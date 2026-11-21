"""Dialog author (phase 1, step 6).

One model call writes the whole dialog between Lena and Felix from the offered facts (plain text
and context only). Every turn carries its fact IDs and lists each claim and each question premise
as an explicit statement with fact IDs; reactions have none (docs/persona.md). The deterministic
parts are pure functions: parsing the answer, the fact-ID gate, the repetition gate and the
minimum-content gate (D5).
The author sees short aliases instead of segment IDs (`aliases`); its answer is mapped back right
after parsing, so turns, gates and the archive only hold segment IDs.
Turn and conversation regeneration belong to the orchestrator (step 8).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from gut_zu_wissen.aliases import (
    build_aliases,
    replace_segment_ids,
    resolve,
    to_aliases,
    unknown_alias_problem,
)
from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.ranking import RankedFact

SPEAKERS = ("Lena", "Felix")
QUESTIONER = "Felix"  # may use premise-exempt facts as question premises again

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "enum": list(SPEAKERS)},
        "text": {"type": "string"},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "fact_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "fact_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["speaker", "text", "fact_ids", "statements"],
    "additionalProperties": False,
}

DIALOG_SCHEMA = {
    "type": "object",
    "properties": {"turns": {"type": "array", "items": TURN_SCHEMA}},
    "required": ["turns"],
    "additionalProperties": False,
}

RULES = """\
Regeln:
- Lena weiß nur, was in den verwendeten Fakten steht. Kein Wissen von außerhalb.
- Jede Behauptung, jede Meinung und jede Prämisse einer Frage braucht Kennungen. Gib sie als \
eigene Einträge in „statements“ an, jeweils mit dem Satz in eigenen Worten und den Kennungen der \
Fakten, die ihn tragen. Eine Aussage, die nur im Kontext steht, ist nicht belegt.
- Sage nur, was der zitierte Fakt-Satz selbst sagt. Kontextsätze dienen nur dem Verständnis; \
was nur im Kontext steht, sprichst du nie aus.
- Zuhörer beziehen jeden Satz auf das, was gerade gesagt wurde. Nennt ein Fakt-Satz seinen \
Gegenstand oder eine beteiligte Person nur mit einem Bezugswort, etwa „dieses Instrument“, „er“, \
„ihm“, „dort“ oder „die Bilder“, dann nennst du sie im Redebeitrag mit Namen, so wie der Kontext \
sie bestimmt, außer der Satz direkt davor nennt sie schon eindeutig. Dieser Name ist die \
einzige Ausnahme von der Regel davor; sonst nimmst du nichts aus dem Kontext. Abgelehnt wird \
etwa Lena: „Es war rund hundertfünfzig Jahre älter als der Mechanismus.“ nach einer Frage zum \
Getriebe, wenn „Dieses Instrument“ im Fakt-Satz eine Sphäre meint.
- Beantwortet Lena eine Frage, handelt der zitierte Fakt-Satz vom Gegenstand der Frage. Passt \
kein Fakt zur Frage, stellt Felix eine andere Frage. Lena verbindet zwei Fakten nicht so, dass \
eine Beziehung entsteht, die keiner der Fakt-Sätze nennt.
- Felix weiß nicht, was Lena gleich erzählt. Seine Fragen legen keine Antwort nahe: keine \
Wertwörter, Vermutungen oder Einzelheiten, die erst in Lenas Antwort stehen. Abgelehnt wird etwa \
Felix: „War die Kritik vielleicht sogar ziemlich scharf?“, wenn erst Lenas Fakt die Schärfe nennt. \
Offene Fragen nach Neuem bleiben erlaubt („Gibt es im Inneren etwas Ungewöhnliches?“).
- „fact_ids“ eines Redebeitrags enthält alle Kennungen, die in ihm vorkommen.
- Erfinde keine Kennungen. Verwende nur Kennungen aus der Liste, ohne Klammern, etwa „F7“.
- Beide Personen machen keine Aussagen über sich selbst: keine Erlebnisse, keine Biografie, \
kein Beruf, kein Wohnort, keine Vorlieben, kein „Das kenne ich“.
- Reaktionen brauchen keine Kennungen und keine Statements. Erlaubt sind: Gefühl im Moment \
(Staunen, Neugier, Belustigung), Nachfrage, Bitte um Erklärung, Überleitung, Wiederaufgreifen des \
eben Gesagten ohne neue Information. Nicht erlaubt sind: neue Information, Wertungen mit \
Faktengehalt („das größte …“), Ankündigungen mit Inhalt, Aussagen über sich selbst über den \
Moment hinaus. Variiere die Reaktionen.
- Eine Reaktion ohne Kennungen wiederholt keine Zahlen, Namen oder Fakten; wer das tut, fasst \
zusammen und braucht Kennungen. Abgelehnt wird etwa Felix: „Vier Monate Suche und kein einziger \
Fund.“ ohne Kennungen.
- Kurze Wertungen, die etwas über die Sache behaupten, etwa „Das ist ein richtiger Coup“ oder \
„Das ist doch ein Widerspruch!“, sind keine Reaktion: Sie stehen nur mit Statement und Kennungen, \
die sie tragen, sonst lässt du sie weg. Frei bleiben reine Gefühlsausrufe wie „Wow!“ oder „Das \
überrascht mich“. Auch „Das habe ich mir gemerkt“ entfällt (Aussage über sich selbst).
- Felix stellt Fragen aus Hörersicht, fasst Gehörtes in eigenen Worten zusammen (dann mit \
Statements und Kennungen) und leitet zum nächsten Fakt über. Prämissen in seinen Fragen müssen \
belegt sein.
- Jede Frage, deren Prämisse sich auf zuvor Gesagtes bezieht, etwa „Wofür stand der Axolotl in \
diesem Film?“, trägt die Kennungen der Fakten hinter dieser Prämisse.
- Lena erzählt jeden Fakt nur einmal. Danach wird er höchstens kurz aufgegriffen: mit einer \
Nachfrage oder Reaktion direkt im Anschluss, ohne seine Details (Zahlen, Namen) zu wiederholen; \
eine solche Nachfrage trägt wie jede Prämisse seine Kennungen. Erst in der Zusammenfassung, dem \
letzten Redebeitrag, darf er wiederkommen. Jeder Fakt ist ein neuer Schritt in der Geschichte: \
Ein späterer Redebeitrag, der einen schon erzählten Fakt noch einmal erzählt oder zusammenfasst, \
ist nicht erlaubt. Beantwortet Lena eine Nachfrage, tut sie es mit einem neuen Fakt.
- Nachfragen wiederholen den Fakt nicht: Felix fragt nach dem Warum, dem Wie oder dem Folgenden, \
ohne den erzählten Inhalt in der Frage noch einmal zu nennen („Warum eigentlich?“, „Wie kam es \
dazu?“). Eine Frage, die einen schon erzählten Fakt als Prämisse nennt, zählt als weiterer \
Redebeitrag für diesen Fakt. Überleitungsfragen zum nächsten Thema bauen nicht auf Fakten auf, die \
schon erzählt wurden; sie fragen offen nach dem Neuen („Und wie ging es dann weiter?“).
- Den ersten Einordnungssatz darf Felix öfter als Prämisse einer Frage verwenden; erzählt wird er \
nur einmal.
- Lena erzählt kurze bis mittellange Sätze, erst die Kernaussage, dann das Detail. Zahlen \
schreibst du aus, wie sie gesprochen werden. Fachbegriffe nur mit Erklärung aus den Fakten.
- Lena verallgemeinert nicht über den Fakt-Satz hinaus. Abgelehnt werden etwa „Es gibt keine \
Zellen, die einfach alles werden“ und „Auf diese Eigenschaft wurde mit dem Namen angespielt“.
- Schlussfolgerungen sind keine Reaktion: Ein „also …“ oder „dann …“, das etwas folgert, behauptet \
etwas. Es braucht ein Statement mit dem Fakt, der die Folgerung selbst trägt, sonst lässt du es \
weg und fragst nur nach. Abgelehnt wird etwa Felix: „Ohne Strafe blieb die vorgesehene Zahl also \
wirkungslos.“, wenn der Fakt nur sagt, dass es keine Strafe gab und die Zahl weiter stieg.
- Lena stellt keine Vergleiche oder Gegensätze her, die der Fakt-Satz nicht selbst enthält: kein \
„anders als“, „im Gegensatz dazu“, „bei X war es anders“, „dagegen“. Sie beginnt direkt mit dem \
Fakt-Satz. Abgelehnt wird etwa Lena: „Bei einer Zerstörung war es anders.“ vor einem Fakt, der \
keinen Vergleich enthält.
- Das Statement eines Redebeitrags gibt den Fakt-Satz wieder, nicht den gesprochenen Satz: Es \
sagt nicht mehr und nicht weniger als der Fakt-Satz.
- Schreibe reinen Text zum Sprechen: kein Markup, keine Aufzählungszeichen, keine Klammern mit \
Quellenangaben, keine Regieanweisungen.
- Am Ende fasst genau eine der beiden Personen das Gespräch einmal kurz zusammen. Keine zweite \
Zusammenfassung, kein abwechselndes Zusammenfassen. Auch die Zusammenfassung braucht Statements \
und Kennungen.
- Ein Redebeitrag reicht bis zum Sprecherwechsel. Die beiden sprechen abwechselnd, aber nicht \
starr; kurze Reaktionen sind eigene Redebeiträge."""

SYSTEM_PROMPT = (
    """\
Du schreibst das Gespräch für einen deutschen Podcast namens „Gut zu wissen“. Zwei feste \
Personen sprechen: Lena, die Expertin, und Felix, der neugierig Fragende. Sie duzen sich und \
sprechen sich mit Vornamen an. Das Gespräch dauert etwa 15 Minuten, das sind 2.000 bis 2.300 \
Wörter. Die Begrüßung und der Abspann sind fester Text außerhalb deines Auftrags: beginne direkt \
beim Thema und schreibe keinen Abschied.

Du bekommst Fakten aus Wikipedia, jeweils mit einer Kennung in eckigen Klammern, etwa [F7], \
und gegebenenfalls einem Kontext. Der Kontext dient nur zum Verstehen (Pronomen, Ortsangaben), \
nie als Beleg. Fakten mit dem Vermerk „(verlinkter Artikel: …)“ stammen aus der Einleitung \
eines verlinkten Artikels. Du wählst aus dem Angebot, was du brauchst, und entwirfst einen \
Erzählbogen; du musst nicht alle Fakten verwenden, aber mindestens {min_used}. Bevorzuge die \
überraschendsten. Fakten mit dem Vermerk „(Pflicht)“ sind die überraschendsten des Artikels und \
müssen im Gespräch vorkommen.

Fakten mit dem Vermerk „(Einordnung)“ stehen am Anfang des Angebots; es sind die ersten Sätze \
des Artikels. Das Gespräch beginnt mit einer kurzen Einordnung aus diesen Fakten, bevor die erste \
Überraschung kommt: was das Thema ist und, wenn die Fakten es sagen, wann und wo. Lena erzählt \
sie; Felix darf mit einer Einstiegsfrage beginnen. Halte die Einordnung kurz: ein oder zwei \
Redebeiträge. Für Einordnungsfakten gelten dieselben Regeln wie für alle Fakten; sie zählen als \
verwendete Fakten und dürfen in der Zusammenfassung wiederkommen.

"""
    + RULES
    + "\n\nAntworte mit JSON im vorgegebenen Schema."
)

TURN_PROMPT = (
    """\
Du schreibst einen einzelnen Redebeitrag eines deutschen Podcast-Gesprächs neu. Es sprechen \
zwei feste Personen: Lena, die Expertin, und Felix, der neugierig Fragende. Sie duzen sich.

Du bekommst alle angebotenen Fakten mit Kennung in eckigen Klammern (der Kontext dient nur zum \
Verstehen, nie als Beleg), dazu den vorherigen und den nächsten Redebeitrag, den zu ersetzenden \
Redebeitrag, jeweils mit ihren Kennungen, und gegebenenfalls den Grund, warum er ersetzt wird. \
Fakten mit dem Vermerk „(Pflicht)“ müssen im Gespräch vorkommen. \
Schreibe nur den neuen Redebeitrag derselben Person. Er muss zum vorherigen und zum nächsten \
Redebeitrag passen. Verwende Fakten aus dem Angebot; behebe den genannten Grund.

"""
    + RULES
    + "\n\nAntworte mit JSON im vorgegebenen Schema: ein Redebeitrag."
)


CONTINUE_PROMPT = (
    """\
Du setzt das Gespräch für einen deutschen Podcast namens „Gut zu wissen“ fort. Zwei feste \
Personen sprechen: Lena, die Expertin, und Felix, der neugierig Fragende. Sie duzen sich und \
sprechen sich mit Vornamen an. Das ganze Gespräch dauert etwa 15 Minuten, das sind 2.000 bis \
2.300 Wörter. Der Abspann ist fester Text außerhalb deines Auftrags: schreibe keinen Abschied.

Du bekommst die angebotenen Fakten, jeweils mit einer Kennung in eckigen Klammern, etwa [F7], und \
gegebenenfalls einem Kontext; der Kontext dient nur zum Verstehen, nie als Beleg. Fakten mit dem \
Vermerk „(verlinkter Artikel: …)“ stammen aus der Einleitung eines verlinkten Artikels, Fakten \
mit dem Vermerk „(Einordnung)“ sind die ersten Sätze des Artikels. Dazu bekommst du das \
bisherige Gespräch. Es bleibt unverändert; du schreibst nur die Fortsetzung, vom genannten \
Redebeitrag an bis zum Ende.

Setze das Gespräch nahtlos an den letzten bisherigen Redebeitrag an: keine Begrüßung, kein \
neuer Einstieg, nichts wiederholen. Fakten, die im bisherigen Gespräch schon erzählt wurden, \
erzählst du nicht noch einmal; nur eine kurze Nachfrage direkt nach der Erzählung darf sich \
darauf beziehen. Fehlt die Einordnung aus den Fakten mit Vermerk „(Einordnung)“ noch, holt Lena \
sie direkt nach. Wähle aus dem Angebot neue Fakten, bevorzuge die überraschendsten, und führe \
den Erzählbogen weiter. Im ganzen Gespräch müssen mindestens {min_used} Fakten verwendet werden, \
dazu alle Fakten mit dem Vermerk „(Pflicht)“, die noch nicht erzählt sind. \
Die Fortsetzung endet mit der einen Zusammenfassung; sie darf auch Fakten aus dem bisherigen \
Gespräch aufgreifen.

"""
    + RULES
    + "\n\nAntworte mit JSON im vorgegebenen Schema: nur die neuen Redebeiträge, ohne die "
    "bisherigen."
)


@dataclass
class Statement:
    text: str
    fact_ids: list[str]


@dataclass
class Turn:
    """A turn; field order is the archive field order."""

    index: int
    speaker: str
    text: str
    fact_ids: list[str]
    statements: list[Statement] = field(default_factory=list)


@dataclass
class TurnProblem:
    turn_index: int
    reason: str


@dataclass
class Dialog:
    turns: list[Turn]
    problems: list[TurnProblem] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- pure functions


EINORDNUNG_NOTE = " (Einordnung)"
MUST_USE_NOTE = " (Pflicht)"


def offer_aliases(
    main: Sequence[RankedFact],
    linked: Sequence[RankedFact],
    einordnung: Sequence[RankedFact] = (),
) -> dict[str, str]:
    """Alias to segment ID for an offer: Einordnung first, then main article, then linked."""
    return build_aliases(r.fact.segment_id for r in [*einordnung, *main, *linked])


def format_offer(
    main: Sequence[RankedFact],
    linked: Sequence[RankedFact],
    einordnung: Sequence[RankedFact] = (),
    must_use: Collection[str] = (),
) -> str:
    """Plain text of the offered facts and their context, in alias order (Einordnung marked,
    then main article with the D8 must-use facts marked, then linked with their article), each
    fact under its alias."""
    alias_of = {sid: alias for alias, sid in offer_aliases(main, linked, einordnung).items()}
    tagged = [
        *((r, EINORDNUNG_NOTE) for r in einordnung),
        *((r, MUST_USE_NOTE if r.fact.segment_id in must_use else "") for r in main),
        *((r, f" (verlinkter Artikel: {r.article})") for r in linked),
    ]
    lines: list[str] = []
    for r, note in tagged:
        lines.append(f"[{alias_of[r.fact.segment_id]}]{note} {r.fact.plain.replace(chr(10), ' ')}")
        if r.fact.context_plain:
            context = " ".join(c.replace(chr(10), " ") for c in r.fact.context_plain)
            lines.append(f"  Kontext: {context}")
    return "\n".join(lines)


def offer_prefix(offer: str) -> str:
    """The start of the user message of a turn regeneration and a continuation: the offer."""
    return f"Angebot:\n{offer}\n\n"


def _strings(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def parse_dialog(answer: dict) -> list[Turn]:
    """Turns from the model's answer. Malformed entries are kept as far as possible so the
    gate can report them; unusable entries (not an object) are dropped."""
    turns: list[Turn] = []
    raw_turns = answer.get("turns", [])
    for item in raw_turns if isinstance(raw_turns, list) else []:
        if not isinstance(item, dict):
            continue
        statements = [
            Statement(str(s.get("text", "")), _strings(s.get("fact_ids")))
            for s in item.get("statements", [])
            if isinstance(s, dict)
        ]
        turns.append(
            Turn(
                len(turns),
                str(item.get("speaker", "")),
                str(item.get("text", "")),
                _strings(item.get("fact_ids")),
                statements,
            )
        )
    return turns


def resolve_turns(turns: Sequence[Turn], aliases: Mapping[str, str]) -> list[Turn]:
    """The author's aliases in fact IDs and statements mapped back to segment IDs (pure)."""
    return [
        Turn(
            t.index,
            t.speaker,
            t.text,
            [resolve(i, aliases) for i in t.fact_ids],
            [Statement(s.text, [resolve(i, aliases) for i in s.fact_ids]) for s in t.statements],
        )
        for t in turns
    ]


def cited(turn: Turn) -> list[str]:
    """Fact IDs of the turn and its statements, first-use order."""
    ids: list[str] = []
    for fid in [*turn.fact_ids, *(i for s in turn.statements for i in s.fact_ids)]:
        if fid not in ids:
            ids.append(fid)
    return ids


def unknown_fact_ids(turn: Turn, offered_ids: Collection[str]) -> list[str]:
    """Cited IDs that no offered fact has, sorted."""
    return sorted({i for i in cited(turn) if i not in offered_ids})


def unknown_ids_reason(unknown: Sequence[str]) -> str:
    """Archived gate problem: the identifiers as the author wrote them, never a guessed segment
    ID."""
    return (
        f"Unknown fact IDs: {', '.join(unknown)} (cited by the author; no offered fact has "
        "this identifier)."
    )


def check_turns(turns: Sequence[Turn], offered_ids: Collection[str]) -> list[TurnProblem]:
    """Gate: speaker and text are valid, every statement has fact IDs, all IDs exist."""
    problems: list[TurnProblem] = []
    for turn in turns:
        if turn.speaker not in SPEAKERS:
            problems.append(
                TurnProblem(
                    turn.index,
                    f"Unbekannte Person „{turn.speaker}“. Es sprechen nur Lena und Felix.",
                )
            )
        if not turn.text.strip():
            problems.append(TurnProblem(turn.index, "Der Redebeitrag ist leer."))
        for statement in turn.statements:
            if not statement.fact_ids:
                problems.append(
                    TurnProblem(
                        turn.index,
                        f"Aussage ohne Kennungen: {statement.text} Jede Aussage braucht die "
                        "Kennungen der Fakten, die sie tragen; sonst lässt du sie weg.",
                    )
                )
        unknown = unknown_fact_ids(turn, offered_ids)
        if unknown:
            problems.append(TurnProblem(turn.index, unknown_ids_reason(unknown)))
    return problems


def statement_fact_ids(turn: Turn) -> list[str]:
    """Fact IDs cited by the turn's statements, first-use order."""
    ids: list[str] = []
    for fid in (i for s in turn.statements for i in s.fact_ids):
        if fid not in ids:
            ids.append(fid)
    return ids


def _german_list(numbers: Sequence[int]) -> str:
    words = [str(n) for n in numbers]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} und {words[-1]}"


def repetition_reason(fact_id: str, earlier: Sequence[int]) -> str:
    """Archived repetition problem: the segment ID and the earlier turns (archive indices) that
    cite the fact. The author sees the alias (`author_reason`)."""
    return (
        f"Der Fakt {fact_id} wurde schon in Redebeitrag {_german_list(earlier)} erzählt. "
        "Erzähle ihn nicht noch einmal; gehe zum nächsten Fakt über oder reagiere kurz ohne "
        "seine Details."
    )


def premise_exempt(turn: Turn, fact_id: str, premise_exempt_ids: Collection[str]) -> bool:
    """Felix citing a premise-exempt fact (the pinned Einordnung #1, the first prose sentence of
    the intro; user decision 2026-11-20): typically a question premise, not counted as telling."""
    return turn.speaker == QUESTIONER and fact_id in premise_exempt_ids


def repetition_problems(
    turns: Sequence[Turn],
    offered_ids: Collection[str],
    max_turns: int,
    premise_exempt_ids: Collection[str] = (),
) -> list[TurnProblem]:
    """Repetition gate (user decision 2026-11-20): an offered fact may be cited by the statements
    of at most `max_turns` turns (the first narration plus one follow-up). Every further turn
    citing it gets a problem, so a repetition always points at the later turn. The last turn is
    the closing summary and is neither counted nor checked. Unknown IDs are the fact-ID gate's
    concern and are ignored here. Felix's turns citing a fact in `premise_exempt_ids` are neither
    counted nor checked for it (`premise_exempt`); Lena's are, as for every fact."""
    earlier: dict[str, list[int]] = {}
    problems: list[TurnProblem] = []
    for turn in turns[:-1]:
        for fid in statement_fact_ids(turn):
            if fid not in offered_ids or premise_exempt(turn, fid, premise_exempt_ids):
                continue
            seen = earlier.setdefault(fid, [])
            if len(seen) >= max_turns:
                problems.append(TurnProblem(turn.index, repetition_reason(fid, seen)))
            seen.append(turn.index)
    return problems


def dialog_problems(
    turns: Sequence[Turn],
    offered_ids: Collection[str],
    max_turns_per_fact: int,
    premise_exempt_ids: Collection[str] = (),
) -> list[TurnProblem]:
    """All gate problems of a whole dialog, in turn order: fact-ID gate, then repetition gate."""
    problems = [
        *check_turns(turns, offered_ids),
        *repetition_problems(turns, offered_ids, max_turns_per_fact, premise_exempt_ids),
    ]
    return sorted(problems, key=lambda p: p.turn_index)


def used_fact_ids(turns: Sequence[Turn], offered_ids: Collection[str]) -> list[str]:
    """Offered facts cited by at least one turn or statement, first-use order."""
    used: list[str] = []
    for turn in turns:
        for fid in cited(turn):
            if fid in offered_ids and fid not in used:
                used.append(fid)
    return used


EXCLUDED_FACT = "[nicht mehr im Angebot]"


def author_reason(
    turn: Turn,
    reason: str | None,
    aliases: Mapping[str, str],
    offered_ids: Collection[str] | None = None,
) -> str | None:
    """The reason for a regeneration as the author sees it: the unknown-ID problem in German and
    actionable, offered segment IDs as aliases. `reason` is the archived one.

    `offered_ids` is the offer the rejected turn was judged against when it differs from the one
    of `aliases` (a continuation, D6): the unknown-ID problem is matched against it, and its
    segment IDs without an alias (excluded since) become `EXCLUDED_FACT`."""
    if reason is None:
        return None
    offered = aliases.values() if offered_ids is None else offered_ids
    unknown = unknown_fact_ids(turn, offered)
    if unknown:
        reason = reason.replace(
            unknown_ids_reason(unknown), unknown_alias_problem(unknown, aliases)
        )
    reason = replace_segment_ids(reason, aliases)
    gone = set(offered) - set(aliases.values())
    for sid in sorted(gone, key=lambda s: (-len(s), s)):
        reason = reason.replace(sid, EXCLUDED_FACT)
    return reason


def top_facts_gate(
    must_use: Collection[str], used: Collection[str], exempt: Collection[str] = ()
) -> str | None:
    """D8: the top-ranked facts (`must_use`) must be used; facts in `exempt` (excluded by D6,
    cited by rejected statements) are not required."""
    missing = [i for i in must_use if i not in used and i not in exempt]
    if missing:
        return (
            f"Gate top facts (D8): {len(missing)} of the {len(must_use)} top-ranked facts "
            f"not used: {', '.join(sorted(missing))}."
        )
    return None


def used_facts_gate(used: int, minimum: int) -> str | None:
    """D5: fewer than `minimum` used facts after the author triggers a conversation regeneration."""
    if used < minimum:
        return (
            f"Gate minimum content (D5): {used} facts used by the author, "
            f"at least {minimum} needed."
        )
    return None


# ---------------------------------------------------------------- author


def write_dialog(
    llm: StructuredLlm,
    main: Sequence[RankedFact],
    linked: Sequence[RankedFact],
    min_used_facts: int,
    max_turns_per_fact: int,
    einordnung: Sequence[RankedFact] = (),
    premise_exempt_ids: Collection[str] = (),
    must_use: Collection[str] = (),
) -> Dialog:
    """One dialog from the offer, checked by the fact-ID and repetition gates. Problems are
    recorded, not raised. `premise_exempt_ids`: see `repetition_problems`."""
    answer = llm.complete_json(
        SYSTEM_PROMPT.format(min_used=min_used_facts),
        format_offer(main, linked, einordnung, must_use),
        DIALOG_SCHEMA,
    )
    aliases = offer_aliases(main, linked, einordnung)
    offered = set(aliases.values())
    turns = resolve_turns(parse_dialog(answer), aliases)
    return Dialog(
        turns,
        dialog_problems(turns, offered, max_turns_per_fact, premise_exempt_ids),
        used_fact_ids(turns, offered),
    )


def _turn_lines(label: str, turn: Turn, aliases: Mapping[str, str]) -> list[str]:
    lines = [f"{label} ({turn.speaker}):", turn.text]
    shown = to_aliases(cited(turn), aliases)
    if shown:
        lines.append(f"Kennungen: {', '.join(shown)}")
    return lines


def format_turn_context(
    turns: Sequence[Turn],
    index: int,
    reason: str | None,
    aliases: Mapping[str, str],
    premise_exempt_ids: Collection[str] = (),
) -> str:
    """The neighbours, the turn to replace and the reason, as plain text for the author. Fact IDs
    appear as aliases; `reason` is already in the author's form (`author_reason`)."""
    lines: list[str] = []
    if index > 0:
        lines += [*_turn_lines("Vorheriger Redebeitrag", turns[index - 1], aliases), ""]
    old = turns[index]
    lines += [*_turn_lines("Zu ersetzender Redebeitrag", old, aliases), ""]
    lines += [
        "Grund für den Ersatz:",
        reason
        or "Der vorherige Redebeitrag wurde neu geschrieben; das Gespräch muss wieder "
        "stimmig sein. Behalte die Sachaussagen des alten Redebeitrags bei, soweit die Fakten "
        "sie tragen.",
        "",
    ]
    if index + 1 < len(turns):
        lines += [*_turn_lines("Nächster Redebeitrag", turns[index + 1], aliases), ""]
    told = told_elsewhere(turns, index, premise_exempt_ids)
    if told:
        lines += [
            "Schon in anderen Redebeiträgen erzählt (nicht noch einmal erzählen; nur eine kurze "
            "Nachfrage direkt nach der Erzählung darf sich darauf beziehen):",
            *_told_lines(told, aliases, premise_exempt_ids),
        ]
    return "\n".join(lines).strip()


PREMISE_EXEMPT_NOTE = " (Felix darf ihn erneut als Prämisse einer Frage verwenden)"


def _told_lines(
    told: Mapping[str, Sequence[int]],
    aliases: Mapping[str, str],
    premise_exempt_ids: Collection[str] = (),
) -> list[str]:
    return [
        f"- {shown[0]}: Redebeitrag {', '.join(map(str, where))}"
        + (PREMISE_EXEMPT_NOTE if fid in premise_exempt_ids else "")
        for fid, where in told.items()
        if (shown := to_aliases([fid], aliases))
    ]


def told_in(
    turns: Sequence[Turn], skip: Collection[int] = (), premise_exempt_ids: Collection[str] = ()
) -> dict[str, list[int]]:
    """Facts cited by the statements of `turns` (positions in `skip` left out), with the turns
    (archive indices) that cite them, in dialog order. Felix's citations of premise-exempt facts
    do not count as telling, as in the repetition gate."""
    told: dict[str, list[int]] = {}
    for pos, turn in enumerate(turns):
        if pos in skip:
            continue
        for fid in dict.fromkeys(f for s in turn.statements for f in s.fact_ids):
            if not premise_exempt(turn, fid, premise_exempt_ids):
                told.setdefault(fid, []).append(turn.index)
    return told


def told_elsewhere(
    turns: Sequence[Turn], index: int, premise_exempt_ids: Collection[str] = ()
) -> dict[str, list[int]]:
    """Facts cited by statements of other turns, with those turns, in dialog order.

    The turn to replace and the closing summary (the last turn) are left out, as are Felix's
    citations of premise-exempt facts, as in the repetition gate. Used to tell the author which
    facts are already told.
    """
    return told_in(turns, (index, len(turns) - 1), premise_exempt_ids)


def format_prefix(
    prefix: Sequence[Turn],
    aliases: Mapping[str, str],
    offered_ids: Collection[str],
    premise_exempt_ids: Collection[str] = (),
) -> str:
    """The kept turns of a continuation as plain text for the author: every turn with its
    speaker and identifiers (aliases), the facts already told, the word and used-fact counts so
    far, and the index the continuation starts at."""
    start = len(prefix)
    lines = [f"Bisheriges Gespräch (Redebeitrag 0 bis {start - 1}, bleibt unverändert):", ""]
    for turn in prefix:
        lines += [*_turn_lines(f"Redebeitrag {turn.index}", turn, aliases), ""]
    told = told_in(prefix, premise_exempt_ids=premise_exempt_ids)
    if told:
        lines += [
            "Im bisherigen Gespräch schon erzählt (nicht noch einmal erzählen; nur eine kurze "
            "Nachfrage direkt nach der Erzählung darf sich darauf beziehen):",
            *_told_lines(told, aliases, premise_exempt_ids),
            "",
        ]
    words = sum(len(t.text.split()) for t in prefix)
    used = len(used_fact_ids(prefix, offered_ids))
    lines += [
        f"Das bisherige Gespräch hat {words} Wörter und verwendet {used} Fakten.",
        f"Schreibe die Fortsetzung ab Redebeitrag {start} bis zum Ende, mit der einen "
        "Zusammenfassung am Schluss. Der erste neue Redebeitrag schließt direkt an "
        f"Redebeitrag {start - 1} ({prefix[-1].speaker}) an.",
    ]
    return "\n".join(lines)


def format_rejections(rejections: Sequence[tuple[int, str]]) -> str:
    """Why the failed chain was discarded, for the author of the continuation (user decision
    2026-11-20): the last rejection (the one that used up the budget) with its turn, the earlier
    rejections of the chain briefly below it (repeats left out), and the instruction to avoid
    them. Each entry is (turn index, reason in the author's form, `author_reason`); empty without
    rejections."""
    if not rejections:
        return ""
    index, reason = rejections[-1]
    lines = [f"Grund, warum der bisherige Redebeitrag {index} verworfen wurde:", reason]
    earlier = [r for r in dict.fromkeys(rejections[:-1]) if r != rejections[-1]]
    if earlier:
        lines += [
            "",
            "Davor in derselben Kette verworfen:",
            *(f"- Redebeitrag {i}: {r}" for i, r in earlier),
        ]
    lines += ["", f"Vermeide {'diese Fehler' if earlier else 'diesen Fehler'} in der Fortsetzung."]
    return "\n".join(lines)


def continue_dialog(
    llm: StructuredLlm,
    offer: str,
    aliases: Mapping[str, str],
    prefix: Sequence[Turn],
    min_used_facts: int,
    rejections: Sequence[tuple[int, str]] = (),
    premise_exempt_ids: Collection[str] = (),
) -> list[Turn]:
    """The continuation of a dialog after its kept `prefix` (turns 0..k-1, user decision
    2026-11-20): the author writes the turns from k to the end, the closing summary included,
    seeing the prefix, the facts already told and why the failed chain was discarded
    (`rejections`, see `format_rejections`). Same schema as the whole dialog; the returned
    turns are re-indexed from k and hold segment IDs. `offer` is the text of `format_offer`,
    `aliases` the offer's aliases, `premise_exempt_ids` as in `repetition_problems`. Gates are the
    orchestrator's concern."""
    assert prefix, "a continuation needs at least one kept turn"
    kept = format_prefix(prefix, aliases, aliases.values(), premise_exempt_ids)
    user = offer_prefix(offer) + kept
    if rejections:
        user += f"\n\n{format_rejections(rejections)}"
    answer = llm.complete_json(CONTINUE_PROMPT.format(min_used=min_used_facts), user, DIALOG_SCHEMA)
    turns = resolve_turns(parse_dialog(answer), aliases)
    for n, turn in enumerate(turns):
        turn.index = len(prefix) + n
    return turns


def regenerate_turn(
    llm: StructuredLlm,
    offer: str,
    aliases: Mapping[str, str],
    turns: Sequence[Turn],
    index: int,
    reason: str | None,
    premise_exempt_ids: Collection[str] = (),
) -> Turn:
    """A new version of turn `index` with its neighbours as context (design: Fehlerbehandlung).

    The speaker stays the one of the replaced turn. `offer` is the text of `format_offer`,
    `aliases` the offer's aliases (`offer_aliases`); `reason` is the archived one;
    `premise_exempt_ids` as in `repetition_problems`.
    """
    context = format_turn_context(
        turns, index, author_reason(turns[index], reason, aliases), aliases, premise_exempt_ids
    )
    prefix = offer_prefix(offer)
    # The offer is the same for every regeneration and cascade step of a conversation and comes
    # first: cached up to its end (spike S7); the neighbours and the reason follow.
    answer = llm.complete_json(TURN_PROMPT, prefix + context, TURN_SCHEMA, cache_user_prefix=prefix)
    (new,) = resolve_turns(parse_dialog({"turns": [answer]}), aliases) or [Turn(index, "", "", [])]
    new.index = index
    if turns[index].speaker in SPEAKERS:
        new.speaker = turns[index].speaker
    return new
