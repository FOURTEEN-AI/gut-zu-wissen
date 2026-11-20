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
- Lena erzählt kurze bis mittellange Sätze, erst die Kernaussage, dann das Detail. Zahlen \
schreibst du aus, wie sie gesprochen werden. Fachbegriffe nur mit Erklärung aus den Fakten.
- Lena verallgemeinert nicht über den Fakt-Satz hinaus. Abgelehnt werden etwa „Es gibt keine \
Zellen, die einfach alles werden“ und „Auf diese Eigenschaft wurde mit dem Namen angespielt“.
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
überraschendsten.

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
Schreibe nur den neuen Redebeitrag derselben Person. Er muss zum vorherigen und zum nächsten \
Redebeitrag passen. Verwende Fakten aus dem Angebot; behebe den genannten Grund.

"""
    + RULES
    + "\n\nAntworte mit JSON im vorgegebenen Schema: ein Redebeitrag."
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
) -> str:
    """Plain text of the offered facts and their context, in alias order (Einordnung marked,
    then main article, then linked with their article), each fact under its alias."""
    alias_of = {sid: alias for alias, sid in offer_aliases(main, linked, einordnung).items()}
    tagged = [
        *((r, EINORDNUNG_NOTE) for r in einordnung),
        *((r, "") for r in main),
        *((r, f" (verlinkter Artikel: {r.article})") for r in linked),
    ]
    lines: list[str] = []
    for r, note in tagged:
        lines.append(f"[{alias_of[r.fact.segment_id]}]{note} {r.fact.plain.replace(chr(10), ' ')}")
        if r.fact.context_plain:
            context = " ".join(c.replace(chr(10), " ") for c in r.fact.context_plain)
            lines.append(f"  Kontext: {context}")
    return "\n".join(lines)


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


def repetition_problems(
    turns: Sequence[Turn], offered_ids: Collection[str], max_turns: int
) -> list[TurnProblem]:
    """Repetition gate (user decision 2026-11-20): an offered fact may be cited by the statements
    of at most `max_turns` turns (the first narration plus one follow-up). Every further turn
    citing it gets a problem, so a repetition always points at the later turn. The last turn is
    the closing summary and is neither counted nor checked. Unknown IDs are the fact-ID gate's
    concern and are ignored here."""
    earlier: dict[str, list[int]] = {}
    problems: list[TurnProblem] = []
    for turn in turns[:-1]:
        for fid in statement_fact_ids(turn):
            if fid not in offered_ids:
                continue
            seen = earlier.setdefault(fid, [])
            if len(seen) >= max_turns:
                problems.append(TurnProblem(turn.index, repetition_reason(fid, seen)))
            seen.append(turn.index)
    return problems


def dialog_problems(
    turns: Sequence[Turn], offered_ids: Collection[str], max_turns_per_fact: int
) -> list[TurnProblem]:
    """All gate problems of a whole dialog, in turn order: fact-ID gate, then repetition gate."""
    problems = [
        *check_turns(turns, offered_ids),
        *repetition_problems(turns, offered_ids, max_turns_per_fact),
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


def author_reason(turn: Turn, reason: str | None, aliases: Mapping[str, str]) -> str | None:
    """The reason for a regeneration as the author sees it: the unknown-ID problem in German and
    actionable, offered segment IDs as aliases. `reason` is the archived one."""
    if reason is None:
        return None
    unknown = unknown_fact_ids(turn, aliases.values())
    if unknown:
        reason = reason.replace(
            unknown_ids_reason(unknown), unknown_alias_problem(unknown, aliases)
        )
    return replace_segment_ids(reason, aliases)


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
) -> Dialog:
    """One dialog from the offer, checked by the fact-ID and repetition gates. Problems are
    recorded, not raised."""
    answer = llm.complete_json(
        SYSTEM_PROMPT.format(min_used=min_used_facts),
        format_offer(main, linked, einordnung),
        DIALOG_SCHEMA,
    )
    aliases = offer_aliases(main, linked, einordnung)
    offered = set(aliases.values())
    turns = resolve_turns(parse_dialog(answer), aliases)
    return Dialog(
        turns,
        dialog_problems(turns, offered, max_turns_per_fact),
        used_fact_ids(turns, offered),
    )


def _turn_lines(label: str, turn: Turn, aliases: Mapping[str, str]) -> list[str]:
    lines = [f"{label} ({turn.speaker}):", turn.text]
    shown = to_aliases(cited(turn), aliases)
    if shown:
        lines.append(f"Kennungen: {', '.join(shown)}")
    return lines


def format_turn_context(
    turns: Sequence[Turn], index: int, reason: str | None, aliases: Mapping[str, str]
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
        lines += _turn_lines("Nächster Redebeitrag", turns[index + 1], aliases)
    return "\n".join(lines).strip()


def regenerate_turn(
    llm: StructuredLlm,
    offer: str,
    aliases: Mapping[str, str],
    turns: Sequence[Turn],
    index: int,
    reason: str | None,
) -> Turn:
    """A new version of turn `index` with its neighbours as context (design: Fehlerbehandlung).

    The speaker stays the one of the replaced turn. `offer` is the text of `format_offer`,
    `aliases` the offer's aliases (`offer_aliases`); `reason` is the archived one.
    """
    context = format_turn_context(
        turns, index, author_reason(turns[index], reason, aliases), aliases
    )
    user = f"Angebot:\n{offer}\n\n{context}"
    answer = llm.complete_json(TURN_PROMPT, user, TURN_SCHEMA)
    (new,) = resolve_turns(parse_dialog({"turns": [answer]}), aliases) or [Turn(index, "", "", [])]
    new.index = index
    if turns[index].speaker in SPEAKERS:
        new.speaker = turns[index].speaker
    return new
