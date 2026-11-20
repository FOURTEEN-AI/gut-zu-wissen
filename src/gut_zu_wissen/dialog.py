"""Dialog author (phase 1, step 6).

One model call writes the whole dialog between Lena and Felix from the offered facts (plain text
and context only). Every turn carries its fact IDs and lists each claim and each question premise
as an explicit statement with fact IDs; reactions have none (docs/persona.md). The deterministic
parts are pure functions: parsing the answer, the fact-ID gate and the minimum-content gate (D5).
Turn and conversation regeneration belong to the orchestrator (step 8).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field

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
- Jede Behauptung, jede Meinung und jede Prämisse einer Frage braucht Fakt-IDs. Gib sie als \
eigene Einträge in „statements“ an, jeweils mit dem Satz in eigenen Worten und den IDs der \
Fakten, die ihn tragen. Eine Aussage, die nur im Kontext steht, ist nicht belegt.
- Sage nur, was der zitierte Fakt-Satz selbst sagt. Kontextsätze dienen nur dem Verständnis; \
was nur im Kontext steht, sprichst du nie aus.
- „fact_ids“ eines Redebeitrags enthält alle Fakt-IDs, die in ihm vorkommen.
- Erfinde keine IDs. Verwende nur IDs aus der Liste.
- Beide Personen machen keine Aussagen über sich selbst: keine Erlebnisse, keine Biografie, \
kein Beruf, kein Wohnort, keine Vorlieben, kein „Das kenne ich“.
- Reaktionen brauchen keine Fakt-IDs und keine Statements. Erlaubt sind: Gefühl im Moment \
(Staunen, Neugier, Belustigung), Nachfrage, Bitte um Erklärung, Überleitung, Wiederaufgreifen des \
eben Gesagten ohne neue Information. Nicht erlaubt sind: neue Information, Wertungen mit \
Faktengehalt („das größte …“), Ankündigungen mit Inhalt, Aussagen über sich selbst über den \
Moment hinaus. Variiere die Reaktionen.
- Eine Reaktion ohne Fakt-IDs wiederholt keine Zahlen, Namen oder Fakten; wer das tut, fasst \
zusammen und braucht Fakt-IDs. Abgelehnt wird etwa Felix: „Vier Monate Suche und kein einziger \
Fund.“ ohne Fakt-IDs.
- Kurze Wertungen, die etwas über die Sache behaupten, etwa „Das ist ein richtiger Coup“ oder \
„Das ist doch ein Widerspruch!“, sind keine Reaktion: Sie stehen nur mit Statement und Fakt-IDs, \
die sie tragen, sonst lässt du sie weg. Frei bleiben reine Gefühlsausrufe wie „Wow!“ oder „Das \
überrascht mich“. Auch „Das habe ich mir gemerkt“ entfällt (Aussage über sich selbst).
- Felix stellt Fragen aus Hörersicht, fasst Gehörtes in eigenen Worten zusammen (dann mit \
Statements und Fakt-IDs) und leitet zum nächsten Fakt über. Prämissen in seinen Fragen müssen \
belegt sein.
- Jede Frage, deren Prämisse sich auf zuvor Gesagtes bezieht, etwa „Wofür stand der Axolotl in \
diesem Film?“, trägt die Fakt-IDs der Fakten hinter dieser Prämisse.
- Lena erzählt kurze bis mittellange Sätze, erst die Kernaussage, dann das Detail. Zahlen \
schreibst du aus, wie sie gesprochen werden. Fachbegriffe nur mit Erklärung aus den Fakten.
- Lena verallgemeinert nicht über den Fakt-Satz hinaus. Abgelehnt werden etwa „Es gibt keine \
Zellen, die einfach alles werden“ und „Auf diese Eigenschaft wurde mit dem Namen angespielt“.
- Schreibe reinen Text zum Sprechen: kein Markup, keine Aufzählungszeichen, keine Klammern mit \
Quellenangaben, keine Regieanweisungen.
- Ein Redebeitrag reicht bis zum Sprecherwechsel. Die beiden sprechen abwechselnd, aber nicht \
starr; kurze Reaktionen sind eigene Redebeiträge."""

SYSTEM_PROMPT = (
    """\
Du schreibst das Gespräch für einen deutschen Podcast namens „Gut zu wissen“. Zwei feste \
Personen sprechen: Lena, die Expertin, und Felix, der neugierig Fragende. Sie duzen sich und \
sprechen sich mit Vornamen an. Das Gespräch dauert etwa 15 Minuten, das sind 2.000 bis 2.300 \
Wörter. Die Begrüßung und der Abspann sind fester Text außerhalb deines Auftrags: beginne direkt \
beim Thema und schreibe keinen Abschied.

Du bekommst Fakten aus Wikipedia, jeweils mit einer ID in eckigen Klammern und gegebenenfalls \
einem Kontext. Der Kontext dient nur zum Verstehen (Pronomen, Ortsangaben), nie als Beleg. \
Fakten mit dem Vermerk „(verlinkter Artikel: …)“ stammen aus der Einleitung eines verlinkten \
Artikels. Du wählst aus dem Angebot, was du brauchst, und entwirfst einen Erzählbogen; du musst \
nicht alle Fakten verwenden, aber mindestens {min_used}. Bevorzuge die überraschendsten.

"""
    + RULES
    + "\n\nAntworte mit JSON im vorgegebenen Schema."
)

TURN_PROMPT = (
    """\
Du schreibst einen einzelnen Redebeitrag eines deutschen Podcast-Gesprächs neu. Es sprechen \
zwei feste Personen: Lena, die Expertin, und Felix, der neugierig Fragende. Sie duzen sich.

Du bekommst alle angebotenen Fakten mit ID in eckigen Klammern (der Kontext dient nur zum \
Verstehen, nie als Beleg), dazu den vorherigen und den nächsten Redebeitrag, den zu ersetzenden \
Redebeitrag und gegebenenfalls den Grund, warum er ersetzt wird. Schreibe nur den neuen \
Redebeitrag derselben Person. Er muss zum vorherigen und zum nächsten Redebeitrag passen. \
Verwende Fakten aus dem Angebot; behebe den genannten Grund.

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


def format_offer(main: Sequence[RankedFact], linked: Sequence[RankedFact]) -> str:
    """Plain text of the offered facts and their context, main article first."""
    lines: list[str] = []
    for r in [*main, *linked]:
        note = "" if r in main else f" (verlinkter Artikel: {r.article})"
        lines.append(f"[{r.fact.segment_id}]{note} {r.fact.plain.replace(chr(10), ' ')}")
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


def check_turns(turns: Sequence[Turn], offered_ids: Collection[str]) -> list[TurnProblem]:
    """Gate: speaker and text are valid, every statement has fact IDs, all IDs exist."""
    problems: list[TurnProblem] = []
    for turn in turns:
        if turn.speaker not in SPEAKERS:
            problems.append(TurnProblem(turn.index, f"Unknown speaker '{turn.speaker}'."))
        if not turn.text.strip():
            problems.append(TurnProblem(turn.index, "Empty text."))
        for statement in turn.statements:
            if not statement.fact_ids:
                problems.append(
                    TurnProblem(turn.index, f"Statement without fact IDs: {statement.text}")
                )
        ids = [*turn.fact_ids, *(i for s in turn.statements for i in s.fact_ids)]
        unknown = sorted({i for i in ids if i not in offered_ids})
        if unknown:
            problems.append(TurnProblem(turn.index, f"Unknown fact IDs: {', '.join(unknown)}"))
    return problems


def used_fact_ids(turns: Sequence[Turn], offered_ids: Collection[str]) -> list[str]:
    """Offered facts cited by at least one turn or statement, first-use order."""
    used: list[str] = []
    for turn in turns:
        for fid in [*turn.fact_ids, *(i for s in turn.statements for i in s.fact_ids)]:
            if fid in offered_ids and fid not in used:
                used.append(fid)
    return used


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
) -> Dialog:
    """One dialog from the offer, checked by the fact-ID gate. Problems are recorded, not raised."""
    answer = llm.complete_json(
        SYSTEM_PROMPT.format(min_used=min_used_facts), format_offer(main, linked), DIALOG_SCHEMA
    )
    offered = {r.fact.segment_id for r in [*main, *linked]}
    turns = parse_dialog(answer)
    return Dialog(turns, check_turns(turns, offered), used_fact_ids(turns, offered))


def format_turn_context(turns: Sequence[Turn], index: int, reason: str | None) -> str:
    """The neighbours, the turn to replace and the reason, as plain text for the author."""
    lines: list[str] = []
    if index > 0:
        lines += [
            f"Vorheriger Redebeitrag ({turns[index - 1].speaker}):",
            turns[index - 1].text,
            "",
        ]
    old = turns[index]
    lines += [f"Zu ersetzender Redebeitrag ({old.speaker}):", old.text, ""]
    lines += [
        "Grund für den Ersatz:",
        reason
        or "Der vorherige Redebeitrag wurde neu geschrieben; das Gespräch muss wieder "
        "stimmig sein. Behalte die Sachaussagen des alten Redebeitrags bei, soweit die Fakten "
        "sie tragen.",
        "",
    ]
    if index + 1 < len(turns):
        lines += [f"Nächster Redebeitrag ({turns[index + 1].speaker}):", turns[index + 1].text]
    return "\n".join(lines).strip()


def regenerate_turn(
    llm: StructuredLlm,
    offer: str,
    turns: Sequence[Turn],
    index: int,
    reason: str | None,
) -> Turn:
    """A new version of turn `index` with its neighbours as context (design: Fehlerbehandlung).

    The speaker stays the one of the replaced turn. `offer` is the text of `format_offer`.
    """
    user = f"Angebot:\n{offer}\n\n{format_turn_context(turns, index, reason)}"
    answer = llm.complete_json(TURN_PROMPT, user, TURN_SCHEMA)
    (new,) = parse_dialog({"turns": [answer]}) or [Turn(index, "", "", [])]
    new.index = index
    if turns[index].speaker in SPEAKERS:
        new.speaker = turns[index].speaker
    return new
