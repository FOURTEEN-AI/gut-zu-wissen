"""STT verification (phase 1, step 10; design: "Verifikation", "Abweichung", "Fehlerbehandlung").

Whisper's transcript of a section is compared word by word with the section's script. Both
sides are normalised first, so that differences in writing alone do not count:

- case (casefold, so ß and ss are equal), punctuation, quotes, apostrophes inside words,
  hyphens and slashes (word separators on both sides);
- German number words become numerals, as Whisper writes them: cardinals up to 999,999
  ("achtzehnhundertneunundachtzig" = 1889, "eine" = 1), ordinals ("dritten" = 3, as "3."
  loses its dot), decades ("zwanziger" = "20er") and "zwei Komma fünf" = "2,5";
- numerals: thousands separators ("10.000", "10 000"), decimal point or comma;
- a few symbols Whisper writes for spoken words (%, ‰, °, €, $, &).

The normalisation, the word diff and the retry decision are pure functions. All remaining
deviations of a section go to the LLM judge model in one call; it decides per deviation whether
the audio changes the statement ("im Zweifel verändert"). A deviation it leaves undecided counts
as a change. Only a change leads to resynthesis of the section (`retry_action`).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

from gut_zu_wissen.llm import StructuredLlm

SYMBOLS = {
    "%": " Prozent ",
    "‰": " Promille ",
    "°": " Grad ",
    "€": " Euro ",
    "$": " Dollar ",
    "&": " und ",
}
_APOSTROPHE = re.compile(r"(?<=\w)['’ʼ](?=\w)")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*[^\W\d_]*|[^\W\d_]+")
_NUMERAL = re.compile(r"(\d+(?:[.,]\d+)*)([^\W\d_]*)")
_GROUPED = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_DECIMAL = re.compile(r"(\d+)[.,](\d+)")
_INFLECTION = ("en", "em", "er", "es", "e")
_DECIMAL_WORD = "komma"


@dataclass(frozen=True)
class Token:
    """A word after normalisation (`norm`) and as written (`raw`)."""

    norm: str
    raw: str


@dataclass
class Deviation:
    """One difference between script and transcript; field order is the archive field order."""

    index: int
    expected: str  # script words (empty: Whisper heard extra words)
    heard: str  # transcript words (empty: Whisper heard nothing for them)
    context_before: str
    context_after: str
    changes_statement: bool | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- number words


def _number_words() -> dict[str, int]:
    units = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
    special = {
        "null": 0,
        "zehn": 10,
        "elf": 11,
        "zwölf": 12,
        "dreizehn": 13,
        "vierzehn": 14,
        "fünfzehn": 15,
        "sechzehn": 16,
        "siebzehn": 17,
        "achtzehn": 18,
        "neunzehn": 19,
    }
    tens = {
        20: "zwanzig",
        30: "dreißig",
        40: "vierzig",
        50: "fünfzig",
        60: "sechzig",
        70: "siebzig",
        80: "achtzig",
        90: "neunzig",
    }
    words = {"eins": 1, "zwo": 2, **special}
    for value, word in enumerate(units[1:], start=1):
        words[word] = value
    for value, word in tens.items():
        words[word] = value
        for unit in range(1, 10):
            words[f"{units[unit]}und{word}"] = value + unit
    return {w.casefold(): v for w, v in words.items()}


BELOW_100 = _number_words()
# Forms of "ein" that stand for one; articles too, which is harmless because both sides are
# normalised the same way.
ONE = {w.casefold(): 1 for w in ("eine", "einen", "einem", "einer", "eines")}


def _below_1000(word: str) -> int | None:
    if word in BELOW_100:
        return BELOW_100[word]
    head, sep, rest = word.partition("hundert")
    if not sep:
        return None
    factor = 1 if head == "" else BELOW_100.get(head)  # "achtzehnhundert" = 1800 as in years
    tail = 0 if rest == "" else BELOW_100.get(rest)
    if not factor or tail is None:
        return None
    return factor * 100 + tail


def cardinal(word: str) -> int | None:
    """Value of a casefolded German cardinal number word below one million, else `None`."""
    if word in ONE:
        return 1
    head, sep, rest = word.partition("tausend")
    if not sep:
        return _below_1000(word)
    factor = 1 if head == "" else _below_1000(head)
    tail = 0 if rest == "" else _below_1000(rest)
    if not factor or tail is None:
        return None
    return factor * 1000 + tail


def ordinal(word: str) -> int | None:
    """Value of a casefolded German ordinal ("dritten", "zwanzigste", "hunderterste")."""
    for suffix in _INFLECTION:
        if not word.endswith(suffix) or len(word) == len(suffix):
            continue
        stem = word[: -len(suffix)]
        if stem.endswith("st"):  # from 20 on: zwanzigst-, hundertst-, tausendst-
            value = cardinal(stem[:-2])
            if value is not None and value >= 20:
                return value
        if stem.endswith("erst"):  # erst-, hunderterst-
            value = cardinal(stem[:-4] + "eins")
            if value is not None:
                return value
        if stem.endswith("dritt"):
            base = stem[:-5] + "drei"
        elif stem.endswith("siebt"):
            base = stem[:-1] + "en"
        elif stem.endswith("acht"):
            base = stem
        elif stem.endswith("t"):
            base = stem[:-1]
        else:
            continue
        value = cardinal(base)
        if value is not None and value >= 2:
            return value
    return None


def decade(word: str) -> int | None:
    """Value of a decade word ("zwanziger", "neunzehnhundertneunziger"), else `None`."""
    if not word.endswith("er"):
        return None
    value = cardinal(word[:-2])
    return value if value is not None and value >= 10 and value % 10 == 0 else None


# ---------------------------------------------------------------- normalisation


def _numeral(text: str) -> str:
    """Canonical form of a numeral: no thousands separators, decimal comma, no leading zeros."""
    if _GROUPED.fullmatch(text):
        text = text.replace(".", "")
    match = _DECIMAL.fullmatch(text)
    if match:
        return f"{int(match.group(1))},{match.group(2)}"
    if text.isdigit():
        return str(int(text))
    return text


def _word(word: str) -> str:
    folded = word.casefold()
    for parse, suffix in ((cardinal, ""), (ordinal, ""), (decade, "er")):
        value = parse(folded)
        if value is not None:
            return f"{value}{suffix}"
    return folded


def tokenize(text: str) -> list[Token]:
    """Normalised words of a text (see module docstring)."""
    text = unicodedata.normalize("NFC", text)
    for symbol, word in SYMBOLS.items():
        text = text.replace(symbol, word)
    text = _APOSTROPHE.sub("", text)
    tokens: list[Token] = []
    for raw in _TOKEN.findall(text):
        numeral = _NUMERAL.fullmatch(raw)
        if numeral:
            norm = _numeral(numeral.group(1)) + numeral.group(2).casefold()
        else:
            norm = _word(raw)
        tokens.append(Token(norm, raw))
    return _merge_numbers(tokens)


def _merge_numbers(tokens: Sequence[Token]) -> list[Token]:
    """Join "10 000" to 10000 and "zwei Komma fünf" to 2,5."""
    out: list[Token] = []
    for token in tokens:
        if (
            out
            and len(token.raw) == 3
            and token.raw.isdigit()
            and out[-1].raw.isdigit()
            and len(out[-1].raw) <= 3
        ):
            prev = out.pop()
            token = Token(str(int(prev.raw + token.raw)), f"{prev.raw} {token.raw}")
        elif (
            len(out) >= 2
            and token.norm.isdigit()
            and out[-1].norm == _DECIMAL_WORD
            and out[-2].norm.isdigit()
        ):
            sep, prev = out.pop(), out.pop()
            token = Token(f"{prev.norm},{token.norm}", f"{prev.raw} {sep.raw} {token.raw}")
        out.append(token)
    return out


# ---------------------------------------------------------------- diff


def diff(script: str, transcript: str, context_words: int) -> list[Deviation]:
    """Word differences between the normalised script and transcript, in script order.

    Each deviation carries the words as written and `context_words` script words on each side.
    """
    expected, heard = tokenize(script), tokenize(transcript)
    matcher = difflib.SequenceMatcher(
        None, [t.norm for t in expected], [t.norm for t in heard], autojunk=False
    )
    deviations: list[Deviation] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        deviations.append(
            Deviation(
                len(deviations),
                _raw(expected[i1:i2]),
                _raw(heard[j1:j2]),
                _raw(expected[max(0, i1 - context_words) : i1]),
                _raw(expected[i2 : i2 + context_words]),
            )
        )
    return deviations


def _raw(tokens: Sequence[Token]) -> str:
    return " ".join(t.raw for t in tokens)


# ---------------------------------------------------------------- LLM decision

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "changes_statement": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "changes_statement", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}
DECISION_PROMPT = (
    "Du prüfst, ob eine Sprachsynthese einen Abschnitt eines Podcast-Gesprächs richtig "
    "wiedergibt. Eine automatische Spracherkennung hat das Audio transkribiert, ein Wortvergleich "
    "mit dem Skript hat Abweichungen gefunden. Zahlwörter und Ziffern, Groß- und Kleinschreibung "
    "und Satzzeichen sind vor dem Vergleich schon angeglichen. Entscheide für jede Abweichung, ob "
    "das Audio damit eine andere Aussage macht als das Skript.\n\n"
    "Die Aussage ist verändert, wenn jemand beim Zuhören etwas anderes verstehen würde als im "
    "Skript steht: eine andere Zahl, ein anderes Datum, ein anderer Name, ein Wort mit anderer "
    "Bedeutung, eine fehlende oder zusätzliche Verneinung, fehlende oder zusätzliche Wörter mit "
    "Inhalt, eine Frage, die zur Feststellung wird, oder umgekehrt.\n"
    "Die Aussage ist nicht verändert bei anderen Schreibweisen derselben gesprochenen Wörter "
    "(Getrennt- oder Zusammenschreibung, Abkürzungen, gleich klingende Schreibweisen von Namen "
    "und Fremdwörtern), bei fehlenden oder zusätzlichen Füllwörtern ohne Inhalt und bei "
    "Abweichungen in der Grammatik, die den Sinn nicht ändern.\n"
    "Die Aussage ist auch nicht verändert, wenn die Spracherkennung statt eines Namens aus dem "
    "Skript ein Wort gehört hat, das deutsch gesprochen fast gleich klingt, etwa „Indianer“ statt "
    "„Indiana“ oder „Lorch“ statt „Lurch“: Das ist ein Hörfehler der Spracherkennung, kein "
    "Fehler der Sprachsynthese. Das gilt nur bei fast gleichem Klang; klingt das Gehörte "
    "deutlich anders, etwa „Lina“ statt „Lena“, ist die Aussage verändert.\n"
    "Im Zweifel gilt die Aussage als verändert.\n\n"
    "Antworte mit JSON im vorgegebenen Schema: für jede Abweichung ihre Nummer, "
    "„changes_statement“ und eine kurze Begründung auf Deutsch."
)
NO_ANSWER = "Keine Entscheidung zu dieser Abweichung; gilt als verändert."
_NOTHING = "(nichts)"


def format_decision_input(
    turns: Sequence[tuple[str, str]], transcript: str, deviations: Sequence[Deviation]
) -> str:
    """Input of the LLM: the section's script per turn, the transcript, the numbered deviations."""
    lines = ["Skript des Abschnitts:", *(f"{speaker}: {text}" for speaker, text in turns)]
    lines += ["", "Transkript der Spracherkennung:", transcript or _NOTHING, "", "Abweichungen:"]
    for d in deviations:
        around = " ".join(
            part
            for part in (d.context_before, f"[{d.expected or _NOTHING}]", d.context_after)
            if part
        )
        lines.append(f"{d.index}. Skript: „{around}“ – gehört: „{d.heard or _NOTHING}“")
    return "\n".join(lines)


def apply_decisions(deviations: Sequence[Deviation], answer: dict) -> None:
    """Write the LLM's decision into each deviation. A deviation without a decision counts as
    changing the statement; unknown indices and repeated decisions (after the first) are ignored."""
    by_index: dict[int, dict] = {}
    for item in answer.get("decisions", []) or []:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            by_index.setdefault(item["index"], item)
    for d in deviations:
        item = by_index.get(d.index)
        if item is None or not isinstance(item.get("changes_statement"), bool):
            d.changes_statement, d.reason = True, NO_ANSWER
        else:
            d.changes_statement, d.reason = item["changes_statement"], str(item.get("reason", ""))


def changes_statement(deviations: Sequence[Deviation]) -> bool:
    """The section's audio changes the statement if any decided deviation does (or is undecided)."""
    return any(d.changes_statement is not False for d in deviations)


def decide_deviations(
    llm: StructuredLlm,
    turns: Sequence[tuple[str, str]],
    transcript: str,
    deviations: Sequence[Deviation],
) -> None:
    """One LLM call for all deviations of a section; none, no call."""
    if not deviations:
        return
    answer = llm.complete_json(
        DECISION_PROMPT, format_decision_input(turns, transcript, deviations), DECISION_SCHEMA
    )
    apply_decisions(deviations, answer)


# ---------------------------------------------------------------- retry budget


class RetryAction(StrEnum):
    ACCEPT = "accept"
    RESYNTHESIZE = "resynthesize"
    ESCALATE = "escalate"


def retry_action(resyntheses: int, changes_statement: bool, budget: int) -> RetryAction:
    """Audio level of the design's error table: a section whose audio changes the statement is
    resynthesised at most `budget` times; after that the run escalates."""
    if not changes_statement:
        return RetryAction.ACCEPT
    if resyntheses < budget:
        return RetryAction.RESYNTHESIZE
    return RetryAction.ESCALATE
