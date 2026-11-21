"""STT verification (phase 1, step 10; design: "Verifikation", "Abweichung", "Fehlerbehandlung").

Whisper's transcript of a section is compared word by word with the section's script. Both
sides are normalised first, so that differences in writing alone do not count:

- case (casefold, so ß and ss are equal), punctuation, quotes, apostrophes inside words,
  hyphens and slashes (word separators on both sides);
- German number words become numerals, as Whisper writes them: cardinals up to 999,999
  ("achtzehnhundertneunundachtzig" = 1889, "eine" = 1), ordinals ("dritten" = 3, as "3."
  loses its dot), decades ("zwanziger" = "20er") and "zwei Komma fünf" = "2,5", also digit by
  digit after the comma ("Komma null acht" = ",08");
- numerals: thousands separators ("10.000", "10 000"), decimal point or comma;
- a few symbols Whisper writes for spoken words (%, ‰, °, €, $, &).

The normalisation, the word diff and the retry decision are pure functions. All remaining
deviations of a section go to the LLM judge model in one call; it decides per deviation whether
the audio changes the statement ("im Zweifel verändert"). A deviation it leaves undecided counts
as a change.

Second hearing (decided 2026-11-21): Whisper mishears at random, so every deviation is heard once
more before the LLM decides; a deviation the second hearing clears is not sent to the LLM. The
word times of the transcript locate the deviation in the audio (`script_times`,
`hearing_window`); the window with a few script words on each side is cut out and transcribed
again. If that transcript has no deviation at the place of the first one
(`heard_again_matches`), the audio says what the script says and the deviation does not count.
Only a remaining deviation that the LLM finds to change the statement leads to resynthesis
(`retry_action`).

Third hearing with a hint (decided 2026-11-21): Whisper writes a rare word it does not know as a
common one that sounds alike ("Rainald" as "Reinhard"), in every synthesis and every hearing, so
resynthesis cannot help. A deviation that still counts after the LLM is heard a third time, with
the expected word given to Whisper as a hint, if `hint_candidate` allows it: one word on each
side, the expected word from the cited fact sentences, no number and no excluded word (negations,
months), and the two words alike (`similarity`). If Whisper with the hint hears the script there
(`hint_matches`), the deviation counts as a mishearing and the pipeline adds the heard word to the
episode's pronunciation lexicon. If it still hears the other word, the voice probably says it, and
the deviation counts as before.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.stt import Word

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
    # Position in the normalised words: script words [script_start, script_end), transcript words
    # [heard_start, heard_end).
    script_start: int = 0
    script_end: int = 0
    heard_start: int = 0
    heard_end: int = 0
    heard_again: str | None = None  # transcript of the second hearing; None: not heard again
    heard_again_matches: bool | None = None  # the second hearing has no deviation at this place
    heard_with_hint: str | None = None  # transcript of the third hearing; None: not heard so
    hint_matches: bool | None = None  # the third hearing, with the hint, has no deviation here

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
    """Join "10 000" to 10000 and "zwei Komma fünf" to 2,5.

    German says decimals digit by digit after the comma ("Komma null acht"): each further single
    digit word extends the fraction (576,08), as long as the fraction so far is single digit words.
    """
    out: list[Token] = []
    digit_by_digit = False  # out[-1] is a decimal whose fraction is single digit words
    for token in tokens:
        single_digit_word = len(token.norm) == 1 and token.norm.isdigit() and token.raw.isalpha()
        if (
            out
            and len(token.raw) == 3
            and token.raw.isdigit()
            and out[-1].raw.isdigit()
            and len(out[-1].raw) <= 3
        ):
            prev = out.pop()
            token = Token(str(int(prev.raw + token.raw)), f"{prev.raw} {token.raw}")
            digit_by_digit = False
        elif (
            len(out) >= 2
            and token.norm.isdigit()
            and out[-1].norm == _DECIMAL_WORD
            and out[-2].norm.isdigit()
        ):
            sep, prev = out.pop(), out.pop()
            token = Token(f"{prev.norm},{token.norm}", f"{prev.raw} {sep.raw} {token.raw}")
            digit_by_digit = single_digit_word
        elif out and digit_by_digit and single_digit_word:
            prev = out.pop()
            token = Token(prev.norm + token.norm, f"{prev.raw} {token.raw}")
        else:
            digit_by_digit = False
        out.append(token)
    return out


# ---------------------------------------------------------------- diff


def _opcodes(expected: Sequence[Token], heard: Sequence[Token]) -> list[tuple]:
    matcher = difflib.SequenceMatcher(
        None, [t.norm for t in expected], [t.norm for t in heard], autojunk=False
    )
    return matcher.get_opcodes()


def diff(script: str, transcript: str, context_words: int) -> list[Deviation]:
    """Word differences between the normalised script and transcript, in script order.

    Each deviation carries the words as written and `context_words` script words on each side.
    """
    return diff_tokens(tokenize(script), tokenize(transcript), context_words)


def diff_tokens(
    expected: Sequence[Token], heard: Sequence[Token], context_words: int
) -> list[Deviation]:
    """`diff` on words that are already normalised."""
    deviations: list[Deviation] = []
    for tag, i1, i2, j1, j2 in _opcodes(expected, heard):
        if tag == "equal":
            continue
        deviations.append(
            Deviation(
                len(deviations),
                _raw(expected[i1:i2]),
                _raw(heard[j1:j2]),
                _raw(expected[max(0, i1 - context_words) : i1]),
                _raw(expected[i2 : i2 + context_words]),
                script_start=i1,
                script_end=i2,
                heard_start=j1,
                heard_end=j2,
            )
        )
    return deviations


# ---------------------------------------------------------------- word times

Span = tuple[float, float]


def token_times(heard: Sequence[Token], words: Sequence[Word]) -> list[Span | None]:
    """Time of each normalised transcript word, from Whisper's word times.

    The transcript (canonicalised by the lexicon) and the words (as Whisper wrote them) are
    normalised separately and aligned; a transcript word without a match has no time.
    """
    timed: list[tuple[Token, Span]] = []
    for word in words:
        for token in tokenize(word.text):
            timed.append((token, (word.start, word.end)))
    times: list[Span | None] = [None] * len(heard)
    for tag, i1, i2, j1, _ in _opcodes(heard, [t for t, _ in timed]):
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = timed[j1 + k][1]
    return times


def script_times(
    expected: Sequence[Token], heard: Sequence[Token], heard_times: Sequence[Span | None]
) -> list[Span | None]:
    """Time of each script word (same alignment as `diff`).

    A word the transcript has unchanged gets the time of its transcript word. Words that Whisper
    heard as other words get the time of all those words together (where the audio says them).
    Words that Whisper did not hear at all have no time.
    """
    times: list[Span | None] = [None] * len(expected)
    for tag, i1, i2, j1, j2 in _opcodes(expected, heard):
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = heard_times[j1 + k]
        elif tag == "replace":
            known = [t for t in heard_times[j1:j2] if t is not None]
            if known:
                for k in range(i1, i2):
                    times[k] = (known[0][0], known[-1][1])
    return times


def hearing_window(
    times: Sequence[Span | None],
    start: int,
    end: int,
    context_words: int,
    duration: float,
    margin: float,
) -> tuple[int, int, float, float]:
    """Script words [first, last) and audio [t0, t1) to hear again for script words [start, end).

    The window reaches `context_words` script words beyond the deviation on each side, further
    out up to the nearest word with a time; without one, to the edge of the audio. `margin`
    seconds are added on both sides.
    """
    first = max(0, start - context_words)
    while first > 0 and times[first] is None:
        first -= 1
    last = min(len(times), end + context_words)  # exclusive
    while last < len(times) and times[last - 1] is None:
        last += 1
    t0 = times[first][0] if first < len(times) and times[first] is not None else 0.0
    t1 = times[last - 1][1] if last > 0 and times[last - 1] is not None else duration
    if first == 0:
        t0 = 0.0
    if last == len(times):
        t1 = duration
    return first, last, max(0.0, t0 - margin), min(duration, t1 + margin)


def overlaps(d: Deviation, start: int, end: int) -> bool:
    """Whether deviation `d` lies in script words [start, end); an insertion (empty range) counts
    where it touches the range."""
    if d.script_start == d.script_end or start == end:
        return start <= d.script_end and d.script_start <= end
    return d.script_start < end and start < d.script_end


def heard_again_matches(
    expected: Sequence[Token], heard: Sequence[Token], first: int, d: Deviation
) -> bool:
    """Whether the second hearing (`heard`, for script words `expected` starting at `first`) has
    no deviation at the place of `d`."""
    start, end = d.script_start - first, d.script_end - first
    return not any(overlaps(x, start, end) for x in diff_tokens(expected, heard, 0))


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
    "Ebenso ist die Aussage nicht verändert, wenn die Spracherkennung ein kleines Wort oder eine "
    "Endung nur verschluckt oder zusätzlich hört (etwa „Zeigt dir“ statt „Zeigte“ oder „Grieche“ "
    "statt „Griechisch“), weil gesprochen ein „-e“ oder „-sch“ leicht untergeht, und der Satz "
    "dabei seinen Sinn behält: gleiche Frage oder Feststellung, gleiche Beteiligte, gleiche Zeit "
    "und Zahl. Das gilt nicht für Zahlen, Daten, Namen, Verneinungen und Wörter mit anderer "
    "Bedeutung; dort bleibt jede Abweichung eine Änderung.\n"
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


def counts(d: Deviation) -> bool:
    """A deviation changes the statement if neither the second nor the third hearing heard the
    script and the LLM did not clear it (undecided counts)."""
    return (
        d.changes_statement is not False
        and d.heard_again_matches is not True
        and d.hint_matches is not True
    )


def changes_statement(deviations: Sequence[Deviation]) -> bool:
    """The audio changes the statement if any deviation counts (`counts`)."""
    return any(counts(d) for d in deviations)


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


# ---------------------------------------------------------------- third hearing with a hint


def similarity(a: str, b: str) -> float:
    """How alike two words are written, 0 to 1 (difflib ratio, case-insensitive)."""
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold(), autojunk=False).ratio()


def fact_words(texts: Sequence[str]) -> frozenset[str]:
    """The words (letters only, casefolded) of the given fact sentences."""
    return frozenset(t.raw.casefold() for text in texts for t in tokenize(text) if t.raw.isalpha())


def hint_candidate(
    d: Deviation,
    words: Collection[str],
    min_similarity: float,
    excluded: Collection[str],
) -> bool:
    """Whether deviation `d` may be heard again with its expected word as a hint (pure).

    One word on each side, both letters only (no number), the expected word one of the cited fact
    sentences' `words`, neither word in `excluded` (casefolded), and the words at least
    `min_similarity` alike.
    """
    if d.script_end - d.script_start != 1 or d.heard_end - d.heard_start != 1:
        return False
    expected, heard = d.expected, d.heard
    if not (expected.isalpha() and heard.isalpha()):
        return False
    skip = {w.casefold() for w in excluded}
    if expected.casefold() not in words or {expected.casefold(), heard.casefold()} & skip:
        return False
    return similarity(expected, heard) >= min_similarity


def add_heard(additions: list[dict], written: str, heard: str, note: str) -> None:
    """Record `heard` as a variant of `written` in the lexicon additions (in place); a second
    variant of the same word joins its entry."""
    for entry in additions:
        if entry["written"] == written:
            if heard not in entry["heard"]:
                entry["heard"].append(heard)
            return
    additions.append({"written": written, "spoken": written, "heard": [heard], "note": note})


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
