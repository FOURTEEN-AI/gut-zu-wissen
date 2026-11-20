"""Markup gate (phase 1, step 9; design: "Audio", "Vorbereitung").

The script that goes to TTS must not contain wiki markup. A finding is a failure of the turn,
which the orchestrator regenerates like any other rejected turn. The rules are pure functions.

Rules: links (`[[`, `]]`, external `[http…`), templates and tables (`{{`, `}}`, `{|`), the pipe,
bold and italic quotes (`''`), headings (`==`), HTML tags and comments, HTML entities, magic words
(`__TOC__`), signatures (`~~~`), list, indent and rule markers at the start of a line, and
rendered footnote markers such as `[1]`, which would be read aloud.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from gut_zu_wissen.dialog import Turn, TurnProblem

MARKUP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Wiki-Link", re.compile(r"\[\[|\]\]")),
    ("externer Link", re.compile(r"\[(?:https?:)?//")),
    ("Vorlage oder Tabelle", re.compile(r"\{\{|\}\}|\{\|")),
    ("senkrechter Strich", re.compile(r"\|")),
    ("Fett- oder Kursiv-Apostrophe", re.compile(r"''")),
    ("Überschrift", re.compile(r"==")),
    ("HTML-Tag oder -Kommentar", re.compile(r"</?[A-Za-z!][^>]*>")),
    ("HTML-Entität", re.compile(r"&(?:[A-Za-z]+|#\d+|#x[0-9A-Fa-f]+);")),
    ("Magic Word", re.compile(r"__[A-Z]+__")),
    ("Signatur", re.compile(r"~~~")),
    ("Zeichen am Zeilenanfang", re.compile(r"^[ \t]*(?:[*#:;]|-{4})", re.MULTILINE)),
    ("Fußnotenmarker", re.compile(r"\[\d+\]")),
)


def find_markup(text: str) -> list[str]:
    """Names of the markup rules the text violates, with the first match, in rule order."""
    found: list[str] = []
    for name, pattern in MARKUP_RULES:
        match = pattern.search(text)
        if match:
            found.append(f"{name} '{match.group(0).strip()}'")
    return found


def markup_problems(turns: Sequence[Turn]) -> list[TurnProblem]:
    """Gate: one problem per turn whose spoken text contains markup."""
    problems: list[TurnProblem] = []
    for turn in turns:
        found = find_markup(turn.text)
        if found:
            problems.append(
                TurnProblem(
                    turn.index,
                    f"Markup im Text: {', '.join(found)}. Schreibe reinen Sprechtext ohne Markup.",
                )
            )
    return problems
