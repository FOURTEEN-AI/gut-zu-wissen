"""Review loop (phase 1, step 12; design: "Feedback und Kalibrierung", "Stichproben").

Sampling: the population is every accepted turn with fact IDs; each one is drawn with probability
r. The draws come from a random generator seeded with the episode ID and are recorded in
`09_review.json`, so the sample is reproducible from the archive. r is read from
`archive/state/review.json`.

Review issue: the pipeline opens one issue per episode, also when no turn was drawn (r < 100 %);
then the issue is only for the explicit label. It shows per drawn turn the text, fact IDs, fact
sentences, context, judge verdict, confidence and reason, and an answer template. The reviewer
answers in comments, one line per turn: `<Nummer>: belegt` or `<Nummer>: nicht belegt`,
optionally followed by a dash and a note.

Labels: the review workflow reads every comment of the issue again on each run and folds the
reviewer's comments in creation order (a later label for a turn wins), so edits, deletions and
skipped runs are picked up. Comments by any other account are ignored and not recorded. A comment
with an invalid review line is rejected as a whole. Turns without a label stay unknown.

Explicit check (D7): the issue also shows the result of `07_explicit.json` (flag, verdict,
findings, reason). The reviewer confirms or corrects the flag with one line in the same format:
`explicit: korrekt`, `explicit: sollte explicit sein` or `explicit: sollte nicht explicit sein`.
The archive keeps the predicted flag and the reviewer's flag side by side; `explicit_accuracy`
computes false-positive and false-negative rates over the archive (step 13). The explicit label
is required (user decision 2026-11-19): a review is answered only when every drawn turn and the
explicit check have a label. With the turns labelled and the explicit label missing, the reply
asks for it and the issue stays open. Whether a review is clean or has an error follows the turn
labels only; a review without drawn turns is clean.

Closing: when the review becomes answered, the bot posts its status reply and closes the issue as
completed. Comments on the closed issue are still read as corrections; the bot never reopens an
issue and closes it only on the change to answered.

Rate: an issue counts once it is answered. Error-free: r * 2/3 (at least r_min);
a turn labelled "nicht belegt": r * 2 (at most r_max). Each answered episode changes r once; a
later correction that turns an error-free review into one with an error doubles r once more; a
correction the other way does not lower it again. With `adjust_rate = false` (phase 1) r stays
where it is and the answered reviews are only recorded.

Deterministic parts are pure functions; GitHub sits behind the `Issues` interface (github.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal

from gut_zu_wissen.archive import (
    EPISODES_DIR,
    REVIEW_FILE,
    REVIEW_STATE_FILE,
    STATE_DIR,
    episode_dir,
    format_timestamp,
    read_json,
    write_json,
)
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.explicit import ExplicitCheck
from gut_zu_wissen.github import Comment, Issues
from gut_zu_wissen.judge import TurnJudgement, cited_ids
from gut_zu_wissen.ranking import RankedFact

Label = Literal["supported", "not_supported"]
ExplicitLabel = Literal["correct", "should_be_explicit", "should_not_be_explicit"]
Outcome = Literal["clean", "error"]
ReviewStatus = Literal["not_sampled", "pending", "answered"]

LABEL_WORDS: dict[str, Label] = {"belegt": "supported", "nicht belegt": "not_supported"}
LABEL_TEXT = {label: word for word, label in LABEL_WORDS.items()}
ISSUE_TITLE_PREFIX = "Review: "
MARKER = "gut-zu-wissen-review"

# `<Nummer>: <Label>`, the label optionally followed by a dash and a note.
REVIEW_LINE = re.compile(r"^\s*(\d+)\s*:(.*)$")
LABEL_VALUE = re.compile(r"^(nicht\s+belegt|belegt)(?:\s*[-–—]\s*(.*))?$", re.IGNORECASE)
# `explicit: <Label>` for the explicit check (D7), same note syntax.
EXPLICIT_KEY = "explicit"
EXPLICIT_LINE = re.compile(r"^\s*explicit\s*:(.*)$", re.IGNORECASE)
EXPLICIT_VALUE = re.compile(
    r"^(korrekt|sollte\s+nicht\s+explicit\s+sein|sollte\s+explicit\s+sein)"
    r"(?:\s*[-–—]\s*(.*))?$",
    re.IGNORECASE,
)
EXPLICIT_WORDS: dict[str, ExplicitLabel] = {
    "korrekt": "correct",
    "sollte explicit sein": "should_be_explicit",
    "sollte nicht explicit sein": "should_not_be_explicit",
}


# ---------------------------------------------------------------- records


@dataclass
class Draw:
    turn_index: int
    u: float
    sampled: bool


@dataclass
class ReviewFact:
    """A cited fact as the reviewer sees it; field order is the archive field order."""

    id: str
    article: str
    revid: int
    sentence: str
    context: list[str]


@dataclass
class ReviewJudgement:
    decided_by: str
    jev_p: dict[str, float]
    jev_error: str | None
    reason: str | None


@dataclass
class ReviewTurn:
    """A drawn turn with everything needed to judge it, and the reviewer's label."""

    turn_index: int
    speaker: str
    text: str
    previous_text: str | None
    fact_ids: list[str]
    statements: list[dict]
    facts: list[ReviewFact]
    judgement: ReviewJudgement
    label: Label | None = None
    note: str | None = None
    comment_id: int | None = None


@dataclass
class ExplicitReview:
    """Result of the explicit check (07_explicit.json) and the reviewer's label for it."""

    model: str
    predicted: bool | None
    verdict: str | None
    findings: list[dict]
    reason: str | None
    label: ExplicitLabel | None = None
    reviewer_flag: bool | None = None
    note: str | None = None
    comment_id: int | None = None


@dataclass
class CommentRecord:
    """A reviewer comment as processed; the body itself stays on GitHub."""

    comment_id: int
    url: str
    created_at: str
    updated_at: str
    review_lines: int
    applied: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class Review:
    """Content of 09_review.json; field order is the archive field order."""

    episode_id: str
    title: str
    rate: float
    seed: str
    population: list[int]
    draws: list[Draw]
    status: ReviewStatus
    outcome: Outcome | None = None
    rate_applied: list[Outcome] = field(default_factory=list)
    issue: int | None = None
    issue_url: str | None = None
    explicit: ExplicitReview | None = None
    turns: list[ReviewTurn] = field(default_factory=list)
    comments: list[CommentRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> Review:
        turns = [
            ReviewTurn(
                **{
                    **t,
                    "facts": [ReviewFact(**f) for f in t["facts"]],
                    "judgement": ReviewJudgement(**t["judgement"]),
                }
            )
            for t in data["turns"]
        ]
        return cls(
            **{
                **data,
                "draws": [Draw(**d) for d in data["draws"]],
                "explicit": ExplicitReview(**data["explicit"]) if data.get("explicit") else None,
                "turns": turns,
                "comments": [CommentRecord(**c) for c in data["comments"]],
            }
        )

    def sampled(self) -> list[int]:
        return [t.turn_index for t in self.turns]


@dataclass
class RateUpdate:
    episode_id: str
    issue: int | None
    outcome: Outcome
    at: str
    r_before: float
    r_after: float


@dataclass
class RateState:
    """Content of archive/state/review.json; field order is the archive field order."""

    r: float
    updates: list[RateUpdate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> RateState:
        return cls(data["r"], [RateUpdate(**u) for u in data["updates"]])


@dataclass
class ParsedLine:
    turn_index: int
    label: Label
    note: str | None


@dataclass
class ParsedExplicit:
    label: ExplicitLabel
    note: str | None


@dataclass
class ParsedComment:
    lines: list[ParsedLine]
    errors: list[str]
    explicit: ParsedExplicit | None = None

    def count(self) -> int:
        return len(self.lines) + len(self.errors) + (self.explicit is not None)


@dataclass
class ExplicitAccuracy:
    """Explicit check against the reviewer's flag over reviewed episodes (step 13).

    Positive = explicit. False-positive rate: share of episodes the reviewer calls not explicit
    that the check flagged; false-negative rate: share of explicit episodes the check missed.
    A rate without a single episode in its denominator is None.
    """

    episodes: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    false_positive_rate: float | None
    false_negative_rate: float | None


# ---------------------------------------------------------------- sampling (pure)


def review_population(turns: Sequence[Turn], judgements: Sequence[TurnJudgement]) -> list[int]:
    """Design: accepted turns with fact IDs (accepted by Jev or the LLM judge)."""
    accepted = {j.turn_index for j in judgements if j.accepted}
    return [t.index for t in turns if t.index in accepted and cited_ids(t)]


def draw_values(seed: str, count: int) -> list[float]:
    """`count` uniform values in [0, 1); the same seed always gives the same values."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(count)]


def sample(population: Sequence[int], draws: Sequence[float], r: float) -> list[Draw]:
    """Each turn is drawn with probability r: drawn when its value is below r (r = 1: all)."""
    return [Draw(i, u, u < r) for i, u in zip(population, draws, strict=True)]


def build_review(
    episode_id: str,
    title: str,
    turns: Sequence[Turn],
    judgements: Sequence[TurnJudgement],
    facts: Mapping[str, RankedFact],
    rate: float,
    explicit: ExplicitCheck,
) -> Review:
    """The review record of a finished run: population, recorded draws, drawn turns, and the
    result of the explicit check."""
    population = review_population(turns, judgements)
    draws = sample(population, draw_values(episode_id, len(population)), rate)
    drawn = {d.turn_index for d in draws if d.sampled}
    by_index = {j.turn_index: j for j in judgements}
    previous = {t.index: turns[n - 1].text if n else None for n, t in enumerate(turns)}
    review_turns = []
    for turn in turns:
        if turn.index not in drawn:
            continue
        judgement = by_index[turn.index]
        review_turns.append(
            ReviewTurn(
                turn_index=turn.index,
                speaker=turn.speaker,
                text=turn.text,
                previous_text=previous[turn.index],
                fact_ids=list(turn.fact_ids),
                statements=[asdict(s) for s in turn.statements],
                facts=[
                    ReviewFact(
                        i,
                        facts[i].article,
                        facts[i].revid,
                        facts[i].fact.plain,
                        list(facts[i].fact.context_plain),
                    )
                    for i in cited_ids(turn)
                    if i in facts
                ],
                judgement=ReviewJudgement(
                    judgement.decided_by,
                    dict(judgement.jev_p),
                    judgement.jev_error,
                    judgement.reason,
                ),
            )
        )
    return Review(
        episode_id=episode_id,
        title=title,
        rate=rate,
        seed=episode_id,
        population=population,
        draws=draws,
        status=review_status([t.turn_index for t in review_turns], [], True, False),
        explicit=ExplicitReview(
            explicit.model,
            explicit.explicit,
            explicit.verdict,
            [f.to_dict() for f in explicit.findings],
            explicit.reason,
        ),
        turns=review_turns,
    )


# ---------------------------------------------------------------- comments (pure)


def is_authorized(comment: Comment, reviewer: str, associations: Collection[str]) -> bool:
    """Only the reviewer's own account, with a repository association, gives labels."""
    return comment.login == reviewer and comment.association in associations


def parse_explicit(value: str, predicted: bool | None) -> tuple[ParsedExplicit | None, str | None]:
    """The value of an `explicit:` line; a correction that names the predicted flag is an
    error (use „korrekt“), as is „korrekt“ when the check gave no flag."""
    match = EXPLICIT_VALUE.match(value)
    if not match:
        return None, f"Explicit: unbekanntes Label „{value}“."
    label = EXPLICIT_WORDS[" ".join(match.group(1).lower().split())]
    if label == "correct" and predicted is None:
        return None, "Explicit: die Prüfung hat kein Ergebnis, „korrekt“ geht nicht."
    if (label == "should_be_explicit" and predicted is True) or (
        label == "should_not_be_explicit" and predicted is False
    ):
        return None, "Explicit: das sagt die Prüfung schon; dann „korrekt“."
    return ParsedExplicit(label, (match.group(2) or "").strip() or None), None


def parse_comment(
    body: str, sampled: Collection[int], predicted: bool | None = None
) -> ParsedComment:
    """Review lines of one comment. Other lines (prose, code fences, quotes) are ignored.

    `<n>:` or `explicit:` with nothing after it is skipped (template line not filled in). An
    unknown label, a turn that was not drawn or a turn or the explicit line named twice is an
    error. `predicted` is the flag of the explicit check.
    """
    lines: list[ParsedLine] = []
    errors: list[str] = []
    seen: set[int] = set()
    explicit: ParsedExplicit | None = None
    explicit_seen = False
    for raw in body.splitlines():
        explicit_match = EXPLICIT_LINE.match(raw)
        if explicit_match:
            value = " ".join(explicit_match.group(1).split())
            if not value:
                continue
            if explicit_seen:
                errors.append("Explicit steht mehrfach im Kommentar.")
                continue
            explicit_seen = True
            explicit, error = parse_explicit(value, predicted)
            if error:
                errors.append(error)
            continue
        match = REVIEW_LINE.match(raw)
        if not match:
            continue
        index, value = int(match.group(1)), " ".join(match.group(2).split())
        if not value:
            continue
        label_match = LABEL_VALUE.match(value)
        if not label_match:
            errors.append(f"Redebeitrag {index}: unbekanntes Label „{value}“.")
            continue
        if index not in sampled:
            errors.append(f"Redebeitrag {index} ist nicht in der Stichprobe.")
            continue
        if index in seen:
            errors.append(f"Redebeitrag {index} steht mehrfach im Kommentar.")
            continue
        seen.add(index)
        word = " ".join(label_match.group(1).lower().split())
        note = (label_match.group(2) or "").strip() or None
        lines.append(ParsedLine(index, LABEL_WORDS[word], note))
    return ParsedComment(lines, errors, explicit)


@dataclass
class Folded:
    labels: dict[int, tuple[ParsedLine, int]]
    explicit: tuple[ParsedExplicit, int] | None
    records: list[CommentRecord]


def fold_labels(
    comments: Sequence[Comment], sampled: Collection[int], predicted: bool | None = None
) -> Folded:
    """Labels from the reviewer's comments in creation order; a later label for a turn (or for
    the explicit check) wins.

    Returns label and comment ID per turn and for the explicit check, and one record per comment
    with review lines or errors. A comment with an error is not applied at all.
    """
    labels: dict[int, tuple[ParsedLine, int]] = {}
    explicit: tuple[ParsedExplicit, int] | None = None
    records: list[CommentRecord] = []
    for comment in sorted(comments, key=lambda c: (c.created_at, c.id)):
        parsed = parse_comment(comment.body, sampled, predicted)
        if not parsed.count():
            continue
        applied = not parsed.errors
        if applied:
            for line in parsed.lines:
                labels[line.turn_index] = (line, comment.id)
            if parsed.explicit:
                explicit = (parsed.explicit, comment.id)
        records.append(
            CommentRecord(
                comment.id,
                comment.url,
                comment.created_at,
                comment.updated_at,
                parsed.count(),
                applied,
                parsed.errors,
            )
        )
    return Folded(labels, explicit, records)


def review_status(
    sampled: Collection[int],
    labelled: Collection[int],
    explicit_required: bool,
    explicit_labelled: bool,
) -> ReviewStatus:
    """answered: every drawn turn labelled and, when the review has an explicit result, the
    explicit label given. not_sampled: nothing to review (no drawn turn, no explicit result; only
    reviews from before the explicit check)."""
    if not sampled and not explicit_required:
        return "not_sampled"
    turns_done = set(sampled) <= set(labelled)
    return "answered" if turns_done and (explicit_labelled or not explicit_required) else "pending"


def closes_issue(before: ReviewStatus, after: ReviewStatus) -> bool:
    """The issue is closed when its review becomes answered, not on later corrections."""
    return after == "answered" and before != "answered"


def review_outcome(turns: Sequence[ReviewTurn]) -> Outcome:
    return "error" if any(t.label == "not_supported" for t in turns) else "clean"


def apply_labels(review: Review, comments: Sequence[Comment]) -> None:
    """Write labels, comment records, status and outcome into `review` (authorized comments
    only; the caller filters)."""
    predicted = review.explicit.predicted if review.explicit else None
    folded = fold_labels(comments, review.sampled(), predicted)
    labels, records = folded.labels, folded.records
    if review.explicit:
        parsed, comment_id = folded.explicit or (None, None)
        review.explicit.label = parsed.label if parsed else None
        review.explicit.note = parsed.note if parsed else None
        review.explicit.comment_id = comment_id
        review.explicit.reviewer_flag = reviewer_flag(predicted, review.explicit.label)
    for turn in review.turns:
        line, comment_id = labels.get(turn.turn_index, (None, None))
        turn.label = line.label if line else None
        turn.note = line.note if line else None
        turn.comment_id = comment_id
    review.comments = records
    review.status = review_status(
        review.sampled(),
        labels,
        review.explicit is not None,
        review.explicit is not None and review.explicit.label is not None,
    )
    review.outcome = review_outcome(review.turns) if review.status == "answered" else None


# ---------------------------------------------------------------- explicit check (pure)


def reviewer_flag(predicted: bool | None, label: ExplicitLabel | None) -> bool | None:
    """The flag the reviewer gives the episode: the predicted one if confirmed, else the
    corrected one; None without a label."""
    if label == "correct":
        return predicted
    if label == "should_be_explicit":
        return True
    if label == "should_not_be_explicit":
        return False
    return None


def explicit_pairs(reviews: Iterable[Review]) -> list[tuple[bool, bool]]:
    """(predicted, reviewer) flags of every episode whose explicit result has a label."""
    pairs = []
    for review in reviews:
        e = review.explicit
        if e is not None and e.predicted is not None and e.reviewer_flag is not None:
            pairs.append((e.predicted, e.reviewer_flag))
    return pairs


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def explicit_confusion(pairs: Iterable[tuple[bool, bool]]) -> ExplicitAccuracy:
    """Confusion counts and rates of the explicit check, positive = explicit."""
    pairs = list(pairs)
    tp = sum(1 for p, r in pairs if p and r)
    fp = sum(1 for p, r in pairs if p and not r)
    fn = sum(1 for p, r in pairs if not p and r)
    tn = sum(1 for p, r in pairs if not p and not r)
    return ExplicitAccuracy(len(pairs), tp, fp, fn, tn, _share(fp, fp + tn), _share(fn, fn + tp))


# ---------------------------------------------------------------- rate (pure)


def rate_due(applied: Sequence[Outcome], outcome: Outcome | None) -> Outcome | None:
    """The rate step an episode is due now: its first answered outcome, or an error found by a
    later correction of an error-free review. Never a second step for the same outcome."""
    if outcome is None or outcome in applied:
        return None
    if applied and outcome == "clean":
        return None  # a correction back to error-free does not lower r again
    return outcome


def next_rate(r: float, outcome: Outcome, review_config: Mapping) -> float:
    """Error-free: r * r_factor_clean, at least r_min. Error: r * r_factor_error, at most r_max."""
    if outcome == "clean":
        value = Fraction(r) * Fraction(str(review_config["r_factor_clean"]))
        return float(max(value, Fraction(review_config["r_min"])))
    value = Fraction(r) * Fraction(str(review_config["r_factor_error"]))
    return float(min(value, Fraction(review_config["r_max"])))


def update_rate(state: RateState, review: Review, review_config: Mapping, now: datetime) -> bool:
    """Apply the due rate step of `review` to `state`; returns whether one was due."""
    due = rate_due(review.rate_applied, review.outcome)
    if due is None:
        return False
    before = state.r
    after = next_rate(before, due, review_config) if review_config["adjust_rate"] else before
    state.r = after
    state.updates.append(
        RateUpdate(review.episode_id, review.issue, due, format_timestamp(now), before, after)
    )
    review.rate_applied.append(due)
    return True


# ---------------------------------------------------------------- rendering (pure)


def escape(text: str) -> str:
    """Text from the pipeline as Markdown: no HTML, no @-mentions."""
    return text.replace("<", "&lt;").replace(">", "&gt;").replace("@", "&#64;")


def quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in escape(text).splitlines() or [""])


def percent(r: float) -> str:
    return f"{r * 100:.4g} %".replace(".", ",")


def format_p(jev_p: Mapping[str, float]) -> str:
    return ", ".join(f"{key} = {p:.3f}" for key, p in jev_p.items())


NO_TURN_NOTE = " Kein Redebeitrag gezogen: dieses Issue dient nur dem Explicit-Label."
JUDGE_TEXT = {"jev": "Jev", "llm_judge": "LLM-Judge"}


def render_turn(turn: ReviewTurn) -> str:
    parts = [f"### Redebeitrag {turn.turn_index} – {turn.speaker}", "", quote(turn.text)]
    if turn.previous_text is not None:
        parts += ["", "<details><summary>Vorheriger Redebeitrag</summary>", ""]
        parts += [quote(turn.previous_text), "", "</details>"]
    parts += ["", f"**Fakt-IDs:** {', '.join(f'`{i}`' for i in turn.fact_ids) or '–'}"]
    if turn.statements:
        parts += ["", "**Aussagen:**"]
        for s in turn.statements:
            ids = ", ".join(f"`{i}`" for i in s["fact_ids"]) or "keine Fakt-IDs"
            parts.append(f"- {escape(s['text'])} ({ids})")
    parts += ["", "**Fakt-Sätze:**"]
    for f in turn.facts:
        sentence = escape(f.sentence.replace("\n", " "))
        parts.append(f"- `{f.id}` ({escape(f.article)}, Revision {f.revid}): {sentence}")
        for c in f.context:
            parts.append(f"  - Kontext: {escape(c.replace(chr(10), ' '))}")
    j = turn.judgement
    parts += ["", f"- **Judge:** angenommen von {JUDGE_TEXT.get(j.decided_by, j.decided_by)}"]
    if j.jev_p:
        parts.append(f"- **Konfidenz (Jev p):** {format_p(j.jev_p)}")
    if j.jev_error:
        parts.append(f"- **Jev-Fehler:** {escape(j.jev_error)}")
    if j.reason:
        parts.append(f"- **Begründung:** {escape(j.reason)}")
    return "\n".join(parts)


CATEGORY_TEXT = {
    "violence": "Gewalt",
    "sexuality": "Sexualität",
    "drugs": "Drogen",
    "profanity": "Schimpfwörter",
}


def flag_text(flag: bool | None) -> str:
    if flag is None:
        return "kein Ergebnis"
    return "explicit" if flag else "nicht explicit"


def render_explicit(e: ExplicitReview) -> str:
    """The result of the explicit check (D7) as the reviewer sees it."""
    parts = [
        "## Explicit-Prüfung (D7)",
        "",
        f"- **Flag:** {flag_text(e.predicted)}",
        f"- **Urteil des Modells:** `{escape(e.verdict or '–')}`",
        f"- **Begründung:** {escape(e.reason or '–')}",
        "- **Fundstellen:**" + ("" if e.findings else " keine"),
    ]
    for f in e.findings:
        index = f.get("turn_index")
        where = f"Redebeitrag {index}" if index is not None else "ohne Redebeitrag"
        category = CATEGORY_TEXT.get(str(f.get("category")), escape(str(f.get("category", ""))))
        parts.append(
            f"  - {where}, {category}: „{escape(str(f.get('quote', '')))}“ – "
            f"{escape(str(f.get('reason', '')))}"
        )
    parts += [
        "",
        "Antwort: `explicit: korrekt`, `explicit: sollte explicit sein` oder "
        "`explicit: sollte nicht explicit sein`, optional mit Notiz nach einem Gedankenstrich. "
        "Explicit ist eine Folge mit drastischer Darstellung von Gewalt, Sexualität oder Drogen "
        "oder mit gesprochenen Schimpfwörtern; sachliche oder beiläufige Erwähnungen zählen "
        "nicht. Ein unsicheres Urteil des Modells gilt als explicit (D7); du beurteilst, ob die "
        "Folge tatsächlich explicit ist. Geprüft wird das ganze Skript (`05_dialog.json`), nicht "
        "nur die gezogenen Redebeiträge.",
    ]
    return "\n".join(parts)


def render_header(review: Review, reviewer: str, links: Mapping[str, str]) -> str:
    data: dict = {"episode_id": review.episode_id, "turns": review.sampled()}
    if review.explicit:
        data["explicit"] = review.explicit.predicted
    marker = json.dumps(data)
    lines = [f"{i}: " for i in review.sampled()]
    if review.explicit:
        lines.append(f"{EXPLICIT_KEY}: ")
    template = "\n".join(lines)
    link_parts = [f"[{name}]({url})" for name, url in links.items()]
    return "\n".join(
        [
            f"<!-- {MARKER} {marker} -->",
            f"Folge `{review.episode_id}` zum Artikel „{escape(review.title)}“. "
            f"Stichprobenrate r = {percent(review.rate)}; gezogen: {len(review.turns)} von "
            f"{len(review.population)} angenommenen Redebeiträgen mit Fakt-IDs."
            + (NO_TURN_NOTE if not review.turns else ""),
            *(["", " · ".join(link_parts)] if link_parts else []),
            "",
            "## So antwortest du",
            "",
            "Pro Redebeitrag eine Zeile `<Nummer>: belegt` oder `<Nummer>: nicht belegt`, "
            "optional mit Notiz nach einem Gedankenstrich, zum Beispiel "
            "`7: nicht belegt – Jahreszahl steht nicht im Fakt-Satz`.",
            "",
            "- **belegt:** Jede Behauptung, Meinung und Prämisse einer Frage folgt aus den "
            "zitierten Fakt-Sätzen; was nur im Kontext steht, zählt nicht als Beleg. Alles "
            "Übrige ist eine Reaktion nach dem Reaktionskriterium (`docs/persona.md`): Gefühl im "
            "Moment, Nachfrage, Bitte um Erklärung, Überleitung, Wiederaufgreifen ohne neue "
            "Information.",
            "- **nicht belegt:** sonst, auch bei einer Aussage der Personas über sich selbst.",
            "",
            f"Es zählen nur Kommentare von `{reviewer}`. Ein späteres Label für denselben "
            "Redebeitrag ersetzt das frühere; leere Zeilen `<Nummer>:` werden übersprungen. "
            "Ein Kommentar mit einer ungültigen Zeile wird ganz verworfen. Labels und Notizen "
            "stehen öffentlich im Archiv. Das Review zählt, sobald jeder gezogene Redebeitrag "
            "und die Explicit-Prüfung ein Label haben; dann wird das Issue geschlossen. "
            "Korrekturen gehen auch danach noch, als Kommentar im geschlossenen Issue.",
            *(["", render_explicit(review.explicit)] if review.explicit else []),
            "",
            "Vorlage zum Kopieren:",
            "",
            "```",
            template,
            "```",
        ]
    )


def pack(header: str, sections: Sequence[str], max_chars: int, continuation: str) -> list[str]:
    """Issue body and follow-up comments, each at most `max_chars`, split at turn boundaries.
    A single section longer than the limit is cut with a pointer to the archive."""
    cut_note = f"\n\n… (gekürzt; vollständig in `{REVIEW_FILE}`)"
    section_max = max_chars - len(continuation) - 2  # a section always fits a follow-up
    messages: list[str] = []
    current = header
    for section in sections:
        if len(section) > section_max:
            section = section[: section_max - len(cut_note)] + cut_note
        if len(current) + 2 + len(section) <= max_chars:
            current += "\n\n" + section
            continue
        messages.append(current)
        current = continuation + "\n\n" + section
    messages.append(current)
    return messages


def render_issue(
    review: Review, reviewer: str, links: Mapping[str, str], max_chars: int
) -> tuple[str, list[str]]:
    """Title and messages (body first, then follow-up comments) of the review issue."""
    title = f"{ISSUE_TITLE_PREFIX}{review.title} ({review.episode_id})"
    header = render_header(review, reviewer, links)
    sections = ["## Redebeiträge", *(render_turn(t) for t in review.turns)] if review.turns else []
    continuation = f"## Redebeiträge (Fortsetzung, Folge `{review.episode_id}`)"
    return title, pack(header, sections, max_chars, continuation)


def render_reply(review: Review, rate_update: RateUpdate | None, closing: bool = False) -> str:
    """Status comment after processing the reviewer's comments; `closing`: the issue is closed
    after this reply."""
    labelled = [t for t in review.turns if t.label]
    lines = [
        f"Labels übernommen: {len(labelled)} von {len(review.turns)} Redebeiträgen "
        f"(Folge `{review.episode_id}`)."
    ]
    missing = [str(t.turn_index) for t in review.turns if not t.label]
    if missing:
        lines.append(f"Ohne Label: {', '.join(missing)}.")
    errors = [t.turn_index for t in review.turns if t.label == "not_supported"]
    if errors:
        lines.append(f"Nicht belegt: {', '.join(str(i) for i in errors)}.")
    for c in review.comments:
        if c.errors:
            lines.append("")
            lines.append(f"Kommentar {c.url} verworfen:")
            lines += [f"- {e}" for e in c.errors]
    if review.explicit:
        e = review.explicit
        if e.label is None:
            lines.append(f"Explicit: noch ohne Label (Prüfung: {flag_text(e.predicted)}).")
        else:
            lines.append(
                f"Explicit: Prüfung {flag_text(e.predicted)}, Review {flag_text(e.reviewer_flag)}."
            )
    if review.status == "answered":
        result = "fehlerfrei" if review.outcome == "clean" else "Fehler gefunden"
        lines += ["", f"Review vollständig: {result}."]
    if rate_update is not None:
        if rate_update.r_after == rate_update.r_before:
            lines.append(
                f"Stichprobenrate bleibt bei {percent(rate_update.r_after)} "
                "(Phase 1: feste Rate bis zu den Abschlusskriterien)."
            )
        else:
            lines.append(
                f"Stichprobenrate: {percent(rate_update.r_before)} → "
                f"{percent(rate_update.r_after)}."
            )
    explicit_missing = review.explicit is not None and review.explicit.label is None
    if review.status == "pending" and not missing and explicit_missing:
        lines += [
            "",
            "Bitte noch das Explicit-Label ergänzen: `explicit: korrekt`, `explicit: sollte "
            "explicit sein` oder `explicit: sollte nicht explicit sein`. Erst dann ist das "
            "Review vollständig und das Issue wird geschlossen.",
        ]
    if closing:
        lines += ["", "Das Issue wird geschlossen."]
        lines.append(
            "Spätere Kommentare werden weiter übernommen (Korrekturen); das Issue bleibt "
            "geschlossen."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- archive


def state_path(archive_root: Path) -> Path:
    return archive_root / STATE_DIR / REVIEW_STATE_FILE


def load_rate_state(archive_root: Path, review_config: Mapping) -> RateState:
    path = state_path(archive_root)
    if not path.exists():
        return RateState(float(review_config["r_initial"]))
    return RateState.from_dict(read_json(path))


def load_review(archive_root: Path, episode_id: str) -> Review:
    return Review.from_dict(read_json(episode_dir(archive_root, episode_id) / REVIEW_FILE))


def save_review(archive_root: Path, review: Review) -> None:
    write_json(episode_dir(archive_root, review.episode_id) / REVIEW_FILE, review.to_dict())


def find_review(archive_root: Path, issue: int) -> Review | None:
    """The review whose issue number is `issue`. The archive decides, not the issue text."""
    for path in sorted((archive_root / EPISODES_DIR).glob(f"*/{REVIEW_FILE}")):
        data = read_json(path)
        if data.get("issue") == issue:
            return Review.from_dict(data)
    return None


def explicit_accuracy(archive_root: Path) -> ExplicitAccuracy:
    """False-positive and false-negative rates of the explicit check over every reviewed
    episode in the archive (step 13)."""
    reviews = [
        Review.from_dict(read_json(path))
        for path in sorted((archive_root / EPISODES_DIR).glob(f"*/{REVIEW_FILE}"))
    ]
    return explicit_confusion(explicit_pairs(reviews))


def open_issue(
    archive_root: Path,
    episode_id: str,
    issues: Issues,
    review_config: Mapping,
    links: Mapping[str, str],
) -> Review:
    """Open the review issue of an episode and record its number; without a drawn turn the issue
    is only for the explicit label. Nothing to review (`not_sampled`, only reviews from before
    the explicit check): no issue. An issue that is already recorded is not opened again."""
    review = load_review(archive_root, episode_id)
    if review.status == "not_sampled" or review.issue is not None:
        return review
    title, messages = render_issue(
        review, review_config["reviewer"], links, review_config["issue_max_chars"]
    )
    number, url = issues.create_issue(title, messages[0])
    review.issue, review.issue_url = number, url
    save_review(archive_root, review)  # recorded before the follow-ups, so a failure keeps it
    for message in messages[1:]:
        issues.create_comment(number, message)
    return review


@dataclass
class ApplyResult:
    review: Review | None
    rate_update: RateUpdate | None
    reply: str | None
    close: bool = False  # close the issue after posting the reply


def apply_issue(
    archive_root: Path,
    issue: int,
    issues: Issues,
    review_config: Mapping,
    now: datetime,
) -> ApplyResult:
    """Read every comment of `issue`, keep the reviewer's, write labels and the rate state.
    `close` is set when the review became answered with these comments."""
    review = find_review(archive_root, issue)
    if review is None:
        return ApplyResult(None, None, None)
    reviewer, associations = review_config["reviewer"], review_config["reviewer_associations"]
    listed = issues.list_comments(issue)
    for c in listed:  # login and association are public; the log shows why a comment is skipped
        ok = is_authorized(c, reviewer, associations)
        print(
            f"Comment {c.id}: login={c.login} association={c.association} authorized={ok}",
            file=sys.stderr,
        )
    comments = [c for c in listed if is_authorized(c, reviewer, associations)]
    before = review.status
    apply_labels(review, comments)
    close = closes_issue(before, review.status)
    state = load_rate_state(archive_root, review_config)
    rate_update = None
    if update_rate(state, review, review_config, now):
        rate_update = state.updates[-1]
        write_json(state_path(archive_root), state.to_dict())
    save_review(archive_root, review)
    return ApplyResult(review, rate_update, render_reply(review, rate_update, close), close)


# ---------------------------------------------------------------- command line


def run_links(env: Mapping[str, str], episode_id: str) -> dict[str, str]:
    """Links in the issue: archive folder on the default branch and the workflow run (audio)."""
    server, repo = env.get("GITHUB_SERVER_URL", "https://github.com"), env.get("GITHUB_REPOSITORY")
    if not repo:
        return {}
    links = {"Archiv": f"{server}/{repo}/tree/main/archive/{EPISODES_DIR}/{episode_id}"}
    if env.get("GITHUB_RUN_ID"):
        links["Audio (Workflow-Lauf)"] = f"{server}/{repo}/actions/runs/{env['GITHUB_RUN_ID']}"
    return links


def _issues_from_env() -> Issues:
    from gut_zu_wissen.github import HttpIssues

    return HttpIssues(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_TOKEN"],
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )


def _summary(text: str) -> None:
    from gut_zu_wissen.run import write_job_summary

    write_job_summary(text)


def main(argv: Iterable[str] | None = None) -> int:
    from gut_zu_wissen.run import ARCHIVE_ROOT, load_config  # run imports this module

    parser = argparse.ArgumentParser(description="Review issues (phase 1, step 12).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_open = sub.add_parser("open", help="open the review issue of an episode")
    p_open.add_argument("episode_id")
    p_apply = sub.add_parser("apply", help="write the reviewer's labels of an issue")
    p_apply.add_argument("issue", type=int)
    p_apply.add_argument("--reply-file", type=Path, required=True)
    p_apply.add_argument(
        "--close-file", type=Path, required=True, help="written when the issue is to be closed"
    )
    p_reply = sub.add_parser("reply", help="post the status comment written by apply")
    p_reply.add_argument("issue", type=int)
    p_reply.add_argument("reply_file", type=Path)
    p_reply.add_argument("--close-file", type=Path, help="close the issue when this file exists")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    review_config = load_config()["review"]

    if args.command == "open":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9-]+", args.episode_id):
            print(f"::error::Not an episode ID: {args.episode_id}", file=sys.stderr)
            return 2
        links = run_links(os.environ, args.episode_id)
        review = open_issue(ARCHIVE_ROOT, args.episode_id, _issues_from_env(), review_config, links)
        if review.issue is None:
            print(f"Episode {args.episode_id}: nothing to review, no review issue.")
            _summary("Nothing to review (no drawn turn, no explicit result); no review issue.")
        else:
            print(f"Episode {args.episode_id}: review issue #{review.issue} ({review.issue_url}).")
            _summary(f"Review issue: {review.issue_url} ({len(review.turns)} turns drawn).")
        return 0

    if args.command == "apply":
        now = datetime.now(UTC)
        result = apply_issue(ARCHIVE_ROOT, args.issue, _issues_from_env(), review_config, now)
        args.reply_file.parent.mkdir(parents=True, exist_ok=True)
        args.close_file.parent.mkdir(parents=True, exist_ok=True)
        args.close_file.unlink(missing_ok=True)  # a rejected push reruns apply
        if result.review is None or result.reply is None:
            args.reply_file.unlink(missing_ok=True)
            print(f"Issue #{args.issue} is not a review issue in the archive; nothing to do.")
            _summary(f"Issue #{args.issue} is not a review issue in the archive; nothing done.")
            return 0
        args.reply_file.write_text(result.reply + "\n", encoding="utf-8")
        if result.close:
            args.close_file.write_text(f"{args.issue}\n", encoding="utf-8")
        review = result.review
        labelled = sum(1 for t in review.turns if t.label)
        print(
            f"Issue #{args.issue} ({review.episode_id}): {labelled} of {len(review.turns)} "
            f"turns labelled, status {review.status}, outcome {review.outcome}."
        )
        _summary(
            f"## Review {review.episode_id}\n\nIssue #{args.issue}: {labelled} of "
            f"{len(review.turns)} turns labelled, status `{review.status}`, "
            f"outcome `{review.outcome}`." + (" The issue is closed." if result.close else "")
        )
        return 0

    if not args.reply_file.exists():
        return 0
    issues = _issues_from_env()
    issues.create_comment(args.issue, args.reply_file.read_text(encoding="utf-8"))
    if args.close_file is not None and args.close_file.exists():
        issues.close_issue(args.issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
