"""Review loop (phase 1, step 12; design: "Feedback und Kalibrierung", "Stichproben").

Sampling: the population is every accepted turn with fact IDs; each one is drawn with probability
r. The draws come from a random generator seeded with the episode ID and are recorded in
`09_review.json`, so the sample is reproducible from the archive. r is read from
`archive/state/review.json`.

Review issue: the pipeline opens one issue per episode with a drawn turn. It shows per turn the
text, fact IDs, fact sentences, context, judge verdict, confidence and reason, and an answer
template. The reviewer answers in comments, one line per turn: `<Nummer>: belegt` or
`<Nummer>: nicht belegt`, optionally followed by a dash and a note.

Labels: the review workflow reads every comment of the issue again on each run and folds the
reviewer's comments in creation order (a later label for a turn wins), so edits, deletions and
skipped runs are picked up. Comments by any other account are ignored and not recorded. A comment
with an invalid review line is rejected as a whole. Turns without a label stay unknown.

Rate: an issue counts once every drawn turn has a label. Error-free: r * 2/3 (at least r_min);
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
from gut_zu_wissen.github import Comment, Issues
from gut_zu_wissen.judge import TurnJudgement, cited_ids
from gut_zu_wissen.ranking import RankedFact

Label = Literal["supported", "not_supported"]
Outcome = Literal["clean", "error"]
ReviewStatus = Literal["not_sampled", "pending", "answered"]

LABEL_WORDS: dict[str, Label] = {"belegt": "supported", "nicht belegt": "not_supported"}
LABEL_TEXT = {label: word for word, label in LABEL_WORDS.items()}
ISSUE_TITLE_PREFIX = "Review: "
MARKER = "gut-zu-wissen-review"

# `<Nummer>: <Label>`, the label optionally followed by a dash and a note.
REVIEW_LINE = re.compile(r"^\s*(\d+)\s*:(.*)$")
LABEL_VALUE = re.compile(r"^(nicht\s+belegt|belegt)(?:\s*[-–—]\s*(.*))?$", re.IGNORECASE)


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
class ParsedComment:
    lines: list[ParsedLine]
    errors: list[str]


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
) -> Review:
    """The review record of a finished run: population, recorded draws, drawn turns."""
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
        status="pending" if review_turns else "not_sampled",
        turns=review_turns,
    )


# ---------------------------------------------------------------- comments (pure)


def is_authorized(comment: Comment, reviewer: str, associations: Collection[str]) -> bool:
    """Only the reviewer's own account, with a repository association, gives labels."""
    return comment.login == reviewer and comment.association in associations


def parse_comment(body: str, sampled: Collection[int]) -> ParsedComment:
    """Review lines of one comment. Other lines (prose, code fences, quotes) are ignored.

    `<n>:` with nothing after it is skipped (template line not filled in). An unknown label, a
    turn that was not drawn or a turn named twice is an error.
    """
    lines: list[ParsedLine] = []
    errors: list[str] = []
    seen: set[int] = set()
    for raw in body.splitlines():
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
    return ParsedComment(lines, errors)


def fold_labels(
    comments: Sequence[Comment], sampled: Collection[int]
) -> tuple[dict[int, tuple[ParsedLine, int]], list[CommentRecord]]:
    """Labels from the reviewer's comments in creation order; a later label for a turn wins.

    Returns label and comment ID per turn, and one record per comment with review lines or
    errors. A comment with an error is not applied at all.
    """
    labels: dict[int, tuple[ParsedLine, int]] = {}
    records: list[CommentRecord] = []
    for comment in sorted(comments, key=lambda c: (c.created_at, c.id)):
        parsed = parse_comment(comment.body, sampled)
        if not parsed.lines and not parsed.errors:
            continue
        applied = not parsed.errors
        if applied:
            for line in parsed.lines:
                labels[line.turn_index] = (line, comment.id)
        records.append(
            CommentRecord(
                comment.id,
                comment.url,
                comment.created_at,
                comment.updated_at,
                len(parsed.lines) + len(parsed.errors),
                applied,
                parsed.errors,
            )
        )
    return labels, records


def review_status(sampled: Collection[int], labelled: Collection[int]) -> ReviewStatus:
    if not sampled:
        return "not_sampled"
    return "answered" if set(sampled) <= set(labelled) else "pending"


def review_outcome(turns: Sequence[ReviewTurn]) -> Outcome:
    return "error" if any(t.label == "not_supported" for t in turns) else "clean"


def apply_labels(review: Review, comments: Sequence[Comment]) -> None:
    """Write labels, comment records, status and outcome into `review` (authorized comments
    only; the caller filters)."""
    labels, records = fold_labels(comments, review.sampled())
    for turn in review.turns:
        line, comment_id = labels.get(turn.turn_index, (None, None))
        turn.label = line.label if line else None
        turn.note = line.note if line else None
        turn.comment_id = comment_id
    review.comments = records
    review.status = review_status(review.sampled(), labels)
    review.outcome = review_outcome(review.turns) if review.status == "answered" else None


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


def render_header(review: Review, reviewer: str, links: Mapping[str, str]) -> str:
    marker = json.dumps({"episode_id": review.episode_id, "turns": review.sampled()})
    template = "\n".join(f"{i}: " for i in review.sampled())
    link_parts = [f"[{name}]({url})" for name, url in links.items()]
    return "\n".join(
        [
            f"<!-- {MARKER} {marker} -->",
            f"Folge `{review.episode_id}` zum Artikel „{escape(review.title)}“. "
            f"Stichprobenrate r = {percent(review.rate)}; gezogen: {len(review.turns)} von "
            f"{len(review.population)} angenommenen Redebeiträgen mit Fakt-IDs.",
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
            "ein Label hat.",
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
    sections = ["## Redebeiträge", *(render_turn(t) for t in review.turns)]
    continuation = f"## Redebeiträge (Fortsetzung, Folge `{review.episode_id}`)"
    return title, pack(header, sections, max_chars, continuation)


def render_reply(review: Review, rate_update: RateUpdate | None) -> str:
    """Status comment after processing the reviewer's comments."""
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


def open_issue(
    archive_root: Path,
    episode_id: str,
    issues: Issues,
    review_config: Mapping,
    links: Mapping[str, str],
) -> Review:
    """Open the review issue of an episode and record its number. No drawn turn: no issue.
    An issue that is already recorded is not opened again."""
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


def apply_issue(
    archive_root: Path,
    issue: int,
    issues: Issues,
    review_config: Mapping,
    now: datetime,
) -> ApplyResult:
    """Read every comment of `issue`, keep the reviewer's, write labels and the rate state."""
    review = find_review(archive_root, issue)
    if review is None:
        return ApplyResult(None, None, None)
    comments = [
        c
        for c in issues.list_comments(issue)
        if is_authorized(c, review_config["reviewer"], review_config["reviewer_associations"])
    ]
    apply_labels(review, comments)
    state = load_rate_state(archive_root, review_config)
    rate_update = None
    if update_rate(state, review, review_config, now):
        rate_update = state.updates[-1]
        write_json(state_path(archive_root), state.to_dict())
    save_review(archive_root, review)
    return ApplyResult(review, rate_update, render_reply(review, rate_update))


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
    p_reply = sub.add_parser("reply", help="post the status comment written by apply")
    p_reply.add_argument("issue", type=int)
    p_reply.add_argument("reply_file", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    review_config = load_config()["review"]

    if args.command == "open":
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9-]+", args.episode_id):
            print(f"::error::Not an episode ID: {args.episode_id}", file=sys.stderr)
            return 2
        links = run_links(os.environ, args.episode_id)
        review = open_issue(ARCHIVE_ROOT, args.episode_id, _issues_from_env(), review_config, links)
        if review.issue is None:
            print(f"Episode {args.episode_id}: no turn drawn, no review issue.")
            _summary("No turn drawn for the review; no review issue.")
        else:
            print(f"Episode {args.episode_id}: review issue #{review.issue} ({review.issue_url}).")
            _summary(f"Review issue: {review.issue_url} ({len(review.turns)} turns drawn).")
        return 0

    if args.command == "apply":
        now = datetime.now(UTC)
        result = apply_issue(ARCHIVE_ROOT, args.issue, _issues_from_env(), review_config, now)
        args.reply_file.parent.mkdir(parents=True, exist_ok=True)
        if result.review is None or result.reply is None:
            args.reply_file.unlink(missing_ok=True)
            print(f"Issue #{args.issue} is not a review issue in the archive; nothing to do.")
            _summary(f"Issue #{args.issue} is not a review issue in the archive; nothing done.")
            return 0
        args.reply_file.write_text(result.reply + "\n", encoding="utf-8")
        review = result.review
        labelled = sum(1 for t in review.turns if t.label)
        print(
            f"Issue #{args.issue} ({review.episode_id}): {labelled} of {len(review.turns)} "
            f"turns labelled, status {review.status}, outcome {review.outcome}."
        )
        _summary(
            f"## Review {review.episode_id}\n\nIssue #{args.issue}: {labelled} of "
            f"{len(review.turns)} turns labelled, status `{review.status}`, "
            f"outcome `{review.outcome}`."
        )
        return 0

    if not args.reply_file.exists():
        return 0
    _issues_from_env().create_comment(args.issue, args.reply_file.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
