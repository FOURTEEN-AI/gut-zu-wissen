"""Exit criteria and measurements of phase 1 (step 13): `python -m gut_zu_wissen.metrics`.

Exit criterion: `exit_criteria.episodes` episodes fully reviewed (review status `answered`: every
drawn turn and the explicit check (D7) have a label) whose sample was drawn at
`exit_criteria.rate` (100 %).

Measured over every run in the archive (one run = one episode folder). An episode resumed from
the audio step (`resumed` in `episode.json`, user decision 2026-11-20) stays one run: its final
status counts, and the report marks it as resumed;

- facts used per minute (completed runs): used facts of the final script (`06_judgement.json`,
  `used_fact_ids`) per minute of dialog audio (sum of the accepted section durations in
  `08_audio.json`), the primary value (user decision 2026-11-19); secondary, per minute of the
  whole episode audio, intro and outro included;
- escalation rate: runs that failed (`status: failed`: gate failures, which abort a manual run
  with a failing job, and crashes, whose reason starts with "Crash:") or are incomplete (still
  `running`: the run has not ended; in folders from before the `completed` status, `running`
  without a review sample) over all runs; plus the share of all judge calls the LLM judge
  decided, those of replaced turn versions and the cascade checks (D1, `cascade_checks` in
  `06_judgement.json`) included (design risk "Eskalationsrate");
- cascade length: per triggering turn regeneration, the number of distinct neighbour turns the
  cascade regenerated (D1), from the regenerations in `05_dialog.json`; plus the cascade checks
  and how many of them the LLM judge decided (two-stage check, user decision 2026-11-20);
- runtime: `finished_at - started_at` of `episode.json`; for a resumed run the sum of its
  attempts (each from its start to its end, the time between them left out);
- cost: as stored in `episode.json` (`cost`) at run time, with the prices then in force; older
  folders without it fall back to TTS from `08_audio.json` and the token counts (`usage`) at
  today's config prices, and the report names them; a missing count or a placeholder price
  leaves that part unknown and is named in the report; for a resumed run every attempt is
  included (token counts summed, TTS of the earlier audio attempts in `previous_attempts`);
- error rate by revision status (D2): share of labelled turns labelled "nicht belegt", by the
  revision status of the articles the turn cites, and share of answered reviews with an error,
  by the main article's revision status;
- explicit check (D7): false-positive and false-negative rates (`review.explicit_confusion`).

Everything except loading the archive and the command line is a pure function.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from gut_zu_wissen.archive import (
    ARTICLE_FILE,
    AUDIO_FILE,
    DIALOG_FILE,
    EPISODE_FILE,
    EPISODES_DIR,
    JUDGEMENT_FILE,
    LINKED_DIR,
    REVIEW_FILE,
    parse_timestamp,
    read_json,
)
from gut_zu_wissen.review import ExplicitAccuracy, Review, explicit_confusion, explicit_pairs
from gut_zu_wissen.usage import (
    PRICE_KEYS,
    TTS_PART,
    USD_DIGITS,
    audio_tts_cost_usd,
    is_price,
    token_cost,
)

Outcome = Literal["completed", "failed", "incomplete"]
COMPLETED_STATUSES = ("completed", "published")  # episode.json status of a run that finished

SECONDS_PER_MINUTE = 60


# ---------------------------------------------------------------- archive data


@dataclass
class RunData:
    """The archive files of one run that the measurements read; None when the file is missing."""

    episode_id: str
    episode: dict
    article: dict | None = None
    dialog: dict | None = None
    judgement: dict | None = None
    audio: dict | None = None
    review: Review | None = None
    linked_status: dict[str, str] = field(default_factory=dict)  # title -> revision status


def _optional(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def load_runs(archive_root: Path) -> list[RunData]:
    """Every episode folder of the archive, in folder order."""
    runs = []
    for path in sorted((archive_root / EPISODES_DIR).glob(f"*/{EPISODE_FILE}")):
        folder = path.parent
        review = _optional(folder / REVIEW_FILE)
        linked = {}
        for linked_path in sorted((folder / LINKED_DIR).glob("*.json")):
            data = read_json(linked_path)
            linked[data["title"]] = data["revision_status"]
        runs.append(
            RunData(
                episode_id=folder.name,
                episode=read_json(path),
                article=_optional(folder / ARTICLE_FILE),
                dialog=_optional(folder / DIALOG_FILE),
                judgement=_optional(folder / JUDGEMENT_FILE),
                audio=_optional(folder / AUDIO_FILE),
                review=Review.from_dict(review) if review else None,
                linked_status=linked,
            )
        )
    return runs


# ---------------------------------------------------------------- helpers (pure)


@dataclass
class Summary:
    count: int
    mean: float | None
    min: float | None
    max: float | None


def summarize(values: Iterable[float]) -> Summary:
    vs = list(values)
    if not vs:
        return Summary(0, None, None, None)
    return Summary(len(vs), sum(vs) / len(vs), min(vs), max(vs))


def share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


# ---------------------------------------------------------------- outcome and escalation


def run_outcome(run: RunData) -> Outcome:
    """failed: recorded as failed (gate failure or crash). completed: status `completed` or
    `published`. incomplete: still `running`, i.e. the run has not ended.

    Folders written before runs were marked completed (and crashes failed) stay `running` after
    the run; there a drawn review sample (last step of a run) means completed, and without one
    the run counts as incomplete."""
    status = run.episode.get("status")
    if status == "failed":
        return "failed"
    if status in COMPLETED_STATUSES:
        return "completed"
    return "completed" if run.review is not None else "incomplete"


def resume_count(episode: Mapping) -> int:
    """How often the run was resumed from the audio step (0 in folders from before resuming)."""
    return len(episode.get("resumed") or [])


@dataclass
class RunEscalation:
    episode_id: str
    outcome: Outcome
    reason: str | None
    resumed: int = 0


@dataclass
class Escalation:
    """Escalation to the owner: every run that did not complete (design: failing job). Crashes
    count as failed."""

    runs: int
    completed: int
    failed: int
    incomplete: int
    rate: float | None
    escalated: list[RunEscalation]
    resumed: list[RunEscalation] = field(default_factory=list)  # every resumed run, any outcome


def escalation(runs: Sequence[RunData]) -> Escalation:
    outcomes = [(r, run_outcome(r)) for r in runs]
    counts = Counter(o for _, o in outcomes)
    every = [
        RunEscalation(r.episode_id, o, r.episode.get("failure"), resume_count(r.episode))
        for r, o in outcomes
    ]
    escalated = [e for e in every if e.outcome != "completed"]
    return Escalation(
        len(runs),
        counts["completed"],
        counts["failed"],
        counts["incomplete"],
        share(len(escalated), len(runs)),
        escalated,
        [e for e in every if e.resumed],
    )


@dataclass
class JudgeEscalation:
    """Share of all judge calls that went to the LLM judge (uncertain p, overlength or transient
    Jev error), turn judgements and cascade checks together. Gate rejections are not judge
    decisions and are left out."""

    judgements: int
    by_jev: int
    by_llm_judge: int
    rate: float | None


def archived_judgements(run: RunData) -> list[dict]:
    """Every judgement of the run: the final ones, those of discarded conversations and those of
    turn versions a regeneration replaced (`superseded_judgements`; missing in folders from
    before they were archived)."""
    judgement = run.judgement or {}
    found = list(judgement.get("judgements", []))
    for discarded in (run.dialog or {}).get("discarded_conversations", []):
        found += discarded.get("judgements", [])
    found += [s["judgement"] for s in judgement.get("superseded_judgements", [])]
    return found


def archived_cascade_checks(run: RunData) -> list[dict]:
    """Every cascade check of the run, those of discarded conversations included."""
    return list((run.judgement or {}).get("cascade_checks", []))


def cascade_check_decider(check: Mapping) -> str | None:
    """Who decided a cascade check: "jev" or "llm_judge".

    Folders from before the two-stage check (user decision 2026-11-20) have no `decided_by`: Jev
    decided when it gave p; after a Jev error nobody decided (the cascade simply went on).
    """
    if "decided_by" in check:
        return check["decided_by"]
    return "jev" if check.get("p") is not None else None


def judge_escalation(runs: Sequence[RunData]) -> JudgeEscalation:
    deciders = Counter(j["decided_by"] for r in runs for j in archived_judgements(r))
    deciders.update(cascade_check_decider(c) for r in runs for c in archived_cascade_checks(r))
    total = deciders["jev"] + deciders["llm_judge"]
    return JudgeEscalation(
        total, deciders["jev"], deciders["llm_judge"], share(deciders["llm_judge"], total)
    )


# ---------------------------------------------------------------- facts per minute


@dataclass
class FactsPerMinute:
    episode_id: str
    used_facts: int
    episode_seconds: float
    dialog_seconds: float | None
    per_minute: float
    per_dialog_minute: float | None


def facts_per_minute(run: RunData) -> FactsPerMinute | None:
    """None unless the run completed with the final judgement and the finished episode audio."""
    if run_outcome(run) != "completed" or run.judgement is None or run.audio is None:
        return None
    if not run.audio.get("seconds"):
        return None
    used = len(run.judgement["used_fact_ids"])
    seconds = run.audio["seconds"]
    sections = [s.get("seconds") for s in run.audio.get("sections", [])]
    dialog = sum(sections) if sections and all(s is not None for s in sections) else None
    return FactsPerMinute(
        run.episode_id,
        used,
        seconds,
        dialog,
        used / (seconds / SECONDS_PER_MINUTE),
        used / (dialog / SECONDS_PER_MINUTE) if dialog else None,
    )


# ---------------------------------------------------------------- cascade length


def cascade_lengths(regenerations: Sequence[Mapping]) -> list[int]:
    """Length of every cascade in the order of the regenerations in `05_dialog.json`.

    A cascade starts with the retry of a rejected turn (repeated retries of the same turn belong
    to it) and continues with the neighbours regenerated after it (kind "cascade"). Its length is
    the number of distinct neighbour turns; 0 when the retried turn was the last one or its
    budget ran out before the cascade began.
    """
    chains: list[set[int]] = []
    previous: Mapping | None = None
    for r in regenerations:
        same_retry = (
            previous is not None
            and previous["kind"] == "retry"
            and previous["turn_index"] == r["turn_index"]
            and previous["conversation"] == r["conversation"]
        )
        if (r["kind"] == "retry" and not same_retry) or not chains:
            chains.append(set())
        if r["kind"] == "cascade":
            chains[-1].add(r["turn_index"])
        previous = r
    return [len(c) for c in chains]


@dataclass
class EpisodeCascades:
    episode_id: str
    regenerations: int
    lengths: list[int]
    mean: float | None
    max: int | None


@dataclass
class Cascades:
    cascades: int
    mean: float | None
    max: int | None
    distribution: dict[int, int]  # length -> number of cascades
    episodes: list[EpisodeCascades]
    checks: int = 0  # cascade checks decided by Jev or the LLM judge
    checks_by_llm_judge: int = 0


def cascades(runs: Sequence[RunData]) -> Cascades:
    episodes = []
    for run in runs:
        if run.dialog is None:
            continue
        regenerations = run.dialog.get("regenerations", [])
        lengths = cascade_lengths(regenerations)
        s = summarize(lengths)
        episodes.append(
            EpisodeCascades(
                run.episode_id,
                len(regenerations),
                lengths,
                s.mean,
                max(lengths) if lengths else None,
            )
        )
    every = [n for e in episodes for n in e.lengths]
    s = summarize(every)
    deciders = Counter(cascade_check_decider(c) for r in runs for c in archived_cascade_checks(r))
    return Cascades(
        len(every),
        s.mean,
        max(every) if every else None,
        dict(sorted(Counter(every).items())),
        episodes,
        deciders["jev"] + deciders["llm_judge"],
        deciders["llm_judge"],
    )


# ---------------------------------------------------------------- runtime


@dataclass
class RunRuntime:
    episode_id: str
    outcome: Outcome
    seconds: float | None
    resumed: int = 0


@dataclass
class Runtime:
    completed: Summary
    runs: list[RunRuntime]


def runtime_seconds(episode: Mapping) -> float | None:
    """From the start of the run to its end; None when the end was not recorded.

    A resumed run sums its attempts: the first from `started_at` to the end of that attempt
    (`previous_finished_at` of the first resume), each resume from its `started_at` to the end
    of the next attempt, the last one to `finished_at`. The time between attempts is left out.
    """
    if not episode.get("started_at") or not episode.get("finished_at"):
        return None
    starts = [episode["started_at"]]
    ends = []
    for resume in episode.get("resumed") or []:
        if not resume.get("previous_finished_at"):
            return None
        ends.append(resume["previous_finished_at"])
        starts.append(resume["started_at"])
    ends.append(episode["finished_at"])
    return sum(
        (parse_timestamp(end) - parse_timestamp(start)).total_seconds()
        for start, end in zip(starts, ends, strict=True)
    )


def runtime(runs: Sequence[RunData]) -> Runtime:
    per_run = [
        RunRuntime(
            r.episode_id, run_outcome(r), runtime_seconds(r.episode), resume_count(r.episode)
        )
        for r in runs
    ]
    return Runtime(
        summarize(r.seconds for r in per_run if r.outcome == "completed" and r.seconds is not None),
        per_run,
    )


# ---------------------------------------------------------------- cost


@dataclass
class CostPart:
    part: str  # "tts" or a usage role
    usd: float | None
    note: str | None = None


@dataclass
class RunCost:
    episode_id: str
    outcome: Outcome
    parts: list[CostPart]
    known_usd: float
    complete: bool
    todays_prices: bool = False  # no cost stored at run time: computed with today's prices
    resumed: int = 0


NO_TTS_COST = "a synthesis without token count or price"
NO_USAGE = CostPart("claude, jev", None, "token counts not recorded")


def tts_cost(run: RunData, outcome: Outcome) -> CostPart:
    if run.audio is None:
        if outcome == "incomplete":
            return CostPart(TTS_PART, None, "no 08_audio.json (run still running or not recorded)")
        return CostPart(TTS_PART, 0.0, "no synthesis")
    usd = audio_tts_cost_usd(run.audio)
    return CostPart(TTS_PART, usd, NO_TTS_COST if usd is None else None)


def _missing_prices(role: str, prices: Mapping, when: str) -> str:
    keys = PRICE_KEYS.get(role)
    if keys is None:
        return "no price key for this role"
    names = ", ".join(k for k in keys if k and not is_price(prices.get(k)))
    return f"price not set{when}: {names}"


def stored_cost(stored: Mapping) -> list[CostPart]:
    """The parts of `cost` in episode.json, at the prices in force at run time."""
    prices = stored.get("prices", {})
    usd = stored.get("usd", {})
    parts = [
        CostPart(TTS_PART, usd.get(TTS_PART), None if is_price(usd.get(TTS_PART)) else NO_TTS_COST)
    ]
    roles = [role for role in usd if role != TTS_PART]
    if not roles:
        parts.append(NO_USAGE)
    for role in roles:
        value = usd[role]
        note = None if value is not None else _missing_prices(role, prices, " at run time")
        parts.append(CostPart(role, value, note))
    return parts


def todays_cost(run: RunData, outcome: Outcome, prices: Mapping) -> list[CostPart]:
    """Folders without a stored cost: the token counts at today's config prices."""
    parts = [tts_cost(run, outcome)]
    usage = run.episode.get("usage") or {}
    if not usage:
        parts.append(NO_USAGE)
    for role, counts in usage.items():
        keys = PRICE_KEYS.get(role)
        usd = token_cost(counts, prices, keys) if keys else None
        parts.append(
            CostPart(role, usd, None if usd is not None else _missing_prices(role, prices, ""))
        )
    return parts


def run_cost(run: RunData, prices: Mapping) -> RunCost:
    """Cost of one run: the values stored at run time in `episode.json` (`cost`); for older
    folders without them, computed with today's `prices`. Whisper runs locally and costs nothing
    per call."""
    outcome = run_outcome(run)
    stored = run.episode.get("cost")
    parts = stored_cost(stored) if stored else todays_cost(run, outcome, prices)
    known = round(sum(p.usd for p in parts if p.usd is not None), USD_DIGITS)
    return RunCost(
        run.episode_id,
        outcome,
        parts,
        known,
        all(p.usd is not None for p in parts),
        todays_prices=not stored,
        resumed=resume_count(run.episode),
    )


@dataclass
class Cost:
    total_known_usd: float
    complete: bool
    completed_mean_usd: float | None  # only when every completed run's cost is complete
    missing: list[str]  # distinct notes of unknown parts
    runs: list[RunCost]
    todays_prices: list[str] = field(default_factory=list)  # runs without a stored cost


def cost(runs: Sequence[RunData], prices: Mapping) -> Cost:
    per_run = [run_cost(r, prices) for r in runs]
    completed = [c for c in per_run if c.outcome == "completed"]
    mean = (
        summarize(c.known_usd for c in completed).mean
        if completed and all(c.complete for c in completed)
        else None
    )
    missing = sorted({f"{p.part}: {p.note}" for c in per_run for p in c.parts if p.usd is None})
    return Cost(
        round(sum(c.known_usd for c in per_run), USD_DIGITS),
        all(c.complete for c in per_run),
        mean,
        missing,
        per_run,
        [c.episode_id for c in per_run if c.todays_prices],
    )


# ---------------------------------------------------------------- error rate by revision status


@dataclass
class ErrorRate:
    labelled: int
    not_supported: int
    rate: float | None


@dataclass
class ErrorsByRevision:
    """Turns: labelled turns by the revision status of the cited articles ("mixed" when they
    differ, "unknown" when none is known). Episodes: answered reviews by the main article's
    revision status; an episode with one turn labelled "nicht belegt" counts as an error."""

    turns: dict[str, ErrorRate]
    episodes: dict[str, ErrorRate]


def turn_revision_status(article_titles: Iterable[str], status: Mapping[str, str]) -> str:
    found = {status[t] for t in article_titles if t in status}
    if not found:
        return "unknown"
    return found.pop() if len(found) == 1 else "mixed"


def revision_statuses(run: RunData) -> dict[str, str]:
    status = dict(run.linked_status)
    if run.article is not None:
        status[run.article["title"]] = run.article["revision_status"]
    return status


def _rates(counts: Mapping[str, list[bool]]) -> dict[str, ErrorRate]:
    return {
        key: ErrorRate(len(errors), sum(errors), share(sum(errors), len(errors)))
        for key, errors in sorted(counts.items())
    }


def errors_by_revision(runs: Sequence[RunData]) -> ErrorsByRevision:
    turns: dict[str, list[bool]] = {}
    episodes: dict[str, list[bool]] = {}
    for run in runs:
        review = run.review
        if review is None:
            continue
        status = revision_statuses(run)
        for turn in review.turns:
            if turn.label is None:
                continue
            key = turn_revision_status((f.article for f in turn.facts), status)
            turns.setdefault(key, []).append(turn.label == "not_supported")
        if review.status == "answered":
            key = run.article["revision_status"] if run.article else "unknown"
            episodes.setdefault(key, []).append(review.outcome == "error")
    return ErrorsByRevision(_rates(turns), _rates(episodes))


# ---------------------------------------------------------------- exit check


@dataclass
class ExitCheck:
    required_episodes: int
    required_rate: float
    qualifying: list[str]
    met: bool


def fully_reviewed(review: Review, rate: float) -> bool:
    """Every drawn turn and the explicit check labelled, sample drawn at `rate` or higher (r = 1:
    every turn drawn). A review without an explicit result does not qualify."""
    return (
        review.status == "answered"
        and review.explicit is not None
        and review.explicit.label is not None
        and review.rate >= rate
    )


def exit_check(runs: Sequence[RunData], criteria: Mapping) -> ExitCheck:
    rate = float(criteria["rate"])
    qualifying = [r for r in runs if r.review is not None and fully_reviewed(r.review, rate)]
    return ExitCheck(
        criteria["episodes"],
        rate,
        [r.episode_id for r in qualifying],
        len(qualifying) >= criteria["episodes"],
    )


# ---------------------------------------------------------------- report


@dataclass
class Report:
    exit: ExitCheck
    facts_per_minute: Summary
    facts_per_dialog_minute: Summary
    facts_per_episode: list[FactsPerMinute]
    escalation: Escalation
    judge_escalation: JudgeEscalation
    cascades: Cascades
    runtime: Runtime
    cost: Cost
    errors_by_revision: ErrorsByRevision
    explicit: ExplicitAccuracy


def build_report(runs: Sequence[RunData], config: Mapping) -> Report:
    per_episode = [f for f in (facts_per_minute(r) for r in runs) if f is not None]
    reviews = [r.review for r in runs if r.review is not None]
    return Report(
        exit=exit_check(runs, config["exit_criteria"]),
        facts_per_minute=summarize(f.per_minute for f in per_episode),
        facts_per_dialog_minute=summarize(
            f.per_dialog_minute for f in per_episode if f.per_dialog_minute is not None
        ),
        facts_per_episode=per_episode,
        escalation=escalation(runs),
        judge_escalation=judge_escalation(runs),
        cascades=cascades(runs),
        runtime=runtime(runs),
        cost=cost(runs, config["prices"]),
        errors_by_revision=errors_by_revision(runs),
        explicit=explicit_confusion(explicit_pairs(reviews)),
    )


def _num(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f} %"


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4f} USD"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    minutes, rest = divmod(round(seconds), SECONDS_PER_MINUTE)
    return f"{minutes} min {rest:02d} s"


def _resumed(count: int) -> str:
    return f", resumed {count}×" if count else ""


def _summary(s: Summary, fmt=_num) -> str:
    if not s.count:
        return "no data"
    return f"mean {fmt(s.mean)}, min {fmt(s.min)}, max {fmt(s.max)} (n = {s.count})"


def _error_rows(rates: Mapping[str, ErrorRate], counted: str, errors: str) -> list[str]:
    if not rates:
        return ["No labels yet."]
    rows = [f"| Revision status | {counted} | {errors} | Error rate |", "|---|---|---|---|"]
    rows += [
        f"| {key} | {r.labelled} | {r.not_supported} | {_pct(r.rate)} |" for key, r in rates.items()
    ]
    return rows


def render_markdown(report: Report) -> str:
    """The report as Markdown, readable in a terminal and in the job summary."""
    e = report.exit
    lines = [
        "## Phase 1 exit criteria (step 13)",
        "",
        f"**{'Met' if e.met else 'Not met'}:** {len(e.qualifying)} of {e.required_episodes} "
        f"episodes fully reviewed (every drawn turn and the explicit check labelled) at "
        f"r ≥ {e.required_rate:.0%}.",
    ]
    if e.qualifying:
        lines.append(f"Qualifying: {', '.join(f'`{i}`' for i in e.qualifying)}.")
    lines += [
        "",
        "### Facts used per minute",
        "",
        f"- Primary, dialog audio only: {_summary(report.facts_per_dialog_minute)}",
        "- Secondary, whole episode audio (intro and outro included): "
        f"{_summary(report.facts_per_minute)}",
    ]
    if report.facts_per_episode:
        lines += [
            "",
            "| Episode | Used facts | Dialog audio | Per dialog minute (primary) | "
            "Episode audio | Per episode minute (secondary) |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| `{f.episode_id}` | {f.used_facts} | {_duration(f.dialog_seconds)} | "
            f"{_num(f.per_dialog_minute)} | {_duration(f.episode_seconds)} | "
            f"{_num(f.per_minute)} |"
            for f in report.facts_per_episode
        ]
    s = report.escalation
    lines += [
        "",
        "### Escalation rate",
        "",
        f"- Runs: {s.runs} (completed {s.completed}, failed {s.failed}, "
        f"incomplete {s.incomplete}); escalated: {_pct(s.rate)}",
    ]
    lines += [
        f"  - `{x.episode_id}` ({x.outcome}{_resumed(x.resumed)}): {x.reason or '–'}"
        for x in s.escalated
    ]
    if s.resumed:
        lines.append(
            "- Resumed from the audio step (each counts as one run with its final outcome): "
            + ", ".join(f"`{x.episode_id}` ({x.outcome}{_resumed(x.resumed)})" for x in s.resumed)
        )
    j = report.judge_escalation
    lines += [
        f"- Judge decisions by the LLM judge: {j.by_llm_judge} of {j.judgements} ({_pct(j.rate)}), "
        "cascade checks included",
        "",
        "### Cascade length (D1)",
        "",
    ]
    c = report.cascades
    if c.cascades:
        dist = ", ".join(f"{k}: {v}" for k, v in c.distribution.items())
        lines.append(
            f"- {c.cascades} cascades; mean {_num(c.mean)}, max {c.max}; "
            f"distribution (length: count) {dist}"
        )
    else:
        lines.append("- No cascades.")
    lines.append(f"- Cascade checks: {c.checks}, decided by the LLM judge: {c.checks_by_llm_judge}")
    lines += [
        f"  - `{x.episode_id}`: {x.regenerations} regenerations, lengths {x.lengths or '–'}, "
        f"mean {_num(x.mean)}, max {x.max if x.max is not None else '–'}"
        for x in c.episodes
    ]
    r = report.runtime
    lines += [
        "",
        "### Runtime",
        "",
        f"- Completed runs: {_summary(r.completed, _duration)}",
    ]
    lines += [
        f"  - `{x.episode_id}` ({x.outcome}{_resumed(x.resumed)}): {_duration(x.seconds)}"
        for x in r.runs
    ]
    if any(x.resumed for x in r.runs):
        lines.append("- Resumed runs: the sum of all attempts, without the time between them.")
    k = report.cost
    lines += [
        "",
        "### Cost",
        "",
        f"- Recorded: {_usd(k.total_known_usd)} over all runs"
        + ("" if k.complete else " (incomplete, see below)"),
        f"- Mean per completed episode: {_usd(k.completed_mean_usd)}",
    ]
    lines += [
        f"  - `{x.episode_id}` ({x.outcome}{_resumed(x.resumed)}): {_usd(x.known_usd)}"
        + ("" if x.complete else " known, parts unknown")
        for x in k.runs
    ]
    if any(x.resumed for x in k.runs):
        lines.append(
            "- Resumed runs: every attempt included (token counts summed, TTS of the earlier "
            "audio attempts), at the prices in force at the last attempt."
        )
    if k.missing:
        lines += ["- Unknown parts:", *(f"  - {m}" for m in k.missing)]
    if k.todays_prices:
        lines.append(
            "- Computed with today's prices (folders from before costs were stored at run "
            f"time): {', '.join(f'`{i}`' for i in k.todays_prices)}"
        )
    b = report.errors_by_revision
    lines += [
        "",
        "### Error rate by revision status (D2)",
        "",
        "Turns (revision status of the cited articles):",
        "",
        *_error_rows(b.turns, "Labelled turns", "Nicht belegt"),
        "",
        "Episodes (answered reviews, main article):",
        "",
        *_error_rows(b.episodes, "Answered reviews", "With error"),
    ]
    x = report.explicit
    lines += [
        "",
        "### Explicit check (D7)",
        "",
        f"- Labelled episodes: {x.episodes} (TP {x.true_positive}, FP {x.false_positive}, "
        f"FN {x.false_negative}, TN {x.true_negative})",
        f"- False-positive rate: {_pct(x.false_positive_rate)}; "
        f"false-negative rate: {_pct(x.false_negative_rate)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- command line


def main(argv: Iterable[str] | None = None) -> int:
    from gut_zu_wissen.run import ARCHIVE_ROOT, load_config, write_job_summary

    parser = argparse.ArgumentParser(description="Phase 1 exit criteria and measurements.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_ROOT, help="archive directory")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    report = build_report(load_runs(args.archive), load_config())
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0
    text = render_markdown(report)
    print(text)
    write_job_summary(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
