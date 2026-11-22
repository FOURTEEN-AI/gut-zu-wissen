"""Resume a failed episode from the audio step: `python -m gut_zu_wissen.resume <episode_id>`.

User decision 2026-11-20: a run that failed only in the audio step (TTS or STT error, STT
budget used up, or a crash inside the step) need not repeat the author and judge work already
archived. The resume loads the final script, its judgements, the offer and the explicit check
from the archive and runs the same audio step and review step as `run.run` (shared functions,
not a copy). Only an episode with `status: failed` whose dialog (`05`), judgement (`06`) and
explicit check (`07`) are complete and whose failure lies in the audio step may be resumed;
everything else is refused (`resume_refusal`).

The episode stays one run: `08_audio.json` is rewritten and keeps the earlier attempts under
`previous_attempts`; `episode.json` gets an entry in `resumed` (when, workflow run, end and
failure reason of the attempt before), `finished_at` of the resume, the token counts of the
resume added to the stored ones (`usage.merge_usage`) and the cost of all attempts at the prices
in force at resume time. On success the status becomes `completed` with `09_review.json`; on
failure it is `failed` again with the new reason, escalated like a run.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from gut_zu_wissen.archive import (
    AUDIO_FILE,
    DIALOG_FILE,
    EPISODE_FILE,
    EXPLICIT_FILE,
    JUDGEMENT_FILE,
    RANKING_FILE,
    STEP_AUDIO,
    STEP_REVIEW,
    episode_dir,
    format_timestamp,
    load_episode,
    read_json,
    save_episode,
)
from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.episode_audio import new_record
from gut_zu_wissen.explicit import ExplicitCheck, Finding
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.intro_outro import ASSETS_DIR, FixedAudioError, load_fixed
from gut_zu_wissen.judge import TurnJudgement
from gut_zu_wissen.llm import AnthropicLlm, StructuredLlm
from gut_zu_wissen.logs import setup_logging
from gut_zu_wissen.orchestrator import GateFailure
from gut_zu_wissen.progress import NoProgress, Progress
from gut_zu_wissen.progress import from_env as progress_from_env
from gut_zu_wissen.ranking import RankedFact
from gut_zu_wissen.run import (
    ARCHIVE_ROOT,
    AUDIO_ROOT,
    REPO_ROOT,
    audio_step,
    crash_reason,
    load_config,
    report_success,
    review_step,
    secret_values,
    write_job_summary,
)
from gut_zu_wissen.stt import Stt, WhisperStt
from gut_zu_wissen.tts import GeminiTts, Tts, voices_of
from gut_zu_wissen.usage import Usage, audio_tts_cost_usd, episode_cost, merge_usage, usage_dict

# Archive folder name: <YYYY-MM-DD>_<slug>, slug as `archive.slugify` makes it, optional `-n`.
# The resume workflow checks its input against the same pattern (test_workflows.py).
EPISODE_ID_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9]+(-[a-z0-9]+)*$"
EXPLICIT_CHECK_PREFIX = "Explicit check"  # failure reason of a failed explicit check (run.py)


class ResumeRefused(Exception):
    """The episode may not be resumed from the audio step; the message says why."""


# ---------------------------------------------------------------- pure functions


def valid_episode_id(value: str) -> bool:
    return re.fullmatch(EPISODE_ID_PATTERN, value) is not None


def resume_refusal(
    episode: Mapping,
    *,
    ranking: Mapping | None,
    dialog: Mapping | None,
    judgement: Mapping | None,
    explicit: Mapping | None,
    audio: Mapping | None,
) -> str | None:
    """Why the episode may not be resumed from the audio step; None when it may.

    Allowed only for `status: failed` with a complete final script (`05_dialog.json` with
    turns, `06_judgement.json` with an accepted judgement for every turn, `04_ranking.json` for
    the review facts) and a successful explicit check (`07_explicit.json` without error and with
    a flag), and only when the failure lies in the audio step: `failed_step` is `audio`, or, in
    folders from before `failed_step` was recorded, `08_audio.json` exists (only the audio step
    writes it) without the finished episode audio (`file` unset; set only once the step
    succeeded, so a later crash is no audio failure).
    """
    status = episode.get("status")
    if status != "failed":
        return f"status is '{status}'; only a failed episode can be resumed."
    failure = episode.get("failure") or ""
    if failure.startswith(EXPLICIT_CHECK_PREFIX) or (explicit or {}).get("error"):
        return "the explicit check failed; the audio step never started."
    missing = [
        name
        for name, data in (
            (RANKING_FILE, ranking),
            (DIALOG_FILE, dialog),
            (JUDGEMENT_FILE, judgement),
            (EXPLICIT_FILE, explicit),
        )
        if data is None
    ]
    if missing:
        return f"{', '.join(missing)} missing: the run failed before the audio step."
    assert dialog is not None and judgement is not None and explicit is not None
    turns = dialog.get("turns") or []
    if not turns:
        return f"{DIALOG_FILE} has no turns."
    judged = {j.get("turn_index"): j.get("accepted") for j in judgement.get("judgements", [])}
    if set(judged) != {t.get("index") for t in turns} or not all(judged.values()):
        return f"{JUDGEMENT_FILE} has no accepted judgement for every turn of the final script."
    if not isinstance(explicit.get("explicit"), bool):
        return f"{EXPLICIT_FILE} has no result."
    step = episode.get("failed_step")
    if step is not None:
        if step != STEP_AUDIO:
            return f"the run failed in the step '{step}', not in the audio step."
        return None
    if audio is None:
        return f"no {AUDIO_FILE}: the run did not fail in the audio step."
    if audio.get("file") is not None:
        return "the audio step finished; the run failed after it."
    return None


def resume_entry(
    started_at: str,
    workflow_run: str | None,
    previous_finished_at: str | None,
    previous_failure: str | None,
) -> dict:
    """One entry of `resumed` in episode.json; field order is the archive field order."""
    return {
        "started_at": started_at,
        "workflow_run": workflow_run,
        "previous_finished_at": previous_finished_at,
        "previous_failure": previous_failure,
    }


def previous_attempts(audio: Mapping | None) -> list[dict]:
    """`previous_attempts` of the new `08_audio.json`: the attempts the old one kept, then the
    old record itself, oldest first."""
    if audio is None:
        return []
    own = {key: value for key, value in audio.items() if key != "previous_attempts"}
    return [*audio.get("previous_attempts", []), own]


# ---------------------------------------------------------------- archive data


def turns_from(dialog: Mapping) -> list[Turn]:
    return [
        Turn(
            t["index"],
            t["speaker"],
            t["text"],
            list(t["fact_ids"]),
            [Statement(s["text"], list(s["fact_ids"])) for s in t.get("statements", [])],
        )
        for t in dialog["turns"]
    ]


def judgements_from(judgement: Mapping) -> list[TurnJudgement]:
    return [TurnJudgement(**j) for j in judgement["judgements"]]


def offer_from(ranking: Mapping) -> dict[str, RankedFact]:
    """The offer of the final conversation by fact ID (`04_ranking.json` is rewritten for every
    offer, so it holds the last one)."""
    offered = [
        *ranking.get("einordnung_facts", []),  # absent in archives before 2026-11-20
        *ranking["main_facts"],
        *ranking["linked_facts"],
    ]
    facts = [RankedFact(r["rank"], r["article"], r["revid"], Fact(**r["fact"])) for r in offered]
    return {r.fact.segment_id: r for r in facts}


def explicit_from(explicit: Mapping) -> ExplicitCheck:
    return ExplicitCheck(
        **{
            **explicit,
            "findings": [Finding(**f) for f in explicit.get("findings", [])],
        }
    )


def _optional(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


# ---------------------------------------------------------------- resume


def resume_audio(
    archive_root: Path,
    episode_id: str,
    config: dict,
    now: datetime,
    *,
    tts: Tts,
    stt: Stt,
    judge_llm: StructuredLlm,
    fixed_dir: Path,
    audio_root: Path,
    workflow_run: str | None = None,
    clock: Callable[[], datetime] | None = None,
    usage: Mapping[str, Usage] | None = None,
    secrets: Collection[str] = (),
    progress: Progress | None = None,
) -> str:
    """Resume the episode from the audio step. Returns the episode ID.

    Raises `ResumeRefused` (nothing changed), `FixedAudioError` (intro or outro do not match;
    nothing changed), `GateFailure` (audio step failed again, recorded) or re-raises a crash
    (recorded as failed with `crash_reason`). `judge_llm` decides the STT deviations; `usage`
    holds the counters of the clients used for the resume.
    """
    if not valid_episode_id(episode_id):
        raise ResumeRefused(f"'{episode_id}' is no episode ID (pattern {EPISODE_ID_PATTERN}).")
    folder = episode_dir(archive_root, episode_id)
    if not (folder / EPISODE_FILE).exists():
        raise ResumeRefused(f"No episode folder {episode_id} in the archive.")
    record = load_episode(archive_root, episode_id)
    data = {
        "ranking": _optional(folder / RANKING_FILE),
        "dialog": _optional(folder / DIALOG_FILE),
        "judgement": _optional(folder / JUDGEMENT_FILE),
        "explicit": _optional(folder / EXPLICIT_FILE),
        "audio": _optional(folder / AUDIO_FILE),
    }
    refusal = resume_refusal(asdict(record), **data)
    if refusal:
        raise ResumeRefused(f"Episode {episode_id} cannot be resumed: {refusal}")
    turns = turns_from(data["dialog"])
    judgements = judgements_from(data["judgement"])
    facts = offer_from(data["ranking"])
    explicit = explicit_from(data["explicit"])
    fixed = load_fixed(fixed_dir, config["models"]["tts"], voices_of(config))

    record.resumed.append(
        resume_entry(format_timestamp(now), workflow_run, record.finished_at, record.failure)
    )
    record.status, record.failure, record.failed_step = "running", None, None
    stored_usage = record.usage
    save_episode(archive_root, record)
    audio_record = new_record(config)
    audio_record.previous_attempts = previous_attempts(data["audio"])
    step = STEP_AUDIO
    # The status page shows the resume's own start, not the first run's.
    (progress or NoProgress()).step(replace(record, started_at=format_timestamp(now)), step, now)
    try:
        audio_step(
            archive_root,
            record,
            audio_record,
            turns,
            tts,
            stt,
            judge_llm,
            fixed,
            audio_root,
            config,
            facts,
        )
        step = STEP_REVIEW
        review_step(archive_root, record, turns, judgements, facts, explicit, config)
        return episode_id
    except GateFailure:
        record.failed_step = step
        raise
    except Exception as exc:  # crash: recorded as failed, then the job fails with the traceback
        max_reason = config["run"]["failure_reason_max_chars"]
        record.status, record.failure = "failed", crash_reason(exc, max_reason, secrets)
        record.failed_step = step
        raise
    finally:  # every exit: end of the resume, token counts and cost of all attempts
        record.finished_at = format_timestamp((clock or (lambda: datetime.now(UTC)))())
        record.usage = merge_usage(stored_usage, usage_dict(usage))
        tts_usd = audio_tts_cost_usd(audio_record.to_dict())
        record.cost = episode_cost(record.usage, tts_usd, config["prices"])
        save_episode(archive_root, record)


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print("Usage: python -m gut_zu_wissen.resume <episode_id>", file=sys.stderr)
        return 2
    setup_logging()
    episode_id = args[0].strip()
    config = load_config()
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("::error::GEMINI_API_KEY is not set.", file=sys.stderr)
        return 2
    models = config["models"]
    # Only the STT decisions call Claude; author and Jev keep their stored counts.
    usage = {"llm_judge": Usage(models["llm_judge"])}
    judge_llm = AnthropicLlm(
        models["llm_judge"], config["quotas"]["llm_max_tokens"], usage=usage["llm_judge"]
    )
    tts_config = config["tts"]
    tts = GeminiTts(
        models["tts"],
        voices_of(config),
        tts_config["max_retries"],
        tts_config["backoff_seconds"],
        tts_config["timeout_seconds"],
        gemini_key,
    )
    stt = WhisperStt(models["stt"], config["stt"])
    secrets = secret_values(os.environ)
    try:
        resume_audio(
            ARCHIVE_ROOT,
            episode_id,
            config,
            datetime.now(UTC),
            tts=tts,
            stt=stt,
            judge_llm=judge_llm,
            fixed_dir=ASSETS_DIR,
            audio_root=AUDIO_ROOT,
            workflow_run=os.environ.get("GITHUB_RUN_ID"),
            usage=usage,
            secrets=secrets,
            progress=progress_from_env(config["status"], REPO_ROOT),
        )
    except ResumeRefused as exc:
        print(f"::error::Refused: {exc}", file=sys.stderr)
        write_job_summary(f"## Audio resume refused\n\n{exc}\n\nThe archive is unchanged.")
        return 1
    except FixedAudioError as exc:
        print(f"::error::Aborted: {exc}", file=sys.stderr)
        write_job_summary(f"## Audio resume aborted\n\n{exc}\n\nThe archive is unchanged.")
        return 1
    except GateFailure as exc:
        print(f"::error::Aborted: {exc}", file=sys.stderr)
        write_job_summary(
            f"## Audio resume of {episode_id} failed\n\n{exc}\n\nDetails in the archive folder."
        )
        return 1
    except Exception as exc:  # crash: escalate, then fail the job with the traceback
        reason = crash_reason(exc, config["run"]["failure_reason_max_chars"], secrets)
        print(f"::error::Aborted: {reason}", file=sys.stderr)
        write_job_summary(
            f"## Audio resume of {episode_id} crashed\n\n{reason}\n\nRecorded as `failed` in "
            "the archive folder if the resume got that far; traceback in the job log."
        )
        raise
    resumes = len(load_episode(ARCHIVE_ROOT, episode_id).resumed)
    report_success(episode_id, f"Resumed from the audio step (resume {resumes}). ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
