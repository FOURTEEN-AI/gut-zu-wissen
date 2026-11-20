"""Run entry point: `python -m gut_zu_wissen.run "<article title>"`.

Phase 1 steps 2 to 12: fetch the article, apply the gates, segment, extract facts, rank, add the
linked intros, write the dialog, judge it, run the explicit check on the final script,
synthesise, verify with STT and normalise the audio, draw the review sample, write the episode
folder. The turn and conversation loops live in orchestrator.py (step 8), the audio step with its
STT verification in episode_audio.py (steps 9 and 10), the explicit check in explicit.py (step
11, run before the audio step so a failed check costs no TTS), the review loop in review.py (step
12; the workflow opens the issue after the run); later steps are added here.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gut_zu_wissen.archive import (
    ARTICLE_FILE,
    AUDIO_FILE,
    DIALOG_FILE,
    EXPLICIT_FILE,
    FACTS_FILE,
    JUDGEMENT_FILE,
    LINKED_DIR,
    RANKING_FILE,
    REVIEW_FILE,
    SEGMENTS_FILE,
    create_episode,
    episode_dir,
    format_timestamp,
    last_published,
    load_episode,
    read_json,
    save_episode,
    slugify,
    write_json,
)
from gut_zu_wissen.episode_audio import (
    EPISODE_AUDIO,
    AudioStepError,
    new_record,
    produce_audio,
)
from gut_zu_wissen.explicit import ExplicitCheck, check_explicit
from gut_zu_wissen.extraction import extract_facts
from gut_zu_wissen.fetch import ArticleNotFoundError, HttpWikipediaApi, WikipediaApi, fetch_article
from gut_zu_wissen.gates import is_living_person, repeat_gate_allows
from gut_zu_wissen.intro_outro import ASSETS_DIR, FixedAudioError, load_fixed
from gut_zu_wissen.jev import HttpJev, Jev
from gut_zu_wissen.linked import LinkedIntro, SkippedLink, fetch_linked_intros
from gut_zu_wissen.llm import AnthropicLlm, LlmError, StructuredLlm
from gut_zu_wissen.orchestrator import GateFailure, Offer, Orchestration, orchestrate
from gut_zu_wissen.ranking import (
    RankedFact,
    bound_targets,
    link_targets,
    offer_count,
    offer_gate,
    offer_summary,
    rank_facts,
    ranked,
    raw_of,
)
from gut_zu_wissen.render import ParseRenderer
from gut_zu_wissen.review import build_review, load_rate_state, load_review
from gut_zu_wissen.segmentation import segment_article
from gut_zu_wissen.stt import Stt, WhisperStt
from gut_zu_wissen.tts import GeminiTts, Tts, voices_of
from gut_zu_wissen.usage import Usage, episode_cost, tts_cost_usd, usage_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.toml"
ARCHIVE_ROOT = REPO_ROOT / "archive"
AUDIO_ROOT = REPO_ROOT / "build" / "audio"  # not committed (.gitignore: build/)
# Their values are masked in a crash reason, which goes to the public archive and job summary.
SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "GEMINI_API_KEY")
CRASH_PREFIX = "Crash: "
SECRET_MASK = "***"
ELLIPSIS = "…"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def crash_reason(exc: BaseException, max_chars: int, secrets: Collection[str] = ()) -> str:
    """Failure reason of a crash: exception type and message on one line, secret values masked,
    cut to `max_chars`."""
    message = " ".join(str(exc).split())
    for secret in secrets:
        if secret:
            message = message.replace(secret, SECRET_MASK)
    reason = f"{CRASH_PREFIX}{type(exc).__name__}" + (f": {message}" if message else "")
    if len(reason) > max_chars:
        reason = reason[: max_chars - len(ELLIPSIS)] + ELLIPSIS
    return reason


def secret_values(env: Mapping[str, str]) -> list[str]:
    return [env[name] for name in SECRET_ENV_VARS if env.get(name)]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def run(
    api: WikipediaApi,
    llm: StructuredLlm,
    judge_llm: StructuredLlm,
    jev: Jev,
    archive_root: Path,
    config: dict,
    title: str,
    now: datetime,
    *,
    tts: Tts,
    stt: Stt,
    fixed_dir: Path,
    audio_root: Path,
    clock: Callable[[], datetime] | None = None,
    usage: Mapping[str, Usage] | None = None,
    secrets: Collection[str] = (),
    dialog_llm: StructuredLlm | None = None,
) -> str:
    """Run the steps implemented so far. Returns the episode ID; raises GateFailure.

    Intro and outro are checked first (`FixedAudioError`), before anything costs money. Audio
    files go to `audio_root/<episode ID>/`. The explicit check (D7, directly after the judge, on
    the final script, before any TTS cost) and the STT deviations are decided by `judge_llm`. The
    explicit flag does not block the episode; a failed check call escalates before the audio
    step. Last, the review sample is
    drawn with the current rate r and written to `09_review.json` (step 12), and the status
    becomes `completed`.

    However the run ends after the episode folder exists, `episode.json` gets `finished_at` (from
    `clock`, default the system clock) and the token counts of `usage`, the counters the API
    clients fill, and `cost`, the USD per role and for TTS at the config prices in force now,
    with those prices (step 13: runtime and cost). A crash (any exception other than `GateFailure`)
    is recorded as `failed` with `crash_reason` (type and message, `secrets` masked) and
    re-raised, so the job fails with the traceback. A crash in the audio step still writes
    `08_audio.json`, so the syntheses paid for stay in the cost log.
    """
    max_reason = config["run"]["failure_reason_max_chars"]
    fixed = load_fixed(fixed_dir, config["models"]["tts"], voices_of(config))
    article = fetch_article(api, title, now)
    record = create_episode(archive_root, article.title, now)
    audio_record = None  # TTS cost of the run, once the audio step started
    try:
        write_json(episode_dir(archive_root, record.episode_id) / ARTICLE_FILE, article.to_dict())

        failure: str | None = None
        if is_living_person(article.categories):
            failure = (
                f"Gate living person: '{article.title}' is a person article without death category."
            )
        else:
            months = config["quotas"]["repeat_gate_months"]
            last = last_published(archive_root, article.title)
            if not repeat_gate_allows(last, now, months):
                assert last is not None
                failure = (
                    f"Gate repeat (D4): '{article.title}' was published on "
                    f"{format_timestamp(last)}, less than {months} months ago."
                )
        if failure:
            record.status, record.failure = "failed", failure
            save_episode(archive_root, record)
            raise GateFailure(failure)
        folder = episode_dir(archive_root, record.episode_id)
        renderer = ParseRenderer(api)
        segments = segment_article(article.wikitext, article.title, article.revid, renderer)
        write_json(
            folder / SEGMENTS_FILE,
            {"revid": article.revid, "segments": [s.to_dict() for s in segments]},
        )
        facts = extract_facts(llm, renderer, article.title, article.revid, segments, config)
        write_json(folder / FACTS_FILE, facts.to_dict())

        quotas = config["quotas"]
        by_target: dict[str, LinkedIntro | None] = {}
        skipped: list[SkippedLink] = []
        written: set[str] = set()

        def make_offer(excluded: frozenset[str]) -> Offer:
            """Both rankings without the excluded facts (D6); linked intros are fetched once."""
            main = rank_facts(
                llm, facts.facts, quotas["main_facts_min"], quotas["main_facts_max"], excluded
            )
            targets, over_limit = bound_targets(
                link_targets(raw_of(main, segments), article.title), quotas["linked_targets_max"]
            )
            for target in targets:
                if target in by_target:
                    continue
                found, not_used = fetch_linked_intros(
                    api, llm, renderer, [target], article.title, config, now
                )
                skipped.extend(not_used)
                by_target[target] = found[0] if found else None
                if found and found[0].title not in written:
                    written.add(found[0].title)
                    write_json(
                        folder / LINKED_DIR / f"{slugify(found[0].title)}.json", found[0].to_dict()
                    )
            current: list[LinkedIntro] = []
            for target in targets:  # an article reached by two links counts once
                intro = by_target[target]
                if intro and intro.title not in {i.title for i in current}:
                    current.append(intro)
            pool = [(f, i) for i in current if i.extraction for f in i.extraction.facts]
            chosen = rank_facts(
                llm,
                [f for f, _ in pool],
                quotas["linked_intro_facts_min"],
                quotas["linked_intro_facts_max"],
                excluded,
            )
            source = {f.segment_id: i for f, i in pool}
            linked = [
                RankedFact(n, source[f.segment_id].title, source[f.segment_id].revid, f)
                for n, f in enumerate(chosen, start=1)
            ]
            main_ranked = ranked(main, article.title, article.revid)
            write_json(
                folder / RANKING_FILE,
                {
                    "main_facts": [r.to_dict() for r in main_ranked],
                    "linked_facts": [r.to_dict() for r in linked],
                    "linked_articles": [
                        {
                            "title": i.title,
                            "revid": i.revid,
                            "file": f"{LINKED_DIR}/{slugify(i.title)}.json",
                        }
                        for i in current
                    ],
                    "skipped_links": [asdict(s) for s in skipped],
                    "excluded_fact_ids": sorted(excluded),
                    "link_targets_over_limit": over_limit,
                    "offer_counts": {
                        "main": offer_count(
                            len(main_ranked), quotas["main_facts_min"], quotas["main_facts_max"]
                        ),
                        "linked": offer_count(
                            len(linked),
                            quotas["linked_intro_facts_min"],
                            quotas["linked_intro_facts_max"],
                        ),
                    },
                },
            )
            failure = offer_gate(len(main_ranked) + len(linked), quotas["min_used_facts"])
            if failure:
                raise GateFailure(failure)
            return Offer(main_ranked, linked)

        state = Orchestration()
        try:
            # The dialog author (dialog and turn regenerations) may use its own client with a
            # larger output ceiling; extraction and ranking keep `llm`.
            orchestrate(state, make_offer, dialog_llm or llm, judge_llm, jev, config)
        except GateFailure as exc:
            if state.dialog:
                write_json(folder / DIALOG_FILE, state.dialog_dict())
                write_json(folder / JUDGEMENT_FILE, state.judgement_dict())
            record.status, record.failure = "failed", str(exc)
            save_episode(archive_root, record)
            raise
        write_json(folder / DIALOG_FILE, state.dialog_dict())
        write_json(folder / JUDGEMENT_FILE, state.judgement_dict())

        assert state.dialog is not None
        # D7 on the final script, before the audio step: a failed check costs no TTS.
        explicit = ExplicitCheck(config["models"]["llm_judge"])
        try:
            check_explicit(explicit, judge_llm, state.dialog.turns)
        except LlmError as exc:
            explicit.error = f"{type(exc).__name__}: {exc}"
            write_json(folder / EXPLICIT_FILE, explicit.to_dict())
            record.status, record.failure = "failed", f"Explicit check (D7): {exc}"
            save_episode(archive_root, record)
            raise GateFailure(record.failure) from exc
        write_json(folder / EXPLICIT_FILE, explicit.to_dict())
        record.explicit = explicit.explicit
        save_episode(archive_root, record)

        audio_record = new_record(config)
        try:
            produce_audio(
                audio_record,
                state.dialog.turns,
                tts,
                stt,
                judge_llm,
                fixed,
                audio_root / record.episode_id,
                config,
            )
        except AudioStepError as exc:
            write_json(folder / AUDIO_FILE, audio_record.to_dict())
            record.status, record.failure = "failed", str(exc)
            save_episode(archive_root, record)
            raise GateFailure(str(exc)) from exc
        except Exception:  # crash: keep the record of the syntheses so far
            write_json(folder / AUDIO_FILE, audio_record.to_dict())
            raise
        write_json(folder / AUDIO_FILE, audio_record.to_dict())

        assert state.offer is not None
        rate = load_rate_state(archive_root, config["review"]).r
        review = build_review(
            record.episode_id,
            record.title,
            state.dialog.turns,
            state.judgements,
            state.offer.by_id(),
            rate,
            explicit,
        )
        write_json(folder / REVIEW_FILE, review.to_dict())
        record.status = "completed"  # saved in `finally`; phase 2 moves it on to published
        return record.episode_id
    except GateFailure:
        raise
    except Exception as exc:  # crash: recorded as failed, then the job fails with the traceback
        record.status, record.failure = "failed", crash_reason(exc, max_reason, secrets)
        raise
    finally:  # every exit, a crash included, records the end and the token counts
        record.finished_at = format_timestamp((clock or _utc_now)())
        record.usage = usage_dict(usage)
        sections = audio_record.to_dict()["sections"] if audio_record is not None else []
        record.cost = episode_cost(record.usage, tts_cost_usd(sections), config["prices"])
        save_episode(archive_root, record)


def write_job_summary(text: str) -> None:
    """Escalation: reason in the job summary of the workflow run (no-op outside Actions)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def write_step_output(name: str, value: str) -> None:
    """Output of the workflow step, for the steps after it (no-op outside Actions)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print('Usage: python -m gut_zu_wissen.run "<article title>"', file=sys.stderr)
        return 2
    config = load_config()
    max_tokens = config["quotas"]["llm_max_tokens"]
    models = config["models"]
    usage = {role: Usage(models[role]) for role in ("author", "llm_judge", "jev")}
    llm = AnthropicLlm(models["author"], max_tokens, usage=usage["author"])
    # The dialog author writes the whole dialog in one answer: its own, larger output ceiling.
    dialog_llm = AnthropicLlm(
        models["author"], config["quotas"]["llm_max_tokens_dialog"], usage=usage["author"]
    )
    judge_llm = AnthropicLlm(models["llm_judge"], max_tokens, usage=usage["llm_judge"])
    jev_config = config["jev"]
    jev = HttpJev(
        config["models"]["jev"],
        jev_config["max_retries"],
        jev_config["backoff_seconds"],
        jev_config["timeout_seconds"],
        os.environ["TYPESAFE_API_KEY"],
        usage=usage["jev"],
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:  # fail before the run spends anything; TTS comes last
        print("::error::GEMINI_API_KEY is not set.", file=sys.stderr)
        return 2
    tts_config = config["tts"]
    tts = GeminiTts(
        config["models"]["tts"],
        voices_of(config),
        tts_config["max_retries"],
        tts_config["backoff_seconds"],
        tts_config["timeout_seconds"],
        gemini_key,
    )
    stt = WhisperStt(config["models"]["stt"], config["stt"])
    secrets = secret_values(os.environ)
    try:
        episode_id = run(
            HttpWikipediaApi(),
            llm,
            judge_llm,
            jev,
            ARCHIVE_ROOT,
            config,
            args[0].strip(),
            datetime.now(UTC),
            tts=tts,
            stt=stt,
            fixed_dir=ASSETS_DIR,
            audio_root=AUDIO_ROOT,
            usage=usage,
            secrets=secrets,
            dialog_llm=dialog_llm,
        )
    except ArticleNotFoundError:
        print(f"Article not found: {args[0]}", file=sys.stderr)
        return 1
    except FixedAudioError as exc:
        print(f"::error::Aborted: {exc}", file=sys.stderr)
        write_job_summary(f"## Episode aborted\n\n{exc}")
        return 1
    except GateFailure as exc:
        print(f"::error::Aborted: {exc}", file=sys.stderr)
        write_job_summary(f"## Episode aborted\n\n{exc}\n\nDetails in the archive folder.")
        return 1
    except Exception as exc:  # crash: escalate, then fail the job with the traceback
        reason = crash_reason(exc, config["run"]["failure_reason_max_chars"], secrets)
        print(f"::error::Aborted: {reason}", file=sys.stderr)
        write_job_summary(
            f"## Episode crashed\n\n{reason}\n\nRecorded as `failed` in the archive folder if "
            "the run got that far; traceback in the job log."
        )
        raise
    write_step_output("episode_id", episode_id)
    audio_file = AUDIO_ROOT / episode_id / EPISODE_AUDIO
    explicit = "yes" if load_episode(ARCHIVE_ROOT, episode_id).explicit else "no"
    review = load_review(ARCHIVE_ROOT, episode_id)
    drawn = f"{len(review.turns)} of {len(review.population)} turns drawn at r = {review.rate}"
    ranking = read_json(episode_dir(ARCHIVE_ROOT, episode_id) / RANKING_FILE)
    offer = offer_summary(ranking["offer_counts"])
    write_job_summary(
        f"## Episode {episode_id}\n\nSteps up to the STT-verified audio, the explicit check and "
        f"the review sample done. {offer}. Explicit: {explicit}. Review: {drawn}. "
        f"Audio: `{audio_file.relative_to(REPO_ROOT)}` (workflow artifact, not committed)."
    )
    print(
        f"Episode {episode_id}: steps up to the review sample done, explicit: {explicit}, "
        f"review: {drawn} ({audio_file})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
