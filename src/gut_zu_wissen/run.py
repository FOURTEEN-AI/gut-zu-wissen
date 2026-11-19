"""Run entry point: `python -m gut_zu_wissen.run "<article title>"`.

Phase 1 steps 2 to 7: fetch the article, apply the gates, segment, extract facts, rank, add the
linked intros, write the dialog, judge it, write the episode folder.
Later steps are added here and move into the orchestrator (step 8).
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gut_zu_wissen.archive import (
    ARTICLE_FILE,
    DIALOG_FILE,
    FACTS_FILE,
    JUDGEMENT_FILE,
    LINKED_DIR,
    RANKING_FILE,
    SEGMENTS_FILE,
    create_episode,
    episode_dir,
    format_timestamp,
    last_published,
    save_episode,
    slugify,
    write_json,
)
from gut_zu_wissen.dialog import used_facts_gate, write_dialog
from gut_zu_wissen.extraction import extract_facts
from gut_zu_wissen.fetch import ArticleNotFoundError, HttpWikipediaApi, WikipediaApi, fetch_article
from gut_zu_wissen.gates import is_living_person, repeat_gate_allows
from gut_zu_wissen.jev import HttpJev, Jev, JevAbortError
from gut_zu_wissen.judge import judge_dialog, judged_facts_gate, used_in_accepted
from gut_zu_wissen.linked import fetch_linked_intros
from gut_zu_wissen.llm import AnthropicLlm, StructuredLlm
from gut_zu_wissen.ranking import RankedFact, link_targets, offer_gate, rank_facts, ranked, raw_of
from gut_zu_wissen.render import ParseRenderer
from gut_zu_wissen.segmentation import segment_article

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.toml"
ARCHIVE_ROOT = REPO_ROOT / "archive"


class GateFailure(Exception):
    """A gate stopped the run; the message is the notice for the owner."""


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
) -> str:
    """Run the steps implemented so far. Returns the episode ID; raises GateFailure."""
    article = fetch_article(api, title, now)
    record = create_episode(archive_root, article.title, now)
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
    main = rank_facts(llm, facts.facts, quotas["main_facts"])
    targets = link_targets(raw_of(main, segments), article.title)
    intros, skipped = fetch_linked_intros(api, llm, renderer, targets, article.title, config, now)
    for intro in intros:
        write_json(folder / LINKED_DIR / f"{slugify(intro.title)}.json", intro.to_dict())
    pool = [(f, i) for i in intros if i.extraction for f in i.extraction.facts]
    chosen = rank_facts(llm, [f for f, _ in pool], quotas["linked_intro_facts"])
    source = {f.segment_id: i for f, i in pool}
    linked = [
        RankedFact(n, source[f.segment_id].title, source[f.segment_id].revid, f)
        for n, f in enumerate(chosen, start=1)
    ]
    write_json(
        folder / RANKING_FILE,
        {
            "main_facts": [r.to_dict() for r in ranked(main, article.title, article.revid)],
            "linked_facts": [r.to_dict() for r in linked],
            "linked_articles": [
                {
                    "title": i.title,
                    "revid": i.revid,
                    "file": f"{LINKED_DIR}/{slugify(i.title)}.json",
                }
                for i in intros
            ],
            "skipped_links": [asdict(s) for s in skipped],
        },
    )
    failure = offer_gate(len(main) + len(linked), quotas["min_used_facts"])
    if failure:
        record.status, record.failure = "failed", failure
        save_episode(archive_root, record)
        raise GateFailure(failure)

    main_ranked = ranked(main, article.title, article.revid)
    dialog = write_dialog(llm, main_ranked, linked, quotas["min_used_facts"])
    write_json(folder / DIALOG_FILE, dialog.to_dict())
    # Turn and conversation regeneration (D1, D5, D6) come with the orchestrator (step 8).
    if dialog.problems:
        failure = (
            f"Gate fact IDs: {len(dialog.problems)} problem(s) in the dialog, see {DIALOG_FILE}."
        )
    else:
        failure = used_facts_gate(len(dialog.used_fact_ids), quotas["min_used_facts"])
    if failure:
        record.status, record.failure = "failed", failure
        save_episode(archive_root, record)
        raise GateFailure(failure)

    facts_by_id = {r.fact.segment_id: r for r in [*main_ranked, *linked]}
    try:
        judgements = judge_dialog(jev, judge_llm, dialog.turns, facts_by_id, config["judge"]["t"])
    except JevAbortError as exc:
        failure = f"Jev aborted the run: {exc}"
        record.status, record.failure = "failed", failure
        save_episode(archive_root, record)
        raise GateFailure(failure) from exc
    used = used_in_accepted(dialog.turns, judgements, facts_by_id)
    write_json(
        folder / JUDGEMENT_FILE,
        {"judgements": [j.to_dict() for j in judgements], "used_fact_ids": used},
    )
    # Turn regeneration and the neighbour cascade (D1) come with the orchestrator (step 8).
    rejected = [j.turn_index for j in judgements if not j.accepted]
    if rejected:
        failure = f"Judge rejected turn(s) {rejected}, see {JUDGEMENT_FILE}."
    else:
        failure = judged_facts_gate(len(used), quotas["min_used_facts"])
    if failure:
        record.status, record.failure = "failed", failure
        save_episode(archive_root, record)
        raise GateFailure(failure)
    return record.episode_id


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print('Usage: python -m gut_zu_wissen.run "<article title>"', file=sys.stderr)
        return 2
    config = load_config()
    max_tokens = config["quotas"]["llm_max_tokens"]
    claude_key = os.environ["CLAUDE_API_KEY"]
    llm = AnthropicLlm(config["models"]["author"], max_tokens, api_key=claude_key)
    judge_llm = AnthropicLlm(config["models"]["llm_judge"], max_tokens, api_key=claude_key)
    jev_config = config["jev"]
    jev = HttpJev(
        config["models"]["jev"],
        jev_config["max_retries"],
        jev_config["backoff_seconds"],
        jev_config["timeout_seconds"],
        os.environ["TYPESAFE_API_KEY"],
    )
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
        )
    except ArticleNotFoundError:
        print(f"Article not found: {args[0]}", file=sys.stderr)
        return 1
    except GateFailure as exc:
        print(f"Aborted: {exc}", file=sys.stderr)
        return 1
    print(f"Episode {episode_id}: steps up to judge done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
