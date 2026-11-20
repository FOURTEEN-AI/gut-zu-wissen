"""Measurements and exit check of step 13 on synthetic archives (no Wikipedia text)."""

import json

import pytest

from gut_zu_wissen.archive import write_json
from gut_zu_wissen.metrics import (
    RunData,
    build_report,
    cascade_check_decider,
    cascade_lengths,
    cascades,
    cost,
    errors_by_revision,
    escalation,
    exit_check,
    facts_per_minute,
    judge_escalation,
    load_runs,
    main,
    render_markdown,
    run_cost,
    run_outcome,
    runtime,
    runtime_seconds,
    summarize,
    turn_revision_status,
)
from gut_zu_wissen.run import load_config

CRITERIA = {"episodes": 2, "rate": 1.0}
PRICES = {
    "claude_sonnet_input": 3.0,
    "claude_sonnet_output": 15.0,
    "jev": 0.042,
    "gemini_tts_input": 0.5,
    "gemini_tts_output": 9.0,
}
USAGE = {
    "author": {"model": "a", "calls": 10, "input_tokens": 1_000_000, "output_tokens": 100_000},
    "llm_judge": {"model": "a", "calls": 3, "input_tokens": 200_000, "output_tokens": 0},
    "jev": {"model": "jev", "calls": 40, "input_tokens": 500_000, "output_tokens": 9_000},
}


def review_turn(index: int, label: str | None, articles: list[str]) -> dict:
    return {
        "turn_index": index,
        "speaker": "Lena",
        "text": "Text.",
        "previous_text": None,
        "fact_ids": [f"1-{n}" for n, _ in enumerate(articles)],
        "statements": [],
        "facts": [
            {"id": f"1-{n}", "article": a, "revid": 1, "sentence": "Satz.", "context": []}
            for n, a in enumerate(articles)
        ],
        "judgement": {"decided_by": "jev", "jev_p": {}, "jev_error": None, "reason": None},
        "label": label,
        "note": None,
        "comment_id": None,
    }


def review(
    episode_id: str,
    turns: list[dict],
    status: str = "answered",
    rate: float = 1.0,
    explicit_label: str | None = None,
    predicted: bool = False,
) -> dict:
    flag = {"correct": predicted, "should_be_explicit": True, "should_not_be_explicit": False}
    labels = [t["label"] for t in turns]
    return {
        "episode_id": episode_id,
        "title": "Rotfuchs",
        "rate": rate,
        "seed": episode_id,
        "population": [t["turn_index"] for t in turns],
        "draws": [{"turn_index": t["turn_index"], "u": 0.5, "sampled": True} for t in turns],
        "status": status,
        "outcome": (
            ("error" if "not_supported" in labels else "clean") if status == "answered" else None
        ),
        "rate_applied": [],
        "issue": 1,
        "issue_url": None,
        "explicit": {
            "model": "m",
            "predicted": predicted,
            "verdict": "clean",
            "findings": [],
            "reason": None,
            "label": explicit_label,
            "reviewer_flag": flag.get(explicit_label) if explicit_label else None,
            "note": None,
            "comment_id": None,
        },
        "turns": turns,
        "comments": [],
    }


def write_run(
    root,
    episode_id: str,
    *,
    status: str = "running",
    failure: str | None = None,
    finished_at: str | None = "2026-11-19T10:45:30Z",
    usage: dict | None = None,
    cost: dict | None = None,
    revision_status: str = "sighted",
    used: int = 12,
    seconds: float = 900.0,
    section_seconds: tuple = (420.0, 360.0),
    section_usd: tuple = (0.05, 0.04),
    regenerations: list | None = None,
    judgements: list | None = None,
    review_data: dict | None = None,
    linked: dict | None = None,
    audio: bool = True,
    resumed: list | None = None,
    previous_audio: list | None = None,
) -> None:
    folder = root / "episodes" / episode_id
    write_json(
        folder / "episode.json",
        {
            "episode_id": episode_id,
            "title": "Rotfuchs",
            "status": status,
            "started_at": "2026-11-19T10:30:00Z",
            "finished_at": finished_at,
            "published_at": None,
            "explicit": False,
            "failure": failure,
            "usage": USAGE if usage is None else usage,
            **({"cost": cost} if cost is not None else {}),
            **({"failed_step": None, "resumed": resumed} if resumed is not None else {}),
        },
    )
    write_json(
        folder / "01_article.json", {"title": "Rotfuchs", "revision_status": revision_status}
    )
    for title, linked_status in (linked or {}).items():
        write_json(
            folder / "linked" / f"{title.lower()}.json",
            {"title": title, "revision_status": linked_status},
        )
    write_json(
        folder / "05_dialog.json",
        {
            "turns": [],
            "regenerations": regenerations or [],
            "discarded_conversations": [],
        },
    )
    write_json(
        folder / "06_judgement.json",
        {
            "judgements": judgements or [],
            "used_fact_ids": [f"1-{n}" for n in range(used)],
            "cascade_checks": [],
        },
    )
    if audio:
        write_json(
            folder / "08_audio.json",
            {
                "sections": [
                    {"index": n, "seconds": s, "usd": u}
                    for n, (s, u) in enumerate(zip(section_seconds, section_usd, strict=True))
                ],
                "seconds": seconds,
                "usd": None,
                **({"previous_attempts": previous_audio} if previous_audio is not None else {}),
            },
        )
    if review_data is not None:
        write_json(folder / "09_review.json", review_data)


def regen(conversation: int, index: int, kind: str) -> dict:
    return {"conversation": conversation, "turn_index": index, "kind": kind}


# ---------------------------------------------------------------- loading and outcome


def test_load_runs_reads_every_episode_folder(tmp_path):
    write_run(tmp_path, "2026-11-19_a", review_data=review("2026-11-19_a", []))
    write_run(tmp_path, "2026-11-19_b", linked={"Wald": "latest"})
    runs = load_runs(tmp_path)
    assert [r.episode_id for r in runs] == ["2026-11-19_a", "2026-11-19_b"]
    assert runs[0].review is not None and runs[1].review is None
    assert runs[1].linked_status == {"Wald": "latest"}
    assert load_runs(tmp_path / "missing") == []


def test_run_outcome():
    assert run_outcome(RunData("a", {"status": "failed"})) == "failed"
    assert run_outcome(RunData("a", {"status": "running"})) == "incomplete"
    assert run_outcome(RunData("a", {"status": "completed"})) == "completed"
    assert run_outcome(RunData("a", {"status": "published"})) == "completed"
    failed = RunData("a", {"status": "failed"}, review=object())  # type: ignore[arg-type]
    assert run_outcome(failed) == "failed"


def test_run_outcome_of_folders_from_before_the_completed_status():
    done = RunData("a", {"status": "running"}, review=object())  # type: ignore[arg-type]
    assert run_outcome(done) == "completed"  # review sample drawn: the run finished


# ---------------------------------------------------------------- facts per minute


def test_facts_per_minute_uses_final_used_facts_and_audio_minutes(tmp_path):
    write_run(
        tmp_path, "e", used=15, seconds=900.0, section_seconds=(400.0, 350.0),
        review_data=review("e", []),
    )  # fmt: skip
    f = facts_per_minute(load_runs(tmp_path)[0])
    assert f.used_facts == 15 and f.per_minute == pytest.approx(1.0)
    assert f.dialog_seconds == 750.0 and f.per_dialog_minute == pytest.approx(1.2)


def test_facts_per_minute_needs_a_completed_run_with_the_finished_audio(tmp_path):
    write_run(tmp_path, "e", audio=False, review_data=review("e", []))
    write_run(tmp_path, "f", status="failed")  # audio done, explicit check failed
    assert [facts_per_minute(r) for r in load_runs(tmp_path)] == [None, None]


def test_dialog_minutes_unknown_when_a_section_has_no_duration(tmp_path):
    write_run(tmp_path, "e", section_seconds=(400.0, None), review_data=review("e", []))
    assert facts_per_minute(load_runs(tmp_path)[0]).per_dialog_minute is None


# ---------------------------------------------------------------- escalation


def test_escalation_rate_counts_failed_and_incomplete_runs(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []))
    write_run(tmp_path, "b", status="failed", failure="Gate living person: …")
    write_run(tmp_path, "c", status="failed", failure="Crash: RuntimeError: connection reset")
    write_run(tmp_path, "d", finished_at=None)  # still running: no end, no review sample
    write_run(tmp_path, "e", status="completed", review_data=review("e", []))
    write_run(tmp_path, "f", status="completed")  # completed without a drawn sample file
    e = escalation(load_runs(tmp_path))
    assert (e.runs, e.completed, e.failed, e.incomplete) == (6, 3, 2, 1)
    assert e.rate == 0.5
    assert [(x.episode_id, x.outcome, x.reason) for x in e.escalated] == [
        ("b", "failed", "Gate living person: …"),
        ("c", "failed", "Crash: RuntimeError: connection reset"),
        ("d", "incomplete", None),
    ]


def test_escalation_rate_without_runs_is_none():
    assert escalation([]).rate is None


RESUMED = [
    {
        "started_at": "2026-11-19T12:00:00Z",
        "workflow_run": "1",
        "previous_finished_at": "2026-11-19T10:40:00Z",
        "previous_failure": "STT verification: …",
    }
]


def test_resumed_run_counts_once_with_its_final_outcome(tmp_path):
    write_run(tmp_path, "a", status="completed", review_data=review("a", []), resumed=RESUMED)
    write_run(tmp_path, "b", status="failed", failure="TTS section 0: 429", resumed=RESUMED * 2)
    write_run(tmp_path, "c", status="failed", failure="Gate living person: …")
    e = escalation(load_runs(tmp_path))
    assert (e.runs, e.completed, e.failed, e.incomplete) == (3, 1, 2, 0)
    assert [(x.episode_id, x.outcome, x.resumed) for x in e.resumed] == [
        ("a", "completed", 1),
        ("b", "failed", 2),
    ]
    assert [(x.episode_id, x.resumed) for x in e.escalated] == [("b", 2), ("c", 0)]


def test_resumed_run_is_marked_in_the_report(tmp_path):
    write_run(
        tmp_path, "a", status="completed", review_data=review("a", []), resumed=RESUMED,
        finished_at="2026-11-19T12:10:00Z",
    )  # fmt: skip
    text = render_markdown(build_report(load_runs(tmp_path), load_config()))
    assert "Runs: 1 (completed 1, failed 0, incomplete 0)" in text
    assert (
        "Resumed from the audio step (each counts as one run with its final outcome): "
        "`a` (completed, resumed 1×)"
    ) in text
    assert "  - `a` (completed, resumed 1×): 20 min 00 s" in text
    assert "without the time between them" in text
    assert "every attempt included" in text


def test_judge_escalation_share_leaves_out_gate_decisions(tmp_path):
    judgements = [{"decided_by": d} for d in ("jev", "jev", "llm_judge", "gate")]
    write_run(tmp_path, "a", judgements=judgements)
    run = load_runs(tmp_path)[0]
    run.dialog["discarded_conversations"] = [{"judgements": [{"decided_by": "llm_judge"}]}]
    j = judge_escalation([run])
    assert (j.judgements, j.by_jev, j.by_llm_judge, j.rate) == (4, 2, 2, 0.5)


def test_judge_escalation_counts_judgements_of_replaced_turn_versions(tmp_path):
    write_run(tmp_path, "a", judgements=[{"decided_by": "jev"}])
    run = load_runs(tmp_path)[0]
    run.judgement["superseded_judgements"] = [
        {"conversation": 1, "turn_index": 0, "version": 0, "replaced_by": 0,
         "judgement": {"decided_by": "llm_judge"}},
        {"conversation": 1, "turn_index": 1, "version": 0, "replaced_by": 1,
         "judgement": {"decided_by": "gate"}},
    ]  # fmt: skip
    j = judge_escalation([run])
    assert (j.judgements, j.by_jev, j.by_llm_judge, j.rate) == (2, 1, 1, 0.5)


def test_judge_escalation_counts_cascade_checks(tmp_path):
    write_run(tmp_path, "a", judgements=[{"decided_by": "jev"}])
    run = load_runs(tmp_path)[0]
    run.judgement["cascade_checks"] = [
        {"conversation": 1, "turn_index": 2, "p": 0.74, "confirmed": True, "error": None,
         "decided_by": "llm_judge", "reason": "Gleich."},
        {"conversation": 1, "turn_index": 3, "p": 0.99, "confirmed": True, "error": None,
         "decided_by": "jev", "reason": None},
        # before the two-stage check: no decided_by
        {"conversation": 2, "turn_index": 4, "p": 0.38, "confirmed": False, "error": None},
        {"conversation": 2, "turn_index": 5, "p": None, "confirmed": False, "error": "429"},
    ]  # fmt: skip
    j = judge_escalation([run])
    assert (j.judgements, j.by_jev, j.by_llm_judge, j.rate) == (4, 3, 1, 0.25)
    c = cascades([run])
    assert (c.checks, c.checks_by_llm_judge) == (3, 1)


@pytest.mark.parametrize(
    "check,expected",
    [
        ({"p": 0.5, "decided_by": "llm_judge"}, "llm_judge"),
        ({"p": 0.99, "decided_by": "jev"}, "jev"),
        ({"p": 0.74}, "jev"),
        ({"p": None, "error": "429"}, None),
    ],
)
def test_cascade_check_decider(check, expected):
    assert cascade_check_decider(check) == expected


# ---------------------------------------------------------------- cascades


def test_cascade_length_counts_distinct_neighbours_per_triggering_turn():
    regenerations = [
        regen(1, 3, "retry"),
        regen(1, 3, "retry"),  # second retry of the same turn: same cascade
        regen(1, 4, "cascade"),
        regen(1, 4, "cascade"),  # neighbour rejected and regenerated again
        regen(1, 5, "cascade"),
        regen(1, 9, "retry"),  # last turn: no neighbour
        regen(2, 0, "retry"),  # new conversation
        regen(2, 1, "cascade"),
    ]
    assert cascade_lengths(regenerations) == [2, 0, 1]


def test_cascade_length_retries_of_the_same_turn_in_another_conversation_are_new():
    assert cascade_lengths([regen(1, 2, "retry"), regen(2, 2, "retry")]) == [0, 0]


def test_cascade_length_without_regenerations():
    assert cascade_lengths([]) == []


def test_cascades_per_episode_and_overall(tmp_path):
    write_run(
        tmp_path,
        "a",
        regenerations=[regen(1, 0, "retry"), regen(1, 1, "cascade"), regen(1, 2, "cascade")],
    )
    write_run(tmp_path, "b", regenerations=[regen(1, 4, "retry"), regen(1, 5, "cascade")])
    write_run(tmp_path, "c")
    c = cascades(load_runs(tmp_path))
    assert (c.cascades, c.mean, c.max, c.distribution) == (2, 1.5, 2, {1: 1, 2: 1})
    assert [(e.episode_id, e.regenerations, e.lengths, e.max) for e in c.episodes] == [
        ("a", 3, [2], 2),
        ("b", 2, [1], 1),
        ("c", 0, [], None),
    ]


# ---------------------------------------------------------------- runtime


def test_runtime_from_start_to_end():
    episode = {"started_at": "2026-11-19T10:30:00Z", "finished_at": "2026-11-19T10:45:30Z"}
    assert runtime_seconds(episode) == 930.0
    assert runtime_seconds({"started_at": "2026-11-19T10:30:00Z"}) is None


def test_runtime_of_a_resumed_run_sums_its_attempts():
    episode = {
        "started_at": "2026-11-19T10:30:00Z",
        "finished_at": "2026-11-19T12:10:00Z",
        "resumed": RESUMED,
    }
    # 10:30-10:40 and 12:00-12:10; the gap is left out
    assert runtime_seconds(episode) == 1200.0
    second = {**RESUMED[0], "started_at": "2026-11-19T13:00:00Z",
              "previous_finished_at": "2026-11-19T12:10:00Z"}  # fmt: skip
    episode = {**episode, "finished_at": "2026-11-19T13:05:00Z", "resumed": [*RESUMED, second]}
    assert runtime_seconds(episode) == 1500.0
    unknown = [{**RESUMED[0], "previous_finished_at": None}]
    assert runtime_seconds({**episode, "resumed": unknown}) is None


def test_runtime_summary_covers_completed_runs_only(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []), finished_at="2026-11-19T10:40:00Z")
    write_run(tmp_path, "b", review_data=review("b", []), finished_at="2026-11-19T11:00:00Z")
    write_run(tmp_path, "c", status="failed", finished_at="2026-11-19T10:31:00Z")
    write_run(tmp_path, "d", review_data=review("d", []), finished_at=None)  # old record
    r = runtime(load_runs(tmp_path))
    assert (r.completed.count, r.completed.mean, r.completed.max) == (2, 1200.0, 1800.0)
    assert [(x.episode_id, x.seconds) for x in r.runs] == [
        ("a", 600.0),
        ("b", 1800.0),
        ("c", 60.0),
        ("d", None),
    ]


# ---------------------------------------------------------------- cost


def test_cost_sums_tts_claude_and_jev(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []), section_usd=(0.05, 0.04))
    c = run_cost(load_runs(tmp_path)[0], PRICES)
    by_part = {p.part: p.usd for p in c.parts}
    # author 1M * 3 + 0.1M * 15 = 4.5; judge 0.2M * 3 = 0.6; Jev 0.5M * 0.042, output free
    assert by_part == {"tts": 0.09, "author": 4.5, "llm_judge": 0.6, "jev": 0.021}
    assert c.known_usd == pytest.approx(5.211) and c.complete


def test_cost_without_stored_cost_includes_earlier_audio_attempts(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []), section_usd=(0.05, 0.04),
              previous_audio=[{"sections": [{"usd": 0.1}]}], resumed=RESUMED)  # fmt: skip
    c = run_cost(load_runs(tmp_path)[0], PRICES)
    assert {p.part: p.usd for p in c.parts}["tts"] == 0.19 and c.resumed == 1


def test_placeholder_prices_leave_the_claude_cost_unknown_and_are_named(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []))
    prices = {**PRICES, "claude_sonnet_input": "PLACEHOLDER", "claude_sonnet_output": "PLACEHOLDER"}
    total = cost(load_runs(tmp_path), prices)
    assert not total.complete and total.completed_mean_usd is None
    assert total.total_known_usd == pytest.approx(0.111)
    assert total.missing == [
        "author: price not set: claude_sonnet_input, claude_sonnet_output",
        "llm_judge: price not set: claude_sonnet_input, claude_sonnet_output",
    ]


def test_runs_without_recorded_usage_or_tts_cost_are_flagged(tmp_path):
    write_run(tmp_path, "a", usage={}, section_usd=(0.05, None))
    c = run_cost(load_runs(tmp_path)[0], PRICES)
    assert [(p.part, p.usd, p.note) for p in c.parts] == [
        ("tts", None, "a synthesis without token count or price"),
        ("claude, jev", None, "token counts not recorded"),
    ]
    assert not c.complete and c.known_usd == 0


def test_failed_run_before_the_audio_step_has_no_tts_cost(tmp_path):
    write_run(tmp_path, "a", status="failed", audio=False)  # gate failure or crash before TTS
    write_run(tmp_path, "b", audio=False, finished_at=None)  # still running: TTS cost unknown
    runs = load_runs(tmp_path)
    assert run_cost(runs[0], PRICES).parts[0].usd == 0.0
    assert run_cost(runs[1], PRICES).parts[0].usd is None


STORED = {
    "prices": {
        "claude_sonnet_input": 2.0,
        "claude_sonnet_output": 10.0,
        "jev": 0.042,
        "gemini_tts_input": 0.5,
        "gemini_tts_output": 9.0,
    },
    "usd": {"author": 3.0, "llm_judge": 0.4, "jev": 0.021, "tts": 0.07},
}


def test_stored_cost_is_used_with_the_prices_of_its_run(tmp_path):
    write_run(tmp_path, "a", cost=STORED, review_data=review("a", []), section_usd=(0.05, 0.04))
    c = run_cost(load_runs(tmp_path)[0], PRICES)  # today's prices differ: not used
    assert {p.part: p.usd for p in c.parts} == {
        "tts": 0.07,
        "author": 3.0,
        "llm_judge": 0.4,
        "jev": 0.021,
    }
    assert c.complete and not c.todays_prices and c.known_usd == pytest.approx(3.491)


def test_stored_placeholder_price_stays_unknown(tmp_path):
    prices = {**STORED["prices"], "claude_sonnet_output": "PLACEHOLDER"}
    stored = {"prices": prices, "usd": {**STORED["usd"], "author": None, "tts": None}}
    write_run(tmp_path, "a", cost=stored)
    c = run_cost(load_runs(tmp_path)[0], PRICES)
    assert [(p.part, p.usd, p.note) for p in c.parts if p.usd is None] == [
        ("tts", None, "a synthesis without token count or price"),
        ("author", None, "price not set at run time: claude_sonnet_output"),
    ]
    assert not c.complete


def test_report_names_folders_costed_at_todays_prices(tmp_path):
    write_run(tmp_path, "a", cost=STORED, review_data=review("a", []))
    write_run(tmp_path, "b", review_data=review("b", []))  # older folder: no stored cost
    total = cost(load_runs(tmp_path), PRICES)
    assert total.todays_prices == ["b"]
    report = build_report(load_runs(tmp_path), {"exit_criteria": CRITERIA, "prices": PRICES})
    assert (
        "- Computed with today's prices (folders from before costs were stored at run time): `b`"
        in render_markdown(report)
    )


def test_no_mean_cost_while_a_completed_run_is_partial(tmp_path):
    write_run(tmp_path, "a", review_data=review("a", []), usage={}, section_usd=(0.1, 0.1))
    write_run(tmp_path, "b", review_data=review("b", []), usage={}, section_usd=(0.3, 0.1))
    total = cost(load_runs(tmp_path), PRICES)
    assert total.completed_mean_usd is None  # usage missing: no mean of partial costs
    assert total.total_known_usd == pytest.approx(0.6)


def test_mean_cost_when_everything_is_known(tmp_path):
    usage = {"jev": {"model": "jev", "calls": 1, "input_tokens": 1_000_000, "output_tokens": 0}}
    write_run(tmp_path, "a", review_data=review("a", []), usage=usage, section_usd=(0.1, 0.1))
    write_run(tmp_path, "b", review_data=review("b", []), usage=usage, section_usd=(0.3, 0.1))
    write_run(tmp_path, "c", status="failed", usage=usage, audio=False)
    total = cost(load_runs(tmp_path), PRICES)
    assert total.complete and total.completed_mean_usd == pytest.approx(0.342)
    assert total.total_known_usd == pytest.approx(0.726)


# ---------------------------------------------------------------- error rate by revision status


def test_turn_revision_status():
    status = {"Rotfuchs": "sighted", "Wald": "latest"}
    assert turn_revision_status(["Rotfuchs"], status) == "sighted"
    assert turn_revision_status(["Rotfuchs", "Rotfuchs"], status) == "sighted"
    assert turn_revision_status(["Rotfuchs", "Wald"], status) == "mixed"
    assert turn_revision_status(["Fehlt"], status) == "unknown"


def test_errors_by_revision_status_for_turns_and_episodes(tmp_path):
    turns_a = [
        review_turn(0, "supported", ["Rotfuchs"]),
        review_turn(1, "not_supported", ["Wald"]),
        review_turn(2, "supported", ["Rotfuchs", "Wald"]),
    ]
    write_run(tmp_path, "a", review_data=review("a", turns_a), linked={"Wald": "latest"})
    turns_b = [review_turn(0, "supported", ["Rotfuchs"]), review_turn(1, None, ["Rotfuchs"])]
    write_run(tmp_path, "b", review_data=review("b", turns_b, status="pending"))
    turns_c = [review_turn(0, "not_supported", ["Rotfuchs"])]
    write_run(tmp_path, "c", revision_status="latest", review_data=review("c", turns_c))
    b = errors_by_revision(load_runs(tmp_path))
    turns = {k: (r.labelled, r.not_supported, r.rate) for k, r in b.turns.items()}
    assert turns == {
        "latest": (2, 2, 1.0),  # Wald in a, Rotfuchs (latest) in c
        "mixed": (1, 0, 0.0),
        "sighted": (2, 0, 0.0),  # unlabelled turn of b left out
    }
    episodes = {k: (r.labelled, r.not_supported, r.rate) for k, r in b.episodes.items()}
    assert episodes == {"latest": (1, 1, 1.0), "sighted": (1, 1, 1.0)}  # b is pending


# ---------------------------------------------------------------- exit check


def test_exit_check_needs_answered_reviews_at_full_rate(tmp_path):
    answered = [review_turn(0, "supported", ["Rotfuchs"])]
    write_run(tmp_path, "a", review_data=review("a", answered, explicit_label="correct"))
    write_run(
        tmp_path, "b", review_data=review("b", answered, rate=2 / 3, explicit_label="correct")
    )
    write_run(tmp_path, "c", review_data=review("c", [review_turn(0, None, [])], "pending"))
    write_run(tmp_path, "d", review_data=review("d", [], status="not_sampled"))
    write_run(tmp_path, "e", review_data=review("e", answered))  # no explicit label
    check = exit_check(load_runs(tmp_path), CRITERIA)
    assert check.qualifying == ["a"] and not check.met
    write_run(tmp_path, "f", review_data=review("f", answered, explicit_label="correct"))
    check = exit_check(load_runs(tmp_path), CRITERIA)
    assert check.qualifying == ["a", "f"] and check.met


def test_real_config_has_the_exit_criteria_of_the_plan():
    assert load_config()["exit_criteria"] == {"episodes": 5, "rate": 1.0}


# ---------------------------------------------------------------- report and command line


def test_summarize():
    assert summarize([]).mean is None
    s = summarize([1, 2, 6])
    assert (s.count, s.mean, s.min, s.max) == (3, 3.0, 1, 6)


def _archive(tmp_path) -> None:
    turns = [review_turn(0, "supported", ["Rotfuchs"])]
    for n in range(3):
        write_run(
            tmp_path,
            f"2026-11-1{n}_e",
            review_data=review(f"2026-11-1{n}_e", turns, explicit_label="correct"),
            regenerations=[regen(1, 0, "retry"), regen(1, 1, "cascade")],
        )
    write_run(tmp_path, "2026-11-19_f", status="failed", failure="Gate repeat (D4): …")


def test_report_and_markdown(tmp_path):
    _archive(tmp_path)
    config = {"exit_criteria": CRITERIA, "prices": PRICES}
    report = build_report(load_runs(tmp_path), config)
    assert report.exit.met and len(report.exit.qualifying) == 3
    assert report.escalation.rate == 0.25
    assert report.explicit.episodes == 3 and report.explicit.false_positive_rate == 0.0
    text = render_markdown(report)
    assert "**Met:** 3 of 2 episodes fully reviewed" in text
    assert "`2026-11-19_f` (failed): Gate repeat (D4): …" in text
    assert "| sighted | 3 | 0 | 0.0 % |" in text
    assert "| Revision status | Answered reviews | With error | Error rate |" in text
    assert report.facts_per_minute.count == 3  # the failed run is left out
    primary = text.index("- Primary, dialog audio only: mean 0.92")
    assert primary < text.index("- Secondary, whole episode audio (intro and outro included): ")
    assert "| Per dialog minute (primary) |" in text
    assert "- Cascade checks: 0, decided by the LLM judge: 0" in text


def test_markdown_on_an_empty_archive():
    text = render_markdown(build_report([], {"exit_criteria": CRITERIA, "prices": PRICES}))
    assert "**Not met:** 0 of 2" in text and "No labels yet." in text


def test_command_line_prints_markdown_and_writes_the_job_summary(tmp_path, monkeypatch, capsys):
    _archive(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert main(["--archive", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Phase 1 exit criteria (step 13)")
    assert "3 of 5 episodes" in out  # real config: 5 episodes
    assert summary.read_text() == out


def test_command_line_json(tmp_path, capsys):
    _archive(tmp_path)
    assert main(["--archive", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["exit"]["met"] is False and len(data["exit"]["qualifying"]) == 3
    assert data["cascades"]["distribution"] == {"1": 3}
