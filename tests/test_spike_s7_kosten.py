"""Pure helpers of the cost spike (spikes/s7_kosten.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spikes"))

from s7_kosten import (  # noqa: E402
    HAIKU,
    SONNET,
    Collect,
    Replay,
    batch_params,
    caching_saving,
    candidate_terms,
    jaccard,
    missing_terms,
    recall,
    request_digest,
    rule_decision,
    rule_margin,
    short_prompt,
    split_for_cache,
    term_found,
    usage_cost,
)

from gut_zu_wissen.judge import VERDICT_PROMPT  # noqa: E402


def test_jaccard_and_recall():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    assert recall(["a", "b", "c", "d"], ["a", "c", "x"]) == 0.5
    assert recall([], ["a"]) == 1.0


def test_usage_cost_counts_cache_and_batch_discount():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
    }
    assert usage_cost(usage, SONNET) == pytest.approx(2.00 + 1.00 + 0.10)
    assert usage_cost(usage, SONNET, batch=True) == pytest.approx(1.55)
    assert usage_cost({"input_tokens": 1_000_000, "output_tokens": 0}, HAIKU) == pytest.approx(0.1)


def test_batch_params_add_effort_only_when_set():
    p = batch_params(SONNET, 100, "sys", "user", {"type": "object"}, None)
    assert p["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert p["messages"] == [{"role": "user", "content": "user"}]
    assert batch_params(SONNET, 100, "s", "u", {}, "low")["output_config"]["effort"] == "low"


def test_split_for_cache_keeps_the_text():
    blocks = split_for_cache("Angebot:\nF1\n\nKontext", 13)
    assert "".join(b["text"] for b in blocks) == "Angebot:\nF1\n\nKontext"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_short_prompt_changes_only_the_reason_instruction():
    short = short_prompt(VERDICT_PROMPT)
    assert short != VERDICT_PROMPT
    assert "höchstens zehn Wörtern" in short
    assert short.startswith(VERDICT_PROMPT[:200])


def test_collect_then_replay_in_order():
    fake = Collect({"candidates": []})
    assert fake.complete_json("s", "u1", {}) == {"candidates": []}
    fake.complete_json("s", "u2", {})
    answers = [(request_digest(s, u), {"n": n}) for n, (s, u, _) in enumerate(fake.requests)]
    replay = Replay(answers)
    assert replay.complete_json("s", "u1", {}) == {"n": 0}
    with pytest.raises(AssertionError):
        replay.complete_json("s", "other", {})


def test_rule_accepts_only_above_both_thresholds():
    assert rule_decision([0.99], [0.97], 0.86, 0.95, 0.85) == "supported"
    assert rule_decision([0.99], [0.90], 0.99, 0.95, 0.85) == "uncertain"  # entity below t
    assert rule_decision([0.99], [], 0.80, 0.95, 0.85) == "uncertain"  # completeness below c
    assert rule_decision([0.04], [0.99], 0.99, 0.95, 0.85) == "rejected"
    assert rule_decision([0.99], [0.02], 0.99, 0.95, 0.85) == "uncertain"  # entity never rejects
    assert rule_margin([0.99], [0.97], 0.86, 0.95, 0.85) == pytest.approx(0.01)


def test_terms_found_by_stem_and_digits():
    assert term_found("Kometen", "Ein Komet zog vorbei.")
    assert not term_found("Kometen", "Einige entstanden in anderen Sternsystemen.")
    assert term_found("1.300", "Dort leben 1300 Menschen.")
    assert not term_found("1908", "im Jahr 1909")


def test_candidate_and_missing_terms():
    text = "Einige Kometen entstanden in anderen Sternsystemen."
    assert candidate_terms(text) == ["Kometen", "Sternsystemen"]
    evidence = "Einige entstanden aber auch in anderen Sternsystemen."
    assert missing_terms(text, evidence) == ["Kometen"]
    assert missing_terms("Im Tagebau Goitzsche wurde Bernstein abgebaut.", "Abbau: Bernstein") == [
        "Tagebau",
        "Goitzsche",
    ]


def test_caching_saving_reads_minus_writes():
    calls = {"extraction": 11, "judge_llm": 0, "regeneration": 4, "conversations": 2}
    saving = caching_saving(calls, {"extraction": 1000, "judge": 800, "offer": 10_000})
    # 10 reads at (2.00 - 0.10) $/M, one write at (2.50 - 2.00) $/M
    assert saving["extraction"] == pytest.approx((10 * 1000 * 1.90 - 1000 * 0.50) / 1e6)
    assert saving["judge"] == 0.0
    # two conversations: two writes, two reads
    assert saving["regeneration"] == pytest.approx((2 * 10_000 * 1.90 - 2 * 10_000 * 0.50) / 1e6)
