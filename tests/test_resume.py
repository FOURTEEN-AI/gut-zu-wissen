"""Audio resume (user decision 2026-11-20): eligibility, successful and failed resumes."""

import json
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.archive import load_episode
from gut_zu_wissen.intro_outro import FixedAudioError
from gut_zu_wissen.llm import LlmError
from gut_zu_wissen.orchestrator import GateFailure
from gut_zu_wissen.resume import (
    ResumeRefused,
    previous_attempts,
    resume_audio,
    resume_entry,
    resume_refusal,
    valid_episode_id,
)
from gut_zu_wissen.tts import TtsError
from gut_zu_wissen.usage import Usage, audio_tts_cost_usd
from test_run import (
    CONFIG,
    LATER,
    NOW,
    FakeApi,
    FakeJudgeLlm,
    FakeLlm,
    FakeStt,
    FakeTts,
    run,
)

EPISODE = "2026-11-19_rotfuchs"
RESUMED_AT = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
RESUME_END = datetime(2026, 11, 20, 9, 12, 30, tzinfo=UTC)
MODEL = "claude-sonnet-5-5"


def folder(root):
    return root / "episodes" / EPISODE


def read(root, name):
    return json.loads((folder(root) / name).read_text())


def failed_in_audio(root):
    """A run that used up the STT budget: everything up to the explicit check is archived."""
    usage = {"author": Usage(MODEL), "llm_judge": Usage(MODEL)}
    usage["author"].add(1000, 200)
    usage["llm_judge"].add(300, 40)
    with pytest.raises(GateFailure, match="STT verification"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), root, CONFIG, "Rotfuchs", NOW,
            stt=FakeStt("Kein Text."), clock=lambda: LATER, usage=usage)  # fmt: skip


def resume(root, tts=None, stt=None, judge=None, usage=None, fixed_dir=None, episode=EPISODE):
    tts = tts or FakeTts()
    return resume_audio(
        root, episode, CONFIG, RESUMED_AT,
        tts=tts, stt=stt or FakeStt(tts=tts), judge_llm=judge or FakeJudgeLlm(),
        fixed_dir=fixed_dir or root / "fixed", audio_root=root / "build",
        workflow_run="123456", clock=lambda: RESUME_END, usage=usage,
    )  # fmt: skip


# ---------------------------------------------------------------- successful resume


def test_resume_completes_the_episode_and_keeps_the_previous_attempt(tmp_path):
    failed_in_audio(tmp_path)
    before = load_episode(tmp_path, EPISODE)
    old_audio = read(tmp_path, "08_audio.json")
    usage = {"llm_judge": Usage(MODEL)}
    usage["llm_judge"].add(50, 10)
    tts = FakeTts()
    assert resume(tmp_path, tts=tts, usage=usage) == EPISODE
    assert len(tts.calls) == 1  # one section, verified at the first synthesis

    record = load_episode(tmp_path, EPISODE)
    assert record.status == "completed" and record.failure is None
    assert record.failed_step is None and record.explicit is False
    assert record.started_at == before.started_at  # still one run
    assert record.finished_at == "2026-11-20T09:12:30Z"
    assert record.resumed == [
        {
            "started_at": "2026-11-20T09:00:00Z",
            "workflow_run": "123456",
            "previous_finished_at": "2026-11-19T10:47:12Z",
            "previous_failure": before.failure,
        }
    ]
    assert record.usage == {
        "author": {"model": MODEL, "calls": 1, "input_tokens": 1000, "output_tokens": 200},
        "llm_judge": {"model": MODEL, "calls": 2, "input_tokens": 350, "output_tokens": 50},
    }
    audio = read(tmp_path, "08_audio.json")
    assert audio["file"] == "episode.flac" and audio["sections"][0]["verified"] is True
    assert audio["previous_attempts"] == [
        {k: v for k, v in old_audio.items() if k != "previous_attempts"}
    ]
    old_tts, new_tts = (
        audio_tts_cost_usd(old_audio),
        audio_tts_cost_usd({**audio, "previous_attempts": []}),
    )
    assert record.cost["usd"]["tts"] == round(old_tts + new_tts, 4)
    prices = CONFIG["prices"]
    assert record.cost["prices"]["claude_sonnet_input"] == prices["claude_sonnet_input"]
    review = read(tmp_path, "09_review.json")
    assert review["episode_id"] == EPISODE and review["population"] == [0, 1]
    assert review["turns"][0]["facts"][0]["id"] == "5-0000"
    assert review["explicit"]["predicted"] is False
    # the final script is untouched
    assert read(tmp_path, "05_dialog.json")["turns"][0]["text"] == "Text."


def test_legacy_folder_without_failed_step_is_resumed(tmp_path):
    # Folders from before `failed_step` (the Tunguska run): 08_audio.json without episode audio.
    failed_in_audio(tmp_path)
    path = folder(tmp_path) / "episode.json"
    data = json.loads(path.read_text())
    del data["failed_step"], data["resumed"]
    path.write_text(json.dumps(data))
    resume(tmp_path)
    assert load_episode(tmp_path, EPISODE).status == "completed"


# ---------------------------------------------------------------- failed resume


def test_failed_resume_records_the_new_reason_and_can_be_resumed_again(tmp_path):
    failed_in_audio(tmp_path)
    first_failure = load_episode(tmp_path, EPISODE).failure
    with pytest.raises(GateFailure, match="TTS section 0"):
        resume(tmp_path, tts=FakeTts(error=TtsError("429")))
    record = load_episode(tmp_path, EPISODE)
    assert record.status == "failed" and record.failure.startswith("TTS section 0")
    assert record.failed_step == "audio"
    assert record.finished_at == "2026-11-20T09:12:30Z"
    assert [r["previous_failure"] for r in record.resumed] == [first_failure]
    assert len(read(tmp_path, "08_audio.json")["previous_attempts"]) == 1
    assert not (folder(tmp_path) / "09_review.json").exists()

    resume(tmp_path)
    record = load_episode(tmp_path, EPISODE)
    assert record.status == "completed"
    assert [r["previous_failure"] for r in record.resumed] == [first_failure, "TTS section 0: 429"]
    assert len(read(tmp_path, "08_audio.json")["previous_attempts"]) == 2


def test_stt_budget_used_up_again_fails_the_resume(tmp_path):
    failed_in_audio(tmp_path)
    tts = FakeTts()
    with pytest.raises(GateFailure, match="STT verification"):
        resume(tmp_path, tts=tts, stt=FakeStt("Kein Text."))
    assert len(tts.calls) == 3
    record = load_episode(tmp_path, EPISODE)
    assert record.status == "failed" and "STT verification" in record.failure
    # TTS cost of both attempts: three syntheses each
    assert record.cost["usd"]["tts"] == audio_tts_cost_usd(read(tmp_path, "08_audio.json"))


def test_crash_in_the_resumed_audio_step_is_recorded(tmp_path):
    failed_in_audio(tmp_path)
    with pytest.raises(RuntimeError, match="socket closed"):
        resume(tmp_path, tts=FakeTts(error=RuntimeError("socket closed")))
    record = load_episode(tmp_path, EPISODE)
    assert record.status == "failed" and record.failure == "Crash: RuntimeError: socket closed"
    assert record.failed_step == "audio"
    assert len(read(tmp_path, "08_audio.json")["previous_attempts"]) == 1


def test_stt_decision_error_fails_the_resume(tmp_path):
    failed_in_audio(tmp_path)

    class Failing(FakeJudgeLlm):
        def complete_json(self, system, user, schema):
            raise LlmError("overloaded")

    with pytest.raises(GateFailure, match="STT verification, section 0"):
        resume(tmp_path, stt=FakeStt("Ganz anderer Text."), judge=Failing())
    assert load_episode(tmp_path, EPISODE).failed_step == "audio"


# ---------------------------------------------------------------- refused


def unchanged(root, action):
    before = {p.name: p.read_bytes() for p in folder(root).iterdir() if p.is_file()}
    with pytest.raises(ResumeRefused) as info:
        action()
    after = {p.name: p.read_bytes() for p in folder(root).iterdir() if p.is_file()}
    assert after == before
    return str(info.value)


def test_completed_episode_is_refused(tmp_path):
    run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    assert "status is 'completed'" in unchanged(tmp_path, lambda: resume(tmp_path))


@pytest.mark.parametrize("status", ["running", "published"])
def test_running_or_published_episode_is_refused(tmp_path, status):
    failed_in_audio(tmp_path)
    path = folder(tmp_path) / "episode.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "status": status}))
    assert f"status is '{status}'" in unchanged(tmp_path, lambda: resume(tmp_path))


def test_gate_failure_before_the_audio_step_is_refused(tmp_path):
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 10}}
    with pytest.raises(GateFailure, match="D5"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, config, "Rotfuchs", NOW)
    assert "before the audio step" in unchanged(tmp_path, lambda: resume(tmp_path))


def test_failed_explicit_check_is_refused(tmp_path):
    judge = FakeJudgeLlm(error=LlmError("The model declined the request."))
    with pytest.raises(GateFailure, match="Explicit check"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, judge=judge)
    assert "explicit check failed" in unchanged(tmp_path, lambda: resume(tmp_path))


def test_missing_intro_refuses_before_anything_changes(tmp_path):
    failed_in_audio(tmp_path)
    before = (folder(tmp_path) / "episode.json").read_bytes()
    with pytest.raises(FixedAudioError):
        resume(tmp_path, fixed_dir=tmp_path / "nowhere")
    assert (folder(tmp_path) / "episode.json").read_bytes() == before


@pytest.mark.parametrize("episode", ["2026-11-19_fehlt", "../state", "2026-11-19_a b", ""])
def test_unknown_or_malformed_episode_id_is_refused(tmp_path, episode):
    with pytest.raises(ResumeRefused):
        resume(tmp_path, episode=episode)


# ---------------------------------------------------------------- pure functions

TURNS = {"turns": [{"index": 0}, {"index": 1}]}
JUDGED = {"judgements": [{"turn_index": 0, "accepted": True}, {"turn_index": 1, "accepted": True}]}
CLEAN = {"explicit": False, "error": None}
FILES = {
    "ranking": {"main_facts": []},
    "dialog": TURNS,
    "judgement": JUDGED,
    "explicit": CLEAN,
    "audio": {"file": None},
}
FAILED = {"status": "failed", "failure": "STT verification: …", "failed_step": "audio"}


def test_refusal_allows_a_failure_in_the_audio_step():
    assert resume_refusal(FAILED, **FILES) is None
    assert resume_refusal(FAILED, **{**FILES, "audio": None}) is None  # crash before 08 exists


@pytest.mark.parametrize(
    ("episode", "files", "expected"),
    [
        ({**FAILED, "status": "completed"}, {}, "status is 'completed'"),
        ({**FAILED, "status": "running"}, {}, "status is 'running'"),
        ({**FAILED, "failed_step": "dialog"}, {}, "step 'dialog'"),
        ({**FAILED, "failed_step": "review"}, {}, "step 'review'"),
        ({**FAILED, "failure": "Explicit check (D7): no"}, {}, "explicit check failed"),
        (FAILED, {"explicit": {"explicit": None, "error": "LlmError: x"}}, "explicit check"),
        (FAILED, {"explicit": {"explicit": None, "error": None}}, "has no result"),
        (FAILED, {"explicit": None}, "07_explicit.json missing"),
        (FAILED, {"dialog": None, "judgement": None}, "05_dialog.json, 06_judgement.json"),
        (FAILED, {"ranking": None}, "04_ranking.json missing"),
        (FAILED, {"dialog": {"turns": []}}, "no turns"),
        (FAILED, {"judgement": {"judgements": JUDGED["judgements"][:1]}}, "every turn"),
        (
            FAILED,
            {"judgement": {"judgements": [{**j, "accepted": False} for j in JUDGED["judgements"]]}},
            "every turn",
        ),
        # legacy folders without failed_step
        ({**FAILED, "failed_step": None}, {"audio": None}, "no 08_audio.json"),
        ({**FAILED, "failed_step": None}, {"audio": {"file": "episode.flac"}}, "after it"),
    ],
)
def test_refusal_reasons(episode, files, expected):
    reason = resume_refusal(episode, **{**FILES, **files})
    assert reason is not None and expected in reason


def test_legacy_failure_in_the_audio_step_is_allowed():
    assert resume_refusal({**FAILED, "failed_step": None}, **FILES) is None
    legacy = {k: v for k, v in FAILED.items() if k != "failed_step"}
    assert resume_refusal(legacy, **FILES) is None


def test_episode_id_pattern():
    assert valid_episode_id("2026-11-20_tunguska-ereignis")
    assert valid_episode_id("2026-11-20_tunguska-ereignis-2")
    for bad in ("2026-11-20_", "tunguska", "2026-11-20_Tunguska", "2026-11-20_a;rm", "x/../y"):
        assert not valid_episode_id(bad)


def test_resume_entry_field_order():
    entry = resume_entry("t", "1", "p", "why")
    assert list(entry) == ["started_at", "workflow_run", "previous_finished_at", "previous_failure"]


def test_previous_attempts_flattens_oldest_first():
    assert previous_attempts(None) == []
    first = {"file": None, "previous_attempts": []}
    second = {"file": None, "previous_attempts": [{"file": None}]}
    assert previous_attempts(first) == [{"file": None}]
    assert previous_attempts(second) == [{"file": None}, {"file": None}]


# ---------------------------------------------------------------- metrics


def test_metrics_count_a_resumed_episode_as_one_run(tmp_path):
    from gut_zu_wissen.metrics import escalation, load_runs, run_cost, runtime

    failed_in_audio(tmp_path)
    resume(tmp_path)
    runs = load_runs(tmp_path)
    assert [r.episode_id for r in runs] == [EPISODE]
    e = escalation(runs)
    assert (e.runs, e.completed, e.failed, e.escalated) == (1, 1, 0, [])
    assert [(x.episode_id, x.outcome, x.resumed) for x in e.resumed] == [(EPISODE, "completed", 1)]
    # 10:30:00-10:47:12 and 09:00:00-09:12:30
    assert runtime(runs).runs[0].seconds == 1032 + 750
    c = run_cost(runs[0], CONFIG["prices"])
    assert c.resumed == 1
    assert {p.part: p.usd for p in c.parts}["tts"] == audio_tts_cost_usd(
        read(tmp_path, "08_audio.json")
    )
