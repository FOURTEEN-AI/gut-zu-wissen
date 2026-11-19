import json
import re
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.archive import create_episode, load_episode, save_episode
from gut_zu_wissen.fetch import ArticleNotFoundError
from gut_zu_wissen.jev import JevAbortError
from gut_zu_wissen.run import GateFailure, load_config
from gut_zu_wissen.run import run as run_pipeline

NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)
CONFIG = {
    "judge": {"t": 0.95},
    "quotas": {
        "repeat_gate_months": 6,
        "max_context_segments": 3,
        "max_chunk_segments": 40,
        "min_fact_words": 4,
        "main_facts": 20,
        "linked_intro_facts": 6,
        "min_used_facts": 1,
        "turn_retries": 2,
        "conversation_retries": 2,
    },
}


class FakeLlm:
    """Extraction: offers every segment of the chunk as a fact without context.
    Ranking: returns the offered IDs in reverse order.
    Dialog: two turns citing every offered ID; `felix` overrides fields of the second turn."""

    def __init__(self, felix: dict | None = None, cite: int | None = None) -> None:
        self.calls: list[str] = []
        self.felix, self.cite = felix or {}, cite

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append(user)
        ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
        if "turns" in schema["properties"]:
            ids = ids[: self.cite]
            statements = [{"text": "Aussage.", "fact_ids": ids}]
            turn = {"speaker": "Lena", "text": "Text.", "fact_ids": ids, "statements": statements}
            return {"turns": [turn, {**turn, "speaker": "Felix", **self.felix}]}
        if "ranked_ids" in schema["properties"]:
            return {"ranked_ids": ids[::-1]}
        return {"candidates": [{"segment_id": i, "context_ids": []} for i in ids]}


class FakeJev:
    """Answers every question with `p`, or raises `error`."""

    def __init__(self, p: float = 0.99, error: Exception | None = None) -> None:
        self.p, self.error = p, error

    def ask(self, state: dict, questions: dict) -> dict:
        if self.error:
            raise self.error
        return {key: self.p for key in questions}


class FakeJudgeLlm:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if "supported" in schema["properties"]:
            return {"supported": self.supported, "reason": "Begründung."}
        return {"reason": "Ablehnungsgrund."}


def run(api, llm, root, config, title, now, jev=None, judge=None):
    return run_pipeline(
        api, llm, judge or FakeJudgeLlm(), jev or FakeJev(), root, config, title, now
    )


class FakeApi:
    """Serves the main article and, optionally, more pages: title -> (categories, wikitext)."""

    def __init__(
        self,
        title: str,
        categories: list[str],
        missing: bool = False,
        text: str = "Ein kurzer Satz im Artikel.",
        others: dict[str, tuple[list[str], str]] | None = None,
    ) -> None:
        self.title, self.missing = title, missing
        self.pages = {title: (categories, text), **(others or {})}
        self.revids = {t: 5 + n for n, t in enumerate(self.pages)}

    def post(self, **params: str) -> dict:
        html = "".join(
            f"<p>{p}</p>" if p.startswith("SEGMENTGRENZE") else f"<div>{p}</div>"
            for p in params["text"].split("\n\n")
        )
        return {"parse": {"text": html}}

    def query(self, **params: str) -> dict:
        if params["prop"] == "info|flagged":
            title = params["titles"]
            if title not in self.pages or (self.missing and title == self.title):
                return {"query": {"pages": [{"title": title, "missing": True}]}}
            return {"query": {"pages": [{"title": title, "lastrevid": self.revids[title]}]}}
        title = next(t for t, r in self.revids.items() if str(r) == params["revids"])
        categories, text = self.pages[title]
        return {
            "query": {
                "pages": [
                    {
                        "title": title,
                        "categories": [{"title": c} for c in categories],
                        "revisions": [
                            {
                                "revid": self.revids[title],
                                "timestamp": "2026-10-24T13:21:55Z",
                                "slots": {"main": {"content": text}},
                            }
                        ],
                    }
                ]
            }
        }


def test_successful_run_archives_article(tmp_path):
    episode_id = run(
        FakeApi("Rotfuchs", ["Kategorie:Hunde"]), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW
    )
    assert episode_id == "2026-11-19_rotfuchs"
    folder = tmp_path / "episodes" / episode_id
    article = json.loads((folder / "01_article.json").read_text())
    assert article["revid"] == 5
    assert article["revision_status"] == "latest"
    segments = json.loads((folder / "02_segments.json").read_text())
    assert segments["revid"] == 5
    assert segments["segments"] == [
        {
            "id": "5-0000",
            "type": "sentence",
            "section": "",
            "raw": "Ein kurzer Satz im Artikel.",
            "plain": None,
        }
    ]
    facts = json.loads((folder / "03_facts.json").read_text())
    assert [f["segment_id"] for f in facts["facts"]] == ["5-0000"]
    assert facts["facts"][0]["plain"] == "Ein kurzer Satz im Artikel."
    assert facts["facts"][0]["context_plain"] == []
    assert load_episode(tmp_path, episode_id).status == "running"


def test_ranking_adds_linked_intros_with_own_revid(tmp_path):
    main_text = (
        "Der [[Fuchs]] jagt im [[Wald]] und im [[Max Muster]] sowie im [[Fehlt]].\n\n== A =="
    )
    others = {
        "Fuchs": ([], "Der Fuchs ist ein Raubtier im Wald.\n\n== Aufbau ==\nNicht dabei hier."),
        "Wald": ([], "Ein Wald ist eine große Fläche mit Bäumen."),
        "Max Muster": (["Kategorie:Mann"], "Ein lebender Mensch lebt hier."),
    }
    api = FakeApi("Rotfuchs", [], text=main_text, others=others)
    episode_id = run(api, FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    folder = tmp_path / "episodes" / episode_id
    ranking = json.loads((folder / "04_ranking.json").read_text())
    assert [r["fact"]["plain"] for r in ranking["main_facts"]] == [
        main_text.split("\n")[0]  # the fake renderer keeps the markup
    ]
    assert [(r["rank"], r["article"], r["revid"]) for r in ranking["linked_facts"]] == [
        (1, "Wald", 7),
        (2, "Fuchs", 6),
    ]
    assert [(a["title"], a["file"]) for a in ranking["linked_articles"]] == [
        ("Fuchs", "linked/fuchs.json"),
        ("Wald", "linked/wald.json"),
    ]
    assert ranking["skipped_links"] == [
        {"title": "Max Muster", "reason": "living_person"},
        {"title": "Fehlt", "reason": "not_found"},
    ]
    fuchs = json.loads((folder / "linked" / "fuchs.json").read_text())
    assert fuchs["revid"] == 6
    assert fuchs["intro_wikitext"] == "Der Fuchs ist ein Raubtier im Wald."
    assert not (folder / "linked" / "max-muster.json").exists()


def test_too_few_offered_facts_fail_the_run(tmp_path):
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 10}}
    with pytest.raises(GateFailure, match="D5"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, config, "Rotfuchs", NOW)
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert record.status == "failed" and "D5" in record.failure
    assert (tmp_path / "episodes" / record.episode_id / "04_ranking.json").exists()


def test_living_person_aborts_and_records_failure(tmp_path):
    api = FakeApi("Max Muster", ["Kategorie:Mann"])
    with pytest.raises(GateFailure, match="living person"):
        run(api, FakeLlm(), tmp_path, CONFIG, "Max Muster", NOW)
    record = load_episode(tmp_path, "2026-11-19_max-muster")
    assert record.status == "failed"
    assert "living person" in record.failure
    assert (tmp_path / "episodes" / record.episode_id / "01_article.json").exists()
    assert not (tmp_path / "episodes" / record.episode_id / "02_segments.json").exists()


def test_repeat_gate_aborts(tmp_path):
    old = create_episode(tmp_path, "Rotfuchs", datetime(2026, 8, 1, tzinfo=UTC))
    old.status, old.published_at = "published", "2026-08-02T03:00:00Z"
    save_episode(tmp_path, old)
    with pytest.raises(GateFailure, match="D4"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").status == "failed"


def test_repeat_after_six_months_runs(tmp_path):
    old = create_episode(tmp_path, "Rotfuchs", datetime(2026, 1, 1, tzinfo=UTC))
    old.status, old.published_at = "published", "2026-01-02T03:00:00Z"
    save_episode(tmp_path, old)
    assert run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)


def test_missing_article_creates_no_episode(tmp_path):
    with pytest.raises(ArticleNotFoundError):
        run(FakeApi("X", [], missing=True), FakeLlm(), tmp_path, CONFIG, "X", NOW)
    assert not (tmp_path / "episodes").exists()


def test_real_config_has_repeat_gate_months():
    assert load_config()["quotas"]["repeat_gate_months"] == 6


def test_dialog_is_archived_with_used_facts(tmp_path):
    episode_id = run(
        FakeApi("Rotfuchs", ["Kategorie:Hunde"]), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW
    )
    dialog = json.loads((tmp_path / "episodes" / episode_id / "05_dialog.json").read_text())
    assert [t["speaker"] for t in dialog["turns"]] == ["Lena", "Felix"]
    assert dialog["problems"] == []
    assert dialog["used_fact_ids"] == ["5-0000"]


def test_unknown_fact_id_in_dialog_fails_the_run(tmp_path):
    llm = FakeLlm(felix={"fact_ids": ["9-9999"]})
    # The fake author cannot repair the turn, so every conversation fails the same way.
    with pytest.raises(GateFailure, match="Conversation regeneration budget used up"):
        run(FakeApi("Rotfuchs", []), llm, tmp_path, CONFIG, "Rotfuchs", NOW)
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").status == "failed"


def test_too_few_used_facts_in_dialog_fail_the_run(tmp_path):
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    api = FakeApi("Rotfuchs", [], text="Ein Satz im Artikel.\n\nNoch ein Satz im Artikel.")
    with pytest.raises(GateFailure, match="used by the author"):
        run(api, FakeLlm(cite=1), tmp_path, config, "Rotfuchs", NOW)


def test_judgements_are_archived(tmp_path):
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    path = tmp_path / "episodes" / episode_id / "06_judgement.json"
    data = json.loads(path.read_text())
    assert [(j["turn_index"], j["accepted"], j["decided_by"]) for j in data["judgements"]] == [
        (0, True, "jev"),
        (1, True, "jev"),
    ]
    assert data["used_fact_ids"] == ["5-0000"]


def test_rejected_turns_exclude_the_fact_and_the_empty_offer_fails_the_run(tmp_path):
    # Jev rejects everything: the turn budget runs out, the only fact is excluded (D6) and the
    # next offer is empty (D5).
    with pytest.raises(GateFailure, match="D5"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, FakeJev(0.01))
    folder = tmp_path / "episodes" / "2026-11-19_rotfuchs"
    judgement = json.loads((folder / "06_judgement.json").read_text())
    assert judgement["judgements"][0]["accepted"] is False
    ranking = json.loads((folder / "04_ranking.json").read_text())
    assert ranking["excluded_fact_ids"] == ["5-0000"]
    dialog = json.loads((folder / "05_dialog.json").read_text())
    assert dialog["discarded_conversations"][0]["excluded_fact_ids"] == ["5-0000"]
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").status == "failed"


def test_uncertain_turn_is_decided_by_llm_judge(tmp_path):
    with pytest.raises(GateFailure, match="D5"):
        run(
            FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
            FakeJev(0.5), FakeJudgeLlm(supported=False),
        )  # fmt: skip


def test_jev_abort_fails_the_run(tmp_path):
    jev = FakeJev(error=JevAbortError("401"))
    with pytest.raises(GateFailure, match="Jev aborted"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, jev)
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").status == "failed"


def test_job_summary_is_appended_when_running_in_actions(tmp_path, monkeypatch):
    from gut_zu_wissen.run import write_job_summary

    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    write_job_summary("## Episode aborted")
    write_job_summary("Reason")
    assert path.read_text() == "## Episode aborted\nReason\n"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    write_job_summary("ignored")  # outside Actions: no-op
