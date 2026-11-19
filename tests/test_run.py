import json
import re
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.archive import create_episode, load_episode, save_episode
from gut_zu_wissen.fetch import ArticleNotFoundError
from gut_zu_wissen.run import GateFailure, load_config, run

NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)
CONFIG = {
    "quotas": {
        "repeat_gate_months": 6,
        "max_context_segments": 3,
        "max_chunk_segments": 40,
        "min_fact_words": 4,
        "main_facts": 20,
        "linked_intro_facts": 6,
        "min_used_facts": 1,
    }
}


class FakeLlm:
    """Extraction: offers every segment of the chunk as a fact without context.
    Ranking: returns the offered IDs in reverse order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append(user)
        ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)
        if "ranked_ids" in schema["properties"]:
            return {"ranked_ids": ids[::-1]}
        return {"candidates": [{"segment_id": i, "context_ids": []} for i in ids]}


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
    config = {"quotas": {**CONFIG["quotas"], "min_used_facts": 10}}
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
