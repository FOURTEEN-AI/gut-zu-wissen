import json
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.archive import create_episode, load_episode, save_episode
from gut_zu_wissen.fetch import ArticleNotFoundError
from gut_zu_wissen.run import GateFailure, load_config, run

NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)
CONFIG = {"quotas": {"repeat_gate_months": 6}}


class FakeApi:
    def __init__(self, title: str, categories: list[str], missing: bool = False) -> None:
        self.title, self.categories, self.missing = title, categories, missing

    def post(self, **params: str) -> dict:
        return {"parse": {"text": "<p>SEGMENTGRENZE00000</p><p>x</p>"}}

    def query(self, **params: str) -> dict:
        if params["prop"] == "info|flagged":
            page = {"title": self.title, "lastrevid": 5}
            if self.missing:
                page["missing"] = True
            return {"query": {"pages": [page]}}
        return {
            "query": {
                "pages": [
                    {
                        "title": self.title,
                        "categories": [{"title": c} for c in self.categories],
                        "revisions": [
                            {
                                "revid": 5,
                                "timestamp": "2026-10-24T13:21:55Z",
                                "slots": {"main": {"content": "Ein kurzer Satz im Artikel."}},
                            }
                        ],
                    }
                ]
            }
        }


def test_successful_run_archives_article(tmp_path):
    episode_id = run(FakeApi("Rotfuchs", ["Kategorie:Hunde"]), tmp_path, CONFIG, "Rotfuchs", NOW)
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
    assert load_episode(tmp_path, episode_id).status == "running"


def test_living_person_aborts_and_records_failure(tmp_path):
    api = FakeApi("Max Muster", ["Kategorie:Mann"])
    with pytest.raises(GateFailure, match="living person"):
        run(api, tmp_path, CONFIG, "Max Muster", NOW)
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
        run(FakeApi("Rotfuchs", []), tmp_path, CONFIG, "Rotfuchs", NOW)
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").status == "failed"


def test_repeat_after_six_months_runs(tmp_path):
    old = create_episode(tmp_path, "Rotfuchs", datetime(2026, 1, 1, tzinfo=UTC))
    old.status, old.published_at = "published", "2026-01-02T03:00:00Z"
    save_episode(tmp_path, old)
    assert run(FakeApi("Rotfuchs", []), tmp_path, CONFIG, "Rotfuchs", NOW)


def test_missing_article_creates_no_episode(tmp_path):
    with pytest.raises(ArticleNotFoundError):
        run(FakeApi("X", [], missing=True), tmp_path, CONFIG, "X", NOW)
    assert not (tmp_path / "episodes").exists()


def test_real_config_has_repeat_gate_months():
    assert load_config()["quotas"]["repeat_gate_months"] == 6
