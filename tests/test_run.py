import io
import json
import re
from datetime import UTC, datetime

import numpy as np
import pytest
import soundfile as sf

from gut_zu_wissen.archive import create_episode, load_episode, save_episode
from gut_zu_wissen.fetch import ArticleNotFoundError
from gut_zu_wissen.intro_outro import FixedAudioError, synthesize_fixed
from gut_zu_wissen.jev import JevAbortError
from gut_zu_wissen.llm import LlmError
from gut_zu_wissen.run import GateFailure, load_config
from gut_zu_wissen.run import run as run_pipeline
from gut_zu_wissen.stt import Transcript
from gut_zu_wissen.tts import Synthesis, TtsError, voices_of
from gut_zu_wissen.usage import Usage
from gut_zu_wissen.verification import NO_ANSWER

NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)
REAL = load_config()
CONFIG = {
    "judge": {"t": 0.95},
    "quotas": {
        "repeat_gate_months": 6,
        "max_context_segments": 3,
        "max_chunk_segments": 40,
        "min_fact_words": 4,
        "main_facts_min": 20,
        "main_facts_max": 40,
        "linked_intro_facts_min": 6,
        "linked_intro_facts_max": 12,
        "einordnung_facts_max": 0,  # off here; the Einordnung tests switch it on
        "linked_targets_max": 30,
        "min_used_facts": 1,
        "max_turns_per_fact": REAL["quotas"]["max_turns_per_fact"],
        "turn_retries": 2,
        "gate_retries": 3,
        "conversation_retries": 2,
        "tts_section_max_chars": 6000,
        "audio_section_retries": 2,
        "target_loudness_lufs": -16,
        "true_peak_ceiling_dbtp": -1.5,
    },
    "models": REAL["models"],
    "tts": REAL["tts"],
    "stt": REAL["stt"],
    "audio": REAL["audio"],
    "prices": REAL["prices"],
    "review": REAL["review"],
    "run": REAL["run"],
}


class FakeLlm:
    """Extraction: offers every segment of the chunk as a fact without context.
    Ranking: returns the offered IDs in reverse order.
    Dialog: two turns citing every offered ID; `felix` overrides fields of the second turn."""

    def __init__(self, felix: dict | None = None, cite: int | None = None) -> None:
        self.calls: list[str] = []
        self.felix, self.cite = felix or {}, cite

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
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
    """Judge verdicts and reasons; the explicit check answers `explicit` or raises `error`."""

    def __init__(
        self,
        supported: bool = True,
        explicit: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.supported, self.error = supported, error
        self.explicit = explicit or {"findings": [], "verdict": "clean", "reason": "Nichts."}
        self.scripts: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        if "verdict" in schema["properties"]:
            self.scripts.append(user)
            if self.error:
                raise self.error
            return self.explicit
        if "supported" in schema["properties"]:
            return {"supported": self.supported, "reason": "Begründung."}
        return {"reason": "Ablehnungsgrund."}


def tone_wav(seconds: float) -> bytes:
    """A quiet tone with edge silence, standing in for speech (24 kHz, 16 bit)."""
    sr = 24_000
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sr)) / sr)
    y = np.concatenate([np.zeros(sr // 4), y, np.zeros(sr // 2)])
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeTts:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list] = []
        self.error = error

    def synthesize(self, turns):
        self.calls.append(list(turns))
        if self.error:
            raise self.error
        return Synthesis(tone_wav(1.0), 100, 250)


class FakeStt:
    """Hears `text`; without it, exactly what the TTS was last asked to say."""

    def __init__(self, text: str | None = None, tts: FakeTts | None = None) -> None:
        self.text, self.tts = text, tts

    def transcribe(self, y, sr):
        if self.text is not None:
            return Transcript(self.text)
        assert self.tts is not None
        return Transcript(" ".join(text for _, text in self.tts.calls[-1]))


def run(
    api, llm, root, config, title, now, jev=None, judge=None, tts=None, fixed=True, stt=None,
    clock=None, usage=None, dialog_llm=None, extraction_llm=None,
):  # fmt: skip
    fixed_dir = root / "fixed"
    tts = tts or FakeTts()
    if fixed and not fixed_dir.exists():
        synthesize_fixed(FakeTts(), fixed_dir, config["models"]["tts"], voices_of(config))
    return run_pipeline(
        api, llm, judge or FakeJudgeLlm(), jev or FakeJev(), root, config, title, now,
        tts=tts, stt=stt or FakeStt(tts=tts), fixed_dir=fixed_dir, audio_root=root / "build",
        clock=clock, usage=usage, dialog_llm=dialog_llm, extraction_llm=extraction_llm,
    )  # fmt: skip


def test_dialog_author_uses_its_own_llm_when_given(tmp_path):
    # Extraction and ranking keep `llm`; the dialog goes to `dialog_llm` (larger output ceiling).
    llm, dialog_llm = FakeLlm(), FakeLlm()
    run(
        FakeApi("Rotfuchs", ["Kategorie:Hunde"]), llm, tmp_path, CONFIG, "Rotfuchs", NOW,
        dialog_llm=dialog_llm,
    )  # fmt: skip
    assert len(dialog_llm.calls) == 1 and llm.calls  # one dialog; extraction/ranking elsewhere
    assert dialog_llm.calls[0] not in llm.calls


def test_extraction_of_main_and_linked_articles_uses_the_extraction_llm(tmp_path):
    # User decision 2026-11-20: extraction on its own model (Haiku 5.5); ranking stays on `llm`.
    main_text = "Der [[Fuchs]] jagt im Wald bei Nacht.\n\n== A =="
    others = {"Fuchs": ([], "Der Fuchs ist ein Raubtier im Wald.")}
    api = FakeApi("Rotfuchs", [], text=main_text, others=others)
    llm, extraction_llm = FakeLlm(), FakeLlm()
    episode_id = run(api, llm, tmp_path, CONFIG, "Rotfuchs", NOW, extraction_llm=extraction_llm)
    extracted = [u for u in extraction_llm.calls if u.startswith("Abschnitt:")]
    assert len(extracted) == len(extraction_llm.calls) == 2  # main article chunk + linked intro
    assert any("Raubtier" in u for u in extracted)
    assert not any(u.startswith("Abschnitt:") for u in llm.calls) and llm.calls  # ranking
    folder = tmp_path / "episodes" / episode_id
    assert json.loads((folder / "linked" / "fuchs.json").read_text())["extraction"]["facts"]


def test_extraction_model_and_prices_are_configured():
    assert REAL["models"]["extraction"] == "claude-haiku-5-5"
    assert REAL["models"]["author"] == REAL["models"]["llm_judge"] == "claude-sonnet-5-5"
    prices = REAL["prices"]
    for key in ("input", "output", "cache_write", "cache_read"):
        assert isinstance(prices[f"claude_haiku_{key}"], float)
    assert prices["claude_haiku_cache_write"] == 1.25 * prices["claude_haiku_input"]


def test_dialog_ceiling_is_its_own_config_value():
    quotas = REAL["quotas"]
    assert quotas["llm_max_tokens_dialog"] > quotas["llm_max_tokens"]


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
    assert load_episode(tmp_path, episode_id).status == "completed"


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
    assert ranking["link_targets_over_limit"] == []
    assert ranking["offer_counts"] == {
        "main": {"offered": 1, "min": 20, "max": 40, "below_min": True},
        "linked": {"offered": 2, "min": 6, "max": 12, "below_min": True},
        "einordnung": {"offered": 0, "min": 0, "max": 0, "below_min": False},
    }
    assert list(ranking)[-1] == "einordnung_facts" and ranking["einordnung_facts"] == []


class SystemRecordingLlm(FakeLlm):
    def __init__(self) -> None:
        super().__init__()
        self.ranking_calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        if "ranked_ids" in schema["properties"]:
            self.ranking_calls.append((system, user))
        return super().complete_json(system, user, schema)


def test_linked_ranking_gets_main_title_and_linked_articles(tmp_path):
    main_text = "Der [[Fuchs]] jagt im [[Wald]].\n\n== A =="
    others = {
        "Fuchs": ([], "Der Fuchs ist ein Raubtier im Wald."),
        "Wald": ([], "Ein Wald ist eine große Fläche mit Bäumen."),
    }
    llm = SystemRecordingLlm()
    run(
        FakeApi("Rotfuchs", [], text=main_text, others=others),
        llm,
        tmp_path,
        CONFIG,
        "Rotfuchs",
        NOW,
    )
    system, user = llm.ranking_calls[-1]
    assert "direkt mit dem Hauptthema „Rotfuchs“ verbunden" in " ".join(system.split())
    assert "(verlinkter Artikel: Fuchs)" in user and "(verlinkter Artikel: Wald)" in user


def test_link_targets_beyond_the_bound_are_not_fetched(tmp_path):
    main_text = (
        "Der [[Fuchs]] jagt im [[Wald]] und im [[Max Muster]] sowie im [[Fehlt]].\n\n== A =="
    )
    others = {
        "Fuchs": ([], "Der Fuchs ist ein Raubtier im Wald."),
        "Wald": ([], "Ein Wald ist eine große Fläche mit Bäumen."),
    }
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "linked_targets_max": 1}}
    api = FakeApi("Rotfuchs", [], text=main_text, others=others)
    episode_id = run(api, FakeLlm(), tmp_path, config, "Rotfuchs", NOW)
    ranking = json.loads((tmp_path / "episodes" / episode_id / "04_ranking.json").read_text())
    assert [a["title"] for a in ranking["linked_articles"]] == ["Fuchs"]
    assert ranking["skipped_links"] == []
    assert ranking["link_targets_over_limit"] == ["Wald", "Max Muster", "Fehlt"]


def test_too_few_offered_facts_fail_the_run(tmp_path):
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 10}}
    with pytest.raises(GateFailure, match="D5"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, config, "Rotfuchs", NOW)
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert record.status == "failed" and "D5" in record.failure
    assert record.failed_step == "dialog"
    assert (tmp_path / "episodes" / record.episode_id / "04_ranking.json").exists()


def test_living_person_aborts_and_records_failure(tmp_path):
    api = FakeApi("Max Muster", ["Kategorie:Mann"])
    with pytest.raises(GateFailure, match="living person"):
        run(api, FakeLlm(), tmp_path, CONFIG, "Max Muster", NOW)
    record = load_episode(tmp_path, "2026-11-19_max-muster")
    assert record.status == "failed"
    assert "living person" in record.failure and record.failed_step == "gates"
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


def test_author_answers_with_aliases_and_the_archive_holds_only_segment_ids(tmp_path):
    api = FakeApi("Rotfuchs", [], text="Erster Satz im Artikel.\n\nZweiter Satz im Artikel.")
    llm = FakeLlm()
    episode_id = run(api, llm, tmp_path, CONFIG, "Rotfuchs", NOW)
    (author_input,) = [c for c in llm.calls if c.startswith("[F1]")]
    assert "[F2]" in author_input and "5-000" not in author_input
    folder = tmp_path / "episodes" / episode_id
    dialog = json.loads((folder / "05_dialog.json").read_text())
    assert [t["fact_ids"] for t in dialog["turns"]] == [["5-0001", "5-0000"]] * 2
    assert dialog["used_fact_ids"] == ["5-0001", "5-0000"]
    judgement = json.loads((folder / "06_judgement.json").read_text())
    assert judgement["used_fact_ids"] == ["5-0001", "5-0000"]
    assert dialog.pop("aliases") == {"F1": "5-0001", "F2": "5-0000"}  # last field, tracing only
    for data in (dialog, judgement):
        assert not re.search(r'"F\d+"', json.dumps(data))


def test_einordnung_is_archived_and_offered_first_outside_the_ranking(tmp_path):
    text = (
        "Erster Satz im Artikel hier.\n\nKurz.\n\nZweiter Satz im Artikel hier.\n\n"
        "Dritter Satz im Artikel hier.\n\n== A ==\nEin Satz im Abschnitt A."
    )
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "einordnung_facts_max": 2}}
    llm = SystemRecordingLlm()
    episode_id = run(FakeApi("Rotfuchs", [], text=text), llm, tmp_path, config, "Rotfuchs", NOW)
    folder = tmp_path / "episodes" / episode_id
    ranking = json.loads((folder / "04_ranking.json").read_text())
    assert list(ranking)[-1] == "einordnung_facts"
    einordnung = ranking["einordnung_facts"]
    # "Kurz." fails the word gate and is skipped; the maximum stops after two
    assert [(r["rank"], r["fact"]["segment_id"]) for r in einordnung] == [
        (1, "5-0000"),
        (2, "5-0002"),
    ]
    assert einordnung[1]["fact"]["context_ids"] == ["5-0000", "5-0001"]
    # extracted too, but offered once only, as Einordnung; the ranking never sees them
    main = [r["fact"]["segment_id"] for r in ranking["main_facts"]]
    assert main == ["5-0004", "5-0003"]
    assert all("[5-0000]" not in user and "[5-0002]" not in user for _, user in llm.ranking_calls)
    assert ranking["offer_counts"]["einordnung"] == {
        "offered": 2,
        "min": 0,
        "max": 2,
        "below_min": False,
    }
    (author_input,) = [c for c in llm.calls if c.startswith("[F1]")]
    lines = [line for line in author_input.split("\n") if line.startswith("[")]
    assert lines == [
        "[F1] (Einordnung) Erster Satz im Artikel hier.",
        "[F2] (Einordnung) Zweiter Satz im Artikel hier.",
        "[F3] Ein Satz im Abschnitt A.",
        "[F4] Dritter Satz im Artikel hier.",
    ]
    dialog = json.loads((folder / "05_dialog.json").read_text())
    assert list(dialog["aliases"].values())[:2] == ["5-0000", "5-0002"]


def test_first_prose_sentence_stays_einordnung_one_after_a_rejection(tmp_path):
    # Every turn of conversation 1 cites F1, the first prose sentence, and is rejected; D6 does
    # not exclude it, so the fresh offer still opens with it (archive 2026-11-20_bernstein).
    text = "Erster Satz im Artikel hier.\n\nZweiter Satz im Artikel hier.\n\n== A ==\nEin Satz."
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "einordnung_facts_max": 2}}
    llm = RetryLlm()
    episode_id = run(
        FakeApi("Rotfuchs", [], text=text), llm, tmp_path, config, "Rotfuchs", NOW, WrongTextJev()
    )
    folder = tmp_path / "episodes" / episode_id
    dialog = json.loads((folder / "05_dialog.json").read_text())
    (discarded,) = dialog["discarded_conversations"]
    assert discarded["regeneration"] == "fresh" and discarded["excluded_fact_ids"] == []
    ranking = json.loads((folder / "04_ranking.json").read_text())
    assert [r["fact"]["segment_id"] for r in ranking["einordnung_facts"]] == ["5-0000", "5-0001"]
    assert ranking["excluded_fact_ids"] == []
    assert dialog["turns"][0]["fact_ids"] == ["5-0000"]


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


class RetryLlm(FakeLlm):
    """The first dialog is bad in turn 0; every regenerated turn is bad again; a second
    conversation from a fresh offer is good."""

    def __init__(self) -> None:
        super().__init__()
        self.dialogs = 0

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        if "turns" in schema["properties"]:
            self.dialogs += 1
            text = "Falsch." if self.dialogs == 1 else "Gut."
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)[:1]
            statements = [{"text": text, "fact_ids": ids}]
            turn = {"speaker": "Lena", "text": text, "fact_ids": ids, "statements": statements}
            return {"turns": [turn, {**turn, "speaker": "Felix"}]}
        if "speaker" in schema["properties"]:
            ids = re.findall(r"^\[([^\]]+)\]", user, re.MULTILINE)[:1]
            statements = [{"text": "Falsch.", "fact_ids": ids}]
            return {"speaker": "Lena", "text": "Falsch.", "fact_ids": ids, "statements": statements}
        return super().complete_json(system, user, schema)


class WrongTextJev(FakeJev):
    def ask(self, state: dict, questions: dict) -> dict:
        p = 0.01 if "Falsch." in state["redebeitrag"]["text"] else 0.99
        return {key: p for key in questions}


def test_conversation_regeneration_succeeds_with_a_fresh_offer(tmp_path):
    api = FakeApi("Rotfuchs", [], text="Erster Satz im Artikel.\n\nZweiter Satz im Artikel.")
    llm = RetryLlm()
    episode_id = run(api, llm, tmp_path, CONFIG, "Rotfuchs", NOW, WrongTextJev())
    folder = tmp_path / "episodes" / episode_id
    dialog = json.loads((folder / "05_dialog.json").read_text())
    assert llm.dialogs == 2 and dialog["conversation"] == 2
    assert [t["text"] for t in dialog["turns"]] == ["Gut.", "Gut."]
    (discarded,) = dialog["discarded_conversations"]
    assert discarded["conversation"] == 1
    assert discarded["excluded_fact_ids"] == ["5-0001"]  # ranked first, so the one that was cited
    assert "budget" in discarded["reason"]
    assert [r["kind"] for r in dialog["regenerations"]] == ["retry", "retry"]
    ranking = json.loads((folder / "04_ranking.json").read_text())
    assert [r["fact"]["segment_id"] for r in ranking["main_facts"]] == ["5-0000"]
    assert ranking["excluded_fact_ids"] == ["5-0001"]
    judgement = json.loads((folder / "06_judgement.json").read_text())
    assert all(j["accepted"] for j in judgement["judgements"])
    assert judgement["used_fact_ids"] == ["5-0000"]


class ContinuationLlm(RetryLlm):
    """The first dialog is bad in turn 1 only; every regenerated turn is bad again; the
    continuation (second dialog call) is good."""

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        answer = super().complete_json(system, user, schema)
        if "turns" in schema["properties"] and self.dialogs == 1:
            good = {**answer["turns"][0], "text": "Gut.", "statements": [
                {**answer["turns"][0]["statements"][0], "text": "Gut."}
            ]}  # fmt: skip
            answer = {"turns": [good, answer["turns"][1]]}
        return answer


def test_exhausted_turn_budget_continues_the_dialog_and_resume_loads_it(tmp_path):
    from gut_zu_wissen.dialog import CONTINUE_PROMPT
    from gut_zu_wissen.resume import resume_refusal, turns_from

    api = FakeApi("Rotfuchs", [], text="Erster Satz im Artikel.\n\nZweiter Satz im Artikel.")
    llm = ContinuationLlm()
    systems: list[str] = []
    inner = llm.complete_json

    def recording(system: str, user: str, schema: dict, **cache) -> dict:
        systems.append(system)
        return inner(system, user, schema, **cache)

    llm.complete_json = recording
    episode_id = run(api, llm, tmp_path, CONFIG, "Rotfuchs", NOW, WrongTextJev())
    folder = tmp_path / "episodes" / episode_id
    dialog = json.loads((folder / "05_dialog.json").read_text())
    assert llm.dialogs == 2 and dialog["conversation"] == 2
    assert CONTINUE_PROMPT.format(min_used=1) in systems
    assert [(t["index"], t["text"]) for t in dialog["turns"]] == [
        (0, "Gut."),
        (1, "Gut."),
        (2, "Gut."),
    ]
    (discarded,) = dialog["discarded_conversations"]
    assert list(discarded)[-4:] == [
        "regeneration",
        "continued_from",
        "fresh_reason",
        "continuation_reason",
    ]
    assert discarded["continuation_reason"] == discarded["judgements"][0]["reason"]
    assert (discarded["regeneration"], discarded["continued_from"]) == ("continuation", 1)
    assert [t["index"] for t in discarded["dialog"]["turns"]] == [1]
    assert [j["turn_index"] for j in discarded["judgements"]] == [1]
    assert [r["budget"] for r in dialog["regenerations"]] == ["judge", "judge"]
    judgement = json.loads((folder / "06_judgement.json").read_text())
    assert [j["turn_index"] for j in judgement["judgements"]] == [0, 1, 2]
    assert all(j["accepted"] for j in judgement["judgements"])
    # resume (audio step) still reads the final dialog
    assert [t.index for t in turns_from(dialog)] == [0, 1, 2]
    refusal = resume_refusal(
        {"status": "failed", "failed_step": "audio"},
        ranking=json.loads((folder / "04_ranking.json").read_text()),
        dialog=dialog,
        judgement=judgement,
        explicit=json.loads((folder / "07_explicit.json").read_text()),
        audio=None,
    )
    assert refusal is None


def test_audio_is_synthesised_normalised_and_archived(tmp_path):
    tts = FakeTts()
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, tts=tts)
    record = json.loads((tmp_path / "episodes" / episode_id / "08_audio.json").read_text())
    assert tts.calls == [[("Lena", "Text."), ("Felix", "Text.")]]  # one section
    assert [s["turn_indices"] for s in record["sections"]] == [[0, 1]]
    assert record["sections"][0]["usd"] == round((100 * 0.5 + 250 * 9.0) / 1e6, 4)
    assert record["loudness"]["output_lufs"] == pytest.approx(-16, abs=0.1)
    assert record["loudness"]["output_true_peak_dbtp"] <= -1.5
    assert set(record["fixed_parts"]) == {"intro", "outro"}
    folder = tmp_path / "build" / episode_id
    assert (folder / "episode.flac").exists() and (folder / "sections" / "00.flac").exists()
    assert record["file"] == "episode.flac"
    assert list(record) == [
        "model", "voices", "section_max_chars", "stt_model", "section_retries", "pronunciation",
            "sections",
        "fixed_parts", "joins", "loudness", "file", "sample_rate", "seconds", "usd",
        "previous_attempts",
    ]  # fmt: skip
    assert record["stt_model"] == "large-v3" and record["section_retries"] == 2
    section = record["sections"][0]
    assert section["verified"] is True
    assert list(section["attempts"][0]) == [
        "attempt", "turn_indices", "seconds", "input_tokens", "output_tokens", "usd", "transcript",
        "deviations", "changes_statement", "action",
    ]  # fmt: skip
    assert section["attempts"][0]["transcript"] == "Text. Text."


def test_missing_intro_and_outro_abort_before_any_api_call(tmp_path):
    llm = FakeLlm()
    with pytest.raises(FixedAudioError):
        run(FakeApi("Rotfuchs", []), llm, tmp_path, CONFIG, "Rotfuchs", NOW, fixed=False)
    assert llm.calls == [] and not (tmp_path / "episodes").exists()


def test_tts_failure_fails_the_run_and_archives_the_sections(tmp_path):
    tts = FakeTts(error=TtsError("TTS answered 401: no key"))
    with pytest.raises(GateFailure, match="TTS section 0"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, tts=tts)
    folder = tmp_path / "episodes" / "2026-11-19_rotfuchs"
    episode = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert episode.status == "failed" and episode.failed_step == "audio"
    record = json.loads((folder / "08_audio.json").read_text())
    assert record["sections"][0]["seconds"] is None and record["file"] is None


def test_audio_that_keeps_changing_the_statement_escalates(tmp_path):
    tts = FakeTts()
    with pytest.raises(GateFailure, match="STT verification: section 0 still changes"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, tts=tts,
            judge=FakeJudgeLlm(), stt=FakeStt("Kein Text."))  # fmt: skip
    assert len(tts.calls) == 3  # first synthesis and two resyntheses
    episode = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert episode.status == "failed" and "STT verification" in episode.failure
    assert episode.failed_step == "audio"
    record = json.loads((tmp_path / "episodes" / episode.episode_id / "08_audio.json").read_text())
    attempts = record["sections"][0]["attempts"]
    assert [a["action"] for a in attempts] == ["resynthesize", "resynthesize", "escalate"]
    assert all(a["transcript"] == "Kein Text." for a in attempts)
    assert attempts[0]["deviations"][0]["reason"] == NO_ANSWER


def test_clean_script_is_archived_and_not_explicit(tmp_path):
    judge = FakeJudgeLlm()
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
                     judge=judge)  # fmt: skip
    assert judge.scripts == ["Skript:\n0. Lena: Text.\n1. Felix: Text."]
    record = json.loads((tmp_path / "episodes" / episode_id / "07_explicit.json").read_text())
    assert record == {
        "model": "claude-sonnet-5-5",
        "explicit": False,
        "verdict": "clean",
        "findings": [],
        "reason": "Nichts.",
        "error": None,
    }
    episode = load_episode(tmp_path, episode_id)
    assert episode.explicit is False and episode.status == "completed"
    assert episode.failed_step is None and episode.resumed == []


@pytest.mark.parametrize(
    "answer",
    [
        {"findings": [], "verdict": "uncertain", "reason": "Unklar."},
        {
            "findings": [
                {"turn_index": 0, "category": "drugs", "quote": "Text.", "reason": "Wein."}
            ],
            "verdict": "explicit",
            "reason": "Alkohol.",
        },
    ],
)
def test_uncertain_or_found_flags_the_episode_explicit_without_blocking_it(tmp_path, answer):
    judge = FakeJudgeLlm(explicit=answer)
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
                     judge=judge)  # fmt: skip
    episode = load_episode(tmp_path, episode_id)
    assert episode.explicit is True and episode.status == "completed" and episode.failure is None
    record = json.loads((tmp_path / "episodes" / episode_id / "07_explicit.json").read_text())
    assert record["explicit"] is True and record["verdict"] == answer["verdict"]


def test_failed_explicit_check_fails_the_run_before_any_tts_cost(tmp_path):
    judge = FakeJudgeLlm(error=LlmError("The model declined the request."))
    tts = FakeTts()
    with pytest.raises(GateFailure, match="Explicit check"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, judge=judge,
            tts=tts)  # fmt: skip
    assert tts.calls == []
    episode = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert episode.status == "failed" and episode.explicit is None
    assert "declined" in episode.failure and episode.failed_step == "explicit"
    folder = tmp_path / "episodes" / episode.episode_id
    record = json.loads((folder / "07_explicit.json").read_text())
    assert record["explicit"] is None and record["error"].startswith("LlmError")
    assert not (folder / "08_audio.json").exists()


def test_explicit_check_runs_before_the_audio_step(tmp_path):
    judge = FakeJudgeLlm()
    tts = FakeTts(error=TtsError("TTS answered 401: no key"))
    with pytest.raises(GateFailure, match="TTS section 0"):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, tts=tts,
            judge=judge)  # fmt: skip
    assert judge.scripts == ["Skript:\n0. Lena: Text.\n1. Felix: Text."]
    folder = tmp_path / "episodes" / "2026-11-19_rotfuchs"
    assert json.loads((folder / "07_explicit.json").read_text())["explicit"] is False
    assert load_episode(tmp_path, "2026-11-19_rotfuchs").explicit is False
    assert (folder / "08_audio.json").exists()


def test_review_sample_is_drawn_with_the_rate_in_the_state_file(tmp_path):
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    review = json.loads((tmp_path / "episodes" / episode_id / "09_review.json").read_text())
    assert review["rate"] == 1.0 and review["seed"] == episode_id
    assert review["population"] == [0, 1] and review["status"] == "pending"
    assert [t["turn_index"] for t in review["turns"]] == [0, 1]
    assert review["turns"][0]["facts"][0]["id"] == "5-0000"
    assert review["issue"] is None
    assert review["explicit"]["predicted"] is False and review["explicit"]["verdict"] == "clean"
    assert review["explicit"]["label"] is None


def test_review_sample_uses_r_from_the_state_file(tmp_path):
    state = tmp_path / "state" / "review.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"r": 0.0, "updates": []}))
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    review = json.loads((tmp_path / "episodes" / episode_id / "09_review.json").read_text())
    assert review["rate"] == 0.0 and review["status"] == "pending" and review["turns"] == []


def test_failed_run_draws_no_review_sample(tmp_path):
    judge = FakeJudgeLlm(error=LlmError("The model declined the request."))
    with pytest.raises(GateFailure):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, judge=judge)
    assert not (tmp_path / "episodes" / "2026-11-19_rotfuchs" / "09_review.json").exists()


def test_step_output_is_appended_when_running_in_actions(tmp_path, monkeypatch):
    from gut_zu_wissen.run import write_step_output

    path = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(path))
    write_step_output("episode_id", "2026-11-19_rotfuchs")
    assert path.read_text() == "episode_id=2026-11-19_rotfuchs\n"


LATER = datetime(2026, 11, 19, 10, 47, 12, tzinfo=UTC)


def test_successful_run_records_end_time_and_token_counts(tmp_path):
    usage = {"author": Usage("claude-sonnet-5-5"), "jev": Usage("jev-1.14.0")}
    usage["author"].add(1000, 200)
    episode_id = run(
        FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
        clock=lambda: LATER, usage=usage,
    )  # fmt: skip
    record = load_episode(tmp_path, episode_id)
    assert record.started_at == "2026-11-19T10:30:00Z"
    assert record.finished_at == "2026-11-19T10:47:12Z"
    assert record.usage == {
        "author": {
            "model": "claude-sonnet-5-5",
            "calls": 1,
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "jev": {
            "model": "jev-1.14.0",
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    prices = REAL["prices"]
    assert record.cost["prices"] == {
        k: prices[k]
        for k in (
            "claude_sonnet_input",
            "claude_sonnet_output",
            "jev",
            "gemini_tts_input",
            "gemini_tts_output",
            "claude_sonnet_cache_write",
            "claude_sonnet_cache_read",
        )
    }
    audio = json.loads((tmp_path / "episodes" / episode_id / "08_audio.json").read_text())
    assert record.cost["usd"] == {
        "author": round(
            (1000 * prices["claude_sonnet_input"] + 200 * prices["claude_sonnet_output"]) / 1e6, 4
        ),
        "jev": 0.0,
        "tts": round(sum(s["usd"] for s in audio["sections"]), 4),
    }
    assert record.status == "completed" and record.failure is None


def test_failed_run_records_end_time(tmp_path):
    with pytest.raises(GateFailure):
        run(
            FakeApi("Rotfuchs", ["Kategorie:Mann"]), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
            clock=lambda: LATER,
        )  # fmt: skip
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert record.status == "failed" and record.finished_at == "2026-11-19T10:47:12Z"


class CrashingLlm(FakeLlm):
    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        raise RuntimeError("connection reset")


def test_crashed_run_is_recorded_as_failed_and_re_raised(tmp_path):
    with pytest.raises(RuntimeError, match="connection reset"):
        run(
            FakeApi("Rotfuchs", []), CrashingLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
            clock=lambda: LATER,
        )  # fmt: skip
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert record.status == "failed" and record.finished_at == "2026-11-19T10:47:12Z"
    assert record.failure == "Crash: RuntimeError: connection reset"
    assert record.failed_step == "extraction"


def test_crash_in_the_audio_step_keeps_the_audio_record(tmp_path):
    tts = FakeTts(error=RuntimeError("socket closed"))
    with pytest.raises(RuntimeError):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW, tts=tts)
    folder = tmp_path / "episodes" / "2026-11-19_rotfuchs"
    episode = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert episode.failure == "Crash: RuntimeError: socket closed"
    assert episode.failed_step == "audio"
    assert (folder / "08_audio.json").exists()
    assert not (folder / "09_review.json").exists()


def test_crash_reason_masks_secrets_and_is_cut():
    from gut_zu_wissen.run import crash_reason

    exc = ValueError("bad key sk-123\n in   header")
    assert crash_reason(exc, 300, ["sk-123", ""]) == "Crash: ValueError: bad key *** in header"
    assert crash_reason(KeyError(), 300) == "Crash: KeyError"
    cut = crash_reason(RuntimeError("x" * 50), 20)
    assert len(cut) == 20 and cut == "Crash: RuntimeError…"


def test_secret_values_come_from_the_named_variables_only():
    from gut_zu_wissen.run import secret_values

    env = {"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "", "HOME": "/root"}
    assert secret_values(env) == ["a"]


def test_main_escalates_a_crash_in_the_job_summary_and_re_raises(tmp_path, monkeypatch):
    import gut_zu_wissen.run as run_module

    def crash(*args, **kwargs):
        raise RuntimeError("upstream said: key sk-live-1 rejected")

    for name, value in [
        ("ANTHROPIC_API_KEY", "sk-live-1"),
        ("TYPESAFE_API_KEY", "ts-1"),
        ("GEMINI_API_KEY", "gm-1"),
        ("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md")),
    ]:
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(run_module, "run", crash)
    with pytest.raises(RuntimeError):
        run_module.main(["Rotfuchs"])
    summary = (tmp_path / "summary.md").read_text()
    assert summary.startswith("## Episode crashed\n\nCrash: RuntimeError: upstream said: key ***")
    assert "sk-live-1" not in summary


def test_completed_run_does_not_count_as_published_for_the_repeat_gate(tmp_path):
    first = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    assert load_episode(tmp_path, first).status == "completed"
    second = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    assert second == "2026-11-19_rotfuchs-2"


def test_failed_run_records_the_cost_so_far(tmp_path):
    with pytest.raises(GateFailure):
        run(FakeApi("Rotfuchs", ["Kategorie:Mann"]), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW)
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    assert record.cost["usd"] == {"tts": 0}  # no client usage passed, no synthesis


def test_crash_in_the_audio_step_records_the_tts_cost(tmp_path):
    with pytest.raises(RuntimeError):
        run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, CONFIG, "Rotfuchs", NOW,
            stt=CrashingStt())  # fmt: skip
    record = load_episode(tmp_path, "2026-11-19_rotfuchs")
    audio = json.loads((tmp_path / "episodes" / record.episode_id / "08_audio.json").read_text())
    (section,) = audio["sections"]
    assert section["usd"] is not None and record.cost["usd"]["tts"] == section["usd"]


class CrashingStt:
    def transcribe(self, y, sr):
        raise RuntimeError("whisper died")


def _lexicon_run(tmp_path, config):
    tts = FakeTts()
    episode_id = run(FakeApi("Rotfuchs", []), FakeLlm(), tmp_path, config, "Rotfuchs", NOW, tts=tts)
    record = json.loads((tmp_path / "episodes" / episode_id / "08_audio.json").read_text())
    return tts, record


def test_config_lexicon_respells_for_the_tts_and_is_archived(tmp_path):
    entry = {"written": "Text", "spoken": "Tekst", "heard": ["Tekst"]}
    tts, record = _lexicon_run(tmp_path, {**CONFIG, "pronunciation": [entry]})
    assert tts.calls == [[("Lena", "Tekst."), ("Felix", "Tekst.")]]  # one section, no resynthesis
    assert record["pronunciation"] == [entry]


def test_episode_lexicon_is_merged_with_the_config_and_wins(tmp_path, monkeypatch):
    config_entries = [
        {"written": "Text", "spoken": "Config", "heard": ["Config"]},
        {"written": "Fuchs", "spoken": "Fux"},
    ]
    episode = [{"written": "Text", "spoken": "Tekst", "heard": ["Tekst"]}]
    monkeypatch.setattr("gut_zu_wissen.run.load_episode_entries", lambda folder: episode)
    tts, record = _lexicon_run(tmp_path, {**CONFIG, "pronunciation": config_entries})
    assert tts.calls == [[("Lena", "Tekst."), ("Felix", "Tekst.")]]
    assert record["pronunciation"] == [*episode, config_entries[1]]
