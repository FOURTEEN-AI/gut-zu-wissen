"""Spike S7: Kosten (user decision 2026-11-20). Measurement only; production code unchanged.

Five cost levers, each measured against archived pipeline output:

1. lower effort (`output_config.effort`) for extraction, ranking and the LLM judge;
2. prompt caching (replay of real call inputs of `2026-11-20_bernstein-2`);
3. Claude Haiku 5.5 for extraction and ranking (comparison only);
4. short reasons on acceptance (LLM-judge prompt variant);
5. a higher Jev share with an extra Jev question per statement (spikes/s7_questions.py).

Requests are built by the production functions (`extract_facts`, `rank_facts`,
`rank_linked_facts`, `format_judge_input`) in two passes: `Collect` records each request the
function makes, the requests go to the Message Batches API (half price, same model behaviour),
and `Replay` hands the answers back in the same order. Extraction, ranking and judge calls do not
depend on each other's answers, so the replayed run is the run the pipeline would make. Each batch
request has the parameters `AnthropicLlm.complete_json` sends (model, max_tokens, system, one user
message, `output_config.format`), plus `output_config.effort` when the variant sets one. No effort
= the model default (`high` on Claude Sonnet 5.5, `medium` on Claude Haiku 5.5), as in production.

The caching replay (`cache`) cannot run as a batch: it sends real call inputs one after the other
with `cache_control` breakpoints and `max_tokens` 1 (caching changes only the input side), and
records `cache_read_input_tokens` and `cache_creation_input_tokens`.

Usage (keys from .env: `set -a; . ./.env; set +a`):
  uv run --group spikes python spikes/s7_kosten.py estimate            # token counts, no cost
  uv run --group spikes python spikes/s7_kosten.py extract NAME        # NAME from EXTRACT_RUNS
  uv run --group spikes python spikes/s7_kosten.py rank NAME           # NAME from RANK_RUNS
  uv run --group spikes python spikes/s7_kosten.py judge NAME          # NAME from JUDGE_RUNS
  uv run --group spikes python spikes/s7_kosten.py cache               # caching replay
  uv run --group spikes python spikes/s7_kosten.py jev                 # lever 5, live, 2 repeats
  uv run --group spikes python spikes/s7_kosten.py batch_extract       # follow-up: batch latency
  uv run --group spikes python spikes/s7_kosten.py report              # tables, no API calls

Results: tests/fixtures/wikipedia/spikes/s7_results/ (answers contain Wikipedia text, CC BY-SA
4.0, see NOTICE). `spend` lists the cost of every saved result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "archive" / "episodes"
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes"
RESULTS = FIXTURES / "s7_results"
JUDGE_CASES = FIXTURES / "s1_sentences_cases.json"  # the 337 judge cases of spike S1
S1_JEV_RESULTS = FIXTURES / "s1_sentences_results"  # recorded production Jev answers
RENDER_CACHE = FIXTURES / "s7_cache"  # Wikipedia rendering per article, not committed

SONNET = "claude-sonnet-5-5"  # config [models] author and llm_judge
HAIKU = "claude-haiku-5-5"
JEV_MODEL = "jev-1.14.0"  # config [models] jev
T = 0.95  # config [judge] t
MAX_TOKENS = 16000  # config [quotas] llm_max_tokens (extraction, ranking, judge)

# USD per million tokens (claude-api skill, models table and prompt-caching economics, cached
# 2026-11-03): input, output, cache read, 5-minute cache write (1.25 x input).
PRICES: dict[str, tuple[float, float, float, float]] = {
    SONNET: (2.00, 10.00, 0.10, 2.50),
    HAIKU: (0.10, 0.50, 0.01, 0.125),
}
BATCH_DISCOUNT = 0.5  # Message Batches: half of every token price
JEV_PRICE = 0.042  # USD per million input tokens, output free (config [prices] jev)

ARTICLES = {
    "tunguska": "2026-11-20_tunguska-ereignis",
    "voynich": "2026-11-20_voynich-manuskript",
    "helgoland": "2026-11-20_helgoland",
}

# name: (article, model, effort); effort None = model default (production for Sonnet).
EXTRACT_RUNS: dict[str, tuple[str, str, str | None]] = {
    f"{a}_{m}_{e or 'default'}": (a, model, e)
    for a in ARTICLES
    for m, model, efforts in (
        ("sonnet", SONNET, (None, "medium", "low")),
        ("haiku", HAIKU, (None, "low")),
    )
    for e in efforts
}
# Ranking: on the archived candidates; Sonnet default twice for the run-to-run noise.
RANK_RUNS: dict[str, tuple[str, str, str | None]] = {
    f"{a}_{m}_{e or 'default'}{suffix}": (a, model, e)
    for a in ARTICLES
    for m, model, efforts in (
        ("sonnet", SONNET, (None, "medium", "low")),
        ("haiku", HAIKU, (None, "low")),
    )
    for e in efforts
    for suffix in (("", "_rep2") if (m, e) == ("sonnet", None) else ("",))
}
# name: (prompt variant, effort, case groups)
JUDGE_RUNS: dict[str, tuple[str, str | None, tuple[str, ...]]] = {
    "verdict_default_critical": ("verdict", None, ("rejected", "s1")),
    "verdict_medium": ("verdict", "medium", ("final", "rejected", "s1")),
    "verdict_low": ("verdict", "low", ("final", "rejected", "s1")),
    "verdict_default_final": ("verdict", None, ("final",)),
    "short_default": ("short", None, ("final", "rejected", "s1")),
    # Follow-up 2026-11-20 (user decision): production judge input with the article title of
    # every cited fact and the article rule in the prompt; the 46 rejected + 239 final turns.
    "title_default": ("verdict", None, ("final", "rejected")),
}
# The two known misses that name the linked article's title (S1, S7 lever 5).
TITLE_CASES = ("2026-11-20_tunguska-ereignis/c1-t21-v0", "2026-11-20_bernstein/c3-t54")

SHORT_FROM = "Begründe kurz auf Deutsch; nenne bei Ablehnung die betroffene Stelle."
SHORT_TO = (
    "Begründe auf Deutsch. Ist „supported“ wahr, genügt eine Zeile mit höchstens zehn Wörtern. "
    "Ist „supported“ falsch, begründe vollständig und nenne die betroffene Stelle."
)

COMPLETENESS_CUTS = (0.95, 0.9, 0.85, 0.8, 0.75)
JEV_REPEATS = 2
JEV_WORKERS = 6


# ---------------------------------------------------------------- pure helpers


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def recall(reference: Iterable[str], found: Iterable[str]) -> float:
    """Share of `reference` that is in `found`; 1.0 for an empty reference."""
    ref = set(reference)
    return len(ref & set(found)) / len(ref) if ref else 1.0


def usage_cost(usage: Mapping, model: str, batch: bool = False) -> float:
    """USD of one response's usage (fields of the API `usage` object)."""
    p_in, p_out, p_read, p_write = PRICES[model]
    usd = (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + (usage.get("cache_read_input_tokens") or 0) * p_read
        + (usage.get("cache_creation_input_tokens") or 0) * p_write
    ) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def batch_params(
    model: str, max_tokens: int, system: str, user: str, schema: dict, effort: str | None
) -> dict:
    """The request `AnthropicLlm.complete_json` sends, with the variant's effort."""
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if effort is not None:
        output_config["effort"] = effort
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": output_config,
    }


def split_for_cache(text: str, prefix_end: int) -> list[dict]:
    """A user message as two text blocks: the stable prefix (cache breakpoint on it) and the
    rest. The concatenation is the original text."""
    assert 0 < prefix_end < len(text)
    return [
        {"type": "text", "text": text[:prefix_end], "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": text[prefix_end:]},
    ]


def short_prompt(verdict_prompt: str) -> str:
    """Lever 4: the production verdict prompt with a one-line reason for acceptances."""
    assert SHORT_FROM in verdict_prompt
    return verdict_prompt.replace(SHORT_FROM, SHORT_TO)


def rule_margin(
    statements: Sequence[float], entities: Sequence[float], completeness: float, t: float, c: float
) -> float:
    """Lever 5: how far a turn is from acceptance by Jev alone under the rule "every statement
    and entity p >= t and completeness p >= c". >= 0 means accepted; the value is the smallest
    distance of any p above its threshold (negative: below)."""
    gaps = [p - t for p in [*statements, *entities]] + [completeness - c]
    return min(gaps)


def rule_decision(
    statements: Sequence[float], entities: Sequence[float], completeness: float, t: float, c: float
) -> str:
    """Lever 5 turn decision: "supported" by Jev alone under the rule; "rejected" as in
    production (any statement or completeness p <= 1 - t; entity questions never reject);
    otherwise "uncertain" (LLM judge)."""
    if any(p <= 1 - t for p in [*statements, completeness]):
        return "rejected"
    if rule_margin(statements, entities, completeness, t, c) >= 0:
        return "supported"
    return "uncertain"


# Words that start a German sentence in capitals without naming anything (lever 5, `terms`).
FUNCTION_WORDS = frozenset(
    """
    aber alle als also am an auch auf aus bei beide beim bis da dabei damals damit dann darauf das
    dass dazu deren einige erst manche
    dem den denn der des deshalb dessen die dies diese diesem diesen dieser dieses doch dort du
    durch ein eine einem einen einer eines er es etwa für gegen heute hier ich ihm ihn ihr ihre
    im in ist jahr jahre jahren jedoch jetzt kein keine man mit nach nicht noch nur ob oder ohne
    schon sein seine seinem seinen seiner sich sie sind so später um und unter viele von vor
    war waren was weil wenn wer wie wir wird wo zu zum zur zwischen über
    """.split()
)
WORD = re.compile(r"[0-9A-Za-zÄÖÜäöüß][0-9A-Za-zÄÖÜäöüß.,\-]*")


def _norm_number(s: str) -> str:
    return re.sub(r"[.,\s]", "", s)


def term_found(term: str, evidence: str) -> bool:
    """Exact lookup in code: the term (a capitalised word or a number) occurs in `evidence`.
    Words match on their stem (inflection: „Kometen“ ~ „Komet“), numbers digit by digit."""
    if any(ch.isdigit() for ch in term):
        return _norm_number(term) in _norm_number(evidence)
    word = term.lower()
    stem = word[: max(4, len(word) - 3)] if len(word) > 4 else word
    return stem in evidence.lower()


def candidate_terms(text: str) -> list[str]:
    """Names, places, things and numbers of a statement: capitalised words (German nouns and
    names) and numbers, function words left out, first occurrence order."""
    terms: list[str] = []
    for raw in WORD.findall(text):
        term = raw.strip(".,-")
        if not term or term.lower() in FUNCTION_WORDS:
            continue
        if (term[0].isupper() or any(ch.isdigit() for ch in term)) and term not in terms:
            terms.append(term)
    return terms


def missing_terms(text: str, evidence: str) -> list[str]:
    """Terms of the statement that code does not find in the evidence; only these go to Jev."""
    return [t for t in candidate_terms(text) if not term_found(t, evidence)]


# ---------------------------------------------------------------- LLM fakes for two passes


class Collect:
    """`StructuredLlm` that records every request and answers with a fixed placeholder."""

    def __init__(self, placeholder: dict) -> None:
        self.placeholder = placeholder
        self.requests: list[tuple[str, str, dict]] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        self.requests.append((system, user, schema))
        return json.loads(json.dumps(self.placeholder))


class Replay:
    """`StructuredLlm` that hands back saved answers in request order and checks that each
    request is the one the answer belongs to."""

    def __init__(self, answers: Sequence[tuple[str, dict]]) -> None:
        self._answers = list(answers)
        self._n = 0

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        digest, answer = self._answers[self._n]
        assert digest == request_digest(system, user), f"request {self._n} differs"
        self._n += 1
        return answer


def request_digest(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- archive access


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _segments(items: Iterable[dict]):
    from gut_zu_wissen.segmentation import Segment

    return [Segment(**s) for s in items]


def _fact(d: dict):
    from gut_zu_wissen.extraction import Fact

    return Fact(**d)


def article_sources(article: str) -> list[dict]:
    """Main article and every linked intro of the run: title, revid, segments, archived
    extraction."""
    folder = EPISODES / ARTICLES[article]
    main = load(folder / "01_article.json")
    out = [
        {
            "source": "main",
            "title": main["title"],
            "revid": main["revid"],
            "segments": load(folder / "02_segments.json")["segments"],
            "extraction": load(folder / "03_facts.json"),
        }
    ]
    for path in sorted((folder / "linked").glob("*.json")):
        linked = load(path)
        if linked.get("extraction") is None:
            continue
        out.append(
            {
                "source": linked["title"],
                "title": linked["title"],
                "revid": linked["revid"],
                "segments": linked["segments"],
                "extraction": linked["extraction"],
            }
        )
    return out


def proposed_ids(extraction: Mapping) -> list[str]:
    """Every segment the model proposed: the offered facts and the gate rejections (unknown IDs
    left out)."""
    ids = [f["segment_id"] for f in extraction["facts"]]
    ids += [r["segment_id"] for r in extraction["rejected"] if r["reason"] != "unknown_segment"]
    return list(dict.fromkeys(ids))


def offered_ids(article: str) -> dict[str, list[str]]:
    """The archived offer: main and linked facts of 04_ranking.json, rank order."""
    ranking = load(EPISODES / ARTICLES[article] / "04_ranking.json")
    return {
        "main": [r["fact"]["segment_id"] for r in ranking["main_facts"]],
        "linked": [r["fact"]["segment_id"] for r in ranking["linked_facts"]],
    }


# ---------------------------------------------------------------- batches


def anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def run_batch(name: str, params: Sequence[dict]) -> list[dict]:
    """Send `params` as one batch (or resume a saved one) and return one result per request in
    order: usage, stop reason and the JSON answer (None on error)."""
    client = anthropic_client()
    pending = RESULTS / "batches" / f"{name}.pending.json"
    pending.parent.mkdir(parents=True, exist_ok=True)
    if pending.exists():
        batch_id = load(pending)["batch_id"]
    else:
        reqs = [{"custom_id": f"r{n:04d}", "params": p} for n, p in enumerate(params)]
        batch_id = client.messages.batches.create(requests=reqs).id
        pending.write_text(json.dumps({"batch_id": batch_id, "requests": len(reqs)}) + "\n")
    print(f"{name}: batch {batch_id}, {len(params)} requests")
    while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        print(f"  {b.request_counts}")
        time.sleep(20)
    by_id: dict[str, dict] = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type != "succeeded":
            by_id[r.custom_id] = {"error": r.result.type, "usage": None, "answer": None}
            continue
        msg = r.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            answer = json.loads(text) if msg.stop_reason == "end_turn" else None
        except json.JSONDecodeError:
            answer = None
        by_id[r.custom_id] = {
            "error": None if answer is not None else f"stop_reason {msg.stop_reason}",
            "usage": msg.usage.model_dump(exclude_none=True),
            "stop_reason": msg.stop_reason,
            "answer": answer,
        }
    pending.unlink()
    return [by_id[f"r{n:04d}"] for n in range(len(params))]


def save(kind: str, name: str, body: dict) -> Path:
    out = RESULTS / kind / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n")
    return out


def usage_summary(results: Sequence[dict], model: str, batch: bool = True) -> dict:
    ok = [r for r in results if r["usage"]]
    return {
        "calls": len(results),
        "errors": sum(r["error"] is not None for r in results),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in ok),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in ok),
        "thinking_tokens": sum(
            (r["usage"].get("output_tokens_details") or {}).get("thinking_tokens", 0) for r in ok
        ),
        "usd_paid": round(sum(usage_cost(r["usage"], model, batch) for r in ok), 4),
        "usd_list": round(sum(usage_cost(r["usage"], model) for r in ok), 4),
    }


# ---------------------------------------------------------------- renderer with disk cache


class CachedRenderer:
    """Production `ParseRenderer` (Wikipedia `action=parse`, descriptive User-Agent) with a
    local cache, so the variants of one article render each sentence once."""

    def __init__(self, article: str) -> None:
        from gut_zu_wissen.fetch import HttpWikipediaApi
        from gut_zu_wissen.render import ParseRenderer

        self._inner = ParseRenderer(HttpWikipediaApi())
        self._path = RENDER_CACHE / f"{article}.json"
        self._cache: dict[str, str] = load(self._path) if self._path.exists() else {}

    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        keys = [f"{title}\x00{s}" for s in snippets]
        todo = [s for k, s in zip(keys, snippets, strict=True) if k not in self._cache]
        if todo:
            todo = list(dict.fromkeys(todo))
            for s, html in zip(todo, self._inner.render_html(title, todo), strict=True):
                self._cache[f"{title}\x00{s}"] = html
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache, ensure_ascii=False))
            time.sleep(1)  # Wikipedia API: be gentle
        return [self._cache[k] for k in keys]


def config() -> dict:
    import tomllib

    return tomllib.loads((ROOT / "config" / "pipeline.toml").read_text())


# ---------------------------------------------------------------- lever 1/3: extraction


def extract(name: str) -> None:
    from gut_zu_wissen.extraction import EXTRACTION_SCHEMA, extract_facts

    article, model, effort = EXTRACT_RUNS[name]
    cfg = config()
    sources = article_sources(article)
    collected: list[tuple[dict, Collect]] = []
    for src in sources:
        fake = Collect({"candidates": []})
        extract_facts(fake, None, src["title"], src["revid"], _segments(src["segments"]), cfg)
        collected.append((src, fake))
    params = [
        batch_params(model, MAX_TOKENS, s, u, EXTRACTION_SCHEMA, effort)
        for _, fake in collected
        for s, u, _ in fake.requests
    ]
    results = run_batch(f"extract_{name}", params)
    renderer = CachedRenderer(article)
    out_sources, n = [], 0
    for src, fake in collected:
        mine = results[n : n + len(fake.requests)]
        n += len(fake.requests)
        answers = [
            (request_digest(s, u), r["answer"] or {"candidates": []})
            for (s, u, _), r in zip(fake.requests, mine, strict=True)
        ]
        res = extract_facts(
            Replay(answers),
            renderer,
            src["title"],
            src["revid"],
            _segments(src["segments"]),
            cfg,
        )
        out_sources.append(
            {
                "source": src["source"],
                "calls": len(mine),
                "errors": sum(r["error"] is not None for r in mine),
                "proposed": proposed_ids(res.to_dict()),
                "facts": [f.segment_id for f in res.facts],
                "extraction": res.to_dict(),
                "usage": [r["usage"] for r in mine],
            }
        )
    body = {
        "name": name,
        "article": article,
        "model": model,
        "effort": effort,
        "summary": usage_summary(results, model),
        "sources": out_sources,
    }
    print(save("extract", name, body), body["summary"])


# ---------------------------------------------------------------- lever 1/3: ranking


def ranking_requests(article: str, extraction: Mapping[str, Mapping] | None = None):
    """The two ranking requests of the archived (final) offer, built by the production functions
    on the archived candidates, or on `extraction` (source -> extraction dict) when given."""
    from gut_zu_wissen.ranking import offer_size, rank_facts, rank_linked_facts

    folder = EPISODES / ARTICLES[article]
    q = config()["quotas"]
    ranking = load(folder / "04_ranking.json")
    title = load(folder / "01_article.json")["title"]
    sources = {s["source"]: s for s in article_sources(article)}
    ext = extraction or {k: s["extraction"] for k, s in sources.items()}
    exclude = set(ranking.get("excluded_fact_ids", []))
    exclude |= {r["fact"]["segment_id"] for r in ranking.get("einordnung_facts", [])}
    main_facts = [_fact(f) for f in ext["main"]["facts"]]
    linked_titles = [a["title"] for a in ranking["linked_articles"]]
    pool, origin = [], {}
    for t in linked_titles:
        for f in ext[t]["facts"] if t in ext else []:
            pool.append(_fact(f))
            origin[f["segment_id"]] = t
    main_fake, linked_fake = Collect({"ranked_ids": []}), Collect({"ranked_ids": []})
    main_max = offer_size(len(main_facts), q["main_offer_share"], q["main_facts_min"])
    rank_facts(main_fake, main_facts, q["main_facts_min"], main_max, exclude)
    rank_linked_facts(
        linked_fake,
        pool,
        origin,
        title,
        q["linked_intro_facts_min"],
        q["linked_intro_facts_max"],
        set(ranking.get("excluded_fact_ids", [])),
    )
    return {
        "main": (main_fake.requests, main_facts, main_max),
        "linked": (linked_fake.requests, pool, q["linked_intro_facts_max"]),
    }


def rank(name: str) -> None:
    from gut_zu_wissen.ranking import RANKING_SCHEMA, select_ranked

    article, model, effort = RANK_RUNS[name]
    reqs = ranking_requests(article)
    keys = [k for k, (r, _, _) in reqs.items() if r]
    params = [
        batch_params(model, MAX_TOKENS, *reqs[k][0][0][:2], RANKING_SCHEMA, effort) for k in keys
    ]
    results = run_batch(f"rank_{name}", params)
    ranked = {}
    for k, r in zip(keys, results, strict=True):
        _, candidates, maximum = reqs[k]
        ids = (r["answer"] or {}).get("ranked_ids", [])
        ranked[k] = [f.segment_id for f in select_ranked(ids, candidates, maximum)]
    body = {
        "name": name,
        "article": article,
        "model": model,
        "effort": effort,
        "summary": usage_summary(results, model),
        "ranked": ranked,
        "usage": {k: r["usage"] for k, r in zip(keys, results, strict=True)},
    }
    print(save("rank", name, body), body["summary"])


def rank_chain(extract_name: str, ranker: str = "same") -> None:
    """Ranking on the candidates of a saved extraction run: the offer the cheaper variant would
    produce end to end (linked articles as archived). `ranker` "same" ranks with the model and
    effort of the extraction run, "sonnet" with Claude Sonnet 5.5 at its default (production)."""
    from gut_zu_wissen.ranking import RANKING_SCHEMA, select_ranked

    saved = load(RESULTS / "extract" / f"{extract_name}.json")
    model, effort = saved["model"], saved["effort"]
    if ranker == "sonnet":
        model, effort = SONNET, None
    tag = f"chain_{extract_name}" + ("" if ranker == "same" else f"_by_{ranker}")
    ext = {s["source"]: s["extraction"] for s in saved["sources"]}
    reqs = ranking_requests(saved["article"], ext)
    keys = [k for k, (r, _, _) in reqs.items() if r]
    params = [
        batch_params(model, MAX_TOKENS, *reqs[k][0][0][:2], RANKING_SCHEMA, effort) for k in keys
    ]
    results = run_batch(tag, params)
    ranked = {}
    for k, r in zip(keys, results, strict=True):
        _, candidates, maximum = reqs[k]
        ids = (r["answer"] or {}).get("ranked_ids", [])
        ranked[k] = [f.segment_id for f in select_ranked(ids, candidates, maximum)]
    body = {
        "name": tag,
        "article": saved["article"],
        "model": model,
        "effort": effort,
        "summary": usage_summary(results, model),
        "ranked": ranked,
        "usage": {k: r["usage"] for k, r in zip(keys, results, strict=True)},
    }
    print(save("rank", tag, body), body["summary"])


# ---------------------------------------------------------------- lever 1/4: LLM judge


def judge_cases() -> list[dict]:
    return load(JUDGE_CASES)["cases"]


def judge_input(case: Mapping) -> str:
    """The LLM judge's input for a case, built by the production `format_judge_input` from the
    case's Jev state (which holds the turn, its statements, the cited facts and context)."""
    from gut_zu_wissen.dialog import Statement, Turn
    from gut_zu_wissen.extraction import Fact
    from gut_zu_wissen.judge import format_judge_input
    from gut_zu_wissen.ranking import RankedFact

    st = case["state"]
    statements = [Statement(a["text"], a["fakten"]) for a in st["aussagen"]]
    turn = Turn(0, st["redebeitrag"]["sprecher"], st["redebeitrag"]["text"], [], statements)
    turn.fact_ids = list(st["fakten"])
    facts = {
        i: RankedFact(
            1,
            f["artikel"],
            0,
            Fact(i, "sentence", "", [], f["satz"], st.get("kontext", {}).get(i, [])),
        )
        for i, f in st["fakten"].items()
    }
    return format_judge_input(turn, facts)


def judge(name: str) -> None:
    from gut_zu_wissen.judge import VERDICT_PROMPT, VERDICT_SCHEMA
    from gut_zu_wissen.llm import request_params

    variant, effort, groups = JUDGE_RUNS[name]
    system = VERDICT_PROMPT if variant == "verdict" else short_prompt(VERDICT_PROMPT)
    cases = [c for c in judge_cases() if c["group"] in groups]
    if name == "title_default":  # the production request builder, cache marker included
        params = [
            request_params(
                SONNET, MAX_TOKENS, system, judge_input(c), VERDICT_SCHEMA, cache_system=True
            )
            for c in cases
        ]
    else:
        params = [
            batch_params(SONNET, MAX_TOKENS, system, judge_input(c), VERDICT_SCHEMA, effort)
            for c in cases
        ]
    results = run_batch(f"judge_{name}", params)
    rows = [
        {
            "id": c["id"],
            "group": c["group"],
            "label": c["accepted"],
            "supported": (r["answer"] or {}).get("supported"),
            "reason": (r["answer"] or {}).get("reason"),
            "error": r["error"],
            "usage": r["usage"],
        }
        for c, r in zip(cases, results, strict=True)
    ]
    body = {
        "name": name,
        "variant": variant,
        "effort": effort,
        "system": system,
        "summary": usage_summary(results, SONNET),
        "cases": rows,
    }
    print(save("judge", name, body), body["summary"])


# ---------------------------------------------------------------- lever 2: caching replay


def cache_calls() -> list[dict]:
    """Real call inputs of `2026-11-20_bernstein-2`, in pipeline order, each with where its
    stable prefix ends (`prefix_end`: in the user text; 0: only the system prompt is cached).

    - extraction: the first 20 chunks of the main article (same system prompt);
    - author: every regeneration of the final conversation, with the final offer; the turn
      context is rebuilt from the final dialog with the replaced version put back (the archive
      does not keep the dialog as it stood, so the tail is approximate; the prefix is exact);
    - judge: the verdict calls of the final turns of the episode (judge cases of spike S1).
    """
    from gut_zu_wissen.dialog import TURN_PROMPT, Statement, Turn, format_turn_context
    from gut_zu_wissen.extraction import SYSTEM_PROMPT, format_chunk, make_chunks
    from gut_zu_wissen.judge import VERDICT_PROMPT, VERDICT_SCHEMA
    from gut_zu_wissen.orchestrator import Offer
    from gut_zu_wissen.ranking import RankedFact

    folder = EPISODES / "2026-11-20_bernstein-2"
    cfg = config()
    q = cfg["quotas"]
    calls: list[dict] = []
    segments = _segments(load(folder / "02_segments.json")["segments"])
    for chunk in make_chunks(segments, q["max_chunk_segments"])[:20]:
        from gut_zu_wissen.extraction import EXTRACTION_SCHEMA

        calls.append(
            {
                "kind": "extraction",
                "system": SYSTEM_PROMPT.format(max_context=q["max_context_segments"]),
                "user": format_chunk(chunk),
                "prefix_end": 0,
                "schema": EXTRACTION_SCHEMA,
            }
        )
    ranking = load(folder / "04_ranking.json")

    def rf(r: dict) -> RankedFact:
        return RankedFact(r["rank"], r["article"], r["revid"], _fact(r["fact"]))

    offer = Offer(
        [rf(r) for r in ranking["main_facts"]],
        [rf(r) for r in ranking["linked_facts"]],
        [rf(r) for r in ranking.get("einordnung_facts", [])],
    )
    dialog = load(folder / "05_dialog.json")

    def turn(d: dict) -> Turn:
        st = [Statement(s["text"], s["fact_ids"]) for s in d.get("statements", [])]
        return Turn(d["index"], d["speaker"], d["text"], d["fact_ids"], st)

    from gut_zu_wissen.dialog import TURN_SCHEMA

    final = [turn(t) for t in dialog["turns"]]
    aliases = offer.aliases()
    for reg in dialog["regenerations"]:
        if reg["conversation"] != dialog["conversation"]:
            continue
        turns = list(final)
        turns[reg["turn_index"]] = turn(reg["previous"])
        context = format_turn_context(turns, reg["turn_index"], reg["reason"], aliases)
        head = f"Angebot:\n{offer.text()}\n\n"
        calls.append(
            {
                "kind": "regeneration",
                "system": TURN_PROMPT,
                "user": head + context,
                "prefix_end": len(head),
                "schema": TURN_SCHEMA,
            }
        )
    for c in judge_cases():
        if c["id"].startswith("2026-11-20_bernstein-2/") and c["group"] == "final":
            calls.append(
                {
                    "kind": "judge",
                    "system": VERDICT_PROMPT,
                    "user": judge_input(c),
                    "prefix_end": 0,
                    "schema": VERDICT_SCHEMA,
                }
            )
    return calls


def cache_request(call: Mapping, cached: bool) -> dict:
    """One replay request: production shape, `max_tokens` 1; with `cached`, a breakpoint on the
    system prompt and, where the user text has a stable prefix, on that prefix."""
    system: str | list[dict] = call["system"]
    user: str | list[dict] = call["user"]
    if cached:
        system = [{"type": "text", "text": call["system"], "cache_control": {"type": "ephemeral"}}]
        if call["prefix_end"]:
            user = split_for_cache(call["user"], call["prefix_end"])
    return {
        "model": SONNET,
        "max_tokens": 1,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": call["schema"]}},
    }


def cache() -> None:
    client = anthropic_client()
    calls = cache_calls()
    rows = []
    for n, call in enumerate(calls):
        r = client.messages.create(**cache_request(call, cached=True))
        u = r.usage.model_dump(exclude_none=True)
        rows.append({"n": n, "kind": call["kind"], "usage": u})
        print(n, call["kind"], u["input_tokens"], u.get("cache_read_input_tokens"),
              u.get("cache_creation_input_tokens"))  # fmt: skip
    body = {
        "episode": "2026-11-20_bernstein-2",
        "note": "Replay with max_tokens 1; output tokens are not affected by caching.",
        "summary": usage_summary(
            [{"usage": r["usage"], "error": None} for r in rows], SONNET, batch=False
        ),
        "calls": rows,
    }
    print(save("cache", "bernstein-2", body), body["summary"])


# ---------------------------------------------------------------- lever 5: Jev


def jev_payload(case: Mapping) -> tuple[dict, dict]:
    """Production state and questions (statement questions + completeness), plus the two new
    question families of spikes/s7_questions.py."""
    from s7_questions import entity_question, term_question

    from gut_zu_wissen.judge import completeness_question

    questions = dict(case["statement_questions"])
    questions["vollstaendig"] = completeness_question()
    st = case["state"]
    for n, a in enumerate(st["aussagen"]):
        ids = [i for i in a["fakten"] if i in st["fakten"]]
        if not ids:
            continue
        questions[f"entities_{n}"] = entity_question(a["text"], ids)
        # The fact sentence only: the LLM judge never sees the article title (format_judge_input),
        # and both known S1 misses name exactly the title of the linked article.
        evidence = " ".join(st["fakten"][i]["satz"] for i in ids)
        for k, term in enumerate(missing_terms(a["text"], evidence)):
            questions[f"term_{n}_{k}"] = term_question(term, a["text"], ids)
    return st, questions


def jev() -> None:
    import httpx
    from s1_completeness import post

    cases = judge_cases()
    key = os.environ["TYPESAFE_API_KEY"]
    with (
        httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client,
        ThreadPoolExecutor(JEV_WORKERS) as pool,
    ):
        for rep in range(JEV_REPEATS):
            out = RESULTS / "jev" / f"{JEV_MODEL}_{rep}.json"
            if out.exists():
                continue

            def one(case: dict) -> dict:
                state, questions = jev_payload(case)
                start = time.perf_counter()
                r = post(client, state, questions)
                assert r["status"] == 200, (case["id"], r["status"], r["response"])
                return {
                    "id": case["id"],
                    "seconds": round(time.perf_counter() - start, 3),
                    "questions": len(questions),
                    "usage": r["response"]["usage"],
                    "answers": {k: a["noul"] for k, a in r["response"]["answers"].items()},
                }

            records = list(pool.map(one, cases))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
            tokens = sum(r["usage"]["input_tokens"] for r in records)
            print(
                f"{out.name}: {len(records)} cases, {tokens} tokens, "
                f"${tokens * JEV_PRICE / 1e6:.4f}"
            )


# ---------------------------------------------------------------- estimate (free)


def estimate() -> None:
    """Input tokens of every planned run by count_tokens (free), with a cost range; output
    tokens are estimated from the archived author/judge averages."""
    from gut_zu_wissen.extraction import EXTRACTION_SCHEMA, extract_facts
    from gut_zu_wissen.judge import VERDICT_PROMPT

    client = anthropic_client()
    cfg = config()

    def count(system: str, user: str) -> int:
        return client.messages.count_tokens(
            model=SONNET, system=system, messages=[{"role": "user", "content": user}]
        ).input_tokens

    for article in ARTICLES:
        reqs = []
        for src in article_sources(article):
            fake = Collect({"candidates": []})
            extract_facts(fake, None, src["title"], src["revid"], _segments(src["segments"]), cfg)
            reqs += fake.requests
        tokens = sum(count(s, u) for s, u, _ in reqs) + 200 * len(reqs)  # + schema overhead
        print(f"extraction {article}: {len(reqs)} calls, {tokens} input tokens "
              f"(Sonnet batch ${tokens * 1e-6:.3f} input)")  # fmt: skip
        rr = ranking_requests(article)
        rt = sum(count(*r[0][0][:2]) for r in rr.values() if r[0])
        print(f"ranking {article}: {rt} input tokens per run")
    sample = judge_cases()[::10]
    mean = statistics.mean(count(VERDICT_PROMPT, judge_input(c)) for c in sample) + 200
    print(f"judge: {mean:.0f} input tokens per case, 337 cases: {337 * mean:.0f}")
    _ = EXTRACTION_SCHEMA


# ---------------------------------------------------------------- report


def _saved(kind: str, name: str) -> dict | None:
    path = RESULTS / kind / f"{name}.json"
    return load(path) if path.exists() else None


def _per_call(summary: Mapping, field: str) -> float:
    return summary[field] / summary["calls"] if summary["calls"] else 0.0


def report_extraction() -> None:
    print("\n## Extraction (lever 1 and 3)\n")
    print(
        "| Run | calls | output tok | thinking tok | list $ | facts | J(default rerun) "
        "| J(archive) | recall of offer | in default rerun | default facts found |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in EXTRACT_RUNS:
        d = _saved("extract", name)
        if d is None:
            continue
        a = d["article"]
        archive = article_sources(a)
        base = _saved("extract", f"{a}_sonnet_default")
        offer = [i for ids in offered_ids(a).values() for i in ids]
        facts = [i for s in d["sources"] for i in s["facts"]]
        afacts = [f["segment_id"] for s in archive for f in s["extraction"]["facts"]]
        bfacts = [i for s in base["sources"] for i in s["facts"]] if base else []
        s = d["summary"]
        print(
            f"| {name} | {s['calls']} | {s['output_tokens']} | {s['thinking_tokens']} "
            f"| {s['usd_list']:.3f} | {len(facts)} | {jaccard(facts, bfacts):.2f} "
            f"| {jaccard(facts, afacts):.2f} | {recall(offer, facts):.2f} "
            f"| {recall(facts, bfacts):.2f} | {recall(bfacts, facts):.2f} |"
        )


def _ranked_ids(d: Mapping, key: str) -> list[str]:
    return d["ranked"].get(key, [])


def report_ranking() -> None:
    print("\n## Ranking on the archived candidates (lever 1 and 3)\n")
    print(
        "| Run | output tok | thinking tok | list $ | main n | J main (archive) "
        "| J main (default) | recall archived main offer | J linked (archive) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    chains = [f"chain_{n}{s}" for n in EXTRACT_RUNS for s in ("", "_by_sonnet")]
    names = [*RANK_RUNS, *chains]
    for name in names:
        d = _saved("rank", name)
        if d is None:
            continue
        a = d["article"]
        arch = offered_ids(a)
        base = _saved("rank", f"{a}_sonnet_default")
        other = _saved("rank", f"{a}_sonnet_default_rep2") if name.endswith("default") else None
        ref = other if name == f"{a}_sonnet_default" else base
        main = _ranked_ids(d, "main")
        s = d["summary"]
        print(
            f"| {name} | {s['output_tokens']} | {s['thinking_tokens']} | {s['usd_list']:.3f} "
            f"| {len(main)} | {jaccard(main, arch['main']):.2f} "
            f"| {jaccard(main, _ranked_ids(ref, 'main')) if ref else float('nan'):.2f} "
            f"| {recall(arch['main'], main):.2f} "
            f"| {jaccard(_ranked_ids(d, 'linked'), arch['linked']):.2f} |"
        )


def report_judge() -> None:
    print("\n## LLM judge (lever 1 and 4)\n")
    print(
        "| Run | final n | final agree | false rejections | rejected n | false acceptances "
        "| S1 FA | S1 FR | output tok/call | thinking tok/call | list $/call |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in JUDGE_RUNS:
        d = _saved("judge", name)
        if d is None:
            continue
        rows = {g: [c for c in d["cases"] if c["group"] == g] for g in ("final", "rejected", "s1")}

        def fa(rs: list[dict]) -> int:
            return sum(c["supported"] is True and not c["label"] for c in rs)

        def fr(rs: list[dict]) -> int:
            return sum(c["supported"] is False and c["label"] for c in rs)

        s = d["summary"]
        final = rows["final"]
        print(
            f"| {name} | {len(final)} | {sum(c['supported'] == c['label'] for c in final)} "
            f"| {fr(final)} | {len(rows['rejected'])} | {fa(rows['rejected'])} "
            f"| {fa(rows['s1'])} | {fr(rows['s1'])} | {_per_call(s, 'output_tokens'):.0f} "
            f"| {_per_call(s, 'thinking_tokens'):.0f} | {s['usd_list'] / s['calls']:.4f} |"
        )
    title = _saved("judge", "title_default")
    if title:
        verdicts = {c["id"]: c["supported"] for c in title["cases"]}
        print("\ntitle_default on the two title cases:", {i: verdicts.get(i) for i in TITLE_CASES})
        reruns = [_saved("judge", n) for n in ("verdict_default_critical", "verdict_default_final")]
        old = {c["id"]: c["supported"] for d in reruns if d for c in d["cases"]}
        changed = sorted(i for i, v in verdicts.items() if i in old and old[i] != v)
        print(f"  verdicts that differ from the S7 default rerun: {len(changed)}", changed)
    base = _saved("judge", "verdict_default_critical")
    if base:
        flips = {c["id"] for c in base["cases"] if c["group"] == "rejected" and c["supported"]}
        print(f"\nDefault rerun accepts {len(flips)} of the rejected turns:", sorted(flips))
        for name in JUDGE_RUNS:
            d = _saved("judge", name)
            if d is None or d is base:
                continue
            mine = {c["id"] for c in d["cases"] if c["group"] == "rejected" and c["supported"]}
            print(
                f"  {name}: {len(mine)} FA, {len(mine & flips)} also in the default rerun, "
                f"new: {sorted(mine - flips)}"
            )


def report_cache() -> None:
    d = _saved("cache", "bernstein-2")
    if d is None:
        return
    print("\n## Caching replay (lever 2, bernstein-2)\n")
    print("| Call kind | calls | input tok/call (all) | cached tok/call | cached share |")
    print("|---|---:|---:|---:|---:|")
    for kind in ("extraction", "regeneration", "judge"):
        rows = [r["usage"] for r in d["calls"] if r["kind"] == kind]
        total = [
            u["input_tokens"]
            + (u.get("cache_read_input_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0)
            for u in rows
        ]
        cached = [u.get("cache_read_input_tokens") or 0 for u in rows[1:]]
        mean_total = statistics.mean(total)
        print(
            f"| {kind} | {len(rows)} | {mean_total:.0f} | {statistics.mean(cached):.0f} "
            f"| {statistics.mean(cached) / mean_total:.0%} |"
        )


def jev_records() -> list[list[dict]]:
    return [load(p) for p in sorted((RESULTS / "jev").glob(f"{JEV_MODEL}_*.json"))]


def jev_ps(answers: Mapping[str, float], family: str):
    stmt = [p for k, p in answers.items() if k.startswith("aussage_")]
    ent = [p for k, p in answers.items() if k.startswith("entities_")]
    term = [p for k, p in answers.items() if k.startswith("term_")]
    extra = {"none": [], "entities": ent, "terms": term, "both": ent + term}[family]
    return stmt, extra, answers["vollstaendig"]


def report_jev() -> None:
    runs = jev_records()
    if not runs:
        return
    cases = {c["id"]: c for c in judge_cases()}
    print("\n## Jev share with a safety net (lever 5)\n")
    print(
        "| Extra question | c | final: Jev alone | rejected: false acceptances (of 46) "
        "| S1 not accepted: FA (of 28) | max margin (rejected + S1) | S1 alone |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for family in ("none", "entities", "terms", "both"):
        for c in COMPLETENESS_CUTS:
            cells = []
            for records in runs:
                alone = n = fa = fa_s1 = s1_alone = s1_n = 0
                margins = []
                for r in records:
                    case = cases[r["id"]]
                    stmt, extra, comp = jev_ps(r["answers"], family)
                    d = rule_decision(stmt, extra, comp, T, c)
                    if case["group"] == "final":
                        n += 1
                        alone += d != "uncertain"
                    if case["group"] == "s1":
                        s1_n += 1
                        s1_alone += d != "uncertain"
                    if not case["accepted"]:
                        margins.append(rule_margin(stmt, extra, comp, T, c))
                        if d == "supported":
                            if case["group"] == "rejected":
                                fa += 1
                            else:
                                fa_s1 += 1
                cells.append((alone / n, fa, fa_s1, max(margins), s1_alone / s1_n))
            col = list(zip(*cells, strict=True))
            print(
                f"| {family} | {c:.2f} | {' / '.join(f'{x:.0%}' for x in col[0])} "
                f"| {' / '.join(map(str, col[1]))} | {' / '.join(map(str, col[2]))} "
                f"| {' / '.join(f'{x:+.2f}' for x in col[3])} "
                f"| {' / '.join(f'{x:.0%}' for x in col[4])} |"
            )
    q = statistics.mean(r["questions"] for r in runs[0])
    tok = statistics.mean(r["usage"]["input_tokens"] for r in runs[0])
    print(f"\nquestions/request {q:.1f}, input tokens/request {tok:.0f}")


EPISODE_COST_SET = {
    "axolotl": "2026-11-20_axolotl",
    "tunguska": "2026-11-20_tunguska-ereignis",
    "voynich": "2026-11-20_voynich-manuskript",
    "bernstein-2": "2026-11-20_bernstein-2",
    "helgoland": "2026-11-20_helgoland",
}


def episode_calls(folder: Path) -> dict[str, int]:
    """Calls per kind of one archived run, counted from its files (pure, reads the archive).

    Rankings and dialog writes: one per conversation that was not a continuation; LLM-judge
    verdict or reason calls: every judgement (final, superseded, discarded) not decided by a gate
    and not accepted by Jev alone; cascade calls: cascade checks the LLM judge decided."""
    dialog = load(folder / "05_dialog.json")
    judgement = load(folder / "06_judgement.json")
    extraction = load(folder / "03_facts.json")["chunk_count"]
    for path in (folder / "linked").glob("*.json"):
        linked = load(path)
        if linked.get("extraction"):
            extraction += linked["extraction"]["chunk_count"]
    discarded = dialog["discarded_conversations"]
    continuations = sum(d.get("regeneration") == "continuation" for d in discarded)
    fresh = 1 + len(discarded) - continuations
    judged = [
        *judgement["judgements"],
        *(s["judgement"] for s in judgement["superseded_judgements"]),
        *(j for d in discarded for j in d["judgements"]),
    ]
    llm = sum(
        j["decided_by"] == "llm_judge" or (j["decided_by"] == "jev" and not j["accepted"])
        for j in judged
    )
    return {
        "extraction": extraction,
        "ranking": 2 * fresh,
        "write": fresh,
        "continuation": continuations,
        "regeneration": len(dialog["regenerations"]),
        "conversations": 1 + len(discarded),
        "judge_llm": llm,
        "judge_jev_alone": sum(j["decided_by"] == "jev" and j["accepted"] for j in judged),
        "cascade_llm": sum(c["decided_by"] == "llm_judge" for c in judgement["cascade_checks"]),
    }


def caching_saving(calls: Mapping[str, int], prefix: Mapping[str, int]) -> dict[str, float]:
    """USD saved per run by prompt caching at Claude Sonnet 5.5 list prices, 5-minute TTL, every
    call within the TTL of the one before it (an upper bound). Each cached read saves
    (input - cache read) per token; each prefix is written once per run (extraction, judge) or
    once per conversation (offer), at (write - input) extra per token."""
    p_in, _, p_read, p_write = PRICES[SONNET]
    save, extra = (p_in - p_read) / 1e6, (p_write - p_in) / 1e6

    def part(n: int, tokens: int, writes: int) -> float:
        return max(n - writes, 0) * tokens * save - writes * tokens * extra if n else 0.0

    return {
        "extraction": part(calls["extraction"], prefix["extraction"], 1),
        "judge": part(calls["judge_llm"], prefix["judge"], 1),
        "regeneration": part(calls["regeneration"], prefix["offer"], calls["conversations"]),
    }


def offers() -> None:
    """Cached prefix of the regeneration call (system + "Angebot:" + final offer) per episode,
    counted with count_tokens (free); saved for `report`."""
    from gut_zu_wissen.dialog import TURN_PROMPT, TURN_SCHEMA
    from gut_zu_wissen.orchestrator import Offer
    from gut_zu_wissen.ranking import RankedFact

    client = anthropic_client()
    out = {}
    for name, folder_name in EPISODE_COST_SET.items():
        ranking = load(EPISODES / folder_name / "04_ranking.json")

        def rf(r: dict) -> RankedFact:
            return RankedFact(r["rank"], r["article"], r["revid"], _fact(r["fact"]))

        offer = Offer(
            [rf(r) for r in ranking["main_facts"]],
            [rf(r) for r in ranking["linked_facts"]],
            [rf(r) for r in ranking.get("einordnung_facts", [])],
        )
        out[name] = client.messages.count_tokens(
            model=SONNET,
            system=TURN_PROMPT,
            messages=[{"role": "user", "content": f"Angebot:\n{offer.text()}\n\n"}],
            output_config={"format": {"type": "json_schema", "schema": TURN_SCHEMA}},
        ).input_tokens
        print(name, out[name])
    save("cache", "offers", {"note": "count_tokens of the cached regeneration prefix", **out})


def report_episodes() -> None:
    cache = _saved("cache", "bernstein-2")
    offer_tokens = _saved("cache", "offers")
    if cache is None or offer_tokens is None:
        return
    reads = {
        k: max(
            (r["usage"].get("cache_read_input_tokens") or 0)
            for r in cache["calls"]
            if r["kind"] == k
        )
        for k in ("extraction", "judge")
    }
    print("\n## Per episode: calls and caching saving (Sonnet list prices)\n")
    print(
        "| Episode | total $ | author $ | judge $ | extraction calls | regenerations | conv. "
        "| LLM judge calls | Jev alone | cache: extraction $ | judge $ | regen. $ | sum $ | % |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, folder_name in EPISODE_COST_SET.items():
        folder = EPISODES / folder_name
        usd = load(folder / "episode.json")["cost"]["usd"]
        total = sum(v for v in usd.values() if v)
        calls = episode_calls(folder)
        prefix = {**reads, "offer": offer_tokens[name]}
        saving = caching_saving(calls, prefix)
        s = sum(saving.values())
        print(
            f"| {name} | {total:.2f} | {usd['author']:.2f} | {usd['llm_judge']:.2f} "
            f"| {calls['extraction']} | {calls['regeneration']} | {calls['conversations']} "
            f"| {calls['judge_llm']} | {calls['judge_jev_alone']} | {saving['extraction']:.3f} "
            f"| {saving['judge']:.3f} | {saving['regeneration']:.3f} | {s:.3f} | {s / total:.0%} |"
        )


# ---------------------------------------------------------------- follow-up: extraction as a batch

# User decision 2026-11-20: measure only. The production setting after the switch: Claude Haiku
# 5.5, the production request builder (`request_params`, system prompt cached), main article
# segments of 02_segments.json.
BATCH_EXTRACT_ARTICLES = ("helgoland", "voynich")
BATCH_EXTRACT_REPEATS = 3
BATCH_POLL_SECONDS = 10


def main_extraction_requests(article: str) -> tuple[dict, list[tuple[str, str, dict]]]:
    """The main article source and the requests `extract_facts` makes for it."""
    from gut_zu_wissen.extraction import extract_facts

    src = article_sources(article)[0]
    fake = Collect({"candidates": []})
    extract_facts(fake, None, src["title"], src["revid"], _segments(src["segments"]), config())
    return src, fake.requests


def production_params(requests: Sequence[tuple[str, str, dict]]) -> list[dict]:
    """Each request as `AnthropicLlm.complete_json` sends it for extraction."""
    from gut_zu_wissen.llm import request_params

    return [
        request_params(HAIKU, MAX_TOKENS, sy, u, schema, cache_system=True)
        for sy, u, schema in requests
    ]


def parse_batch_result(r) -> dict:
    if r.result.type != "succeeded":
        return {"error": r.result.type, "usage": None, "answer": None}
    msg = r.result.message
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        answer = json.loads(text) if msg.stop_reason == "end_turn" else None
    except json.JSONDecodeError:
        answer = None
    return {
        "error": None if answer is not None else f"stop_reason {msg.stop_reason}",
        "usage": msg.usage.model_dump(exclude_none=True),
        "stop_reason": msg.stop_reason,
        "answer": answer,
    }


def replay_extraction(article: str, src: dict, requests, answers: Sequence[dict]) -> dict:
    """The extraction result of the answers, through the production `extract_facts`."""
    from gut_zu_wissen.extraction import extract_facts

    replay = Replay(
        [(request_digest(sy, u), a or {"candidates": []}) for (sy, u, _), a in
         zip(requests, answers, strict=True)]
    )  # fmt: skip
    res = extract_facts(
        replay, CachedRenderer(article), src["title"], src["revid"],
        _segments(src["segments"]), config(),
    )  # fmt: skip
    d = res.to_dict()
    return {"proposed": proposed_ids(d), "facts": [f["segment_id"] for f in d["facts"]]}


def sync_extraction(article: str) -> dict:
    """One synchronous run as the pipeline makes it: streamed `AnthropicLlm`, one call after the
    other, with the cache marker; wall time of all calls."""
    from gut_zu_wissen.extraction import extract_facts
    from gut_zu_wissen.llm import AnthropicLlm
    from gut_zu_wissen.usage import Usage

    src, requests = main_extraction_requests(article)
    usage = Usage(HAIKU)
    answers: list[dict] = []

    class Recording(AnthropicLlm):
        def complete_json(self, system, user, schema, **cache):
            answer = super().complete_json(system, user, schema, **cache)
            answers.append(answer)
            return answer

    llm = Recording(HAIKU, MAX_TOKENS, anthropic_client(), usage)
    start = time.time()
    res = extract_facts(
        llm, CachedRenderer(article), src["title"], src["revid"], _segments(src["segments"]),
        config(),
    )  # fmt: skip
    seconds = time.time() - start  # includes the (cached) render, a few ms
    counts = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
    }
    d = res.to_dict()
    return {
        "calls": usage.calls,
        "seconds": round(seconds, 1),
        "usage": counts,
        "usd_list": round(usage_cost(counts, HAIKU), 4),
        "proposed": proposed_ids(d),
        "facts": [f["segment_id"] for f in d["facts"]],
    }


def run_timed_batches(jobs: Mapping[str, list[dict]]) -> dict[str, dict]:
    """Submit one batch per job at once, poll until all ended, and record per batch the wall
    time from submit to the end of processing as seen by polling, and the API's own
    created_at/ended_at. A batch still open on an error is cancelled."""
    client = anthropic_client()
    open_ids: dict[str, str] = {}
    submitted: dict[str, float] = {}
    try:
        for name, params in jobs.items():
            reqs = [{"custom_id": f"r{n:04d}", "params": p} for n, p in enumerate(params)]
            submitted[name] = time.time()
            open_ids[name] = client.messages.batches.create(requests=reqs).id
            print(f"{name}: batch {open_ids[name]}, {len(reqs)} requests")
        done: dict[str, dict] = {}
        while len(done) < len(jobs):
            time.sleep(BATCH_POLL_SECONDS)
            for name, bid in open_ids.items():
                if name in done:
                    continue
                b = client.messages.batches.retrieve(bid)
                if b.processing_status != "ended":
                    continue
                polled = time.time() - submitted[name]
                by_id = {r.custom_id: parse_batch_result(r) for r in
                         client.messages.batches.results(bid)}  # fmt: skip
                done[name] = {
                    "batch_id": bid,
                    "seconds_polled": round(polled, 1),
                    "seconds_api": round((b.ended_at - b.created_at).total_seconds(), 1),
                    "results": [by_id[f"r{n:04d}"] for n in range(len(jobs[name]))],
                }
                print(f"  {name} ended after {polled:.0f} s ({b.request_counts})")
        open_ids.clear()
        return done
    finally:
        for name, bid in open_ids.items():
            if client.messages.batches.retrieve(bid).processing_status != "ended":
                client.messages.batches.cancel(bid)
                print(f"  cancelled {name} ({bid})")


def batch_extract() -> None:
    """Follow-up: extraction of two archived articles as Message Batches (Haiku 5.5, production
    requests), BATCH_EXTRACT_REPEATS repeats, against one synchronous run each."""
    prepared = {a: main_extraction_requests(a) for a in BATCH_EXTRACT_ARTICLES}
    body: dict = {"model": HAIKU, "articles": {}}
    for a in BATCH_EXTRACT_ARTICLES:
        print(f"sync {a} ...")
        body["articles"][a] = {"sync": sync_extraction(a), "batches": []}
        print(f"  {body['articles'][a]['sync']['seconds']} s")
    for rep in range(BATCH_EXTRACT_REPEATS):
        jobs = {f"{a}_r{rep}": production_params(prepared[a][1]) for a in prepared}
        done = run_timed_batches(jobs)
        for a, (src, requests) in prepared.items():
            d = done[f"{a}_r{rep}"]
            results = d.pop("results")
            d["summary"] = usage_summary(results, HAIKU)
            d["usage"] = [r["usage"] for r in results]
            d.update(replay_extraction(a, src, requests, [r["answer"] for r in results]))
            body["articles"][a]["batches"].append(d)
    paid = sum(
        b["summary"]["usd_paid"] for x in body["articles"].values() for b in x["batches"]
    ) + sum(x["sync"]["usd_list"] for x in body["articles"].values())
    body["summary"] = {"usd_paid": round(paid, 4)}
    print(save("batch_extract", "haiku_production", body), body["summary"])


def report_batch_extract() -> None:
    d = _saved("batch_extract", "haiku_production")
    if d is None:
        return
    print("\n## Follow-up: extraction as a batch (Haiku 5.5, production requests)\n")
    print(
        "| Article | run | calls | wall s | cache read tok | list $ | paid $ "
        "| facts | J facts vs sync | J proposed vs sync |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, x in d["articles"].items():
        s = x["sync"]
        print(
            f"| {a} | sync | {s['calls']} | {s['seconds']:.0f} "
            f"| {s['usage']['cache_read_input_tokens']:,} | {s['usd_list']:.4f} "
            f"| {s['usd_list']:.4f} | {len(s['facts'])} | 1.00 | 1.00 |"
        )
        for n, b in enumerate(x["batches"]):
            cached = sum((u or {}).get("cache_read_input_tokens", 0) for u in b["usage"])
            print(
                f"| {a} | batch {n + 1} | {b['summary']['calls']} | {b['seconds_polled']:.0f} "
                f"(API {b['seconds_api']:.0f}) | {cached:,} | {b['summary']['usd_list']:.4f} "
                f"| {b['summary']['usd_paid']:.4f} | {len(b['facts'])} "
                f"| {jaccard(b['facts'], s['facts']):.2f} "
                f"| {jaccard(b['proposed'], s['proposed']):.2f} |"
            )
        bs = x["batches"]
        pairs = [(i, j) for i in range(len(bs)) for j in range(i + 1, len(bs))]
        if pairs:
            js = [jaccard(bs[i]["facts"], bs[j]["facts"]) for i, j in pairs]
            print(f"| {a} | batch vs batch | | | | | | | {min(js):.2f}-{max(js):.2f} | |")


def report() -> None:
    report_episodes()
    report_extraction()
    report_ranking()
    report_judge()
    report_cache()
    report_jev()
    report_batch_extract()
    print()
    spend()


def spend() -> float:
    """USD actually paid for every saved result (batch prices where batched)."""
    total = 0.0
    for kind in ("extract", "rank", "judge", "cache", "batch_extract"):
        for p in sorted((RESULTS / kind).glob("*.json")):
            body = load(p)
            if "summary" not in body:
                continue
            s = body["summary"]
            total += s["usd_paid"]
            print(f"{kind:8} {p.stem:40} ${s['usd_paid']:.4f}")
    jev_tokens = sum(
        r["usage"]["input_tokens"] for p in (RESULTS / "jev").glob("*.json") for r in load(p)
    )
    print(f"jev      {jev_tokens} tokens ${jev_tokens * JEV_PRICE / 1e6:.4f}")
    total += jev_tokens * JEV_PRICE / 1e6
    print(f"TOTAL ${total:.4f}")
    return total


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    command, *args = sys.argv[1:]
    {
        "estimate": estimate,
        "extract": extract,
        "rank": rank,
        "chain": rank_chain,
        "judge": judge,
        "cache": cache,
        "jev": jev,
        "spend": spend,
        "report": report,
        "offers": offers,
        "batch_extract": batch_extract,
    }[command](*args)
