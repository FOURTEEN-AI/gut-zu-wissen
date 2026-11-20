"""Spike S1 follow-up: completeness per sentence (user decision 2026-11-20). Measurement only.

The judge asks one completeness question per turn (`old`, production). This spike splits
`redebeitrag.text` into sentences (SoMaJo, the tokenizer of the segmentation step) and asks
about each sentence whether it is a pure reaction or covered by `aussagen`. Wording:
spikes/s1_sentences_questions.py. Turn decision: gut_zu_wissen.judge.decide over the statement
p and the sentence p (complete if every sentence >= t, incomplete if any <= 1 - t).

Cases (`cases`):
- `final`: every final turn of the five completed episodes of 2026-11-20;
- `rejected`: every turn version the judge (Jev or LLM judge, not a gate) rejected, in any
  archived folder: final turns of failed runs, superseded judgements, discarded conversations;
- `s1`: the hand-labelled S1 turns with their round-2 state and statement questions.
State and statement questions as production builds them (`build_request`); facts: every fact
extracted in the run (main article, linked articles, Einordnung), since discarded conversations
had their own offers. Neighbour turn of a superseded version: see `_neighbour`.

Labels: the final verdict of each turn version (LLM judge, or Jev where it decided alone); S1:
the hand labels.

Usage (TYPESAFE_API_KEY in the environment for `run`):
  uv run python spikes/s1_sentences.py cases    # build the case file
  uv run python spikes/s1_sentences.py run      # live requests, REPEATS repeats
  uv run python spikes/s1_sentences.py report   # tables from saved results

Per case and repeat, one request per variant: the production statement questions plus the
variant's completeness questions, on the production state. The results store answers, usage and
latency; `payload` rebuilds the exact request from the case file.
"""

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from s1_completeness import S1_CASES, S1_V2, post
from s1_completeness_questions import old
from s1_sentences_questions import VARIANTS, sentence_questions

from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.judge import Decision, build_request, decide
from gut_zu_wissen.ranking import RankedFact
from gut_zu_wissen.segmentation import default_tokenizer

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "archive" / "episodes"
COMPLETED = [
    "2026-11-20_axolotl",
    "2026-11-20_tunguska-ereignis",
    "2026-11-20_voynich-manuskript",
    "2026-11-20_bernstein-2",
    "2026-11-20_helgoland",
]
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes"
CASES = FIXTURES / "s1_sentences_cases.json"
RESULTS = FIXTURES / "s1_sentences_results"
MODEL = "jev-1.14.0"  # config/pipeline.toml [models] jev
T = 0.95  # config/pipeline.toml [judge] t
REPEATS = 2
WORKERS = 6
PRICE_PER_MTOK = 0.042  # docs.typesafe.ai/models: Jev input, output free
ALL_VARIANTS = ("old", *VARIANTS)


# ---------------------------------------------------------------- pure helpers


def split_sentences(text: str, tokenizer) -> list[str]:
    """Sentences of a turn text by SoMaJo (character offsets, original spelling kept)."""
    out = []
    for tokens in tokenizer.tokenize_text([text]):
        s = text[tokens[0].character_offset[0] : tokens[-1].character_offset[1]].strip()
        if s:
            out.append(s)
    return out


def sentence_p(variant: str, answers: dict[str, float], n: int) -> float:
    """p that sentence n is fine. `split*`: a sentence is fine if it is a pure reaction or
    covered, so p = max(reaction, covered); it is bad only if both are low."""
    if variant == "either":
        return answers[f"either_ok_{n}"]
    return max(answers[f"{variant}_reaction_{n}"], answers[f"{variant}_covered_{n}"])


def completeness_ps(variant: str, answers: dict[str, float], sentences: int) -> list[float]:
    """Completeness p of a turn: the old single question, or one p per sentence."""
    if variant == "old":
        return [answers["vollstaendig"]]
    return [sentence_p(variant, answers, n) for n in range(sentences)]


def statement_ps(answers: dict[str, float]) -> list[float]:
    return [p for k, p in answers.items() if k.startswith("aussage_")]


def turn_decision(statements: list[float], completeness: list[float], t: float) -> Decision:
    """Jev's decision on the turn: production `decide` over all p."""
    return decide([*statements, *completeness], t)


# ---------------------------------------------------------------- cases


def _turn(d: dict) -> Turn:
    statements = [Statement(s["text"], s["fact_ids"]) for s in d.get("statements", [])]
    return Turn(d["index"], d["speaker"], d["text"], d["fact_ids"], statements)


def _ranked(r: dict) -> RankedFact:
    return RankedFact(r["rank"], r["article"], r["revid"], Fact(**r["fact"]))


def _facts(folder: Path) -> dict[str, RankedFact]:
    """Every fact extracted in the run; the offer of `04_ranking.json` wins where present."""
    main = json.loads((folder / "01_article.json").read_text())
    sources = [(main["title"], json.loads((folder / "03_facts.json").read_text()))]
    for path in sorted((folder / "linked").glob("*.json")):
        linked = json.loads(path.read_text())
        if "extraction" in linked:
            sources.append((linked["title"], linked["extraction"]))
    facts: dict[str, RankedFact] = {}
    for title, extraction in sources:
        for rank, f in enumerate(extraction["facts"], start=1):
            facts[f["segment_id"]] = RankedFact(rank, title, extraction["revid"], Fact(**f))
    ranking = json.loads((folder / "04_ranking.json").read_text())
    for key in ("einordnung_facts", "main_facts", "linked_facts"):
        for r in ranking.get(key, []):
            facts[r["fact"]["segment_id"]] = _ranked(r)
    return facts


def _neighbour(regenerations: list[dict], conv: int, n: int, replaced_by: int, turns: dict):
    """Turn n-1 as it stood when the version replaced by regeneration `replaced_by` of turn n
    was judged: the version that the first later regeneration of n-1 replaced, if there is one,
    else the conversation's last version of n-1. (Approximation: the archive does not record
    the neighbour at judging time.)"""
    if n == 0:
        return None
    for k, r in enumerate(regenerations):
        if k > replaced_by and r["conversation"] == conv and r["turn_index"] == n - 1:
            return r["previous"]
    return turns.get(n - 1)


def archive_items(folder: Path) -> list[dict]:
    """(group, id, turn, previous, judgement) of one folder."""
    name = folder.name
    dialog = json.loads((folder / "05_dialog.json").read_text())
    judgement = json.loads((folder / "06_judgement.json").read_text())
    completed = name in COMPLETED
    convs = {judgement["conversation"]: (dialog["turns"], judgement["judgements"])}
    for c in dialog["discarded_conversations"]:
        convs[c["conversation"]] = (c["dialog"]["turns"], c["judgements"])
    items = []
    for conv, (turns, judgements) in convs.items():
        by_index = {t["index"]: t for t in turns}
        final = conv == judgement["conversation"]
        for t, j in zip(turns, judgements, strict=True):
            judged = j["decided_by"] != "gate"
            if final and completed:
                group = "final"
            elif not j["accepted"] and judged:
                group = "rejected"
            else:
                continue
            prev = by_index.get(t["index"] - 1)
            items.append((group, f"{name}/c{conv}-t{t['index']}", t, prev, j))
    for s in judgement["superseded_judgements"]:
        j = s["judgement"]
        if j["accepted"] or j["decided_by"] == "gate":
            continue
        conv, n = s["conversation"], s["turn_index"]
        old_turn = dialog["regenerations"][s["replaced_by"]]["previous"]
        by_index = {t["index"]: t for t in convs[conv][0]}
        prev = _neighbour(dialog["regenerations"], conv, n, s["replaced_by"], by_index)
        items.append(("rejected", f"{name}/c{conv}-t{n}-v{s['version']}", old_turn, prev, j))
    facts = _facts(folder)
    out = []
    for group, cid, t, prev, j in items:
        state, questions = build_request(_turn(t), _turn(prev) if prev else None, facts)
        out.append(
            {
                "group": group,
                "id": cid,
                "speaker": t["speaker"],
                "accepted": j["accepted"],
                "decided_by": j["decided_by"],
                "reason": j["reason"],
                "archived_jev_p": j["jev_p"],
                "state": state,
                "statement_questions": {k: q for k, q in questions.items() if k != "vollstaendig"},
            }
        )
    return out


def s1_items() -> list[dict]:
    labelled = json.loads(S1_CASES.read_text())
    gold = {c["id"]: c for g in ("turns", "holdout_turns") for c in labelled[g]}
    out = []
    for r in json.loads(S1_V2.read_text()):
        if r["kind"] != "turn":
            continue
        c = gold[r["id"]]
        questions = r["request"]["questions"]
        out.append(
            {
                "group": "s1",
                "id": f"s1-{c['id']}",
                "speaker": r["request"]["state"]["redebeitrag"]["sprecher"],
                "accepted": c["vollstaendig_gold"] and all(a["gold"] for a in c["aussagen"]),
                "decided_by": "hand",
                "reason": c["category"],
                "complete_by_hand": c["vollstaendig_gold"],
                "state": r["request"]["state"],
                "statement_questions": {
                    k: q for k, q in questions.items() if k.startswith("aussage_")
                },
            }
        )
    return out


def cases() -> None:
    tokenizer = default_tokenizer()
    folders = sorted(p.parent for p in EPISODES.glob("*/06_judgement.json"))
    items = [c for f in folders for c in archive_items(f)] + s1_items()
    seen: set[str] = set()
    out = []
    for c in items:
        # The same turn version can appear twice (e.g. superseded and final of a failed run).
        key = json.dumps([c["state"]["redebeitrag"], c["state"]["aussagen"]], ensure_ascii=False)
        if c["group"] == "rejected" and key in seen:
            continue
        seen.add(key)
        c["sentences"] = split_sentences(c["state"]["redebeitrag"]["text"], tokenizer)
        out.append(c)
    body = {
        "_note": "Turns from archive/episodes (all folders with a judgement) and the S1 turns, "
        "state and statement questions as built by gut_zu_wissen.judge.build_request. Fact "
        "sentences and context are text from the German Wikipedia (CC BY-SA 4.0, see NOTICE). "
        "`accepted` is the final verdict of the turn version; `sentences` is the SoMaJo split.",
        "cases": out,
    }
    CASES.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    for g in ("final", "rejected", "s1"):
        print(g, sum(c["group"] == g for c in out))


# ---------------------------------------------------------------- live run


def payload(case: dict, variant: str) -> tuple[dict, dict]:
    """State and questions of one request: statement questions plus the variant's questions."""
    questions = dict(case["statement_questions"])
    if variant == "old":
        questions["vollstaendig"] = old()
    else:
        questions |= sentence_questions(variant, case["sentences"])
    return case["state"], questions


def _one(client: httpx.Client, case: dict, variant: str) -> dict:
    state, questions = payload(case, variant)
    start = time.perf_counter()
    r = post(client, state, questions)
    elapsed = time.perf_counter() - start
    assert r["status"] == 200, (case["id"], variant, r["status"], r["response"])
    return {
        "variant": variant,
        "questions": len(questions),
        "seconds": round(elapsed, 3),
        "usage": r["response"]["usage"],
        "answers": {k: a["noul"] for k, a in r["response"]["answers"].items()},
    }


def run() -> None:
    all_cases = json.loads(CASES.read_text())["cases"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    key = os.environ["TYPESAFE_API_KEY"]
    with (
        httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        for rep in range(REPEATS):
            out = RESULTS / f"{MODEL}_{rep}.json"
            if out.exists():
                continue
            jobs = [(c, v) for c in all_cases for v in ALL_VARIANTS]
            done = list(pool.map(lambda job: _one(client, *job), jobs))
            records = []
            for n, c in enumerate(all_cases):
                chunk = done[n * len(ALL_VARIANTS) : (n + 1) * len(ALL_VARIANTS)]
                records.append({"id": c["id"], "requests": chunk})
            out.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
            print(f"{out.name}: {len(records)} cases")


# ---------------------------------------------------------------- report


def evaluate(case: dict, request: dict) -> dict:
    variant, answers = request["variant"], request["answers"]
    stmt = statement_ps(answers)
    compl = completeness_ps(variant, answers, len(case["sentences"]))
    return {
        "decision": turn_decision(stmt, compl, T),
        "completeness": decide(compl, T),
        "stmt_ok": all(p >= T for p in stmt),
        "min_compl": min(compl),
        "low_sentence": case["sentences"][compl.index(min(compl))] if variant != "old" else None,
    }


def report() -> None:
    cases_by_id = {c["id"]: c for c in json.loads(CASES.read_text())["cases"]}
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{MODEL}_*.json"))]
    tokens = sum(q["usage"]["input_tokens"] for rec in runs for r in rec for q in r["requests"])
    print(f"repeats {len(runs)}, input tokens {tokens}, cost ${tokens * PRICE_PER_MTOK / 1e6:.3f}")
    for group in ("final", "rejected", "s1"):
        print(f"\n=== {group}")
        print(
            f"{'variant':10} {'rep':>3} {'n':>4} {'alone':>6} {'acc':>5} {'rej':>5} "
            f"{'unsure':>6} {'c>=t':>5} {'c<=1-t':>6} {'FA':>3} {'FR':>3} {'maxRej':>6} "
            f"{'q/turn':>6} {'s/req':>6} {'tok/req':>7}"
        )
        for v in ALL_VARIANTS:
            for rep, records in enumerate(runs):
                rows = []
                for r in records:
                    case = cases_by_id[r["id"]]
                    if case["group"] != group:
                        continue
                    req = next(q for q in r["requests"] if q["variant"] == v)
                    rows.append((case, req, evaluate(case, req)))
                n = len(rows)
                acc = sum(e["decision"] is Decision.SUPPORTED for _, _, e in rows)
                rej = sum(e["decision"] is Decision.REJECTED for _, _, e in rows)
                chi = sum(e["completeness"] is Decision.SUPPORTED for _, _, e in rows)
                clo = sum(e["completeness"] is Decision.REJECTED for _, _, e in rows)
                fa = [
                    c["id"]
                    for c, _, e in rows
                    if not c["accepted"] and e["decision"] == "supported"
                ]
                fr = [
                    c["id"] for c, _, e in rows if c["accepted"] and e["completeness"] == "rejected"
                ]
                rej_p = [e["min_compl"] for c, _, e in rows if not c["accepted"]]
                print(
                    f"{v:10} {rep:>3} {n:>4} {(acc + rej) / n:6.0%} {acc / n:5.0%} {rej / n:5.0%} "
                    f"{(n - acc - rej) / n:6.0%} {chi / n:5.0%} {clo / n:6.0%} {len(fa):>3} "
                    f"{len(fr):>3} {max(rej_p) if rej_p else float('nan'):6.2f} "
                    f"{statistics.mean(q['questions'] for _, q, _ in rows):6.1f} "
                    f"{statistics.mean(q['seconds'] for _, q, _ in rows):6.2f} "
                    f"{statistics.mean(q['usage']['input_tokens'] for _, q, _ in rows):7.0f}"
                )
                for cid in fa:
                    print(f"    FALSE ACCEPT {cid}")
                for cid in fr:
                    print(f"    false reject {cid}")


def borderline(limit: int = 12) -> None:
    """Rejected turns with the highest completeness p per variant (min over repeats shown)."""
    cases_by_id = {c["id"]: c for c in json.loads(CASES.read_text())["cases"]}
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob(f"{MODEL}_*.json"))]
    for v in ALL_VARIANTS:
        print(f"\n=== {v}")
        rows = []
        for cid, case in cases_by_id.items():
            if case["accepted"]:
                continue
            evals = []
            for records in runs:
                r = next(x for x in records if x["id"] == cid)
                evals.append(evaluate(case, next(q for q in r["requests"] if q["variant"] == v)))
            rows.append((max(e["min_compl"] for e in evals), cid, evals))
        for _p, cid, evals in sorted(rows, reverse=True)[:limit]:
            ps = "/".join(f"{e['min_compl']:.2f}" for e in evals)
            stmt = "stmt ok" if all(e["stmt_ok"] for e in evals) else "stmt <t"
            print(f"  {ps:11} {stmt}  {cid}  low: {evals[0]['low_sentence']}")


if __name__ == "__main__":
    {"cases": cases, "run": run, "report": report, "borderline": borderline}[sys.argv[1]]()
