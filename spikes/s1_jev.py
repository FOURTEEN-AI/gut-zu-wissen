"""Spike S1: Jev on German turn/fact pairs, D1 content preservation, error codes, latency.

Usage (TYPESAFE_API_KEY in the environment):
  uv run --group spikes python spikes/s1_jev.py run      # cases × models × languages × repeats
  uv run --group spikes python spikes/s1_jev.py errors   # overlength and other error codes
  uv run --group spikes python spikes/s1_jev.py report   # evaluate saved results

Every request payload and response is saved, so any result can be rerun.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from s1_questions import (
    completeness_question,
    d1_conveys,
    d1_holistic,
    d1_in_before,
    d1_nothing_added,
    sentence_is_covered,
    sentence_is_reaction,
    statement_question,
)
from somajo import SoMaJo

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes" / "s1_cases.json"
RESULTS = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes" / "s1_results"
API = "https://api.typesafe.ai/v1/systemone"
MODELS = ["jev-1.13.0", "jev-1.14.0"]
LANGS = ["de", "en"]
REPEATS = 2
T = 0.95  # config/pipeline.toml [judge] t
T_SKILL = 0.7  # TypeSafe's suggested starting threshold, for comparison only


def turn_request(case: dict, cases: dict, lang: str) -> dict:
    cited = sorted({f for a in case["aussagen"] for f in a["fakten"]})
    state: dict = {"redebeitrag": {"sprecher": case["sprecher"], "text": case["text"]}}
    if "vorher" in case:
        state["vorheriger_redebeitrag"] = case["vorher"]
    state["aussagen"] = [{"text": a["text"], "fakten": a["fakten"]} for a in case["aussagen"]]
    state["fakten"] = {f: cases["fakten"][f] for f in cited}
    kontext = {f: cases["kontext"][f] for f in cited if f in cases["kontext"]}
    if kontext:
        state["kontext"] = kontext
    questions = {
        f"aussage_{i}": statement_question(lang, a["text"], a["fakten"])
        for i, a in enumerate(case["aussagen"])
    }
    questions["vollstaendig"] = completeness_question(lang)
    return {"state": state, "questions": questions}


def d1_request(pair: dict, lang: str) -> dict:
    questions = {"holistic": d1_holistic(lang), "nothing_added": d1_nothing_added(lang)}
    for i, s in enumerate(pair["aussagen_vorher"]):
        questions[f"conveys_{i}"] = d1_conveys(lang, s)
    return {"state": {"vorher": pair["vorher"], "nachher": pair["nachher"]}, "questions": questions}


def turn_request_v2(case: dict, cases: dict, tokenizer: SoMaJo) -> tuple[dict, list[str]]:
    """Round 2: article title per fact; completeness asked per sentence (two Nouls each)."""
    body = turn_request(case, cases, "de")
    state = body["state"]
    state["fakten"] = {
        f: {"artikel": cases["artikel"][f.split("-")[0]], "satz": s}
        for f, s in state["fakten"].items()
    }
    sentences = [
        case["text"][toks[0].character_offset[0] : toks[-1].character_offset[1]]
        for toks in tokenizer.tokenize_text([case["text"]])
    ]
    for j, sentence in enumerate(sentences):
        body["questions"][f"satz_{j}_reaktion"] = sentence_is_reaction(sentence)
        body["questions"][f"satz_{j}_erfasst"] = sentence_is_covered(sentence)
    return body, sentences


def d1_request_v2(pair: dict) -> dict:
    body = d1_request(pair, "de")
    for k, s in enumerate(pair["aussagen_nachher"]):
        body["questions"][f"in_before_{k}"] = d1_in_before(s)
    return body


def post(client: httpx.Client, model: str, body: dict) -> tuple[int, dict, float]:
    payload = {"model": model, **body}
    for attempt in range(5):
        started = time.perf_counter()
        r = client.post(API, json=payload)
        elapsed = time.perf_counter() - started
        if r.status_code in (429, 529) and attempt < 4:
            time.sleep(2**attempt)
            continue
        return r.status_code, r.json(), elapsed
    raise RuntimeError("unreachable")


def client() -> httpx.Client:
    key = os.environ["TYPESAFE_API_KEY"]
    return httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)


def run() -> None:
    cases = json.loads(CASES.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    with client() as c:
        for model in MODELS:
            for lang in LANGS:
                for rep in range(REPEATS):
                    records = []
                    items = [("turn", t, turn_request(t, cases, lang)) for t in cases["turns"]]
                    items += [("d1", p, d1_request(p, lang)) for p in cases["d1_pairs"]]
                    for kind, item, body in items:
                        status, resp, secs = post(c, model, body)
                        records.append(
                            {
                                "kind": kind,
                                "id": item["id"],
                                "request": {"model": model, **body},
                                "status": status,
                                "response": resp,
                                "seconds": round(secs, 3),
                            }
                        )
                    out = RESULTS / f"{model}_{lang}_{rep}.json"
                    out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
                    print(f"{out.name}: {len(records)} requests")


def run2() -> None:
    cases = json.loads(CASES.read_text())
    tokenizer = SoMaJo("de_CMC", character_offsets=True)
    with client() as c:
        for model in MODELS:
            for rep in range(REPEATS):
                records = []
                for group in ("turns", "holdout_turns", "d1_pairs", "holdout_d1"):
                    for item in cases[group]:
                        if group.endswith("turns"):
                            body, _ = turn_request_v2(item, cases, tokenizer)
                            kind = "turn"
                        else:
                            body, kind = d1_request_v2(item), "d1"
                        status, resp, secs = post(c, model, body)
                        records.append(
                            {
                                "kind": kind,
                                "group": group,
                                "id": item["id"],
                                "request": {"model": model, **body},
                                "status": status,
                                "response": resp,
                                "seconds": round(secs, 3),
                            }
                        )
                out = RESULTS / f"v2_{model}_{rep}.json"
                out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
                print(f"{out.name}: {len(records)} requests")


def report2() -> None:
    cases = json.loads(CASES.read_text())
    items = {
        i["id"]: i for g in ("turns", "holdout_turns", "d1_pairs", "holdout_d1") for i in cases[g]
    }
    for path in sorted(RESULTS.glob("v2_*.json")):
        records = json.loads(path.read_text())
        secs = [r["seconds"] for r in records]
        tokens = sum(r["response"]["usage"]["input_tokens"] for r in records)
        print(
            f"\n=== {path.stem}  input_tokens={tokens}  latency median="
            f"{statistics.median(secs):.2f}s max={max(secs):.2f}s"
        )
        for group in ("turns", "holdout_turns"):
            for variant in ("v1_completeness", "v2_per_sentence"):
                out = {
                    "accept_ok": 0,
                    "FALSE_ACCEPT": 0,
                    "reject_ok": 0,
                    "false_reject": 0,
                    "judge": 0,
                }
                notes = []
                for r in (r for r in records if r["group"] == group):
                    case = items[r["id"]]
                    ans = {k: v["noul"] for k, v in r["response"]["answers"].items()}
                    ps = [ans[f"aussage_{i}"] for i in range(len(case["aussagen"]))]
                    if variant == "v1_completeness":
                        ps.append(ans["vollstaendig"])
                    else:
                        j = 0
                        while f"satz_{j}_reaktion" in ans:
                            ps.append(max(ans[f"satz_{j}_reaktion"], ans[f"satz_{j}_erfasst"]))
                            j += 1
                    gold = all(a["gold"] for a in case["aussagen"]) and case["vollstaendig_gold"]
                    v = verdict(ps, T)
                    if v == "accept":
                        out["accept_ok" if gold else "FALSE_ACCEPT"] += 1
                        if not gold:
                            notes.append(f"FA:{case['id']}")
                    elif v == "reject":
                        out["reject_ok" if not gold else "false_reject"] += 1
                        if gold:
                            notes.append(f"FR:{case['id']}")
                    else:
                        out["judge"] += 1
                        notes.append(f"J:{case['id']}")
                print(f"  {group:13} {variant:16} {out}")
                print(f"  {'':13} {'':16} {' '.join(notes)}")
        for group in ("d1_pairs", "holdout_d1"):
            for mode in ("holistic", "atomic_v1", "atomic_v2"):
                res = []
                for r in (r for r in records if r["group"] == group):
                    pair = items[r["id"]]
                    ans = {k: v["noul"] for k, v in r["response"]["answers"].items()}
                    if mode == "holistic":
                        ps = [ans["holistic"]]
                    elif mode == "atomic_v1":
                        ps = [v for k, v in ans.items() if k.startswith(("conveys", "nothing"))]
                    else:
                        ps = [v for k, v in ans.items() if k.startswith(("conveys", "in_before"))]
                    v = verdict(ps, T)
                    ok = (v == "accept") == pair["gold"]
                    res.append(f"{pair['id']}:{v[:3]}{'' if ok else '✗'}")
                print(f"  {group:13} D1 {mode:10} {' '.join(res)}")


def errors() -> None:
    """Probe error codes. Long state: real article wikitext, repeated."""
    text = (CASES.parent / "FC_Bayern_München.json").read_text()
    q = {"q": {"type": "noul", "instructions": "Geht es in `text` um Fußball?"}}
    probes = {
        "per_question_over_32k": {"state": {"text": text[:120_000]}, "questions": q},
        "per_request_over_64k": {"state": {"text": (text * 2)[:260_000]}, "questions": q},
        "request_split_over_64k": {
            # each question small, state ~40k: per question < 32k? no: tests the sum rule
            "state": {"text": text[:100_000]},
            "questions": {f"q{i}": q["q"] for i in range(3)},
        },
        "malformed_question": {"state": "x", "questions": {"q": {"type": "noul", "foo": 1}}},
        "unknown_model": None,
        "bad_key": None,
    }
    results = {}
    with client() as c:
        for name, body in probes.items():
            if name == "unknown_model":
                status, resp, secs = post(c, "jev-0.0.0", {"state": "x", "questions": q})
            elif name == "bad_key":
                with httpx.Client(headers={"Authorization": "Bearer invalid"}, timeout=60) as b:
                    r = b.post(API, json={"model": MODELS[0], "state": "x", "questions": q})
                    status, resp, secs = r.status_code, r.json(), 0.0
            else:
                status, resp, secs = post(c, MODELS[1], body)
            results[name] = {"status": status, "response": resp, "seconds": round(secs, 3)}
            print(name, status, json.dumps(resp, ensure_ascii=False)[:300])
    (RESULTS / "errors.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


def verdict(ps: list[float], t: float) -> str:
    """Threshold decision from the design: all p >= t supported; any p <= 1-t not; else judge."""
    if all(p >= t for p in ps):
        return "accept"
    if any(p <= 1 - t for p in ps):
        return "reject"
    return "uncertain"


def report() -> None:
    cases = json.loads(CASES.read_text())
    turns = {t["id"]: t for t in cases["turns"]}
    pairs = {p["id"]: p for p in cases["d1_pairs"]}
    for path in sorted(RESULTS.glob("jev-*.json")):
        records = json.loads(path.read_text())
        q_rows, turn_rows, d1_rows, secs, tokens = [], [], [], [], 0
        for r in records:
            if r["status"] != 200:
                print(path.name, r["id"], "status", r["status"])
                continue
            secs.append(r["seconds"])
            tokens += r["response"]["usage"]["input_tokens"]
            ans = {k: v["noul"] for k, v in r["response"]["answers"].items()}
            if r["kind"] == "turn":
                case = turns[r["id"]]
                for i, a in enumerate(case["aussagen"]):
                    q_rows.append(("aussage", case["id"], i, a["gold"], ans[f"aussage_{i}"]))
                q_rows.append(
                    ("vollst", case["id"], -1, case["vollstaendig_gold"], ans["vollstaendig"])
                )
                gold = all(a["gold"] for a in case["aussagen"]) and case["vollstaendig_gold"]
                turn_rows.append((case["id"], case["category"], gold, list(ans.values())))
            else:
                pair = pairs[r["id"]]
                atomic = [v for k, v in ans.items() if k != "holistic"]
                d1_rows.append((pair["id"], pair["gold"], ans["holistic"], atomic))
        print(
            f"\n=== {path.stem}  requests={len(records)}  input_tokens={tokens}  "
            f"latency median={statistics.median(secs):.2f}s max={max(secs):.2f}s"
        )
        for kind in ("aussage", "vollst"):
            rows = [x for x in q_rows if x[0] == kind]
            for t in (T, T_SKILL):
                stats = {"right": 0, "WRONG": 0, "unsure": 0}
                for _, _, _, gold, p in rows:
                    if p >= t:
                        stats["right" if gold else "WRONG"] += 1
                    elif p <= 1 - t:
                        stats["WRONG" if gold else "right"] += 1
                    else:
                        stats["unsure"] += 1
                print(f"  {kind:8} t={t}: {stats}  (n={len(rows)})")
        for t in (T, T_SKILL):
            outcome = {
                "accept_ok": 0,
                "FALSE_ACCEPT": 0,
                "reject_ok": 0,
                "false_reject": 0,
                "judge": 0,
            }
            bad = []
            for cid, cat, gold, ps in turn_rows:
                v = verdict(ps, t)
                if v == "accept":
                    outcome["accept_ok" if gold else "FALSE_ACCEPT"] += 1
                    if not gold:
                        bad.append(f"FA:{cid}/{cat}")
                elif v == "reject":
                    outcome["reject_ok" if not gold else "false_reject"] += 1
                    if gold:
                        bad.append(f"FR:{cid}/{cat}")
                else:
                    outcome["judge"] += 1
            print(f"  turns    t={t}: {outcome} {' '.join(bad)}")
        for mode in ("holistic", "atomic"):
            res = []
            for pid, gold, hol, atomic in d1_rows:
                ps = [hol] if mode == "holistic" else atomic
                v = verdict(ps, T)
                ok = (v == "accept") == gold  # D1: anything but accept means continue
                res.append(f"{pid}:{v[:3]}{'' if ok else '✗'}")
            print(f"  D1 {mode:8} t={T}: {' '.join(res)}")


def detail(name: str) -> None:
    """Per-question probabilities for one result file."""
    cases = json.loads(CASES.read_text())
    turns = {t["id"]: t for t in cases["turns"]}
    for r in json.loads((RESULTS / f"{name}.json").read_text()):
        ans = {k: v["noul"] for k, v in r["response"].get("answers", {}).items()}
        if r["kind"] == "turn":
            case = turns[r["id"]]
            golds = [a["gold"] for a in case["aussagen"]] + [case["vollstaendig_gold"]]
            cells = [
                f"{p:.2f}{'' if g else '*'}" for p, g in zip(ans.values(), golds, strict=False)
            ]
            print(f"{r['id']} {case['category']:22} {' '.join(cells)}")
        else:
            print(f"{r['id']} " + " ".join(f"{k}={p:.2f}" for k, p in ans.items()))


if __name__ == "__main__":
    {"run": run, "run2": run2, "report2": report2, "errors": errors, "report": report}.get(
        sys.argv[1], lambda: detail(sys.argv[2])
    )()
