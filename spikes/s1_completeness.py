"""Spike S1 follow-up: rework of the completeness question (user decision 2026-11-20).

Measures the old and new completeness wordings on real turns from the Axolotl run
(archive/episodes/2026-11-19_axolotl-2, all three conversations plus superseded turn versions)
and on the hand-labelled S1 turns. Labels: the final verdict per turn (LLM judge, or Jev where
Jev decided alone); for rejected turns also a completeness label by hand (Claude).

Usage (TYPESAFE_API_KEY in the environment for `run`):
  uv run --group spikes python spikes/s1_completeness.py cases   # build the case file
  uv run --group spikes python spikes/s1_completeness.py run     # round 1: live requests
  uv run --group spikes python spikes/s1_completeness.py run2    # round 2: against `fakten`
  uv run --group spikes python spikes/s1_completeness.py report  # tables from saved results

Every request payload and response is saved, so any result can be rerun.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
from s1_completeness_questions import FULL_STATE, INVERTED, REDUCED_STATE, ROUND2

from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.judge import build_request
from gut_zu_wissen.ranking import RankedFact

ROOT = Path(__file__).resolve().parent.parent
EPISODE = ROOT / "archive" / "episodes" / "2026-11-19_axolotl-2"
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes"
CASES = FIXTURES / "s1_completeness_cases.json"
S1_CASES = FIXTURES / "s1_cases.json"
S1_V2 = FIXTURES / "s1_results" / "v2_jev-1.14.0_0.json"  # round-2 state of every S1 turn
RESULTS = FIXTURES / "s1_completeness_results"
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # config/pipeline.toml [models] jev
T = 0.95  # config/pipeline.toml [judge] t
REPEATS = 2
COMPLETENESS = "vollstaendig"

# Completeness label by hand for the turns the judge rejected (Claude, from the judge's reason):
# False = the text says something that is missing from its statements.
HAND_LABELS = {
    "c3-t3": (False, "„bleibt sein Leben lang eine Art Larve“ fehlt unter den Aussagen"),
    "c3-t14": (False, "„Es gibt keine Zellen, die einfach alles werden“ fehlt"),
    "c3-t34": (False, "Prämisse der Frage (Axolotl im Film) ohne Aussage"),
    "c1-t16": (False, "Grenzfall: „ergibt Sinn“ ist eine Wertung, Rest steht in den Aussagen"),
    "c1-t34": (False, "„starkes Wort für abgeschriebene Sätze“ und Prämisse fehlen"),
    "c2-t26": (False, "„kulturelle Spuren“, „berühmten Theaterstück“ fehlen"),
    "c2-t34": (False, "„Erst gefeiert, dann beschuldigt“ fehlt"),
    "c1-t1-v0": (False, "Grenzfall: „Auf diese Eigenschaft …“ geht über die Aussage hinaus"),
    "c2-t14-v0": (True, "Grenzfall: Text steht in der Aussage; die Aussage ist nicht belegt"),
    "c3-t2-v0": (False, "Zusammenfassung des Gesagten als eigene Feststellung ohne Aussage"),
}


# ---------------------------------------------------------------- cases


def _turn(d: dict) -> Turn:
    statements = [Statement(s["text"], s["fact_ids"]) for s in d["statements"]]
    return Turn(d["index"], d["speaker"], d["text"], d["fact_ids"], statements)


def _facts() -> dict[str, RankedFact]:
    """Every extracted fact of the run: main article and linked articles."""
    main = json.loads((EPISODE / "01_article.json").read_text())
    facts: dict[str, RankedFact] = {}
    sources = [(main["title"], json.loads((EPISODE / "03_facts.json").read_text()))]
    for path in sorted((EPISODE / "linked").glob("*.json")):
        linked = json.loads(path.read_text())
        sources.append((linked["title"], linked["extraction"]))
    for title, extraction in sources:
        for rank, f in enumerate(extraction["facts"], start=1):
            facts[f["segment_id"]] = RankedFact(rank, title, extraction["revid"], Fact(**f))
    return facts


def cases() -> None:
    dialog = json.loads((EPISODE / "05_dialog.json").read_text())
    judgement = json.loads((EPISODE / "06_judgement.json").read_text())
    facts = _facts()
    convs = [(judgement["conversation"], dialog["turns"], judgement["judgements"])]
    convs += [
        (c["conversation"], c["dialog"]["turns"], c["judgements"])
        for c in dialog["discarded_conversations"]
    ]
    previous_versions = {(r["conversation"], r["turn_index"]): r for r in dialog["regenerations"]}
    items = []  # (id, conversation, turn, previous, judgement)
    for conv, turns, judgements in convs:
        for n, (t, j) in enumerate(zip(turns, judgements, strict=True)):
            prev = turns[n - 1] if n else None
            items.append((f"c{conv}-t{t['index']}", conv, t, prev, j))
    for s in judgement["superseded_judgements"]:
        conv, n = s["conversation"], s["turn_index"]
        turns = next(ts for c, ts, _ in convs if c == conv)
        old = previous_versions[(conv, n)]["previous"]
        # At judging time the neighbour before was its own old version, if it was replaced too.
        before = previous_versions.get((conv, n - 1))
        prev = before["previous"] if before else (turns[n - 1] if n else None)
        items.append((f"c{conv}-t{n}-v{s['version']}", conv, old, prev, s["judgement"]))
    out = []
    for cid, conv, t, prev, j in items:
        state, _ = build_request(_turn(t), _turn(prev) if prev else None, facts)
        hand = HAND_LABELS.get(cid)
        out.append(
            {
                "id": cid,
                "conversation": conv,
                "speaker": t["speaker"],
                "accepted": j["accepted"],
                "decided_by": j["decided_by"],
                "reason": j["reason"],
                "archived_jev_p": j["jev_p"],
                "complete_by_hand": hand[0] if hand else j["accepted"],
                "hand_note": hand[1] if hand else None,
                "state": state,
            }
        )
    body = {
        "_note": "Turns of the Axolotl run (archive/episodes/2026-11-19_axolotl-2), state as "
        "built by gut_zu_wissen.judge.build_request. Fact sentences and context are text from "
        "the German Wikipedia (CC BY-SA 4.0, see NOTICE). Labels: `accepted` is the run's final "
        "verdict; `complete_by_hand` is Claude's label for rejected turns, otherwise the verdict.",
        "cases": out,
    }
    CASES.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    print(f"{CASES.name}: {len(out)} cases")


def s1_cases() -> list[dict]:
    """S1 turns with their round-2 state and the hand completeness label."""
    labelled = json.loads(S1_CASES.read_text())
    gold = {c["id"]: c for g in ("turns", "holdout_turns") for c in labelled[g]}
    out = []
    for r in json.loads(S1_V2.read_text()):
        if r["kind"] != "turn":
            continue
        c = gold[r["id"]]
        out.append(
            {
                "id": f"s1-{c['id']}",
                "accepted": c["vollstaendig_gold"] and all(a["gold"] for a in c["aussagen"]),
                "complete_by_hand": c["vollstaendig_gold"],
                "category": c["category"],
                "archived_jev_p": {
                    k: a["noul"]
                    for k, a in r["response"]["answers"].items()
                    if k.startswith("aussage_")
                },
                "state": r["request"]["state"],
            }
        )
    return out


# ---------------------------------------------------------------- live run


def reduced(state: dict) -> dict:
    return {k: v for k, v in state.items() if k not in ("fakten", "kontext")}


def post(client: httpx.Client, state: dict, questions: dict) -> dict:
    payload = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(5):
        r = client.post(API, json=payload)
        if r.status_code in (429, 529) and attempt < 4:
            time.sleep(2**attempt)
            continue
        return {"request": payload, "status": r.status_code, "response": r.json()}
    raise RuntimeError("unreachable")


def _items() -> list[tuple[str, dict]]:
    items = [("axolotl", c) for c in json.loads(CASES.read_text())["cases"]]
    return items + [("s1", c) for c in s1_cases()]


def _client() -> httpx.Client:
    key = os.environ["TYPESAFE_API_KEY"]
    return httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)


def run2() -> None:
    with _client() as client:
        for rep in range(REPEATS):
            records = [
                {
                    "group": group,
                    "id": c["id"],
                    "requests": [post(client, c["state"], {k: q() for k, q in ROUND2.items()})],
                }
                for group, c in _items()
            ]
            out = RESULTS / f"r2_{MODEL}_{rep}.json"
            out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
            print(f"{out.name}: {len(records)} cases")


def run() -> None:
    items = [("axolotl", c) for c in json.loads(CASES.read_text())["cases"]]
    items += [("s1", c) for c in s1_cases()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    key = os.environ["TYPESAFE_API_KEY"]
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        for rep in range(REPEATS):
            records = []
            for group, c in items:
                full = post(client, c["state"], {k: q() for k, q in FULL_STATE.items()})
                red = post(client, reduced(c["state"]), {k: q() for k, q in REDUCED_STATE.items()})
                records.append({"group": group, "id": c["id"], "requests": [full, red]})
            out = RESULTS / f"{MODEL}_{rep}.json"
            out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
            print(f"{out.name}: {len(records)} cases")


# ---------------------------------------------------------------- report


def completeness_p(record: dict) -> dict[str, float]:
    """p(complete) per variant; inverted variants are read as 1 - p."""
    ps: dict[str, float] = {}
    for req in record["requests"]:
        assert req["status"] == 200, req
        for k, a in req["response"]["answers"].items():
            ps[k] = round(1 - a["noul"], 4) if k in INVERTED else a["noul"]
    return ps


def report() -> None:
    labels = {c["id"]: c for c in json.loads(CASES.read_text())["cases"]}
    labels |= {c["id"]: c for c in s1_cases()}
    for path in sorted(RESULTS.glob("*.json")):
        records = json.loads(path.read_text())
        variants = list(completeness_p(records[0]))
        tokens = sum(q["response"]["usage"]["input_tokens"] for r in records for q in r["requests"])
        print(f"\n=== {path.stem}  input_tokens={tokens}")
        for group in ("axolotl", "s1"):
            rows = [r for r in records if r["group"] == group]
            print(f"--- {group} (n = {len(rows)})")
            print(
                f"  {'variant':16} {'>=t':>5} {'<=1-t':>6} {'unsure':>6} {'mean p':>6}  "
                "| rejected turn p>=t | incomplete(hand) p>=t | turn false accept"
            )
            for v in variants:
                ps, fa_turn, fa_hand, fa_all = [], [], [], []
                for r in rows:
                    case = labels[r["id"]]
                    p = completeness_p(r)[v]
                    ps.append(p)
                    if p >= T and not case["accepted"]:
                        fa_turn.append(r["id"])
                    if p >= T and not case["complete_by_hand"]:
                        fa_hand.append(r["id"])
                    # Statement questions are unchanged: their p from the run (Axolotl) or
                    # from S1 round 2.
                    stmt = [p for k, p in case["archived_jev_p"].items() if k != COMPLETENESS]
                    if p >= T and all(s >= T for s in stmt) and not case["accepted"]:
                        fa_all.append(r["id"])
                n = len(ps)
                hi = sum(p >= T for p in ps)
                lo = sum(p <= 1 - T for p in ps)
                print(
                    f"  {v:16} {hi / n:5.0%} {lo / n:6.0%} {(n - hi - lo) / n:6.0%} "
                    f"{statistics.mean(ps):6.3f}  | {fa_turn} | {fa_hand} | {fa_all}"
                )
            for v in variants:
                acc = [completeness_p(r)[v] for r in rows if labels[r["id"]]["accepted"]]
                rej = [completeness_p(r)[v] for r in rows if not labels[r["id"]]["accepted"]]
                print(
                    f"  {v:16} accepted turns: mean {statistics.mean(acc):.3f} "
                    f"(>=t {sum(p >= T for p in acc)}/{len(acc)}); rejected turns: "
                    f"{' '.join(f'{p:.2f}' for p in rej)}"
                )


def detail(name: str) -> None:
    """p(complete) per case and variant for one result file."""
    labels = {c["id"]: c for c in json.loads(CASES.read_text())["cases"]}
    labels |= {c["id"]: c for c in s1_cases()}
    for r in json.loads((RESULTS / f"{name}.json").read_text()):
        case = labels[r["id"]]
        mark = "" if case["accepted"] else " REJ"
        mark += "" if case["complete_by_hand"] else " INC"
        cells = " ".join(f"{k}={p:.2f}" for k, p in completeness_p(r).items())
        print(f"{r['id']:12}{mark:9} {cells}")


if __name__ == "__main__":
    {"cases": cases, "run": run, "run2": run2, "report": report}.get(
        sys.argv[1], lambda: detail(sys.argv[2])
    )()
