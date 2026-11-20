# Spike S1: Jev

**Question (plan, Phase 0):** German accuracy on about 30 hand-labelled turn/fact pairs; error code for overlength; latency; how reliable the content-preservation question (D1) is.
**Output:** go/no-go for Jev, complete error table.
**Result:** **Go**, with **Jev 1.14** (`jev-1.14.0`, pinned) and German questions. Across 52 turns × 2 models × 2 repeats, Jev never accepted a wrong turn and never rejected a correct one at t = 0.95. The cost is a high judge rate: 39 % of turns go to the LLM judge. D1 never confirmed changed or lost content, but it confirms only about half of the unchanged turns, so cascades run longer than necessary.

Code: [spikes/s1_jev.py](../../spikes/s1_jev.py), questions in [spikes/s1_questions.py](../../spikes/s1_questions.py). Cases: [tests/fixtures/wikipedia/spikes/s1_cases.json](../../tests/fixtures/wikipedia/spikes/s1_cases.json). Every request payload and response: [tests/fixtures/wikipedia/spikes/s1_results/](../../tests/fixtures/wikipedia/spikes/s1_results/). Run with `jev-1.13.0` and `jev-1.14.0` on 2026-11-15. Total cost: 544k input tokens, about **$0.023**.

> **Labels.** The project owner delegated S1. All turns and labels were written by Claude, not hand-checked. The labels are deliberately unambiguous, but they are not the independent human labels the plan asks for. The first reviewed episodes (step 13, r = 100 %) are the real check.

## Cases

All fact sentences are real segments from the S2 fixtures (7 articles). The turns are written in persona style, with numbers spelled out as Lena speaks.

- **Tuning set:** 36 turns with 39 statements, and 6 D1 pairs.
- **Holdout:** 16 turns and 4 D1 pairs, written before the round-2 variants were run. Round 1 never saw them.
- **Categories:**
  - correct statements;
  - changed number, date, name or ordinal; swapped numbers;
  - negation; overgeneralisation ("attempted" → "reached", "2010s" → "since opening"); added detail or superlative;
  - claim supported only by the context;
  - supported and unsupported question premises;
  - pure reaction; reaction with a hidden claim or evaluation; self-statement ("Das kenne ich …");
  - a claim missing from the statement list;
  - correct arithmetic (2000 + 5000 → "more than 7000").

About half the cases are planted errors. The judge rate on real dialogs will differ.

## Request design (chosen)

One request per turn. The state is shared, so it is billed once.

```json
{
  "redebeitrag": {"sprecher": "Lena", "text": "…"},
  "vorheriger_redebeitrag": "…",            // only when present
  "aussagen": [{"text": "…", "fakten": ["271593273-0100"]}],
  "fakten": {"271593273-0100": {"artikel": "Eiffelturm", "satz": "…"}},
  "kontext": {"271593273-0100": ["…"]}      // only when the fact has context
}
```

- **One Noul per statement:** „Folgt die Aussage „…“ aus `fakten.<id>`?“. The criteria name what does *not* count: a different number, date or name, an added detail, anything only in `kontext`.
- **One completeness Noul:** „Ist jede Sachbehauptung und jede Prämisse einer Frage in `redebeitrag.text` unter `aussagen` erfasst?“. The reaction criterion from [persona.md](../persona.md) goes into the criteria.
- **Article title per fact** (`artikel`): this is something code knows. Without it, "the visitor numbers *of the Eiffel Tower*" was uncertain because the fact sentence never names the tower.

The exact wording is in [spikes/s1_questions.py](../../spikes/s1_questions.py).

## Results

### Statement questions (Jev 1.14, German, round 1, t = 0.95)

| | right | wrong | uncertain |
|---|---:|---:|---:|
| Statements (n = 39) | 31–32 | 0 | 7–8 |
| Completeness (n = 36) | 13–14 | 0 | 22–23 |

- **Planted errors:** in round 1, all changed numbers, dates and names get p = 0.02–0.07, a clear reject.
- **Context-only claims:** p = 0.39–0.72, never accepted. They go to the judge.
- **Arithmetic** (c25, "more than 7000"): p = 0.72, uncertain, as TypeSafe documents for numbers.
- **Completeness:** this is the bottleneck. Pure reactions ("Okay, damit hätte ich nicht gerechnet …") score 0.4–0.7, and Felix's questions with a premise score about 0.65. Both always go to the judge.

### Turn decision (all 52 turns, round 2 state, both repeats)

| Model | t | accepted correctly | wrong accept | rejected correctly | wrong reject | to judge |
|---|---:|---:|---:|---:|---:|---:|
| jev-1.13.0 | 0.95 | 23 | 0 | 32 | 0 | 49 (47 %) |
| jev-1.14.0 | 0.95 | 28 | 0 | 35 | 0 | 41 (39 %) |
| jev-1.14.0 | 0.9 | 34 | 0 | 44 | 0 | 26 (25 %) |
| jev-1.14.0 | 0.8 | 40 | 0 | 50 | 0 | 14 (13 %) |

The holdout set behaves like the tuning set: 5 accepted correctly, 5 rejected correctly, 6 to the judge, no wrong decision. t stays at 0.95 as configured. The lower rows are data for the later recalibration, not a recommendation, since n is too small and the labels are Claude's.

### Variants tried and dropped

| Variant | Result |
|---|---|
| English questions on German state | Judge rate higher than German on both models (1.14: 16–17 vs 13 of 36). |
| Per-sentence completeness (code splits the turn with SoMaJo; per sentence "reine Reaktion?" + "von `aussagen` wiedergegeben?", combined in code) | Worse: 23 of 36 to the judge. Jev rates clearly covered sentences only 0.65–0.95. |

### D1: content preservation (10 pairs, 4 of them holdout)

| Variant | Changed/lost content (5 pairs) | Same content (5 pairs) |
|---|---|---|
| One holistic question „Vermittelt `nachher` dieselben Sachaussagen wie `vorher`?“ | never confirmed | 1 of 5 confirmed |
| **Atomic, both directions (chosen):** per old statement „Vermittelt `nachher` die Aussage „…“?“ and per new statement „Vermittelt `vorher` die Aussage „…“?“; confirmed only if all p ≥ t | never confirmed | 2–3 of 5 confirmed |

The atomic variant uses the explicit statement lists the dialog author outputs for both versions. Under D1, uncertain counts as content lost, so the weakness is safe. It only makes cascades longer and uses up the 2× budget sooner. Cascade length is a step 13 metric.

### Latency and tokens

- **Latency:** median 0.16–0.18 s per request, maximum 0.55 s. Both models are the same.
- **Tokens:** about 800–1,100 input tokens per turn request.
- **Cost:** an episode of about 150 turns plus regenerations costs well under $0.05 in Jev calls.

## Error table (complete)

Measured with real requests ([errors.json](../../tests/fixtures/wikipedia/spikes/s1_results/errors.json)) plus the [TypeSafe error docs](https://docs.typesafe.ai/api/errors.md):

| Case | Response | Reaction (design table) |
|---|---|---|
| Overlength: state + question > 32k tokens, or request > 64k | `400`, `detail.error_type = "max_tokens_exceeded"` | LLM judge |
| Rate limit, overload | `429`, `529` | Retry with backoff, then LLM judge |
| Server error, timeout | `500`, timeout | Retry with backoff, then LLM judge |
| Invalid key | `401`, `error_type = "authentication_error"` | Abort |
| Unknown model | `400`, `error_type = "api_usage_error"` (not 404) | Abort |
| Malformed question | `400`, `detail` as string (not 422) | Abort |
| Validation error | `422` | Abort |
| Body > 32 MiB, unsupported compression | `413`, `415` (documented, not triggered) | Abort (invalid request) |
| `402`, `403`, `404` | not observed; kept from the design | Abort |

Overlength is recognised by `status == 400 and detail.error_type == "max_tokens_exceeded"`. Every other 400 is an invalid request. That resolves the design's open point "bei 422 Trennung von ungültiger Anfrage über Fehlertext": overlength is a 400 with its own error type.

## Decisions taken in this spike

Delegated by the project owner.

1. **Model:** `jev-1.14.0`, pinned by version, not by alias. It sends fewer turns to the judge than 1.13 (39 % vs 47 %), at the same price and latency. Set in [config/pipeline.toml](../../config/pipeline.toml).
2. **Question language:** German.
3. **State:** as above, with an article title per fact and context as a separate field, per fact.
4. **Completeness:** one holistic question per turn, as in the design.
5. **D1:** atomic, in both directions, over the author's statement lists. Confirmed only if every p ≥ t.
6. **Error table:** as above. 413/415 → abort.
7. **Jev price:** `0.042` USD per million input tokens in the config.

## Changes to the design document

Applied to [wikipedia-podcast-pipeline.md](../wikipedia-podcast-pipeline.md) with the project owner's approval:

- **Judge:** Jev 1.13 → Jev 1.14 (`jev-1.14.0`).
- **Error table:** overlength = `400` + `max_tokens_exceeded`; unknown model = `400 api_usage_error`; add `413`/`415` → abort.
- **Open points:** remove "Jev-Überlänge: Fehlercode per Test ermitteln" and "Jev: Überlänge-Fehlercode testen; Verhalten auf Deutsch prüfen". Both are answered.

## Risks this spike confirms or adds

- **Judge rate:** about 40 % at t = 0.95. It's driven by pure reactions and question premises (completeness question) and by context-only and arithmetic claims. The design expected this ("Eskalationsrate: bei t = 0,95 anfangs voraussichtlich hoch").
- **Pure reactions always escalate.** Felix's reactions are frequent in a dialog, so many LLM-judge calls will be for turns with no facts at all.
- **D1 cascades** will often run longer than needed, because Jev rarely confirms paraphrases with certainty.
- **Small n, Claude's labels:** no false accept in 208 turn decisions is encouraging, but it says little about the real false-accept rate. The review samples (r = 100 % at first) remain the check.

## Follow-up 2026-11-20: completeness question on real turns

**Question (user decision 2026-11-20):** can a reworded completeness question stop sending almost every real turn to the LLM judge, while keeping reactions allowed and never accepting a rejected turn?
**Result:** **no switch.** No variant clearly reduces the uncertain share without risk; the production question stays as it is.

Code: [spikes/s1_completeness.py](../../spikes/s1_completeness.py), wording in [spikes/s1_completeness_questions.py](../../spikes/s1_completeness_questions.py). Cases: [s1_completeness_cases.json](../../tests/fixtures/wikipedia/spikes/s1_completeness_cases.json) (131 turns of the Axolotl run `2026-11-19_axolotl-2`: three conversations plus six superseded turn versions; state built by `build_request`). Payloads and responses: [s1_completeness_results/](../../tests/fixtures/wikipedia/spikes/s1_completeness_results/). `jev-1.14.0`, two repeats per round, t = 0.95, about 1.5 M input tokens ($0.06).

Labels: the run's final verdict per turn (121 accepted, 10 rejected); for the rejected turns also a completeness label by Claude. The 52 S1 turns ran again with their round-2 state.

| Variant | Axolotl ≥ t | ≤ 1−t | uncertain | mean p | Axolotl turns to judge | S1 turns to judge | rejected turn ≥ t |
|---|---:|---:|---:|---:|---:|---:|---|
| `old` (production) | 5 % | 0 % | 95 % | 0.729 | 95–96 % | 38 % | none (max 0.91) |
| `content`: "Steht alles, was … an Inhalt mitteilt, auch unter `aussagen`?" | 5 % | 0 % | 95 % | 0.700 | 95 % | 38 % | none |
| `structured`: same, lists of what counts as content | 15 % | 0 % | 85 % | 0.762 | 85–88 % | 40–42 % | none (max 0.94) |
| `adds`: reversed polarity | 0 % | 0 % | 100 % | 0.491 | 99–100 % | 54–56 % | none |
| `old`, state without `fakten`/`kontext` | 3 % | 0 % | 97 % | 0.712 | 97–98 % | 42 % | none |
| `content`, same reduced state | 3 % | 0 % | 97 % | 0.682 | 98 % | 40–44 % | none |
| `facts`: text against `fakten` instead of `aussagen` | 5 % | 0 % | 95 % | 0.734 | 95 % | 33–38 % | none |
| `facts_structured` | 11 % | 0 % | 89 % | 0.788 | 89 % | 37–38 % | **1** (c2-t14-v0, 0.95) |

- **Little signal on real turns.** On the Axolotl turns, completeness p barely separates accepted from rejected turns (AUC 0.58 for `old`, at most 0.70 for any variant). Rejected turns reach p = 0.91–0.95, so lowering t for this question would not help either.
- **Why:** the Felix turns are full of short evaluations ("Das ist ein richtiger Coup", "Das ist doch ein Widerspruch!", "Das habe ich mir gemerkt") that sit on the line between "Gefühl im Moment" and "Wertung mit Faktengehalt" or a self-statement. The LLM judge let them through; Jev stays uncertain. Lena's turns often say more than their statement list.
- **`structured`** is the best variant, but the gain is small (5 % → 15 % at ≥ t on Axolotl, slightly worse on S1), and a rejected turn reaches 0.94. Run-to-run noise is about 0.03 on average and up to 0.14, so a false acceptance is within reach.
