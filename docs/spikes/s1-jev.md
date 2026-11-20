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

## Follow-up 2026-11-20: completeness per sentence

**Question (user decision 2026-11-20):** does asking Jev about each sentence of `redebeitrag.text` ("pure reaction, or covered by `aussagen`?") let Jev decide clearly more turns alone than the one completeness question per turn, with no false acceptance and a safety margin? Measurement only; production code unchanged.
**Result:** **no switch.** The only variant with more Jev-alone decisions (`either`, 12 % → 22 %) accepts three to four turns the LLM judge rejected, and its highest p on a rejected turn is 0.98. The two-question variants decide fewer turns than production.

Code: [spikes/s1_sentences.py](../../spikes/s1_sentences.py), wording in [spikes/s1_sentences_questions.py](../../spikes/s1_sentences_questions.py), unit tests of the pure helpers in `tests/test_spike_s1_sentences.py`. Cases: [s1_sentences_cases.json](../../tests/fixtures/wikipedia/spikes/s1_sentences_cases.json); answers, usage and latency: [s1_sentences_results/](../../tests/fixtures/wikipedia/spikes/s1_sentences_results/) (the exact payload is rebuilt by `payload()` from the case file).

- **Cases (337).** `final`: all 239 final turns of the five completed episodes of 2026-11-20 (Axolotl, Tunguska-Ereignis, Voynich-Manuskript, Bernstein-2, Helgoland). `rejected`: all 46 turn versions the judge rejected (all by the LLM judge; gate rejections left out) in every archived folder with a judgement: superseded judgements, discarded conversations and final turns of failed runs. `s1`: the 52 hand-labelled S1 turns with their round-2 state. State and statement questions from `build_request`; facts: every fact extracted in the run (main, linked, Einordnung), because discarded conversations had their own offers. The neighbour turn of a superseded version is approximated (the archive does not record it).
- **Labels:** the final verdict of each turn version (LLM judge, or Jev where it decided alone); S1: the hand labels.
- **Sentences:** SoMaJo (`de_CMC`, the segmentation tokenizer), original text kept; 2.6 sentences per turn on average, at most 12.
- **Requests:** one request per turn and variant: the production statement questions plus the variant's completeness questions, production state. Turn decision: `judge.decide` over statement p and sentence p (complete if every sentence ≥ t, incomplete if any ≤ 1 − t, otherwise uncertain). `jev-1.14.0`, t = 0.95, two repeats, six requests in parallel.

Wordings (true = the sentence is fine; reaction criterion from `docs/persona.md`, sentence named in the question):

- `either`: one Noul per sentence, „Ist der Satz „…“ aus `redebeitrag.text` eine reine Reaktion oder vollständig unter `aussagen` erfasst?“
- `split`: two Nouls per sentence, „… eine reine Reaktion ohne eigenen Inhalt?“ and „Steht alles, was der Satz „…“ an Inhalt mitteilt, in einer Aussage unter `aussagen`?“; sentence p = max of the two.
- `split_ctx`: `split` with the sentence before and after in structured instructions (`satz_davor`, `satz_danach`).

All variants count a question's premise as content.

| Variant | final: Jev alone | final: uncertain | rejected: Jev accepts (false acceptance) | max completeness p on a rejected turn | false rejections | S1: Jev alone | questions / request | input tokens / request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `old` (production) | 12–13 % | 87–88 % | 0 | 0.93 | 0 | 58–60 % | 2.5 | 1,143 |
| `either` | 21–22 % | 78–79 % | **3–4** | **0.98** | 0 | 69–73 % | 4.5 | 1,919 |
| `split` | 9–10 % | 90–91 % | 0 | 0.91 | 0 | 52–58 % | 7.4 | 2,243 |
| `split_ctx` | 12–13 % | 87–88 % | 0 | 0.94 | 0 | 54 % | 7.4 | 2,558 |

Ranges are the two repeats. Jev never rejected a turn of the `final` and `rejected` groups alone (no p ≤ 1 − t); every decision it took alone there was an acceptance. False rejections: none in any variant.

**False acceptances of `either`** (statements and every sentence ≥ t; sentence p of both repeats):

| Case | LLM judge's reason (short) | lowest sentence p | `old` completeness p |
|---|---|---|---|
| `2026-11-20_tunguska-ereignis/c1-t21-v0` | "Kometen" is not named in the cited fact sentence | 0.98 / 0.98 | 0.90 / 0.92 |
| `2026-11-20_bernstein/c3-t54` | "Tagebau Goitzsche" is in no fact sentence | 0.97 / 0.96 | 0.79 / 0.83 |
| `2026-11-20_tunguska-ereignis-2/c3-t29` | "hält für möglich" turns a theory about the event into a general possibility | 0.96 / 0.96 | 0.88 / 0.89 |
| `2026-11-20_bernstein/c1-t45` (repeat 2 only) | "leicht zu erreichen" goes beyond "leicht zu fördern" | 0.94 / 0.96 | 0.93 / 0.93 |

The first three are not completeness errors: the text matches its statements, but the statements do not follow from the fact sentences, and the statement questions still answered 0.98–0.99. Production catches them only because its completeness question stays uncertain. So the per-turn completeness question also works as a safety net for statement-question misses, and any completeness question that reaches ≥ t more often lets these through.

**Borderline rejected turns** (highest completeness p, both repeats): `split_ctx`: Tunguska c1-t21-v0 0.88/0.94, S1 c35 0.89/0.93, S1 c30 0.93/0.85; `split`: S1 c35 0.92/0.93, Tunguska c1-t21-v0 0.89/0.91, Bernstein-2 c1-t50 0.91/0.91; `old`: S1 h07 0.94/0.93, S1 c35 0.94/0.93, Bernstein c1-t45 0.93/0.93.

- **Pure reactions** (45 final turns without statements): no variant helps; Jev decides 0–2 of them alone in every variant. The `either` gain comes from Lena turns whose sentences each match a statement.
- **The two-question variants** split the probability: `reaction` is high for reactions, `covered` for content sentences, but sentences that mix both ("Selbstheilungsfähigkeiten, das klingt fast nach einem Superhelden.") stay at 0.5–0.8 on both. The neighbour sentences change little.
- **Latency:** one request per turn in every variant (questions run in parallel inside the request), about 0.17 s per request for all variants (six in parallel). Per-sentence questions add tokens, not calls: 1.7–2.2 × the input tokens of production.
- **Cost:** 4.94 M input tokens for both repeats (2,696 requests), $0.21 at $0.042 per million; plus about 30 k tokens for a ten-case probe.

**Recommendation:** do not switch. `either` gives more Jev-alone decisions but fails both conditions (false acceptances, no safety margin); `split` and `split_ctx` give no gain at 2 × the tokens. The finding that the completeness question hides statement-question misses matters for any later attempt: a better completeness question would need a stronger statement check first. Three of the four misses are fact sentences whose subject or place stands only in `kontext` or nowhere; how to check that is open.
