# Spike S7: Kosten

**Question (user decision 2026-11-20):** which of five levers lowers the cost per episode without losing fact fidelity? The levers are lower effort, prompt caching, Claude Haiku 5.5 for extraction and ranking, short reasons on acceptance, and a higher Jev share with a safety net. This spike only measures; production code is unchanged, and every switch is the owner's decision.
**Output:** the saving per lever in $ and %, its quality effect, and a recommendation; a combined estimate; risks.
**Result:** only **prompt caching** saves money without any quality effect: **13–20 % per episode**, Helgoland **$1.55 → about $1.28**. **Haiku 5.5 for extraction** (ranking stays on Sonnet) saves another **~27 %** (Helgoland about **$1.01** with caching), but the offer moves a bit further from today's than two Sonnet runs differ from each other. That trade-off is the owner's call. Lower effort saves 1–4 %, but it doubles the LLM judge's false acceptances; short reasons save about 3 % but may raise false rejections. A higher Jev share saves about 1 % at a margin of 0.01–0.02. Neither is recommended.

Code: [spikes/s7_kosten.py](../../spikes/s7_kosten.py), Jev wording in [spikes/s7_questions.py](../../spikes/s7_questions.py), unit tests of the pure helpers in `tests/test_spike_s7_kosten.py`. Every answer, usage record and cost: [tests/fixtures/wikipedia/spikes/s7_results/](../../tests/fixtures/wikipedia/spikes/s7_results/) (CC BY-SA, see NOTICE). Judge cases: the 337 cases of the S1 follow-up ([s1_sentences_cases.json](../../tests/fixtures/wikipedia/spikes/s1_sentences_cases.json)). Run on 2026-11-20 with `claude-sonnet-5-5`, `claude-haiku-5-5` and `jev-1.14.0`. **Total spend: $4.84** (budget $6): $4.79 Claude, $0.04 Jev.

## Method

- **Same requests as production.** The production functions `extract_facts`, `rank_facts`, `rank_linked_facts` and `format_judge_input` build every request. The spike runs them twice. In the first pass, a fake `StructuredLlm` (`Collect`) records the requests. They go to the Message Batches API, and in the second pass `Replay` hands the answers back in order, checking a hash of each request. Extraction, ranking and judge calls do not depend on each other's answers, so the replay gives the run the pipeline would make. Each request carries the parameters `AnthropicLlm.complete_json` sends: model, `max_tokens` 16000, system prompt, one user message and `output_config.format`. A variant adds `output_config.effort`. Without effort, the model default applies: `high` on Sonnet 5.5, `medium` on Haiku 5.5. Production sets no effort, so it runs at `high`.
- **Batches at half price.** The batch price only lowers the spike's bill. Every cost below is at **list price**, as the pipeline pays.
- **Baselines.** The 2026-11-20 runs of Helgoland and Bernstein-2 used today's extraction prompt. Tunguska and Voynich ran before the prompt changed (10:44), so their archived extraction is not comparable. For all three articles the spike therefore re-ran the production setting ("default rerun"). Ranking ran twice at the default to measure the run-to-run noise. The judge default ran once on all 337 cases.
- **Extraction metrics.** Facts: the segments that pass all gates (rendered with the production `ParseRenderer`). Linked intros: the intros archived in the run (the link targets depend on the ranking, so they were kept fixed). *Recall of offer:* the share of the archived offer (main and linked facts in `04_ranking.json`) that the variant still extracts.
- **Ranking metrics.** Ranking runs on the archived candidates with the archived exclusions (Einordnung). Jaccard of the chosen main facts against the archive and against the default rerun. *Recall:* the share of the archived main offer that the variant chooses.
- **Judge labels.** The archived final verdict per turn version; for the S1 turns, the hand labels. A false acceptance is a supported verdict on one of the 46 turns the archive rejected.
- **Caching** cannot be measured in a batch. The spike replayed real call inputs of `2026-11-20_bernstein-2` one after another: 20 extraction chunks, the 8 regenerations of the final conversation and the 56 LLM-judge calls of its final turns. It used `cache_control` breakpoints and `max_tokens` 1 (caching changes only the input side), and recorded `cache_read_input_tokens`.

## Where the money goes today

Per call kind at list price, measured on Helgoland ($1.55 in total):

| Part | $ | Share | Mostly |
|---|---:|---:|---|
| Extraction, 95 calls (65 chunks + 30 linked intros) | 0.51 | 33 % | **input** (183k tokens); 155 output tokens per call, **no thinking** |
| Ranking, 2 calls | 0.14 | 9 % | input (63k tokens); about 750 thinking tokens |
| Dialog: write + 4 regenerations (author minus the two above) | 0.41 | 26 % | output (about 30k tokens) |
| LLM judge, 52 calls | 0.30 | 19 % | 1.0–1.1k input, 230–300 output per call, of which 60–120 thinking |
| TTS | 0.19 | 12 % | |

The brief assumed that output tokens dominate the author role because thinking counts as output. That holds only for the dialog author. At the default effort Sonnet 5.5 **does not think at all** during extraction and thinks little during ranking. Both are input-bound, so effort cannot save much there.

## Results per lever

### Overview

| Lever | Saving per episode | Quality effect | Recommendation |
|---|---|---|---|
| 1 Effort `low` for extraction + ranking | Helgoland $0.024 (1.5 %) | extraction recall of the offer 0.98 → 0.91; ranking: fewer facts (37 → 29) | no |
| 1 Effort `low`/`medium` for the LLM judge | $0.04 Helgoland, $0.13 Bernstein-2 (2.5–4 %) | **false acceptances 13 → 22–23 of 46**; false rejections 5 → 13 of 239 | no |
| 2 Prompt caching | **$0.18–0.60 (13–20 %)**, Helgoland $0.27 | none (same request bytes) | **yes** |
| 3 Haiku 5.5 for extraction, Sonnet ranks | Helgoland $0.42 (27 %) | offer J 0.41 against the archive (Sonnet-vs-Sonnet noise 0.48–0.54) | owner's decision |
| 3 Haiku 5.5 for ranking | Helgoland $0.13 (8 %) | offer J 0.17–0.24, prefers climate and administrative facts | no |
| 4 Short reasons on acceptance | $0.04 Helgoland, $0.15 Bernstein-2 (3–5 %) | false acceptances 12 (noise level 13); **false rejections 5 → 12 of 239** | not now |
| 5 Jev alone with `terms` and completeness ≥ 0.90 | about $0.015 (1 %) | 0 false acceptances, margin −0.01 to −0.02 | no |

### Lever 1: lower effort

**Extraction** (production setting = `sonnet_default`):

| Run | output tok | thinking tok | list $ | facts | J (default rerun) | J (archive) | recall of offer | default facts found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tunguska_sonnet_default | 4,480 | 0 | 0.190 | 137 | 1.00 | (old prompt) | 0.86 | 1.00 |
| tunguska_sonnet_low | 4,033 | 0 | 0.186 | 120 | 0.85 | | 0.83 | 0.86 |
| voynich_sonnet_default | 6,823 | 0 | 0.256 | 188 | 1.00 | (old prompt) | 0.91 | 1.00 |
| voynich_sonnet_low | 6,443 | 0 | 0.252 | 178 | 0.85 | | 0.82 | 0.89 |
| helgoland_sonnet_default | 14,689 | 0 | 0.514 | 449 | 1.00 | **0.84** | 0.98 | 1.00 |
| helgoland_sonnet_medium | 13,495 | 0 | 0.502 | 410 | 0.83 | 0.80 | 0.91 | 0.87 |
| helgoland_sonnet_low | 13,102 | 0 | 0.498 | 406 | 0.82 | 0.80 | 0.91 | 0.86 |

- **Noise:** the default rerun matches the archived Helgoland extraction at J 0.84; that is the floor for this measure.
- **Low** cuts output by 6–11 % but saves only 1–3 % of the extraction cost, because input dominates. It extracts 5–12 % fewer facts and misses 3–9 % of the archived offer.

**Ranking** on the archived candidates:

| Run | output tok | thinking tok | list $ | main n | J main (archive) | J main (default rerun) | recall archived main offer |
|---|---:|---:|---:|---:|---:|---:|---:|
| tunguska default / rep2 | 1,030 / 1,061 | 799 / 830 | 0.033 | 22 / 22 | 0.91 / 0.91 | 0.91 | 0.95 / 0.95 |
| tunguska low | 210 | 0 | 0.025 | 20 | 0.91 | 0.83 | 0.91 |
| voynich default / rep2 | 771 / 711 | 512 / 473 | 0.042 | 30 / 30 | 0.71 / 0.71 | 0.71 | 0.83 / 0.83 |
| voynich low | 238 | 0 | 0.036 | 25 | 0.77 | 0.67 | 0.80 |
| helgoland default / rep2 | 1,101 / 851 | 744 / 501 | 0.137 | 37 / 37 | 0.54 / 0.48 | **0.48** | 0.70 / 0.65 |
| helgoland medium | 322 | 0 | 0.130 | 35 | 0.44 | 0.67 | 0.59 |
| helgoland low | 280 | 0 | 0.130 | 29 | 0.50 | 0.47 | 0.59 |

- **The ranking is noisy.** Choosing 37 of 367 Helgoland candidates, two runs at the same setting share only J 0.48. Low and medium stay within that noise in overlap. On Helgoland they pick fewer facts (29–35 instead of 37) and recall the archived offer slightly less (0.59 against 0.65–0.70).
- **The saving is tiny:** $0.008 on Helgoland, because ranking is input-bound.

**LLM judge** (verdict prompt, 337 cases; the "rejected" turns are the 46 the archive rejected):

| Run | final: false rejections (of 239) | rejected: **false acceptances** (of 46) | S1: FA / FR (of 28 / 24) | output tok/call | thinking tok/call | list $/call (weighted over the 337) |
|---|---:|---:|---:|---:|---:|---:|
| default rerun (production) | **5** | **13** | 3 / 2 | 255 | 78 | 0.00473 |
| medium | 13 | **22** | 4 / 2 | 178 | 0 | 0.00395 |
| low | 13 | **23** | 4 / 3 | 171 | 0 | 0.00389 |

- **The judge is not stable on its own rejections.** The default rerun accepts 13 of the 46 archived rejections (28 %). It also rejects 5 of the 239 accepted final turns (2 %). So the archived labels on borderline turns are themselves only one draw.
- **Medium and low do not think at all** and accept 22–23 of the 46. They include 10 of the 13 that the default rerun also accepts, plus 12–13 new ones, among them the two known S1 misses (Tunguska c1-t21-v0 „Kometen“ and, at low, Bernstein c3-t54 „Tagebau Goitzsche“). That is a real loss of strictness, not noise.
- **The saving** is about $0.0008 per call: $0.04 on Helgoland (46 calls) and $0.13 on Bernstein-2 (153 calls).

**Explicit check and STT decision:** not measured. The explicit check is one call per episode (about $0.02, S6), and the STT decision is at most one small call per audio section. The saving would be below one cent.

**Recommendation:** keep the default effort everywhere. Lower effort saves 1–4 % in total and costs fidelity in the judge.

### Lever 2: prompt caching

The minimum cacheable prefix for Sonnet 5.5 and Haiku 5.5 is 512 tokens (claude-api skill). Cache reads cost $0.10 per million tokens on Sonnet 5.5 (0.05 × input) and writes $2.50 (1.25 ×, 5-minute TTL).

| Call kind | stable prefix | breakpoint | replayed calls | input tok/call | cached tok/call | cached share |
|---|---|---|---:|---:|---:|---:|
| Extraction | system prompt (one for all chunks and linked intros) | system | 20 | 2,561 | 872 | 34 % |
| Regeneration | `TURN_PROMPT` + „Angebot:“ + offer | system + end of the offer in the user text | 8 | 9,194 | 8,195 | 89 % |
| LLM judge (verdict) | `VERDICT_PROMPT` | system | 56 | 1,040 | 797 | 77 % |

The cached token counts include about 210 tokens of structured-output schema that the API puts in front of the system prompt. Every call after the first hit the cache, on all three kinds.

Not cacheable or no reuse: ranking (one call per list), the dialog write call (once per conversation, different system prompt from the regeneration), the cascade prompt (289 tokens plus schema, at the limit) and Jev.

**Saving per episode** (archived call counts, Sonnet list prices, every call inside the 5-minute TTL; one write per run for extraction and judge, one per conversation for the offer; offer prefix by `count_tokens` per episode):

| Episode | total $ | extraction calls | regenerations | conversations | LLM-judge calls | saving: extraction | judge | regeneration | **sum** | **%** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Axolotl | 1.66 | 47 | 4 | 2 | 80 | 0.076 | 0.119 | 0.019 | **0.21** | **13 %** |
| Tunguska | 1.43 | 43 | 6 | 1 | 41 | 0.069 | 0.060 | 0.068 | **0.20** | **14 %** |
| Voynich | 1.22 | 61 | 2 | 1 | 49 | 0.099 | 0.072 | 0.013 | **0.18** | **15 %** |
| Bernstein-2 | 3.07 | 91 | 18 | 3 | 153 | 0.149 | 0.230 | 0.221 | **0.60** | **20 %** |
| Helgoland | 1.55 | 95 | 4 | 1 | 46 | 0.155 | 0.068 | 0.048 | **0.27** | **18 %** |

**What caching would need in the code** (not implemented):

1. `StructuredLlm.complete_json` needs a way to name the stable prefix, for example a `cache_prefix` argument. `AnthropicLlm` would then send the system prompt as a text block with `cache_control`, and a user prefix as its own block with `cache_control`. All fakes gain the argument.
2. `regenerate_turn` and `continue_dialog` pass `Angebot:\n{offer}\n\n` as the prefix. It already comes first and is deterministic (`format_offer`); keep it that way. Nothing that varies (turn index, reason, timestamps) may go before it.
3. **The cost log must learn cache tokens.** `usage.input_tokens` excludes cached tokens, so `Usage` has to record `cache_read_input_tokens` and `cache_creation_input_tokens`, and `[prices]` needs cache read and write prices. Without that, `episode.json` `cost` would be wrong.
4. Optional, worth little: the write, regeneration and continuation calls have three different system prompts, so they cannot share the offer cache. Putting the offer before the task-specific text would save about one cache write per conversation plus the continuation's offer.
5. **TTL:** all estimates assume each call follows the previous one of its kind within 5 minutes. Extraction chunks follow each other in seconds, and judge calls do too. A regeneration can come several minutes after the previous one (judging a whole 50-turn dialog takes 3–5 minutes); a miss there costs one extra write ($0.02). The replay ran back to back, so its hit rate is an upper bound. The 1-hour TTL (2 × write) would only pay off if real runs show regular gaps over 5 minutes.

**Recommendation:** implement it, with the cost-log change. It saves 13–20 % and does not change a single byte the model reads.

### Lever 3: Claude Haiku 5.5 for extraction and ranking (comparison only)

`claude-haiku-5-5` costs $0.10 / $0.50 per million tokens (Sonnet 5.5: $2 / $10).

**Extraction:**

| Run | output tok | thinking tok | list $ | facts | J (Sonnet default) | share also in Sonnet | Sonnet facts found | recall of offer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tunguska_haiku_default | 29,643 | 22,713 | 0.022 | 233 | 0.54 | 0.56 | 0.95 | 1.00 |
| tunguska_haiku_low | 16,116 | 9,854 | 0.015 | 212 | 0.57 | 0.60 | 0.93 | 0.97 |
| voynich_haiku_default | 43,389 | 33,672 | 0.031 | 307 | 0.54 | 0.56 | 0.92 | 0.91 |
| voynich_haiku_low | 22,435 | 13,874 | 0.021 | 273 | 0.57 | 0.61 | 0.89 | 0.91 |
| helgoland_haiku_default | 82,586 | 61,152 | 0.060 | 714 | 0.52 | 0.56 | 0.89 | 0.93 |
| helgoland_haiku_low | 48,796 | 30,202 | 0.043 | 617 | 0.57 | 0.63 | 0.86 | 0.98 |

- **Haiku finds most of what Sonnet finds.** It finds 86–95 % of Sonnet's facts and 91–100 % of the archived offer.
- **It adds 40–70 % more facts.** Unlike Sonnet, it thinks a lot (640 tokens per call at its default), and it is still 8–12 × cheaper.
- **Haiku only** (Helgoland, random sample): definitions and inventory facts, such as the long definition of Küstenmeer under the Seerechtsübereinkommen, „Die Häuser im Oberland beginnen mit 301 … und enden bei 762“ and „Das Meerwasser wird auf Helgoland für Kuranwendungen genutzt.“
- **Sonnet only** (sample): „Bis heute ist der 1. März auf Helgoland ein Feiertag.“, „Die Felsformationen der Insel Helgoland wurden 2006 in die Liste der 77 ausgezeichneten Nationalen Geotope aufgenommen.“ Two facts of the archived offer that Haiku does not extract: „Aus Gründen des Protokolls betrat Kaiser Wilhelm II. die Insel erst am 10. August …“ and „Durch die enorme Menge an Sprengstoff erschien es zudem möglich, die gesamte Insel vollständig zu vernichten …“.

**Ranking and the end-to-end offer:**

| Run | list $ | main n | J main (archive) | recall archived main offer | Sonnet noise (J, recall) |
|---|---:|---:|---:|---:|---|
| helgoland_haiku_default (Haiku ranks archived candidates) | 0.007 | 40 | 0.24 | 0.41 | 0.48–0.54, 0.65–0.70 |
| chain helgoland: Haiku extracts, Haiku ranks | 0.009 | 40 | 0.13 | 0.24 | |
| chain helgoland: **Haiku extracts, Sonnet ranks** | 0.169 | 39 | **0.41** | **0.59** | |
| voynich_haiku_default | 0.003 | 26 | 0.47 | 0.60 | 0.71, 0.83 |
| chain voynich: Haiku extracts, Sonnet ranks | 0.091 | 27 | 0.42 | 0.57 | |
| tunguska_haiku_default | 0.002 | 23 | 0.80 | 0.91 | 0.91, 0.95 |
| chain tunguska: Haiku extracts, Sonnet ranks | 0.084 | 24 | 0.48 | 0.68 | |

(For Tunguska and Voynich, the chain compares with an offer the archive built from the older, smaller extraction, so their overlap is lower for any new extraction.)

- **Haiku ranks by a different taste.** Its Helgoland top 15, none chosen by any Sonnet run, includes the winter climate (6.4 °C), the hours of sunshine, the temperature extremes, the name etymology and the 1544 land division.
- **All three Sonnet runs choose, but Haiku leaves out:** the two pedestrian lights on the Düne that warn of landing aircraft, Hoffmann von Fallersleben writing the „Lied der Deutschen“ on Helgoland, Heisenberg retreating to the island because of his hay fever, the 1950 occupation by two Heidelberg students, the price of houses rising tenfold between 1807 and 1810, and Gunther von Hagens as an assistant doctor in the island hospital. For a podcast about surprise, that is the wrong taste.
- **Haiku extraction with Sonnet ranking** keeps the Sonnet taste. Its offer is a little further from the archive (J 0.41, recall 0.59) than two Sonnet runs are from each other (J 0.48–0.54, recall 0.65–0.70). The larger candidate list raises the ranking input, from $0.14 to $0.17 on Helgoland.

**Saving on Helgoland:** extraction + ranking $0.65 → $0.23 with Haiku extraction and Sonnet ranking (**−$0.42, 27 % of the episode**). Haiku for both: $0.07 (−$0.58), but at a clear loss of surprise.

**Recommendation:** Haiku for ranking: no. Haiku for extraction is the owner's decision. The saving is large, and the quality loss on the offer is measurable but small. Whether listeners would notice is not measured; a test episode with a review would show it.

### Lever 4: short reasons on acceptance

Prompt variant (`short_prompt`): „Begründe kurz auf Deutsch; nenne bei Ablehnung die betroffene Stelle.“ becomes „Begründe auf Deutsch. Ist „supported“ wahr, genügt eine Zeile mit höchstens zehn Wörtern. Ist „supported“ falsch, begründe vollständig und nenne die betroffene Stelle.“ The run used the default effort, on all 337 cases.

| Run | final: false rejections (of 239) | rejected: false acceptances (of 46) | S1 FA / FR | output tok per accepted verdict | per rejected | list $/call (weighted) |
|---|---:|---:|---:|---:|---:|---:|
| default (production prompt) | 5 | 13 | 3 / 2 | 231 | 409 | 0.00473 |
| short reasons | **12** | 12 | 2 / 1 | **103** | 320 | 0.00377 |

- **Output on acceptance drops 55 %:** reasons shrink from about 320 to 83 characters, and thinking drops a little too. Example: „Nur Staunen und Nachfrage, keine Behauptung, daher belegfrei.“
- **False acceptances** stay at the noise level (12 against 13). 5 of the 12 are among the 13 the default rerun accepts.
- **False rejections** rise from 5 to 12 of 239. With one repeat each, that may be noise or a real shift. Each false rejection costs a regeneration, a judge call and a cascade step (about $0.05–0.07). At +3 percentage points that is about 1.4 extra regenerations on Helgoland, which would eat the saving.
- **Saving:** $0.001 per call, $0.04 on Helgoland and $0.15 on Bernstein-2 (3–5 %).

**Recommendation:** do not switch now. If the owner wants it, run a second repeat of both variants on the 239 final turns first (about $1.10 at list price, $0.55 as a batch) to see whether the false rejections are noise.

### Lever 5: a higher Jev share with a safety net

**New questions** (written with the typesafe skill, [s7_questions.py](../../spikes/s7_questions.py); production state of `build_request`, two repeats, `jev-1.14.0`):

- `entities`: one Noul per statement, close to the user's wording: „Kommt jeder Name, jeder Ort und jede Zahl der Aussage „…“ in `fakten.<id>.satz` selbst vor?“
- `terms`: code does the exact lookup (capitalised words and numbers of the statement, stem match against the fact sentence). Only the terms code cannot find go to Jev, one Noul per term: „Nennt `fakten.<id>.satz` „Kometen“ aus der Aussage „…“?“ The true criterion allows another word form or a word meaning exactly the same; the false criterion names a broader term, a related term, or the term standing only in `kontext`. This follows the skill: arithmetic and exact lookups in code, one judgment per question. The TypeSafe jaggedness page lists finding occurrences of a term as a weak spot, which is what `entities` asks for.

The 337 cases got 495 `entities` questions and 182 `terms` questions; 123 cases had at least one term code could not find. The request also contains the production statement questions and the production completeness question, so every rule is computed from p values of the same request.

**Rule:** accept by Jev alone if every statement p ≥ t (0.95), every extra-question p ≥ t, and completeness ≥ c; reject as in production (any statement or completeness p ≤ 1 − t); otherwise the LLM judge decides. The new questions never reject. *Max margin* is the highest value, over all not-accepted turns, of the smallest distance of any p above its threshold. Below 0 means no false acceptance; the closer to 0, the thinner the safety. Values: repeat 1 / repeat 2.

| Extra question | c | final: Jev alone | false acceptances (of 46 rejected) | S1 not accepted: FA (of 28) | max margin |
|---|---:|---:|---:|---:|---:|
| none (production at c = 0.95) | 0.95 | 13 % / 13 % | 0 / 0 | 0 / 0 | −0.02 / −0.03 |
| none | 0.90 | 26 % / 27 % | 3 / 3 | 0 / 0 | +0.03 / +0.02 |
| none | 0.85 | 42 % / 41 % | 8 / 8 | 0 / 0 | +0.04 / +0.04 |
| none | 0.80 | 56 % / 56 % | 19 / 18 | 0 / 0 | +0.08 / +0.09 |
| none | 0.75 | 65 % / 69 % | 21 / 21 | 0 / 0 | +0.13 / +0.14 |
| entities | 0.95 | 4 % / 3 % | 0 / 0 | 0 / 0 | −0.02 / −0.03 |
| entities | 0.90 | 10 % / 10 % | 1 / 1 | 0 / 0 | +0.00 / +0.00 |
| entities | 0.85 | 13 % / 13 % | 5 / 5 | 0 / 0 | +0.03 / +0.04 |
| entities | 0.80 | 18 % / 17 % | 9 / 9 | 0 / 0 | +0.08 / +0.09 |
| entities | 0.75 | 22 % / 25 % | 9 / 9 | 0 / 0 | +0.13 / +0.14 |
| terms | 0.95 | 9 % / 9 % | 0 / 0 | 0 / 0 | −0.06 / −0.06 |
| **terms** | **0.90** | **19 % / 20 %** | **0 / 0** | **0 / 0** | **−0.02 / −0.01** |
| terms | 0.85 | 30 % / 29 % | 3 / 3 | 0 / 0 | +0.03 / +0.04 |
| terms | 0.80 | 41 % / 40 % | 9 / 9 | 0 / 0 | +0.08 / +0.09 |
| terms | 0.75 | 47 % / 52 % | 9 / 9 | 0 / 0 | +0.13 / +0.14 |
| both | 0.90 | 10 % / 10 % | 0 / 0 | 0 / 0 | −0.02 / −0.01 |
| both | 0.85 | 13 % / 13 % | 3 / 3 | 0 / 0 | +0.03 / +0.04 |
| both | 0.75 | 22 % / 24 % | 7 / 7 | 0 / 0 | +0.13 / +0.14 |

- **`entities`** rarely reaches 0.95 even on good statements, so it mostly blocks acceptances. Asking about a whole set („jeder Name …“) is the question Jev is not built for.
- **`terms`** catches the „Kometen“ miss: the term question answers 0.57 / 0.49, while the statement question says 0.99. At c = 0.90 it raises the Jev-alone share from 13 % to 19–20 % with no false acceptance in either repeat. The margin is only 0.01–0.02, and S1 run-to-run noise is about 0.03 on average and up to 0.14. At c = 0.85 the false acceptances are Axolotl-2 c3-t34, Bernstein c1-t22 and Bernstein-2 c1-t50 (or c1-t6-v0); the first two are among the 13 that the LLM judge itself flips on a rerun.
- **Saving:** 6–7 % more Jev-alone turns are about 3 LLM-judge calls fewer on a Helgoland-sized dialog, about $0.015 per episode (1 %). With caching, the judge call gets cheaper still. Jev cost rises negligibly (a few hundred tokens per request at $0.042 per million).
- **Finding: the article title.** Both known S1 misses („Kometen“ in Tunguska c1-t21-v0, „Tagebau Goitzsche“ in Bernstein c3-t54) name exactly the **title of the linked article** the fact comes from („Komet“, „Tagebau Goitzsche“). Jev sees the title in `fakten.<id>.artikel`, so its statement question says 0.98–0.99. The LLM judge never sees the title (`format_judge_input` omits it), so it rejects. The two judges disagree about what counts as evidence, not about the text.

**Recommendation:** do not switch. About 1 % saving at a margin of 0.01–0.02 is not worth a risk to fact fidelity. Keep `terms` (code lookup + one Noul per missing term) as the safety net to use if the completeness threshold is ever lowered.

## Combined estimate

The recommended set is **prompt caching only**:

| Episode | today | with caching |
|---|---:|---:|
| Axolotl | $1.66 | about $1.45 |
| Tunguska | $1.43 | about $1.23 |
| Voynich | $1.22 | about $1.04 |
| Bernstein-2 | $3.07 | about $2.47 |
| **Helgoland** | **$1.55** | **about $1.28** |

**If the owner also chooses Haiku 5.5 for extraction** (Sonnet keeps the ranking): Helgoland **$1.55 → about $1.01** (−35 %). The breakdown is $1.55 − $0.45 (extraction $0.51 → $0.06) + $0.03 (larger ranking input) − $0.07 (judge cache) − $0.05 (offer cache). The extraction cache saving no longer applies once extraction runs on Haiku.

## Risks

- **The LLM judge is not reproducible on borderline turns.** At production settings, it accepts 13 of 46 previously rejected turns on a rerun, and rejects 5 of 239 previously accepted ones. This is independent of cost, but it matters for every comparison in this spike and for fact fidelity in general: a turn the judge rejects once might pass on a regeneration with nearly the same text.
- **Caching TTL:** the replay ran back to back. Real runs with gaps over 5 minutes lose hits and pay extra writes; the cost log will show it once cache tokens are recorded.
- **Cost log:** caching without recording cache tokens would make `episode.json` `cost` wrong (too low input, missing writes).
- **Haiku extraction:** the offer shifts slightly. Linked intros were fixed in this spike; with Haiku the link targets (from the most surprising facts) would shift too. Also not measured: how many fewer facts the dialog author can use (D5) and how the episodes sound.
- **Small samples.** Three articles for extraction and ranking, one or two repeats per variant, one regeneration-heavy episode for caching.
- **Labels are one draw** (see the first risk). False acceptances are counted against the archive. A turn the default rerun also accepts is probably borderline rather than wrong.

## Open for the owner

1. **Prompt caching:** implement it, including cache tokens in `Usage` and cache prices in `[prices]`? (Recommended.)
2. **Haiku 5.5 for extraction:** switch (−27 %, offer slightly further from today's), or test it on one episode with a full review first?
3. **Article title as evidence:** should a statement that names the linked article's subject („Kometen“ for a fact from „Komet“) count as supported? Today Jev says yes (it sees `artikel`) and the LLM judge says no (it does not). Either the LLM judge gets the title, or Jev's state drops it.
4. **Short reasons:** run a second repeat to settle the false-rejection question (about $0.55 as a batch), or drop the lever?
5. **Not in scope, but the biggest remaining output item:** the dialog author (write and regenerations, about 26 % of Helgoland). Measuring effort there needs a quality check of whole dialogs (judge pass rate, review), not only token counts.
6. **Message Batches for extraction (an observation, outside the five levers):** extraction calls are independent of each other. As a batch they would cost half (Helgoland −$0.26 without caching). The batches in this spike took about 2–10 minutes each.

## Nachtrag: Extraktion als Batch

**Question (user decision 2026-11-20, measure only):** is the Message Batches API worth it for extraction after the switch to Claude Haiku 5.5, given the run time and the 90-minute GitHub Actions timeout?

**Method.** The main-article extraction of Helgoland (65 chunks) and Voynich (45 chunks) from `02_segments.json`, with the production setting: `claude-haiku-5-5`, the production request builder (`request_params`, system prompt with a `cache_control` breakpoint), no effort (Haiku default `medium`). One synchronous run per article as the pipeline makes it (streamed `AnthropicLlm`, one call after the other), then three batch repeats; in each repeat the two articles were submitted at the same time. Wall time = submit until polling (every 10 s) sees the batch ended; the API's own `created_at` → `ended_at` is given in brackets. Overlap = Jaccard of the facts that pass all gates (`extract_facts` with the production renderer). Run on 2026-11-20, `batch_extract` in [spikes/s7_kosten.py](../../spikes/s7_kosten.py), results in `s7_results/batch_extract/haiku_production.json`. Spend $0.15.

| Article | Run | Calls | Wall s | Cache read tok | List $ | Paid $ | Facts | J facts vs sync |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Helgoland | sync | 65 | 342 | 55,808 | 0.041 | 0.041 | 615 | 1.00 |
| Helgoland | batch 1 / 2 / 3 | 65 | 342 / 376 / 386 (API 335 / 374 / 383) | 56,680 | 0.039–0.040 | 0.020 | 589 / 581 / 584 | 0.78 / 0.80 / 0.80 |
| Voynich | sync | 45 | 192 | 39,240 | 0.019 | 0.019 | 229 | 1.00 |
| Voynich | batch 1 / 2 / 3 | 45 | 342 / 376 / 386 (API 335 / 371 / 376) | 38,368–39,240 | 0.019–0.020 | 0.010 | 236 / 241 / 237 | 0.80 / 0.81 / 0.83 |

- **Cost:** a batch halves the price, as documented: $0.020 instead of $0.041 for Helgoland, $0.010 instead of $0.019 for Voynich. Prompt caching also hit inside the batches (the same read counts as the synchronous run). On Haiku the whole main extraction costs only 2–4 cents, so the saving is **$0.01–0.02 per article**; with the linked intros (30 more calls on Helgoland) about **$0.03 per episode, 2–3 %** of the roughly $1.0 Helgoland would cost with Haiku extraction and caching.
- **Latency:** every batch took 5.6–6.4 minutes, independent of size. Synchronously, Helgoland took 5.7 minutes, Voynich 3.2. A batch is never faster here, and slower for small articles. The skill names "most batches within 1 hour, at most 24 hours"; S7 saw 2–10 minutes.
- **Results:** batch and synchronous runs differ as much as two batches differ from each other (J 0.78–0.83 against 0.80–0.85). There is no sign of a systematic difference; the batch is the same model behaviour.
- **Caching across articles:** the synchronous Voynich run read its system prompt from the cache the Helgoland run had written minutes before (0 writes). The extraction prompt does not depend on the article.

**Pipeline structure.** Extraction is not one batch per run: the linked intros can only be extracted after the main ranking has chosen the link targets, and a fresh conversation (D6) fetches new targets again. A batched run therefore waits for at least two batches one after the other, about 12 minutes or more, where today the synchronous calls take about as long or less. Completed runs take 22–46 minutes today (metrics); a batch that takes an hour, which the API allows, would push a run past the 90-minute timeout.

**Recommendation: do not batch the extraction.** It saves about 3 cents (2–3 %) per episode, makes no run faster, and adds a waiting time without a firm upper bound against a 90-minute timeout, plus the code for submitting, polling, resuming and cancelling batches. Batching would only pay off for a workload that is not on the run's critical path (for example a backfill of many articles), or if extraction ever moves back to a more expensive model.

## Rerun

Re-evaluate the saved results without API calls:

```
uv run --group spikes python spikes/s7_kosten.py report
uv run --group spikes python spikes/s7_kosten.py spend
```

Live runs (keys from `.env`):

```
set -a; . ./.env; set +a
uv run --group spikes python spikes/s7_kosten.py extract helgoland_sonnet_low    # names: EXTRACT_RUNS
uv run --group spikes python spikes/s7_kosten.py rank helgoland_sonnet_low       # names: RANK_RUNS
uv run --group spikes python spikes/s7_kosten.py chain helgoland_haiku_default sonnet
uv run --group spikes python spikes/s7_kosten.py judge short_default             # names: JUDGE_RUNS
uv run --group spikes python spikes/s7_kosten.py cache                           # caching replay
uv run --group spikes python spikes/s7_kosten.py offers                          # offer prefix tokens (free)
uv run --group spikes python spikes/s7_kosten.py jev                             # lever 5, 2 repeats
uv run --group spikes python spikes/s7_kosten.py judge title_default             # follow-up: judge with article titles
uv run --group spikes python spikes/s7_kosten.py batch_extract                   # follow-up: extraction as a batch
```

A run overwrites its result file. An unfinished batch resumes from `s7_results/batches/<name>.pending.json`. The render cache for extraction (`tests/fixtures/wikipedia/spikes/s7_cache/`) is not committed and is rebuilt from the Wikipedia API.
