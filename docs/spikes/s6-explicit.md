# Spike S6: Explicit check

**Question:** how accurate is the explicit check (D7, step 11) under the rule of 2026-11-19? Explicit means a *drastic* depiction of violence, sexuality or drugs (vivid, detailed or sensational, glorifying, or instructions), or spoken profanity. A factual, historical, scientific or passing mention does not count. Uncertain counts as explicit.
**Output:** error rates against hand-labelled excerpts, run-to-run stability, cost and latency, and suggestions for the prompt.
**Result (provisional):** on 40 excerpts × 3 repeats, the check made **no error**. There were 0/81 false positives, 0/39 false negatives, no `uncertain`, and the same verdict in every repeat. A call costs about **$0.005** and takes about **2 s**. The set is too easy to show where the boundary lies (see Limits). No prompt change is needed for the cases tested.

Code: [spikes/s6_explicit.py](../../spikes/s6_explicit.py). Cases: [tests/fixtures/wikipedia/spikes/s6_cases.json](../../tests/fixtures/wikipedia/spikes/s6_cases.json). Labelling sheet: [spikes/s6_labels.md](../../spikes/s6_labels.md). Every answer: [tests/fixtures/wikipedia/spikes/s6_results/](../../tests/fixtures/wikipedia/spikes/s6_results/). Run with `claude-sonnet-5-5` (`models.llm_judge`) on 2026-11-19. Total cost: 177k input and 22k output tokens, about **$0.58**.

> **Labels are a proposal.** Claude wrote all excerpts and labels. The project owner has not confirmed them yet, so every number here is measured against proposed labels. The spike is done only after the owner confirms the labels (`confirmed: true` per item). See [Rerun](#rerun).

## Cases

There are 40 short excerpts (2–5 turns, Lena and Felix, German, in persona style). Each is written on the topic of a real German Wikipedia article, with its title and sighted revid recorded (CC BY-SA, see NOTICE). The text is written for the spike and is not copied from the article. The drastic excerpts go beyond what the articles say on purpose.

| Group | n | Proposed label | Content |
|---|---:|---|---|
| clean | 13 | clean | Construction, geography, maths, astronomy, biography; one "Ach du meine Güte" |
| borderline | 14 | clean | Battle (Hastings), guillotine (Ludwig XVI.), fox killing a mouse, Riesling, ibuprofen, Great Train Robbery (driver injured), Black Death, Verdun casualties, coca leaves and cocaine, bonobo sexual behaviour, Götz quote referred to but not spoken, witch trials (torture, executions), white shark attacks, Opium Wars |
| explicit | 13 | explicit | Violence 4 (gore at Hastings, Damiens' execution in detail, impalement glorified, Ripper mutilations); sexuality 3 (Kamasutra positions, Casanova scene, Spargel innuendo); drugs 3 (LSD trip glorified, Absinth binge recommended, psilocybin consumption tips); profanity 3 (Götz quote spoken, "Scheiße" as reaction, "Arschloch" as an insult) |

Each case goes through the production code path unchanged: `check_explicit` → `AnthropicLlm.complete_json` with `EXPLICIT_PROMPT`, `EXPLICIT_SCHEMA` and `max_tokens` from the config. The runner only wraps the SDK client to record usage and latency. The flag is `is_explicit(verdict, findings)`.

## Results

### Flag (40 cases × 3 repeats = 120 calls, 0 errors)

| | Rate |
|---|---:|
| False positive (clean or borderline → explicit) | 0/81 (0 %) |
| False negative (explicit → clean, the critical error) | 0/39 (0 %) |
| Verdict `uncertain` | 0/120 (0 %) |
| Same flag in all 3 repeats | 40/40 |
| Same verdict in all 3 repeats | 40/40 |

| Group / category | Runs flagged explicit |
|---|---:|
| clean | 0/39 |
| borderline violence | 0/24 |
| borderline drugs | 0/12 |
| borderline sexuality | 0/3 |
| borderline profanity | 0/3 |
| explicit violence | 12/12 |
| explicit sexuality | 9/9 |
| explicit drugs | 9/9 |
| explicit profanity | 9/9 |

### Misclassified items

None.

### Findings

- **Category:** in every explicit run, the findings name exactly the intended category.
- **Quotes:** every quote is verbatim from the turn the finding points to.
- **Reasons:** the reasons use the rule's own terms. For example, b07: "sachlich und historisch … ohne sie anschaulich oder reißerisch auszumalen". For e11: "wörtlich ausgesprochen … zählt als Schimpfwort …, auch wenn es literarisch zitiert wird". For b11: "angedeutet, aber nicht ausgesprochen".
- **Verdict and flag never disagree.** `uncertain` never occurs, so D7's "uncertain → explicit" did not affect any result.

### Cost and latency

- **Latency:** median 2.2 s, maximum 4.0 s per call, at 4 calls in parallel.
- **Tokens:** about 1,480 input and 180 output tokens per call. About 1,300 of the input tokens are the prompt.
- **Per episode:** the check runs once on a script of about 150 turns, roughly 6–8k input tokens. That is about $0.02 per episode at $2/$10 per million tokens. The `[prices]` values for Claude are still `PLACEHOLDER` in the config, so the spike uses these Sonnet 5.5 list prices.

## Limits

- **The set is easy.** Each excerpt is written to be clearly on one side of the rule. The real boundary cases are excerpts with moderate detail: a sentence or two of description, a quoted contemporary account of a battle, a medical description of a symptom. The set has none of these, so the 0 % rates are an upper bound on quality, not an estimate.
- **The set overlaps the prompt.** The prompt names "eine Schlacht, ein Raubtier, das Beute jagt, oder der Weinbau" as non-explicit, and three borderline items (b01, b03, b04) are exactly those examples.
- **Short excerpts, not whole scripts.** A real script has about 150 turns. One drastic sentence among many could be overlooked, and the prompt asks only for the first finding per category. The spike does not test this.
- **Written by the same model family.** Claude wrote both the excerpts and the labels, and Claude Sonnet 5.5 judges them.

## Prompt-change suggestions

No change is needed for the cases tested. Do not apply any of these until the labels are confirmed and a harder set shows an error. They are listed in order of value.

1. **Make quoting explicit.** State whether a quoted vulgar word counts as "tatsächlich ausgesprochen". Suggested wording: „Auch zitierte Schimpfwörter zählen, sobald sie wörtlich ausgesprochen werden; ein bloßer Verweis auf ein derbes Zitat zählt nicht.“ The model already behaves this way (e11, b11), but the rule does not say so. This depends on owner decision Q1.
2. **Say what "Anleitung" covers.** Suggested wording: „Anleitung heißt auch: Tipps zu Dosis, Zubereitung oder Konsum, ohne Warnung als Empfehlung formuliert.“ This makes e10-type cases independent of the glorifying tone that accompanied it here.
3. **Name mild expressions.** The rule does not say whether "verdammt", "Mist" or "Mann!" count as profanity (Q2). Once the owner decides, one sentence in the prompt removes the ambiguity, for example „Milde Ausrufe wie „Mist“ oder „verdammt“ zählen (nicht)“.
4. **Long scripts.** Optionally, add „Prüfe jeden Redebeitrag einzeln“. This is a guard against one drastic sentence being overlooked in 150 turns. Test it on a real full script before adding it.

## Rerun

After you change labels in `s6_cases.json`, re-evaluate the saved answers. This needs no API calls:

```
uv run --group spikes python spikes/s6_explicit.py sheet    # refresh the labelling sheet
uv run --group spikes python spikes/s6_explicit.py report   # rates against the current labels
```

Only rerun the model for a prompt change or new cases. That costs about $0.60 for 40 cases × 3 repeats:

```
set -a; . ./.env; set +a
uv run --group spikes python spikes/s6_explicit.py run
uv run --group spikes python spikes/s6_explicit.py report
```

`run` overwrites `s6_results/<model>.json`. The file also stores the prompt it used. For new cases, run `revids` to record the source revisions.

## Open for the owner

- **Q1:** does a quoted vulgar sentence (the Götz quote spoken aloud, e11) count as spoken profanity? The proposal says yes, following the wording of the rule.
- **Q2:** do mild expressions ("verdammt", "Mist") count as profanity? There is no case for this yet.
- **Q3:** should S6 get a second, harder set before ✅? This would cover moderate-detail borderline cases and drastic sentences inside a full-length script, ideally taken from real author output (step 13).
