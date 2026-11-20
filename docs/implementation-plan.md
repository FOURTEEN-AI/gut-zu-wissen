# Implementation plan: „Gut zu wissen“

## Principles

- **Archive JSON is pipeline state.** Each step reads from the archive and writes to it, so runs can be resumed, reviewed and replayed.
- **Pure functions for deterministic logic, LLM and API calls behind interfaces.** This covers gates, the threshold decision and retry budgets. Unit tests run on recorded fixtures, with Wikipedia fixtures kept separate under CC BY-SA.
- **Retry budgets as an explicit state machine.** Turn, conversation and audio levels are tested with fakes before any real API is wired in.
- **Config file** for t, r, model IDs, quotas and prices. Prices feed the per-episode cost log.

## Phase 0: Spikes

These retire the open risks before the build starts.

| Spike | Question | Output |
|---|---|---|
| S1 Jev ✅ | German accuracy on ~30 hand-labelled turn/fact pairs; overlength error code; latency; reliability of the content-preservation question (see D1) | Go/no-go for Jev, error table complete |
| S2 SoMaJo + masking ✅ | Robust on 10 diverse articles (nested templates, tables, lists)? | Masking rules, segment ID scheme |
| S3 Rendering ✅ | Does `action=parse` per segment differ from in-article rendering? How do infobox parameters map to rendered row labels (see D3)? | Decision: per segment, or render once and map; label-mapping method |
| S4 Gemini TTS ✅ | Voice choice, section length under the output limit (16,384 tokens per call), audibility of seams | Two voice names, section strategy |
| S5 Whisper ✅ | Runtime for 15 min of audio on an Actions CPU runner (large vs. faster-whisper int8) | Model size, cache setup |
| S6 Explicit ✅ | Accuracy of the explicit check (D7, step 11) on ~40 hand-labelled excerpts: false positives on factual mentions, false negatives on drastic depictions, share of `uncertain`, run-to-run stability | Error rates against owner-confirmed labels, prompt-change suggestions |
| S7 Kosten | Five cost levers per episode, measured without production changes: lower effort, prompt caching, Claude Haiku 5.5 for extraction and ranking, short reasons on acceptance, higher Jev share with a safety net | Saving per lever in $ and %, effect on fact fidelity, recommendation; decisions left to the owner |

## Phase 1: Manual run to audio (Ausbaustufe 1)

1. **Repo skeleton.** ✅ Done. Python package, Apache 2.0 for code, CC BY-SA for the archive directory, README mapping, CI (lint, tests), secrets, `workflow_dispatch` with a `concurrency` group.
2. **Fetch.** ✅ Done.
   - Latest sighted revision; fallback to latest revision if none exists (D2).
   - Wikitext, revid, revision status (sighted or latest), UTC timestamp, categories.
   - Gates as pure functions: living person; episode for this article published less than 6 months ago (D4).
3. **Segmentation.** ✅ Done. Masking, then SoMaJo, then unmasking. Infobox rows from `Infobox *` and `Taxobox` only, as their own segment type (D3). Stable segment IDs (revid + index).
4. **Extraction per chunk.** ✅ Done.
   - Structured output (JSON schema).
   - Gates for uniqueness, ≥ 8 words, full sentence, and context (contiguous, same chunk, ≤ 3 segments).
   - Plaintext rendering of offered facts and their context. Infobox rows as „Infobox type – section header – label: value“, header omitted when there is none (D3).
5. **Ranking.** ✅ Done.
   - 20–40 main facts. Linked intros: fetch (same revision rule), gate, extract, rank for 6–12 more, each with its own revid for the shownotes. More than the minimum only with genuinely surprising, non-duplicate facts; fewer than the minimum when extraction yields fewer (decided 2026-11-20).
   - Gate: fewer than 10 offered facts → gate failure before the dialog author (D5).
   - Einordnung (decided 2026-11-20): the first intro sentences of the main article (up to 2, `einordnung_facts_max`) that pass the fact gates are always offered first, outside the ranking and the main quota; archived as `einordnung_facts` in `04_ranking.json`. The first prose sentence of the intro is always Einordnung #1 when it passes the gates and is never excluded by D6; Einordnung sentences are never ranked or offered as main facts as well.
6. **Dialog author.** ✅ Done.
   - Output schema per turn (text, fact IDs, explicit statements with fact IDs). Gate: fact IDs exist.
   - Gate: fewer than 10 used facts → conversation regeneration (D5).
7. **Judge.** ✅ Done.
   - Question builder: one question per statement plus the completeness question.
   - Jev client implementing the error table.
   - Threshold decision as a pure function, with tests at t, 1−t and between.
   - LLM judge in two modes: justify a rejection, or give a full verdict.
   - Gate after judging: fewer than 10 used facts in accepted turns → gate failure (D5).
8. **Orchestrator.** ✅ Done.
   - Turn regeneration with neighbours as context.
   - Neighbour cascade (D1): after a turn regeneration, the next turn is always regenerated. A two-stage check decides whether the regenerated turn conveys the same statements as before. Jev first: p ≥ t → confirmed, cascade stops; p ≤ 1−t → content lost, cascade continues. Uncertain, or a Jev error the error table sends to the LLM judge → the LLM judge decides (sees only the previous text and statements and the new text); an undecided or malformed answer counts as content lost. Two budgets per triggering turn (changed 2026-11-20): judge rejections use the 2× budget (`turn_retries`), gate rejections (decided by "gate": unknown fact ID or alias, repetition, markup, empty text, statement without identifiers, unknown speaker) a separate 3× gate budget (`gate_retries`). A retry charges the type of the rejection it fixes; a cascade step charges the type of the rejection that triggered the cascade; a cascade step rejected again charges that rejection's type for its retry. Each regeneration records the budget it charged (`budget` in `05_dialog.json`).
   - Conversation regeneration (D6, changed 2026-11-20): exclude every fact cited by a rejected statement (cumulatively). When a turn budget is used up, keep turns 0..k-1 with their judgements (k: the triggering turn if the failure happened inside a cascade, otherwise the failing turn) and let the author continue the dialog from turn k to the end, closing summary included, on the same offer without the excluded facts (no re-ranking); it sees the kept turns and the facts already told in them and must not retell them, and receives the reason of the chain's last rejection (the one that used up the budget; earlier rejections of the chain briefly) in its own form (aliases, German gate texts), to avoid that error (2026-11-20). Only the new turns are judged; the repetition gate and the D5 gates look at the whole dialog. Fall back to fresh rankings and a fresh dialog when k = 0, when the remaining offer cannot reach the D5 minimum, or after D5 fails after the author. Counts against `conversation_retries`; the discarded tail (or whole dialog) is archived with the path taken (`regeneration`, `continued_from`, `fresh_reason`) and the archived reason given to the continuation (`continuation_reason`).
   - Gate failures: manual run → abort with notice.
   - Escalation: failing job plus job summary.
9. **Markup gate**, then TTS: ✅ Done.
   - Sections at turn boundaries, at most `tts_section_max_chars` (6,000, spike S4).
   - Intro and outro synthesized once and committed.
   - Concatenation and loudness normalization to −16 LUFS (`target_loudness_lufs`) with a true-peak limiter (ceiling `true_peak_ceiling_dbtp`, −1.5 dBTP). A test checks loudness and peak ceiling of the output.
10. **STT verification.** ✅ Done. Normalization, diff per section, LLM decision on whether a deviation changes the statement, resynthesis of the affected section.
11. **Explicit check (D7).** ✅ Done. LLM judges the final script; drastic depiction of violence, sexuality or drugs, or spoken profanity → episode flagged explicit (a factual or passing mention does not count). Uncertain → explicit. Result stored in the archive.
12. **Review loop.** ✅ Done.
    - Sampling with rate r; r state in the repo.
    - Review issue in machine-readable format.
    - `issue_comment` workflow writes labels to the archive, accepting comments from you only.
13. **Exit criteria.** ✅ Done (2026-11-20: 5 of 5 episodes fully reviewed, no `nicht belegt`; measurements from `python -m gut_zu_wissen.metrics`). 5 episodes fully reviewed at r = 100 %. Measured: facts used per minute, escalation rate, cascade length, runtime, cost, error rate by revision status.

## Phase 2: Publication (Ausbaustufe 2)

- **Blocker:** resolve the ⚠ placeholders in datenschutz.md. ✅ Done (2026-11-20; Stand 22 November 2026).
- **Cover (D7).** ✅ Done (2026-11-20; template approved by Andreas, font bundled in `assets/cover/fonts/`).
  - SVG template with a title box: drafted by Claude, finalized by Andreas.
  - Pipeline renders the article title per episode: wrap at word boundaries, no line limit, then shrink the font until the block fits; no ellipsis.
  - SVG → PNG (1400–3000 px, square), bundled font, no external resources.
- Internet Archive upload via the S3 API (audio and episode cover). ✅ Done (2026-11-20; verified with a live test upload to `test_collection`).
  - Decided 2026-11-20: MP3 encoded by the pipeline from the FLAC master (128 kbit/s, `[hosting]` in the config); publishing is a manual workflow (`publish.yml`, input: episode ID and the producing run's ID) after the review, not part of the run.
  - One item per episode, `gut-zu-wissen_<episode id>`; result in `10_publication.json`; the episode becomes `published` only once the item lists both files.
  - Needs the GitHub secrets `ARCHIVE_S3_ACCESS_KEY` and `ARCHIVE_S3_SECRET_KEY`.
- Static Pages site: no external resources, `preload="none"`, legal pages linked on every page. ✅ Done in code (2026-11-20; generator `gut_zu_wissen.site`, workflow `pages.yml`; Pages source set to "GitHub Actions" on 2026-11-20).
  - Publish the plain-language explainer `site/so-funktionierts.html`: ✅ Done (2026-11-20; live, identical to `main`). Before publishing:
    - check the Impressum/Datenschutz link targets against how the site is actually served: ✅ done (generated as `impressum.html`, `datenschutz.html`);
    - replace the "trial phase, not yet published" wording: ✅ done;
    - update the steps if the pipeline has changed since: ✅ checked (2026-11-20; numbers match `[quotas]`, D2, D4, D5; no change needed).
- **RSS** with iTunes tags and stable GUIDs: ✅ Done in code (2026-11-20; `gut_zu_wissen.feed`, built with the site).
  - Category: Education.
  - Owner email: gut-zu-wissen@fourteen-it.de.
  - Explicit: `false` at feed level; per episode from the explicit check.
  - Episode image: rendered cover.
- Shownotes: revision permalinks, history links, mailto link with episode title and ID. ✅ Done in code (2026-11-20; episode pages of the site).
- Spotify submission (manual, once; verification mail goes to the owner email). Submitted 2026-11-20 with the github.io feed URL; Spotify's review takes up to 24 hours.
- Custom domain `gut-zu-wissen.fourteen-it.de` for the Pages site (later, after Spotify has approved the feed): DNS CNAME to `fourteen-ai.github.io`, set the custom domain in the Pages settings, change `base_url` in `[site]`, announce the move in the old feed with `itunes:new-feed-url` (GitHub redirects the old address). Check Impressum and Datenschutz for addresses. A plain HTTP redirect is not an option; apps and Spotify do not follow it reliably.

## Phase 3: Article of the day (Ausbaustufe 3)

All items ✅ done in code (2026-11-20; `gut_zu_wissen.daily`, `gut_zu_wissen.autopublish`, workflows `daily.yml` and `auto-publish.yml`). The first live runs are still to be watched (see `docs/phase3-assumptions.md`).

- Resolve the article of the day. ✅ Spike S8 (`docs/spikes/s8-artikel-des-tages.md`): chronology page `Wikipedia:Hauptseite/Artikel des Tages/Chronologie <year>` (written by a bot at 23:00 UTC), the weekday page as fallback.
- Cron `17 3 * * *` (`daily.yml`). ✅
- Gate failures (living person, recent episode, too few facts) and used-up budgets → substitute article. ✅
- Substitute article: random from Exzellent/Lesenswert (`Kategorie:Wikipedia:Exzellent`, `Kategorie:Wikipedia:Lesenswert`), seeded with the date, living-person and recent-episode gates checked before selection, max 2 (`substitute_articles_max`), otherwise the episode is dropped. ✅
- Jev abort: escalation and the episode is dropped, no substitute. ✅ A dropped episode or a crash fails the workflow run; GitHub mails the owner.
- Delayed automatic publication (decided 2026-11-20): `auto-publish.yml` publishes the episode once a day at a fixed time (10:00 UTC) if its run ended at least `min_age_hours` (config `[publishing]`, 2) before. The owner holds it with the label `hold` on the review issue; an episode whose review found an error is never published automatically. ✅

## Decisions

| ID | Topic | Decision |
|---|---|---|
| D1 | Neighbour coherence | After a turn regeneration, the next turn is always regenerated. The cascade continues until a regenerated turn is confirmed to convey the same statements as before. Two-stage check (changed 2026-11-20): Jev p ≥ t → confirmed; p ≤ 1−t → content lost; uncertain or a Jev error that goes to the LLM judge → the LLM judge decides; in doubt (undecided or malformed answer) → content lost. Budgets (changed 2026-11-20): judge rejections use the triggering turn's 2× budget, gate rejections (no judge call: unknown fact ID or alias, repetition, markup, empty text, statement without identifiers, unknown speaker) a separate 3× gate budget per triggering turn. Retries charge the type of their rejection; cascade steps charge the type of the rejection that triggered the cascade. |
| D2 | Revision | Latest sighted revision; latest revision if none exists. Revision status recorded in the archive. |
| D3 | Infobox rows | `Infobox *` and `Taxobox` only (`Personendaten` excluded). Plaintext: „Infobox type – section header – label: value“ (header omitted when the row has none); a single-cell row with a colon is split at the first colon into label and value (spike S3). |
| D4 | Repeats | New episode for an article allowed 6 months after the last episode's publication date; otherwise gate failure. |
| D5 | Minimum content | 10 used facts. Offer < 10 → gate failure before the dialog author. After the author < 10 → conversation regeneration. After the judge < 10 → gate failure. |
| D6 | Conversation regeneration | Exclude facts cited by rejected statements. Changed 2026-11-20: after a used-up turn budget, keep turns 0..k-1 (k: the turn that started the failing chain) and continue the dialog from turn k on the same offer without the excluded facts (no re-ranking); the author receives the reason of the chain's last rejection. Rerun both rankings and write a fresh dialog only when k = 0, when the remaining offer cannot reach the D5 minimum, or when D5 fails after the author. |
| D7 | Feed metadata | Static SVG cover template with per-episode title (wrap, no line limit, shrink to fit). Category Education. Owner email gut-zu-wissen@fourteen-it.de. Explicit: feed `false`, per episode by LLM check of the final script (drastic depiction of violence, sexuality or drugs, or spoken profanity; a factual or passing mention does not count); uncertain → explicit. |
