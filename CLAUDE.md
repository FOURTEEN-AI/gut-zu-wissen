# Gut zu wissen

German podcast generated from German Wikipedia articles. For each article, a Python pipeline on GitHub Actions extracts the most surprising facts and writes a ~15-minute dialog between two fixed AI personas: Lena (expert) and Felix (curious questioner). It checks every claim against the extracted facts, synthesizes the audio and publishes the episode. Provider: FOURTEEN IT UG (haftungsbeschränkt).

## Source of truth

Read the relevant document before changing anything in its area. Do not change these documents without asking.

| File | Content |
|---|---|
| `docs/wikipedia-podcast-pipeline.md` | Design: pipeline, gates, judge, error handling, risks |
| `docs/persona.md` | Personas, reaction criterion (part of the judge rubric) |
| `docs/implementation-plan.md` | Phases, steps, decisions D1–D7 |
| `site/impressum.md`, `site/datenschutz.md` | Legal pages; the Pages site must satisfy what they state |

Current phase: see `docs/implementation-plan.md`. Work in plan order, one step per change.

## Rules

- **Fact fidelity.** Every claim and every question premise in the script carries fact IDs. Personas make no statements about themselves. Reactions follow the reaction criterion in `docs/persona.md`.
- **Decided is decided.** D1–D7 in the plan are binding. If a case is not covered, ask; do not invent behaviour.
- **Pure functions for deterministic logic.** Gates, threshold decision, retry budgets, title fitting: pure functions with unit tests.
- **LLM and API calls behind interfaces.** Tests use fakes and recorded fixtures. CI never calls live APIs.
- **Archive JSON is pipeline state.** Indented, stable field order. Every step reads from it and writes to it.
- **Public repo.** Logs, prompts and archive are public. Secrets only via environment variables or GitHub secrets; never in code, logs or archive. No listener data anywhere; the pipeline never accesses the feedback mailbox.
- **Licences.** Code: Apache 2.0 (`LICENSE`). Archive: CC BY-SA 4.0 (own licence file in the archive directory). Files containing Wikipedia text, including test fixtures, live in `tests/fixtures/wikipedia/` with a CC BY-SA notice.
- **Pages site.** No cookies, no tracking, no external resources (including fonts). Audio player uses `preload="none"`. Impressum and Datenschutz linked on every page.
- **Wikipedia API.** Descriptive User-Agent with contact address; respect rate limits.

## Cover

- Template: `assets/cover/cover-template.svg`. The fitting contract is a comment in the file.
- Font: DejaVu Sans Bold from the system package `fonts-dejavu-core`. Nothing bundled. The workflow checks `fc-list | grep "DejaVu Sans"` before rendering.
- Rendering: `cairosvg` (needs `libcairo2`). Title measurement with Pillow against `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`.
- Output: PNG, 3000 × 3000.

## Conventions

- Python 3.12, `pytest`, `ruff` (lint and format).
- Config file for t, r, model IDs, quotas and prices; no magic numbers in code.
- Code and identifiers in English. Podcast content, prompts for German output and legal texts in German.

## Working style

- Ask before deciding anything the documents leave open.
- Run tests and lint before every commit.
- After completing a plan step, mark it in `docs/implementation-plan.md`.
