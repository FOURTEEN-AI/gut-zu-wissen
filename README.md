# Gut zu wissen

German podcast generated from German Wikipedia articles. For each article, a Python pipeline on GitHub Actions extracts the most surprising facts and writes a ~15-minute dialog between two fixed AI personas: Lena (expert) and Felix (curious questioner). It checks every claim against the extracted facts, synthesizes the audio and publishes the episode. Provider: FOURTEEN IT UG (haftungsbeschränkt).

Design and plan: [docs/](docs/).

## Licences

| Part | Licence |
|---|---|
| Code (everything not listed below) | Apache 2.0, [LICENSE](LICENSE) |
| Archive (`archive/`) | CC BY-SA 4.0, [archive/LICENSE](archive/LICENSE) |
| Wikipedia fixtures (`tests/fixtures/wikipedia/`) | CC BY-SA 4.0, [NOTICE](tests/fixtures/wikipedia/NOTICE) |
| Intro and outro audio (`assets/audio/`) | CC BY-SA 4.0, [assets/audio/LICENSE](assets/audio/LICENSE) |
| Show art and cover template (`assets/cover/`) | CC BY-SA 4.0, [assets/cover/LICENSE](assets/cover/LICENSE) |

The podcast itself is published under CC BY-SA 4.0. Wikipedia content is from the German Wikipedia (CC BY-SA 4.0). This is an independent project, not affiliated with the Wikimedia Foundation.

## Development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and the system package `libcairo2` (cover rendering; the font is bundled). `ffmpeg` is optional locally: the audio test uses it as an independent loudness meter when it is installed (CI installs it).

```sh
uv sync                      # install dependencies incl. dev group
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format --check . # format check (drop --check to format)
```

Archive layout: `archive/episodes/<date>_<slug>/episode.json` plus one JSON file per pipeline step in step order: `01_article.json`, `02_segments.json`, `03_facts.json`, `04_ranking.json`, `05_dialog.json`, `06_judgement.json`, `07_explicit.json` (the explicit check runs right after the judge, before any TTS cost), `08_audio.json`, `09_review.json`; see [design](docs/wikipedia-podcast-pipeline.md).

Intro and outro audio (`assets/audio/`) is synthesised once with the episode voices and committed; the pipeline refuses to run when it is missing or no longer matches the text, model or voices:

```sh
uv run python -m gut_zu_wissen.intro_outro   # needs GEMINI_API_KEY; --force to redo
```

Configuration (threshold t, sampling rate r, model IDs, quotas, prices): [config/pipeline.toml](config/pipeline.toml). Values marked `PLACEHOLDER` are not decided yet.

## Resuming from the audio step

A run that failed only in the audio step (TTS or STT error, STT budget used up, crash inside the step) can be resumed without repeating the author and judge work: start the Resume audio workflow (`.github/workflows/resume-audio.yml`) with the archive folder name as `episode_id`, or locally

```sh
uv run python -m gut_zu_wissen.resume 2026-11-20_tunguska-ereignis   # needs ANTHROPIC_API_KEY, GEMINI_API_KEY
```

It reruns sections, TTS, STT verification and normalisation on the archived final script, then draws the review sample; the workflow opens the review issue as after a normal run. Any other episode is refused with a reason and the archive stays unchanged: running, completed or published episodes, gate failures before the audio step, a failed explicit check. `08_audio.json` keeps the earlier attempts under `previous_attempts`; `episode.json` lists the resume under `resumed`. The metrics count the episode as one run with its final status, marked as resumed; runtime and cost include every attempt.

## Publishing and the daily run

- **By hand:** the Publish workflow (`publish.yml`) takes an episode ID and the ID of the run that produced the audio and publishes the episode to the Internet Archive; the Pages workflow then rebuilds the site and the RSS feed.
- **Daily (phase 3):** the Daily workflow (`daily.yml`, cron `17 3 * * *` UTC) runs the German Wikipedia's article of the day (`python -m gut_zu_wissen.daily`; substitutes from the featured and recommended articles after a failure, at most two). The Auto-publish workflow (`auto-publish.yml`, hourly) publishes the episode 6 hours after its run (`[publishing]` in the config). To stop it, put the label `hold` on the episode's review issue; remove the label to release it. An episode whose review found an error is never published automatically. A dropped episode or a crash fails the workflow run, and GitHub sends the usual mail about it.

## GitHub secrets

The repository is public. Secrets exist only as GitHub secrets and are passed as environment variables. 

| Secret | Used for | Needed from |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Sonnet 5.5: extraction, ranking, dialog author, LLM judge, explicit check | Phase 1, step 4 |
| `TYPESAFE_API_KEY` | Jev (TypeSafe) | Phase 1, step 7 |
| `GEMINI_API_KEY` | TTS (Gemini 3.8 Flash TTS) | Phase 1, step 9 |
| `ARCHIVE_S3_ACCESS_KEY`, `ARCHIVE_S3_SECRET_KEY` | Internet Archive upload (S3 API) | Phase 2 |

The built-in `GITHUB_TOKEN` covers commits to the archive and the review issues. The pipeline never accesses the feedback mailbox.

## Review

After a successful run the episode workflow draws each accepted turn with fact IDs with the sampling rate r (`archive/state/review.json`; phase 1 keeps r at 100 %) and opens a review issue for the drawn turns (`09_review.json` in the episode folder); when r < 100 % draws no turn, the issue is still opened, only for the explicit label. Answer with comments, one line per turn: `<number>: belegt` or `<number>: nicht belegt`, optionally followed by a dash and a note. The issue also shows the explicit check (`07_explicit.json`); confirm or correct its flag with `explicit: korrekt`, `explicit: sollte explicit sein` or `explicit: sollte nicht explicit sein`. The explicit label is required. `review.explicit_accuracy` computes the false-positive and false-negative rates of the explicit check over the archive. The review workflow (`.github/workflows/review.yml`) writes the labels to `09_review.json`; once every drawn turn and the explicit check have a label, the review is answered and recorded in the state file, which updates r when `adjust_rate` is on (off in phase 1); whether it is clean or has an error follows the turn labels only (no drawn turn: clean). The bot then posts its status reply and closes the issue as completed. While the turns are labelled but the explicit label is missing, the reply asks for it and the issue stays open. Comments on a closed issue are still applied as corrections; the issue stays closed and is never reopened automatically. Only comments by the reviewer configured in `config/pipeline.toml` count. To read an issue again (for example after a skipped run), start the review workflow manually with the issue number.

## Metrics

Exit criteria of phase 1 (step 13) and the measurements over the archive: facts used per minute (primary: dialog audio only; secondary: the whole episode audio with intro and outro), escalation rate, cascade length, runtime, cost, error rate by revision status and the explicit check's false-positive and false-negative rates.

```sh
uv run python -m gut_zu_wissen.metrics              # Markdown report
uv run python -m gut_zu_wissen.metrics --json       # the same as JSON
uv run python -m gut_zu_wissen.metrics --archive DIR
```

The exit criteria are met once `[exit_criteria]` in `config/pipeline.toml` holds: 5 episodes whose review is answered (every drawn turn and the explicit check labelled) with the sample drawn at r = 100 %. Costs are stored per run in `episode.json` (`cost`: USD for the Claude author, the Claude LLM judge, Jev and TTS, with the config prices in force at run time); the report uses these and falls back to today's prices only for older folders without them, naming those folders. Cost parts without a recorded token count or with a `PLACEHOLDER` price are listed as unknown. The Metrics workflow (`.github/workflows/metrics.yml`, manual start) prints the report to the job summary.
