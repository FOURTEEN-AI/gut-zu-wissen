# Gut zu wissen

German podcast generated from German Wikipedia articles. For each article, a Python pipeline on GitHub Actions extracts the most surprising facts and writes a ~15-minute dialog between two fixed AI personas: Lena (expert) and Felix (curious questioner). It checks every claim against the extracted facts, synthesizes the audio and publishes the episode. Provider: FOURTEEN IT UG (haftungsbeschränkt).

Design and plan: [docs/](docs/).

## Licences

| Part | Licence |
|---|---|
| Code (everything not listed below) | Apache 2.0, [LICENSE](LICENSE) |
| Archive (`archive/`) | CC BY-SA 4.0, [archive/LICENSE](archive/LICENSE) |
| Wikipedia fixtures (`tests/fixtures/wikipedia/`) | CC BY-SA 4.0, [NOTICE](tests/fixtures/wikipedia/NOTICE) |

The podcast itself is published under CC BY-SA 4.0. Wikipedia content is from the German Wikipedia (CC BY-SA 4.0). This is an independent project, not affiliated with the Wikimedia Foundation.

## Development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and the system packages `libcairo2` and `fonts-dejavu-core` (cover rendering).

```sh
uv sync                      # install dependencies incl. dev group
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format --check . # format check (drop --check to format)
```

Archive layout: `archive/episodes/<date>_<slug>/episode.json` plus one JSON file per pipeline step; see [design](docs/wikipedia-podcast-pipeline.md).

Configuration (threshold t, sampling rate r, model IDs, quotas, prices): [config/pipeline.toml](config/pipeline.toml). Values marked `PLACEHOLDER` are not decided yet.

## GitHub secrets

The repository is public. Secrets exist only as GitHub secrets and are passed as environment variables. The names are proposals.

| Secret | Used for | Needed from |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Sonnet 5.5: extraction, ranking, dialog author, LLM judge, explicit check | Phase 1, step 4 |
| `TYPESAFE_API_KEY` | Jev (TypeSafe) | Phase 1, step 7 |
| `GEMINI_API_KEY` | TTS (Gemini 3.8 Flash TTS) | Phase 1, step 9 |
| `IA_S3_ACCESS_KEY`, `IA_S3_SECRET_KEY` | Internet Archive upload (S3 API) | Phase 2 |

The built-in `GITHUB_TOKEN` covers commits to the archive and the review issues. The pipeline never accesses the feedback mailbox.
