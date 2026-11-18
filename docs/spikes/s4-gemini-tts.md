# Spike S4: Gemini TTS

**Question (plan, Phase 0, replaced ElevenLabs by Gemini 3.8 Flash TTS):** which two voices, how long may a section be, are the seams between sections audible?
**Output:** two voice names, section strategy.
**Result:**
- **Voices:** Lena = `de-de-advisor-3`, Felix = `de-de-tutor-4`.
- **Model:** `gemini-3.8-flash-tts`.
- **Sections:** at most 6,000 characters per section at turn boundaries, so 3 sections for a typical episode.
- **Seams:** the measured differences at a seam are within the normal variation inside a section.
- **Limits:** the API's documented size limits are not enforced. A whole episode fits in one call, but I recommend keeping sections for cheap retries.

**Correction (2026-11-17), applied.** The first version of this report used WER numbers that were wrong: my transcription passed Gemini's 24 kHz audio to Whisper, which reads a raw array as 16 kHz, so Whisper heard everything 1.5× too fast (for example the intro scored WER 0.364 and lost its first sentence). I fixed `transcribe()`, re-measured every WER from the saved audio (`spikes/s4_gemini.py retranscribe`, no API calls) and replaced the numbers below with the corrected ones. The old values stay in the result files as `wer_24khz_bug`. Pitch, loudness, level gaps, token counts, durations and costs never used Whisper and are unchanged. None of the choices changed.

**What I could not verify.** I can't listen. The project owner listened to the samples on 2026-11-18 and approved the voices. Everything below is a measurement: Whisper word error rate (intelligibility), Praat pitch, pyloudnorm loudness, token usage. Naturalness, pleasantness, German pronunciation of names and the audibility of a seam to a human ear are **not measured**.

Code: [spikes/s4_gemini.py](../../spikes/s4_gemini.py). Test dialog (Eiffelturm facts): [s4_script.json](../../tests/fixtures/wikipedia/spikes/s4_script.json). Results: [tests/fixtures/wikipedia/spikes/s4_results/](../../tests/fixtures/wikipedia/spikes/s4_results/). Audio is not committed (the archive holds audio only in the published form). Total cost of the spike: about $1.50 (the 35-minute probe alone was about $0.48).

## API facts (measured and from the docs)

| | |
|---|---|
| Endpoint | `POST https://generativelanguage.googleapis.com/v1beta/interactions`, header `x-goog-api-key`. Two speakers: `generation_config.speech_config.speakers = [{speaker, voice}]`; each turn is a `text` content item with a `speech_metadata` annotation naming the speaker. `response_format: {"type": "audio"}`. |
| Response | `steps[].content[]` with `type: "audio"`, base64 WAV, 24 kHz, mono, 16-bit. Usage in `usage.total_input_tokens` / `total_output_tokens`. |
| Audio tokens | Exactly **25.0 per second** of audio, in every request measured. |
| Voices | 30 named voices (Kore, Puck, …) and 87 German library voices (`de-de-podcaster-3`, …; `GET /v1beta/voices?language_code=de-DE`, with gender, pitch, persona, age). **Both kinds work in two-speaker requests**, although the docs mention prebuilt voices only. |
| Rate limit | **10 requests per minute** (Tier 1). Response `429`, `error.code = "too_many_requests"`, text contains `Please retry in 25s`. The runner parses this and waits. |
| Speed | 63 s for 12 min of audio, 81 s for 16 min, 173 s for 35.5 min (real-time factor 0.09 to 0.08). |
| Price | $0.50 per million text tokens, $9.00 per million audio tokens through 2026-12-31; $1.00 and $18.00 from 2027-01-01 (Flash-Lite: $6.00 / $12.00). **About $0.22 per 16-minute episode** in 2026 and about $0.44 in 2027, plus retries. Already in the config. |

## Voices

29 candidates (14 for Lena, 15 for Felix), each synthesised on the same German text and measured ([voices.json](../../tests/fixtures/wikipedia/spikes/s4_results/voices.json)). Selection by the persona profile (Lena: female, medium pitch, calm, clear; Felix: male, more energy, clearly distinguishable).

- **Intelligibility:** all 29 voices have WER ≤ 0.02 on the sample (mean 0.002). This does not separate the candidates.
- **WER up to 1.4 in the first run** (`tutor-4`) came from the sample-rate bug, not from the voice: at the correct rate `tutor-4` scores 0.02.

Four pairs were then tested in the full dialog ([pairs.json](../../tests/fixtures/wikipedia/spikes/s4_results/pairs.json)):

| Pair | Lena | Felix | WER | Pitch separation | Level gap (Lena louder) |
|---|---|---|---:|---:|---:|
| **A** | `de-de-advisor-3` | `de-de-tutor-4` | 0.062 | 7.7 semitones | 2.9 dB |
| B | Sulafat | Puck | 0.051 | 9.0 | 3.4 |
| C | `de-de-tutor-2` | `de-de-concierge-9` | 0.065 | 8.0 | 3.5 |
| D | Despina | `de-de-assistant-10` | 0.084 | 5.1 | 2.8 |

The WER differences are mostly proper nouns (SETE, Champ de Mars) and are not significant. **Pair A** was chosen for two reasons that are not measurements: the library voices are tagged "accent: German, from Germany" (the persona document asks for native speakers; the named voices are multilingual), and Google's own descriptions match the roles (Lena "reflective, calm, and reassuring", Felix "enthusiastic, engaging, and fun"). Pair B is the fallback. In every pair Lena is about 3 dB louder than Felix; that is a property of the model's voices; the only setting I tried (`style`) did not change it.

## Model and style

`gemini-3.8-flash-tts` against `gemini-3.8-flash-lite-tts`, each twice on pair A ([models.json](../../tests/fixtures/wikipedia/spikes/s4_results/models.json)):

| | WER | Request time for 2 min | Cost for 2 min |
|---|---:|---:|---:|
| Flash | 0.058, 0.076 | 24 s | $0.026–0.029 |
| Flash-Lite | 0.065, 0.051 | 14–15 s | $0.018 |
| Flash-Lite + `style` annotations | 0.062, 0.091 | 14–15 s | $0.019 |

No measurable quality difference; the run-to-run noise within one model (audio length varies by ±5 %) is larger than the difference between models. **Flash** is used because the project owner named it and Google positions it for multi-speaker dialogue; it costs about $0.07 more per episode. `speech_metadata.style` had no measurable effect on pitch variation or intelligibility and is not used (the docs also warn that extra prompt text can increase voice drift).

## Section length

Documented limits ([model page](https://aistudio.google.com/docs/models/gemini-3.8-flash-tts)): 8,192 input tokens, 16,384 output tokens (about 10.9 minutes). Measured with real requests ([limit.json](../../tests/fixtures/wikipedia/spikes/s4_results/limit.json)):

| Request | Input tokens | Output tokens | Audio | Last words of the script in the transcript? | WER |
|---|---:|---:|---:|---|---:|
| 12,257 chars | 3,181 | 18,034 | 12.0 min | yes | 0.060 |
| 16,343 chars | 4,241 | 24,124 | 16.1 min | yes | 0.066 |
| 36,773 chars | 9,541 | 53,296 | 35.5 min | yes | 0.072 |

**The limits are not enforced.** All requests returned `200`, `status: completed`, with complete audio, up to 3.3× the output limit and 1.2× the input limit. Within the 35.5-minute call there is no drift: loudness −0.01 LU per 10 minutes, Lena's pitch −0.13 and Felix's +0.02 semitones per 10 minutes, the speaker level gap −0.06 dB per 10 minutes ([drift_limit_input_over_8192_tokens.json](../../tests/fixtures/wikipedia/spikes/s4_results/drift_limit_input_over_8192_tokens.json)). Speaking rate is 17.0 characters per second of audio in all three, so **15,000 characters are about 14.7 minutes**.

So a single call could synthesise a whole episode. Recommendation: **still use sections**, for these reasons:
1. **Retries:** the design resynthesises only the affected section after a Whisper finding (up to 2× per section). A section of 6 minutes costs about $0.07 to redo, the whole episode $0.22, and a retry of the whole episode changes audio that was fine.
2. **Documented limits:** relying on behaviour that the model page contradicts is a risk. At 6,000 characters (about 5.9 minutes, about 8,800 output tokens, about 1,600 input tokens) a section stays at 54 % of the documented output limit, which also covers the slowest rate seen (16.5 characters per second) with margin.
3. **Cost of sections:** a seam is nearly free, as shown next.

**Strategy:** `tts_section_max_chars = 6000`. Split into the fewest sections of at most that size, balanced, at turn boundaries (`split_sections` in the spike script; to become a pure function with unit tests in step 9). A 16,000-character episode gives 3 sections of about 5,400 characters. Rate limit: an episode needs about 3 calls plus retries, far below 10 per minute.

## Seams

Full episode (16,343 characters), 3 sections, joined ([episode_3sections.json](../../tests/fixtures/wikipedia/spikes/s4_results/episode_3sections.json)). Each section is a separate call with the same voices.

| Between | Loudness step | Lena pitch | Felix pitch | Speaking rate |
|---|---:|---:|---:|---:|
| sections 0 and 1 | +0.7 LU | −0.71 semitones | +1.04 | +0.1 % |
| sections 1 and 2 | 0.0 LU | +0.43 | −0.13 | −4.2 % |
| **Normal variation inside a section** (std of 30 s windows) | 0.6 LU | 0.66 | 1.18 | |

Every seam step lies inside the normal variation of 30-second windows within a section. By these measures there is nothing at a seam that does not also happen within a section. Further evidence: two-minute dialogs from independent calls with the same voices differ by at most 1.5 LU in loudness (−19.6 to −21.1 LUFS).

Joined and loudness-normalised, the whole episode has WER 0.065, no worse than the single calls (0.060–0.072).

**What the measurements cannot show:** a change in intonation style or breathing between calls. The sample `seam_excerpt_3sec.wav` (12 s either side of a join) is for that.

**Edge silence:** sections start with 0.20–0.25 s of silence and end with 0.2–0.8 s. Joined with a plain 0.5 s pause, the gap at a seam is therefore 0.9–1.5 s, longer than a turn gap. Proposal for step 9: trim silence at section edges to 0.15 s, then insert a 0.4 s pause. I did not test the result by ear.

## Fixed intro and outro

Synthesised once with the same voices ([intro_outro.json](../../tests/fixtures/wikipedia/spikes/s4_results/intro_outro.json)). Outro (34 s): Lena +0.05 and Felix +1.23 semitones against the episode, WER 0.031. The intro (8 s) has only about 3 s of Felix's voice, so its pitch figure (−2.4 semitones) is not reliable. The audio contains the greeting; the transcript made with the bug lacked it, and at the correct rate it is complete (WER 0.045). Intro and outro are not part of the STT loop in the design, so this has no consequence there.

## Findings for later steps

- **Loudness (step 9):** the output is about −20 LUFS with peaks around −1 dBFS. Normalising to the proposed **−16 LUFS needs +4 dB, which pushes peaks to about +3 dBFS**. A plain gain change clipped 6,963 samples in the joined episode (0.03 %). Without a limiter the loudest possible level is about −20 LUFS. Step 9 needs a true-peak limiter (for example ffmpeg `loudnorm=I=-16:TP=-1.5`, preinstalled on GitHub runners, not in the dev container), or a lower target. The −16 LUFS value is still marked "proposal, not yet decided" in the config.
- **Whisper (step 10 and S5):** retracted. I had reported that Whisper hallucinates repeated text after the speech, drops short first sentences and garbles foreign names, and recommended VAD settings. These observations came from the sample-rate bug. At the correct rate, default and VAD settings give identical results on the three voices I re-tested. Not yet re-checked: numbers (Whisper writes digits, the script spells them out) and foreign names such as SETE; step 10's normalisation has to handle both anyway. Whisper speed was **not** measured here; S5 does that.
- **Rate limit:** the TTS client must read `Please retry in Ns` from a `429` and wait. TTS error handling is otherwise not covered by the design table; observed: `429 too_many_requests`; not observed: all other codes.

## Decisions (project owner, 2026-11-18)

1. **Voices:** the project owner listened to the samples and approved Lena = `de-de-advisor-3`, Felix = `de-de-tutor-4`.
2. **Loudness:** target −16 LUFS **with a true-peak limiter** (ceiling −1.5 dBTP). Without the limiter the peaks would clip (see findings above). Step 9 adds a test for loudness and peak ceiling of the output.
3. **Design document:** the proposed changes were applied.

## Changes applied

- [config/pipeline.toml](../../config/pipeline.toml): `tts_voice_lena = "de-de-advisor-3"`, `tts_voice_felix = "de-de-tutor-4"`, `tts_section_max_chars = 6000`, `target_loudness_lufs = -16` (decided), new `true_peak_ceiling_dbtp = -1.5`.
- [Design document](../wikipedia-podcast-pipeline.md), section Audio: sections of at most 6,000 characters (documented API limits are respected but not enforced, sections exist for cheap retries), new bullets for the rate limit (10 requests per minute, wait time from the `429` response) and for loudness. The open point "Stimmen" is removed.
- [Persona document](../persona.md): voice names for Lena and Felix.
- [Plan](../implementation-plan.md), step 9: loudness −16 LUFS with limiter and the test.

## Still open

- **Remaining WER of 5–9 % in dialogs** (0.03 for the outro) is not analysed. Likely causes: foreign names (SETE heard as SETI, "Champ de Mars"), and numbers that the script spells out but Whisper writes as digits (numerals are removed from the metric by a heuristic). Step 10's normalisation and LLM decision have to cope with both; S5 measures Whisper speed.
- **Gap at section joins:** edge silences add up to 0.9–1.5 s with a 0.5 s pause. Proposal for step 9: trim edge silence to 0.15 s, then insert 0.4 s. Not tested by ear; a decision for step 9.
