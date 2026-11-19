# Spike S5: Whisper runtime on an Actions runner

**Question (plan, Phase 0):** how long does Whisper take for 15 minutes of audio on a GitHub Actions CPU runner (large vs. faster-whisper int8)?
**Output:** model size, cache setup.
**Result:**
- **Model:** `large-v3` with faster-whisper, `compute_type=int8`, 4 threads. It fits comfortably: 978.6 s (16.3 min) of audio take about 11 min, a real-time factor (RTF) of 0.66–0.68.
- **Cache:** `actions/cache` on `~/.cache/huggingface/hub`, key `whisper-large-v3`. It saves only about 22 s per run (model load 27.8 s cold, 5.4 s warm), so the cache is a convenience, not a requirement.
- **Smaller models were not measured.** `large-v3` already meets the time budget, so I did not run `medium` or `small`. The design says "start with the large model".

## Measurements

Runner: `ubuntu-latest`, 4 CPUs, 15 GB RAM. Test audio: the S4 Eiffelturm dialog repeated to 16.3 min, synthesized with Gemini TTS (voices `de-de-advisor-3` and `de-de-tutor-4`), 24 kHz, resampled to 16 kHz for Whisper. WER is measured against the reference script with numerals removed on both sides, as in S4.

| Run | Model load | Transcription | RTF | Peak RSS | WER | Job wall time |
|---|---|---|---|---|---|---|
| Cold (no cache) | 27.8 s | 649.7 s | 0.664 | 4,333 MB | 0.038 | ~12 min |
| Warm (cache restored) | 5.4 s | 666.4 s | 0.681 | 3,236 MB | 0.038 | ~12 min |

The 17 s difference in transcription time between the two runs is runner noise (the two jobs ran on different runner images, kernel 7.0.0-1019 and -1020). Both runs gave the same WER. Whether the peak RSS differs by 1.1 GB because of the cold download or for another reason I did not investigate.

## What follows for the pipeline

- Whisper adds about 11–12 min of runner time per episode (the job wall time includes checkout, `uv sync` and cache restore). The 120 min job timeout in the spike workflow is far above that.
- Peak memory is 3–4.3 GB, well under the 15 GB of the runner.
- An episode of 15 min is the target; the test audio is 16.3 min, so a real episode will be a little faster.
- The WER of 3.8 % is in line with S4 (5–9 % in dialogs, 3 % for the outro). Foreign names and spelled-out numbers still need the normalisation in step 10; this spike did not analyse the remaining errors.

## Not measured

- Other models (`medium`, `large-v3-turbo`, `small`) and other thread counts.
- Runner-to-runner variance beyond the two runs above.
- Audio from a full pipeline run, as opposed to the repeated S4 test dialog.

Code: [spikes/s5_whisper.py](../../spikes/s5_whisper.py), workflow [spike-s5-whisper.yml](../../.github/workflows/spike-s5-whisper.yml). Results: [tests/fixtures/wikipedia/spikes/s5_results/](../../tests/fixtures/wikipedia/spikes/s5_results/). The test audio is the release asset `spike-s5-audio` (not committed).
