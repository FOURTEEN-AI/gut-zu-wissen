"""Pure helpers of the per-sentence completeness spike (spikes/s1_sentences.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spikes"))

from s1_sentences import completeness_ps, sentence_p, split_sentences, turn_decision  # noqa: E402

from gut_zu_wissen.judge import Decision  # noqa: E402
from gut_zu_wissen.segmentation import default_tokenizer  # noqa: E402


def test_split_sentences_keeps_original_text():
    text = "Wahnsinn! Das klingt fast unwirklich. Wie groß ist er, Lena?"
    assert split_sentences(text, default_tokenizer()) == [
        "Wahnsinn!",
        "Das klingt fast unwirklich.",
        "Wie groß ist er, Lena?",
    ]


def test_split_variant_takes_the_better_of_reaction_and_covered():
    answers = {"split_reaction_0": 0.97, "split_covered_0": 0.10}
    assert sentence_p("split", answers, 0) == 0.97


def test_completeness_ps_old_is_the_single_question():
    assert completeness_ps("old", {"vollstaendig": 0.8, "aussage_0": 0.99}, 3) == [0.8]


def test_turn_decision():
    assert turn_decision([0.99], [0.96, 0.97], 0.95) is Decision.SUPPORTED
    assert turn_decision([0.99], [0.96, 0.04], 0.95) is Decision.REJECTED
    assert turn_decision([0.99], [0.96, 0.90], 0.95) is Decision.UNCERTAIN
    assert turn_decision([], [0.96], 0.95) is Decision.SUPPORTED
