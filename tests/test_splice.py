import numpy as np

from gut_zu_wissen.audio import quietest_sample
from gut_zu_wissen.splice import affected_turns, blocks, boundary_cuts, turn_offsets
from gut_zu_wissen.verification import Deviation

SR = 1000


def dev(start: int, end: int) -> Deviation:
    return Deviation(0, "x", "y", "", "", script_start=start, script_end=end)


def test_turn_offsets_count_normalised_words():
    assert turn_offsets(["Es waren neun.", "Echt?", "Ja, 324 Meter."]) == [0, 3, 4, 7]


def test_turn_offsets_refuse_a_number_split_across_turns():
    assert turn_offsets(["Es waren 10", "000 Leute."]) is None


def test_affected_turns_of_replacements_and_insertions():
    offsets = [0, 3, 4, 7]
    assert affected_turns([dev(1, 2)], offsets) == {0}
    assert affected_turns([dev(2, 4)], offsets) == {0, 1}  # spans a border
    assert affected_turns([dev(3, 3)], offsets) == {0, 1}  # extra words at a border
    assert affected_turns([dev(5, 5)], offsets) == {2}
    assert affected_turns([dev(7, 7)], offsets) == {2}  # extra words at the end
    assert affected_turns([], offsets) == set()


def test_quietest_sample_lies_in_the_quietest_frame():
    y = np.ones(1000)
    y[600:620] = 0
    assert 600 <= quietest_sample(y, SR, 0.5, 0.8, 0.01) < 620
    assert quietest_sample(y, SR, 0.5, 0.505, 0.01) == 502  # shorter than a frame: middle


def test_boundary_cuts_need_both_word_times_and_a_gap():
    y = np.ones(10 * SR)
    y[2200:2210] = 0
    offsets = [0, 2, 3, 4, 5]
    times = [(0, 1), (1, 2), (2.5, 3), None, (5, 6)]
    cuts = boundary_cuts(times, offsets, y, SR, 0.01)
    assert cuts[0] is not None and 2200 <= cuts[0] < 2210  # between 2.0 and 2.5 s
    assert cuts[1] is None and cuts[2] is None  # word 3 has no time
    overlapping = [(0, 1), (1, 2.6), (2.5, 3), (3, 4), (5, 6)]
    assert boundary_cuts(overlapping, offsets, y, SR, 0.01)[0] is None


def test_blocks_widen_to_the_next_border_that_can_be_cut_and_merge():
    assert blocks({1}, 4, [True, True, True]) == [(1, 2)]
    assert blocks({1}, 4, [True, False, True]) == [(1, 3)]
    assert blocks({1}, 4, [False, True, True]) == [(0, 2)]
    assert blocks({0, 2}, 4, [True, False, True]) == [(0, 3)]
    assert blocks({0, 3}, 4, [True, True, True]) == [(0, 1), (3, 4)]
    assert blocks({1}, 3, [False, False]) == [(0, 3)]
