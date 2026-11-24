from interfaces import Scorer
from map import build_map


def test_interfaces_uses_the_real_scorer():
    assert Scorer.__module__ == "visibility"


def test_direction_and_walls_matter_on_tiny():
    s = Scorer(build_map("tiny"))
    sizes = [int(s.masks[i].sum()) for i, c in enumerate(s.candidates) if c[:2] == (8, 7)]
    assert len(set(sizes)) > 1      # directions differ
    assert max(sizes) < 60          # not a 13 x 13 square