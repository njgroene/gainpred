"""The baseline's verdict rule (gainpred.baseline): order and thresholds."""
from gainpred.baseline import verdict


def cell(**kw):
    base = {"new": True, "change": 0.0, "easy_change": 0.0, "gap": 0.0}
    return {**base, **kw}


def test_pays_off_at_new_length():
    assert verdict(cell(change=30), g=20, h=3, d=30) == "pays off"


def test_far_below_best_is_harm_even_when_up():
    # harm is checked first: up 40 at a new length but 50 below the best teacher's probe
    assert verdict(cell(change=40, gap=50), g=20, h=3, d=30) == "does harm"


def test_learned_length_down_is_harm():
    assert verdict(cell(new=False, change=-5), g=20, h=3, d=30) == "does harm"


def test_easy_drop_is_harm():
    assert verdict(cell(change=30, easy_change=-4), g=20, h=3, d=30) == "does harm"


def test_otherwise_nothing():
    assert verdict(cell(change=10), g=20, h=3, d=30) == "does nothing"
    assert verdict(cell(new=False, change=2), g=20, h=3, d=30) == "does nothing"
