"""Tests for the P9b scoring: outcomes under the fixed rules, verdicts at given levels."""

from gainpred.p9b import cells_of, give_verdicts, score_outcomes
from gainpred.report import RoundsRun


def study(acc8, acc9, acc10=0.0, acc11=0.0):
    out = {}
    for d, a in ((8, acc8), (9, acc9), (10, acc10), (11, acc11)):
        out[f"{d}/no-carry"] = out[f"{d}/carry"] = a
    return out


def run(name, teacher, evals, checks, easy=None):
    easy = easy or {r: {"3": 1.0, "7": 0.99, "all": 0.995} for r in range(3)}
    events = [{"event": "eval", "round": r, "set": "study", "accuracy": a} for r, a in evals.items()]
    events += [{"event": "eval", "round": r, "set": "easy", "accuracy": a} for r, a in easy.items()]
    events += [{"event": "checks", "round": r, "digits": d, **c} for (r, d), c in checks.items()]
    return RoundsRun(name, {"start_run": "start-seed1-authors", "rounds": {"teacher": teacher, "seed": 1}}, events)


CHECK = {"check_1": 1.0, "check_2b": 0.7, "check_2_kept": 0.35, "teachable": 0.38}


def test_outcomes_at_new_and_learned_lengths():
    good = run("a", "agreement", {0: study(0, 0), 1: study(0.99, 0.98), 2: study(0.99, 0.985, 0.97, 0.96)},
               {(1, 8): CHECK, (1, 9): CHECK, (2, 8): {**CHECK, "check_1": 0.01}})
    bad = run("b", "no_selection", {0: study(0, 0), 1: study(0.40, 0.02)}, {(1, 8): CHECK, (1, 9): CHECK})
    cells = cells_of(good) + cells_of(bad)
    score_outcomes(cells)
    out = {(c.run, c.round, c.digits): c.outcome for c in cells}
    assert out[("a", 1, 8)] == "paid off"            # 0 -> 99, best
    assert out[("b", 1, 8)] == "taught wrong answers"  # up, but 59 points below the best
    assert out[("b", 1, 9)] == "did nothing"         # up less than 3 points
    assert out[("a", 2, 8)] == "did nothing"         # already learned, unchanged
    assert out[("a", 2, 10)] == "paid off"           # new in round 2
    assert all(not c.new_length for c in cells if c.round == 2 and c.digits <= 9)


def test_easy_drop_is_harm_and_verdict_order():
    r = run("c", "no_selection", {0: study(0.9, 0.9), 1: study(0.9, 0.9)}, {(1, 8): {**CHECK, "check_2b": 0.2}},
            easy={0: {"3": 1.0, "all": 1.0}, 1: {"3": 0.9, "all": 0.9}})
    cells = cells_of(r)
    score_outcomes(cells)
    give_verdicts(cells, level_2b=0.5, level_1=0.1)
    c8 = next(c for c in cells if c.digits == 8)
    assert c8.outcome == "did harm" and c8.verdict == "does harm"  # 2b below the level comes first
