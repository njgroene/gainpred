"""What the control-arm students answer on long crossing runs (protocol amendment 23).

Defined after all experiments; inference only, on saved models; nothing is trained and no
logged file is touched.

The question
============
In the control arm, the starting training includes short strings with a run of 7-8 inside a
single piece. After the withheld round (join at the cut, trained without crossing runs whose
left part is 3-4), accuracy on crossing runs of 7 fell sharply, although every answer the
teacher showed was right. The teacher never shows an answer above 6 on long strings, so the
guess is that the student learned "a long string's answer is at most 6". Recorded before any
answer was computed: after the round, at a crossing run of 7, at least half of the control-arm
students' wrong answers give the right symbol with length 6.

Models, per starting model s = 0-4
==================================
a  control arm, before the round: the round's starting model (round 0 of the -longruns run)
b  control arm, after the round
c  withheld version without the control arm, after the round
d  edges-differ rule on the full pool, after the round

Identity check
==============
Each model answers the whole study test set exactly as the run's own evaluation did (all
10,000 rows, file order, batches of 1,000, greedy), so the batches pad the same way. The
accuracy on the groups with a crossing run of 4-8 must equal the logged one exactly;
``fetch`` refuses otherwise.

Words
=====
A **crossing run** is the run across the split between the two pieces (``joined_length_n``
groups, length a + b, a symbols in the left piece and b in the right). The string's correct
answer is its longest run, which is usually the crossing run but can be a longer run inside a
piece. **Pick the longer** answers with the longer piece's own longest run, the left piece's on
a tie (the teacher of that name, given right piece answers).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROLES = {"a": "Control arm, before the round", "b": "Control arm, after the round",
         "c": "Withheld version, after the round", "d": "Edges-differ rule, full pool, after the round"}
SEEDS = range(5)
EVALUATION_SEEDS = (2, 3, 4)
CHECK_LENGTHS = range(4, 9)
SAVED_LENGTHS = range(5, 9)
RESULTS = Path("results/control_answers.jsonl")


def models(seeds=SEEDS) -> list[dict]:
    """The 20 models: role, seed, the run whose log holds its evaluation, and that round."""
    out = []
    for s in seeds:
        control, plain, edges = (f"runs-join_at_the_cut-withheld-seed{s}-longruns",
                                 f"runs-join_at_the_cut-withheld-seed{s}-u30000", f"runs-edges_differ-seed{s}-u30000")
        out += [{"role": "a", "seed": s, "run": control, "round": 0},
                {"role": "b", "seed": s, "run": control, "round": 1},
                {"role": "c", "seed": s, "run": plain, "round": 1},
                {"role": "d", "seed": s, "run": edges, "round": 1}]
    return out


def checkpoint(runs_dir: Path, m: dict) -> Path:
    """Round 0: the checkpoint the run started from (its config); round 1: the round's last one."""
    if m["round"] == 0:
        return runs_dir / json.loads((runs_dir / m["run"] / "config.json").read_text())["start_checkpoint"]
    return sorted((runs_dir / m["run"] / f"round_{m['round']}" / "checkpoints").glob("step_*.pt"))[-1]


def logged_accuracy(runs_dir: Path, m: dict) -> dict[str, float]:
    log = [json.loads(line) for line in (runs_dir / m["run"] / "rounds.jsonl").read_text().splitlines()]
    return next(e["accuracy"] for e in log if e["event"] == "eval" and e["set"] == "study" and e["round"] == m["round"])


def answer_text(answer) -> str:
    """A Run as "s|n"; an unreadable answer as its raw text, prefixed with "?"."""
    from gainpred.formats import Run, run_answer_text
    return run_answer_text(answer) if isinstance(answer, Run) else "?" + answer.raw


def answer_all(name: str, runs_dir: Path, data_dir: Path, device, seeds=SEEDS) -> None:
    """Modal side: every model answers the study set; one ``model`` event each, then ``done``.

    Writes ``runs/<name>/attempt_<n>/answers.jsonl``; a retried call starts a new attempt and
    skips models an earlier attempt finished.
    """
    import torch  # noqa: F401

    from gainpred.data import load
    from gainpred.run_train import answer_runs_greedily, question_of
    from gainpred.train import load_model

    records = load(data_dir / "test_study.jsonl")
    questions = [question_of(r) for r in records]
    base = runs_dir / name
    base.mkdir(parents=True, exist_ok=True)
    earlier = sorted(base.glob("attempt_*/answers.jsonl"))
    done = {(e["role"], e["seed"]) for log in earlier for e in map(json.loads, log.read_text().splitlines())
            if e["event"] == "model"}
    out = base / f"attempt_{len(earlier) + 1}"
    out.mkdir()
    with open(out / "answers.jsonl", "a") as f:
        for m in models(seeds):
            if (m["role"], m["seed"]) in done:
                continue
            path = checkpoint(runs_dir, m)
            model = load_model(path, device)
            model.eval()
            answers = answer_runs_greedily(model, questions, 1_000, device)
            f.write(json.dumps({"event": "model", **m, "checkpoint": str(path.relative_to(runs_dir)),
                                "answers": [answer_text(a) for a in answers]}) + "\n")
            f.flush()
            print(m["role"], m["seed"], "answered")
        f.write(json.dumps({"event": "done"}) + "\n")


# ---------------------------------------------------------------- local: check, save, tabulate


def piece_facts(record: dict) -> dict:
    """The facts each saved answer carries: the crossing run's parts and each piece's longest run."""
    from gainpred.formats import longest_run, run_answer_text
    left, right = record["string"][: record["join"]], record["string"][record["join"]:]
    ll, rl = longest_run(left), longest_run(right)
    return {"left_part": record["a"], "right_part": record["b"], "left_longest": run_answer_text(ll),
            "right_longest": run_answer_text(rl),
            "pick_the_longer": run_answer_text(ll if ll.length >= rl.length else rl)}


def identity_check(events: list[dict], records: list[dict], runs_dir: Path) -> list[str]:
    """Models whose accuracy on crossing runs of 4-8 differs from the logged one (should be none)."""
    bad = []
    for e in events:
        logged = logged_accuracy(runs_dir, e)
        for n in CHECK_LENGTHS:
            group = f"joined_length_{n}"
            rows = [i for i, r in enumerate(records) if r["group"] == group]
            now = sum(e["answers"][i] == records[i]["answer"] for i in rows) / len(rows)
            if now != logged[group]:
                bad.append(f"{e['role']} seed {e['seed']} {group}: logged {logged[group]}, now {now}")
    return bad


def fetch(name: str = "control-answers") -> list[dict]:
    """Download, check identity, write ``results/control_answers.jsonl`` (one row per answer)."""
    import subprocess

    from gainpred.data import load
    from gainpred.reproduction import wall

    subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{name}", "runs/"],
                   capture_output=True, check=True)
    lines = [json.loads(line) for log in sorted((Path("runs") / name).glob("attempt_*/answers.jsonl"))
             for line in log.read_text().splitlines()]
    events = [e for e in lines if e["event"] == "model"]
    if sorted((e["role"], e["seed"]) for e in events) != sorted((m["role"], m["seed"]) for m in models()):
        raise ValueError(f"{len(events)} models answered, {len(models())} wanted")
    records = load(Path("data/longest_run/test_study.jsonl"))
    bad = identity_check(events, records, Path("runs"))
    if bad:
        raise ValueError("re-scored accuracy differs from the logged one:\n" + "\n".join(bad))
    rows = []
    for e in events:
        for i, r in enumerate(records):
            if r["group"] in {f"joined_length_{n}" for n in SAVED_LENGTHS}:
                rows.append({"role": e["role"], "seed": e["seed"], "run": e["run"], "round": e["round"],
                             "problem": i, "group": r["group"], "crossing": r["a"] + r["b"], "string": r["string"],
                             "correct": r["answer"], **piece_facts(r), "answer": e["answers"][i]})
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw = Path("results/control_answers") / f"{name}.jsonl"
    raw.parent.mkdir(exist_ok=True)
    # The raw log without the answers (they are in RESULTS), plus the wall time for the spending total.
    raw.write_text("".join(json.dumps({k: v for k, v in e.items() if k != "answers"}) + "\n" for e in lines)
                   + json.dumps(wall(name)) + "\n")
    print(f"{RESULTS}: {len(rows)} answers from {len(events)} models; identity check passed")
    return rows


CATEGORIES = ("correct", "right symbol, length 6", "right symbol, other wrong length", "anything else")


def category(row: dict) -> tuple[str, int | None]:
    """(category, length): the length is given for "right symbol, other wrong length"."""
    if row["answer"] == row["correct"]:
        return "correct", None
    if row["answer"].startswith("?"):
        return "anything else", None
    symbol, length = row["answer"].split("|")
    if symbol != row["correct"].split("|")[0]:
        return "anything else", None
    return ("right symbol, length 6", None) if length == "6" else ("right symbol, other wrong length", int(length))


def tabulate(rows: list[dict]) -> dict:
    """Counts per category, the wrong-length distribution and the pick-the-longer count."""
    cats = Counter(category(r)[0] for r in rows)
    lengths = Counter(category(r)[1] for r in rows if category(r)[0] == "right symbol, other wrong length")
    return {"n": len(rows), **{c: cats[c] for c in CATEGORIES}, "lengths": dict(sorted(lengths.items())),
            "pick the longer": sum(r["answer"] == r["pick_the_longer"] for r in rows)}


def load_rows() -> list[dict]:
    return [json.loads(line) for line in RESULTS.read_text().splitlines()]


def prediction(rows: list[dict], seeds) -> tuple[int, int, bool]:
    """Amendment 23's prediction on model b at a crossing run of 7: (length-6 wrong, all wrong, held)."""
    picked = [r for r in rows if r["role"] == "b" and r["seed"] in seeds and r["crossing"] == 7]
    t = tabulate(picked)
    wrong = t["n"] - t["correct"]
    return t["right symbol, length 6"], wrong, wrong > 0 and 2 * t["right symbol, length 6"] >= wrong


if __name__ == "__main__":
    fetch()
