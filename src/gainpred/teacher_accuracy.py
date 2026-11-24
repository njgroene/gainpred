"""How accurate each teacher's kept answers are (protocol amendment 24, Part B).

Defined after all experiments; inference only, on saved models; nothing is trained and no
logged file is touched. It never enters the verdict, and P9b is not rescored.

Usage::

    uv run python -m gainpred.teacher_accuracy             # list the 18 runs and their cases
    uv run python -m gainpred.teacher_accuracy --launch    # one Modal call per run
    uv run python -m gainpred.teacher_accuracy --fetch     # -> results/teacher_accuracy.csv

The question
============
The checks never look at the right answer: novelty (check 1) asks whether the teacher's kept
answer differs from what the student already says, the confirmed share (check 2b) whether a
second split builds the same answer. Here we *do* look, with plain arithmetic, at the kept
answers of every evaluation case (seeds 2-4) of the teachers that split -- the no-carry rule,
the agreement rule, no selection, no selection after the no-carry rule, and the weak starting
models (all no-carry rule) -- to see how often a kept answer is wrong, and how often a wrong
one is confirmed anyway.

How
===
Exactly as check 3 (amendment 22, ``gainpred.check3``): per run, round and length, the model as
it was *before* that round's training answers the round's check sample, redrawn from the
same seeded stream (2,000 problems, the cuts the run used). Greedy answers do not depend on
any random state, so the teacher's decisions and the logged kept count, novelty and confirmed
share must come out *exactly* as logged; ``fetch`` refuses otherwise.

Every problem's answers are kept (``results/teacher_accuracy/<call>.answers.jsonl.gz``): the
operands, the kept answer and the cut it came from, every cut's built answer, and the
student's direct answer. Every quantity below is computed locally from those rows, so a later
question needs no new run.

Quantities, over a case's kept answers (``case_row``)
=====================================================
* teacher accuracy -- share equal to the true sum;
* confirmed share -- share equal to the built answer of at least one *other* cut (check 2b);
* wrong-but-confirmed -- share that are wrong AND confirmed (``wrong_confirmed``);
* accuracy among confirmed answers, and among unconfirmed answers;
* student accuracy -- the student's greedy direct answer on the same (kept) problems, and on
  all 2,000.

By logic, teacher accuracy >= confirmed share - wrong-but-confirmed (the confirmed answers that
are not wrong are right). A failure would be a bug, not a finding; ``fetch`` checks it.
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from dataclasses import fields
from pathlib import Path

CELLS = Path("results/p9b_cells.csv")
OUT_DIR = Path("results/teacher_accuracy")
CSV = Path("results/teacher_accuracy.csv")
EVALUATION_SEEDS = (2, 3, 4)
SPLITTING = ("no_carry", "agreement", "no_selection")

KINDS = ("No-carry rule", "Agreement rule", "No selection", "No selection after the no-carry rule",
         "Weak starting models")


def kind(run: str, teacher: str) -> str:
    """The kind of teacher, as the paper names it (the weak starting models are their own kind)."""
    if "weak" in run:
        return "Weak starting models"
    if "after-nocarry" in run:
        return "No selection after the no-carry rule"
    return {"no_carry": "No-carry rule", "agreement": "Agreement rule", "no_selection": "No selection",
            "sample_and_vote": "Sample and vote"}[teacher]


def plan() -> list[dict]:
    """One Modal call per evaluation run of a splitting teacher, with its P9b cases (round, digits)."""
    runs: dict[str, dict] = {}
    for r in csv.DictReader(CELLS.open()):
        if int(r["seed"]) in EVALUATION_SEEDS and r["teacher"] in SPLITTING:
            runs.setdefault(r["run"], {"name": f"teacheracc-{r['run']}", "run": r["run"], "seed": int(r["seed"]),
                                       "teacher": r["teacher"], "kind": kind(r["run"], r["teacher"]),
                                       "cases": []})["cases"].append((int(r["round"]), int(r["digits"])))
    return list(runs.values())


def text(answer) -> str:
    """An answer as text: a number as its digits, an unreadable answer as "?" + its raw text
    (so two unreadable answers never compare equal to a number, and are kept apart)."""
    from gainpred.formats import Unreadable
    return "?" + answer.raw if isinstance(answer, Unreadable) else str(answer)


# ---------------------------------------------------------------- Modal side


def redraw(original: str, out_run: str, runs_dir: Path, data_dir: Path, device,
           only: list[tuple[int, int]]) -> None:
    """The ``only`` cases of ``original``: redraw, run the teacher, record every answer.

    Writes ``runs/<out_run>/attempt_<n>/teacher_accuracy.jsonl``: a ``config`` event, one
    ``case`` event per (round, length) with the redrawn and logged checks and every problem's
    answers, a ``round_done`` per round, then ``done``.
    """
    import torch

    from gainpred.check3 import redrawn_rounds, teacher_with_pieces
    from gainpred.checks import check_1_and_teachable, confirmed_share
    from gainpred.probe import attempt_dir, events
    from gainpred.rounds import RoundConfig
    from gainpred.teachers import build_all, cuts_of, teacher_key

    src = runs_dir / original
    config = json.loads((src / "config.json").read_text())
    known = {f.name for f in fields(RoundConfig)}
    cfg = RoundConfig(**{k: v for k, v in config["rounds"].items() if k in known})
    teacher = teacher_key(cfg.teacher)
    first_round = (config.get("start_round") or 0) + 1
    logged = {(e["round"], e["digits"]): e for e in events(src) if e["event"] == "checks"}
    out, finished = attempt_dir(runs_dir, out_run)
    with open(out / "teacher_accuracy.jsonl", "a") as f:
        def log(e: dict) -> None:
            f.write(json.dumps(e) + "\n")
            f.flush()

        log({"event": "config", "of": original, "teacher": teacher, "seed": cfg.seed, "only": only})
        for k, start, solver, cells in redrawn_rounds(cfg, config, src, runs_dir, data_dir, device, first_round,
                                                      logged, finished, {tuple(c) for c in only}):
            for d, sample in cells:
                t0 = time.monotonic()
                decisions = [dec for dec, _ in teacher_with_pieces(teacher, sample, solver)]
                # The checks as run_checks computes them (greedy direct answers; every cut built once).
                direct = solver(sample)
                check_1, teachable = check_1_and_teachable(decisions, direct)
                built = build_all(sample, [cuts_of(p) for p in sample], solver)
                check_2b = confirmed_share(decisions, built)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                orig = logged[(k, d)]
                log({"event": "case", "round": k, "digits": d,
                     "redrawn": {"kept": sum(dec.kept is not None for dec in decisions), "check_1": check_1,
                                 "check_2b": check_2b, "teachable": teachable},
                     "logged": {key: orig.get(key) for key in ("problems", "kept", "check_1", "check_2b", "teachable")},
                     "seconds": time.monotonic() - t0, "start_checkpoint": str(start.relative_to(runs_dir)),
                     "problems": [{"a": p.a, "b": p.b, "kept": dec.kept, "source": dec.source,
                                   "built": [text(b.answer) for b in bs], "direct": text(a)}
                                  for p, dec, bs, a in zip(sample, decisions, built, direct, strict=True)]})
            log({"event": "round_done", "round": k})
        log({"event": "done"})


# ---------------------------------------------------------------- local: per-case quantities


def case_row(problems: list[dict]) -> dict:
    """The quantities of one case from its problems' answers (shares of kept answers, 0-1)."""
    kept = [p for p in problems if p["kept"] is not None]
    n = len(kept)

    def right(p):
        return p["kept"] == p["a"] + p["b"]

    def confirmed(p):
        return any(b == str(p["kept"]) for j, b in enumerate(p["built"]) if j != p["source"])

    conf = [p for p in kept if confirmed(p)]
    unconf = [p for p in kept if not confirmed(p)]

    def share(xs, of):
        return len(xs) / len(of) if of else None

    return {"problems": len(problems), "kept": n,
            "teacher_accuracy": share([p for p in kept if right(p)], kept),
            "confirmed_share": share(conf, kept),
            "wrong_confirmed": share([p for p in conf if not right(p)], kept),
            "accuracy_confirmed": share([p for p in conf if right(p)], conf),
            "accuracy_unconfirmed": share([p for p in unconf if right(p)], unconf),
            "unconfirmed": len(unconf),
            "novelty": share([p for p in kept if p["direct"] != str(p["kept"])], kept),
            "student_accuracy": share([p for p in kept if p["direct"] == str(p["a"] + p["b"])], kept),
            "student_accuracy_all": share([p for p in problems if p["direct"] == str(p["a"] + p["b"])], problems)}


def consistent(row: dict) -> bool:
    """Teacher accuracy >= confirmed share - wrong-but-confirmed (true by logic; tiny float slack)."""
    if row["kept"] == 0:
        return True
    return row["teacher_accuracy"] >= row["confirmed_share"] - row["wrong_confirmed"] - 1e-12


def answers_path(name: str) -> Path:
    return OUT_DIR / f"{name}.answers.jsonl.gz"


def load_answers(name: str) -> list[dict]:
    """The saved ``case`` events of one call, with every problem's answers."""
    with gzip.open(answers_path(name), "rt") as f:
        return [json.loads(line) for line in f]


def launch() -> None:
    import subprocess
    for p in plan():
        subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "teacher-accuracy", p["name"], "--of",
                        p["run"], "--only", ",".join(f"{k}:{d}" for k, d in p["cases"])])


def fetch() -> list[dict]:
    """Download every call, check it reproduces the logged checks exactly, write the CSV.

    Refuses (raises) if a call is missing a case, if any redrawn kept count, novelty or
    confirmed share differs from the logged one, or if the consistency check fails.
    """
    import subprocess

    OUT_DIR.mkdir(exist_ok=True)
    rows, bad = [], []
    for p in plan():
        subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{p['name']}", "runs/"],
                       capture_output=True, check=True)
        lines = [json.loads(line) for log in sorted((Path("runs") / p["name"]).glob("attempt_*/teacher_accuracy.jsonl"))
                 for line in log.read_text().splitlines()]
        if not any(e["event"] == "done" for e in lines):
            raise ValueError(f"{p['name']}: not done")
        # A retried call (a new attempt) redoes its cases; the last attempt's copy is used.
        cases = list({(e["round"], e["digits"]): e for e in lines if e["event"] == "case"}.values())
        if sorted((e["round"], e["digits"]) for e in cases) != sorted(p["cases"]):
            raise ValueError(f"{p['name']}: {len(cases)} cases, {len(p['cases'])} wanted")
        # The log without the answers (they go to the .gz), for the run list and spending total.
        (OUT_DIR / f"{p['name']}.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in e.items() if k != "problems"}) + "\n" for e in lines))
        with gzip.open(answers_path(p["name"]), "wt") as f:
            f.writelines(json.dumps(e) + "\n" for e in cases)
        for e in sorted(cases, key=lambda e: (e["round"], e["digits"])):
            red, log = e["redrawn"], e["logged"]
            row = {"run": p["run"], "kind": p["kind"], "teacher": p["teacher"], "seed": p["seed"],
                   "round": e["round"], "digits": e["digits"], **case_row(e["problems"]),
                   "logged_kept": log["kept"], "logged_novelty": log["check_1"], "logged_confirmed": log["check_2b"]}
            if (red["kept"], red["check_1"], red["check_2b"]) != (log["kept"], log["check_1"], log["check_2b"]) or \
                    (row["kept"], row["novelty"], row["confirmed_share"]) != (log["kept"], log["check_1"], log["check_2b"]):
                bad.append(f"{p['run']} round {e['round']} {e['digits']} digits: redrawn {red} / local "
                           f"{row['kept'], row['novelty'], row['confirmed_share']}, logged {log}")
            if not consistent(row):
                bad.append(f"{p['run']} round {e['round']} {e['digits']} digits: consistency check fails {row}")
            rows.append(row)
    if bad:
        raise ValueError("teacher accuracy does not reproduce the logged checks:\n" + "\n".join(bad))
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{CSV}: {len(rows)} cases from {len(plan())} runs; all reproduce the logged checks")
    return rows


def load_rows() -> list[dict]:
    """results/teacher_accuracy.csv with numbers parsed (empty -> None)."""
    def num(v):
        return None if v == "" else float(v)
    out = []
    for r in csv.DictReader(CSV.open()):
        out.append({**r, **{k: int(r[k]) for k in ("seed", "round", "digits", "problems", "kept", "unconfirmed")},
                    **{k: num(r[k]) for k in ("teacher_accuracy", "confirmed_share", "wrong_confirmed",
                                              "accuracy_confirmed", "accuracy_unconfirmed", "novelty",
                                              "student_accuracy", "student_accuracy_all")}})
    return out


def predictions(rows: list[dict]) -> dict:
    """P1 and P2 (amendment 24), over the fully trained starting models: per seed and pooled.

    P1: wrong-but-confirmed <= 2% of kept answers in every case. P2: accuracy among confirmed
    answers >= 98% in every case (a case with no confirmed answer has no accuracy among them and
    is listed, not counted as held).
    """
    full = [r for r in rows if r["kind"] != "Weak starting models"]
    out = {}
    for label, rs in [*((f"seed {s}", [r for r in full if r["seed"] == s]) for s in EVALUATION_SEEDS),
                      ("pooled", full)]:
        p1 = [r for r in rs if r["wrong_confirmed"] is not None and r["wrong_confirmed"] > 0.02]
        p2 = [r for r in rs if r["accuracy_confirmed"] is None or r["accuracy_confirmed"] < 0.98]
        out[label] = {"cases": len(rs), "p1_fail": p1, "p2_fail": p2, "p1": not p1, "p2": not p2}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.launch:
        launch()
    elif a.fetch:
        fetch()
    else:
        for p in plan():
            print(p["name"], p["kind"], len(p["cases"]))
