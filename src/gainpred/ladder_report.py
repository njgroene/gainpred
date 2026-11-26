"""The ladder's report (``results/ladder/report.md``), generated from the fetched logs only.

Usage: ``uv run python -m gainpred.ladder_report`` (after ``python -m gainpred.ladder fetch``).
Every number comes from ``results/ladder/runs/*`` and ``results/ladder/launches.jsonl``.

Layout: the calls of every seed; per seed, its seed model, its four arms (with every
``diverged`` event and redo of amendment 30 next to the first attempt) and what arm 4 built and
skipped; the smoke test; and at the end a summary over every seed that has run.

Which test of a round counts (amendment 30): its redo's if the round was redone, else the last
test logged for it. A continuation call (``ladder_s5_arm3:redo8``) appends to the arm's log; the
events up to the first ``call_done`` are the original call's.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gainpred.ladder import (ARM_NAMES, CAP_DOLLARS, DOLLARS_PER_HOUR, LADDER_RESULTS, LENGTHS, STAGES,
                             records, round_lengths)

RUNS = LADDER_RESULTS / "runs"
SEEDS = (5, 6, 7)
PARTS = ("sec_check", "sec_build", "sec_train", "sec_test")


def events(run: str) -> list[dict]:
    path = RUNS / run / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def of(ev: list[dict], kind: str, **match) -> list[dict]:
    return [e for e in ev if e["event"] == kind and all(e.get(k) == v for k, v in match.items())]


def pct(x, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def when(t: float | None) -> str:
    return "-" if t is None else datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def seeds_run() -> list[int]:
    """The seeds with at least one launched call (seed model or arm)."""
    return [s for s in SEEDS if any(r["run"].startswith(f"ladder_s{s}_") for r in records())]


def original_part(ev: list[dict]) -> list[dict]:
    """The events of the arm's first call (up to and including its first ``call_done``)."""
    for i, e in enumerate(ev):
        if e["event"] == "call_done":
            return ev[:i + 1]
    return ev


def counting_tests(ev: list[dict]) -> dict[int, dict]:
    """{round: acc} of the test that counts: the redo's if the round was redone, else the last."""
    out: dict[int, dict] = {}
    redone: set[int] = set()
    for e in of(ev, "test"):
        if e.get("redo"):
            out[e["round"]] = e["acc"]
            redone.add(e["round"])
        elif e["round"] not in redone:
            out[e["round"]] = e["acc"]
    return out


def first_attempt_tests(ev: list[dict]) -> dict[int, dict]:
    return {e["round"]: e["acc"] for e in of(ev, "test") if not e.get("redo")}


def stages_of(ev: list[dict]) -> dict[int, bool]:
    return {e["stage"]: e["learned"] for e in of(ev, "stage")}


def call_file(run: str, segment: str) -> dict:
    path = RUNS / run / "calls" / f"{segment}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def dollars_of(secs: float | None) -> float | None:
    return None if secs is None else secs / 3600 * DOLLARS_PER_HOUR


# ---------------------------------------------------------------- calls and spending

def spent(seed: int) -> float:
    total = 0.0
    for r in records():
        if r["run"].startswith(f"ladder_s{seed}_"):
            d = dollars_of(call_file(r["run"], r["key"].split(":", 1)[1] or "main").get("wall_seconds"))
            total += d or 0.0
    return total


def calls_section() -> list[str]:
    rows = []
    for r in records():
        run, seg = r["run"], r["key"].split(":", 1)[1] or "main"
        c = call_file(run, seg)
        secs = c.get("wall_seconds")
        dollars = dollars_of(secs)
        rows.append([f"`{run}`", f"`{r['key']}`", when(c.get("started")), when(c.get("ended")),
                     "-" if secs is None else f"{secs:,.0f}", "-" if dollars is None else f"{dollars:.2f}",
                     c.get("reason", "not ended")])
    out = ["## Calls", "", "Every call launched for the ladder (`results/ladder/launches.jsonl`), crashed and "
           "relaunched calls included. Dollars = wall seconds of the call x $1.10 / 3,600 (decision D11).", ""]
    out += table(["Run", "Key", "Start", "End", "Wall seconds", "Dollars", "How it ended"], rows)
    out += [""]
    for seed in seeds_run():
        out += [f"**Random seed {seed}: ${spent(seed):.2f} of the ${CAP_DOLLARS:.2f} cap** (seed model and arm calls; "
                "the smoke test `ladder_smoke` is reported above but not counted, decision D11).", ""]
    return out


# ---------------------------------------------------------------- one seed

def seed_section(seed: int) -> list[str]:
    log = RUNS / f"ladder_s{seed}_seed" / "log.jsonl"
    if not log.exists():
        return ["## Seed model", "", "Not fetched.", ""]
    ev = [json.loads(line) for line in log.read_text().splitlines()]
    easy = [e for e in ev if e.get("event") == "eval" and e.get("set") == "easy"][-1]
    start = [e for e in ev if e.get("event") == "eval" and e.get("set") == "start"][-1]
    acc = {d: easy["accuracy"][d] for d in "34567"}
    out = ["## Seed model", "", f"`ladder_s{seed}_seed`, step {easy['step']:,}: accuracy on the existing easy test set "
           "(1,000 sums per length; greedy).", ""]
    out += table(["Length", *acc], [["Accuracy (%)", *(pct(v, 1) for v in acc.values())]])
    out += ["", f"* Entry check (D13, at least 98.0% at each length 3-7): "
            f"**{'passed' if all(v >= 0.98 for v in acc.values()) else 'FAILED'}**.",
            f"* Protocol P9 acceptance for starting models (at least 99% per length on the easy set): "
            f"**{'met' if all(v >= 0.99 for v in acc.values()) else 'not met'}**.",
            "* P9 also logs 8 and 9 digits asked directly (study test set, by carry across the study's cut 1): "
            + ", ".join(f"{k} {pct(v)}%" for k, v in start["accuracy"].items() if k != "all") + ".", ""]
    return out


def ending(ev: list[dict]) -> str:
    done = of(ev, "call_done")
    return done[-1]["reason"] if done else "not ended"


def redo_lines(ev: list[dict]) -> list[str]:
    """Amendment 30: every diverged event, the first attempt and the redo side by side."""
    div = of(ev, "diverged")
    out = []
    for c in of(ev, "start_check"):
        out += [f"Start check of the continuation (round {c['round']}; start weights tested against the test logged "
                f"after round {c['compared_with_round']}, at every length 3-"
                f"{max(map(int, c['diff_points']))}): largest difference {c['max_diff_points']:.2f} points, "
                f"**{'passed' if c['passed'] else 'FAILED'}** (limit 0.2).", ""]
    if not div:
        return out + ["Divergence rule (amendment 30): no `diverged` event.", ""]
    first, redo = first_attempt_tests(ev), {e["round"]: e["acc"] for e in of(ev, "test") if e.get("redo")}
    for e in div:
        r = e["round"]
        how = " (after the fact, in the continuation call; the first attempt is the original call's)" \
            if e.get("after_the_fact") else ""
        verdict = "not redone (log ends)" if r not in redo else (
            "redo diverged too: `diverged_twice`" if any(redo[r].get(str(d), {}).get("acc_total", 0) < 0.90 - 1e-12
                                                        for d in range(3, 8)) else "redo not diverged; the redo counts")
        out += [f"`diverged` in round {r}{how}: acc_total at 3-7 "
                + ", ".join(f"{d}: {pct(v)}%" for d, v in e["acc_short"].items()) + f". Outcome: {verdict}.", ""]
        lengths = sorted({int(d) for d in first.get(r, {})} | {int(d) for d in redo.get(r, {})})
        rows = [[d, pct(first.get(r, {}).get(str(d), {}).get("acc_total")),
                 pct(redo.get(r, {}).get(str(d), {}).get("acc_total"))] for d in lengths]
        out += table(["Length", f"round {r}, first attempt" + (" (original call)" if e.get("after_the_fact") else ""),
                      f"round {r}, redo"], rows) + [""]
    return out


def arm_sections(seed: int) -> list[str]:
    out = ["## Arms", ""]
    for arm in (1, 2, 3, 4):
        run = f"ladder_s{seed}_arm{arm}"
        ev = events(run)
        out += [f"### Arm {arm}: {ARM_NAMES[arm]} (`{run}`)", ""]
        if not ev:
            out += ["No log fetched.", ""]
            continue
        tests = counting_tests(ev)
        stages = stages_of(ev)
        learned = [s for s in sorted(stages) if stages[s]]
        out += [f"Ended: **{ending(ev)}** (all calls: " + ", ".join(e["reason"] for e in of(ev, "call_done")) + "). "
                f"Last round finished: {max((e['round'] for e in of(ev, 'round_done')), default=0)}. "
                f"Stages learned: {', '.join(map(str, learned)) or 'none'}"
                + (f"; not learned: {', '.join(str(s) for s in sorted(stages) if not stages[s])}" if
                   any(not v for v in stages.values()) else "") + ".", ""]
        if len(of(ev, "call_done")) > 1:
            orig = original_part(ev)
            o_stages = stages_of(orig)
            out += [f"Original call alone: ended **{ending(orig)}** after round "
                    f"{max((e['round'] for e in of(orig, 'round_done')), default=0)}; stages learned: "
                    f"{', '.join(str(s) for s in sorted(o_stages) if o_stages[s]) or 'none'}. The tables below count "
                    "the redo of a redone round; the original's first attempt is shown next to its redo.", ""]
        # acc_total after rounds 4, 8, 12, 16
        ends = [r for r in (4, 8, 12, 16) if r in tests]
        if ends:
            lengths = range(3, 2 * max(ends) + 8)
            rows = []
            for d in lengths:
                rows.append([d] + [pct(tests[r].get(str(d), {}).get("acc_total")) for r in ends])
            out += [f"`acc_total` (%) after rounds {', '.join(map(str, ends))}:", ""]
            out += table(["Length", *(f"after round {r}" for r in ends)], rows) + [""]
            low = []
            for s, lens in STAGES.items():
                r = 4 * s
                if r in tests:
                    v = min((tests[r].get(str(d), {}).get("acc_total", 0.0), d) for d in lens)
                    low.append([s, f"{lens.start}-{lens.stop - 1}", r, pct(v[0]), v[1],
                                "yes" if stages.get(s) else "no"])
            out += table(["Stage", "Lengths", "After round", "Lowest acc_total (%)", "at length", "Learned"],
                         low) + [""]
        out += redo_lines(ev)
        # per new length
        rows = []
        for c in of(ev, "check"):
            d, r = c["digits"], c["round"]
            b = of(ev, "build", round=r, digits=d)
            b = b[-1] if b else {}
            after = tests.get(r, {}).get(str(d), {})
            rows.append([d, r, pct(c["direct_acc"]), pct(c["cut1_acc"]), pct(c["confirmed_share"]),
                         pct(c["agreed_share"]), pct(c["agreed_acc"]), pct(c["novelty_arm"]), pct(c["novelty_agreed"]),
                         c["arm4_verdict"], f"{b['sums_drawn']:,}" if b else "-", f"{b['answers_kept']:,}" if b else "-",
                         pct(b.get("kept_acc")) if b else "-", pct(b.get("kept_carry_share")) if b else "-",
                         pct(after.get("acc_total")), pct(after.get("acc_carry")), pct(after.get("acc_no_carry"))])
        out += ["Per new length: the check before its round (2,000 fresh sums), the build, and the test after the "
                "round that taught it (all in %; `kept_acc`, `direct_acc`, `cut1_acc` and the accuracy among "
                "pair-agreed answers use correct answers and are for analysis only).", ""]
        out += table(["Length", "Round", "direct_acc", "cut1_acc", "confirmed", "agreed", "agreed acc", "novelty_arm",
                      "novelty_agreed", "arm4_verdict", "sums drawn", "kept", "kept_acc", "kept carry",
                      "acc_total after", "acc_carry", "acc_no_carry"], rows) + [""]
    return out


def round_seconds(ev: list[dict]) -> dict[int, dict]:
    """Per round, the seconds of every round_done of it summed (a continuation's redo is added to
    the original attempt's round)."""
    out: dict[int, dict] = {}
    for e in of(ev, "round_done"):
        o = out.setdefault(e["round"], {k: 0.0 for k in (*PARTS, "sec_total")})
        for k in o:
            o[k] += e[k]
    return out


def arm4_section(seed: int) -> list[str]:
    ev4, ev3 = events(f"ladder_s{seed}_arm4"), events(f"ladder_s{seed}_arm3")
    out = ["## Arm 4: what it built and skipped", ""]
    if not ev4:
        return out + ["No log fetched.", ""]
    done3 = round_seconds(ev3)
    rows = []
    for e in of(ev4, "round_done"):
        r = e["round"]
        skipped = [d for d in round_lengths(r) if d not in e["built"]]
        o = done3.get(r, {})
        rows.append([r, ", ".join(map(str, e["built"])) or "-", ", ".join(map(str, skipped)) or "-",
                     "no" if e["round_skipped"] else "yes", f"{e['n_raw_examples']:,}", f"{e['n_effective_examples']:,}",
                     f"{e['num_updates']:,}",
                     " / ".join(f"{e[k]:.0f}" for k in ("sec_check", "sec_build", "sec_train", "sec_test", "sec_total")),
                     " / ".join(f"{o[k]:.0f}" for k in ("sec_check", "sec_build", "sec_train", "sec_test", "sec_total"))
                     if o else "-"])
    out += table(["Round", "Built", "Skipped", "Trained", "n_raw_examples", "n_effective_examples", "num_updates",
                  "arm 4 seconds: check / build / train / test / total",
                  "arm 3 seconds: check / build / train / test / total"], rows) + [""]
    tot = lambda ev: sum(e["sec_total"] for e in of(ev, "round_done"))  # noqa: E731
    out += [f"Round seconds summed: arm 4 {tot(ev4):,.0f}, arm 3 {tot(ev3):,.0f}.", ""]
    skips = of(ev4, "skip_length")
    tests = counting_tests(ev4)
    if skips:
        out += ["`acc_total` (%) of every length arm 4 skipped, after the round that skipped it and every later round:", ""]
        later = sorted(r for r in tests if r >= min(e["round"] for e in skips))
        rows = [[e["digits"], e["round"], e["pool"]] + [pct(tests[r].get(str(e["digits"]), {}).get("acc_total"))
                                                         if r >= e["round"] else "" for r in later] for e in skips]
        out += table(["Length", "Skipped in round", "Pool (pair-agreed answers)", *(f"r{r}" for r in later)], rows)
        out += [""]
    else:
        out += ["Arm 4 skipped no length.", ""]
    return out


def smoke_section() -> list[str]:
    ev = events("ladder_smoke")
    path = RUNS / "ladder_smoke" / "smoke.json"
    if not ev:
        return []
    s = json.loads(path.read_text()) if path.exists() else {}
    out = ["## GPU smoke test (`ladder_smoke`, not counted in the cap)", "",
           "Seed model `start-seed2-authors` (step 10,000), loaded with max_len 136; round 1 of arm 3 with 200 kept "
           "answers per length and 100 updates; nothing from it is used by any ladder run.", ""]
    out += table(["Length", "agreed_share", "confirmed_share", "agreed acc", "cut1_acc", "direct_acc"],
                 [[c["digits"], pct(c["agreed_share"], 2), pct(c["confirmed_share"], 2), pct(c["agreed_acc"]),
                   pct(c["cut1_acc"]), pct(c["direct_acc"])] for c in of(ev, "check")])
    d = of(ev, "round_done")
    if d:
        d = d[0]
        out += ["", "Seconds: " + ", ".join(f"{k} {d[k]:.1f}" for k in
                                             ("sec_check", "sec_build", "sec_train", "sec_test", "sec_total")) + "."]
    if s:
        out += [f"At 39 digits: 500 direct answers {s['direct_seconds_500']:.2f} s, 500 composed answers (three cuts) "
                f"{s['composed_seconds_500']:.2f} s, peak GPU memory {s['peak_memory_gib']:.2f} GiB of "
                f"{s['gpu_memory_gib']:.0f}; no out-of-memory error."]
    return out + [""]


# ---------------------------------------------------------------- the summary over seeds

def arm_summary(ev: list[dict]) -> dict:
    tests, stages = counting_tests(ev), stages_of(ev)
    lows = {}
    for s in sorted(stages):
        acc = tests.get(4 * s, {})
        lows[s] = min((acc.get(str(d), {}).get("acc_total", 0.0), d) for d in STAGES[s])
    done = of(ev, "round_done")
    last = max((e["round"] for e in done), default=0)
    return {"calls": [e["reason"] for e in of(ev, "call_done")], "learned": max((s for s, v in stages.items() if v),
                                                                                default=None),
            "lows": lows, "diverged": len(of(ev, "diverged")), "redos": len([e for e in of(ev, "test") if e.get("redo")]),
            "last_round": last, "last_acc": tests.get(last, {})}


def summary_section() -> list[str]:
    seeds = seeds_run()
    out = [f"## Summary over seeds {', '.join(map(str, seeds))}", "",
           "Tests that count: a redone round's redo (amendment 30). Rounds 4, 8, 12 and 16 end stages 1-4.", ""]
    # 1. how each arm ended
    rows = []
    for seed in seeds:
        for arm in (1, 2, 3, 4):
            run = f"ladder_s{seed}_arm{arm}"
            ev = events(run)
            if not ev:
                rows.append([seed, arm, f"`{run}`", "not run or not fetched", "-", "-", "-", "-", "-", "-", "-"])
                continue
            parts = [("original call", original_part(ev)), ("with the continuation", ev)] \
                if len(of(ev, "call_done")) > 1 else [("", ev)]
            for label, part in parts:
                s = arm_summary(part)
                rows.append([seed, arm, f"`{run}`" + (f" ({label})" if label else ""),
                             ", ".join(s["calls"]) or "not ended", s["learned"] or "none"]
                            + [f"{pct(s['lows'][k][0])} ({s['lows'][k][1]})" if k in s["lows"] else "-"
                               for k in (1, 2, 3, 4)] + [s["diverged"], s["redos"]])
    out += ["How each arm ended. Lowest `acc_total` (%) at the stage's lengths after the stage's last round (at "
            "length), for every stage whose last round was finished.", ""]
    out += table(["Seed", "Arm", "Run", "How the calls ended", "Last stage learned", "Stage 1", "Stage 2", "Stage 3",
                  "Stage 4", "`diverged` events", "Redos"], rows) + [""]
    # 2. arm 3 against arm 4 after the last round each finished
    header, cols = ["Length"], []
    for seed in seeds:
        s3, s4 = arm_summary(events(f"ladder_s{seed}_arm3")), arm_summary(events(f"ladder_s{seed}_arm4"))
        header += [f"s{seed} arm 3 (r{s3['last_round']})", f"s{seed} arm 4 (r{s4['last_round']})",
                   f"s{seed} arm 4 - arm 3"]
        cols.append((s3["last_acc"], s4["last_acc"]))
    rows = []
    for d in LENGTHS:
        row = [d]
        for a3, a4 in cols:
            v3, v4 = a3.get(str(d), {}).get("acc_total"), a4.get(str(d), {}).get("acc_total")
            row += [pct(v3), pct(v4), "-" if v3 is None or v4 is None else f"{100 * (v4 - v3):+.1f}"]
        rows.append(row)
    out += ["`acc_total` (%) at 8-39 digits after the last round each arm finished (round in brackets); "
            "difference in points.", ""]
    out += table(header, rows) + [""]
    # 3. what arm 4 built, skipped and spent, next to arm 3
    rows = []
    for seed in seeds:
        for arm in (3, 4):
            ev = events(f"ladder_s{seed}_arm{arm}")
            if not ev:
                continue
            built = sorted({e["digits"] for e in of(ev, "build")})
            skipped = sorted({e["digits"] for e in of(ev, "skip_length")})
            done = of(ev, "round_done")
            trained = sorted({e["round"] for e in done if not e["round_skipped"]})
            updates = sum(e["num_updates"] * (2 if e.get("redone") and not e.get("continuation") else 1) for e in done)
            secs = {k: sum(e[k] for e in done) for k in PARTS}
            rows.append([seed, arm, _ranges(built), _ranges(skipped) or "-", f"{len(trained)} ({_ranges(trained)})",
                         f"{updates:,}"] + [f"{secs[k]:,.0f}" for k in PARTS] + [f"{sum(secs.values()):,.0f}"])
    out += ["Arm 4 against arm 3: lengths built and skipped, rounds trained, updates (a redo's updates included) and "
            "GPU seconds by part (summed over the rounds' `round_done` events; the round-0 test and the "
            "continuation's start check are not included).", ""]
    out += table(["Seed", "Arm", "Lengths built", "Lengths skipped", "Rounds trained", "Updates", "check s", "build s",
                  "train s", "test s", "sum s"], rows) + [""]
    # 4. every length arm 4 skipped: its lowest acc_total afterwards
    rows = []
    for seed in seeds:
        ev = events(f"ladder_s{seed}_arm4")
        tests = counting_tests(ev)
        for e in of(ev, "skip_length"):
            vals = [(tests[r][str(e["digits"])]["acc_total"], r) for r in sorted(tests)
                    if r >= e["round"] and str(e["digits"]) in tests[r]]
            low = min(vals) if vals else None
            rows.append([seed, e["digits"], e["round"], e["pool"], "-" if low is None else f"{pct(low[0])} (r{low[1]})",
                         len(vals)])
    out += ["Every length arm 4 skipped: its lowest `acc_total` (%) over the test of the round that skipped it and "
            "every later round (round in brackets).", ""]
    out += (table(["Seed", "Length", "Skipped in round", "Pool", "Lowest acc_total", "Tests"], rows) if rows
            else ["Arm 4 skipped no length."]) + [""]
    # 5. arm 3 where the check said there was nothing new
    rows = []
    for seed in seeds:
        ev = events(f"ladder_s{seed}_arm3")
        tests = counting_tests(ev)
        for c in of(ev, "check"):
            if c["arm4_verdict"] == "skip_length":
                after = tests.get(c["round"], {}).get(str(c["digits"]), {})
                rows.append([seed, c["digits"], c["round"], pct(c["direct_acc"]), pct(after.get("acc_total"))])
    out += ["Arm 3, at every length where its check gave `arm4_verdict` = `skip_length` (D16): `direct_acc` in the "
            "check and `acc_total` after the round (%).", ""]
    out += (table(["Seed", "Length", "Round", "direct_acc", "acc_total after"], rows) if rows
            else ["No such length."]) + [""]
    # 6. dollars
    rows = [[seed, f"{spent(seed):.2f}", f"{CAP_DOLLARS:.2f}"] for seed in seeds]
    rows.append(["total", f"{sum(spent(s) for s in seeds):.2f}", f"{CAP_DOLLARS * len(seeds):.2f}"])
    out += ["Dollars per seed against its cap (seed model and arm calls; smoke test not counted).", ""]
    out += table(["Seed", "Dollars", "Cap"], rows) + [""]
    return out


def _ranges(xs: list[int]) -> str:
    """[8, 9, 10, 12] -> "8-10, 12"."""
    out, start = [], None
    for i, x in enumerate(xs):
        if start is None:
            start = x
        if i + 1 == len(xs) or xs[i + 1] != x + 1:
            out.append(str(start) if start == x else f"{start}-{x}")
            start = None
    return ", ".join(out)


NOTE_D14 = ("**Note (decision D14).** At lengths 8 to 15 the ladder's carry groups are defined by a carry across "
            "the ladder's own Cut 1 (left piece h - 1). They differ from the carry groups of the earlier study, whose "
            "cut 1 differs at 8, 10 and 12-15, so the ladder's `acc_carry` and `acc_no_carry` at those lengths are "
            "not comparable with the study's.")


def write_report(path: Path = LADDER_RESULTS / "report.md") -> str:
    seeds = seeds_run()
    lines = [f"# The ladder: random seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))}", "",
             "Generated by `uv run python -m gainpred.ladder_report` from the fetched logs; do not edit by hand. "
             "Rules: protocol amendments 29 and 30 and `docs/ladder_spec.md`. Exploratory: no outcome was predicted.",
             "", NOTE_D14, ""]
    lines += calls_section()
    for seed in seeds:
        body = seed_section(seed) + arm_sections(seed) + arm4_section(seed)
        lines += [f"# Random seed {seed}", ""] + body
    lines += smoke_section() + summary_section()
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text)
    return text


if __name__ == "__main__":
    write_report()
    print(f"wrote {LADDER_RESULTS / 'report.md'}")
