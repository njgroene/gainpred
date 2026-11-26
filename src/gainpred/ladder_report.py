"""The ladder's report (``results/ladder/report.md``), generated from the fetched logs only.

Usage: ``uv run python -m gainpred.ladder_report`` (after ``python -m gainpred.ladder fetch``).
Every number comes from ``results/ladder/runs/*`` and ``results/ladder/launches.jsonl``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gainpred.ladder import (ARM_NAMES, CAP_DOLLARS, DOLLARS_PER_HOUR, LADDER_RESULTS, STAGES, call_seconds,
                             records, round_lengths)

RUNS = LADDER_RESULTS / "runs"
SEED = 5


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


def calls_section() -> tuple[list[str], float]:
    rows, total = [], 0.0
    for r in records():
        run, seg = r["run"], r["key"].split(":", 1)[1] or "main"
        path = RUNS / run / "calls" / f"{seg}.json"
        c = json.loads(path.read_text()) if path.exists() else {}
        secs = c.get("wall_seconds")
        dollars = None if secs is None else secs / 3600 * DOLLARS_PER_HOUR
        if run.startswith(f"ladder_s{SEED}_") and dollars is not None:
            total += dollars
        rows.append([f"`{run}`", f"`{r['key']}`", when(c.get("started")), when(c.get("ended")),
                     "-" if secs is None else f"{secs:,.0f}", "-" if dollars is None else f"{dollars:.2f}",
                     c.get("reason", "not ended")])
    out = ["## Calls", "", "Every call launched for the ladder (`results/ladder/launches.jsonl`), crashed and "
           "relaunched calls included. Dollars = wall seconds of the call x $1.10 / 3,600 (decision D11).", ""]
    out += table(["Run", "Key", "Start", "End", "Wall seconds", "Dollars", "How it ended"], rows)
    out += ["", f"**Random seed {SEED}: ${total:.2f} of the ${CAP_DOLLARS:.2f} cap** (seed model and arm calls; "
            "the smoke test `ladder_smoke` is reported above but not counted, decision D11).", ""]
    return out, total


def seed_section() -> list[str]:
    log = RUNS / f"ladder_s{SEED}_seed" / "log.jsonl"
    if not log.exists():
        return ["## Seed model", "", "Not fetched.", ""]
    ev = [json.loads(line) for line in log.read_text().splitlines()]
    easy = [e for e in ev if e.get("event") == "eval" and e.get("set") == "easy"][-1]
    start = [e for e in ev if e.get("event") == "eval" and e.get("set") == "start"][-1]
    acc = {d: easy["accuracy"][d] for d in "34567"}
    out = ["## Seed model", "", f"`ladder_s{SEED}_seed`, step {easy['step']:,}: accuracy on the existing easy test set "
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


def arm_sections() -> list[str]:
    out = ["## Arms", ""]
    for arm in (1, 2, 3, 4):
        run = f"ladder_s{SEED}_arm{arm}"
        ev = events(run)
        out += [f"### Arm {arm}: {ARM_NAMES[arm]} (`{run}`)", ""]
        if not ev:
            out += ["No log fetched.", ""]
            continue
        tests = {e["round"]: e["acc"] for e in of(ev, "test")}
        stages = {e["stage"]: e["learned"] for e in of(ev, "stage")}
        learned = [s for s in sorted(stages) if stages[s]]
        out += [f"Ended: **{ending(ev)}** (all calls: " + ", ".join(e["reason"] for e in of(ev, "call_done")) + "). "
                f"Last round finished: {max((e['round'] for e in of(ev, 'round_done')), default=0)}. "
                f"Stages learned: {', '.join(map(str, learned)) or 'none'}"
                + (f"; not learned: {', '.join(str(s) for s in sorted(stages) if not stages[s])}" if
                   any(not v for v in stages.values()) else "") + ".", ""]
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


def arm4_section() -> list[str]:
    ev4, ev3 = events(f"ladder_s{SEED}_arm4"), events(f"ladder_s{SEED}_arm3")
    out = ["## Arm 4: what it built and skipped", ""]
    if not ev4:
        return out + ["No log fetched.", ""]
    done3 = {e["round"]: e for e in of(ev3, "round_done")}
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
    tests = {e["round"]: e["acc"] for e in of(ev4, "test")}
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


NOTE_D14 = ("**Note (decision D14).** At lengths 8 to 15 the ladder's carry groups are defined by a carry across "
            "the ladder's own Cut 1 (left piece h - 1). They differ from the carry groups of the earlier study, whose "
            "cut 1 differs at 8, 10 and 12-15, so the ladder's `acc_carry` and `acc_no_carry` at those lengths are "
            "not comparable with the study's.")


def write_report(path: Path = LADDER_RESULTS / "report.md") -> str:
    calls, _ = calls_section()
    lines = ["# The ladder: random seed 5", "",
             "Generated by `uv run python -m gainpred.ladder_report` from the fetched logs; do not edit by hand. "
             "Rules: protocol amendment 29 and `docs/ladder_spec.md`. Exploratory: no outcome was predicted.", "",
             NOTE_D14, ""]
    lines += calls + seed_section() + arm_sections() + arm4_section() + smoke_section()
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text)
    return text


if __name__ == "__main__":
    write_report()
    print(f"wrote {LADDER_RESULTS / 'report.md'}")
