"""Pilot report: reads ``results/pilot/runs/*/rounds.jsonl`` and applies the spec's fixed definitions.

Definitions (spec, "Definitions fixed now"; owner's answers 1-3, 6, 12)
=======================================================================
* Accuracy at a length: mean of the carry and no-carry groups of the test set (8-15: the study's
  set, 13-15 from the five-cut set; 16-39: the pilot's set); easy lengths: per length.
* **Learned**: after the floor's last round, at least 90% at every length of the floor, in at
  least two of three repeats (the answer key has two repeats: both).
* **Failed**: below 70% at one or more lengths of the floor, in at least two of three repeats.
* **Partial**: neither.
* **Eroded** (floor 1): a floor-1 length more than 3 points below that repeat's accuracy just
  before the floor's first round, in at least two of three repeats. Test 1 also reports the
  largest drop at any length from 3 up to the top of the floor below.
* **Seen coming**: the check's value in every repeat of a failing condition lies beyond its value
  in every repeat of the solid condition (lower confirmed share; higher novelty at some floor-1
  length). The looser comparison (mean lower or higher) is reported beside it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from gainpred.floors import FLOORS, PILOT_RESULTS, round_lengths, seconds_of
from gainpred.floors_cost import dollars

RUNS = PILOT_RESULTS / "runs"


def events(run: str) -> list[dict]:
    path = RUNS / run / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def accuracy(evs: list[dict], floor: int | None, round_: int) -> dict[int, float]:
    """Percent per length after ``round_`` of ``floor`` (None: an under-trained floor-1 run)."""
    out: dict[int, float] = {}
    sets = {e["set"]: e["accuracy"] for e in evs if e["event"] == "eval" and e["round"] == round_
            and e.get("floor") == floor}
    for name in ("easy", "study", "floors", "study5"):  # study5 last: it replaces 13-15
        groups = defaultdict(list)
        for k, v in sets.get(name, {}).items():
            if k != "all":
                groups[int(k.split("/")[0])].append(v)
        out |= {d: 100 * sum(v) / len(v) for d, v in groups.items()}
    return dict(sorted(out.items()))


def finished_rounds(evs: list[dict], floor: int) -> list[int]:
    return sorted(e["round"] for e in evs if e["event"] == "round_done" and e.get("floor") == floor)


def verdict(runs: list[str], floor: int) -> dict:
    """Learned / failed / partial over the repeats that finished the floor's four rounds."""
    finals, eroded, worst_drop, learned_each = {}, {}, {}, {}
    for run in runs:
        evs = events(run)
        if 4 not in finished_rounds(evs, floor):
            continue
        after, before = accuracy(evs, floor, 4), accuracy(evs, floor, 0)
        finals[run] = after
        learned_each[run] = all(after.get(d, 0) >= 90 for d in FLOORS[floor])
        eroded[run] = any(before[d] - after.get(d, 0) > 3 for d in FLOORS[1] if d in before)
        drops = [(before[d] - after.get(d, 0), d) for d in before if d < FLOORS[floor].start]
        worst_drop[run] = max(drops) if drops else None
    n = len(finals)
    need = 2 if len(runs) >= 3 else len(runs)
    learned = sum(learned_each.values()) >= need
    failed = sum(any(a.get(d, 0) < 70 for d in FLOORS[floor]) for a in finals.values()) >= need
    return {"repeats": n, "learned": learned, "failed": failed and not learned,
            "verdict": "learned" if learned else "failed" if failed else "partial" if n else "not run",
            "learned_each": learned_each, "floor1_eroded": sum(eroded.values()) >= need, "eroded_each": eroded,
            "worst_drop": worst_drop, "finals": finals}


def seen_coming(runs: list[str]) -> list[dict]:
    """The two numbers, before floor-2 training, per repeat (from the run that computed them)."""
    out = []
    for run in runs:
        e = next((e for e in events(run) if e["event"] == "seen_coming"), None)
        if e:
            out.append({"run": run, "confirmed": {d: v["confirmed_share"] for d, v in e["confirmed_share"].items()},
                        "novelty": {d: v["novelty"] for d, v in e["floor1_novelty"].items()}})
    return out


def compare_seen(cond: list[dict], solid: list[dict]) -> dict:
    """Strict ("beyond every solid repeat") and loose (mean) comparisons."""
    if not cond or not solid:
        return {}
    res = {}
    for d in ("16", "17"):
        c, s = [x["confirmed"][d] for x in cond], [x["confirmed"][d] for x in solid]
        res[f"confirmed {d}"] = {"strict": max(c) < min(s), "loose": sum(c) / len(c) < sum(s) / len(s)}
    strict = loose = False
    for d in map(str, FLOORS[1]):
        c, s = [x["novelty"][d] or 0 for x in cond], [x["novelty"][d] or 0 for x in solid]
        strict |= min(c) > max(s)
        loose |= sum(c) / len(c) > sum(s) / len(s)
    res["floor-1 novelty"] = {"strict": strict, "loose": loose}
    return res


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def acc_table(runs: list[str], floor: int | None, round_: int, lengths) -> list[str]:
    head = "| Run | " + " | ".join(map(str, lengths)) + " |"
    lines = [head, "|" + " --- |" * (len(lengths) + 1)]
    for run in runs:
        a = accuracy(events(run), floor, round_)
        if a:
            lines.append(f"| `{run}` | " + " | ".join(fmt(a.get(d)) for d in lengths) + " |")
    return lines


def rounds_table(runs: list[str]) -> list[str]:
    lines = ["| Run | Floor | Round | Examples | Updates | Kept per new length | Kept right (%) | Minutes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        evs = events(run)
        for e in evs:
            if e["event"] != "round_done":
                continue
            builds = [b for b in evs if b["event"] == "build" and b["floor"] == e["floor"] and b["round"] == e["round"]]
            kept = ", ".join(f"{b['digits']}: {b['kept']:,} of {b['drawn']:,}" for b in builds)
            right = ", ".join(fmt(100 * b["kept_right"], 2) if b["kept_right"] is not None else "–" for b in builds)
            lines.append(f"| `{run}` | {e['floor']} | {e['round']} | {e['examples']:,} | {e['updates']:,} | {kept} | "
                         f"{right} | {sum(e['seconds'].values()) / 60:.1f} |")
    return lines


def stage_dollars(stages: dict[str, list[str]]) -> list[str]:
    lines = ["| Stage | Runs | Logged GPU-hours | Dollars |", "| --- | --- | --- | --- |"]
    total = 0.0
    for stage, runs in stages.items():
        s = sum(seconds_of(r) for r in runs)
        total += s
        lines.append(f"| {stage} | {len(runs)} | {s / 3600:.2f} | {dollars(s):.2f} |")
    lines.append(f"| **Total** | | {total / 3600:.2f} | **{dollars(total):.2f}** |")
    return lines


# ---------------------------------------------------------------------------------------------
# The report (results/pilot/report.md)
# ---------------------------------------------------------------------------------------------
# Run names used by the pilot. A condition is (version of floor 1, teacher); "r<k>" is the repeat.
REPEATS = (0, 1, 2)
SOLID_AGR = [f"solid-agr-r{r}" for r in REPEATS]
# Repeat 1's floor-4 call was restarted by Modal after its round 2 and refused to overwrite its
# round 1; it was relaunched once as solid-agr-r1-again (same repeat, same streams, same start).
FLOOR4_AGR = ["solid-agr-r0", "solid-agr-r1-again", "solid-agr-r2"]
SOLID_KEY = ["solid-key-r0", "solid-key-r1"]  # owner's change A: the answer key has two repeats
CONDITIONS = {  # condition -> runs (floor 2)
    "solid / agreement": SOLID_AGR,
    "solid / answer key": SOLID_KEY,
    "solid / no-carry": [f"solid-nc-r{r}" for r in REPEATS],
    "corrupted / agreement": [f"corrupt-agr-r{r}" for r in REPEATS],
    "corrupted / no-carry": [f"corrupt-nc-r{r}" for r in REPEATS],
    "under-trained / agreement": [f"under-agr-r{r}" for r in REPEATS],
    "under-trained / no-carry": [f"under-nc-r{r}" for r in REPEATS],
    "under-trained / answer key": [f"under-key-r{r}" for r in REPEATS],
}
UNDERTRAINED = {500: "under-500", 1_000: "under-1000", 1_500: "under-1500"}
KEY_CONTROL = {3: [f"solid-key3-r{r}" for r in REPEATS], 4: [f"solid-key4-r{r}" for r in REPEATS]}
STAGES = {  # stage -> runs whose logged seconds count for it (a run spanning stages is split below)
    "Timing and rehearsal set": ["timing", "rehearsal-build-seed2"],
}


def present(runs: list[str]) -> list[str]:
    return [r for r in runs if events(r) or (RUNS / r / "call.json").exists()]


def floor_mean(acc: dict[int, float], lengths) -> float | None:
    vals = [acc[d] for d in lengths if d in acc]
    return sum(vals) / len(vals) if len(vals) == len(list(lengths)) else None


def block_tables(runs: list[str], floor: int, round_: int) -> list[str]:
    """Accuracy per length and repeat after ``round_`` of ``floor``: one table per floor of
    lengths (the easy lengths 3-7 in the first), with a 'before' row (round 0) for lower floors."""
    out = []
    blocks = [("3-7 and floor 1 (8-15)", list(range(3, 16)))] + \
             [(f"floor {f} ({FLOORS[f].start}-{FLOORS[f].stop - 1})", list(FLOORS[f])) for f in range(2, floor + 1)]
    for name, lengths in blocks:
        out += [f"*{name}*", "", "| Run | When | " + " | ".join(map(str, lengths)) + " |",
                "|" + " --- |" * (len(lengths) + 2)]
        for run in runs:
            evs = events(run)
            rows = [("before", accuracy(evs, floor, 0))] if lengths[-1] < FLOORS[floor].start else []
            rows.append((f"after round {round_}", accuracy(evs, floor, round_)))
            for when, a in rows:
                if a:
                    out.append(f"| `{run}` | {when} | " + " | ".join(fmt(a.get(d)) for d in lengths) + " |")
        out.append("")
    return out


def verdict_lines(runs: list[str], floor: int) -> list[str]:
    v = verdict(runs, floor)
    if not v["repeats"]:
        return ["Not run (or no repeat finished round 4)."]
    lines = [f"* **Verdict: {v['verdict']}** ({v['repeats']} repeats finished round 4; "
             f"learned needs {'2 of 3' if len(runs) >= 3 else 'all'} repeats at >= 90% at every length {FLOORS[floor].start}-"
             f"{FLOORS[floor].stop - 1}).",
             f"* Floor 1 eroded (> 3 points below the repeat's own level before the floor's first round, "
             f"in {'2 of 3' if len(runs) >= 3 else 'all'} repeats): **{'yes' if v['floor1_eroded'] else 'no'}**."]
    for run in v["finals"]:
        drop, at = v["worst_drop"][run] or (None, None)
        lo = min(v["finals"][run].get(d, 0) for d in FLOORS[floor])
        lines.append(f"  * `{run}`: learned {'yes' if v['learned_each'][run] else 'no'} (lowest on the floor "
                     f"{lo:.1f}%); floor 1 eroded {'yes' if v['eroded_each'][run] else 'no'}; largest drop at any "
                     f"length 3-{FLOORS[floor].start - 1}: {fmt(drop)} points" + (f" at {at} digits" if at else "")
                     + ".")
    return lines


def checks_table(runs: list[str], floor: int) -> list[str]:
    """The round's checks at the new lengths, with the condition's own teacher (and corruption)."""
    lines = ["| Run | Round | Digits | Model's direct answer right | Kept share | Novelty | Confirmed share | Kept right (audit) | Self-taught pieces replaced |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        for e in events(run):
            if e["event"] == "checks" and e.get("floor") == floor:
                c = e["confirmed_share"]
                lines.append(f"| `{run}` | {e['round']} | {e['digits']} | {fmt(100 * e['audit_direct_right'])} | "
                             f"{fmt(100 * e['kept_share'])} | "
                             f"{fmt(None if e['novelty'] is None else 100 * e['novelty'])} | "
                             f"{c if isinstance(c, str) else fmt(None if c is None else 100 * c)} | "
                             f"{fmt(None if e['audit_kept_right'] is None else 100 * e['audit_kept_right'], 2)} | "
                             f"{fmt(100 * e.get('replaced_share', 0), 2)} |")
    return lines


def floor1_novelty_table(runs: list[str], floor: int) -> list[str]:
    lines = ["| Run | Round | " + " | ".join(map(str, FLOORS[1])) + " |", "|" + " --- |" * 10]
    for run in runs:
        for e in events(run):
            if e["event"] == "floor1_checks" and e.get("floor") == floor:
                n = e["novelty"]
                lines.append(f"| `{run}` | {e['round']} | " + " | ".join(
                    fmt(None if n[str(d)]["novelty"] is None else 100 * n[str(d)]["novelty"], 2) for d in FLOORS[1]) + " |")
    return lines


def floor_below_table(runs: list[str], floor: int) -> list[str]:
    lines = ["| Run | " + " | ".join(f"{d}: novelty / confirmed" for d in FLOORS[floor - 1]) + " |",
             "|" + " --- |" * 9]
    for run in runs:
        e = next((e for e in events(run) if e["event"] == "floor_below_checks" and e.get("floor") == floor), None)
        if e:
            c = e["checks"]
            lines.append(f"| `{run}` | " + " | ".join(
                f"{fmt(None if c[str(d)]['novelty'] is None else 100 * c[str(d)]['novelty'])} / "
                f"{fmt(None if c[str(d)]['confirmed_share'] is None else 100 * c[str(d)]['confirmed_share'])}"
                for d in FLOORS[floor - 1]) + " |")
    return lines


def seen_section() -> list[str]:
    out = ["The two numbers, computed with the no-carry rule on the floor-1 model before any floor-2 training "
           "(2,000 fresh problems per length; with corrupted pieces the corruption is applied to the confirmed "
           "share; floor-1 novelty uses pieces of 3-7 digits, which are never corrupted, and covers only problems "
           "with no carry at cut 1). Percent.", "",
           "| Version | Run | Confirmed 16 | Confirmed 17 | " + " | ".join(f"Novelty {d}" for d in FLOORS[1]) + " |",
           "|" + " --- |" * 12]
    groups = {"solid": SOLID_AGR, "corrupted": CONDITIONS["corrupted / agreement"],
              "under-trained": CONDITIONS["under-trained / agreement"]}
    vals = {}
    for version, runs in groups.items():
        vals[version] = seen_coming(runs)
        for x in vals[version]:
            out.append(f"| {version} | `{x['run']}` | {fmt(100 * x['confirmed']['16'])} | {fmt(100 * x['confirmed']['17'])} | "
                       + " | ".join(fmt(None if x["novelty"][str(d)] is None else 100 * x["novelty"][str(d)], 2)
                                    for d in FLOORS[1]) + " |")
    out.append("")
    for version in ("corrupted", "under-trained"):
        res = compare_seen(vals[version], vals["solid"])
        if not res:
            continue
        floor2_fails = {}
        for teacher in ("agreement", "no-carry"):
            runs = CONDITIONS[f"{version} / {teacher}"]
            if present(runs):
                floor2_fails[teacher] = verdict(runs, 2)["verdict"]
        out.append(f"* **{version} vs solid** (floor 2 with that version: "
                   + ", ".join(f"{t} {v}" for t, v in floor2_fails.items()) + "): "
                   + "; ".join(f"{k}: strict {'yes' if v['strict'] else 'no'}, loose {'yes' if v['loose'] else 'no'}"
                               for k, v in res.items()) + ".")
    out += ["", "*Strict* (the spec's rule): every repeat of the condition lies beyond every solid repeat (lower "
            "confirmed share; for novelty, higher at one or more floor-1 lengths). *Loose*: the condition's mean "
            "lies beyond the solid mean. \"Seen coming\" applies only to a condition where floor 2 fails.", ""]
    return out


def undertrained_section() -> list[str]:
    runs = {s: r for s, r in UNDERTRAINED.items() if present([r])}
    if not runs:
        return ["Not run."]
    out = ["Accuracy after round 4 (percent; 13-15 from the five-cut set).", "",
           "| Updates per round | Run | " + " | ".join(map(str, FLOORS[1])) + " | Mean 8-15 |", "|" + " --- |" * 11]
    means = {}
    for steps, run in runs.items():
        a = accuracy(events(run), None, 4)
        means[steps] = floor_mean(a, FLOORS[1])
        out.append(f"| {steps:,} | `{run}` | " + " | ".join(fmt(a.get(d)) for d in FLOORS[1]) + f" | {fmt(means[steps])} |")
    out += ["", "Mean over 8-15 after each round (the study's run with 3,000 updates per round is the solid floor 1):",
            "", "| Updates per round | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |", "| --- | --- | --- | --- | --- | --- |"]
    for steps, run in runs.items():
        evs = events(run)
        out.append(f"| {steps:,} | " + " | ".join(fmt(floor_mean(accuracy(evs, None, k), FLOORS[1])) for k in range(5)) + " |")
    choice = undertrained_choice()
    out += ["", f"* Choice (mean closest to 85%): **{choice['choice'] or 'none'}**. "
            f"Every length below 20% or above 98% in all three runs: **{'yes' if choice['all_extreme'] else 'no'}**"
            + (" -> version B not run (a finding)." if choice["all_extreme"] else "."), ""]
    return out


def undertrained_choice() -> dict:
    """The spec's rule: the run whose mean over 8-15 after round 4 is closest to 85%, unless in all
    three runs every length is below 20% or above 98% (then version B is not run)."""
    means, extreme = {}, {}
    for steps, run in UNDERTRAINED.items():
        a = accuracy(events(run), None, 4)
        if len([d for d in FLOORS[1] if d in a]) == 8:
            means[steps] = floor_mean(a, FLOORS[1])
            extreme[steps] = all(a[d] < 20 or a[d] > 98 for d in FLOORS[1])
    all_extreme = len(extreme) == 3 and all(extreme.values())
    choice = None if all_extreme or not means else UNDERTRAINED[min(means, key=lambda s: (abs(means[s] - 85), s))]
    return {"means": means, "extreme": extreme, "all_extreme": all_extreme, "choice": choice}


def rehearsal_section() -> list[str]:
    meta = json.loads((RUNS / "rehearsal-floor1-seed2" / "meta.json").read_text())
    audit = meta["audit"]
    wrong = sum(v["wrong"] for v in audit.values())
    return [f"Built once from `{meta['checkpoint']}` with the no-carry rule (pieces of 3-7 digits); "
            f"{meta['rows']:,} rows, sha256 `{meta['sha256'][:16]}...`. Not filtered: **{wrong} of {meta['rows']:,} "
            "answers are wrong**.", "",
            "| Digits | " + " | ".join(audit) + " |", "|" + " --- |" * (len(audit) + 1),
            "| Kept | " + " | ".join(f"{v['kept']:,}" for v in audit.values()) + " |",
            "| Drawn | " + " | ".join(f"{v['drawn']:,}" for v in audit.values()) + " |",
            "| Wrong | " + " | ".join(str(v["wrong"]) for v in audit.values()) + " |", ""]


def segment_seconds(run: str) -> dict[str, float]:
    """Logged seconds of a run, per call: "floor <f> rounds <a>-<b>" (each pilot_floor call logs
    call_done with its rounds); an under-trained or other single call: "floor 1"."""
    out: dict[str, float] = defaultdict(float)
    if (RUNS / run / "call.json").exists():
        return {"floor 1": seconds_of(run)}
    pending: dict[int, float] = defaultdict(float)
    for e in events(run):
        if e["event"] in ("round_done", "before_floor"):
            pending[e["floor"]] += sum(e["seconds"].values())
        if e["event"] == "call_done":
            a, b = e["rounds"]
            out[f"floor {e['floor']} rounds {a}-{b}"] += e["wall_seconds"]
            pending.pop(e["floor"], None)
    for f, s in pending.items():
        out[f"floor {f} unfinished"] += s
    return dict(out)


def dollars_section(stages: dict[str, list[tuple[str, str | None]]]) -> list[str]:
    """``stages``: stage -> (run, floor label or None for the whole run)."""
    lines = ["| Stage | Runs | Logged GPU-hours | Dollars |", "| --- | --- | --- | --- |"]
    total = 0.0
    for stage, parts in stages.items():
        s = 0.0
        for run, part in parts:
            seg = segment_seconds(run) if (RUNS / run).exists() else {}
            s += sum(v for k, v in seg.items() if part is None or k.startswith(part))
        if parts:
            total += s
            lines.append(f"| {stage} | {len(parts)} | {s / 3600:.2f} | {dollars(s):.2f} |")
    lines.append(f"| **Pilot total** | | {total / 3600:.2f} | **{dollars(total):.2f}** |")
    return lines


CONTROL_RUNS = [f"direct-r{r}" for r in REPEATS]


def control_section(notes: Path = PILOT_RESULTS / "notes_control.md") -> list[str]:
    """Protocol amendment 27: the direct-answers control (prediction C1: floor 2 is NOT learned)."""
    out = ["## Control: direct answers (amendment 27)", ""]
    runs = present(CONTROL_RUNS)
    if not runs:
        return out + ["Not run.", ""]
    v2 = verdict(CONTROL_RUNS, 2)
    if v2["repeats"]:
        held = not v2["learned"]
        out += [f"**Prediction C1 (floor 2 is not learned with direct answers): {'held' if held else 'failed'}** "
                f"(floor 2 {v2['verdict']}; learned in {sum(v2['learned_each'].values())} of 3 repeats).", ""]
    floors = [f for f in (2, 3, 4) if any(e["event"] == "build" and e.get("floor") == f for r in runs for e in events(r))]
    for f in floors:
        out += [f"### Floor {f}", ""] + verdict_lines(CONTROL_RUNS, f) + ["", "Accuracy per length and repeat after "
                "each round (percent; the floor's lengths, and floor 1):", ""]
        lengths = list(FLOORS[1]) + list(FLOORS[f])
        out += ["| Run | Round | " + " | ".join(map(str, lengths)) + " |", "|" + " --- |" * (len(lengths) + 2)]
        for run in runs:
            evs = events(run)
            for k in range(0, 5):
                a = accuracy(evs, f, k)
                if a:
                    out.append(f"| `{run}` | {k} | " + " | ".join(fmt(a.get(d)) for d in lengths) + " |")
        out += ["", "Kept answers per length (built from the model's direct answers):", "",
                "| Run | Digits | Drawn | Kept | Shortfall | Kept right (%) |", "| --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            for e in events(run):
                if e["event"] == "build" and e.get("floor") == f:
                    out.append(f"| `{run}` | {e['digits']} | {e['drawn']:,} | {e['kept']:,} | {max(0, e['shortfall']):,} | "
                               f"{fmt(None if e['kept_right'] is None else 100 * e['kept_right'], 2)} |")
        out += ["", "Checks before each round (agreement rule, for comparison only; it decides nothing):", ""]
        out += checks_table(runs, f) + [""]
    spent = sum(seconds_of(r) for r in runs)
    out += [f"**Dollars spent on the control: {dollars(spent):.2f}** (logged GPU wall seconds x $1.10/h; cap $18).", ""]
    out += [notes.read_text().rstrip(), ""] if notes.exists() else []
    return out


def stage_a_gate() -> tuple[bool | None, list[str]]:
    """Stage A passes if both answer-key repeats are at >= 90% at every length 16-19 after round 2."""
    lines, ok = [], True
    for run in SOLID_KEY:
        a = accuracy(events(run), 2, 2)
        if not a:
            return None, ["Not run."]
        low = min(a.get(d, 0) for d in range(16, 20))
        ok &= low >= 90
        lines.append(f"  * `{run}`: lowest at 16-19 after round 2: {low:.1f}%.")
    return ok, [f"* **Gate A: {'passed' if ok else 'failed'}** (>= 90% at every length 16-19 in both repeats)."] + lines


def floor_section(title: str, runs: list[str], floor: int, gate: str | None) -> list[str]:
    planned, runs = runs, present(runs)  # the rule counts planned repeats (2 of 3), not finished ones
    if not runs:
        return [f"## {title}", "", "Not run.", ""]
    out = [f"## {title}", ""]
    if gate:
        v = verdict(planned, floor)
        out += [f"**{gate}: {'passed' if v['learned'] else 'not passed'}** (floor {floor} {v['verdict']}).", ""]
    out += verdict_lines(planned, floor) + ["", f"### Accuracy per length and repeat (percent)", ""]
    out += block_tables(runs, floor, 4)
    if floor >= 3:
        out += [f"### Floor {floor - 1} before floor {floor}'s first round (logged only, no gate)", "",
                "No-carry rule on split 1, current model, 2,000 fresh problems per length; percent.", ""]
        out += floor_below_table(runs, floor) + [""]
    out += [f"### Checks at the new lengths, each round (the condition's own teacher)", ""]
    out += checks_table(runs, floor) + ["", f"### Floor-1 novelty each round (trusted-piece teacher; problems "
                                            "with no carry at cut 1 only; percent)", ""]
    out += floor1_novelty_table(runs, floor) + [""]
    return out


def write_report(path: Path = PILOT_RESULTS / "report.md", notes: Path = PILOT_RESULTS / "notes.md") -> str:
    """The pilot report: generated tables, with the hand-written status, decisions and problems
    from ``notes.md`` (which opens the report)."""
    out = [notes.read_text().rstrip(), ""] if notes.exists() else ["# Pilot report", ""]
    ok, lines = stage_a_gate()
    out += ["## Stage A: answer key, floor 2, rounds 1-2 (repeats 0 and 1)", ""] + lines + [""]
    out += block_tables(present(SOLID_KEY), 2, 2)
    out += floor_section("Stage B: agreement rule, floor 2 (solid floor 1)", SOLID_AGR, 2, "Gate B")
    out += floor_section("Stage B: answer key, floor 2, rounds 3-4 continued (control)", SOLID_KEY, 2, None)
    out += floor_section("Stage B2: agreement rule, floor 3", SOLID_AGR, 3, "Gate B2")
    out += floor_section("Answer-key control, floor 3", KEY_CONTROL[3], 3, None)
    out += floor_section("Stage B3: agreement rule, floor 4", FLOOR4_AGR, 4, "Gate B3")
    out += floor_section("Answer-key control, floor 4", KEY_CONTROL[4], 4, None)
    out += ["## Stage C (iii): under-trained floor 1", ""] + undertrained_section()
    for cond, runs in CONDITIONS.items():
        if cond not in ("solid / agreement", "solid / answer key"):
            out += floor_section(f"Stage C: {cond}, floor 2", runs, 2, None)
    out += ["## Seen coming", ""] + seen_section()
    all_runs = present(SOLID_KEY + SOLID_AGR + ["solid-agr-r1-again"] + KEY_CONTROL[3] + KEY_CONTROL[4]
                       + [r for c, rs in CONDITIONS.items() for r in rs if c not in ("solid / agreement",
                                                                                     "solid / answer key")])
    out += ["## Examples and updates per round", ""] + rounds_table(all_runs) + [""]
    c_runs = [r for c, rs in CONDITIONS.items() if c not in ("solid / agreement", "solid / answer key") for r in rs]
    stages = {
        "Timing and rehearsal set": [("timing", None), ("rehearsal-build-seed2", None)],
        "A (answer key, floor 2 rounds 1-2)": [(r, "floor 2 rounds 1-2") for r in SOLID_KEY],
        "B (agreement floor 2; answer key rounds 3-4)": [(r, "floor 2") for r in SOLID_AGR] + [(r, "floor 2 rounds 3-4") for r in SOLID_KEY],
        "B2 (agreement floor 3)": [(r, "floor 3") for r in SOLID_AGR] + [(r, None) for r in present(KEY_CONTROL[3])],
        "B3 (agreement floor 4; repeat 1 crashed and relaunched)": [(r, "floor 4") for r in SOLID_AGR]
        + [(r, None) for r in present(["solid-agr-r1-again"] + KEY_CONTROL[4])],
        "C": [(r, None) for r in present(c_runs + list(UNDERTRAINED.values()))],
    }
    out += ["## Dollars (logged GPU wall seconds x $1.10/h)", ""] + dollars_section(stages) + [""]
    out += ["## Rehearsal set audit", ""] + rehearsal_section()
    out += control_section()
    text = "\n".join(out) + "\n"
    path.write_text(text)
    return text


if __name__ == "__main__":
    write_report()
    print(f"wrote {PILOT_RESULTS / 'report.md'}")
