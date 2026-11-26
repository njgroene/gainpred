"""The ladder (protocol amendment 29; ``docs/ladder_spec.md``): teaching itself from 8 to 39 digits.

The idea in one paragraph
=========================
A seed model learns 3-7 digit sums from correct answers. Then, two new lengths per round
(8-9, 10-11, ..., 38-39: sixteen rounds, four stages of eight lengths), the model makes its
own training answers for the new lengths and trains on them. No correct answer and no
hand-written filter is ever used above 7 digits. Four arms differ only in how the answers
are made and kept:

1. **direct**: the model's answer to the whole sum, kept if readable;
2. **composed**: the Cut 1 composed answer (two piece answers side by side), kept if readable;
3. **composed + agreed**: the answer at least two of the three cuts produce;
4. **self-paced**: as 3, but a new length is built only if the check says work remains;
   otherwise the check's own pair-agreed answers are stored for it (and upsampled in training).

Cuts (the same rule at every length)
====================================
With h = n // 2, the three cuts have left pieces of h - 1, h and h + 1 digits::

    n = 12:  5+7, 6+6, 7+5        n = 39:  18+21, 19+20, 20+19

Every piece is shorter than both new lengths of its round, so it was taught earlier.

One round (``run_arm``)
=======================
1. **Check**: 2,000 fresh sums per new length; direct answer and all three cuts; no update.
   Logged identically in every arm (``check_report``).
2. **Build**: for each length the arm builds, draw until 10,000 answers are kept or
   200,000 sums are drawn (``collect``).
3. **Train** on new answers + stored answers of every earlier length + 5,000 easy sums per
   length 3-7 answered by the model itself; U = max(3000, 100 ceil(7 N_raw / 25,600)) updates.
4. **Test** every length 3 .. 2r + 7.

Everything lives under ``runs/ladder/`` (volume) and ``results/ladder/`` (here), never next to
the legacy loop (``gainpred.floors``) or the study, so none of their reports can change.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gainpred.cuts import EASY_DIGITS, Cut, carry_crosses, piece_starts
from gainpred.data import draw_easy, draw_number, load, problems_of, record, to_jsonl
from gainpred.floors import _dump, last_checkpoint, load_long, updates_for
from gainpred.formats import Addition, Answer, Unreadable, solve
from gainpred.teachers import ModelSolver, Solver, build_all, majority, readable

# ============================================================================
# Lengths, rounds, stages and cuts
# ============================================================================

LENGTHS = range(8, 40)
ROUNDS = range(1, 17)
STAGES: dict[int, range] = {1: range(8, 16), 2: range(16, 24), 3: range(24, 32), 4: range(32, 40)}

CUTS: dict[int, tuple[Cut, Cut, Cut]] = {
    n: tuple((left, n - left) for left in (n // 2 - 1, n // 2, n // 2 + 1)) for n in LENGTHS  # type: ignore[misc]
}
"""``cuts_middle_8_39``: Cut 1, Cut 2, Cut 3 as (left piece, right piece)."""

ARMS = (1, 2, 3, 4)
ARM_NAMES = {1: "direct", 2: "composed", 3: "composed + agreed", 4: "self-paced"}


def round_lengths(round_: int) -> tuple[int, int]:
    """Round r teaches 2r + 6 and 2r + 7: round 1 is 8-9, round 16 is 38-39."""
    return 2 * round_ + 6, 2 * round_ + 7


def stage_of(round_: int) -> int:
    return (round_ - 1) // 4 + 1


def nonzero_positions(digits: int) -> frozenset[int]:
    """Positions where a piece starts under any of the three cuts: never 0."""
    return frozenset().union(*(piece_starts(cut) for cut in CUTS[digits]))


def draw(rng: random.Random, digits: int) -> Addition:
    """A sum of this length: the ladder's no-zero rule from 8 digits, the study's easy rule below."""
    if digits in CUTS:
        nz = nonzero_positions(digits)
        return Addition(draw_number(rng, digits, nz), draw_number(rng, digits, nz))
    return draw_easy(rng, digits)


def draw_fresh(rng: random.Random, digits: int, n: int, exclude: set[Addition]) -> list[Addition]:
    """``n`` new sums of this length, none in ``exclude`` (which grows)."""
    out = []
    while len(out) < n:
        p = draw(rng, digits)
        if p not in exclude:
            exclude.add(p)
            out.append(p)
    return out


def carry_at_cut1(p: Addition) -> bool:
    return carry_crosses(p, CUTS[p.n_digits][0])


# ============================================================================
# Random streams (decision D9: no arm in the key)
# ============================================================================


def stream_key(seed: int | str, round_: int, digits: int, purpose: str) -> str:
    return f"ladder | s{seed} | {round_} | {digits} | {purpose}"


def stream(seed: int | str, round_: int, digits: int, purpose: str) -> random.Random:
    digest = hashlib.sha256(stream_key(seed, round_, digits, purpose).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def stream_seed(seed: int | str, round_: int, purpose: str) -> int:
    digest = hashlib.sha256(stream_key(seed, round_, 0, purpose).encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


# ============================================================================
# The test set (decision D17)
# ============================================================================

TEST_SEED = 1_006  # 1,001-1,005 and 2,001 are taken (gainpred.data.SEEDS, gainpred.floors.TEST_SEED)
TEST_PER_GROUP = 1_000
LADDER_DATA_DIR = Path("data/ladder")
TEST_FILE = "test_middle_8_39.jsonl"


def existing_test_problems(data_dir: Path, floor_data_dir: Path) -> set[Addition]:
    """Every sum of an existing test set (and of the starting-training file): never drawn again."""
    from gainpred.rounds import known_problems

    out = known_problems(data_dir)
    if (floor_data_dir / "test_floors.jsonl").exists():
        out |= set(problems_of(load(floor_data_dir / "test_floors.jsonl")))
    return out


def make_test_set(exclude: set[Addition], per_group: int = TEST_PER_GROUP) -> list[dict]:
    """Per length 8-39: ``per_group`` sums with a carry across Cut 1, as many without."""
    rng = random.Random(TEST_SEED)
    records, seen = [], set()
    for digits in LENGTHS:
        groups: dict[bool, list[Addition]] = {True: [], False: []}
        while any(len(g) < per_group for g in groups.values()):
            p = draw(rng, digits)
            group = groups[carry_at_cut1(p)]
            if p not in seen and p not in exclude and len(group) < per_group:
                seen.add(p)
                group.append(p)
        for carry in (False, True):
            records += [record(p, carry_across_cut1=carry) for p in groups[carry]]
    return records


def write_test_set(data_dir: Path, floor_data_dir: Path, out_dir: Path = LADDER_DATA_DIR) -> dict:
    """Write the test set and its manifest (sha256 of the bytes); refuses to replace a different set."""
    content = to_jsonl(make_test_set(existing_test_problems(data_dir, floor_data_dir)))
    manifest = {"test_middle_8_39": {"seed": TEST_SEED, "rows": content.count(b"\n"),
                                     "sha256": hashlib.sha256(content).hexdigest()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "manifest.json").exists() and json.loads((out_dir / "manifest.json").read_text()) != manifest:
        raise RuntimeError(f"{out_dir} already holds a different ladder test set")
    (out_dir / TEST_FILE).write_bytes(content)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ============================================================================
# Arm rules and the check
# ============================================================================


def _text(a: Answer | None) -> str | None:
    if a is None:
        return None
    return "?" + a.raw if isinstance(a, Unreadable) else str(a)


def arm_answer(arm: int, cut_answers: list[Answer] | None, direct: Answer | None) -> int | None:
    """What the arm keeps for one sum (None = nothing). Arm 1 needs only the direct answer,
    arm 2 only Cut 1's composed answer, arms 3-4 all three cuts."""
    if arm == 1:
        return readable(direct)
    if arm == 2:
        return readable(cut_answers[0])
    if arm in (3, 4):
        return majority(cut_answers, needed=2)
    raise ValueError(f"unknown arm {arm!r}")


def arm4_builds(m_agreed: int, novelty_agreed: float | None, check_sample: int,
                skip_fraction: float = 0.05, novelty_level: float = 0.02) -> bool:
    """Arm 4's rule (D5): build if fewer than 5% of check sums are pair-agreed, or if at least
    2% of the pair-agreed answers differ from the direct answer."""
    return m_agreed < skip_fraction * check_sample or (novelty_agreed is not None and novelty_agreed >= novelty_level)


def _share(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def check_report(arm: int, problems: list[Addition], cut_answers: list[list[Answer]], direct: list[Answer],
                 skip_fraction: float = 0.05, novelty_level: float = 0.02) -> tuple[dict, list[tuple[Addition, int]]]:
    """Every field of specification section 6, step 1, and the pair-agreed answers (arm 4's pool).

    Correct answers enter only the analysis fields (``direct_acc``, ``cut1_acc``, ``agreed_acc``)."""
    n = len(problems)
    truth = [solve(p) for p in problems]
    cut1 = [c[0] for c in cut_answers]
    readable1 = [i for i, a in enumerate(cut1) if not isinstance(a, Unreadable)]
    agreed = [majority(c, needed=2) for c in cut_answers]
    agreed_idx = [i for i, a in enumerate(agreed) if a is not None]
    kept = [arm_answer(arm, c, d) for c, d in zip(cut_answers, direct)]
    kept_idx = [i for i, a in enumerate(kept) if a is not None]
    novelty_agreed = _share([agreed[i] != direct[i] for i in agreed_idx])
    m = len(agreed_idx)
    report = {
        "check_sums": n,
        "direct_acc": _share([d == t for d, t in zip(direct, truth)]),
        "direct_readable": _share([not isinstance(d, Unreadable) for d in direct]),
        "cut1_acc": _share([a == t for a, t in zip(cut1, truth)]),
        "cut1_readable": len(readable1) / n,
        "confirmed_share": _share([cut1[i] == cut_answers[i][1] or cut1[i] == cut_answers[i][2] for i in readable1]),
        "agreed_share": m / n,
        "m_agreed": m,
        "agreed_acc": _share([agreed[i] == truth[i] for i in agreed_idx]),
        "arm_kept_share": len(kept_idx) / n,
        "novelty_arm": _share([kept[i] != direct[i] for i in kept_idx]),
        "novelty_agreed": novelty_agreed,
        "arm4_verdict": "build" if arm4_builds(m, novelty_agreed, n, skip_fraction, novelty_level) else "skip_length",
    }
    pool = [(problems[i], agreed[i]) for i in agreed_idx]
    return report, pool


def answer_cuts(problems: list[Addition], solver: Solver, cuts: tuple[int, ...] = (0, 1, 2)) -> list[list[Answer]]:
    """The composed answer of each listed cut, for every sum, in one call to the solver."""
    built = build_all(problems, [tuple(CUTS[p.n_digits][c] for c in cuts) for p in problems], solver)
    return [[b.answer for b in bs] for bs in built]


def collect(arm: int, solver: Solver, rng: random.Random, digits: int, quota: int, cap: int,
            exclude: set[Addition], rows: list[dict], max_chunk: int = 20_000) -> tuple[list[tuple[Addition, int]], int]:
    """Draw sums from ``rng`` until ``quota`` are kept by the arm's rule or ``cap`` are drawn.
    The arm asks the model only for what its rule needs. Every sum drawn is logged in ``rows``."""
    kept: list[tuple[Addition, int]] = []
    drawn, rate = 0, 0.3
    while len(kept) < quota and drawn < cap:
        need = quota - len(kept)
        chunk = min(cap - drawn, max_chunk, max(500, math.ceil(1.15 * need / max(rate, 0.02))))
        problems = draw_fresh(rng, digits, chunk, exclude)
        direct = solver(problems) if arm == 1 else [None] * len(problems)
        cuts = (answer_cuts(problems, solver, (0,)) if arm == 2 else
                answer_cuts(problems, solver) if arm in (3, 4) else [None] * len(problems))
        for p, c, d in zip(problems, cuts, direct):
            drawn += 1
            a = arm_answer(arm, c, d)
            rows.append({"a": str(p.a), "b": str(p.b), "direct": _text(d),
                         "cuts": None if c is None else [_text(x) for x in c], "kept": None if a is None else str(a)})
            if a is not None:
                kept.append((p, a))
                if len(kept) == quota:
                    break
        rate = max(len(kept) / drawn, 0.001)
    return kept, drawn


# ============================================================================
# Stored answers and upsampling
# ============================================================================


def kept_path(run_dir: Path, name: str) -> Path:
    return run_dir / "kept" / f"{name}.jsonl.gz"


def save_pool(run_dir: Path, digits: int, round_: int, rows: list[tuple[Addition, int]], source: str,
              tag: str = "") -> str:
    """Store a length's answers once. ``source``: "built" or "check_pool" (arm 4's skip).
    A relaunch that rebuilds a length writes a new file (``tag``), never over an old one."""
    content = "".join(json.dumps({"a": p.a, "b": p.b, "answer": a}) + "\n" for p, a in rows).encode()
    digest = hashlib.sha256(content).hexdigest()
    name = f"{digits}" if not tag else f"{digits}.{tag}"
    path = kept_path(run_dir, name)
    if path.exists():
        raise FileExistsError(f"{path} exists; stored answers are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content, mtime=0))
    index_path = run_dir / "kept" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[str(digits)] = {"round": round_, "rows": len(rows), "sha256": digest, "source": source, "file": path.name}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return digest


def load_pool(run_dir: Path, digits: int) -> tuple[list[tuple[Addition, int]], str]:
    """(answers, source) as last stored for this length; refuses a changed file."""
    entry = json.loads((run_dir / "kept" / "index.json").read_text())[str(digits)]
    content = gzip.decompress((run_dir / "kept" / entry["file"]).read_bytes())
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"stored answers for {digits} digits do not match their fingerprint")
    return [(Addition(r["a"], r["b"]), r["answer"]) for r in map(json.loads, content.decode().splitlines())], entry["source"]


def expand(pool: list, target: int, rng: random.Random) -> list:
    """D7: the whole pool ``target // M`` times, plus a seeded random subset for the remainder."""
    m = len(pool)
    if m == 0 or m >= target:
        return list(pool)
    return pool * (target // m) + rng.sample(pool, target % m)


# ============================================================================
# One arm: rounds 1-16
# ============================================================================


@dataclass(frozen=True)
class LadderConfig:
    seed: int | str = 5
    arm: int = 3
    kept_per_length: int = 10_000
    draw_cap_factor: int = 20
    pool_target: int = 10_000
    replay_per_length: int = 5_000
    check_sample: int = 2_000
    batch_size: int = 256
    passes: int = 7
    min_steps: int = 3_000
    decay_fraction: float = 1_000 / 3_000
    solver_batch_size: int = 1_000
    eval_per_length: int | None = None
    eval_batch_size: int = 1_000
    max_chunk: int = 20_000
    skip_fraction: float = 0.05
    novelty_level: float = 0.02
    setup_level: float = 0.20
    learned_level: float = 0.90
    last_round: int = 16
    steps_override: int | None = None
    """Only the smoke test sets it (100 updates); None = the update count of the rules."""
    diverged_level: float = 0.90
    """Amendment 30 (R1): a trained round below this at a length 3-7 is redone once."""


MAX_LEN = 136  # loaded as in the legacy loop (floors.PILOT_MAX_LEN): 39 digits fit


def eval_sets(data_dir: Path, ladder_data_dir: Path, max_digits: int, per_length: int | None) -> dict:
    from gainpred.train import EvalSet, by_length, by_length_and_carry, select

    easy = [r for r in load(data_dir / "test_easy.jsonl") if r["digits"] <= max_digits]
    sets = {"easy": EvalSet(select(easy, by_length, per_length), by_length)}
    long = [r for r in load(ladder_data_dir / TEST_FILE) if r["digits"] <= max_digits]
    if long:
        sets["ladder"] = EvalSet(select(long, by_length_and_carry, per_length), by_length_and_carry)
    return sets


def per_length(evals: dict[str, dict[str, float]]) -> dict[str, dict]:
    """{digits: {acc_total, acc_carry, acc_no_carry}}; acc_total over both (equal) groups."""
    out = {}
    for k, v in evals.get("easy", {}).items():
        if k != "all":
            out[k] = {"acc_total": v}
    groups = evals.get("ladder", {})
    for d in sorted({k.split("/")[0] for k in groups if k != "all"}, key=int):
        c, nc = groups.get(f"{d}/carry"), groups.get(f"{d}/no-carry")
        out[d] = {"acc_total": (c + nc) / 2 if c is not None and nc is not None else (c if nc is None else nc),
                  "acc_carry": c, "acc_no_carry": nc}
    return out


def stage_learned(acc: dict[str, dict], stage: int, level: float = 0.90) -> bool:
    return all(acc.get(str(d), {}).get("acc_total", 0.0) >= level - 1e-12 for d in STAGES[stage])


SHORT_LENGTHS = range(3, 8)


def diverged(acc: dict[str, dict], level: float = 0.90) -> bool:
    """Amendment 30 (R1): a trained round is diverged if, after its test, ``acc_total`` is below
    ``level`` at one or more of the short lengths 3-7 (a missing length counts as 0)."""
    return any(acc.get(str(d), {}).get("acc_total", 0.0) < level - 1e-12 for d in SHORT_LENGTHS)


def _examples_digest(examples: list[tuple[Addition, int]]) -> str:
    """Fingerprint of a training set in its order (shows that a redo trains on the same examples)."""
    h = hashlib.sha256()
    for p, a in examples:
        h.update(f"{p.a} {p.b} {a}\n".encode())
    return h.hexdigest()


def _trained(dirs: list[Path]) -> list[Path]:
    return [d for d in dirs if d.is_dir() and list((d / "checkpoints").glob("step_*.pt"))]


def _round_result(run_dir: Path, r: int) -> Path | None:
    """The weights a trained round ended with (None: the round trained nothing).

    Amendment 30: if the round was redone, its redo folder (``round_{r}-redo``, or
    ``round_{r}-redo-<tag>`` if a relaunch had to redo it again) is the result, and the first
    attempt's folder is never chosen, whatever the other folders are called. Otherwise the
    last of ``round_{r}`` and a relaunch's ``round_{r}-<tag>`` folders, as before."""
    def is_redo(d: Path) -> bool:
        return d.name == f"round_{r}-redo" or d.name.startswith(f"round_{r}-redo-")

    redo = _trained([run_dir / f"round_{r}-redo", *sorted(d for d in run_dir.glob(f"round_{r}-redo-*"))])
    if redo:
        return last_checkpoint(redo[-1])
    first = _trained([run_dir / f"round_{r}", *sorted(d for d in run_dir.glob(f"round_{r}-*") if not is_redo(d))])
    return last_checkpoint(first[-1]) if first else None


def _latest_weights(run_dir: Path, before: int) -> Path | None:
    """The result of the last trained round before ``before`` (skipped rounds have none)."""
    for r in range(before - 1, 0, -1):
        found = _round_result(run_dir, r)
        if found is not None:
            return found
    return None


def _read_log(run_dir: Path) -> list[dict]:
    path = run_dir / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _first_attempt_test(events: list[dict], round_: int) -> dict | None:
    """The accuracies the first attempt of ``round_`` was tested at (the last such test logged)."""
    found = [e["acc"] for e in events if e.get("event") == "test" and e["round"] == round_ and not e.get("redo")]
    return found[-1] if found else None


def _counting_test(events: list[dict], round_: int) -> dict | None:
    """The test of ``round_`` that counts: its redo's if it was redone, else the last one logged."""
    found = [e for e in events if e.get("event") == "test" and e["round"] == round_]
    redo = [e for e in found if e.get("redo")]
    return (redo or found or [{"acc": None}])[-1]["acc"]


def refuse_continuation(run_dir: Path, round_: int, level: float = 0.90) -> None:
    """Amendment 30 (R3): the continuation of a round is possible only if the round's stored answers
    and those of every earlier round exist, and the first attempt was tested and diverged.
    Raises before anything is loaded, trained or written."""
    index_path = run_dir / "kept" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    missing = [d for d in range(8, round_lengths(round_)[1] + 1)
               if str(d) not in index or not (run_dir / "kept" / index[str(d)]["file"]).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name}: no stored answers for {missing} digits; cannot continue round "
                                f"{round_} without its check and build")
    wrong = [d for d in round_lengths(round_) if index[str(d)]["round"] != round_]
    if wrong:
        raise ValueError(f"{run_dir.name}: stored answers for {wrong} digits are not those of round {round_}")
    events = _read_log(run_dir)
    first = _first_attempt_test(events, round_)
    if first is None or _counting_test(events, round_ - 1) is None:
        raise ValueError(f"{run_dir.name}: rounds {round_ - 1} and {round_} must both have been tested")
    if not diverged(first, level):
        raise ValueError(f"{run_dir.name}: round {round_} did not diverge; there is nothing to redo")
    if any(e.get("event") == "test" and e["round"] == round_ and e.get("redo") for e in events):
        raise ValueError(f"{run_dir.name}: round {round_} was redone already; a round is redone at most once")


def run_arm(cfg: LadderConfig, run: str, seed_checkpoint: Path, ladder_dir: Path, data_dir: Path,
            floor_data_dir: Path, ladder_data_dir: Path, device: torch.device, budget_seconds: float = float("inf"),
            prior_seconds: float = 0.0, first_round: int = 1, tag: str = "",
            make_solver: Callable[[object, int], Solver] | None = None,
            after_round: Callable[[], None] | None = None, clock: Callable[[], float] = time.time,
            call_started: float | None = None, continue_round: int | None = None,
            test_hook: Callable[[int, str, dict], dict] | None = None) -> str:
    """Rounds ``first_round``..``cfg.last_round`` of one arm. Returns how the call ended:
    finished, stage_not_learned, setup_stop, budget_stop, shortfall_stop, diverged_twice or
    start_check_failed.

    ``budget_seconds``: the arm's budget; before each round, the call's own wall seconds plus
    ``prior_seconds`` (earlier calls of the same arm) are compared with it. ``first_round`` > 1
    continues after a crash from the arm's stored answers and its last checkpoint; ``tag``
    names that call's new files. ``call_started``: when the Modal call began (the budget counts
    the call's wall seconds from there); None = now.

    Amendment 30. After the test of every trained round the divergence rule (``settle``) runs:
    a round below ``cfg.diverged_level`` at a length 3-7 is trained once more from the same
    start weights on the same examples with the batch seed "batches-redo", into
    ``round_{r}-redo``, and tested again; diverged twice ends the arm. ``continue_round``: the
    continuation of a round whose first attempt diverged in an earlier call: after a start check
    of its start weights (within 0.2 points of the previous round's logged test at every length),
    it rebuilds that round's training set from the stored answers (no check, no build) and runs
    the same redo, then continues normally. ``test_hook(round, kind, acc)`` (tests only) may
    replace the accuracies of a test; kind is "test", "redo" or "start_check"."""
    from gainpred.train import Stopwatch, TrainConfig, device_name, evaluate, train

    started = clock() if call_started is None else call_started
    run_dir = ladder_dir / run
    if continue_round is not None:
        refuse_continuation(run_dir, continue_round, cfg.diverged_level)
        first_round = continue_round
    earlier = _read_log(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = seed_checkpoint if first_round == 1 else (_latest_weights(run_dir, first_round) or seed_checkpoint)
    model = load_long(weights, device)
    (run_dir / f"config-rounds{first_round}{'-' + tag if tag else ''}.json").write_text(json.dumps(
        {"ladder": asdict(cfg), "run": run, "first_round": first_round, "start_checkpoint": str(weights),
         "seed_checkpoint": str(seed_checkpoint), "budget_seconds": budget_seconds, "prior_seconds": prior_seconds,
         "tag": tag, "continue_round": continue_round, "device": device_name(device), "torch": torch.__version__,
         "max_len": MAX_LEN}, indent=2) + "\n")
    log_file = open(run_dir / "rounds.jsonl", "a")

    def log(event: dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    exclude = existing_test_problems(data_dir, floor_data_dir) | set(problems_of(load(ladder_data_dir / TEST_FILE)))

    def solver_for(round_: int) -> Solver:
        return (make_solver(model, round_) if make_solver else
                ModelSolver(model, device, cfg.solver_batch_size, seed=stream_seed(cfg.seed, round_, "solver")))

    def test(round_: int, max_digits: int, watch, kind: str = "test") -> dict:
        model.eval()
        with watch.time("test"):
            evals = {name: evaluate(model, s.records, s.group_key, cfg.eval_batch_size, device)
                     for name, s in eval_sets(data_dir, ladder_data_dir, max_digits, cfg.eval_per_length).items()}
        acc = per_length(evals)
        if test_hook:
            acc = test_hook(round_, kind, acc)
        if kind != "start_check":
            log({"event": "test", "round": round_, "acc": acc, **({"redo": True} if kind == "redo" else {})})
        return acc

    def end(reason: str, **extra) -> str:
        log({"event": "call_done", "reason": reason, "wall_seconds": clock() - started, **extra})
        log_file.close()
        return reason

    def training_set(round_: int, solver: Solver, watch) -> tuple[list[tuple[Addition, int]], int, dict[str, int]]:
        """Every stored length 8 .. the round's longer length (arm 4's skipped pools expanded,
        D7) and the replay sums of lengths 3-7 drawn from the round's own stream, labelled by
        ``solver`` (the model the round starts from). The same call rebuilds it exactly."""
        examples: list[tuple[Addition, int]] = []
        composition: dict[str, int] = {}
        n_raw = 0
        for d in range(8, round_lengths(round_)[1] + 1):
            rows_, source = load_pool(run_dir, d)
            n_raw += len(rows_)
            if source == "check_pool":
                rows_ = expand(rows_, cfg.pool_target, stream(cfg.seed, round_, d, "upsample"))
            examples += rows_
            composition[str(d)] = len(rows_)
        with watch.time("build"):
            replay_problems = [p for d in EASY_DIGITS
                               for p in draw_fresh(stream(cfg.seed, round_, d, "replay"), d,
                                                   cfg.replay_per_length, exclude)]
            replay_answers = solver(replay_problems)
        replay = [(p, a) for p, a in zip(replay_problems, replay_answers) if isinstance(a, int)]
        examples += replay
        n_raw += len(replay)
        composition["replay"] = len(replay)
        return examples, n_raw, composition

    def train_once(round_: int, examples, steps: int, start: Path, attempt: str, n_raw: int, composition, watch) -> Path:
        """One training of a round: ``attempt`` "first" (batch seed "batches", folder ``round_{r}``)
        or "redo" (reloads ``start``, batch seed "batches-redo", folder ``round_{r}-redo``)."""
        nonlocal model
        redo = attempt == "redo"
        tcfg = TrainConfig(steps=steps, batch_size=cfg.batch_size,
                           seed=stream_seed(cfg.seed, round_, "batches-redo" if redo else "batches"),
                           eval_every=steps, checkpoint_every=steps, eval_per_length=cfg.eval_per_length,
                           warmup_fraction=0.0, decay_fraction=cfg.decay_fraction)
        out = run_dir / (f"round_{round_}-redo" if redo else f"round_{round_}")
        if out.exists():
            out = run_dir / f"{out.name}-{tag or 'again'}"
        if redo:
            model = load_long(start, device)
        with watch.time("train"):
            model = train(model, examples, tcfg, out, {}, device,
                          extra_config={"task": "ladder/round", "round": round_, "arm": cfg.arm, "attempt": attempt,
                                        "start_checkpoint": str(start), "n_raw_examples": n_raw,
                                        "composition": composition, "examples_sha256": _examples_digest(examples)})
        return out

    def settle(round_: int, acc: dict, examples, steps: int, start: Path, n_raw: int, composition, watch,
               after_the_fact: bool = False) -> tuple[dict, Path | None, str | None]:
        """The divergence rule (amendment 30, R1), the one place a round is redone: returns the
        accuracies that count, the redo's folder (None: not redone) and "diverged_twice" or None."""
        if not diverged(acc, cfg.diverged_level):
            return acc, None, None
        log({"event": "diverged", "round": round_,
             "acc_short": {str(d): acc.get(str(d), {}).get("acc_total") for d in SHORT_LENGTHS},
             **({"after_the_fact": True} if after_the_fact else {})})
        out = train_once(round_, examples, steps, start, "redo", n_raw, composition, watch)
        acc = test(round_, round_lengths(round_)[1], watch, "redo")
        return acc, out, ("diverged_twice" if diverged(acc, cfg.diverged_level) else None)

    last_acc: dict = {}
    if first_round == 1:
        watch = Stopwatch(device)
        last_acc = test(0, 9, watch)
        log({"event": "round0", "sec_test": watch.totals["test"]})
    else:
        last_acc = _counting_test(earlier, first_round - 1) or {}

    if continue_round is not None:
        # R3's start check: the start weights must test as the previous round's logged test did
        watch = Stopwatch(device)
        before = continue_round - 1
        now = test(before, round_lengths(before)[1] if before else 9, watch, "start_check")
        diff = {d: 100 * abs(now[d]["acc_total"] - v["acc_total"]) if d in now else None for d, v in last_acc.items()}
        failed = not last_acc or any(v is None or v > 0.2 + 1e-9 for v in diff.values())
        log({"event": "start_check", "round": continue_round, "compared_with_round": before,
             "start_checkpoint": str(weights), "max_diff_points": max((v for v in diff.values() if v is not None),
                                                                      default=None),
             "diff_points": diff, "acc": now, "passed": not failed, "sec_test": watch.totals["test"]})
        if failed:
            log({"event": "start_check_failed", "round": continue_round})
            return end("start_check_failed", round=continue_round)

    start_weights = weights
    for round_ in range(first_round, cfg.last_round + 1):
        # stops before the round, in the order of the task: budget, then the stage rule
        spent = clock() - started + prior_seconds
        if spent >= budget_seconds:
            return end("budget_stop", before_round=round_, spent_seconds=spent)
        if round_ > 1 and (round_ - 1) % 4 == 0 and not stage_learned(last_acc, stage_of(round_ - 1), cfg.learned_level):
            return end("stage_not_learned", stage=stage_of(round_ - 1))
        round_start = clock()
        watch = Stopwatch(device)
        solver = solver_for(round_)
        new = round_lengths(round_)
        composition: dict[str, int] = {}
        redo_out, stop, digest = None, None, None
        continuation = round_ == continue_round
        if continuation:
            # R3: no check and no build; the stored answers, the replay labelled by the start model,
            # then the redo of R1 against the first attempt's logged test
            build: list[int] = []
            round_skipped = False
            examples, n_raw, composition = training_set(round_, solver, watch)
            n_eff, digest = len(examples), _examples_digest(examples)
            steps = cfg.steps_override or updates_for(n_raw, cfg.passes, cfg.batch_size, cfg.min_steps)
            acc, redo_out, stop = settle(round_, _first_attempt_test(earlier, round_), examples, steps, start_weights,
                                         n_raw, composition, watch, after_the_fact=True)
            out = redo_out
        else:
            # 1. check (identical in every arm)
            checks, pools = {}, {}
            for d in new:
                sample = draw_fresh(stream(cfg.seed, round_, d, "check"), d, cfg.check_sample, exclude)
                with watch.time("check"):
                    cuts = answer_cuts(sample, solver)
                    direct = solver(sample)
                report, pool = check_report(cfg.arm, sample, cuts, direct, cfg.skip_fraction, cfg.novelty_level)
                checks[d], pools[d] = report, pool
                log({"event": "check", "round": round_, "digits": d, **report})
                _dump(run_dir / "logs" / f"round{round_}-{d}-check{'-' + tag if tag else ''}.jsonl.gz",
                      [{"a": str(p.a), "b": str(p.b), "direct": _text(a), "cuts": [_text(x) for x in c]}
                       for p, c, a in zip(sample, cuts, direct)])
            if round_ == 1 and any(checks[d]["agreed_share"] < cfg.setup_level for d in new):
                return end("setup_stop", agreed_share={str(d): checks[d]["agreed_share"] for d in new})
            # 2. build (arm 4 builds only where its rule says so)
            build = [d for d in new if cfg.arm != 4 or checks[d]["arm4_verdict"] == "build"]
            for d in new:
                if d not in build:
                    save_pool(run_dir, d, round_, pools[d], "check_pool", tag)
                    log({"event": "skip_length", "round": round_, "digits": d, "pool": len(pools[d])})
                    continue
                rows: list[dict] = []
                with watch.time("build"):
                    kept, drawn = collect(cfg.arm, solver, stream(cfg.seed, round_, d, "build"), d,
                                          cfg.kept_per_length, cfg.draw_cap_factor * cfg.kept_per_length, exclude,
                                          rows, cfg.max_chunk)
                _dump(run_dir / "logs" / f"round{round_}-{d}-build{'-' + tag if tag else ''}.jsonl.gz", rows)
                log({"event": "build", "round": round_, "digits": d, "sums_drawn": drawn, "answers_kept": len(kept),
                     "shortfall_flag": len(kept) < cfg.kept_per_length,
                     "kept_acc": _share([a == solve(p) for p, a in kept]),
                     "kept_carry_share": _share([carry_at_cut1(p) for p, _ in kept])})
                if not kept:
                    return end("shortfall_stop", round=round_, digits=d)
                save_pool(run_dir, d, round_, kept, "built", tag)
            round_skipped = not build
            out = None
            if round_skipped:
                # 3-4. nothing built: no replay, no training (so never redone); the model is tested again
                acc = test(round_, new[1], watch)
                n_raw = n_eff = steps = 0
            else:
                # 3. training set and training; 4. test; then the divergence rule
                examples, n_raw, composition = training_set(round_, solver, watch)
                n_eff, digest = len(examples), _examples_digest(examples)
                steps = cfg.steps_override or updates_for(n_raw, cfg.passes, cfg.batch_size, cfg.min_steps)
                out = train_once(round_, examples, steps, start_weights, "first", n_raw, composition, watch)
                acc = test(round_, new[1], watch)
                acc, redo_out, stop = settle(round_, acc, examples, steps, start_weights, n_raw, composition, watch)
                out = redo_out or out
        if out is not None:
            start_weights = last_checkpoint(out)
        last_acc = acc
        t = watch.totals
        log({"event": "round_done", "round": round_, "built": build, "round_skipped": round_skipped,
             "n_raw_examples": n_raw, "n_effective_examples": n_eff, "num_updates": steps,
             "composition": composition, "sec_check": t["check"], "sec_build": t["build"], "sec_train": t["train"],
             "sec_test": t["test"], "sec_total": clock() - round_start, "redone": redo_out is not None,
             "examples_sha256": digest, **({"continuation": True} if continuation else {})})
        if stop:
            return end(stop, round=round_)
        if round_ % 4 == 0:
            log({"event": "stage", "stage": stage_of(round_), "learned": stage_learned(acc, stage_of(round_),
                                                                                       cfg.learned_level)})
        if after_round:
            after_round()
    return end("finished")


# ============================================================================
# Launching (deployed app, spawned calls; recorded in results/ladder/launches.jsonl)
# ============================================================================

LADDER_RESULTS = Path("results/ladder")
RECORD = LADDER_RESULTS / "launches.jsonl"
DOLLARS_PER_HOUR = 1.10
CAP_DOLLARS = 28.00


def records() -> list[dict]:
    return [json.loads(line) for line in RECORD.read_text().splitlines()] if RECORD.exists() else []


def spawn(function: str, run: str, kwargs: dict, segment: str = "") -> str:
    """Spawn a ladder call with the legacy launch checks: committed code, the deployed app at
    this commit, identical data manifests (addition, legacy floors, ladder), never a key twice."""
    import modal

    from gainpred import launch
    from gainpred.floors import FLOOR_DATA_DIR

    if not launch.code_is_committed():
        raise SystemExit("uncommitted changes in src/ or modal_app.py: commit first")
    key = f"{run}:{segment}"
    if any(r.get("key") == key for r in records()):
        raise SystemExit(f"{key} was launched before; runs are never overwritten")
    commit = launch.git("rev-parse", "HEAD")
    launch.ensure_deployed(commit)
    for fn, local in (("make_data", Path("data/addition/manifest.json")),
                      ("pilot_data", FLOOR_DATA_DIR / "manifest.json"),
                      ("ladder_data", LADDER_DATA_DIR / "manifest.json")):
        if modal.Function.from_name(launch.APP, fn).remote() != json.loads(local.read_text()):
            raise SystemExit(f"cloud data of {fn} differ from {local}; not launching")
    call = modal.Function.from_name(launch.APP, function).spawn(run=run, **kwargs)
    LADDER_RESULTS.mkdir(parents=True, exist_ok=True)
    with RECORD.open("a") as f:
        f.write(json.dumps({"key": key, "run": run, "function": function, "args": kwargs, "commit": commit,
                            "call_id": call.object_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    print(f"{key}: spawned ({call.object_id}) from {commit[:7]}")
    return call.object_id


def call_states() -> dict[str, str]:
    import modal

    out = {}
    for r in records():
        try:
            modal.FunctionCall.from_id(r["call_id"]).get(timeout=0)
            out[r["key"]] = "done"
        except TimeoutError:
            out[r["key"]] = "running"
        except Exception as e:  # noqa: BLE001
            out[r["key"]] = f"failed: {type(e).__name__}: {str(e)[:200]}"
    return out


def fetch(runs: list[str] | None = None) -> None:
    """Download each run's summary files: top-level .json/.jsonl, calls/*.json, kept/index.json
    (never checkpoints or per-sum logs)."""
    import subprocess

    names = runs or sorted({r["run"] for r in records()})
    for name in names:
        for sub in ("", "/calls"):
            listing = subprocess.run(["uv", "run", "modal", "volume", "ls", "--json", "gainpred",
                                      f"runs/ladder/{name}{sub}"], capture_output=True, text=True)
            if listing.returncode:
                continue
            for entry in json.loads(listing.stdout):
                path = entry["filename"]
                if entry["type"] == "file" and path.endswith((".json", ".jsonl")):
                    dest = LADDER_RESULTS / "runs" / name / sub.strip("/") / Path(path).name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", path, str(dest)],
                                   capture_output=True, check=True)
        dest = LADDER_RESULTS / "runs" / name / "kept_index.json"
        subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred",
                        f"runs/ladder/{name}/kept/index.json", str(dest)], capture_output=True)


def call_seconds(name: str) -> dict[str, float]:
    """{call key: wall seconds} of one run, from the calls/*.json each call writes when it ends."""
    folder = LADDER_RESULTS / "runs" / name / "calls"
    return {p.stem: json.loads(p.read_text())["wall_seconds"] for p in sorted(folder.glob("*.json"))} \
        if folder.exists() else {}


def seconds_of(name: str) -> float:
    return sum(call_seconds(name).values())


def spending(seed: int | str = 5) -> float:
    """Logged dollars of every ladder call of this seed (smoke test excluded, D11)."""
    folders = (LADDER_RESULTS / "runs").glob(f"ladder_s{seed}_*") if (LADDER_RESULTS / "runs").exists() else []
    return sum(seconds_of(p.name) for p in folders) / 3600 * DOLLARS_PER_HOUR


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="The ladder (amendment 29).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="write data/ladder (the 8-39 digit test set)")
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("runs", nargs="*")
    sp = sub.add_parser("spending")
    sp.add_argument("--seed", default="5")
    s = sub.add_parser("spawn")
    s.add_argument("function")
    s.add_argument("run")
    s.add_argument("--segment", default="")
    s.add_argument("--kwargs", default="{}", help="JSON")
    args = parser.parse_args()
    if args.command == "data":
        from gainpred.data import DEFAULT_OUT_DIR
        from gainpred.floors import FLOOR_DATA_DIR

        print(write_test_set(DEFAULT_OUT_DIR, FLOOR_DATA_DIR))
    elif args.command == "status":
        for k, v in call_states().items():
            print(f"{k:45s} {v}")
    elif args.command == "fetch":
        fetch(args.runs or None)
    elif args.command == "spending":
        print(f"ladder spending, seed {args.seed} (wall seconds x $1.10/h): ${spending(args.seed):.2f} "
              f"of ${CAP_DOLLARS:.2f}")
    elif args.command == "spawn":
        spawn(args.function, args.run, json.loads(args.kwargs), args.segment)


if __name__ == "__main__":
    main()
