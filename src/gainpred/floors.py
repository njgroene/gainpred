"""Pilot: building on self-taught pieces (floors 2-4). Exploratory; nothing here goes into the paper.

The idea in one paragraph
=========================
The existing rounds teach 8-15 digit sums ("floor 1") from answers built out of 3-7 digit
pieces, sums the model learned from correct answers. Here the model goes further: a 16-23
digit sum ("floor 2") is split into **two** pieces, at least one of 8 digits or more -- a
length the model only ever taught *itself*. The model answers both piece sums, the two
answers are joined, and a teacher keeps or discards the result. Floor 3 (24-31) is then
built from floor-2 pieces, floor 4 (32-39) from floor-3 pieces. Correct answers (``solve``)
never make or filter a training answer of 8 digits or more, except in the answer-key control.

Example, 18 digits, split 8 + 10::

    347530151542738677 + 199090414712738008
    left piece:  34753015 + 19909041    = 54662056       (8-digit sum, self-taught)
    right piece: 1542738677 + 4712738008 = 6255476685    (10-digit sum, self-taught)
    joined:      54662056 | 6255476685  = 546620566255476685

Words used here (the spec's terms)
==================================
floor   a range of lengths: 1 = 8-15, 2 = 16-23, 3 = 24-31, 4 = 32-39.
split   one cut position for both numbers; three per length (``SPLITS``).
piece   one of the two shorter sums a split makes; *self-taught* if 8 digits or more.
repeat  the same condition again with its own random streams (owner's answer 13).

One round (four per floor, two new lengths per round, no decisions)
==================================================================
1. **Checks** with the model before the round: novelty, share kept and confirmed share
   at the two new lengths (with this condition's teacher), and novelty at every floor-1
   length against the teacher that uses trusted 3-7 digit pieces (the existing no-carry rule).
2. **Build** 10,000 kept answers per new length, from at most 200,000 problems drawn.
3. **Rehearse**: the stored kept answers of every length taught earlier on this floor and
   on floors 2..f-1, the floor-1 rehearsal set (10,000 per length, built once by the
   solid model), and 5,000 easy problems per length 3-7 answered by the model (replay).
4. **Train** ``updates_for(examples)`` updates: 7 passes over the round's examples, at
   least 3,000 (owner's change D); the learning-rate schedule stretched in proportion.
5. **Test** every length from 3 up to the highest taught, asked directly (never split).

Everything goes under ``runs/pilot/<run>/`` (and ``results/pilot/`` locally), never next to
the study's runs, so the reports, P9b and the write-up never see the pilot.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gainpred.config import ModelConfig
from gainpred.cuts import EASY_DIGITS, HARD_DIGITS, Cut, carry_crosses, piece_problems, piece_starts
from gainpred.data import draw_easy, draw_hard, draw_number, load, problems_of, record, to_jsonl
from gainpred.formats import Addition, Answer, Unreadable, solve
from gainpred.model import Transformer
from gainpred.teachers import Built, Decision, ModelSolver, Solver, combine, majority, readable

# ============================================================================
# Floors and splits
# ============================================================================

FLOORS: dict[int, range] = {1: range(8, 16), 2: range(16, 24), 3: range(24, 32), 4: range(32, 40)}
SELF_TAUGHT = 8
"""A piece of this many digits or more is self-taught: the model never saw a correct answer there."""

SPLITS: dict[int, tuple[Cut, Cut, Cut]] = {
    # length: (split 1, split 2, split 3) as (left piece, right piece) -- the spec's table, verbatim.
    16: ((7, 9), (8, 8), (9, 7)),
    17: ((7, 10), (8, 9), (9, 8)),
    18: ((8, 10), (9, 9), (10, 8)),
    19: ((8, 11), (9, 10), (10, 9)),
    20: ((9, 11), (10, 10), (11, 9)),
    21: ((9, 12), (10, 11), (11, 10)),
    22: ((10, 12), (11, 11), (12, 10)),
    23: ((10, 13), (11, 12), (12, 11)),
    24: ((8, 16), (16, 8), (7, 17)),
    25: ((8, 17), (17, 8), (9, 16)),
    26: ((8, 18), (18, 8), (10, 16)),
    27: ((8, 19), (19, 8), (11, 16)),
    28: ((8, 20), (20, 8), (12, 16)),
    29: ((8, 21), (21, 8), (13, 16)),
    30: ((8, 22), (22, 8), (14, 16)),
    31: ((8, 23), (23, 8), (15, 16)),
    32: ((8, 24), (24, 8), (7, 25)),
    33: ((8, 25), (25, 8), (9, 24)),
    34: ((8, 26), (26, 8), (10, 24)),
    35: ((8, 27), (27, 8), (11, 24)),
    36: ((8, 28), (28, 8), (12, 24)),
    37: ((8, 29), (29, 8), (13, 24)),
    38: ((8, 30), (30, 8), (14, 24)),
    39: ((8, 31), (31, 8), (15, 24)),
}
"""Split 1 (index 0) is the one the no-carry rule uses."""

PILOT_MAX_LEN = 136
"""A 39-digit problem with its answer and end token is 133 characters; the study's limit is
128. The model has no positional encoding, so the limit is only a sanity check and no weight
depends on it: pilot models are loaded with 136 (owner's answer 15). Nothing else changes."""


def floor_of(digits: int) -> int:
    for f, lengths in FLOORS.items():
        if digits in lengths:
            return f
    return 0


def split_rule(n: int) -> tuple[Cut, Cut, Cut]:
    """How the spec made the table: left pieces h-1, h, h+1 on floor 2 (h = n // 2);
    8, n-8, n-16 on floor 3 and 8, n-8, n-24 on floor 4, with 7 where the third would equal the first."""
    f = floor_of(n)
    if f == 2:
        lefts = (n // 2 - 1, n // 2, n // 2 + 1)
    else:
        third = n - (16 if f == 3 else 24)
        lefts = (8, n - 8, 7 if third == 8 else third)
    return tuple((left, n - left) for left in lefts)  # type: ignore[return-value]


def round_lengths(floor: int, round_: int) -> tuple[int, int]:
    """The two new lengths of a round: floor 2 round 1 is 16-17, round 4 is 22-23."""
    low = FLOORS[floor].start + 2 * (round_ - 1)
    return low, low + 1


def nonzero_positions(digits: int) -> frozenset[int]:
    """Positions that start a piece under any of the three splits: never 0 (as now)."""
    return frozenset().union(*(piece_starts(cut) for cut in SPLITS[digits]))


def draw_floor(rng: random.Random, digits: int) -> Addition:
    nz = nonzero_positions(digits)
    return Addition(draw_number(rng, digits, nz), draw_number(rng, digits, nz))


def draw_any(rng: random.Random, digits: int) -> Addition:
    """Floor lengths with the pilot's no-zero rule; 8-15 and 3-7 exactly as the study draws them."""
    if digits in SPLITS:
        return draw_floor(rng, digits)
    return draw_hard(rng, digits) if digits in HARD_DIGITS else draw_easy(rng, digits)


def draw_fresh(rng: random.Random, digits: int, n: int, exclude: set[Addition]) -> list[Addition]:
    """``n`` new problems of this length, none in ``exclude`` (which grows)."""
    out = []
    while len(out) < n:
        p = draw_any(rng, digits)
        if p not in exclude:
            exclude.add(p)
            out.append(p)
    return out


# ============================================================================
# Random streams (owner's answers 13 and section 2.5)
# ============================================================================


def stream_key(model: str, repeat: int | str, floor: int, round_: int, digits: int, purpose: str) -> str:
    return f"pilot | {model} | {repeat} | {floor} | {round_} | {digits} | {purpose}"


def stream(model: str, repeat: int | str, floor: int, round_: int, digits: int, purpose: str) -> random.Random:
    """A random stream derived from a hash of its whole description, so two streams can only
    coincide if every part is equal: collisions are impossible by construction.

    The condition (teacher, version of floor 1) is *not* part of it: repeat r of every
    condition draws the same fresh problems. The repeat number is."""
    digest = hashlib.sha256(stream_key(model, repeat, floor, round_, digits, purpose).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def stream_seed(model: str, repeat: int | str, floor: int, round_: int, purpose: str) -> int:
    """An integer seed (for the order of training batches) from the same kind of hash."""
    digest = hashlib.sha256(stream_key(model, repeat, floor, round_, 0, purpose).encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


# ============================================================================
# Corrupted pieces (test 2, version A; owner's answers 9 and 10)
# ============================================================================

CORRUPTION_RATE = 0.15


def _uniform(*parts: object) -> float:
    digest = hashlib.sha256(" | ".join(["corrupt", *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def corruption_selected(model: str, problem: Addition, split: int, piece: int, rate: float) -> bool:
    """Is this piece answer replaced? A fixed function of (starting model, problem, split,
    piece): it never looks at what the model answered, and it is the same in every repeat."""
    return _uniform(model, problem.a, problem.b, split, piece, "replace") < rate


def corrupt(answer: Answer, model: str, problem: Addition, split: int, piece: int, rate: float) -> tuple[Answer, bool]:
    """(answer to use, whether it was replaced). One digit position, chosen uniformly, gets a
    different digit chosen uniformly; the first digit never becomes 0. Unreadable answers are
    never replaced (there is no digit to pick)."""
    if rate <= 0 or isinstance(answer, Unreadable) or not corruption_selected(model, problem, split, piece, rate):
        return answer, False
    text = str(answer)
    position = int(_uniform(model, problem.a, problem.b, split, piece, "position") * len(text))
    choices = [d for d in "0123456789" if d != text[position] and not (position == 0 and d == "0")]
    digit = choices[int(_uniform(model, problem.a, problem.b, split, piece, "digit") * len(choices))]
    return int(text[:position] + digit + text[position + 1:]), True


# ============================================================================
# Building answers from two pieces
# ============================================================================

TEACHERS = ("agreement", "no_carry", "answer_key", "direct")
TEACHER_NAMES = {"agreement": "Agreement rule", "no_carry": "No-carry rule", "answer_key": "Answer key",
                 "direct": "Direct answers"}
"""``direct`` is the control of protocol amendment 27: the model's own answer to the whole problem,
kept if readable. No splits, no pieces, no combine; the true sum never makes or filters an answer."""


def _text(answer: Answer) -> str:
    return "?" + answer.raw if isinstance(answer, Unreadable) else str(answer)


@dataclass
class Corruption:
    model: str = "seed2"
    rate: float = 0.0


def build_floor(problems: list[Addition], splits: tuple[int, ...], solver: Solver,
                corruption: Corruption | None = None) -> tuple[list[list[Built]], list[list[dict]]]:
    """Build the listed splits of every problem with one call to the solver.

    Returns the built answers (``teachers.Built``, one per listed split) and, per problem
    and split, a log entry: each piece, the model's answer, the answer used (after any
    replacement), whether it was replaced, and whether the piece is self-taught.
    """
    rate = corruption.rate if corruption else 0.0
    jobs = [(i, s, SPLITS[p.n_digits][s], piece_problems(p, SPLITS[p.n_digits][s]))
            for i, p in enumerate(problems) for s in splits]
    answers = iter(solver([piece for *_, pieces in jobs for piece in pieces]))
    built: list[list[Built]] = [[] for _ in problems]
    info: list[list[dict]] = [[] for _ in problems]
    for i, s, cut, pieces in jobs:
        used, rows = [], []
        for j, (piece, width) in enumerate(zip(pieces, cut)):
            said = next(answers)
            final, replaced = (corrupt(said, corruption.model, problems[i], s, j, rate)
                               if rate and width >= SELF_TAUGHT else (said, False))
            selected = bool(rate and width >= SELF_TAUGHT
                            and corruption_selected(corruption.model, problems[i], s, j, rate))
            used.append(final)
            rows.append({"a": piece.a, "b": piece.b, "said": _text(said), "used": _text(final),
                         "replaced": replaced, "selected": selected, "self_taught": width >= SELF_TAUGHT,
                         "right": final == solve(piece), "said_right": said == solve(piece)})
        b = Built(cut, tuple(used), combine(used, cut))
        built[i].append(b)
        info[i].append({"split": s, "pieces": rows, "joined": _text(b.answer)})
    return built, info


def teacher_splits(teacher: str) -> tuple[int, ...]:
    return {"agreement": (0, 1, 2), "no_carry": (0,), "answer_key": (), "direct": ()}[teacher]


def decide(teacher: str, problems: list[Addition], built: list[list[Built]],
           direct: list[Answer] | None = None) -> list[Decision]:
    """Keep or discard. ``built[i]`` must start with split 1 and hold every split the rule needs.

    * no-carry rule: split 1's joined answer, unless the right answer is longer than its piece;
    * agreement rule: an answer at least two of the three splits produce;
    * answer key (control): the true sum, always;
    * direct answers (control, amendment 27): ``direct[i]``, the model's answer to the whole
      problem, kept if it is readable. Nothing else is filtered.
    """
    if teacher == "direct":
        if direct is None or len(direct) != len(problems):
            raise ValueError("the direct-answers teacher needs one direct answer per problem")
        return [Decision(p, readable(a), (a,), None) for p, a in zip(problems, direct)]
    out = []
    for p, bs in zip(problems, built):
        if teacher == "no_carry":
            b = bs[0]
            out.append(Decision(p, readable(b.answer) if b.fits else None, (b.answer,), 0))
        elif teacher == "agreement":
            answers = [b.answer for b in bs[:3]]
            kept = majority(answers, needed=2)
            out.append(Decision(p, kept, tuple(answers), None if kept is None else answers.index(kept)))
        elif teacher == "answer_key":
            out.append(Decision(p, solve(p), ()))
        else:
            raise ValueError(f"unknown teacher {teacher!r}")
    return out


def confirmed_by_other(d: Decision, bs: list[Built]) -> bool:
    """Is the kept answer identical to the joined answer of a split other than its source?"""
    return any(b.answer == d.kept for j, b in enumerate(bs) if j != d.source)


def piece_stats(info: list[list[dict]]) -> dict:
    """With correct answers, for analysis only: how often piece answers are right."""
    n = defaultdict(int)
    for per_problem in info:
        for split in per_problem:
            for row in split["pieces"]:
                kind = "self_taught" if row["self_taught"] else "trusted"
                n[f"{kind}"] += 1
                n[f"{kind}_right"] += row["right"]
                n[f"{kind}_said_right"] += row["said_right"]
                n["replaced"] += row["replaced"]
                n["selected"] += row["selected"]
                n["selected_unreadable"] += row["selected"] and not row["replaced"]
    out = {"pieces_self_taught": n["self_taught"], "pieces_trusted": n["trusted"],
           "replaced": n["replaced"], "selected_unreadable": n["selected_unreadable"]}
    for kind in ("self_taught", "trusted"):
        if n[kind]:
            out[f"{kind}_right"] = n[f"{kind}_right"] / n[kind]
            out[f"{kind}_said_right"] = n[f"{kind}_said_right"] / n[kind]
    if n["self_taught"]:
        out["replaced_share"] = n["replaced"] / n["self_taught"]
    return out


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode(), mtime=0))


def _undump(path: Path) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]


def problem_row(p: Addition, d: Decision, info: list[dict], direct: Answer | None = None) -> dict:
    row = {"a": str(p.a), "b": str(p.b), "splits": info, "kept": None if d.kept is None else str(d.kept),
           "source": d.source, "right": None if d.kept is None else d.kept == solve(p)}
    if direct is not None:
        row["direct"] = _text(direct)
    return row


def collect(teacher: str, solver: Solver, rng: random.Random, digits: int, quota: int, cap: int,
            exclude: set[Addition], corruption: Corruption | None, rows: list[dict],
            max_chunk: int = 20_000) -> tuple[list[tuple[Addition, int]], int, list[Decision], list[list[Built]], list]:
    """Draw problems in order from ``rng`` until ``quota`` answers are kept or ``cap`` are drawn.

    Every problem drawn is logged in ``rows``. The teacher sees a prefix of the same stream
    whatever its keep rate, so two teachers given the same repeat see the same problems.
    """
    kept: list[tuple[Addition, int]] = []
    decisions: list[Decision] = []
    built_all: list[list[Built]] = []
    info_all: list = []
    drawn, rate = 0, 0.3
    while len(kept) < quota and drawn < cap:
        need = quota - len(kept)
        chunk = min(cap - drawn, max_chunk, max(500, math.ceil(1.15 * need / max(rate, 0.02))))
        problems = draw_fresh(rng, digits, chunk, exclude)
        splits = teacher_splits(teacher)
        built, info = build_floor(problems, splits, solver, corruption) if splits else (
            [[] for _ in problems], [[] for _ in problems])
        direct = solver(problems) if teacher == "direct" else None  # whole problems, greedy
        for i, (p, d, bs, inf) in enumerate(zip(problems, decide(teacher, problems, built, direct), built, info)):
            drawn += 1
            rows.append(problem_row(p, d, inf, None if direct is None else direct[i]))
            decisions.append(d)
            built_all.append(bs)
            info_all.append(inf)
            if d.kept is not None:
                kept.append((p, d.kept))
                if len(kept) == quota:
                    break
        rate = max(len(kept) / drawn, 0.001)
    return kept, drawn, decisions, built_all, info_all


# ============================================================================
# Test set for 16-39 digits (made before any training data)
# ============================================================================

TEST_SEED = 1_005  # the study's sets use 1,001-1,004 and 2,001 (gainpred.data.SEEDS)
TEST_PER_GROUP = 1_000
FLOOR_DATA_DIR = Path("data/floors")


def make_test_floors(per_group: int = TEST_PER_GROUP) -> list[dict]:
    """Per length 16-39: ``per_group`` problems with a carry across split 1 and as many without
    (as the study's test set does at 8-15). Fresh draws have about 61% with that carry; this
    set has 50%. Problems never repeat; other test sets and training data are shorter."""
    rng = random.Random(TEST_SEED)
    records, seen = [], set()
    for digits in SPLITS:
        groups: dict[bool, list[Addition]] = {True: [], False: []}
        while any(len(g) < per_group for g in groups.values()):
            p = draw_floor(rng, digits)
            group = groups[carry_crosses(p, SPLITS[digits][0])]
            if p not in seen and len(group) < per_group:
                seen.add(p)
                group.append(p)
        for carry in (False, True):
            records += [record(p, carry_across_cut1=carry) for p in groups[carry]]
    return records


def write_floor_data(out_dir: Path = FLOOR_DATA_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    content = to_jsonl(make_test_floors())
    (out_dir / "test_floors.jsonl").write_bytes(content)
    manifest = {"test_floors": {"seed": TEST_SEED, "rows": content.count(b"\n"),
                                "sha256": hashlib.sha256(content).hexdigest()}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ============================================================================
# Models
# ============================================================================


def load_long(path: Path, device: torch.device | str = "cpu") -> Transformer:
    """A checkpoint loaded with ``max_len`` 136 (pilot only). The weights are loaded unchanged."""
    checkpoint = torch.load(path, map_location=device)
    cfg = dict(checkpoint["model_config"], betas=tuple(checkpoint["model_config"]["betas"]), max_len=PILOT_MAX_LEN)
    model = Transformer(ModelConfig(**cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def last_checkpoint(round_dir: Path) -> Path:
    return sorted((round_dir / "checkpoints").glob("step_*.pt"))[-1]


# ============================================================================
# The floor-1 rehearsal set (built once per starting model, the same file in every condition)
# ============================================================================

REHEARSAL_PER_LENGTH = 10_000


def rehearsal_dir(pilot_dir: Path, model: str) -> Path:
    return pilot_dir / f"rehearsal-floor1-{model}"


def build_rehearsal(pilot_dir: Path, model: str, checkpoint: Path, data_dir: Path, floor_data_dir: Path,
                    device: torch.device, per_length: int = REHEARSAL_PER_LENGTH,
                    make_solver: Callable[[Transformer], Solver] | None = None) -> dict:
    """10,000 answers per length 8-15, built by the solid model with the no-carry rule from
    trusted 3-7 digit pieces (the study's teacher, unchanged). Audited, never filtered."""
    from gainpred.rounds import collect_kept, known_problems
    from gainpred.teachers import no_carry

    out = rehearsal_dir(pilot_dir, model)
    if (out / "meta.json").exists():
        raise FileExistsError(f"{out} already holds the rehearsal set; it is built once")
    started = time.time()
    m = load_long(checkpoint, device)
    solver = make_solver(m) if make_solver else ModelSolver(m, device, 1_000, seed=0)
    exclude = known_problems(data_dir) | set(problems_of(load(floor_data_dir / "test_floors.jsonl")))
    rows, audit = [], {}
    for digits in FLOORS[1]:
        kept, drawn, _ = collect_kept(no_carry, solver, stream(model, "-", 1, 0, digits, "rehearsal"), digits,
                                      per_length, 20 * per_length, exclude)
        wrong = sum(a != solve(p) for p, a in kept)
        audit[digits] = {"kept": len(kept), "drawn": drawn, "wrong": wrong}
        rows += [{"a": p.a, "b": p.b, "answer": a, "digits": digits} for p, a in kept]
    content = "".join(json.dumps(r) + "\n" for r in rows).encode()
    out.mkdir(parents=True)
    (out / "answers.jsonl.gz").write_bytes(gzip.compress(content, mtime=0))
    meta = {"model": model, "checkpoint": str(checkpoint), "sha256": hashlib.sha256(content).hexdigest(),
            "rows": len(rows), "audit": audit, "seconds": time.time() - started}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_rehearsal(pilot_dir: Path, model: str) -> tuple[list[tuple[Addition, int]], str]:
    """The set and its fingerprint; refuses a file whose fingerprint changed."""
    out = rehearsal_dir(pilot_dir, model)
    meta = json.loads((out / "meta.json").read_text())
    content = gzip.decompress((out / "answers.jsonl.gz").read_bytes())
    if hashlib.sha256(content).hexdigest() != meta["sha256"]:
        raise RuntimeError("the floor-1 rehearsal set does not match its recorded fingerprint")
    rows = [json.loads(line) for line in content.decode().splitlines()]
    return [(Addition(r["a"], r["b"]), r["answer"]) for r in rows], meta["sha256"]


# ============================================================================
# A pilot run: one condition, one repeat
# ============================================================================


@dataclass(frozen=True)
class PilotConfig:
    teacher: str = "agreement"
    version: str = "solid"
    """Floor 1: "solid", "corrupted" (the solid model, piece answers replaced) or "undertrained"."""
    repeat: int = 0
    model: str = "seed2"
    """Starting model: names the random streams, the corruption function and the rehearsal set."""
    corruption: float = 0.0
    kept_per_length: int = 10_000
    draw_cap_factor: int = 20
    replay_per_length: int = 5_000
    check_sample: int = 2_000
    batch_size: int = 256
    passes: int = 7
    min_steps: int = 3_000
    decay_fraction: float = 1_000 / 3_000
    """The study's 1,000 decay updates of 3,000, stretched in proportion to the updates."""
    solver_batch_size: int = 1_000
    eval_per_length: int | None = None
    max_chunk: int = 20_000


def updates_for(examples: int, passes: int = 7, batch_size: int = 256, minimum: int = 3_000) -> int:
    """Owner's change D: 7 x examples / 256, rounded up to the next 100, never below 3,000."""
    return max(minimum, math.ceil(passes * examples / batch_size / 100) * 100)


def kept_path(run_dir: Path, digits: int) -> Path:
    return run_dir / "kept" / f"{digits}.jsonl.gz"


def save_kept(run_dir: Path, digits: int, floor: int, round_: int, kept: list[tuple[Addition, int]]) -> str:
    content = "".join(json.dumps({"a": p.a, "b": p.b, "answer": a}) + "\n" for p, a in kept).encode()
    path = kept_path(run_dir, digits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content, mtime=0))
    digest = hashlib.sha256(content).hexdigest()
    index_path = run_dir / "kept" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[str(digits)] = {"floor": floor, "round": round_, "rows": len(kept), "sha256": digest}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return digest


def load_kept(run_dir: Path, digits: int) -> list[tuple[Addition, int]]:
    """The answers kept when this length was last built; refuses a changed file."""
    content = gzip.decompress(kept_path(run_dir, digits).read_bytes())
    index = json.loads((run_dir / "kept" / "index.json").read_text())
    if hashlib.sha256(content).hexdigest() != index[str(digits)]["sha256"]:
        raise RuntimeError(f"stored answers for {digits} digits do not match their fingerprint")
    return [(Addition(r["a"], r["b"]), r["answer"]) for r in map(json.loads, content.decode().splitlines())]


def eval_sets(data_dir: Path, floor_data_dir: Path, max_digits: int, per_length: int | None) -> dict:
    from gainpred.train import EvalSet, by_length, by_length_and_carry, select

    sets = {
        "easy": EvalSet(select(load(data_dir / "test_easy.jsonl"), by_length, per_length), by_length),
        "study": EvalSet(select(load(data_dir / "test_study.jsonl"), by_length_and_carry, per_length),
                         by_length_and_carry),
        "study5": EvalSet(select(load(data_dir / "test_study_5cut.jsonl"), by_length_and_carry, per_length),
                          by_length_and_carry),
    }
    floors = [r for r in load(floor_data_dir / "test_floors.jsonl") if r["digits"] <= max_digits]
    if floors:
        sets["floors"] = EvalSet(select(floors, by_length_and_carry, per_length), by_length_and_carry)
    return sets


def run_floor(cfg: PilotConfig, run: str, floor: int, first_round: int, last_round: int, start_checkpoint: Path,
              pilot_dir: Path, data_dir: Path, floor_data_dir: Path, device: torch.device,
              base_run: str | None = None, seen: bool = False,
              make_solver: Callable[[Transformer, int], Solver] | None = None,
              after_round: Callable[[], None] | None = None,
              copy_lower_from: str | None = None,
              expect_direct_right: dict[str, float] | None = None) -> Transformer:
    """Rounds ``first_round``..``last_round`` of ``floor`` for one condition and repeat.

    ``start_checkpoint``: the model to start from when ``first_round`` is 1; otherwise the run
    continues from its own last round. ``base_run``: on floors 3-4, the run whose stored
    lower-floor answers are rehearsed (itself by default; the agreement run of the same repeat
    for the answer-key control, owner's change C). ``seen``: compute the two "seen coming"
    numbers before the first round of floor 2.

    Amendment 28 (the control one floor up) adds two options, used only on a floor's round 1:

    * ``copy_lower_from``: copy that run's stored answers for every floor from 2 up to the floor
      below into this run's folder, once, checking each copy's sha256 against the other run's index.
      Unlike ``base_run``, later floors of this run then rehearse its *own* answers of the floors it
      taught itself (``base_run`` would take every lower floor from the other run, on every floor).
    * ``expect_direct_right``: ``{digits: share}``, the ``audit_direct_right`` the other run logged in
      its own round 1 with the same model and the same fresh problems. If this run's value differs by
      more than 0.2 points at any listed length, it stops before building or training anything
      (the wrong model or the wrong problems were loaded).
    """
    from gainpred.checks import run_checks
    from gainpred.teachers import no_carry
    from gainpred.train import Stopwatch, TrainConfig, device_name, evaluate, train

    started = time.time()
    run_dir = pilot_dir / run
    for k in range(first_round, last_round + 1):
        if (run_dir / f"floor{floor}" / f"round_{k}").exists():
            raise FileExistsError(f"{run_dir} already holds floor {floor} round {k}; runs are never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    if first_round == 1:
        checkpoint = start_checkpoint
    else:
        checkpoint = last_checkpoint(run_dir / f"floor{floor}" / f"round_{first_round - 1}")
    model = load_long(checkpoint, device)
    floor1, floor1_sha = load_rehearsal(pilot_dir, cfg.model)
    base_dir = pilot_dir / (base_run or run)
    if base_run and base_run != run:
        # The control rehearses the very answers the agreement run kept on the lower floors.
        for f in range(2, floor):
            for d in FLOORS[f]:
                rows = load_kept(base_dir, d)
                index = json.loads((base_dir / "kept" / "index.json").read_text())[str(d)]
                save_kept(run_dir, d, index["floor"], index["round"], rows)
    copied = copy_lower(pilot_dir / copy_lower_from, run_dir, floor) if copy_lower_from and first_round == 1 else None
    segment = {"pilot": asdict(cfg), "run": run, "floor": floor, "rounds": [first_round, last_round],
               "start_checkpoint": str(checkpoint), "base_run": base_run, "seen": seen,
               "floor1_rehearsal_sha256": floor1_sha, "device": device_name(device), "torch": torch.__version__,
               "max_len": PILOT_MAX_LEN, "copy_lower_from": copy_lower_from, "copied": copied,
               "expect_direct_right": expect_direct_right}
    (run_dir / f"config-floor{floor}-rounds{first_round}-{last_round}.json").write_text(
        json.dumps(segment, indent=2) + "\n")
    log_file = open(run_dir / "rounds.jsonl", "a")

    def log(event: dict) -> None:
        log_file.write(json.dumps({"floor": floor, **event}) + "\n")
        log_file.flush()

    corruption = Corruption(cfg.model, cfg.corruption) if cfg.corruption else None
    exclude = _known(data_dir, floor_data_dir) | {p for p, _ in floor1}

    def solver_for(round_: int) -> Solver:
        return (make_solver(model, round_) if make_solver else
                ModelSolver(model, device, cfg.solver_batch_size, seed=stream_seed(cfg.model, cfg.repeat, floor,
                                                                                  round_, "solver")))

    def run_evals(round_: int, max_digits: int, watch) -> None:
        model.eval()
        with watch.time("test"):
            for name, s in eval_sets(data_dir, floor_data_dir, max_digits, cfg.eval_per_length).items():
                log({"event": "eval", "round": round_, "set": name,
                     "accuracy": evaluate(model, s.records, s.group_key, 1_000, device)})

    def logs(round_: int, digits: int, purpose: str, rows: list[dict]) -> None:
        _dump(run_dir / "logs" / f"floor{floor}-round{round_}-{digits}-{purpose}.jsonl.gz", rows)

    def floor_checks(round_: int, digits: int, teacher: str, purpose: str, solver: Solver, watch,
                     corrupt: bool) -> dict:
        """Novelty, share kept and confirmed share at one floor length, on fresh problems.
        All three splits are built (with the corruption, if any); direct answers never corrupted."""
        sample = draw_fresh(stream(cfg.model, cfg.repeat, floor, round_, digits, purpose), digits,
                            cfg.check_sample, exclude)
        with watch.time("checks"):
            built, info = build_floor(sample, (0, 1, 2), solver, corruption if corrupt else None)
            direct = solver(sample)
        decisions = decide(teacher, sample, built)
        kept = [(d, bs, a) for d, bs, a in zip(decisions, built, direct) if d.kept is not None]
        agree = decide("agreement", sample, built)
        report = {
            "digits": digits, "teacher": teacher, "problems": len(sample), "kept": len(kept),
            "kept_share": len(kept) / len(sample),
            "novelty": sum(d.kept != a for d, _, a in kept) / len(kept) if kept else None,
            "confirmed_share": ("not applicable" if teacher == "answer_key" else
                                sum(confirmed_by_other(d, bs) for d, bs, _ in kept) / len(kept) if kept else None),
            "agreement_kept_share": sum(d.kept is not None for d in agree) / len(sample),
            # with correct answers, for analysis only
            "audit_kept_right": sum(d.kept == solve(d.problem) for d, _, _ in kept) / len(kept) if kept else None,
            "audit_wrong_confirmed": (sum(d.kept != solve(d.problem) and confirmed_by_other(d, bs)
                                          for d, bs, _ in kept) / len(kept) if kept else None),
            "audit_direct_right": sum(a == solve(p) for p, a in zip(sample, direct)) / len(sample),
            **piece_stats(info),
        }
        logs(round_, digits, purpose, [problem_row(p, d, inf, a) for p, d, inf, a in
                                       zip(sample, decisions, info, direct)])
        return report

    def floor1_novelty(round_: int, purpose: str, solver: Solver, watch) -> dict:
        """Novelty at every floor-1 length against the trusted-piece teacher (the study's
        no-carry rule, pieces of 3-7 digits). It covers only problems with no carry at cut 1."""
        out = {}
        for d in FLOORS[1]:
            sample = draw_fresh(stream(cfg.model, cfg.repeat, floor, round_, d, purpose), d, cfg.check_sample,
                                exclude)
            with watch.time("checks"):
                report = run_checks(no_carry(sample, solver), solver)
            out[str(d)] = {"novelty": report.check_1, "kept": report.kept, "problems": report.problems,
                           "confirmed_share": report.check_2b}
        return out

    watch = Stopwatch(device)
    lo = FLOORS[floor].start
    if first_round == 1:
        run_evals(0, lo - 1, watch)  # where every shorter length stands before the floor
        if seen and floor == 2:
            solver = solver_for(0)
            log({"event": "seen_coming", "version": cfg.version, "repeat": cfg.repeat,
                 "confirmed_share": {str(d): floor_checks(0, d, "no_carry", "seen", solver, watch, True)
                                     for d in (16, 17)},
                 "floor1_novelty": floor1_novelty(0, "seen_floor1", solver, watch)})
        if floor >= 3:
            solver = solver_for(0)
            log({"event": "floor_below_checks",
                 "checks": {str(d): floor_checks(0, d, "no_carry", "below", solver, watch, False)
                            for d in FLOORS[floor - 1]}})
        log({"event": "before_floor", "seconds": dict(watch.totals)})

    for round_ in range(first_round, last_round + 1):
        watch = Stopwatch(device)
        solver = solver_for(round_)
        new = round_lengths(floor, round_)
        # 1. checks, before anything is built
        # The direct-answers control logs the agreement rule's checks, for comparison only.
        check_teacher = "agreement" if cfg.teacher == "direct" else cfg.teacher
        measured = {}
        for d in new:
            report = floor_checks(round_, d, check_teacher, "checks", solver, watch, True)
            measured[str(d)] = report["audit_direct_right"]
            log({"event": "checks", "round": round_, **report})
        if expect_direct_right and round_ == 1:
            # Amendment 28: same model and same problems as the other run, so the same answers.
            ok = start_check_passes(measured, expect_direct_right)
            log({"event": "start_check", "measured": measured, "expected": expect_direct_right, "passed": ok})
            if not ok:
                log_file.close()
                raise RuntimeError(f"start check failed: direct answers right {measured}, expected "
                                   f"{expect_direct_right}; nothing was built or trained")
        log({"event": "floor1_checks", "round": round_, "novelty": floor1_novelty(round_, "floor1", solver, watch)})
        # 2. build
        examples: list[tuple[Addition, int]] = []
        composition: dict[str, int] = {}
        for d in new:
            rows: list[dict] = []
            with watch.time("build"):
                kept, drawn, decisions, built, info = collect(
                    cfg.teacher, solver, stream(cfg.model, cfg.repeat, floor, round_, d, "build"), d,
                    cfg.kept_per_length, cfg.draw_cap_factor * cfg.kept_per_length, exclude, corruption, rows,
                    cfg.max_chunk)
            logs(round_, d, "build", rows)
            kept_decisions = [(dd, bs) for dd, bs in zip(decisions, built) if dd.kept is not None]
            by_pieces = defaultdict(lambda: [0, 0])
            for dd, bs in kept_decisions:
                if dd.source is not None:
                    n_self = sum(w >= SELF_TAUGHT for w in bs[dd.source].cut)
                    by_pieces[n_self][0] += 1
                    by_pieces[n_self][1] += dd.kept == solve(dd.problem)
            digest = save_kept(run_dir, d, floor, round_, kept)
            log({"event": "build", "round": round_, "digits": d, "drawn": drawn, "kept": len(kept),
                 "shortfall": cfg.kept_per_length - len(kept), "kept_share": len(kept) / drawn if drawn else 0.0,
                 "kept_right": sum(a == solve(p) for p, a in kept) / len(kept) if kept else None,
                 "kept_wrong_confirmed": (sum(dd.kept != solve(dd.problem) and confirmed_by_other(dd, bs)
                                              for dd, bs in kept_decisions) / len(kept)
                                          if kept and cfg.teacher == "agreement" else None),
                 "kept_right_by_self_taught_pieces": {str(k): v[1] / v[0] for k, v in sorted(by_pieces.items())},
                 "sha256": digest, **piece_stats(info)})
            examples += kept
            composition[f"new_{d}"] = len(kept)
        # 3. rehearse: earlier lengths of this floor, every lower floor, floor 1, replay
        for d in range(lo, new[0]):
            rows_ = load_kept(run_dir, d)
            examples += rows_
            composition[f"stored_{d}"] = len(rows_)
        for f in range(2, floor):
            for d in FLOORS[f]:
                rows_ = load_kept(run_dir, d)
                examples += rows_
                composition[f"stored_{d}"] = len(rows_)
        examples += floor1
        composition["floor1_rehearsal"] = len(floor1)
        with watch.time("build"):
            replay_problems = [p for d in EASY_DIGITS
                               for p in draw_fresh(stream(cfg.model, cfg.repeat, floor, round_, d, "replay"), d,
                                                   cfg.replay_per_length, exclude)]
            replay_answers = solver(replay_problems)
        replay = [(p, a) for p, a in zip(replay_problems, replay_answers) if isinstance(a, int)]
        examples += replay
        composition["replay"] = len(replay)
        # 4. train
        steps = updates_for(len(examples), cfg.passes, cfg.batch_size, cfg.min_steps)
        tcfg = TrainConfig(steps=steps, batch_size=cfg.batch_size,
                           seed=stream_seed(cfg.model, cfg.repeat, floor, round_, "batches"),
                           eval_every=steps, checkpoint_every=steps, eval_per_length=cfg.eval_per_length,
                           warmup_fraction=0.0, decay_fraction=cfg.decay_fraction)
        with watch.time("train"):
            model = train(model, examples, tcfg, run_dir / f"floor{floor}" / f"round_{round_}", {}, device,
                          extra_config={"task": "pilot/floor-round", "floor": floor, "round": round_,
                                        "examples": len(examples), "composition": composition})
        # 5. test
        run_evals(round_, new[1], watch)
        log({"event": "round_done", "round": round_, "examples": len(examples), "updates": steps,
             "decay_updates": round(steps * cfg.decay_fraction), "composition": composition,
             "seconds": dict(watch.totals)})
        if after_round:
            after_round()
    log({"event": "call_done", "rounds": [first_round, last_round], "wall_seconds": time.time() - started})
    log_file.close()
    return model


START_CHECK_POINTS = 0.2
"""Amendment 28: the largest allowed difference, in percentage points, in the start check."""


def start_check_passes(measured: dict[str, float], expected: dict[str, float]) -> bool:
    """Does every expected length's share of right direct answers match within 0.2 points?"""
    return all(k in measured and abs(measured[k] - v) * 100 <= START_CHECK_POINTS + 1e-9
               for k, v in expected.items())


def copy_lower(source: Path, run_dir: Path, floor: int) -> dict[str, str]:
    """Copy ``source``'s stored answers for floors 2 .. floor-1 (files and index entries) into
    ``run_dir``. Each file is copied byte for byte and its content's sha256 checked against the
    source's index before and after; nothing in ``source`` is written. Refuses to replace a
    length this run already stores. Returns {digits: sha256}."""
    src_index = json.loads((source / "kept" / "index.json").read_text())
    out = {}
    for f in range(2, floor):
        for d in FLOORS[f]:
            if kept_path(run_dir, d).exists():
                raise FileExistsError(f"{run_dir} already stores {d} digits; runs are never overwritten")
            raw = kept_path(source, d).read_bytes()
            entry = src_index[str(d)]
            if hashlib.sha256(gzip.decompress(raw)).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"{source} stored answers for {d} digits do not match its index")
            dest = kept_path(run_dir, d)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            index_path = run_dir / "kept" / "index.json"
            index = json.loads(index_path.read_text()) if index_path.exists() else {}
            index[str(d)] = dict(entry)
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            if hashlib.sha256(gzip.decompress(dest.read_bytes())).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"the copy of {d} digits does not match {source}'s index")
            out[str(d)] = entry["sha256"]
    return out


def _known(data_dir: Path, floor_data_dir: Path) -> set[Addition]:
    from gainpred.rounds import known_problems

    return known_problems(data_dir) | set(problems_of(load(floor_data_dir / "test_floors.jsonl")))


# ============================================================================
# Timing measurement (before stage A)
# ============================================================================


def measure_timing(checkpoint: Path, device: torch.device, out: Path) -> dict:
    """Time 100 training updates at widths 85, 109 and 133 (23, 31 and 39 digits), and building
    1,000 agreement-rule answers (three splits, six pieces) and 1,000 direct answers at 23, 31, 39."""
    from gainpred.train import Stopwatch, TrainConfig, train

    model = load_long(checkpoint, device)
    watch = Stopwatch(device)
    result: dict = {"train": {}, "build": {}, "direct": {}}
    rng = random.Random(0)
    for digits in (23, 31, 39):
        problems = [draw_floor(rng, digits) for _ in range(1_000)]
        solver = ModelSolver(model, device, 1_000, seed=0)
        with watch.time(f"build{digits}"):
            built, _ = build_floor(problems, (0, 1, 2), solver)
        with watch.time(f"direct{digits}"):
            solver(problems)
        result["build"][digits] = watch.totals[f"build{digits}"] / len(problems)
        result["direct"][digits] = watch.totals[f"direct{digits}"] / len(problems)
        examples = [(p, solve(p)) for p in (draw_floor(rng, digits) for _ in range(256 * 4))]
        cfg = TrainConfig(steps=100, batch_size=256, seed=0, eval_every=100, checkpoint_every=100, warmup_fraction=0.0,
                          decay_fraction=0.0, save_checkpoints=False)
        copy = load_long(checkpoint, device)
        t0 = time.time()
        train(copy, examples, cfg, out / f"train{digits}", {}, device)
        result["train"][digits] = (time.time() - t0) / 100
    result["width"] = {d: 3 * d + 16 for d in (23, 31, 39)}
    (out / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ============================================================================
# Launching (deployed app, spawned calls; recorded in results/pilot/launches.jsonl)
# ============================================================================

PILOT_RESULTS = Path("results/pilot")
RECORD = PILOT_RESULTS / "launches.jsonl"
A10_DOLLARS_PER_HOUR = 1.10


def records() -> list[dict]:
    return [json.loads(line) for line in RECORD.read_text().splitlines()] if RECORD.exists() else []


def spawn(function: str, run: str, kwargs: dict, segment: str = "") -> str:
    """Spawn a pilot call on the deployed app, with the study's launch checks: committed code,
    the deployed app at this commit, identical data fingerprints, and never a second call for
    the same run and segment."""
    import modal

    from gainpred import launch

    if not launch.code_is_committed():
        raise SystemExit("uncommitted changes in src/ or modal_app.py: commit first")
    key = f"{run}:{segment}"
    if any(r.get("key") == key for r in records()):
        raise SystemExit(f"{key} was launched before; runs are never overwritten")
    commit = launch.git("rev-parse", "HEAD")
    launch.ensure_deployed(commit)
    remote = modal.Function.from_name(launch.APP, "make_data").remote()
    if remote != json.loads(Path("data/addition/manifest.json").read_text()):
        raise SystemExit("cloud addition data differ from local; not launching")
    remote = modal.Function.from_name(launch.APP, "pilot_data").remote()
    if remote != json.loads((FLOOR_DATA_DIR / "manifest.json").read_text()):
        raise SystemExit("cloud floor test set differs from local; not launching")
    call = modal.Function.from_name(launch.APP, function).spawn(run=run, **kwargs)
    PILOT_RESULTS.mkdir(parents=True, exist_ok=True)
    with RECORD.open("a") as f:
        f.write(json.dumps({"key": key, "run": run, "function": function, "args": kwargs, "commit": commit,
                            "call_id": call.object_id,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
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
    """Download each run's top-level summary files (never checkpoints or per-problem logs)."""
    import subprocess

    names = runs or sorted({r["run"] for r in records()})
    for name in names:
        listing = subprocess.run(["uv", "run", "modal", "volume", "ls", "--json", "gainpred", f"runs/pilot/{name}"],
                                 capture_output=True, text=True)
        if listing.returncode:
            continue
        for entry in json.loads(listing.stdout):
            path = entry["filename"]
            if entry["type"] == "file" and path.endswith((".json", ".jsonl")):
                dest = PILOT_RESULTS / "runs" / name / Path(path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", path, str(dest)],
                               capture_output=True, check=True)
        for sub in ("kept/index.json",):
            dest = PILOT_RESULTS / "runs" / name / sub.replace("/", "_")
            subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred",
                            f"runs/pilot/{name}/{sub}", str(dest)], capture_output=True)


def seconds_of(name: str) -> float:
    """Logged seconds of one pilot run folder in results/pilot/runs: wall seconds of finished
    calls, and the logged parts of a call that did not finish."""
    folder = PILOT_RESULTS / "runs" / name
    total = 0.0
    if (folder / "call.json").exists():  # timing, rehearsal and under-trained calls
        return json.loads((folder / "call.json").read_text())["wall_seconds"]
    if not (folder / "rounds.jsonl").exists():
        return 0.0
    pending = 0.0
    for e in map(json.loads, (folder / "rounds.jsonl").read_text().splitlines()):
        if e["event"] in ("round_done", "before_floor"):
            pending += sum(e["seconds"].values())
        if e["event"] == "call_done":
            total += e["wall_seconds"]
            pending = 0.0
    return total + pending


def spending() -> float:
    folders = (PILOT_RESULTS / "runs").glob("*") if (PILOT_RESULTS / "runs").exists() else []
    return sum(seconds_of(p.name) for p in folders) / 3600 * A10_DOLLARS_PER_HOUR


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pilot: building on self-taught pieces.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="write data/floors (the 16-39 digit test set)")
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("runs", nargs="*")
    sub.add_parser("spending")
    sp = sub.add_parser("spawn")
    sp.add_argument("function")
    sp.add_argument("run")
    sp.add_argument("--segment", default="")
    sp.add_argument("--kwargs", default="{}", help="JSON")
    args = parser.parse_args()
    if args.command == "data":
        print(write_floor_data())
    elif args.command == "status":
        for k, v in call_states().items():
            print(f"{k:55s} {v}")
    elif args.command == "fetch":
        fetch(args.runs or None)
    elif args.command == "spending":
        print(f"pilot spending (logged GPU dollars): ${spending():.2f}")
    elif args.command == "spawn":
        spawn(args.function, args.run, json.loads(args.kwargs), args.segment)


if __name__ == "__main__":
    main()
