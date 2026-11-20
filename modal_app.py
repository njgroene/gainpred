"""Run gainpred on Modal's cloud GPUs.

Usage (from the repo root)::

    uv run modal run modal_app.py --run start-seed0 --seed 0     # full P1 starting training
    uv run modal run modal_app.py --run smoke --steps 200 --eval-every 100 --eval-per-length 100

    uv run modal run --detach modal_app.py --run start-seed0     # keeps running if you close the terminal
    uv run modal volume get gainpred runs/start-seed0 runs/      # download a finished run

    # self-improvement rounds (P3), continuing from a starting run on the volume:
    uv run modal run --detach modal_app.py::rounds --run si-nocarry-seed1 --start start-seed1-stable \
        --teacher no_carry --rounds 1

How Modal works, briefly
========================
You write ordinary Python functions and decorate them with ``@app.function``.
``modal run`` packages the code, starts a container in Modal's cloud with the
requested GPU, runs the function there, streams its prints back to your
terminal, and shuts the container down. You pay per second while it runs.

Three pieces are declared below:

* **Image** -- the software inside the container: Python 3.12, torch, numpy,
  and this repo's ``gainpred`` package (copied from your machine on each run,
  so local code changes are picked up automatically).
* **Volume** -- persistent cloud storage, mounted at ``/vol`` in the
  container. It outlives the container, so datasets and runs survive between
  calls. Layout: ``/vol/data/addition/`` and ``/vol/runs/<run>/``.
* **Functions** -- ``make_data`` (CPU only), ``train_start`` and
  ``self_improve`` (on a GPU); for longest run, ``make_run_data``,
  ``train_start_runs`` and ``self_improve_runs`` (data in
  ``/vol/data/longest_run/``; launched with ``gainpred.launch runs-main`` and
  ``runs-rounds``). ``main`` and ``rounds`` are *local
  entrypoints*: they run on your machine and call the remote functions with
  ``.remote(...)``. ``modal run modal_app.py`` runs ``main``;
  ``modal run modal_app.py::rounds`` runs ``rounds``.

Why the data is generated in the cloud, not uploaded
====================================================
The datasets are fully determined by their seeds (see gainpred.data), so the
container regenerates them instead of uploading 16 MB. ``main`` then checks
the cloud manifest against your local ``data/addition/manifest.json``: if the
SHA-256 fingerprints differ, it stops before training on different data.

GPU choice
==========
An NVIDIA A10G: a 14M-parameter model leaves bigger GPUs mostly idle, and
the A10G is cheap and usually available. (We first chose the L4, but Modal
had no L4 capacity on 13 November 2026, and no run had used it yet.) All
confirming runs must use the same GPU type (P9), which is why it is fixed
here rather than a command-line option.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

GPU = "A10G"
VOLUME_PATH = Path("/vol")

app = modal.App("gainpred")

# Versions match uv.lock, so cloud and laptop run the same libraries.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.15.0", "numpy==2.5.4")
    .add_local_python_source("gainpred")
)

volume = modal.Volume.from_name("gainpred", create_if_missing=True)


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, timeout=15 * 60)
def make_data() -> dict:
    """Generate the addition datasets on the volume (once) and return the manifest."""
    from gainpred.data import write_all

    from gainpred.data import SEEDS

    out_dir = VOLUME_PATH / "data" / "addition"
    manifest_path = out_dir / "manifest.json"
    # Regenerate when a set was added since the volume's copy was made (e.g. test_study_5cut).
    if manifest_path.exists() and set(json.loads(manifest_path.read_text())) == set(SEEDS):
        return json.loads(manifest_path.read_text())
    manifest = write_all(out_dir)
    volume.commit()  # make the files visible to later containers
    return manifest


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=4 * 60 * 60)
def train_start(run: str, seed: int, steps: int, eval_every: int, eval_per_length: int, checkpoint_every: int,
                norm_eps: float = 0.0, decay_fraction: float = 0.2) -> None:
    """P1 starting training on one GPU. Output goes to /vol/runs/<run>/."""
    import torch

    from gainpred.train import TrainConfig, starting_training

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    cfg = TrainConfig(
        steps=steps,
        seed=seed,
        eval_every=eval_every,
        eval_per_length=eval_per_length or None,  # 0 means "full test sets"
        checkpoint_every=checkpoint_every,
        decay_fraction=decay_fraction,
    )
    try:
        starting_training(
            cfg,
            run,
            norm_eps=norm_eps or None,  # 0 means the default of ModelConfig
            data_dir=VOLUME_PATH / "data" / "addition",
            device=torch.device("cuda"),
            runs_dir=VOLUME_PATH / "runs",
        )
    finally:
        volume.commit()  # save logs and checkpoints even if training crashed


@app.local_entrypoint()
def main(
    run: str,
    seed: int = 0,
    steps: int = 10_000,
    eval_every: int = 1_000,
    eval_per_length: int = 0,
    checkpoint_every: int = 2_000,
    norm_eps: float = 0.0,
    decay_fraction: float = 0.2,
) -> None:
    """Check the cloud data matches local data, then train. Runs on your machine."""
    local_manifest = Path("data/addition/manifest.json")
    if not local_manifest.exists():
        raise SystemExit("run `uv run python -m gainpred.data` first, so there is a local manifest to compare with")
    remote = make_data.remote()
    if remote != json.loads(local_manifest.read_text()):
        raise SystemExit("cloud datasets differ from local ones (fingerprints in manifest.json); not training")
    print("datasets match local fingerprints")
    train_start.remote(run, seed, steps, eval_every, eval_per_length, checkpoint_every, norm_eps, decay_fraction)
    print(f"done. download with: uv run modal volume get gainpred runs/{run} runs/")


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=12 * 60 * 60)
def self_improve(run: str, start: str, teacher: str, rounds: int, start_round: int, build_seconds: float) -> None:
    """P3 self-improvement rounds on one GPU. Output goes to /vol/runs/<run>/."""
    # A warm (reused) container keeps the volume as it was when the container started;
    # reload so runs that finished since then (our starting point) are visible.
    volume.reload()
    import torch

    from gainpred.rounds import RoundConfig, self_improve, start_seed

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    runs_dir = VOLUME_PATH / "runs"
    from_round = start_round or None  # 0 means "from a starting-training run"
    # Bigger generation batches than the default: a GPU answers 4,096 short prompts at once easily.
    cfg = RoundConfig(teacher=teacher, rounds=rounds, seed=start_seed(runs_dir, start, from_round),
                      solver_batch_size=4_096, build_seconds_per_length=build_seconds or None)
    try:
        self_improve(cfg, run, start, data_dir=VOLUME_PATH / "data" / "addition",
                     device=torch.device("cuda"), runs_dir=runs_dir, start_round=from_round)
    finally:
        volume.commit()


@app.local_entrypoint()
def rounds(run: str, start: str, teacher: str = "no_carry", rounds: int = 4, start_round: int = 0,
           build_seconds: float = 0) -> None:
    """Self-improvement rounds from a run already on the volume.

    ``--start-round K`` continues from round K of the self-improvement run
    ``--start`` (P9b run f); ``--build-seconds S`` stops drawing for a length
    after S seconds and logs the shortfall (0 = no limit).
    """
    remote = make_data.remote()
    if remote != json.loads(Path("data/addition/manifest.json").read_text()):
        raise SystemExit("cloud datasets differ from local ones (fingerprints in manifest.json); not training")
    print("datasets match local fingerprints")  # gainpred.launch waits for this line
    self_improve.remote(run, start, teacher, rounds, start_round, build_seconds)
    print(f"done. download with: uv run modal volume get gainpred runs/{run} runs/")


# ----------------------------------------------------------------------------
# Longest run (experiment 2): data, starting training, one round.
# Launched with ``uv run python -m gainpred.launch runs-main ...`` / ``runs-rounds ...``.
# ----------------------------------------------------------------------------


@app.function(image=image, volumes={str(VOLUME_PATH): volume}, timeout=15 * 60)
def make_run_data() -> dict:
    """Generate the longest-run datasets on the volume (once) and return the manifest."""
    from gainpred.run_data import write_all

    from gainpred.run_data import SEEDS

    out_dir = VOLUME_PATH / "data" / "longest_run"
    manifest_path = out_dir / "manifest.json"
    # Regenerate when a set was added since the volume's copy (e.g. train_start_longruns).
    if manifest_path.exists() and set(json.loads(manifest_path.read_text())) == set(SEEDS):
        return json.loads(manifest_path.read_text())
    manifest = write_all(out_dir)
    volume.commit()
    return manifest


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=6 * 60 * 60)
def train_start_runs(run: str, seed: int, steps: int = 15_000, eval_every: int = 1_000, checkpoint_every: int = 3_000,
                     norm_eps: float = 1e-6, decay_fraction: float = 0.1, long_runs: bool = False) -> None:
    """Longest-run starting training (P1) on one GPU. Output goes to /vol/runs/<run>/."""
    import torch

    from gainpred.run_train import starting_training_runs
    from gainpred.train import TrainConfig

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    make_run_data.local()  # data on the volume, generated here if missing
    cfg = TrainConfig(steps=steps, seed=seed, eval_every=eval_every, checkpoint_every=checkpoint_every,
                      decay_fraction=decay_fraction)
    try:
        starting_training_runs(cfg, run, data_dir=VOLUME_PATH / "data" / "longest_run", device=torch.device("cuda"),
                               runs_dir=VOLUME_PATH / "runs", norm_eps=norm_eps or None, long_runs=long_runs)
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=12 * 60 * 60)
def self_improve_runs(run: str, start: str, teacher: str, build_seconds: float = 0.0, withheld: bool = False,
                      long_runs: bool = False) -> None:
    """One longest-run round (P3) on one GPU, from a starting run on the volume."""
    # A warm (reused) container keeps the volume as it was when the container started;
    # reload so runs that finished since then (our starting point) are visible.
    volume.reload()
    import torch

    from gainpred.rounds import start_seed
    from gainpred.run_train import RunRoundConfig, self_improve_runs as run_rounds

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    make_run_data.local()
    runs_dir = VOLUME_PATH / "runs"
    cfg = RunRoundConfig(teacher=teacher, seed=start_seed(runs_dir, start), solver_batch_size=4_096,
                         build_seconds_per_length=build_seconds or None, withheld=withheld,
                         long_runs=long_runs)
    try:
        run_rounds(cfg, run, start, data_dir=VOLUME_PATH / "data" / "longest_run", device=torch.device("cuda"),
                   runs_dir=runs_dir)
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=12 * 60 * 60)
def probe(run: str, of: str, only_rounds: list[int] | None = None) -> None:
    """"Train briefly and look" baseline: rebuild each round of run ``of``, train 200 updates, evaluate."""
    volume.reload()  # see self_improve
    import torch

    from gainpred.probe import probe_addition, probe_runs

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    runs_dir = VOLUME_PATH / "runs"
    try:
        if of.startswith("runs-"):
            make_run_data.local()
            probe_runs(of, run, runs_dir, VOLUME_PATH / "data" / "longest_run", torch.device("cuda"))
        else:
            probe_addition(of, run, runs_dir, VOLUME_PATH / "data" / "addition", torch.device("cuda"), only_rounds)
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=2 * 60 * 60)
def check3(run: str, of: str, three_cuts: bool = False, only: list[list[int]] | None = None) -> None:
    """Check 3 on addition (amendment 22): redraw each round's check sample of run ``of`` with the
    model before that round, record the teacher's piece answers, compute check 3. Inference only."""
    volume.reload()  # see self_improve
    import torch

    from gainpred.check3 import redraw

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    try:
        redraw(of, run, VOLUME_PATH / "runs", VOLUME_PATH / "data" / "addition", torch.device("cuda"),
               three_cuts=three_cuts, only=[tuple(c) for c in only] if only else None)
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=60 * 60)
def reproduction(run: str) -> None:
    """Reproduction table (appendix): score the evaluation-run checkpoints of the no-carry rule
    and no selection on test_reproduction (3-15 digits, 100 each). Inference only."""
    volume.reload()  # see self_improve
    import torch

    from gainpred.reproduction import score

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    try:
        score(run, VOLUME_PATH / "runs", VOLUME_PATH / "data" / "addition", torch.device("cuda"))
    finally:
        volume.commit()


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=60 * 60)
def control_answers(run: str) -> None:
    """Protocol amendment 23: the 20 longest-run models (control arm before and after the withheld
    round, withheld version, edges-differ rule; starting models 0-4) answer the study test set.
    Inference only."""
    volume.reload()  # see self_improve
    import torch

    from gainpred.control_answers import answer_all

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    make_run_data.local()
    try:
        answer_all(run, VOLUME_PATH / "runs", VOLUME_PATH / "data" / "longest_run", torch.device("cuda"))
    finally:
        volume.commit()
