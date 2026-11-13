"""Run gainpred on Modal's cloud GPUs.

Usage (from the repo root)::

    uv run modal run modal_app.py --run start-seed0 --seed 0     # full P1 starting training
    uv run modal run modal_app.py --run smoke --steps 200 --eval-every 100 --eval-per-length 100

    uv run modal run --detach modal_app.py --run start-seed0     # keeps running if you close the terminal
    uv run modal volume get gainpred runs/start-seed0 runs/      # download a finished run

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
* **Functions** -- ``make_data`` (CPU only) and ``train_start`` (on a GPU).
  ``main`` is the *local entrypoint*: it runs on your machine and calls the
  remote functions with ``.remote(...)``.

Why the data is generated in the cloud, not uploaded
====================================================
The datasets are fully determined by their seeds (see gainpred.data), so the
container regenerates them instead of uploading 16 MB. ``main`` then checks
the cloud manifest against your local ``data/addition/manifest.json``: if the
SHA-256 fingerprints differ, it stops before training on different data.

GPU choice
==========
An NVIDIA L4: a 14M-parameter model leaves bigger GPUs mostly idle, and the
L4 is among the cheapest. All confirming runs must use the same GPU type
(P9), which is why it is fixed here rather than a command-line option.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

GPU = "L4"
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

    out_dir = VOLUME_PATH / "data" / "addition"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    manifest = write_all(out_dir)
    volume.commit()  # make the files visible to later containers
    return manifest


@app.function(image=image, gpu=GPU, volumes={str(VOLUME_PATH): volume}, timeout=4 * 60 * 60)
def train_start(run: str, seed: int, steps: int, eval_every: int, eval_per_length: int, checkpoint_every: int) -> None:
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
    )
    try:
        starting_training(
            cfg,
            run,
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
) -> None:
    """Check the cloud data matches local data, then train. Runs on your machine."""
    local_manifest = Path("data/addition/manifest.json")
    if not local_manifest.exists():
        raise SystemExit("run `uv run python -m gainpred.data` first, so there is a local manifest to compare with")
    remote = make_data.remote()
    if remote != json.loads(local_manifest.read_text()):
        raise SystemExit("cloud datasets differ from local ones (fingerprints in manifest.json); not training")
    print("datasets match local fingerprints")
    train_start.remote(run, seed, steps, eval_every, eval_per_length, checkpoint_every)
    print(f"done. download with: uv run modal volume get gainpred runs/{run} runs/")
