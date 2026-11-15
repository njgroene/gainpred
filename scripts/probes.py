"""The "train briefly and look" probes (protocol amendment 20): which runs, which rounds, and launching them.

    uv run python scripts/probes.py            # list
    uv run python scripts/probes.py --launch   # spawn every probe not yet launched

Learning runs: the rounds P9b scores (rounds 1-2 of ``-authors``, rounds 3-4 of the five-cut
reruns ``-5cut``, run f, the four seed-1 weak models). Confirming runs: every P9b round of
starting points 2-4 (the ``-r2`` reruns where the original failed) and the five longest-run
rounds per starting point.
"""
import argparse
import subprocess

ARMS = ("nocarry", "agreement", "noselect", "vote")


def plan() -> list[tuple[str, str, str]]:
    """(probe run, original run, rounds) -- rounds '' means all."""
    p = []
    for s in (0, 1):
        for a in ARMS:
            p.append((f"probe-si-{a}-seed{s}-authors", f"si-{a}-seed{s}-authors", "1,2"))
            p.append((f"probe-si-{a}-seed{s}-5cut", f"si-{a}-seed{s}-5cut", ""))
        p.append((f"probe-si-noselect-after-nocarry-seed{s}-authors", f"si-noselect-after-nocarry-seed{s}-authors", ""))
    for n in (2250, 2500, 3400, 3500):
        p.append((f"probe-si-nocarry-weak-s{n}-seed1", f"si-nocarry-weak-s{n}-seed1", ""))
    for s in (2, 3, 4):
        for a in ARMS:
            p.append((f"probe-si-{a}-seed{s}-authors", f"si-{a}-seed{s}-authors", ""))
        f = f"si-noselect-after-nocarry-seed{s}-authors" + ("-r2" if s == 3 else "")
        p.append((f"probe-{f}", f, ""))
    for w in ("s1425-seed2-r2", "s2200-seed2", "s3500-seed3", "s3450-seed3", "s3000-seed4", "s2250-seed4"):
        p.append((f"probe-si-nocarry-weak-{w}", f"si-nocarry-weak-{w}", ""))
    for s in (2, 3, 4):
        for t in ("edges_differ", "pick_the_longer", "join_at_the_cut", "join_at_the_cut-withheld"):
            p.append((f"probe-runs-{t}-seed{s}-u30000", f"runs-{t}-seed{s}-u30000", ""))
        p.append((f"probe-runs-join_at_the_cut-withheld-seed{s}-longruns",
                  f"runs-join_at_the_cut-withheld-seed{s}-longruns", ""))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    for probe, of, rounds in plan():
        print(probe, of, rounds or "all")
        if args.launch:
            subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "probe", probe, "--of", of,
                            *(["--rounds", rounds] if rounds else [])])
