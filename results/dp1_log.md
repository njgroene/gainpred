# Running log for decision point 1

Kept as work proceeds (runbook section 5). Each entry: date, what, evidence.

## Decided by rule since the last report

- 14 Nov. **Authors' setup adopted** (runbook A4): `start-seed0-authors` and `start-seed1-authors` both 99.9% on easy problems (norm_eps 1e-6, 1,000-step decay). Learning batch rerun from them; old-setup runs kept and labelled superseded.
- 14 Nov. **Weak starting model, 60% target: 2,500 steps** (57.7%). One round of the no-carry rule run (`si-nocarry-weak-s2500-seed1`).
- 14 Nov. **Weak starting model, 97.5% target: 3,500 steps** (97.5%), replacing an earlier pick of 2,250 steps (95.9%) once a closer candidate appeared. Both rounds run and reported (`si-nocarry-weak-s3500-seed1`, `si-nocarry-weak-s2250-seed1`).
- 14 Nov. **Weak starting model, 60% target: now 3,400 steps** (61.5%), replacing 2,500 steps (57.7%) once a closer candidate appeared; both rounds run and reported.
- 14 Nov. **Weak starting model, 87.5% target: no candidate within 5 points** after 20 step counts (1,000–8,000); a further six launched (2,200–3,750). If none lands within 5 points, the target stays open for decision point 1 (I will not keep searching indefinitely; this bound is my judgment, not a rule).
- 14 Nov. **Longest-run starting training raised to 30,000 updates** (P3: "updates raised until check 0 passes"): `runs-start-seed1` failed check 0 in one group (run length 5: 3 of 7 right) though 100.0% overall. Both seeds retrained at 30,000 so the recipe stays equal; seed 0's rounds from the 15,000-update model run meanwhile as learning data.
- 14 Nov. Longest-run readings where the protocol is silent: recorded as P10 assumptions (see `longest_run_readings.md`).

## Surprises

- **Sample and vote keeps only wrong answers and teaches nothing** (seed 0, four rounds): every kept answer wrong in every round, test accuracy 0% throughout; check 1 was 7% (8 digits) and 15% (9 digits) before round 1, 2b 0%.
- **Longest run, pick the longer harms runs across the join**: joined run length 3: 97.6% → 49.2%, length 4: 96.1% → 1.8% (seed 0, one round); the edges-differ rule lifts random joins 94.8% → 98.6% but never teaches joined runs of 6+.

- **Detached Modal runs died when the laptop was switched off** (13 Nov); fixed by deploying and spawning.
- **Seed 0's 8-digit skill is gone with the new setup**: 0% on 8 digits (old setup: 58%). P9b's case "run 0 at 8 digits against 9" assumed run 0 knows some 8-digit problems; with these starting models it has nothing to compare.
- **Weak starting models: final easy accuracy is erratic in the step count.** Seed 1, authors' setup: 1,500 steps 41.8%, 1,750 2.5%, 2,000 2.1%, 2,125 44.8%, 2,250 95.9%, 2,375 2.1%, 2,500 57.7%, 2,625 99.8%, 2,750 22.2%, 2,875 22.4%, 3,000 99.9%, 3,250 22.3%, 3,500 97.5%, 4,000+ 99.5–100%. Where a run ends seems to depend on whether a loss spike was under way when the decay began. Confirming seeds reuse the step counts "whatever accuracy comes out", so their accuracies may land far from the targets.
- **Check 0 for longest run includes a 7-problem group** (run length 5): one or two errors fail it. The per-run-length pass mark is mostly noise there.
- **The agreement rule teaches carries less well than the no-carry rule**, with both near-100% right training answers: after round 4, carry groups 95–98% (agreement) against 99–100% (no-carry rule).
- **The new setup closes the old 9-digit carry gap**: no-carry rule round 1, 9-digit carry 98–99% (old setup: 70%).
