# Running log for decision point 1

Kept as work proceeds (runbook section 5). Each entry: date, what, evidence.

## Decided by rule since the last report

- 14 Nov. **Authors' setup adopted** (runbook A4): `start-seed0-authors` and `start-seed1-authors` both 99.9% on easy problems (norm_eps 1e-6, 1,000-step decay). Learning batch rerun from them; old-setup runs kept and labelled superseded.
- 14 Nov. **Weak starting model, 60% target: 2,500 steps** (57.7%). One round of the no-carry rule run (`si-nocarry-weak-s2500-seed1`).
- 14 Nov. **Weak starting model, 97.5% target: 3,500 steps** (97.5%), replacing an earlier pick of 2,250 steps (95.9%) once a closer candidate appeared. Both rounds run and reported (`si-nocarry-weak-s3500-seed1`, `si-nocarry-weak-s2250-seed1`).
- 14 Nov. **Weak starting model, 60% target: now 3,400 steps** (61.5%), replacing 2,500 steps (57.7%) once a closer candidate appeared; both rounds run and reported.
- 14 Nov. **Weak starting model, 87.5% target: no candidate within 5 points** after 20 step counts (1,000–8,000); a further six launched (2,200–3,750). If none lands within 5 points, the target stays open for decision point 1 (I will not keep searching indefinitely; this bound is my judgment, not a rule).
- 14 Nov. **Longest-run starting training raised to 30,000 updates** (P3: "updates raised until check 0 passes"): `runs-start-seed1` failed check 0 in one group (run length 5: 3 of 7 right) though 100.0% overall. Both seeds retrained at 30,000 so the recipe stays equal; seed 0's rounds from the 15,000-update model run meanwhile as learning data.
- 14 Nov. **Longest-run starting training: 30,000 updates adopted** (P3 rule): `runs-start-seed0-u30000` and `runs-start-seed1-u30000` pass check 0 in every group (100.0%). Rounds for both seeds rerun from them; seed 0's rounds from the 15,000-update model are kept and labelled superseded.
- 14 Nov. **Weak starting model, 87.5% target: left open** after 26 step counts (none within 5 points; nearest 95.9% and 62.1%).
- 14 Nov. Longest-run readings where the protocol is silent: recorded as P10 assumptions (see `longest_run_readings.md`).

## Confirming runs (from 14-15 Nov)

- Weak models by rule: seed 3: 3,500 steps (59.9%) and 3,450 steps (99.1%); seed 4: 3,000 steps (61.5%) and 2,250 steps (97.6%); seed 2: 2,200 steps (97.3%) for 97.5%, and for 60% nothing within 5 points after 34 step counts (nearest 47.1% at 1,400); one more batch of 8 (1,310-1,490) launched; if none lands, the target stays open for seed 2 (my bound, not the rule). Result: 1,425 steps landed at 62.2% and is seed 2's 60% pick.

- Two runs failed at start (15 Nov, ~00:00): si-nocarry-weak-s1425-seed2 and si-noselect-after-nocarry-seed3-authors could not find their starting run's files because Modal reused a warm container whose view of the volume predated them. Fixed (volume.reload() at the start of round functions, commit a7a2efa); both rerun under -r2 names; the failed runs stay listed.

## Surprises

- **Control arm (14 Nov): familiar answer lengths do not protect joined runs of 7-8.** With long-run strings in starting training the model answers joined runs of 7 at 87-96% and of 8 at 31-60% before any round (0% without them). The withheld round (joined runs up to 6 only) then cuts joined runs of 7 to 34-44% and leaves 8 at 33-50%, while joined runs of 6 go 0-1% -> 94-98%. Random joins also start lower with the long-run strings (86% against 94-97%).

- **Withheld version of experiment 2 (both seeds): no generalisation to longer joined runs.** Join at the cut on the pool without run-across problems of left part 3–4 lifts joined runs of 6 to 95–100% but leaves joined runs of 7–8 at 0% (full pool: 93–99% and 75–98%). Its check-2 diagnostic used the old ±3 cuts (launched minutes before the change).

- **Longest run behaves exactly as designed, on both seeds** (30,000-update models, one round): join at the cut lifts the study set 56% → 95–99% (joined runs of 6: 0 → 97–100%); the edges-differ rule barely moves it (55–59%) and never teaches joined runs of 6+ (check 3: hands back); pick the longer harms (56% → 24–27%; joined runs of 4: 89–94% → 2–3%). Check 1 for the edges-differ rule is low (0.04–0.08) and it did little: a 'does nothing' case for P9b's longest-run analogue.

- **Sample and vote keeps only wrong answers and teaches nothing** (seed 0, four rounds): every kept answer wrong in every round, test accuracy 0% throughout; check 1 was 7% (8 digits) and 15% (9 digits) before round 1, 2b 0%.
- **Longest run, pick the longer harms runs across the join**: joined run length 3: 97.6% → 49.2%, length 4: 96.1% → 1.8% (seed 0, one round); the edges-differ rule lifts random joins 94.8% → 98.6% but never teaches joined runs of 6+.

- **Detached Modal runs died when the laptop was switched off** (13 Nov); fixed by deploying and spawning.
- **Seed 0's 8-digit skill is gone with the new setup**: 0% on 8 digits (old setup: 58%). P9b's case "run 0 at 8 digits against 9" assumed run 0 knows some 8-digit problems; with these starting models it has nothing to compare.
- **Weak starting models: final easy accuracy is erratic in the step count.** Seed 1, authors' setup: 1,500 steps 41.8%, 1,750 2.5%, 2,000 2.1%, 2,125 44.8%, 2,250 95.9%, 2,375 2.1%, 2,500 57.7%, 2,625 99.8%, 2,750 22.2%, 2,875 22.4%, 3,000 99.9%, 3,250 22.3%, 3,500 97.5%, 4,000+ 99.5–100%. Where a run ends seems to depend on whether a loss spike was under way when the decay began. Confirming seeds reuse the step counts "whatever accuracy comes out", so their accuracies may land far from the targets.
- **Check 0 for longest run includes a 7-problem group** (run length 5): one or two errors fail it. The per-run-length pass mark is mostly noise there.
- **The agreement rule teaches carries less well than the no-carry rule**, with both near-100% right training answers: after round 4, carry groups 95–98% (agreement) against 99–100% (no-carry rule).
- **The new setup closes the old 9-digit carry gap**: no-carry rule round 1, 9-digit carry 98–99% (old setup: 70%).

## Decision point 2 (signed 15 November 2026)

Signed on the page; protocol amendments 14-19; runbook cap on weak-grid batches.

**Open item (R1 wording).** The owner's sentence for the write-up says the weak starting models near 60% did nothing and none did harm. The confirming cells say otherwise for three of the six 60%-model cells:

| Run | Digits | Before | After | Best teacher | Outcome |
| --- | --- | --- | --- | --- | --- |
| si-nocarry-weak-s1425-seed2-r2 | 8 | 0.2% | 85.5% | 100% | unclassified (up 85, 14.5 below best) |
| si-nocarry-weak-s1425-seed2-r2 | 9 | 0% | 39.0% | 100% | taught wrong answers (counted with did harm) |
| si-nocarry-weak-s3000-seed4 | 8 | 0.3% | 78.4% | 99.2% | unclassified (up 78, 21 below best) |
| si-nocarry-weak-s3000-seed4 | 9 | 0% | 7.4% | 98.3% | did nothing |
| si-nocarry-weak-s3500-seed3 | 8 | 0% | 8.5% | 99.8% | did nothing |
| si-nocarry-weak-s3500-seed3 | 9 | 0% | 0.6% | 100% | did nothing |

The one matching weak cell (1 of 10) is the taught-wrong-answers cell. Models at 97-99% paid off in all six of their cells. The write-up sentence is left for the owner; the report states the cells as above.
