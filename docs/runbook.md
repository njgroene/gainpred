# Runbook: running the study with little intervention

Agreed with the project owner on 14 November 2026. Any session working on this repo (human or Claude) follows it. It sits beside `docs/protocol.md`: the protocol says *what* is measured; this says *how the work proceeds* and *who decides what*.

## 1. Standing permissions

Within these, work proceeds without asking.

- **Launch any run on the work list** (section 4), within the budget.
- **Commit and push to main**: code, tests, results, and protocol edits of the kinds allowed in section 2.
- **Fix bugs that do not change what is measured** (crashes, launch failures, reporting errors), add a test, relaunch once.

**Budget.** $80 for Modal in total (raised from $50 to $60 and then to $80 on 15 November 2026, the second time for the baseline probes and their reruns; no separate notice level). Stop and ask before exceeding it. Spending is tracked in the status section of `results/README.md`.

## 2. Hard lines: never without the owner

- **No confirming runs** (starting points 2–4) before the owner signs decision point 1.
- **No change to what is measured**: cases, check definitions, verdict rule, outcome thresholds, cuts, quotas, test sets. If one looks wrong, write it up for the next decision point; do not fix it.
- **Never delete or overwrite a run.** A rerun gets a new name, and both are reported. Every run appears in the report, including failed and parked ones.
- **Protocol edits are limited to**: recording outcomes; recording values chosen by an agreed rule; P10 assumptions. **Every P10 assumption added goes on the list for the next decision point.**

## 3. Rules for decisions that would otherwise wait

| Situation | Rule |
| --- | --- |
| Weak starting models (P9b) | Pick the step counts whose easy accuracy is closest to 60%, 87.5% and 97.5%; ties go to fewer steps. If no candidate is within 5 points of a target, train more step counts, at most two extra batches (decision point 2); after that the target stays open. Never pick a distant one. Record the step counts in P9b |
| A run crashes or hits a Modal limit | Diagnose. A bug that does not change what is measured: fix, test, relaunch once under a new name. A second failure: park it, report it, continue with other work |
| A result looks wrong | Investigate and write it up. Do not change the design. It goes to the next decision point |
| The protocol is silent (any task) | Take the most literal reading, record it as a P10 assumption, and list it for the next decision point |
| A teacher hits the 20x cap or a time limit | Log the shortfall; it is a result, not a failure |
| The freeze | Tag the frozen commit `stage1-freeze`. Every confirming run records the commit it ran from, which must be that tag or a descendant that changes no measured behaviour |

## 4. Tripwires: stop that line of work and notify the owner

- The no-carry rule falls below 90% at any length in a four-round run.
- A standard starting model fails check 0 (under 99% on easy problems).
- Any overlap between test and training problems.
- A rerun differs from its original by more than 3 points on any group.
- Anything suggesting the checks were computed after training, not before.

Other lines of work continue while a tripwire is open.

## 5. Work list

### A. Before decision point 1 (no owner decisions needed)

1. Download the batch launched on 13 November (runs a–f and `start-seed1-s*`), rebuild the reports, and report back on the owner's list (see `results/README.md`).
2. Weak starting models: pick by the rule in section 3, record in P9b, run one round of the no-carry rule from each.
3. Seed 0's missing arms: the agreement rule, no selection, and run f.
4. **The authors' setup**: `norm_eps` 1e-6 and a 1,000-step decay in starting training. Retrain seeds 0 and 1. If both pass check 0 (at least 99% on easy problems), adopt it and rerun the learning batch from the new starting models; if not, keep the 2,000-step decay and record the departure. Report which happened.
5. Longest run: finish (review the partial agent work on branch `worktree-agent-ad8c7c459185755f2`), test, paper test, merge. **Before its learning runs, send the owner the paper test and the list of readings taken where the protocol was silent.** Learning runs may go ahead without waiting for the answer; nothing there is frozen until decision point 1.
6. Longest-run learning runs: starting training for seeds 0 and 1, one round with each teacher.
7. P9b scoring code: verdict against outcome per case, run, round and length, using the fixed scoring rules; run it on the learning runs.

### Decision point 1 (freeze stage 1)

A page the owner can comment on. For **every** item: the raw table, a recommendation, and what to do if the owner disagrees. Items:

- the cuts question for 13–15 digits (three cuts, or four to five);
- the 2b and check-1 levels;
- the outcome thresholds at new lengths (within 10 points of the best = paid off; 30 or more below = taught wrong answers), with the data that bears on them;
- longest run: the open questions and readings taken;
- every P10 assumption added since the last report;
- the freeze itself.

Plus two lists: **what was decided by rule** since the last report, and **what was surprising**.

### B. After decision point 1

Confirming runs for starting points 2–4: starting training, every addition arm (five cuts at 13–15), the weak starting models (matched on target accuracy, 60% and 97.5%), run f, sample and vote, longest run (three teachers, plus the withheld version of join at the cut, plus its control arm: long-run starting training, then the same withheld round). Score P9b.

### Decision point 2

Sign-off on the P9b scoring outcome.

### C. Write-up

Results tables, plots, and how P9b turned out.

## 6. Mechanics

- **Launching**: `uv run python -m gainpred.launch main|rounds NAME ...` deploys the app (when the commit changed) and *spawns* the run on Modal, so it survives this laptop being off; `uv run python -m gainpred.launch status` shows every spawned run as running, done or failed. Never use `modal run --detach` for real runs: on 13 November such runs died when the laptop was switched off.
- **Status**: a generated section at the top of `results/README.md`: runs in flight, done, failed or parked; spending; open tripwires; decisions waiting.
- **Decision documents**: a page the owner can comment on.
- **Pacing**: runs are checked every 20–30 minutes while a session is open; finished runs are downloaded and reported, and newly unblocked runs are launched. Modal runs continue when the laptop is off; the session's own work pauses until it is reopened, and resumes from this runbook and the status section.
