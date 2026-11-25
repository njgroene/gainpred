### Decisions I took (amendment 28: the control one floor up, its paper files, and the macros for hand-typed numbers)

Every number in the tables above, in `results/pilot/macros.md` and in the new macros of
`paper/generated/numbers_pilot.tex` and `numbers.tex` is generated from the logs; none is typed by hand.
No definition, threshold or number of the task was changed.

1. **Order.** Amendment 28 was committed and pushed (19fc186) before any code of this control; the code
   and its tests followed (2bfd137), then the launch.
2. **Starting model.** Each run was launched with `start = "floor:solid-agr-rN"`, which the existing
   launcher resolves to the last checkpoint in `/vol/runs/pilot/solid-agr-rN/floor2/round_4/checkpoints/`.
   The `start_checkpoint` recorded in each run's `config-floor3-rounds1-4.json` is the one that
   `solid-agr-rN` recorded for floor 3 (`step_005100.pt`; for N = 1 the one of `solid-agr-r1`).
3. **Copying the floor-2 answers.** A new option, `copy_lower_from`, copies the eight files and their
   index entries byte for byte before round 1. Each copy's sha256 is checked against `solid-agr-rN`'s
   index before and after copying. The index entries keep the floor and round they had in `solid-agr-rN`.
   `base_run` is None in all three runs. Nothing is written to the folder of `solid-agr-rN`, and a test
   checks that the source folder is unchanged.
4. **Check before training.**
   * The expected values were read from `solid-agr-rN`'s local logs and passed to the call. They are
     recorded in `launches.jsonl` and in the run's config.
   * The run compares them with its own `audit_direct_right` at 24 and 25 after the round-1 checks,
     and before building anything. A difference of more than 0.2 points stops the run.
   * As in `solid-agr-rN`, the floor-2 checks before floor 3 run first, so the fresh problems, and the
     problems they exclude, are the same.
   * All six values matched exactly.
5. **Cap.** The $12 counts the logged wall seconds of every `direct3-*` run. The estimate before launch
   was $4.08 for floor 3, from the pilot's cost model with one whole answer per problem. Floor 3 was not
   learned, so floor 4 was not run.
6. **Paper table.** `table_climb_control3.tex` keeps the columns of `table_climb_control.tex` and adds one
   for the control: "Round 4", the test accuracy after round 4 of the floor (part 3 item 2 asks for that
   number). It is 362 pt wide at \small, unscaled, against 380 pt for the widest existing unscaled table.
   It compiles with both numbers files without an overfull or underfull box. No macros for floor 4 were
   made, because it was not run.
7. **"Largest drop at any shorter length"** is computed as in the pilot: from the floor's round-0 test to
   after round 4, at every length from 3 to 23. It is 0.05 points in each repeat, which prints as 0.0.
8. **Part 4.** None of the five numbers had a macro, so all were added. Earlier macros are unchanged and
   the new lines are added only. Readings:
   * (a) `\gpClimbGainGapRight`: the cases of analysis C whose outcome is "unclassified", in the three
     named conditions. Novelty is the logged share (in percent); gain = after minus before, where
     before is `audit_direct_right`.
   * (b) `\gpClimbRehearsalRows` and `\gpClimbRehearsalWrong`: from the rehearsal set's `meta.json`,
     whose audit compared each answer with the true sum.
   * (c) `\gpClimbUpdatesLow` and `\gpClimbUpdatesHigh`: every `round_done` of floors 2-4 in the three
     agreement repeats (36 rounds).
   * (d) `\gpDidNothingPct`: the 258 cases of `\gpCases`; marked "% after".
   * (e) `\gpLrEdgesRandomJoinLow` and `\gpLrEdgesRandomJoinHigh`: the test group `random_join`, after the
     edges-differ rule's one round, over the starting models of the evaluation runs (2, 3, 4), as for
     the other `\gpLr...` macros; marked "% after".

### Anything that looked wrong (control one floor up)

1. **Nothing went wrong in the runs.**
   * There was no crash and no shortfall, and every check before training matched exactly.
   * Examples and updates per round equal `solid-agr-rN`'s on floor 3 (5,700, 6,200, 6,700, 7,300
     updates).
2. **The control kept what it already knew and copied its own errors.**
   * At each new length, the share of kept answers that were right was about the share of right direct
     answers before the round. Accuracy after the round stayed close to that share.
   * Lengths the model already answered (24-26) stayed at 96% or more.
   * Beyond about 27 digits, accuracy fell as length grew in every repeat, down to 0.1%, 82.8% and 39.2% at
     31 digits.
   * No shorter length got worse.
3. **The three repeats differ a lot.** Each repeat starts from its own agreement run's model, and those
   models answered 27-31 digits differently before floor 3. For example, at 27 digits 75.9% of repeat
   0's kept answers were right, against 99.6% in repeat 1. Repeat 1 came closest (82.8% at 31 digits),
   but none of the three reached 90% at every length.
