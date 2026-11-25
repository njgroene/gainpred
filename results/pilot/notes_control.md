### Decisions I took (amendment 27: the control, the analyses and the paper files)

The numbers in the tables above and in `results/pilot/analysis.md`, `results/pilot/macros.md` and
`paper/generated/*climb*` are generated from the logs; none is typed by hand. No definition,
threshold or number of the task was changed.

1. **Order.** Amendment 27 was committed and pushed (38a2082) before any code of the control was
   written; the control's code, its tests and its cost estimate followed (7afa892), then the launch.
2. **Where the teacher lives.** The direct-answers teacher is a fourth teacher in the pilot's own module
   (`gainpred.floors`: `decide`, `collect`, `teacher_splits`); none of `cuts.py`, `teachers.py`,
   `rounds.py`, `data.py`, `train.py`, `p9b.py` or `writeup.py` was changed. The three existing teachers'
   code paths are unchanged (their tests pass as before).
3. **Checks of the control.** Before each round the control logs the agreement rule's checks at the two
   new lengths (novelty, kept share, confirmed share), with `audit_direct_right`, computed by the same
   code as in the pilot; they decide nothing. The floor-1 novelty checks of every pilot round are also
   logged, as in the pilot. The "seen coming" numbers are not computed (the control's floor 1 is the
   solid one, whose numbers the agreement runs already computed).
4. **Cap.** The task's $18 is counted for this task only, as the logged wall seconds of the control's
   calls. The estimate before launch ($2.44 for floor 2) used the pilot's cost model with one whole answer
   per problem drawn (keep rate assumed 0.95).
5. **Stop after floor 2.** Floor 2 was not learned, so by the task's rule floors 3 and 4 were not run.
6. **Analyses (part 3).**
   * *Basics check* (C): P9b's check 0, i.e. every easy length 3-7 at least 98% on the test set before
     the round (the floor's eval after the previous round; round 0 for round 1).
   * *Outcome* (C): exactly the thresholds of the task. P9b's extra clause (an easy-length fall of more
     than 3 points counts as harm) is not applied. For "before 50% or lower", the tests are applied in
     the order the task lists them (paid off, did nothing, taught wrong answers); "did nothing" includes
     a fall.
   * *Best* (C): the highest "after" at that length over every run of the 24-25 November pilot with the
     same repeat number (solid agreement, solid no-carry, corrupted agreement, corrupted no-carry, and the
     answer key for repeats 0 and 1). The control is a separate amendment and is not among them.
   * *Novelty in A* is the value logged in the round's checks event; "first round" is in teaching order
     across floors 2-4.
   * *Error after* (B) is 100 minus the test-set accuracy (mean of the two carry groups) after the round
     that first taught the length.
7. **Paper files (part 4).**
   * *Text width.* The task forbids opening anything under `paper/` other than the generated files, so
     the text width was not read from the paper. The widest existing generated table that is set at
     \small without scaling measures 380 pt (`table_withheld_slim.tex`, `table_rehearsal.tex`); every new
     table was measured with tectonic and is at most that wide (widest: 376 pt). All new files were
     compiled together (with `\input` of `numbers_pilot.tex`) without an overfull or underfull box.
   * *"Before" in `table_climb_control.tex`* is the share of the model's own direct answers that were
     right before the round, on 2,000 fresh problems (`audit_direct_right`): the test set is not scored
     at a length before the round that teaches it.
   * *Confirmed share in `table_climb_weak.tex`* for the no-carry rows is the one computed in the
     agreement run of the same version of floor 1 (same model, computed once; pilot decision 2).
   * *Lowest kept right* in `table_climb.tex` is the lowest share over the floor's built sets and repeats
     (for the answer key it is read from the logs too).
   * *Macros*: one per number in the tables, the figure (the mean bar at each length and the caption's
     range) and the control's result; macro names spell digits as words because LaTeX names hold letters
     only. `results/pilot/macros.md` lists each.
   * The figure prints each mean of 4% or more rounded to a whole percent, directly on top of its bar.

### Anything that looked wrong (control)

1. **Nothing went wrong in the runs**: no crash, no shortfall; almost every direct answer was readable,
   so the control drew barely more problems than it kept. Examples and updates per round equal the
   agreement runs' exactly.
2. **The control learned nothing new.** At 17 digits it trained on its own answers, of which only about
   a fifth were right, and stayed near where it started; from 18 digits on it started at about 0%, its
   kept answers were almost all wrong, and it stayed there. It did not get worse at the lengths it
   already knew (16 digits and floor 1). This is what C1 predicted.
3. **The agreement-rule model could already answer the even lengths before training them** (18, 20, 22
   at about 99% before their round; see `table_climb_control.tex`), while the control's model could not.
   So in the pilot the lengths actually taught by the splits are mostly the odd ones (17, 19, 21, 23),
   where the error before the round was high (`figure_climb_errors`). This was read after the runs.
4. **Four misses of the frozen rule**, all with corrupted pieces and the no-carry rule: the confirmed
   share was below the frozen 2b level, so the verdict was "does harm", yet the round paid off (the model
   learned despite the random wrong answers; see the pilot's anomaly 2).
