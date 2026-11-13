## Appendix: Protocol

The working rules for building and running the experiments. Draft of 13 November 2026. It is frozen stage by stage, before that stage's confirming runs; after that, changes go only in the amendments log at the end, with a date and a reason. Items marked *to decide* are settled on the learning runs.

### P1. Settings taken from Shin et al.

From their Section A.1, A.2 and Table 4, re-read on 12 November.

| Setting | Value |
| --- | --- |
| Model | Character-level Transformer, 6 layers, hidden size 384, 6 attention heads, no positional encoding, about 14.2M parameters |
| Optimizer | AdamW, β1 0.9, β2 0.99, ε 1e-12, weight decay 0.1 |
| Learning rate, batch size | 5e-4 at the peak (schedule below), 256 |
| Precision, decoding | bfloat16; greedy decoding for all built answers |
| Addition, starting training | Both numbers have the same length, 3–7 digits, with no leading zero; 50,000 problems per length, 10,000 updates. Equal length holds at every size in this study |
| Addition, each round | 10,000 new built problems per length from 8 up to the current maximum, plus 5,000 new easy problems per easy length answered by the current model; 3,000 updates |
| Longest run, starting training | Strings of length 6–10, 50,000 per length, 15,000 updates |
| Longest run, each round | 2,000 new built strings per length from 12 up to the current maximum, plus the starting strings replayed with true answers; 3,000 updates |

**Not stated in their paper; our assumptions.** Learning-rate schedule and gradient clipping, taken from the public code of Lee et al. (2025), from which their setup derives (same architecture, same AdamW ε): warmup-stable-decay with 10% warm-up and the last 20% of steps decaying along a cosine to 1% of the peak, and gradient clipping at norm 1.0. Loss computed on answer characters only. All are checked against their code if it arrives, and listed under P10.

### P2. Formats

| Item | Rule |
| --- | --- |
| Addition question and answer | `Q: 6133 + 8894 = ?` then `A: 15027`. Digits are written forward, most significant first; their Section C.2 confirms “forward addition, as in our setting” |
| Built addition answer | The piece sums written side by side. The leading sum may be one digit longer than its piece; every lower sum is padded with zeros to its piece's width |
| Longest-run question and answer | `Q: runlen(493036) = ?` then `A: 4\|1` (symbol, then length). The leftmost run wins ties: here every run has length 1, so the answer is the first symbol. A second example: `Q: runlen(887476) = ?` then `A: 8\|2` |
| Edge-run questions | `Q: leftrun(493036) = ?` and `Q: rightrun(493036) = ?`, answered in the same symbol-and-length form |
| No piece starts with 0 | Addition problems are drawn so that, under each of the three cuts, no piece of either number begins with 0. Otherwise a piece would look unlike any easy problem |
| Unreadable answers | An answer that cannot be parsed counts as wrong when scoring, and as its own distinct answer when comparing answers. It is never dropped |

### P3. Sizes and rounds in this study

| Task | Rounds | Hard sizes | Kept answers per size, per round |
| --- | --- | --- | --- |
| Addition | 4 | Maximum 9, then 11, 13, 15 digits; every length from 8 to the maximum is rebuilt each round | 10,000 |
| Longest run | 1 | Lengths 12–20, each string made of two pieces of length 6–10 | 2,000 |

Every version of an experiment trains on the same number of kept answers. Candidates are drawn until the quota is met, capped at twenty times the quota; the number drawn and any shortfall are logged. The paper test puts the need at about three candidates per kept answer up to 12 digits and about fourteen at 15 digits for the agreement rule.

*To decide:* how the two edge-run questions enter the longest-run starting training (default: the same 50,000 strings per length, each asked all three questions, with updates raised until check 0 passes).

### P4. Cuts

| Size | Cut 1 | Cut 2 | Cut 3 |
| --- | --- | --- | --- |
| 8 digits | 4∣4 | 3∣5 | 5∣3 |
| 9 digits | 3∣6 | 4∣5 | 5∣4 |
| 10 digits | 5∣5 | 4∣6 | 6∣4 |
| 11 digits | 4∣7 | 5∣6 | 6∣5 |
| 12 digits | 6∣6 | 5∣7 | 7∣5 |
| 13 digits | 6∣7 | 7∣6 | 4∣5∣4 |
| 14 digits | 7∣7 | 4∣5∣5 | 6∣4∣4 |
| 15 digits | 5∣5∣5 | 4∣7∣4 | 6∣3∣6 |

Two rules govern the table. Pieces are always 3–7 digits, the easy sizes. And no two cuts of a problem may place a cut at the same position: when they do, a carry at that position makes both wrong in the same way, and agreement stops being evidence. Every row was checked in simulation on 12 November. Cut 1 supplies the answer for “the no-carry rule” and “no selection”. *To decide* on the learning runs: whether to use four or five cuts at 13–15 digits to raise the yield. In longest run, the training answer comes from one cut at the join; cuts three positions either side are used only to compute check 2 as a diagnostic.

### P5. Teachers and selection rules

The no-carry rule and the edges-differ rule together are *the hand-written rules*: the filters of Shin et al. (their Table 1), which use knowledge of where each task's combining step fails. In code: `no_carry`, `edges_differ`, `agreement`, `no_selection`, `sample_and_vote` (`src/gainpred/teachers.py`). Renamed on 13 November 2026 from "their rule" and "two cuts agree"; run `si-their-seed1-r1` and earlier logs use the old names `their_rule` and `two_cuts_agree`.

| Name | Definition |
| --- | --- |
| The no-carry rule | Keep the cut-1 answer if every lower piece sum fits in its piece's width, that is, no carry crosses a cut |
| The edges-differ rule | Keep the answer if the last symbol of the left piece differs from the first symbol of the right piece |
| The agreement rule | Build the answer from all three cuts. If at least two give the identical string, keep that string; otherwise set the problem aside |
| No selection | Keep the cut-1 answer always |
| Sample and vote | Five direct answers sampled at temperature 1.0. If at least three are identical, keep that answer |
| Pick the longer | Keep the left piece's longest run if it is at least as long as the right piece's; otherwise the right piece's |
| Join at the cut | Candidates: the left piece's longest run; if the left piece's right-edge symbol equals the right piece's left-edge symbol, a run of that symbol whose length is the two edge runs added; the right piece's longest run. Keep the longest, leftmost on ties |

### P6. Hard problems for longest run

| Kind | Share | Construction |
| --- | --- | --- |
| Random join | Half | Two strings drawn like the easy strings, placed side by side |
| Run across the join | Half | Two such strings; then the last *a* symbols of the left and the first *b* symbols of the right are set to one symbol, with *a* and *b* each drawn from 1 to 4. The symbols just outside this stretch must differ from it |

The withheld version of experiment 2 discards run-across-the-join problems with *a* of 3 or 4. All versions draw from one fixed pool.

### P7. Test problems

Generated first, from their own random seed, before any training data, and excluded from every training pool.

| Set | Size | Groups |
| --- | --- | --- |
| Addition, study | 1,000 per group | Each length 8–15, split by whether a carry crosses cut 1 |
| Addition, reproduction | 100 per length | Lengths 3–15, as in Shin et al., for comparison with their curves |
| Easy problems, for check 0 | 1,000 per piece type and size | Addition lengths 3–7; the three longest-run questions at lengths 6–10, also reported by run length |
| Longest run, study | 1,000 per group | Random joins; run-across-the-join by joined length 2–8 and by left part 1–2 versus 3–4 |

### P8. The checks, exactly

| Check | Computed on | Definition | Uses right answers? |
| --- | --- | --- | --- |
| 0 | Easy test problems | Share answered correctly, per piece type, size and run length. Pass mark 98% | Yes, for easy problems only |
| 1 | 2,000 fresh problems per length, at **every** length from 8 to the current maximum, each round | Share of the teacher's kept answers that differ from the greedy direct answer | No |
| 2 | Same sample, or all training problems when used for selection | The agreement rule of P5; reports share kept and share set aside | No |
| 2b, confirmed share | Same sample | Share of the teacher's kept answers that are identical to the answer built from at least one cut other than the one that produced them (for sample and vote: from any cut). 100% for the agreement rule by construction | No |
| 3 | 10,000 sets of piece answers per teacher | Share of teacher outputs identical to one of the piece answers. “Hands back” if 100% | No |
| Teachable share | Same sample as check 1 | Share of hard problems with a kept teacher answer that differs from the direct answer. A cost measure: with a fixed quota it says how many candidates must be drawn, not how much the student learns | No |

### P9. Runs, logging and acceptance tests

**Runs.** Five random starting points per configuration, numbered 0 to 4. Starting points 0 and 1 are learning runs. Starting points 2, 3 and 4 are confirming runs and are not started until the stage is frozen.

**Logged per run, round and size.** Share kept; share of kept answers that are right; candidates drawn; checks 1, 2 and 2b and the teachable share **at every length from 8 to the current maximum**; accuracy on every test group; tokens and wall-clock time, separately for building answers, for the checks, and for training.

**Acceptance tests for stage 1,** in order.

| Test | Passes when |
| --- | --- |
| Generators | Every answer matches plain arithmetic; regenerating gives identical files |
| Format | A problem converts to characters and back unchanged |
| Model | Between 14.15M and 14.20M parameters (ours: 14,186,112); memorises 64 problems to zero error |
| Starting training | At least 99% on easy addition test problems |
| Starting point | Accuracy on 8- and 9-digit problems asked directly is recorded; no pass mark at 8 digits. This model can reach one digit beyond its training lengths (Shin et al., Section C.2). Low accuracy is expected from 9 digits up |
| Cutting and combining | With a perfect calculator in place of the model, reproduces the paper-test table of experiment 1. Checked 13 November 2026: rule errors of unfiltered labels up to 9 digits 61.6% against 60.8% in Shin et al.'s Table 7, 0.0% with their filter as in theirs; the agreement rule needs 2.9 draws per kept answer up to 12 digits and 14.1 at 15 (`results/paper_test.md`) |
| One round, the no-carry rule | At least 95% on both 8-digit groups and on 9-digit no-carry; 9-digit carry is recorded. Passed 13 November 2026 (`si-their-seed1-r1`: 99.6 / 98.6 and 99.8 / 69.9) |
| The gate: four rounds, the no-carry rule | Accuracy holds to 15 digits, in the direction of their Figure 4. Passed 13 November 2026 (`si-nocarry-seed1`: 93.7–99.5% at every length 8–15) |

### P9b. Experiment 3: cases, verdict and outcome

Experiment 3 was redesigned on 13 November 2026. In the gate run every check said "go" and every round paid off, so there was nothing for a predictor to tell apart. It now compares a verdict given before training with what training then did.

| Case | Expected | Check that should say so | Runs |
| --- | --- | --- | --- |
| Agreement rule, new lengths | Pays off | Checks 0, 1, 2 pass | Experiment 1 |
| No-carry rule, new lengths | Pays off | Checks 0, 1 pass; 2b confirms only part | Experiment 1 and the gate |
| No selection | Does harm | 2b low | Experiment 1 |
| Sample and vote | Does little or nothing | Check 1 low | Five new runs |
| Lengths already learned | Does nothing | Check 1 near 0% | Every run |
| Starting model stopped early (weak pieces) | Stalls or does harm | Check 0 fails | Three new starting models |
| Run 0 at 8 digits against 9 digits | Smaller gain at 8 | Check 1 lower at 8 | The gate from run 0 |

**Verdict, decided in this order.** *Does harm* if check 0 fails or 2b is below a set level. *Does nothing* if check 1 is below a set level. *Pays off* otherwise. *To decide* on the learning runs, then frozen: the two levels, and the three stopping points for the weak starting models.

**Outcome, from the examiner.** Per case, run, round and length: *paid off* (accuracy up at least 20 points), *did nothing* (within 3 points), *did harm* (down more than 3 points at that length or on easy problems).

**Support.** In the confirming runs the verdict matches the outcome in at least 9 of 10 cells, with at least five cells in each of the did-nothing and did-harm groups. Verdicts are also scored per round over all lengths together, because gains travel between lengths (9-digit carry: 69.5% to 98.4% in round 2 of the gate).

### P10. Assumptions, departures and amendments

**Assumptions where their paper is silent.** Learning-rate schedule and gradient clipping as in Lee et al.'s code (`experiments/run_forward_vanilla.sh` at github.com/JackCai1206/arithmetic-self-improve): warmup-stable-decay (10% warm-up, last 20% cosine decay to 1% of the peak) and clipping at gradient norm 1.0. A first run with a constant learning rate and no clipping (start-seed0, 13 November 2026) spiked repeatedly and ended at 77% on easy problems after reaching 98.8% at step 8,000; loss on answer characters only; how pieces are chosen within a size (ours: the fixed cuts of P4). Model details (`src/gainpred/model.py`): a Llama-style block, inferred from the parameter count (pre-norm RMSNorm, SwiGLU feed-forward of width 1536, no biases; 14,186,112 parameters with the 33-token vocabulary of `src/gainpred/tokenizer.py`, where a GPT-2-style block would give 10.6M); separate input and output embeddings, as in Llama; weights initialised from a normal distribution with standard deviation 0.02; weight decay on weight matrices and embeddings but not on norm gains; float32 weights and optimizer state with bfloat16 computation; no dropout. Formats (`src/gainpred/formats.py`): a newline separates question and answer, and the prompt ends with `A: `, so the model writes only the answer and then an end token; an answer is readable only in canonical form (no leading zeros) and only if the model stopped by itself; two unreadable answers never count as agreeing, even when their text is identical. Addition data (`src/gainpred/data.py`): both numbers in a problem have the same length; each digit is drawn uniformly, with no 0 at the first digit or, for 8–15 digits, at the start of any piece under the three cuts; problems are unique within a set; each set avoids every exact problem in the sets generated before it (test sets first, then training); the reproduction test set obeys only the leading-digit rule; fixed seeds 1001 (study test), 1002 (reproduction test), 1003 (easy test) and 2001 (starting training), with the file fingerprints frozen in `tests/test_data.py`. Training (`src/gainpred/train.py`): batches are drawn by shuffling the training set once per epoch, with the shuffle seeded by the run's starting point, and the last incomplete batch of each epoch is dropped; greedy answers for evaluation are generated under the same bfloat16 mixed precision as training. Teachers (`src/gainpred/teachers.py`): a lower piece sum wider than its piece is written out in full, so the built answer is one digit too long; a built answer is unreadable if any piece answer is unreadable or the joined digits are not a canonical number; the no-carry rule tests the model's own piece sums, never the true carries; no selection sets an unreadable cut-1 answer aside (it cannot be trained on) and counts it; the agreement rule and sample and vote count only readable answers towards agreement. With a perfect calculator, the no-carry rule keeps about 38% at 8–14 digits rather than half, because no piece may start with 0, which makes carries more likely (`results/paper_test.md`). Starting point (13 November 2026): the two learning runs differ at 8 digits asked directly (seed 0: 58%, right length, tens digit often wrong; seed 1: 0%, every answer one digit short), while both are near 0% at 9 digits. We keep 8 digits as the first hard size and measure every run's own starting accuracy on all test groups before round 1, reporting gains against it. Rounds (`src/gainpred/rounds.py`): each round continues from the previous round's weights with a fresh AdamW state; the learning rate is held at the peak and then decays along a cosine to 1% over the last 1,000 of the 3,000 updates, with no warm-up (Lee et al.'s code for self-improvement rounds); "5,000 per seed length" is read as 5,000 for each of 3–7 digits, 25,000 in all, matching Shin et al.'s Table 4; replay answers are the current model's greedy answers, unchecked, and an unreadable replay answer is dropped and counted; candidates are drawn per starting point, round and length from their own seeds, so all teachers with the same starting point see the same candidate stream; rounds never draw a test or starting-training problem, nor a problem drawn earlier in the run.

**Departures from their setup.** Four rounds of addition where they run eight; one round of longest run where they run eight; three cuts per addition problem where they use one; two edge-run questions added to longest-run starting training; half the longest-run hard problems built with a run across the join; no piece starting with 0 under any cut.

**Amendments.** *13 November 2026, before any freeze.* (1) Constant learning rate replaced by warm-up, stable, decay, with clipping at 1.0; reason: `start-seed0` spiked and ended at 77% on easy problems. (2) The starting-point test no longer requires close to 0% at 8 digits; reason: run 0 answers 58% of 8-digit problems directly. (3) "Their rule" and "two cuts agree" renamed the no-carry rule and the agreement rule. (4) Experiment 3 redesigned around seven contrast cases and a three-way verdict (P9b); check 1 replaces the teachable share as the predictor; reason: in the gate run the teachable share was 14% at 15 digits and about 38% elsewhere with the same gain. (5) Checks computed at every length each round, and check 2b added.
