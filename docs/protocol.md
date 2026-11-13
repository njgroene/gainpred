## Appendix: Protocol

The working rules for building and running the experiments. Draft of 12 November 2026. It is frozen stage by stage, before that stage's confirming runs; after that, changes go only in the amendments log at the end, with a date and a reason. Items marked *to decide* are settled on the learning runs.

### P1. Settings taken from Shin et al.

From their Section A.1, A.2 and Table 4, re-read on 12 November.

| Setting | Value |
| --- | --- |
| Model | Character-level Transformer, 6 layers, hidden size 384, 6 attention heads, no positional encoding, about 14.2M parameters |
| Optimizer | AdamW, β1 0.9, β2 0.99, ε 1e-12, weight decay 0.1 |
| Learning rate, batch size | 5e-4, 256 |
| Precision, decoding | bfloat16; greedy decoding for all built answers |
| Addition, starting training | Both numbers 3–7 digits, 50,000 problems per length, 10,000 updates |
| Addition, each round | 10,000 new built problems per length from 8 up to the current maximum, plus 5,000 new easy problems per easy length answered by the current model; 3,000 updates |
| Longest run, starting training | Strings of length 6–10, 50,000 per length, 15,000 updates |
| Longest run, each round | 2,000 new built strings per length from 12 up to the current maximum, plus the starting strings replayed with true answers; 3,000 updates |

**Not stated in their paper; our assumptions.** Constant learning rate with no warm-up. Loss computed on answer characters only. Both are checked against their code if it arrives, and listed under P10.

### P2. Formats

| Item | Rule |
| --- | --- |
| Addition question and answer | `Q: 6133 + 8894 = ?` then `A: 15027`. Digits are written forward, most significant first; their Section C.2 confirms “forward addition, as in our setting” |
| Built addition answer | The piece sums written side by side. The leading sum may be one digit longer than its piece; every lower sum is padded with zeros to its piece's width |
| Longest-run question and answer | `Q: runlen(493036) = ?` then `A: 4\|1` (symbol, then length). The leftmost run wins ties |
| Edge-run questions | `Q: leftrun(493036) = ?` and `Q: rightrun(493036) = ?`, answered in the same symbol-and-length form |
| No piece starts with 0 | Addition problems are drawn so that, under each of the three cuts, no piece of either number begins with 0. Otherwise a piece would look unlike any easy problem |
| Unreadable answers | An answer that cannot be parsed counts as wrong when scoring, and as its own distinct answer when comparing answers. It is never dropped |

### P3. Sizes and rounds in this study

| Task | Rounds | Hard sizes | Kept answers per size, per round |
| --- | --- | --- | --- |
| Addition | 4 | Maximum 9, then 11, 13, 15 digits; every length from 8 to the maximum is rebuilt each round | 10,000 |
| Longest run | 1 | Lengths 12–20, each string made of two pieces of length 6–10 | 2,000 |

Every version of an experiment trains on the same number of kept answers. Candidates are drawn until the quota is met, capped at twenty times the quota; the number drawn and any shortfall are logged. The paper test puts the need at about three candidates per kept answer up to 12 digits and about fourteen at 15 digits for the two-cuts-agree rule.

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

Two rules govern the table. Pieces are always 3–7 digits, the easy sizes. And no two cuts of a problem may place a cut at the same position: when they do, a carry at that position makes both wrong in the same way, and agreement stops being evidence. Every row was checked in simulation on 12 November. Cut 1 supplies the answer for “their rule” and “no selection”. *To decide* on the learning runs: whether to use four or five cuts at 13–15 digits to raise the yield. In longest run, the training answer comes from one cut at the join; cuts three positions either side are used only to compute check 2 as a diagnostic.

### P5. Teachers and selection rules

| Name | Definition |
| --- | --- |
| Their rule, addition | Keep the cut-1 answer if every lower piece sum fits in its piece's width, that is, no carry crosses a cut |
| Their rule, longest run | Keep the answer if the last symbol of the left piece differs from the first symbol of the right piece |
| Two cuts agree | Build the answer from all three cuts. If at least two give the identical string, keep that string; otherwise set the problem aside |
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
| 1 | 2,000 hard problems per new size, each round | Share where the greedy direct answer differs from the kept teacher answer | No |
| 2 | Same sample, or all training problems when used for selection | The two-cuts-agree rule of P5; reports share kept and share set aside | No |
| 3 | 10,000 sets of piece answers per teacher | Share of teacher outputs identical to one of the piece answers. “Hands back” if 100% | No |
| Teachable share | Same sample as check 1 | Share of hard problems with a kept teacher answer that differs from the direct answer | No |

### P9. Runs, logging and acceptance tests

**Runs.** Five random starting points per configuration, numbered 0 to 4. Starting points 0 and 1 are learning runs. Starting points 2, 3 and 4 are confirming runs and are not started until the stage is frozen.

**Logged per run, round and size.** Share kept; share of kept answers that are right; candidates drawn; checks 1 and 2 and teachable share; accuracy on every test group; tokens and wall-clock time, separately for building answers, for the checks, and for training.

**Acceptance tests for stage 1,** in order.

| Test | Passes when |
| --- | --- |
| Generators | Every answer matches plain arithmetic; regenerating gives identical files |
| Format | A problem converts to characters and back unchanged |
| Model | About 14.2M parameters; memorises 64 problems to zero error |
| Starting training | At least 99% on easy addition test problems |
| Starting point | Close to 0% when asked 8- and 9-digit problems directly |
| Cutting and combining | With a perfect calculator in place of the model, reproduces the paper-test table of experiment 1 |
| One round, their rule | High accuracy at 8 and 9 digits, including problems with a carry across the cut |
| The gate: four rounds, their rule | Accuracy holds to 15 digits, in the direction of their Figure 4, by the end of week 3 |

### P10. Assumptions, departures and amendments

**Assumptions where their paper is silent.** Constant learning rate; loss on answer characters only; how pieces are chosen within a size (ours: the fixed cuts of P4). Model details (`src/gainpred/model.py`): a Llama-style block, inferred from the parameter count (pre-norm RMSNorm, SwiGLU feed-forward of width 1536, no biases; 14,186,112 parameters with the 33-token vocabulary of `src/gainpred/tokenizer.py`, where a GPT-2-style block would give 10.6M); separate input and output embeddings, as in Llama; weights initialised from a normal distribution with standard deviation 0.02; weight decay on weight matrices and embeddings but not on norm gains; float32 weights and optimizer state with bfloat16 computation; no dropout; no gradient clipping. Formats (`src/gainpred/formats.py`): a newline separates question and answer, and the prompt ends with `A: `, so the model writes only the answer and then an end token; an answer is readable only in canonical form (no leading zeros) and only if the model stopped by itself; two unreadable answers never count as agreeing, even when their text is identical. Addition data (`src/gainpred/data.py`): both numbers in a problem have the same length; each digit is drawn uniformly, with no 0 at the first digit or, for 8–15 digits, at the start of any piece under the three cuts; problems are unique within a set; each set avoids every exact problem in the sets generated before it (test sets first, then training); the reproduction test set obeys only the leading-digit rule; fixed seeds 1001 (study test), 1002 (reproduction test), 1003 (easy test) and 2001 (starting training), with the file fingerprints frozen in `tests/test_data.py`.

**Departures from their setup.** Four rounds of addition where they run eight; one round of longest run where they run eight; three cuts per addition problem where they use one; two edge-run questions added to longest-run starting training; half the longest-run hard problems built with a run across the join; no piece starting with 0 under any cut.
