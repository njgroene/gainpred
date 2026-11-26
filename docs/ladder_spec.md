# Experiment Specification: A Self-Teaching Ladder from 8 to 39 Digits

**Status:** Finalized specification for implementation (design review incorporated; four corrections of 26 November 2026 applied, marked *[corrected]*)  
**Base commit:** `8f19604` (`src/gainpred/floors.py`)

---

## 1. Summary & Objective

This experiment tests what a small Transformer needs to teach itself multi-digit addition far beyond its supervised lengths ($3\text{--}7$ digits), using **no ground-truth labels above $7$ digits** and **no hand-written error filter**. It runs a clean four-arm ablation where each arm adds exactly one mechanism to the arm before it:

| Arm | Training answers at new lengths | Mechanism added | Core question |
| :--- | :--- | :--- | :--- |
| **1. Direct** | Direct answers to longer sums, all valid number strings kept | Baseline | Does naive unfiltered self-training work? |
| **2. Composed** | Answers composed from two shorter pieces (Cut 1), all valid number strings kept | Composition | Is decomposing into shorter pieces enough on its own? |
| **3. Composed + Agreed** | Composed answers kept only when at least $2$ of $3$ neighbouring cuts produce the exact same answer *[corrected: the cuts are neighbouring positions, not disjoint]* | Selection by split agreement | Does label-free split invariance let the model climb end-to-end? |
| **4. Composed + Agreed + Self-paced** | As Arm 3, but built only at lengths where pre-round check shows work remains | Autonomous go / no-go pacing | Can the model pace its own curriculum and save compute without losing accuracy? |

All four arms start from the **exact same saved seed checkpoint** (trained with ground-truth answers on $3\text{--}7$ digits) and run one unified protocol across $16$ rounds ($4$ stages covering $8$ to $39$ digits) using the same middle-cut rule everywhere.

* **Epistemic status:** Exploratory ablation with operational definitions, thresholds, stopping rules, and seed order frozen before any GPU run (to be recorded accordingly in Appendix A, *"What was fixed when"*). No pass/fail outcome predictions are pre-registered for the arms.
* **Budget:** Expected $\approx \$13.80$ per seed ($\approx \$41$ across seeds $5, 6, 7$), with a hard spending cap of $\$28.00$ on Seed 5 (at $\$1.10$/GPU-hour).

---

## 2. Why This Replaces the Earlier Loop Runs

1. **Removes the Stage 1 hand-written filter dependency:** The earlier $16\text{--}39$ digit climb stood on an $8\text{--}15$ digit checkpoint trained with Shin et al.'s hand-written no-carry filter. Starting at $8$ digits from the $3\text{--}7$ digit seed makes the climb label-free and filter-free end-to-end.
2. **Unconfounds composition and selection:** Earlier controls jumped directly from `composed + agreement` to `direct + unfiltered`, changing two ingredients at once. Arm 2 (`composed, unfiltered`) isolates composition from selection.
3. **Tests active check-based steering:** Earlier pre-round verdicts were logged passively, and the frozen $50\%$ single-round novelty cutoff was too coarse to gate an iterative climb. Arm 4 tests active per-length gating at a $2\%$ novelty threshold.
4. **Unifies cuts and expands seeds:** Uses one uniform cutting rule across all lengths ($8\text{--}39$) and tests across three fresh random seeds ($5, 6, 7$) rather than three data repeats on a single seed.

---

## 3. Disambiguated Terminology

Every term below has a single strict definition in code, logs, and paper text. Note especially the distinction between **first-cut confirmed** (diagnostic metric $c$) and **pair-agreed** (selection rule for Arms 3 & 4):

| Term | Exact definition |
| :--- | :--- |
| **Sum** | Addition problem of two $n$-digit integers ($n$ is the *length*). Neither full addend nor any cut piece may begin with digit `0`. |
| **Readable number** | *[corrected]* An answer that the existing answer parser of the repository does **not** mark as `Unreadable` (the test used by `gainpred.teachers.readable` and `gainpred.teachers.majority`). The ladder uses that same parser, so that "readable" means the same as in every earlier run. Intended meaning, for comparison in the read-through: the output string `s` after stripping EOS/padding satisfies `len(s) > 0 and s.isascii() and s.isdigit()`. If the existing parser accepts or rejects any string differently from this rule, report it in the read-through and keep the existing parser. |
| **Seed model** | Model trained from scratch with ground-truth answers on sums of $3\text{--}7$ digits. One checkpoint per random seed ($5, 6, 7$). |
| **Direct answer** | Model's output when given an $n$-digit sum directly without splitting. |
| **Cut** | Division of both addends of an $n$-digit sum at index $k$, producing a left sum of $k$ digits and a right sum of $n - k$ digits (written $k + (n - k)$). |
| **Composed answer** | Direct answers to the left and right pieces concatenated side-by-side without carry handling. Valid only if both piece outputs (and thus their concatenation) are readable numbers; wrong whenever a carry crosses the cut or a piece sum is wrong. |
| **First-cut confirmed ($c$)** | Diagnostic metric (`confirmed_share`): among readable composed answers from **Cut 1**, the fraction reproduced identically by **Cut 2 or Cut 3** ($\approx 24.6\%$ on clean sums). |
| **Pair-agreed** | Selection criterion for Arms 3 & 4 (`agreed_share`): at least **two of the three cuts** (Cut 1==Cut 2, Cut 1==Cut 3, or Cut 2==Cut 3) produce the **exact same readable number string** ($\approx 34.4\%$ of fresh sums, yielding $\approx 640\text{--}720$ sums per $2,000$). Since there are only $3$ cuts, a matching pair is always unique. |
| **Novelty ($\nu_{\text{arm}}$, $\nu_{\text{agreed}}$)** | Disagreement rate against the model's own direct answer on the $2,000$-sum pre-round check. Both the arm's own training-answer novelty ($\nu_{\text{arm}}$) and the pair-agreed novelty ($\nu_{\text{agreed}}$) are logged in every arm; Arm 4 gates on $\nu_{\text{agreed}}$. |
| **Round & Stage** | One **round** introduces $2$ new lengths, builds data, trains, and evaluates. One **stage** ($4$ rounds) covers $8$ lengths: Stage 1 ($8\text{--}15$), Stage 2 ($16\text{--}23$), Stage 3 ($24\text{--}31$), Stage 4 ($32\text{--}39$). |

---

## 4. Shared Setup Across All Four Arms

### 4.1 Model & Seed Checkpoints
* **Architecture:** Shin et al.'s character-level Transformer ($\approx 14\text{M}$ parameters, no positional encodings, max sequence length set to $136$ characters as in the $39$-digit runs; no learned weights depend on this buffer limit).
* **Seeds:** Random seeds `5`, `6`, and `7` (unused in all prior runs).
* **Shared initialization:** Train the seed model once per random seed on $3\text{--}7$ digit ground-truth sums using the existing starting-model recipe, save the checkpoint, and load that exact file to initialize Arms 1, 2, 3, and 4.
* **Seed entry check:** The seed model must score $\ge 98\%$ test accuracy at **each** length in $\{3, 4, 5, 6, 7\}$. If a seed fails, log and report it as failed and increment to the next unused seed integer.

### 4.2 Uniform Cutting Rule ($8$ to $39$ Digits)
At every length $n \in \{8, \dots, 39\}$, let $h = \lfloor n / 2 \rfloor$. Every problem uses three middle cuts with left-piece lengths:
$$\text{Cut 1: } h - 1, \qquad \text{Cut 2: } h, \qquad \text{Cut 3: } h + 1$$
*(Example: $n = 12 \to 5+7, 6+6, 7+5$; $n = 39 \to 18+21, 19+20, 20+19$.)*

**Structural properties (note for implementation and paper prose):**
* Every piece has length $\ge 3$ and strictly less than both new lengths of its round.
* **Tree depth structure:** Lengths $8\text{--}12$ use only seed pieces ($3\text{--}7$, depth $1$); lengths $13\text{--}17$ mix seed and self-taught pieces; lengths $18\text{--}39$ use *only* self-taught pieces ($\ge 8$); specifically, lengths $13\text{--}22$ have depth $2$ and lengths $23\text{--}39$ have depth $3$, with **maximum piece length $21$** (lengths $22\text{--}39$ are never used as pieces).
* **Reference values with exact piece answers** *[corrected: added]*: simulated on 200,000 fresh sums at each of the lengths 8, 9, 12, 16, 17, 23, 30 and 39, a carry crosses Cut 1 in 61.5 to 61.9% of sums (so the Cut 1 composed answer is right in 38.1 to 38.5%), `confirmed_share` is 24.5 to 24.7%, `agreed_share` is 34.2 to 34.6%, and no pair-agreed answer was wrong. A model whose piece answers are right should log values close to these in its pre-round check.
* **Carry-chain independence:** Algebraically, when piece answers are exact, a carry across cut $k$ yields integer value $S + 9 L_k 10^{n-k}$; two cuts $k_1 < k_2$ cannot agree on a carry-corrupted string unless all digits in $[k_1, k_2)$ are `0` in both addends, which prevents any carry from propagating across both cuts. Thus multi-position carry runs never cause false pair-agreement ($\varepsilon = 0$ on exact pieces).

### 4.3 Held-Out Test Sets
Generate a dedicated, frozen test set for lengths $8\text{--}39$ before generating any training data:
* At each length $n \in \{8, \dots, 39\}$: **$1,000$ sums with a carry across Cut 1** ($h - 1$) and **$1,000$ sums without a carry across Cut 1** ($2,000$ total per length; no addend or piece across any of the $3$ cuts begins with `0`).
* Retain existing $3\text{--}7$ digit easy test sets. Do not reuse the old $16\text{--}39$ test sets because Cut 1 changed on $24\text{--}39$ digits.

### 4.4 Anatomy of One Round ($r = 1 \dots 16$)
 Round $r$ introduces two new lengths $(n_a, n_b) = (2r + 6, 2r + 7)$:

1. **Check (`2,000` fresh sums per new length, no weight updates):**
   Evaluate the current model directly and across all $3$ cuts on $2,000$ fresh sums at $n_a$ and $2,000$ at $n_b$. Compute all logged pre-round metrics (Section 6).
2. **Build (`10,000` target per built length, `200,000` draw cap):**
   For each new length scheduled to build, draw fresh sums (with no leading `0` on any cut piece) until $10,000$ valid answers are kept under the arm's rule, or until $200,000$ sums ($20\times$ quota) have been drawn.
   * *Shortfall guardrail:* If $0$ answers are kept after $200,000$ draws at a length scheduled to build, log the shortfall, mark the round/stage failed, and stop the arm cleanly. If $1 \le \text{kept} < 10,000$, log the shortfall warning and proceed with the kept answers.
3. **Train:**
   Combine:
   * New training answers from this round,
   * Stored training answers from all prior rounds ($8$ up to $2r + 5$), and
   * $5,000$ fresh easy sums at each length $3\text{--}7$ labeled by the current model itself (as in existing code).
   * **Update count formula:** Let $N_{\text{raw}}$ be the actual unrepeated number of examples in this combined set. Train with batch size $256$ for:
     $$U = \max\!\left(3000,\; 100 \cdot \left\lceil \frac{7 \cdot N_{\text{raw}}}{256 \cdot 100} \right\rceil\right)$$
     updates (matching $3,000$ updates in Round 1, scaling as $7$ epochs rounded up to the next $100$).
4. **Test:**
   Evaluate test accuracy at every length from $3$ up to $2r + 7$ (reporting overall accuracy and accuracy split by carry / no-carry across Cut 1).

---

## 5. Exact Rules for the Four Arms & Guardrails

| Arm | Candidate answer | Kept when | Rehearsal & training |
| :--- | :--- | :--- | :--- |
| **1. Direct** | Direct answer on $n$-digit sum | Output is a readable number | Standard $10,000$ stored per length |
| **2. Composed** | Cut 1 composed answer | Both piece outputs (and joined string) are readable numbers | Standard $10,000$ stored per length |
| **3. Composed + Agreed** | Unique string produced by $\ge 2$ of the $3$ cuts | At least $2$ of $3$ cuts produce identical readable numbers | Standard $10,000$ stored per length |
| **4. Self-paced** | Same as Arm 3 | Built only if pre-round check triggers go; otherwise reuses check's agreed answers | Upsampled rehearsal for skipped lengths (see 5.1) |

### 5.1 Arm 4 Decision Rule, Collapse Guardrail, and Rehearsal Weighting
Before each round, Arm 4 evaluates each of its two new lengths independently on the $2,000$-sum pre-round check:
1. Let $M_{\text{agreed}}$ be the number of check sums (out of $2,000$) that are **pair-agreed** (at least $2$ of $3$ cuts match on a readable number).
2. Let $\nu_{\text{agreed}}$ be the fraction of those $M_{\text{agreed}}$ answers that differ from the model's direct answer on the same sum.
3. **Go / Build condition:** Build $10,000$ answers for length $n$ if and only if:
   $$M_{\text{agreed}} < 100 \quad (\text{i.e., } < 5\% \text{ retention}) \qquad \text{OR} \qquad \nu_{\text{agreed}} \ge 0.02 \quad (\ge 2\%)$$
   *(Rationale for $M_{\text{agreed}} < 100$: mirrors the $5\%$ retention floor of Section 2.4. If fewer than $5\%$ of sums achieve 2-of-3 agreement, the teacher is breaking down rather than the student having mastered the length, so $\nu_{\text{agreed}}$ must not trigger a false skip.)*
4. **No-Go / Skip handling ($M_{\text{agreed}} \ge 100$ and $\nu_{\text{agreed}} < 0.02$):**
   * **Skip Build:** Do $0$ additional build draws for length $n$. Save the $M_{\text{agreed}}$ pair-agreed answers already computed during the check ($\approx 640\text{--}720$ examples) as length $n$'s rehearsal pool.
   * **Prevent rehearsal starvation without inflating update counts:** Lengths like $18$ and $20$ are critical pieces for Stage 4 ($36\text{--}39$ digits). If stored at raw size $\approx 680$ against $10,000$ for built lengths, they receive $15\times$ less sampling weight. Therefore:
     * Count only the **$M_{\text{agreed}}$ unrepeated examples** inside $N_{\text{raw}}$ when computing the update count $U$ (so skipping a length genuinely reduces gradient steps as well as build time).
     * During **uniform batch sampling**, weight/upsample every skipped length's stored pool so that each length in $8 \dots 2r+7$ is sampled with equal effective pool weight ($10,000$ examples per active self-taught length, alongside $5,000$ per easy length $3\text{--}7$).
5. **Full-round skip:** If **neither** $n_a$ nor $n_b$ triggers Build, skip Step 2 (Build) and Step 3 (Train) entirely—carry the model weights forward unchanged, store both check pools ($M_{\text{agreed}}$ each) for rehearsal in future rounds that *do* train, and run Step 4 (Test) as usual.

### 5.2 Stage Stopping Rule & Spending Cap
* **Stage learned:** After the $4$th round of a stage, test accuracy is $\ge 90.0\%$ at **every** one of the stage's $8$ lengths.
* **Early stopping per arm:** Run stage by stage. If an arm fails to learn a stage, stop that arm at the end of that stage (applies identically to all $4$ arms). Note that Stage 1 ($8\text{--}15$) already tests both depth-$1$ ($8\text{--}12$) and depth-$2$ ($13\text{--}15$) composition for Arm 2.
* **Seed progression:** Run Seed `5` completely first. Proceed to Seeds `6` and `7` unless Arm 3 fails to learn Stage 1 or Stage 2 on Seed `5`.
* **Spending cap:** Track cumulative logged GPU seconds on Seed `5` priced at $\$1.10 / 3600$ per second. If cumulative spend reaches $\$28.00$, allow the currently executing round to finish cleanly and start no further rounds until reviewed.
* **Reporting** *[corrected: added]*: every run that is started is reported, including runs that crash and are restarted. No run is dropped or replaced without saying so.

---

## 6. Unified Logging Schema

Every arm must log the exact same schema so any pair of arms can be compared directly:

1. **Pre-round check (`2,000` sums per new length, logged in ALL arms):**
   * `direct_acc`: ground-truth accuracy of direct answers (for post-hoc analysis only).
   * `cut1_acc`: ground-truth accuracy of Cut 1 composed answers (analysis only).
   * `confirmed_share` ($c$): fraction of readable Cut 1 composed answers confirmed by Cut 2 or Cut 3.
   * `agreed_share`: fraction of the $2,000$ sums where at least $2$ of $3$ cuts match on a readable number ($M_{\text{agreed}} / 2000$), plus ground-truth accuracy among those pair-agreed answers (analysis only).
   * `novelty_arm` ($\nu_{\text{arm}}$): disagreement between the arm's candidate selection rule and direct answers ($0$ by construction in Arm 1).
   * `novelty_agreed` ($\nu_{\text{agreed}}$): disagreement between pair-agreed answers and direct answers, plus `arm4_verdict` (`build` vs. `skip_length`).
2. **Build step (per new length):**
   * `sums_drawn`, `answers_kept`, `shortfall_flag`, `kept_acc` (ground-truth accuracy of kept answers, analysis only), `kept_carry_share` (fraction of kept answers with a carry across Cut 1).
3. **Train & Test step (per round):**
   * `n_raw_examples`, `n_effective_examples`, `num_updates`, `round_skipped` (bool).
   * Wall-clock GPU seconds broken down cleanly: `sec_check`, `sec_build`, `sec_train`, `sec_test`, `sec_total`.
   * Per-length test accuracy for every length $3 \dots 2r+7$: `acc_total`, `acc_no_carry`, `acc_carry` (at Cut 1 for $n \ge 8$).

---

## 7. Implementation Workflow & Non-Destructive Constraints

1. **Mandatory Pre-Code Read-Through Gate:**
   Before writing or modifying any code, inspect `src/gainpred/floors.py` (and associated test/training scripts at `8f19604`) against this specification. Report:
   * Any hidden assumptions in `floors.py` (e.g., hardcoded `n >= 16` bounds, tokenization/EOS stripping, batch sampler mechanics, sequence length padding, or test-set loader assumptions).
   * Any ambiguity or conflict between existing helpers and this spec.
   * Wait for confirmation if anything substantive disagrees; otherwise proceed to implementation.
2. **Zero Regression on Existing Paper Artifacts:**
   * Existing run logs, generated tables, figures, and `\gp...` LaTeX macros from `8f19604` **must regenerate identically**.
   * Isolate the new cut table (`cuts_middle_8_39`), new test set (`test_middle_8_39`), and new run identifiers (`ladder_s{seed}_arm{1..4}`) cleanly from the legacy `floors` runs.
3. **Two-Stage Smoke Test Before Seed 5:**
   * **CPU dry run:** Tiny model ($2$ layers, small hidden dim), $20$ check sums, quota $50$, $10$ updates across $2$ rounds on all $4$ arms (explicitly testing both `build` and `skip_length` / `skip_round` branches in Arm 4, plus the $0$-kept shortfall branch).
   * **GPU smoke test ($< \$1$):** Verify throughput, memory at $39$ digits (`max_len=136`), and GPU timer logging before launching Seed `5`.

---

## 8. Notes for Paper Integration (Post-Run)

* **Section 4 stays untouched:** The pre-registered single-round evaluation ($96.5\%$ across $258$ cases) remains unchanged.
* **Section 5 becomes the 4-arm ladder ($8 \to 39$ digits):**
  * Avoid phrasing the new loop as *"composing each stage from the stage directly below"*—under middle cuts ($h-1, h, h+1$), describe it accurately as balanced binary-tree composition where lengths $8\text{--}39$ bootstrap through three levels of pieces up to $21$ digits.
  * State plainly that Arm 1 tests *unfiltered* direct self-training (noting Lee et al.'s filtered single-digit approach), and state transparently that Arm 4's $2\%$ threshold was selected on the earlier $16\text{--}39$ logs prior to running Seeds $5\text{--}7$.
* **Preserve valuable earlier Section 5 findings in the Supplement:**
  * Keep the $15$-digit and $23$-digit direct-answer controls in the Supplement (as stronger-checkpoint counterparts to Arm 1).
  * Keep the **$15\%$ random piece-corruption stress test** (comparing uncorrelated random label noise vs. systematic carry errors under SGD) in the Supplement, with a concise reference in the main text (Section 4.3 or Section 6).