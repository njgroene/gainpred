# Longest run: paper test and readings taken where the protocol is silent

For the owner (runbook A5), 14 November 2026. Learning runs may proceed; nothing here is frozen until decision point 1.

## Paper test (perfect solver, 10,000 strings per length 12–20, seed 3101)

Means over lengths 12–20 (each length within about 1.5 points); per length in `paper_test_longest_run.md`.

| Teacher | Kept, all | Right, all | Kept, random join | Right, random join | Kept, run across | Right, run across | Check 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Edges-differ rule | ~45 | 100 | 90.2 | 100 | 0.0 | – | 100 (hands back) |
| Pick the longer (= no selection) | 100 | 49.1 | 100 | 94.2 | 100 | 3.8 | 100 (hands back) |
| Join at the cut | 100 | 100 | 100 | 100 | 100 | 100 | 49.1 |

Against Shin et al. (random joins are the comparable rows): Table 7 unfiltered rule errors at length ≤ 20: 5.9 theirs, 5.8 ours; filtered kept 90.0 vs 90.2, rule errors 0.0 vs 0.0. Section C.7's 66.2% of strings with no run crossing a block boundary: 66.1 in a simulation of their setting (not ours). Independently checked on 14 November: `longest_run` matches a brute-force count on 20,000 strings; with a perfect solver, join at the cut is right on 7,000 of 7,000 problems and the edges-differ rule keeps only right answers.

## Readings taken (now P10 assumptions)

1. Easy strings: each symbol uniform and independent over 0–9 (the paper is silent; its 90.0% kept and 66.2% both match).
2. Strings may start with 0 (Shin's example "022227"); the no-leading-0 rule is addition only.
3. Join position: the left-piece length is uniform over the values that keep both pieces 6–10.
4. "Half" run across the join: a fair coin per problem, so half in expectation, not exactly.
5. "Symbols just outside must differ": an outside symbol equal to the run symbol is redrawn from the other nine, so the planted run has length exactly a + b.
6. Run symbol uniform over 0–9; a and b each uniform over 1–4, independently.
7. Withheld filter: drops run-across problems with construction parameter a of 3 or 4; a random join whose edges happen to match is kept.
8. Study-set lengths: each problem's length uniform over 12–20 (P7 gives none).
9. Study-set groups drawn separately: random joins; a + b = 2..8 with (a, b) uniform over pairs with that sum; a in {1, 2} versus {3, 4} with b uniform over 1–4.
10. Easy test set: each (question, length) group has its own 1,000 strings.
11. Exclusion from training is by string: a test string is never used in training under any question.
12. Starting training asks every string all three questions (P3's stated default; parameter `TRAIN_START_QUESTIONS`).
13. The edges-differ rule reads the input string (as Shin's G(x, y)), not the model's answers.
14. Join at the cut reads edge symbols from the string and edge-run lengths from the model's leftrun/rightrun answers (`EDGE_SYMBOLS_FROM`; "answers" is the alternative).
15. An unreadable piece answer that a rule needs sets the problem aside, for every teacher.
16. The check-2 diagnostic combines each of the three cuts (join, join − 3, join + 3) with join at the cut.
17. Check 3 compares against every piece answer the teacher asked for (with or without edge-run answers gives the same result).
18. Check 3 is computed on the piece answers of 10,000 real hard problems.
19. Seeds 1101 (study test), 1102 (easy test), 2101 (starting training), 3101 (paper test); round pools to use 8,000,000+.

## Open questions (for decision point 1)

- **No selection is identical to pick the longer** (pick the longer never sets anything aside). Drop one name or define them differently?
- **The ±3 diagnostic cuts mostly break the 6–10 piece rule**: they hold only when the join is at least 9 and the right piece at most 7, so check 2 asks about pieces of 3–13 symbols, outside the training lengths. Use only ±3 cuts whose pieces are 6–10, or define the cuts as alternative 6–10 splits?
- P4/P8 do not say whether check 2 and 2b combine each cut with join at the cut or pick the longer.
- "Half random join": a coin (current) or exact halves? Matters slightly for quotas, since the edges-differ rule keeps none of the run-across half.
- The study set has two overlapping groupings drawn separately; a crossed design (length × left part) would separate them.
- No longest-run reproduction test set (Shin: 100 per length, 6–83). Wanted?
- P1 says 15,000 updates for longest-run starting training, but asking three questions per string triples the rows; P3 already allows raising updates until check 0 passes.
- Run-across problems are far outside Shin's regime: pick the longer is right on 3.8% of them, so with half the pool built this way unfiltered labels are about 51% wrong, against 5.9% in the paper.
