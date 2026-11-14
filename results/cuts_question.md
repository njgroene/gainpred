# The cuts question for 13–15 digits (for decision point 1)

Perfect-calculator simulation, 20,000 problems per row (`uv run python scripts/cuts_sim.py`). Extra cuts obey P4's two rules (pieces 3–7 digits; no position shared with another cut), and problems are drawn with no piece starting with 0 under *all* the cuts in use.

| Size | Cuts | Extra cuts | Agreement rule keeps | Draws per kept | No-carry rule 2b |
| --- | --- | --- | --- | --- | --- |
| 13 | 3 | – | 23.7% | 4.2 | 53.5% |
| 13 | 4 | 5∣3∣5 | 30.0% | 3.3 | 60.0% |
| 13 | 5 | 5∣3∣5, 3∣7∣3 | 34.9% | 2.9 | 65.1% |
| 14 | 3 | – | 13.3% | 7.5 | 30.3% |
| 14 | 4 | 5∣6∣3 | 19.3% | 5.2 | 39.8% |
| 14 | 5 | 5∣6∣3, 3∣5∣6 | 25.2% | 4.0 | 48.4% |
| 15 | 3 | – | 7.2% | 13.8 | 37.1% |
| 15 | 4 | 7∣5∣3 | 11.7% | 8.6 | 44.1% |
| 15 | 5 | 7∣5∣3, 3∣5∣7 | 17.2% | 5.8 | 52.8% |

Why it matters for P9b: in the learning batch, the lowest 2b among cells that *paid off* is 0.30 and the highest among cells that *did harm or taught wrong answers* is 0.29 (`p9b.md`). The low paid-off values come from the no-carry rule at 14–15 digits, where 2b is low by construction with three cuts. More cuts raise them (14 digits: 30% → 48% with five), widening the gap a 2b level must fall into. The agreement rule's yield rises too (15 digits: 13.8 → 5.8 draws per kept answer).

Side effects of adding cuts, to weigh:
- The no-piece-starts-with-0 rule (P2) applies under every cut, so new cuts add forbidden-zero positions: hard problems at 13–15 digits would be drawn differently from now on. The test sets were drawn under the three-cut rule; a test problem may have a 0 at a new cut's piece start. That affects nothing measured on the test sets, but it is a difference between training and test problems.
- The learning batch so far used three cuts; changing now means rerunning the agreement and no-carry arms' rounds 3–4 (13–15 digits) before the freeze.
