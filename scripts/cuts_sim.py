"""Perfect-calculator simulation: 3 vs 4 vs 5 cuts at 13-15 digits (analysis only)."""
import random
from itertools import combinations
from gainpred.cuts import ADDITION_CUTS, carry_crosses, cut_positions, piece_starts
from gainpred.data import draw_number
from gainpred.formats import Addition

EXTRA = {  # candidate extra cuts: pieces 3-7, no position shared with any other cut
    13: [(5, 3, 5), (3, 7, 3)],
    14: [(5, 6, 3), (3, 5, 6)],
    15: [(7, 5, 3), (3, 5, 7)],
}
def valid(cuts):
    pos = [cut_positions(c) for c in cuts]
    return all(all(3 <= w <= 7 for w in c) for c in cuts) and all(not (a & b) for a, b in combinations(pos, 2))
for d, extra in EXTRA.items():
    base = list(ADDITION_CUTS[d])
    for k in (3, 4, 5):
        cuts = base + extra[: k - 3]
        assert valid(cuts), (d, cuts)
        nonzero = frozenset().union(*(piece_starts(c) for c in cuts))
        rng = random.Random(5000 + d * 10 + k)
        n = 20000; kept_agree = 0; nc_kept = 0; nc_conf = 0
        for _ in range(n):
            p = Addition(draw_number(rng, d, nonzero), draw_number(rng, d, nonzero))
            ok = [not carry_crosses(p, c) for c in cuts]
            kept_agree += sum(ok) >= 2
            if ok[0]:
                nc_kept += 1; nc_conf += any(ok[1:])
        print(f"{d} digits, {k} cuts {cuts[3:] or ''}: agreement keeps {100*kept_agree/n:.1f}% "
              f"({n/kept_agree:.1f} draws/kept); no-carry rule keeps {100*nc_kept/n:.1f}%, 2b {100*nc_conf/nc_kept:.1f}%")
