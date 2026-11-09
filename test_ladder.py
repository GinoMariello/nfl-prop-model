"""
Checks OverModel.ladder, the hit-rate ladder on the Prop Board. Synthetic history,
no network or database.

    python3 test_ladder.py
"""
import numpy as np
import pandas as pd

import site_data
from site_data import LEVELS, OverModel

rng = np.random.default_rng(1)
pred = rng.uniform(20, 120, 3000)
hist = pd.DataFrame({"pred_receiving_yards": pred, "actual_receiving_yards": np.maximum(0, pred + rng.normal(0, 25, 3000))})
m = OverModel(hist, "receiving_yards")

lad = m.ladder(70, 60.5)
over = [lad["over"][f"{lv:.2f}"] for lv in LEVELS]
under = [lad["under"][f"{lv:.2f}"] for lv in LEVELS]
# A surer over needs a lower line; a surer under needs a higher one.
o, u = [v for v in over if v is not None], [v for v in under if v is not None]
assert all(a >= b for a, b in zip(o, o[1:])), over
assert all(a <= b for a, b in zip(u, u[1:])), under
assert all(v % 1 == 0.5 for v in o + u), (over, under)
# Each rung really clears its level, and the next yard further does not.
c = site_data.LADDER_W * 70 + (1 - site_data.LADDER_W) * 60.5
final = c + m.misses[int(np.digitize(c, m.edges))]
cal = lambda q: 0.5 + site_data.LADDER_K * (q - 0.5)
for lv, x, y in zip(LEVELS, over, under):
    if x is not None:
        assert cal(np.mean(final > x)) >= lv > cal(np.mean(final > x + 1)), (lv, x)
    if y is not None:
        assert cal(np.mean(final < y)) >= lv > cal(np.mean(final < y - 1)), (lv, y)

# Projection above the line leans over, below leans under.
hi, mid, lo = (m.ladder(p, 60.5)["p_over"] for p in (90, 60.5, 30))
assert hi > mid > lo, (hi, mid, lo)

print("ladder checks passed")
