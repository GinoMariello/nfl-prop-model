"""
Checks site_data.live_p_over, the in-game P(over) the Live tab mirrors in JS.
Synthetic history, no network or database.

    python3 test_live_p_over.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from site_data import OverModel, live_p_over

rng = np.random.default_rng(0)
pred = rng.uniform(20, 120, 2000)
hist = pd.DataFrame({"pred_receiving_yards": pred, "actual_receiving_yards": np.maximum(0, pred + rng.normal(0, 25, 2000))})
m = OverModel(hist, "receiving_yards", shrink=0.8)
live = m.live()

# Pregame (whole game left): the 100 quantiles reproduce the full-sample p_over.
for proj, line in [(40, 35.5), (60, 70.5), (100, 80.5), (75, 75.5)]:
    assert abs(live_p_over(live, proj, line, 0, 1.0) - m.p_over(proj, line)) < 0.03, (proj, line)

# Game over: certain, no shrink left.
assert live_p_over(live, 60, 70.5, 71, 0.0) == 1.0
assert live_p_over(live, 60, 70.5, 70, 0.0) == 0.0

# More yards so far never lowers the chance.
ps = [live_p_over(live, 60, 70.5, c, 0.5) for c in range(0, 80, 5)]
assert all(a <= b for a, b in zip(ps, ps[1:])), ps
# Less time left hurts a player short of the line; one already past it is near certain.
assert live_p_over(live, 60, 70.5, 65, 0.1) < live_p_over(live, 60, 70.5, 65, 0.5)
assert live_p_over(live, 60, 70.5, 75, 0.5) == 1.0

print("live_p_over checks passed")
