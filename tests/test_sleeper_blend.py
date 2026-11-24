"""
Checks for the Sleeper blend: parse_sleeper_projections() maps ids and stat names
and drops what it cannot key, and blend_sources() takes the weighted mean of the
sources a player has, rescaling when one is missing.

Synthetic, no network, no database.

Run: python3 tests/test_sleeper_blend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import PlayerProjections as pp
from StatScraper import parse_sleeper_projections


def demo_parse():
    records = [
        {"player_id": "123", "player": {"position": "WR"}, "last_modified": 1759809647014,
         "stats": {"pts_ppr": 14.1, "rec": 4.5, "rec_yd": 64.0}},
        {"player_id": "999", "player": {"position": "WR"}, "stats": {"pts_ppr": 3.0}},  # no gsis id
        {"player_id": "123", "player": {"position": "WR"}, "stats": {"rec": 2.0}},       # no PPR projection
    ]
    d = parse_sleeper_projections(records, {"123": "00-0031234"}, 2026, 6, "2026-10-12T00:00:00+00:00")
    assert len(d) == 1 and d.player_id.iloc[0] == "00-0031234"
    assert d.sleeper_fantasy_points_ppr.iloc[0] == 14.1 and d.sleeper_receiving_yards.iloc[0] == 64.0
    assert d.sleeper_receiving_tds.iloc[0] == 0.0, "a stat Sleeper omits is a projected zero"
    assert d.last_modified.iloc[0].startswith("2025-10-07") and d.retrieved_at.iloc[0] == "2026-10-12T00:00:00+00:00"
    print("OK -- Sleeper records map to gsis ids and project column names; unkeyable ones are dropped.")


def demo_blend():
    ours = pd.DataFrame({"player_id": ["a", "b", "c"], "pred_receptions": [4.0, 6.0, 5.0]})
    sleeper = pd.DataFrame({"player_id": ["a", "b"], "sleeper_receptions": [8.0, 2.0]})
    gbm = pd.DataFrame({"player_id": ["a"], "gbm_receptions": [6.0]})
    out = pp.blend_sources(ours, {"sleeper": sleeper, "gbm": gbm}, {"sleeper": 0.5, "own": 0.0, "gbm": 0.5})
    assert out.pred_receptions.tolist() == [7.0, 2.0, 5.0], out.pred_receptions.tolist()  # a: both; b: sleeper only; c: own fallback
    assert out.own_receptions.tolist() == [4.0, 6.0, 5.0]
    out = pp.blend_sources(ours, {"sleeper": sleeper}, {"sleeper": 0.25, "own": 0.75})
    assert out.pred_receptions.tolist() == [5.0, 5.0, 5.0], out.pred_receptions.tolist()
    print("OK -- blend_sources() weights what each player has, rescales around gaps, and falls back to ours.")



def demo_mean_columns():
    out = pd.DataFrame({"player_id": ["a"], "pred_receiving_yards": [50.0], "pred_wopr": [0.4]})
    m = pp.add_mean_columns(out, "WR")
    assert m.mean_receiving_yards.iloc[0] == 50.0 * pp.MEAN_RATIOS["WR"]["receiving_yards"]
    assert "mean_wopr" not in m and "pred_wopr" in m, "share stats are not totals and get no mean_"
    assert not any(c.startswith("mean_") for c in pp.add_mean_columns(out, "QB")), "QB is not corrected"
    print("OK -- mean_ columns scale RB/WR/TE counting stats for totals and leave shares and QB alone.")


if __name__ == "__main__":
    demo_parse()
    demo_blend()
    demo_mean_columns()
