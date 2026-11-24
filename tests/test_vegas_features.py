"""
Checks for the Vegas-line context features (implied_team_total, team_spread_line):
the sign convention, the missing-line path and the column order. That models
saved before these features still predict is test_predict_next_game_refactor.py's
job (it runs one).

Synthetic, no network, no database.

Run: python3 test_vegas_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import PlayerProjections as pp


def demo_implied_totals_follow_the_spread():
    """A 47.5 total with the home side favored by 3.5: home expects 25.5, away 22.
    A game with no line posted yet stays NaN for fillna_context() to fill."""
    sc = pd.DataFrame({
        "season": [2026] * 4, "week": [6, 6, 7, 7], "team": ["KC", "LV", "KC", "LV"], "opponent": ["LV", "KC", "LV", "KC"],
        "is_home": [1, 0, 0, 1], "rest_days": 7, "opponent_rest_days": 7, "div_game": 1, "roof": "outdoors",
        "temp": 60.0, "wind": 5.0, "starting_qb_id": "q",
        "total_line": [47.5, 47.5, None, None], "team_spread_line": [3.5, -3.5, None, None],
    })
    out = pp.build_schedule_features(sc)
    assert out["implied_team_total"].tolist()[:2] == [25.5, 22.0], out["implied_team_total"].tolist()
    assert out["team_spread_line"].tolist()[:2] == [3.5, -3.5]
    assert out["implied_team_total"].iloc[2:].isna().all(), "no line yet must stay missing, not become 0"
    filled, fills = pp.fillna_context(out.assign(**{c: 0.0 for c in pp.TEAM_CONTEXT_COLUMNS if c not in out}))
    assert not filled["implied_team_total"].isna().any() and "implied_team_total" in fills
    print("OK -- implied totals split the total by the spread, and a missing line is filled, not zeroed.")


def demo_context_order_matches():
    """The live vector is assembled by hand; its length must match the column list."""
    assert pp.CONTEXT_COLUMNS[len(pp.TEAM_CONTEXT_COLUMNS) - 2:len(pp.TEAM_CONTEXT_COLUMNS)] == pp.VEGAS_CONTEXT_COLUMNS
    assert len(pp.CONTEXT_COLUMNS) == 31
    print("OK -- Vegas features close the team block of CONTEXT_COLUMNS.")


if __name__ == "__main__":
    demo_implied_totals_follow_the_spread()
    demo_context_order_matches()
