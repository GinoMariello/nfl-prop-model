"""
A second, independent projection model: gradient-boosted trees (scikit-learn's
HistGradientBoostingRegressor) on recent form, usage share, opponent and Vegas
features, retrained from scratch for every (season, week) on all earlier games.
Shares no modelling code with PlayerProjections.py -- only the raw SQLite tables
StatScraper.py fills -- so its errors are its own, which is what makes it worth
blending (see PlayerProjections.BLEND_WEIGHTS).

Leak rule: the week being projected enters only as empty "target" rows (player,
team, opponent, Vegas line); every player/opponent feature is computed from games
strictly before it (shift(1) before any ewm/rolling), and nothing from that week
or later is in the training set.

Run directly for a self-check: python3 -m GbmProjections
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FORM = ["fantasy_points_ppr", "targets", "receptions", "receiving_yards", "carries", "rushing_yards",
        "attempts", "passing_yards", "passing_tds", "target_share", "wopr", "air_yards_share"]
FEATURES = ([f"ewm3_{c}" for c in FORM] + [f"last_{c}" for c in FORM]
            + ["l8_fantasy_points_ppr", "l8_targets", "l8_carries", "l8_attempts",
               "games_before", "same_team_as_last", "opp_allowed_l8",
               "is_home", "total_line", "team_spread_line", "implied_team_total"])


def build_features(history: pd.DataFrame, schedule: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """
    Feature rows for `history` (completed games) plus `targets` (season, week,
    player_id, team, opponent_team, position_group, player_name -- no stats).
    Each row's features use only rows before it for that player/opponent.
    """
    rows = pd.concat([history.assign(is_target=False), targets.assign(is_target=True)], ignore_index=True)
    rows["t"] = rows.season * 100 + rows.week
    rows = rows.sort_values(["player_id", "t"]).reset_index(drop=True)
    for col in FORM:
        rows[col] = rows[col].astype(float)
        rows.loc[~rows.is_target, col] = rows.loc[~rows.is_target, col].fillna(0.0)

    by_player = rows.groupby("player_id")
    for col in FORM:
        prev = by_player[col].shift(1)
        rows[f"ewm3_{col}"] = prev.groupby(rows.player_id).transform(lambda s: s.ewm(halflife=3, min_periods=1).mean())
        rows[f"last_{col}"] = prev
    for col in ["fantasy_points_ppr", "targets", "carries", "attempts"]:
        prev = by_player[col].shift(1)
        rows[f"l8_{col}"] = prev.groupby(rows.player_id).transform(lambda s: s.rolling(8, min_periods=1).mean())
    rows["games_before"] = by_player.cumcount()
    rows["same_team_as_last"] = (by_player["team"].shift(1) == rows.team).astype(float)

    # PPR allowed by the opponent to this position group, trailing 8 games, strictly before.
    allowed = (rows[~rows.is_target]
               .groupby(["opponent_team", "position_group", "t"], as_index=False)["fantasy_points_ppr"].sum())
    upcoming = rows.loc[rows.is_target, ["opponent_team", "position_group", "t"]].drop_duplicates()
    allowed = pd.concat([allowed, upcoming.assign(fantasy_points_ppr=np.nan)], ignore_index=True)
    allowed = allowed.drop_duplicates(["opponent_team", "position_group", "t"]).sort_values("t")
    grouped = allowed.groupby(["opponent_team", "position_group"])["fantasy_points_ppr"]
    allowed["opp_allowed_l8"] = grouped.shift(1).groupby([allowed.opponent_team, allowed.position_group]).transform(
        lambda s: s.rolling(8, min_periods=1).mean())
    rows = rows.merge(allowed[["opponent_team", "position_group", "t", "opp_allowed_l8"]],
                      on=["opponent_team", "position_group", "t"], how="left")

    rows = rows.merge(schedule[["season", "week", "team", "is_home", "total_line", "team_spread_line"]],
                      on=["season", "week", "team"], how="left")
    rows["implied_team_total"] = (rows.total_line + rows.team_spread_line) / 2
    return rows


def gbm_project_week(history: pd.DataFrame, schedule: pd.DataFrame, position_group: str, season: int, week: int,
                     targets: List[Tuple[str, str]], stat_columns: List[str], loss: str = "absolute_error") -> pd.DataFrame:
    """
    Projections for `targets` [(player_id, opponent), ...] in (season, week):
    one model per stat, trained on every earlier game of `position_group`.
    `history` is player_weekly_stats rows; anything from this week on is ignored.
    Returns player_id, player_name, pred_<stat>. Players with no earlier game are
    skipped (nothing to build features from).

    loss="absolute_error" (default) targets the typical (median) game, which sits
    below the average for skewed stats like yards and fantasy points -- so it
    projects lower than Sleeper and PlayerProjections, which target the average.
    It stays the default because it blends better: on 2026 weeks 2-5 the blend
    missed PPR by less at every position with it than with loss="poisson" (the
    average game), whose re-fitted blend weights fell to 0-0.1. It is also the
    right comparison for an over/under prop line, which is won at the median.
    With "poisson", a stat that can go negative (fantasy points, rushing yards)
    falls back to squared error, which also targets the average.
    """
    cut = season * 100 + week
    history = history[(history.season * 100 + history.week < cut) & (history.position_group == position_group)]
    last = history.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
    keep = [(p, o) for p, o in targets if p in last.index]
    if not keep:
        return pd.DataFrame(columns=["player_id", "player_name"] + [f"pred_{s}" for s in stat_columns])
    target_rows = pd.DataFrame({
        "season": season, "week": week, "position_group": position_group,
        "player_id": [p for p, _ in keep], "opponent_team": [o for _, o in keep],
        "team": [last.at[p, "team"] for p, _ in keep], "player_name": [last.at[p, "player_name"] for p, _ in keep],
    })
    rows = build_features(history, schedule, target_rows)
    train = rows[~rows.is_target & (rows.games_before >= 1)]
    test = rows[rows.is_target]
    out = test[["player_id", "player_name"]].copy()
    for stat in stat_columns:
        y = train[stat].fillna(0.0)
        stat_loss = "squared_error" if loss == "poisson" and (y < 0).any() else loss
        model = HistGradientBoostingRegressor(loss=stat_loss, max_iter=300, learning_rate=0.05,
                                              min_samples_leaf=40, random_state=0)
        model.fit(train[FEATURES], y)
        out[f"pred_{stat}"] = np.clip(model.predict(test[FEATURES]), 0, None)
    return out.reset_index(drop=True)


def _self_check() -> None:
    """A target row's features must not change when its own week's real stats exist in history."""
    rng = np.random.default_rng(0)
    n = 12
    hist = pd.DataFrame({
        "season": 2025, "week": np.tile(np.arange(1, n + 1), 2), "player_id": np.repeat(["a", "b"], n),
        "player_name": np.repeat(["A", "B"], n), "team": np.repeat(["KC", "LV"], n),
        "opponent_team": np.repeat(["LV", "KC"], n), "position_group": "WR",
    })
    for col in FORM:
        hist[col] = rng.uniform(0, 10, len(hist))
    sched = pd.DataFrame({"season": 2025, "week": np.tile(np.arange(1, n + 1), 2), "team": np.repeat(["KC", "LV"], n),
                          "is_home": 1.0, "total_line": 45.0, "team_spread_line": 0.0})
    targets = [("a", "LV"), ("b", "KC")]
    clean = gbm_project_week(hist[hist.week < n], sched, "WR", 2025, n, targets, ["receptions"])
    leaked = hist.copy()
    leaked.loc[leaked.week == n, FORM] = 999.0  # the projected week's real stats, wildly off
    dirty = gbm_project_week(leaked, sched, "WR", 2025, n, targets, ["receptions"])
    assert np.allclose(clean.pred_receptions, dirty.pred_receptions), "the projected week's own stats leaked in"
    assert gbm_project_week(hist, sched, "WR", 2025, n, [("zz", "KC")], ["receptions"]).empty
    print("OK -- GBM projections ignore the projected week's own stats and skip players with no history.")


if __name__ == "__main__":
    _self_check()
