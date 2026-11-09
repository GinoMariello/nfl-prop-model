"""Optional real-data regression, adapted from the earlier cached rolling test.

Run: .venv/bin/python tests/test_cached_features.py --db data/nfl_games.db
Reads the supplied database in read-only mode; rebuilds only a temporary copy.
"""
import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.rolling_features import (
    STAT_COLUMNS, add_rolling_features, build_matchup_table, run, select_features,
)
from src.features.team_game_stats import build_team_games


def demo(db_path="data/nfl_games.db"):
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        games = pd.read_sql_query("SELECT * FROM games", conn)
        stats = pd.read_sql_query("SELECT * FROM team_game_stats", conn)
    assert "completed" not in games, "This regression exercises the existing ingestion schema"
    # Exercise the complete persisted-input path, without touching source tables.
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "snapshot.db")
        with sqlite3.connect(path) as conn:
            games.to_sql("games", conn, index=False)
            stats.to_sql("team_game_stats", conn, index=False)
        first = run(path)
        assert_frame_equal(first, run(path))
        with sqlite3.connect(path) as conn:
            assert_frame_equal(games, pd.read_sql_query("SELECT * FROM games", conn))
            assert_frame_equal(stats, pd.read_sql_query("SELECT * FROM team_game_stats", conn))
            # Cross-table invariant: exclusion_report counts supervised_eligible
            # straight from `games`, while the label count comes from the merged
            # matchup table. Two independent paths to the same number, so a
            # drift between them means one of the two is wrong.
            report = pd.read_sql_query("SELECT * FROM feature_exclusions", conn)
        supervised_eligible = int(report.loc[report.reason.eq("supervised_eligible"), "count"].sum())
        assert supervised_eligible == int(first.home_win.notna().sum()), (
            f"feature_exclusions supervised_eligible ({supervised_eligible}) must match "
            f"matchup_features' labeled rows ({int(first.home_win.notna().sum())})")

    regular = games.loc[games.game_type.eq("REG")]
    assert set(first.game_id) == set(regular.game_id)
    scores = regular[["home_score", "away_score"]]
    expected_labels = (np.isfinite(scores).all(axis=1)
                       & regular.home_score.ne(regular.away_score)
                       & pd.to_datetime(regular.gameday, format="%Y-%m-%d", errors="coerce").notna())
    assert first.home_win.notna().sum() == expected_labels.sum() > 0
    assert first.home_games_played_before.gt(0).any()

    metrics = [c for c in STAT_COLUMNS if c not in ("points_for", "points_against")]
    rows = build_team_games(games).merge(stats[["game_id", "team", *metrics]],
                                        on=["game_id", "team"], how="left", validate="one_to_one")
    baseline = add_rolling_features(rows)
    # Independent, deliberately simple reference, ordered by dates rather than weeks.
    kc = rows.loc[rows.team.eq("KC") & rows.game_type.eq("REG")].sort_values(["gameday", "game_id"])
    computed = baseline.loc[baseline.team.eq("KC")].set_index("game_id")
    for row in kc.itertuples():
        prior = kc.loc[(kc.gameday < row.gameday) & kc.points_for.notna() & kc.points_against.notna()]
        assert computed.loc[row.game_id, "games_played_before"] == len(prior)
        np.testing.assert_allclose(computed.loc[row.game_id, "roll_off_epa_per_play"],
                                   prior.tail(8).off_epa_per_play.mean(), equal_nan=True)
    # Retain the real-data corruption proof, through both home/away reshaping.
    target = kc.loc[kc.points_for.notna() & kc.points_against.notna()].iloc[3]
    corrupted = rows.copy()
    corrupted.loc[corrupted.game_id.eq(target.game_id) & corrupted.team.eq("KC"), "off_epa_per_play"] = 999999.
    after = build_matchup_table(add_rolling_features(corrupted), games).set_index("game_id")
    before = first.set_index("game_id")
    earlier_ids = regular.loc[regular.gameday.le(target.gameday), "game_id"]
    assert_frame_equal(select_features(before).loc[earlier_ids], select_features(after).loc[earlier_ids])
    next_game = kc.loc[kc.gameday.gt(target.gameday)].iloc[0]
    side = "home" if next_game.is_home else "away"
    column = f"{side}_roll_off_epa_per_play"
    assert before.loc[next_game.game_id, column] != after.loc[next_game.game_id, column]
    print(f"OK: {len(games)} raw games, {len(first)} REG matchups, "
          f"{first.home_win.notna().sum()} labels, {len(kc)} KC reference rows; "
          "real-data corruption, repeatability, and raw-table preservation passed.")


def check_market_coverage_bounds_the_team_market_window(games):
    """The market variant cannot start where the plan says the window starts.

    model-plan.md §5 sets development outer seasons 2005-2024 and, for each
    outer season T, fits on 1999..T-2 and validates on T-1. `team_market` fits
    only on paired-quote rows -- and the source carries no moneylines at all
    before 2006, so the first three outer seasons have a training fold with
    zero quoted games and `fit_model` raises by design.

    Pinning it here means Stage 4's fold loop is written around a measured
    fact, and that a future re-ingest which backfills early odds shows up as a
    test failure rather than a silently wider window.
    """
    reg = games.loc[games.game_type.eq("REG")].copy()
    quotes = reg[["home_moneyline", "away_moneyline"]].apply(pd.to_numeric, errors="coerce")
    reg["quoted"] = quotes.notna().all(axis=1)

    per_season = reg.groupby("season").quoted.agg(["sum", "size"])
    early = per_season.loc[per_season.index <= 2005, "sum"]
    assert early.sum() == 0, (
        f"expected no paired moneylines before 2006, found {int(early.sum())} -- "
        f"if odds were backfilled, team_market's window moves and §5 needs revisiting")

    # First outer season whose training fold (1999..T-2) holds any quoted game.
    quoted_seasons = set(per_season.index[per_season["sum"] > 0])
    first_fittable = next(T for T in range(2005, 2025)
                          if quoted_seasons & set(range(1999, T - 1)))
    assert first_fittable == 2008, (
        f"team_market's first fittable outer season is {first_fittable}, not 2008")

    team_only_seasons = len(range(2005, 2025))
    team_market_seasons = len(range(first_fittable, 2025))
    assert (team_only_seasons, team_market_seasons) == (20, 17)

    # model-plan.md §6 forbids comparing one model's full sample against
    # another's restricted sample, so the head-to-head lives on the overlap.
    print(f"OK: no paired moneylines before 2006; team_only gets {team_only_seasons} outer "
          f"seasons (2005-2024), team_market {team_market_seasons} ({first_fittable}-2024); "
          f"matched comparison is {first_fittable}-2024.")


def check_neutral_site_rows_are_flagged(games):
    """The 58 games whose "home" team has no home field are marked, not silently
    scored as home games. See rolling_features._neutral_site."""
    reg = games.loc[games.game_type.eq("REG")]
    window = reg.loc[reg.season.between(2005, 2024)]
    assert set(games.location.dropna().unique()) <= {"Home", "Neutral"}, \
        "an unclassified location value would otherwise read as a home game"

    neutral = window.location.eq("Neutral")
    assert int(neutral.sum()) == 58, f"expected 58 neutral-site REG games, found {int(neutral.sum())}"

    scores = window[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
    decided = scores.notna().all(axis=1) & scores.home_score.ne(scores.away_score)
    home_win = scores.home_score.gt(scores.away_score)
    at_neutral = home_win[decided & neutral].mean()
    at_home = home_win[decided & ~neutral].mean()
    assert at_neutral < at_home, (
        "the nominal home team is not expected to hold a home-field edge at a neutral site; "
        f"got {at_neutral:.4f} neutral vs {at_home:.4f} home")

    print(f"OK: {int(neutral.sum())} neutral-site REG games flagged; home-win rate "
          f"{at_neutral:.4f} there vs {at_home:.4f} at real home sites "
          f"(n={int((decided & neutral).sum())}, so direction only, not a measured effect).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/nfl_games.db")
    db = parser.parse_args().db
    demo(db)
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        raw_games = pd.read_sql_query("SELECT * FROM games", conn)
    check_market_coverage_bounds_the_team_market_window(raw_games)
    check_neutral_site_rows_are_flagged(raw_games)
