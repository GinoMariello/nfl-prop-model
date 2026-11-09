"""Offline regression tests: .venv/bin/python -m unittest discover -s tests"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.team_game_stats import build_team_games, build_team_game_stats
from src.features.rolling_features import (
    STAT_COLUMNS, TEAM_FEATURE_COLUMNS, add_rolling_features, build_matchup_table,
    exclusion_report, normalized_home_probability, run, select_features,
)


def games_fixture(n=12):
    return pd.DataFrame({
        "game_id": [f"g{i:02}" for i in range(n)],
        "season": [2022] * 2 + [2023] * (n - 2),
        "week": list(range(n, 0, -1)),  # Deliberately wrong chronological order.
        "game_type": "REG", "gameday": pd.date_range("2023-01-01", periods=n, freq="7D").strftime("%Y-%m-%d"),
        "home_team": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "away_team": ["B" if i % 2 == 0 else "A" for i in range(n)],
        "home_score": 21., "away_score": 14.,
        "home_moneyline": -150., "away_moneyline": 130.,
    })


def stats_fixture(games):
    rows = build_team_games(games)
    for column in STAT_COLUMNS:
        if column not in rows:
            rows[column] = rows.game_id.str[1:].astype(float) + rows.team.eq("B") * 100
    return rows


def matchup(games, stats=None):
    return build_matchup_table(add_rolling_features(stats_fixture(games) if stats is None else stats), games)


class FeatureTests(unittest.TestCase):
    def test_reference_window_order_seasons_missing_metrics(self):
        games = games_fixture()
        stats = stats_fixture(games)
        stats.loc[stats.game_id.eq("g09"), "off_epa_per_play"] = np.nan
        actual = add_rolling_features(stats.sample(frac=1, random_state=4))
        for team in ("A", "B"):
            history = stats.loc[stats.team.eq(team)].sort_values("gameday")
            got = actual.loc[actual.team.eq(team)].reset_index(drop=True)
            for i in range(len(history)):
                expected = history.iloc[max(0, i - 8):i].off_epa_per_play.mean()
                np.testing.assert_allclose(got.loc[i, "roll_off_epa_per_play"], expected, equal_nan=True)
                self.assertEqual(got.loc[i, "games_played_before"], i)
        assert_frame_equal(actual, add_rolling_features(stats))

    def test_current_and_future_corruption_through_both_sides(self):
        games = games_fixture()
        stats = stats_fixture(games)
        baseline = matchup(games, stats)
        corrupted = stats.copy()
        mask = corrupted.game_id.ge("g02")
        corrupted.loc[mask, STAT_COLUMNS] = 999999.
        changed_games = games.copy()
        changed_games.loc[changed_games.game_id.ge("g02"), ["home_score", "away_score"]] = [100, 0]
        changed = matchup(changed_games, corrupted)
        assert_frame_equal(select_features(baseline).iloc[:3], select_features(changed).iloc[:3])
        for side in ("home", "away"):
            self.assertNotEqual(baseline.loc[3, f"{side}_roll_off_epa_per_play"],
                                changed.loc[3, f"{side}_roll_off_epa_per_play"])

    def test_placeholders_postseason_inprogress_and_ties(self):
        games = games_fixture()
        games["completed"] = True
        games.loc[2, "completed"] = False  # Even a non-null live score is not final.
        games.loc[3, "home_score"] = np.nan
        games.loc[4, "game_type"] = "POST"
        games.loc[5, "game_type"] = "PRE"
        games.loc[6, "away_score"] = 21  # Completed ties do count as history.
        result = matchup(games).set_index("game_id")
        self.assertTrue(result.loc[["g02", "g03", "g06"], "home_win"].isna().all())
        self.assertNotIn("g04", result.index)
        self.assertEqual(result.loc["g07", "home_games_played_before"], 3)
        reduced = games.drop(index=[2, 3, 4, 5])
        expected = matchup(reduced).set_index("game_id")
        assert_frame_equal(select_features(result).loc[expected.index], select_features(expected))

    def test_same_day_and_unknown_dates(self):
        games = games_fixture()
        games.loc[1, "gameday"] = games.loc[0, "gameday"]
        games.loc[2, "gameday"] = "01/10/2023"
        result = matchup(games).set_index("game_id")
        self.assertEqual(result.loc["g01", "home_games_played_before"], 0)
        self.assertTrue(pd.isna(result.loc["g02", "home_games_played_before"]))
        self.assertEqual(result.loc["g03", "home_games_played_before"], 2)
        report = exclusion_report(games)
        self.assertEqual(report.loc[report.reason.eq("ambiguous_date"), "count"].sum(), 1)

    def test_ingestion_schema_works_without_completion_column(self):
        games = games_fixture()
        self.assertNotIn("completed", games)
        self.assertNotIn("completed", build_team_games(games))
        result = matchup(games)
        self.assertTrue(result.home_win.notna().all())
        self.assertEqual(result.home_games_played_before.tolist(), list(range(len(games))))
        report = exclusion_report(games)
        self.assertEqual(report.query("reason == 'score_proxy_used'")["count"].sum(), len(games))

    def test_explicit_unknown_status_is_not_replaced_by_proxy(self):
        games = games_fixture()
        games["completed"] = pd.NA
        result = matchup(games)
        self.assertTrue(result.home_win.isna().all())
        self.assertTrue(result.home_games_played_before.eq(0).all())

    def test_invalid_scores_are_excluded(self):
        games = games_fixture()
        games.loc[:3, "home_score"] = [np.inf, -1, .5, np.nan]
        result = matchup(games)
        self.assertTrue(result.home_win.iloc[:4].isna().all())
        self.assertEqual(result.loc[4, "home_games_played_before"], 0)

    def test_missing_allowlisted_column_raises(self):
        with self.assertRaises(KeyError):
            select_features(pd.DataFrame({TEAM_FEATURE_COLUMNS[0]: [1.]}))

    def test_realistic_offseason_and_all_future_history(self):
        games = games_fixture(2)
        games["season"] = [2023, 2024]
        games["gameday"] = ["2024-01-07", "2024-09-08"]
        result = matchup(games)
        self.assertEqual(result.loc[1, "home_games_played_before"], 1)
        games[["home_score", "away_score"]] = np.nan
        result = matchup(games)
        self.assertTrue(result.home_games_played_before.eq(0).all())
        self.assertTrue(select_features(result).filter(like="roll_").isna().all().all())

    def test_same_day_final_state_and_strict_dates(self):
        games = games_fixture()
        games.loc[:8, "gameday"] = "2023-01-01"
        stats = stats_fixture(games)
        result = add_rolling_features(stats)
        shuffled = add_rolling_features(stats.sample(frac=1, random_state=7))
        assert_frame_equal(result, shuffled)
        row = result.loc[result.game_id.eq("g09") & result.team.eq("A")].iloc[0]
        self.assertEqual(row.games_played_before, 9)
        self.assertEqual(row.roll_off_epa_per_play, 4.5)  # last eight: 1..8
        for date in ("2023-1-01", "2023-01-01 13:00:00", "01/02/2023"):
            games.loc[0, "gameday"] = date
            self.assertTrue(pd.isna(matchup(games).loc[0, "home_win"]))

    def test_labels_allowlist_and_market(self):
        games = games_fixture()
        games.loc[0, "away_score"] = 30
        games.loc[1, "away_score"] = np.nan
        games.loc[2, "home_moneyline"] = 0
        games.loc[3, "away_moneyline"] = np.inf
        games.loc[4, "away_moneyline"] = -99
        games.loc[5, "home_moneyline"] = np.nan
        result = matchup(games)
        self.assertEqual(result.loc[0, "home_win"], 0)
        self.assertTrue(pd.isna(result.loc[1, "home_win"]))
        self.assertEqual(result.loc[2, "home_win"], 1)
        result["future_target"] = 1234
        self.assertEqual(tuple(select_features(result).columns), TEAM_FEATURE_COLUMNS)
        self.assertTrue(select_features(result).iloc[0].filter(like="roll_").isna().all())
        market = select_features(result, "team_market")
        self.assertEqual(market.index.tolist(), [0, 1, 6, 7, 8, 9, 10, 11])
        self.assertAlmostEqual(result.loc[0, "market_home_probability"], .6 / (.6 + 100 / 230))
        games[["home_moneyline", "away_moneyline"]] = [100, -100]
        self.assertTrue(normalized_home_probability(games).eq(.5).all())
        with self.assertRaises(ValueError):
            select_features(result, "unknown")

    def test_bad_keys_and_sides_rejected(self):
        games = games_fixture()
        with self.assertRaises(ValueError):
            build_team_games(pd.concat([games, games.iloc[:1]]))
        stats = stats_fixture(games)
        with self.assertRaises(ValueError):
            add_rolling_features(pd.concat([stats, stats.iloc[:1]]))
        features = add_rolling_features(stats)
        with self.assertRaises(ValueError):
            build_matchup_table(features.iloc[1:], games)
        features.loc[0, "is_home"] = not features.loc[0, "is_home"]
        with self.assertRaises(ValueError):
            build_matchup_table(features, games)
        with self.assertRaises(ValueError):
            add_rolling_features(stats, window=0)

    def test_relocations(self):
        games = games_fixture()
        games["home_team"] = "STL"
        games["away_team"] = "SD"
        games.loc[2:, ["home_team", "away_team"]] = ["LA", "LAC"]
        rows = build_team_games(games)
        self.assertEqual(set(rows.team), {"LA", "LAC"})
        stats = rows.copy()
        for col in STAT_COLUMNS:
            if col not in stats:
                stats[col] = 1.
        result = matchup(games, stats)
        self.assertEqual(result.loc[2, "home_games_played_before"], 2)
        games["home_team"] = "OAK"
        games.loc[2:, "home_team"] = "LV"
        self.assertEqual(matchup(games).loc[2, "home_games_played_before"], 2)

    def test_raw_plays_to_matchup(self):
        games = games_fixture()
        plays = pd.DataFrame([
            {"game_id": "g00", "posteam": offense, "defteam": defense,
             "play_type": "pass", "epa": epa, "success": 1., "pass": 1,
             "rush": 0, "yards_gained": 10., "down": 3,
             "third_down_converted": 1., "yardline_100": 10.,
             "touchdown": 1., "interception": 0., "fumble_lost": 0., "sack": 0.}
            for offense, defense, epa in [("A", "B", 2.), ("B", "A", -1.)]
        ])
        stats = build_team_game_stats(games, plays)
        result = matchup(games, stats)
        self.assertEqual(len(stats), 2 * len(games))
        self.assertEqual(result.loc[1, "away_roll_off_epa_per_play"], 2.)
        self.assertEqual(result.loc[1, "home_roll_def_epa_per_play"], 2.)
        self.assertEqual(result.loc[1, "away_roll_points_for"], 21.)
        self.assertEqual(result.loc[2, "home_games_played_before"], 2)
        # Eight missing box scores eventually evict the only observed stats.
        self.assertTrue(pd.isna(result.loc[9, "home_roll_off_epa_per_play"]))

    def test_sqlite_rebuild_and_raw_preservation(self):
        games = games_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "games.db")
            with sqlite3.connect(path) as conn:
                games.to_sql("games", conn, index=False)
                # Missing an entire box score still counts as a completed game.
                stats_fixture(games).query("game_id != 'g01'").to_sql("team_game_stats", conn, index=False)
            first = run(path)
            second = run(path)
            assert_frame_equal(first, second)
            self.assertEqual(first.loc[2, "home_games_played_before"], 2)
            with sqlite3.connect(path) as conn:
                raw = pd.read_sql_query("SELECT * FROM games", conn)
                self.assertEqual(len(pd.read_sql_query("SELECT * FROM feature_exclusions", conn)), 20)
            assert_frame_equal(raw, games)

    def test_empty_regular_schedule(self):
        games = games_fixture()
        games["game_type"] = "POST"
        self.assertTrue(matchup(games).empty)


if __name__ == "__main__":
    unittest.main()
