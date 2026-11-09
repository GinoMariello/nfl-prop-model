"""
Sanity check for src/features/team_game_stats.py. Not a leakage test (see
that module's docstring -- there's nothing rolling yet to leak). Checks two
kinds of things on one real season:

1. League-wide conservation invariants that must hold regardless of any
   aggregation bug: every giveaway is someone's takeaway, every sack
   allowed is someone's sack made, and every point scored is someone's
   point allowed.
2. Rates are valid probabilities, and a couple of well-known final scores
   from the 2023 season come out right (independent proof the box scores
   attach to the correct team/game, not just internally consistent).

Run: python3 tests/test_team_game_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from StatScraper import query_sqlite
from src.features.team_game_stats import build_team_game_stats

SEASON = 2023


def demo():
    # Full history, not just one season: the relocated-franchise abbreviation
    # mismatch (STL/SD/OAK in games vs LA/LAC/LV in plays) only shows up
    # across seasons, never within one -- a single-season test missed it.
    games = query_sqlite("SELECT * FROM games", db_path="data/nfl_games.db")
    plays = query_sqlite("SELECT * FROM plays", db_path="data/nfl_games.db")
    df_all = build_team_game_stats(games, plays)

    assert df_all["off_turnovers"].sum() == df_all["def_takeaways"].sum(), \
        "giveaways != takeaways across full history -- check for team-abbreviation mismatches"
    assert df_all["off_sacks_allowed"].sum() == df_all["def_sacks"].sum(), \
        "sacks allowed != sacks made across full history -- check for team-abbreviation mismatches"
    # Only check completed games -- future/unplayed games have no plays yet
    # by definition, that's correct, not a bug.
    completed = df_all[df_all["points_for"].notna()]
    null_rate = completed["off_epa_per_play"].isna().mean()
    # nflverse itself has zero pbp rows for a handful of 1999-2000 games
    # (verified: 1999_01_BAL_STL, 2000_03_SD_KC, 2000_06_BUF_MIA -- a real
    # source gap, not a merge bug). Threshold allows for that, not more.
    assert null_rate < 0.001, f"{null_rate:.2%} of completed team-games have no offensive stats -- likely a merge/key mismatch"

    assert len(df_all) == 2 * len(games), "expected exactly 2 rows (one per team) per game"
    df = df_all[df_all["season"] == SEASON]

    assert not df.duplicated(subset=["game_id", "team"]).any(), "duplicate (game_id, team) rows"

    # Conservation invariants -- true by definition of the stat, independent of this code.
    assert df["off_turnovers"].sum() == df["def_takeaways"].sum(), "giveaways != takeaways league-wide"
    assert df["off_sacks_allowed"].sum() == df["def_sacks"].sum(), "sacks allowed != sacks made league-wide"
    assert df["points_for"].sum() == df["points_against"].sum(), "points for != points against league-wide"

    for col in ["off_success_rate", "off_third_down_rate", "off_redzone_td_rate",
                "def_success_rate", "def_third_down_rate", "def_redzone_td_rate"]:
        valid = df[col].dropna()
        assert valid.between(0, 1).all(), f"{col} has values outside [0, 1]"

    # Independent ground truth: real final scores from the 2023 opener.
    kc = df[(df["team"] == "KC") & (df["opponent"] == "DET")].iloc[0]
    assert (kc["points_for"], kc["points_against"]) == (20, 21), "KC-DET week 1 2023 score mismatch"
    atl = df[(df["team"] == "ATL") & (df["opponent"] == "CAR")].iloc[0]
    assert (atl["points_for"], atl["points_against"]) == (24, 10), "ATL-CAR week 1 2023 score mismatch"

    print(f"OK -- {len(df)} team-game rows for {SEASON}, conservation invariants and known scores check out.")


if __name__ == "__main__":
    demo()
