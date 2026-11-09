"""
Per-team, per-game box scores, built from raw plays + games (src/data/ingest.py).

One row per (game_id, team): that team's own offensive numbers for the game
(EPA/play, success rate, third-down rate, red-zone TD rate, sacks taken,
giveaways) and what their defense allowed the opponent (same stats, mirrored).
Long format -- two rows per game, one per team -- because the next step
(rolling pregame features) needs a per-team time series to run a trailing
window over.

*** NOT LEAK-SAFE ON ITS OWN ***
A row here describes a team's performance IN that game -- using it as a
feature to predict that same game's outcome would be using the answer to
predict itself. It only becomes a safe model input after the next step
computes a *trailing* average per team using strictly earlier rows (shift-by-
one before rolling). Do not wire this table straight into a model.

Run: python3 -m src.features.team_game_stats
"""
import numpy as np
import pandas as pd

from StatScraper import save_to_sqlite, query_sqlite, drop_table_if_exists

GAMES_DB_PATH = "data/nfl_games.db"

# EPA/success/yardage are only meaningful on an actual scrimmage snap --
# excludes kneels, spikes, timeouts, and no-plays (declined penalties etc).
SCRIMMAGE = "play_type in ('pass', 'run')"

# games (import_schedules) labels relocated franchises with the abbreviation
# they used *at the time* (STL/SD/OAK); plays (import_pbp_data) labels every
# season of the same franchise with its *current* abbreviation (LA/LAC/LV).
# No season range overlaps (verified: STL max season 2015, LA min season
# 2016, etc.), so this is a safe direct remap, not a guess.
TEAM_ABBR_FIXES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

# nflverse itself has zero pbp rows for a handful of completed early games
# (verified: 1999_01_BAL_STL, 2000_03_SD_KC, 2000_06_BUF_MIA) -- these will
# always come out with null offensive/defensive stats below. That's a real
# gap in the source data, not a bug in this merge.


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape `games` (one row/game, home+away columns) into long format:
    one row per (game_id, team) with that team's own perspective."""
    if games.game_id.isna().any() or games.game_id.duplicated().any():
        raise ValueError("Expected non-null unique game_id")
    games = games.replace({"home_team": TEAM_ABBR_FIXES, "away_team": TEAM_ABBR_FIXES})
    if games[["home_team", "away_team"]].isna().any().any() or games.home_team.eq(games.away_team).any():
        raise ValueError("Expected two distinct non-null teams per game")
    games = games.copy()
    for column in ("home_rest", "away_rest"):
        if column not in games:
            games[column] = float("nan")
    home = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest_days",
    }).assign(is_home=True)
    away = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest_days",
    }).assign(is_home=False)
    cols = ["game_id", "season", "week", "game_type", "gameday", "team", "opponent",
            "is_home", "points_for", "points_against", "rest_days"]
    if "completed" in games:
        cols.append("completed")
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def _redzone_td_rate(df: pd.DataFrame) -> float:
    # Known limit: per-play proxy, not per-red-zone-*trip* (needs drive-level
    # grouping). Upgrade to per-trip if red-zone efficiency needs to match
    # the standard broadcast stat exactly.
    rz = df[df["yardline_100"] <= 20]
    return rz["touchdown"].mean() if len(rz) else np.nan


def _third_down_rate(df: pd.DataFrame) -> float:
    third = df[df["down"] == 3]
    return third["third_down_converted"].mean() if len(third) else np.nan


def build_team_game_pbp_stats(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team): offensive box score (as posteam) merged
    with defensive box score (what they allowed, as defteam)."""
    snaps = plays.query(SCRIMMAGE)

    off = snaps.groupby(["game_id", "posteam"]).apply(lambda g: pd.Series({
        "off_plays": len(g),
        "off_epa_per_play": g["epa"].mean(),
        "off_success_rate": g["success"].mean(),
        "off_pass_epa": g.loc[g["pass"] == 1, "epa"].mean(),
        "off_rush_epa": g.loc[g["rush"] == 1, "epa"].mean(),
        "off_yards_per_play": g["yards_gained"].mean(),
        "off_third_down_rate": _third_down_rate(g),
        "off_redzone_td_rate": _redzone_td_rate(g),
        "off_turnovers": g["interception"].sum() + g["fumble_lost"].sum(),
    }), include_groups=False).reset_index().rename(columns={"posteam": "team"})

    off_sacks = plays.groupby(["game_id", "posteam"])["sack"].sum().reset_index() \
        .rename(columns={"posteam": "team", "sack": "off_sacks_allowed"})

    defn = snaps.groupby(["game_id", "defteam"]).apply(lambda g: pd.Series({
        "def_epa_per_play": g["epa"].mean(),
        "def_success_rate": g["success"].mean(),
        "def_third_down_rate": _third_down_rate(g),
        "def_redzone_td_rate": _redzone_td_rate(g),
        "def_takeaways": g["interception"].sum() + g["fumble_lost"].sum(),
    }), include_groups=False).reset_index().rename(columns={"defteam": "team"})

    def_sacks = plays.groupby(["game_id", "defteam"])["sack"].sum().reset_index() \
        .rename(columns={"defteam": "team", "sack": "def_sacks"})

    out = off.merge(off_sacks, on=["game_id", "team"], how="left") \
             .merge(defn, on=["game_id", "team"], how="left") \
             .merge(def_sacks, on=["game_id", "team"], how="left")
    return out


def build_team_game_stats(games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team): the schedule skeleton left-joined to its
    box score.

    Left join on purpose. nflverse has zero play rows for a handful of
    completed early games, and those must survive as scheduled rows with
    null stats rather than vanishing -- the rolling layer counts them as
    completed history with missing metrics.

    NOT leak-safe: every column here describes what happened *in* the game.
    Only `rolling_features` makes it usable as model input.
    """
    team_games = build_team_games(games)
    pbp_stats = build_team_game_pbp_stats(plays)
    return team_games.merge(pbp_stats, on=["game_id", "team"], how="left")


def run(db_path: str = GAMES_DB_PATH) -> pd.DataFrame:
    """Always recomputes from ALL rows currently in games/plays (fast --
    seconds, not a network fetch) and fully replaces the table. A partial
    `seasons` filter isn't offered: it previously combined with upsert-by-key
    to leave stale rows behind whenever the aggregation logic itself changed
    (e.g. the STL/SD/OAK abbreviation fix) -- dropping the whole table
    before rewriting it is what actually guarantees this table matches its
    inputs."""
    games = query_sqlite("SELECT * FROM games", db_path=db_path)
    plays = query_sqlite("SELECT * FROM plays", db_path=db_path)

    df = build_team_game_stats(games, plays)
    drop_table_if_exists("team_game_stats", db_path=db_path)
    save_to_sqlite(df, "team_game_stats", key_columns=["game_id", "team"], db_path=db_path)
    return df


if __name__ == "__main__":
    run()
