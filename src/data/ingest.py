"""
Data ingestion for the NFL game predictor (win probability / point
differential / spread coverage).

SOURCE: nflverse, via the nfl_data_py package -- already a dependency in
this repo (see StatScraper.py). nfl_data_py pulls versioned parquet/csv
releases published by the nflverse project (the community-maintained
successor to nflfastR), which is the standard free source for NFL analytics:

  - import_schedules(): one row per game -- final score, home/away teams,
    kickoff date/time, rest days (nflverse computes these from each team's
    prior game date), stadium/surface/weather, referee, starting QBs, and
    the closing betting lines (spread, total, moneylines). Verified: 100%
    non-null spread_line/total_line/home_qb_name back to the 1999 season
    (7,276 games checked). temp/wind are null for ~20-40%/season of games
    by design (dome/closed-roof games have no outdoor weather), not missing
    data.
  - import_pbp_data(): one row per play -- EPA, success rate, air yards,
    down/distance, sacks, turnovers, etc. Sourced from nflfastR's play
    model, also reliable back to 1999.

Rejected alternatives:
  - Scraping Pro-Football-Reference / ESPN: nflverse already structures
    and publishes this exact data for free, so scraping would be strictly
    worse (slower, brittle to markup changes, ToS-ambiguous) for no benefit.
  - Kaggle NFL datasets: static snapshots of unclear/unreproducible
    provenance that can't be extended to new seasons.
  - Paid providers (SportsDataIO etc.): unnecessary cost for data nflverse
    already provides.

LEAKAGE NOTE (for the feature-engineering step, not this one): this module
only fetches and stores raw source data -- it computes nothing. But since
some columns here are only known *after* a game finishes, mark them now so
future-you doesn't reach for them as inputs:
  - Label / post-game only, never a feature: home_score, away_score,
    result, total (these come from the same completed game being predicted).
  - Every play-level stat in `plays` (epa, success, yards_gained, ...) is
    post-game for the game it belongs to -- safe to use only as history for
    *rolling* features computed from strictly earlier games.
  - Pre-game safe (known before kickoff): spread_line, total_line,
    moneylines, home_rest/away_rest, weather, stadium/surface, referee,
    home_qb_name/away_qb_name (starters are announced pre-game), div_game.

Run: python3 -m src.data.ingest
"""
from typing import List, Optional

import nfl_data_py as nfl
import pandas as pd

from StatScraper import save_to_sqlite  # reuse the existing upsert helper

# nflverse's own pbp/schedule data starts getting thin before 1999 (fewer
# tracked columns, spottier betting lines), so this is the earliest season
# worth pulling automatically.
EARLIEST_SEASON = 1999

GAMES_DB_PATH = "data/nfl_games.db"

# Curated subset of nfl_data_py's ~400 raw pbp columns -- just what's needed
# to later build EPA/play, success rate, third-down/red-zone rate, sack
# rate, and turnover features. Cuts row width ~9x, no loss for this project.
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type",
    "home_team", "away_team", "posteam", "defteam", "posteam_type",
    "down", "ydstogo", "yardline_100", "goal_to_go", "qtr", "game_seconds_remaining",
    "play_type", "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike", "special_teams_play",
    "epa", "success", "wpa",
    "yards_gained", "air_yards", "yards_after_catch", "complete_pass", "incomplete_pass", "interception",
    "sack", "fumble", "fumble_lost",
    "touchdown", "pass_touchdown", "rush_touchdown",
    "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    "penalty", "penalty_team", "penalty_yards", "drive",
]

# The subset of PBP_COLUMNS that src/features/team_game_stats.py actually
# reads. Their absence is not a degraded feature, it is a broken pipeline, so
# a fetch that cannot supply them fails here rather than in the feature
# builder one layer down -- after ingest has already written a truncated
# `plays` table to SQLite.
#
# Everything else in PBP_COLUMNS stays best-effort on purpose: nflverse
# genuinely does not carry some of it for the earliest seasons (`air_yards` is
# empty for every 1999 and 2005 play in the current snapshot,
# `special_teams_play` for every 1999 play), and nothing downstream reads them
# yet. Promote a column here when something starts depending on it.
PBP_REQUIRED_COLUMNS = [
    "game_id", "play_id",                      # upsert key (see ingest_pbp)
    "season",                                  # season filtering downstream
    "posteam", "defteam",                      # whose play it is
    "play_type",                               # the SCRIMMAGE filter
    "down", "yardline_100", "third_down_converted", "touchdown",
    "epa", "success", "pass", "rush", "yards_gained",
    "interception", "fumble_lost", "sack",
]


def _select_pbp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to PBP_COLUMNS, but refuse to return a frame missing a required one.

    The tolerant intersection below is what lets an optional column disappear
    without comment. Applied to a required column it is a silent data loss:
    the curated list shrinks to match whatever arrived, so a check that
    compares the result against that same filtered list can never notice.
    """
    missing = [column for column in PBP_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"play-by-play source is missing required column(s): {missing}. "
            f"src/features/team_game_stats.py reads these directly, so ingesting without "
            f"them writes a `plays` table that fails later with a bare KeyError. Check "
            f"whether nflverse renamed them before widening PBP_REQUIRED_COLUMNS."
        )
    return df[[column for column in PBP_COLUMNS if column in df.columns]]


def default_seasons() -> List[int]:
    """All seasons nflverse can supply: 1999 through the current calendar year (the current
    season may be partial/in-progress; unplayed games simply have null scores)."""
    from datetime import datetime
    return list(range(EARLIEST_SEASON, datetime.now().year + 1))


def fetch_schedules(seasons: List[int]) -> pd.DataFrame:
    """One row per game: score, betting lines, weather, rest days, starters."""
    return nfl.import_schedules(seasons)


def fetch_pbp(seasons: List[int]) -> pd.DataFrame:
    """One row per play, trimmed to PBP_COLUMNS, raising if the source dropped
    a column the feature layer needs (see `_select_pbp_columns`). `downcast=True`
    shrinks float64->float32 (nflverse ships them oversized); harmless for stats
    used as model inputs."""
    return _select_pbp_columns(nfl.import_pbp_data(seasons, downcast=True, cache=False))


def ingest_schedules(seasons: List[int], db_path: str = GAMES_DB_PATH) -> pd.DataFrame:
    """Fetch schedules and upsert them into `games`, keyed on `game_id`.

    Upsert, not replace: re-running a season corrects rows whose scores were
    null at first pull without discarding seasons this call did not request.
    """
    df = fetch_schedules(seasons)
    save_to_sqlite(df, "games", key_columns=["game_id"], db_path=db_path)
    return df


def ingest_pbp(seasons: List[int], db_path: str = GAMES_DB_PATH) -> pd.DataFrame:
    """Fetch play-by-play and upsert it into `plays`, keyed on
    `(game_id, play_id)` -- `play_id` alone repeats across games.

    Raises if the source dropped a column the feature layer needs; see
    `_select_pbp_columns`.
    """
    df = fetch_pbp(seasons)
    save_to_sqlite(df, "plays", key_columns=["game_id", "play_id"], db_path=db_path)
    return df


def run_ingestion(seasons: Optional[List[int]] = None, db_path: str = GAMES_DB_PATH) -> None:
    """Both raw tables for `seasons`, defaulting to every season nflverse
    carries. Network-bound and the slowest entry point in the repo.

    Writes raw facts only. Derived tables are rebuilt separately by
    `src.features.*`, so re-ingesting never silently changes a feature.
    """
    seasons = seasons or default_seasons()
    print(f"Ingesting {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}) into {db_path}")
    ingest_schedules(seasons, db_path)
    ingest_pbp(seasons, db_path)


if __name__ == "__main__":
    run_ingestion()
