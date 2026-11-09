"""
StatScraper.py
================
NFL player & team stat scraper -- pulls player/team stats from nflverse's
official data releases (nfl_data_py) and stores them in SQLite.

Storage: every pull upserts into a SQLite database (see save_to_sqlite()),
keyed on the natural identity of each row (player_id+season+week, or
season+team+season_type), so re-running the scraper for weeks you already
have refreshes those rows instead of duplicating them. Pass save_path=None /
db_path=None to skip CSV/SQLite output on any given call.

INSTALL:
    pip install nfl_data_py pandas pyarrow
    (pyarrow reads the .parquet files nflverse publishes; sqlite3 needs no
    install, it's in the Python standard library.)

DOCS: https://github.com/nflverse/nfl_data_py
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# SHARED HELPERS -- used by every approach below
# ---------------------------------------------------------------------------
def default_recent_seasons(n: int = 3) -> List[int]:
    """
    Return the n most recent NFL season years, ending with the current
    calendar year. Used as the default `years` value across this file since
    this project only needs a rolling recent-seasons window, not full
    history.

    Args:
        n: How many seasons back to include (default 3).

    Returns:
        List of season years, oldest first, e.g. [2024, 2025, 2026] if run
        during the 2026 season.

    Notes:
        NFL seasons are labeled by the year they start in (a game played in
        January 2026 still belongs to the "2025 season"), and a new season
        starts in September. Using the current calendar year as the window's
        end is correct for both mid-season and off-season use -- worst case,
        early in a new season, is that the newest year briefly has only a
        few weeks of data, not a missing one.
    """
    current_year = datetime.now().year
    return list(range(current_year - n + 1, current_year + 1))


def save_to_csv(df: pd.DataFrame, filename: str, out_dir: str = "data") -> str:
    """
    Save a DataFrame to a CSV file, creating the output directory if needed.

    Args:
        df: The DataFrame to save.
        filename: Name of the CSV file (e.g. "team_stats_2025.csv").
        out_dir: Directory to save into. Created automatically if missing.
                 Defaults to "./data".

    Returns:
        The full path to the saved CSV file.

    Notes:
        Uses index=False so pandas' auto-generated row index isn't written
        as a spurious extra column in the CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved {len(df)} rows -> {full_path}")
    return full_path


def save_to_sqlite(df: pd.DataFrame, table_name: str, key_columns: List[str], db_path: str = "data/nfl_stats.db") -> str:
    """
    Save (upsert) a DataFrame into a SQLite table, creating the table and a
    UNIQUE index on `key_columns` the first time it's called, and replacing
    any rows with matching keys on every later call -- so re-running a
    scrape for weeks/seasons you already have doesn't create duplicate rows,
    it just refreshes them.

    Args:
        df: The DataFrame to save.
        table_name: SQLite table to write into (created if it doesn't exist).
        key_columns: Column(s) that uniquely identify a row, e.g.
            ["player_id", "season", "week"] for weekly player stats, or
            ["season", "team"] for team-season stats. Used to build the
            UNIQUE index that makes upserts possible.
        db_path: Path to the SQLite database file. Created automatically
            (along with its parent directory) if it doesn't exist yet.
            Defaults to "data/nfl_stats.db".

    Returns:
        The full path to the database file.

    Notes:
        Implementation: the DataFrame is written to a throwaway staging
        table, then `INSERT OR REPLACE INTO <table_name> SELECT * FROM
        staging` copies it into the real table -- SQLite's "OR REPLACE"
        resolves conflicts using the UNIQUE index on key_columns, so a row
        with a key that already exists gets overwritten rather than
        duplicated. The staging table is dropped afterward either way.

        A single .db file can hold many tables, so calling this repeatedly
        with different table_name values (e.g. "player_weekly_stats" and
        "team_season_stats") builds up one combined database rather than
        scattering data across separate files like the CSV approach does.
    """
    import sqlite3

    out_dir = os.path.dirname(db_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        staging_table = f"_staging_{table_name}"
        df.to_sql(staging_table, conn, if_exists="replace", index=False)

        cursor = conn.cursor()
        table_exists = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()

        if not table_exists:
            # First time writing this table: create it from the staging
            # table's inferred schema, then add the UNIQUE index that makes
            # future "INSERT OR REPLACE" upserts possible.
            cursor.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM "{staging_table}"')
            key_cols_sql = ", ".join(f'"{c}"' for c in key_columns)
            index_name = f"idx_{table_name}_key"
            cursor.execute(f'CREATE UNIQUE INDEX "{index_name}" ON "{table_name}" ({key_cols_sql})')
        else:
            cursor.execute(f'INSERT OR REPLACE INTO "{table_name}" SELECT * FROM "{staging_table}"')

        cursor.execute(f'DROP TABLE "{staging_table}"')
        conn.commit()
    finally:
        conn.close()

    print(f"Upserted {len(df)} rows -> {db_path} (table: {table_name})")
    return db_path


def _drop_if_missing_columns(table_name: str, columns: List[str], db_path: str) -> None:
    """
    Drop `table_name` when it lacks any of `columns`, so save_to_sqlite()'s
    `INSERT ... SELECT *` does not fail on a table written before a column was
    added. Only for raw tables re-pulled from source on every load.
    """
    import sqlite3

    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    try:
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    finally:
        conn.close()
    if have and not set(columns) <= have:
        drop_table_if_exists(table_name, db_path=db_path)


def drop_table_if_exists(table_name: str, db_path: str = "data/nfl_stats.db") -> None:
    """
    Drop `table_name` if it exists. For tables that are fully recomputed
    from source data on every run (e.g. a derived feature table), call this
    before save_to_sqlite() instead of relying on its upsert: upsert only
    refreshes rows whose key still matches, so a row whose key VALUE changed
    (e.g. a bugfix that relabels some rows) is never overwritten and lingers
    as a stale duplicate. Safe for raw scraped tables too, just unnecessary
    there since their keys don't change between runs.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()
    finally:
        conn.close()


def query_sqlite(sql: str, db_path: str = "data/nfl_stats.db", params: Optional[list] = None) -> pd.DataFrame:
    """
    Run a SQL query against the stats database and return the result as a
    DataFrame. A thin convenience wrapper so you don't need to import
    sqlite3/pandas separately when exploring the data.

    Args:
        sql: Any SQL query, e.g. "SELECT * FROM team_season_stats WHERE season = 2025".
            Use "?" placeholders (not f-string/.format() interpolation) for
            any user- or variable-supplied values, and pass them via `params`.
        db_path: Path to the SQLite database file. Defaults to "data/nfl_stats.db".
        params: Values to bind to "?" placeholders in `sql`, in order. Omit
            for queries with no placeholders.

    Returns:
        Query results as a DataFrame.

    Example:
        query_sqlite("SELECT player_name, passing_yards FROM player_weekly_stats "
                     "WHERE season = 2025 ORDER BY passing_yards DESC LIMIT 10")

        query_sqlite("SELECT * FROM player_weekly_stats WHERE season IN (?, ?)",
                     params=[2024, 2025])
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


try:
    import nfl_data_py as nfl
    _HAS_NFL_DATA_PY = True
except ImportError:
    _HAS_NFL_DATA_PY = False


_STATS_PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{}.parquet"
)


def get_player_weekly_stats_nflverse(
    years: Optional[List[int]] = None,
    save_path: str = "player_weekly_stats.csv",
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Pull week-by-week player stat lines (passing/rushing/receiving/etc.)
    from nflverse's current weekly data release, and save the result to a
    SQLite database and a CSV by default.

    Args:
        years: List of season years to pull, e.g. [2024, 2025, 2026].
            Defaults to the last 3 seasons (see default_recent_seasons())
            if not given, since this project only needs recent data.
        save_path: CSV filename to write (passed to save_to_csv()). Pass
            save_path=None to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "player_weekly_stats", keyed on player_id/season/week -- see
            save_to_sqlite()). Pass db_path=None to skip SQLite.

    Returns:
        DataFrame containing one row per player per week (~150 columns),
        including player_name, position, team, opponent_team, season_type,
        and every offense/defense/special-teams stat nflverse tracks.

    Example:
        get_player_weekly_stats_nflverse()  # last 3 seasons -> sqlite + csv
        get_player_weekly_stats_nflverse([2024, 2025], db_path=None)  # csv only

    Notes:
        This reads directly from nflverse's "stats_player" release tag
        rather than going through nfl_data_py's `import_weekly_data()`.
        That function is hardcoded to an older release tag ("player_stats")
        which nflverse stopped updating after the 2024 season -- requesting
        2025+ through it 404s. "stats_player" is nflverse's current,
        actively-updated tag and covers every season including the one in
        progress, so pulling from it directly sidesteps the stale pin
        without needing to wait on an nfl_data_py update.

        Column ambiguity warning -- interceptions: nflverse tracks two
        separate, unrelated "interception" columns on every row, and their
        names alone don't make the distinction obvious:
          - passing_interceptions: passes THIS player THREW that were
            intercepted (an offensive stat -- what shows up on a QB's line).
          - def_interceptions: passes THIS player CAUGHT as a defender (a
            defensive stat -- what shows up on a cornerback/safety's line).
        A given player-week row is realistically only one or the other
        (a QB's def_interceptions is 0; a defensive back's
        passing_interceptions is 0) -- treat them as two different stats
        about two different roles, never sum or average them together as if
        "interceptions" were a single column. The same offense/defense
        prefix convention (passing_/rushing_/receiving_ vs def_) disambiguates
        several other columns too, e.g. rushing_fumbles vs def_fumbles.
    """
    years = years if years is not None else default_recent_seasons()

    frames = [pd.read_parquet(_STATS_PLAYER_WEEK_URL.format(year)) for year in years]
    df = pd.concat(frames, ignore_index=True)

    # Drop team-level aggregate rows (e.g. total team penalties for the game)
    # that nflverse includes in this file alongside individual player stats
    # -- they carry no player_id, so they don't belong in a per-player table
    # and would violate the player_id/season/week primary key below anyway.
    df = df.dropna(subset=["player_id"])

    key_columns = ["player_id", "season", "week"]
    if db_path:
        save_to_sqlite(df, "player_weekly_stats", key_columns=key_columns, db_path=db_path)
    if save_path:
        save_to_csv(df, save_path)
    return df


def get_team_season_stats_nflverse(
    years: Optional[List[int]] = None,
    season_type: str = "REG",
    save_path: str = "team_season_stats.csv",
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Build a team-level season summary (points for/against, wins/losses,
    etc.) by pulling nflverse's schedule/results data and aggregating it
    per team per season. Saved to a SQLite database and a CSV by default.

    Args:
        years: List of season years to pull, e.g. [2024, 2025, 2026].
            Defaults to the last 3 seasons (see default_recent_seasons())
            if not given, since this project only needs recent data.
        season_type: Which games to include:
            - "REG"  -- regular season only (default). This is the
                        conventional meaning of a team's record/standings,
                        e.g. what ESPN/TV report.
            - "POST" -- playoff games only (wild card through Super Bowl).
            - "ALL"  -- regular season + playoffs combined.
        save_path: CSV filename to write. Pass save_path=None to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "team_season_stats", keyed on season/team/season_type -- see
            save_to_sqlite()). Pass db_path=None to skip SQLite.

    Returns:
        DataFrame with one row per team per season (per season_type): team,
        season, season_type, wins, losses, ties, points_for, points_against,
        point_diff.

    Raises:
        ImportError: if nfl_data_py isn't installed.
        ValueError: if season_type isn't one of "REG", "POST", "ALL".

    Notes:
        This aggregates the game-by-game schedule data rather than pulling
        a pre-built "team stats" table (nflverse doesn't publish one
        directly) -- if you need deeper team stats (yards/turnovers/etc.)
        aggregate `nfl.import_pbp_data(years)` by `posteam` instead; that
        data is much larger (millions of rows) so only pull the years you
        actually need.

        Unlike get_player_weekly_stats_nflverse() above, this one still
        goes through nfl_data_py's import_schedules(), which reads a single
        continuously-updated CSV (not a frozen per-year archive) -- so it
        isn't affected by the stale-release-tag issue.

        An earlier version of this function always combined regular season
        and playoff games into one total with no way to tell them apart --
        e.g. it reported the Chiefs' 2024 record as 17-3 (their real 15-2
        regular season plus a 2-1 playoff run to the Super Bowl), which
        silently disagreed with the 15-2 every other source reports as
        "the" 2024 record. season_type now makes that choice explicit, and
        it's part of the upsert key so calling this with different
        season_type values doesn't overwrite each other in the database --
        you can have REG, POST, and ALL rows for the same team/season side
        by side.
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    season_type = season_type.upper()
    valid_season_types = {"REG", "POST", "ALL"}
    if season_type not in valid_season_types:
        raise ValueError(f"season_type must be one of {sorted(valid_season_types)}, got {season_type!r}")

    years = years if years is not None else default_recent_seasons()
    games = nfl.import_schedules(years)

    # nflverse tags each game's game_type as REG, or one of four playoff
    # round codes (WC, DIV, CON, SB) -- "POST" here bundles those four.
    if season_type == "REG":
        games = games[games["game_type"] == "REG"]
    elif season_type == "POST":
        games = games[games["game_type"] != "REG"]
    # "ALL": no filter, keep every game_type.

    home = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
    })[["season", "team", "opponent", "points_for", "points_against"]]

    away = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
    })[["season", "team", "opponent", "points_for", "points_against"]]

    all_games = pd.concat([home, away], ignore_index=True).dropna(subset=["points_for", "points_against"])
    all_games["win"] = (all_games["points_for"] > all_games["points_against"]).astype(int)
    all_games["loss"] = (all_games["points_for"] < all_games["points_against"]).astype(int)
    all_games["tie"] = (all_games["points_for"] == all_games["points_against"]).astype(int)

    summary = all_games.groupby(["season", "team"], as_index=False).agg(
        wins=("win", "sum"),
        losses=("loss", "sum"),
        ties=("tie", "sum"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
    )
    summary["point_diff"] = summary["points_for"] - summary["points_against"]
    summary["season_type"] = season_type

    key_columns = ["season", "team", "season_type"]
    if db_path:
        save_to_sqlite(summary, "team_season_stats", key_columns=key_columns, db_path=db_path)
    if save_path:
        save_to_csv(summary, save_path)
    return summary


# Default stat used to rank players AT a given `position` -- the single most
# standard box-score stat for judging who's playing at an elite level there.
# "CB" isn't a real column value on its own (see below) -- it's handled as a
# special case in get_position_leaders() by combining two coverage-related
# stats, since neither alone captures coverage quality well.
POSITION_RANKING_STAT: Dict[str, str] = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "WR": "receiving_yards",
    "TE": "receiving_yards",
}


def get_position_leaders(
    df: pd.DataFrame,
    position: str,
    season: int,
    top_n: int = 10,
    stat_column: Optional[str] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rank players at a given position by a season-to-date total of a chosen
    stat and return the top N -- e.g. "top 10 WRs by receiving yards in
    2025" or "top 10 CBs by coverage plays in 2025".

    Args:
        df: Weekly player stats, e.g. from get_player_weekly_stats_nflverse().
        position: An exact `position` value (e.g. "WR", "CB", "QB"), not
            `position_group` -- lets you separate e.g. CB from Safety even
            though nflverse groups both under position_group "DB".
        season: Which season to rank within.
        top_n: How many players to return (default 10).
        stat_column: Which column to rank by. Defaults to
            POSITION_RANKING_STAT[position] for QB/RB/WR/TE. For "CB", the
            default instead combines def_interceptions + def_pass_defended
            into one "coverage_score" column, since neither stat alone
            fully captures coverage quality, and there's no public
            play-by-play charting data (e.g. PFF grades) available to do
            better -- see the "Column ambiguity warning" note in
            get_player_weekly_stats_nflverse for the
            def_interceptions/passing_interceptions distinction this
            builds on. Required (raises if omitted) for any other position.
        save_path: Optional CSV filename to also save the result to.

    Returns:
        DataFrame, one row per player, sorted descending by the ranking
        stat: rank (1 = best), player_id, player_name, team (their most
        recent team that season, so a mid-season trade doesn't split their
        total across two rows), and the ranking stat's season total.

    Raises:
        ValueError: if stat_column isn't given and `position` has no
            default in POSITION_RANKING_STAT (and isn't "CB").

    Notes:
        Ranks by the season's REGULAR-SEASON totals as they stand in `df`
        -- pass a season still in progress and you get "top N so far,"
        which shifts week to week as more games are played. That's the
        correct behavior for a live leaderboard; if you need a frozen,
        no-lookahead ranking for model training (e.g. "who was the best CB
        heading into week 6, using only weeks 1-5"), use
        PlayerProjections.py's trailing/leak-safe ranking functions
        instead -- this one is a straightforward season-to-date leaderboard.

    Example:
        get_position_leaders(df, "WR", season=2025, top_n=10)
        get_position_leaders(df, "CB", season=2025, top_n=10, save_path="cb_leaders_2025.csv")
    """
    position_df = df[(df["position"] == position) & (df["season"] == season) & (df["season_type"] == "REG")].copy()

    if stat_column is None:
        if position == "CB":
            position_df["coverage_score"] = position_df["def_interceptions"].fillna(0) + position_df["def_pass_defended"].fillna(0)
            stat_column = "coverage_score"
        elif position in POSITION_RANKING_STAT:
            stat_column = POSITION_RANKING_STAT[position]
        else:
            raise ValueError(
                f"No default ranking stat for position={position!r}; pass stat_column explicitly "
                f"(known defaults: {sorted(POSITION_RANKING_STAT)} plus 'CB')."
            )

    totals = position_df.groupby(["player_id", "player_name"], as_index=False)[stat_column].sum()
    most_recent_team = position_df.sort_values("week").groupby("player_id")["team"].last()
    totals["team"] = totals["player_id"].map(most_recent_team)

    totals = totals.sort_values(stat_column, ascending=False).head(top_n).reset_index(drop=True)
    totals.insert(0, "rank", totals.index + 1)

    if save_path:
        save_to_csv(totals, save_path)
    return totals


def get_player_bio(save_path: Optional[str] = None, db_path: Optional[str] = "data/nfl_stats.db") -> pd.DataFrame:
    """
    Pull nflverse's player master table -- one row per player, ever, with
    their identity across multiple ID systems (gsis_id, the "player_id"
    used throughout this project's own player_weekly_stats table, plus
    pfr_id, espn_id, etc.) and bio info: height, weight, position, college,
    draft info.

    Args:
        save_path: CSV filename to write. Pass None (default) to skip the CSV.
        db_path: SQLite database file to upsert into (table "player_bio",
            keyed on player_id). Pass None to skip SQLite.

    Returns:
        DataFrame with one row per player: player_id (gsis_id, renamed for
        consistency with the rest of this project's tables), pfr_id,
        display_name, height (inches), weight (lbs), position, college_name,
        draft_year, and a few other bio columns nflverse tracks.

    Notes:
        This is the crosswalk that makes it possible to join
        Pro-Football-Reference-sourced data (which identifies players by
        pfr_id, e.g. get_weekly_defensive_target_stats() below) onto this
        project's nflverse-play-by-play-sourced tables (which use gsis_id).
        Unlike the weekly/season pulls elsewhere in this file, there's no
        `years` parameter -- nflverse publishes ONE current master player
        table, not a per-season archive, so this always returns the latest
        version for every player nflverse has ever tracked.
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    players = nfl.import_players()
    bio = players.rename(columns={"gsis_id": "player_id"})
    bio = bio[bio["player_id"].notna()]

    if db_path:
        save_to_sqlite(bio, "player_bio", key_columns=["player_id"], db_path=db_path)
    if save_path:
        save_to_csv(bio, save_path)
    return bio


def get_weekly_defensive_target_stats(
    years: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Pull Pro-Football-Reference's advanced weekly DEFENSIVE stats -- the
    "targeted" numbers for a specific defender (how many times they were
    thrown at, how often that pass was completed, yards/TDs allowed, passer
    rating allowed when targeted) -- and attach each row's real player_id so
    it can be joined against this project's other player_weekly_stats-based
    tables.

    Args:
        years: Season years to pull, e.g. [2023, 2024, 2025]. Defaults to
            StatScraper's default_recent_seasons() (last 3 seasons) if not given.
        save_path: CSV filename to write. Pass None (default) to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "weekly_defensive_target_stats", keyed on player_id/season/week).
            Pass None to skip SQLite.

    Returns:
        DataFrame with one row per defender per week: player_id, season,
        week, game_type ("REG" or a playoff round code), team, opponent,
        def_targets (times thrown at), def_ints,
        def_completions_allowed, def_completion_pct, def_yards_allowed,
        def_yards_allowed_per_tgt, def_receiving_td_allowed,
        def_passer_rating_allowed (the standard "how good is this defender
        in coverage" summary stat -- lower is better, same convention as a
        quarterback's own passer rating), def_adot (average depth of the
        passes thrown at them).

    Notes:
        This is real PFR advanced-stats data (via nfl_data_py's
        import_weekly_pfr('def', years)), not a derived proxy -- unlike
        computing a rough "coverage score" from the basic box score's
        def_interceptions + def_pass_defended (what an earlier version of
        this project's matchup model used), these numbers directly answer
        "how many times was this guy targeted, and how did he do." The join
        from PFR's own pfr_player_id to this project's player_id (gsis_id)
        goes through get_player_bio()'s crosswalk and matches essentially
        every row (verified: 7924/7926 for the 2025 season alone; PFR's
        weekly def data doesn't include a `position` column itself, so
        get_player_bio()'s `position` is what you'd filter on for e.g. "CB
        only" -- this function returns all defenders, not just corners).
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    years = years if years is not None else default_recent_seasons()

    pfr_def = nfl.import_weekly_pfr("def", years)
    bio = get_player_bio(db_path=None)[["player_id", "pfr_id"]]

    merged = pfr_def.merge(bio, left_on="pfr_player_id", right_on="pfr_id", how="left")
    merged = merged[merged["player_id"].notna()]

    keep_cols = [
        "player_id", "season", "week", "game_type", "team", "opponent",
        "def_targets", "def_ints", "def_completions_allowed", "def_completion_pct",
        "def_yards_allowed", "def_yards_allowed_per_tgt", "def_receiving_td_allowed",
        "def_passer_rating_allowed", "def_adot",
    ]
    result = merged[keep_cols].reset_index(drop=True)

    if db_path:
        save_to_sqlite(result, "weekly_defensive_target_stats", key_columns=["player_id", "season", "week"], db_path=db_path)
    if save_path:
        save_to_csv(result, save_path)
    return result


def get_schedule_context(
    years: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Pull per-TEAM game context from nflverse's schedule data: home/away,
    rest days (own and opponent's), divisional-matchup flag, weather
    (roof/temp/wind), and each team's starting QB for that game. Reshaped
    from nflverse's one-row-per-game schedule (home_team/away_team columns)
    into one row per team per game, matching the shape of this project's
    other team-level tables (see StatScraper.get_team_season_stats_nflverse(),
    which reshapes the same underlying schedule data the same way).

    Args:
        years: Season years to pull, e.g. [2023, 2024, 2025]. Defaults to
            default_recent_seasons() if not given. Unlike most of this
            file's pulls, nflverse's schedule includes the CURRENT season's
            not-yet-played games too (with NaN scores) -- so this also
            covers upcoming weeks, useful for live pregame context.
        save_path: CSV filename to write. Pass None (default) to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "schedule_context", keyed on season/week/team). Pass None to skip.

    Returns:
        DataFrame with one row per (season, week, team): game_type, opponent,
        is_home (1/0), rest_days (this team's days of rest before the game),
        opponent_rest_days, div_game (1/0 divisional matchup), roof
        ("outdoors"/"dome"/"closed"/"open"), temp (NaN for domes -- see
        Notes), wind (NaN for domes), starting_qb_id (that team's own
        starting QB's player_id for this game), total_line (the game's
        points over/under) and team_spread_line (points THIS team is
        favored by; negative when it is the underdog).

        The lines are nflverse's: the closing line for played games, the
        current line for upcoming ones (posted about a week ahead; NaN
        further out). Both are set before kickoff and carry no result.

    Notes:
        temp/wind are only meaningful for outdoor/retractable-roof games --
        nflverse leaves them NaN for dome games rather than reporting a
        fake indoor reading (verified: 190/285 games populated for the 2025
        season, matching the number of non-dome games). Downstream code
        that uses these should fill dome games with a "climate-controlled,
        no weather effect" constant (e.g. ~70F, 0 wind) rather than a raw
        league-average fill, since a dome's true condition is known, not
        missing information.
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    years = years if years is not None else default_recent_seasons()
    games = nfl.import_schedules(years)

    shared_cols = ["season", "week", "game_type", "div_game", "roof", "temp", "wind"]

    home = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_rest": "rest_days", "away_rest": "opponent_rest_days", "home_qb_id": "starting_qb_id",
    })
    home["is_home"] = 1
    home["team_spread_line"] = games["spread_line"]  # nflverse: positive means the home team is favored

    away = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_rest": "rest_days", "home_rest": "opponent_rest_days", "away_qb_id": "starting_qb_id",
    })
    away["is_home"] = 0
    away["team_spread_line"] = -games["spread_line"]

    keep_cols = shared_cols + ["team", "opponent", "is_home", "rest_days", "opponent_rest_days", "starting_qb_id",
                               "total_line", "team_spread_line"]
    result = pd.concat([home[keep_cols], away[keep_cols]], ignore_index=True)

    if db_path:
        _drop_if_missing_columns("schedule_context", list(result.columns), db_path)
        save_to_sqlite(result, "schedule_context", key_columns=["season", "week", "team"], db_path=db_path)
    if save_path:
        save_to_csv(result, save_path)
    return result


def get_weekly_player_age(
    years: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Pull each player's age AND years of NFL experience for every week they
    were on an active roster, from nflverse's weekly roster snapshots --
    nflverse's own computation for both (real birth date vs. that week's
    actual date for age; seasons played for experience), not derived here.

    Args:
        years: Season years to pull, e.g. [2023, 2024, 2025]. Defaults to
            default_recent_seasons() if not given.
        save_path: CSV filename to write. Pass None (default) to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "weekly_player_age", keyed on player_id/season/week). Pass None
            to skip SQLite.

    Returns:
        DataFrame with one row per (player_id, season, week): age
        (fractional years), years_exp (integer seasons played, 0 for a
        rookie -- distinct from age: a 24-year-old day-one rookie and a
        24-year-old entering his 3rd season have the same age but very
        different NFL experience, which is the actual signal for
        "mistake-prone young player" vs. "established veteran").

    Notes:
        nflverse's weekly roster snapshots (import_weekly_rosters) carry a
        small number of rows with no player_id or no age -- an empty
        player_id string (not a real player row), or a roster entry with no
        birth date on file yet (verified: 194/46849 rows missing age for
        the 2025 season alone; years_exp itself had zero nulls in the same
        check). Both are dropped here rather than passed through as
        unusable rows; a handful of (player_id, season, week) keys in
        nflverse's own source data also repeat (8/46849 for 2025 -- e.g. a
        mid-week team change), so duplicates are resolved by keeping the
        first occurrence, same as any other minor upstream data quirk this
        file already works around (e.g. the stale-release-tag note in
        get_player_weekly_stats_nflverse()).
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    years = years if years is not None else default_recent_seasons()
    rosters = nfl.import_weekly_rosters(years)

    age = rosters[["player_id", "season", "week", "age", "years_exp"]]
    age = age[(age["player_id"] != "") & age["player_id"].notna() & age["age"].notna()]
    age = age.drop_duplicates(subset=["player_id", "season", "week"]).reset_index(drop=True)

    key_columns = ["player_id", "season", "week"]
    if db_path:
        save_to_sqlite(age, "weekly_player_age", key_columns=key_columns, db_path=db_path)
    if save_path:
        save_to_csv(age, save_path)
    return age


def get_weekly_injury_reports(
    years: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    db_path: Optional[str] = "data/nfl_stats.db",
) -> pd.DataFrame:
    """
    Pull the official weekly injury report -- each player's game-status
    designation (Questionable/Doubtful/Out) as announced ahead of that
    week's game, by real player_id (nflverse's injury data already carries
    gsis_id directly, no Pro-Football-Reference crosswalk needed here,
    unlike get_weekly_defensive_target_stats()).

    Args:
        years: Season years to pull, e.g. [2023, 2024, 2025]. Defaults to
            default_recent_seasons() if not given. nflverse publishes the
            current season's reports as they're announced, so this also
            covers the upcoming week once a team's report is out (usually
            Wednesday-Friday before a Sunday game).
        save_path: CSV filename to write. Pass None (default) to skip the CSV.
        db_path: SQLite database file to upsert into (table
            "weekly_injury_reports", keyed on player_id/season/week). Pass
            None to skip SQLite.

    Returns:
        DataFrame with one row per (player_id, season, week) that had a
        report at all -- a player with NO row for a given week has no
        reported injury concern that week. Columns: team, report_status
        ("Questionable"/"Doubtful"/"Out"), report_primary_injury (free-text,
        e.g. "Hamstring").

    Notes:
        A player ruled fully "Out" essentially never has a real stat line
        for that game (they didn't play), so this status mostly won't
        co-occur with a real player_weekly_stats row to train against --
        the useful training signal here is overwhelmingly from
        Questionable/Doubtful players who DID play through a designation,
        often with reduced usage. Absence of any row for a player-week is
        NOT the same as "confirmed healthy" -- it just means no game-status
        concern was ever reported, which is the normal case for the vast
        majority of players most weeks.
    """
    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")

    years = years if years is not None else default_recent_seasons()
    injuries = nfl.import_injuries(years)

    result = injuries.rename(columns={"gsis_id": "player_id"})
    result = result[["player_id", "season", "week", "team", "report_status", "report_primary_injury"]]
    result = result.dropna(subset=["player_id", "report_status"]).reset_index(drop=True)

    key_columns = ["player_id", "season", "week"]
    if db_path:
        save_to_sqlite(result, "weekly_injury_reports", key_columns=key_columns, db_path=db_path)
    if save_path:
        save_to_csv(result, save_path)
    return result


# ===========================================================================
# PLAYER PROP LINES -- sportsbook over/under lines, logged pregame for paper trading
# ===========================================================================
# The Odds API (the-odds-api.com, key in env ODDS_API_KEY). Each event-odds call
# costs one credit per market per region; whether the free plan includes player
# props is unconfirmed -- the first call tells (an empty or 4xx reply).
PROP_MARKETS = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds", "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions", "player_reception_yds": "receiving_yards",
}
# What get_prop_lines pulls by default: the markets the Prop Board and paper trades use
# (receptions since 2026-10-17, for the rec55 rule). Each market costs a credit per game.
YARD_MARKETS = ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]


def parse_prop_lines(event: dict, retrieved_at: str) -> pd.DataFrame:
    """
    One Odds API event-odds reply -> one row per bookmaker, market, player and
    side (Over/Under): line, American price, plus the event's id and kickoff.
    Markets outside PROP_MARKETS and outcomes without a line are dropped.
    """
    rows = []
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            if market.get("key") not in PROP_MARKETS:
                continue
            for outcome in market.get("outcomes") or []:
                if outcome.get("point") is None or not outcome.get("description"):
                    continue
                rows.append({
                    "retrieved_at": retrieved_at, "event_id": event.get("id"), "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"), "away_team": event.get("away_team"),
                    "bookmaker": book.get("key"), "book_last_update": market.get("last_update") or book.get("last_update"),
                    "market": market["key"], "stat": PROP_MARKETS[market["key"]],
                    "player_name": outcome["description"], "side": outcome.get("name"),
                    "line": float(outcome["point"]), "price": float(outcome["price"]),
                })
    return pd.DataFrame(rows)


def parse_prop_sids(event: dict, retrieved_at: str) -> pd.DataFrame:
    """
    DraftKings' own outcome ids from an event-odds reply pulled with includeSids=true: one row per market,
    player and side with DraftKings' line and sid. The Prop Board builds betslip links from them:
    https://sportsbook.draftkings.com/?outcomes=<sid>[+<sid>...] (# in a sid is %23). Multi-leg links
    confirmed by hand on 2026-10-29.
    """
    rows = []
    for book in event.get("bookmakers") or []:
        if book.get("key") != "draftkings":
            continue
        for market in book.get("markets") or []:
            if market.get("key") not in PROP_MARKETS:
                continue
            for outcome in market.get("outcomes") or []:
                if outcome.get("point") is None or not outcome.get("description") or not outcome.get("sid"):
                    continue
                rows.append({"retrieved_at": retrieved_at, "event_id": event.get("id"), "market": market["key"],
                             "stat": PROP_MARKETS[market["key"]], "player_name": outcome["description"],
                             "side": outcome.get("name"), "line": float(outcome["point"]), "sid": str(outcome["sid"])})
    return pd.DataFrame(rows)


def get_prop_lines(api_key: str, days_ahead: float = 7.0, markets: Optional[List[str]] = None,
                   db_path: Optional[str] = "data/nfl_stats.db", table: str = "prop_lines") -> pd.DataFrame:
    """
    Pull current NFL player prop lines for every game kicking off in the next
    `days_ahead` days and append them to "prop_lines" (one snapshot per call,
    keyed on retrieval time, so repeated pulls show line movement).

    Only games not yet started are pulled -- a line logged here is pregame by
    construction, which is the point: it is what a bet would have been made at.
    Costs one credit per market per game (US region only).
    """
    import json
    import urllib.parse
    import urllib.request
    from datetime import timedelta, timezone

    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
    now = datetime.now(timezone.utc)
    retrieved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch(path: str, **params):
        query = urllib.parse.urlencode({"apiKey": api_key, **params})
        with urllib.request.urlopen(f"{base}{path}?{query}", timeout=60) as response:
            return json.load(response), response.headers.get("x-requests-remaining")

    events, _ = fetch("/events", dateFormat="iso")
    horizon = now + timedelta(days=days_ahead)
    upcoming = [ev for ev in events
                if now < datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) <= horizon]
    frames, sid_frames, remaining = [], [], None
    for ev in upcoming:
        # includeSids costs no extra credits; DraftKings' sids feed the board's betslip links (prop_sids)
        reply, remaining = fetch(f"/events/{ev['id']}/odds", regions="us", oddsFormat="american",
                                 markets=",".join(markets or YARD_MARKETS), dateFormat="iso", includeSids="true")
        frames.append(parse_prop_lines(reply, retrieved_at))
        sid_frames.append(parse_prop_sids(reply, retrieved_at))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sids = pd.concat(sid_frames, ignore_index=True) if sid_frames else pd.DataFrame()
    print(f"{len(upcoming)} upcoming games, {len(result)} prop lines ({len(sids)} DraftKings sids); API credits remaining: {remaining}")
    if db_path and len(result):
        save_to_sqlite(result, table, key_columns=["retrieved_at", "event_id", "bookmaker", "market",
                                                   "player_name", "side"], db_path=db_path)
    if db_path and len(sids):
        save_to_sqlite(sids, "prop_sids", key_columns=["retrieved_at", "event_id", "market", "player_name", "side"], db_path=db_path)
    return result


GAME_MARKETS = ["h2h", "spreads", "totals"]


def parse_game_lines(events: list, retrieved_at: str) -> pd.DataFrame:
    """
    The Odds API /odds reply (many events) -> one row per event, bookmaker,
    market (h2h = moneyline, spreads, totals) and outcome: the outcome's name (a
    team, or Over/Under), its point (spread or total; none for moneyline) and its
    American price.
    """
    rows = []
    for ev in events:
        for book in ev.get("bookmakers") or []:
            for market in book.get("markets") or []:
                if market.get("key") not in GAME_MARKETS:
                    continue
                for outcome in market.get("outcomes") or []:
                    rows.append({
                        "retrieved_at": retrieved_at, "event_id": ev.get("id"), "commence_time": ev.get("commence_time"),
                        "home_team": ev.get("home_team"), "away_team": ev.get("away_team"),
                        "bookmaker": book.get("key"), "book_last_update": market.get("last_update") or book.get("last_update"),
                        "market": market["key"], "outcome": outcome.get("name"),
                        "point": None if outcome.get("point") is None else float(outcome["point"]),
                        "price": float(outcome["price"]),
                    })
    return pd.DataFrame(rows)


def get_game_lines(api_key: str, db_path: Optional[str] = "data/nfl_stats.db") -> pd.DataFrame:
    """
    Pull moneyline, spread and total from every US book for all NFL games not yet
    started and append them to "game_lines" (one snapshot per call, keyed on
    retrieval time, so repeated pulls show line movement). One call covers every
    game: about 3 credits (markets x regions).
    """
    import json
    import urllib.parse
    import urllib.request
    from datetime import timezone

    now = datetime.now(timezone.utc)
    retrieved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.urlencode({"apiKey": api_key, "regions": "us", "markets": ",".join(GAME_MARKETS),
                                    "oddsFormat": "american", "dateFormat": "iso"})
    with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?{query}", timeout=60) as r:
        events, remaining = json.load(r), r.headers.get("x-requests-remaining")
    upcoming = [ev for ev in events if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) > now]
    result = parse_game_lines(upcoming, retrieved_at)
    print(f"{len(upcoming)} upcoming games, {len(result)} game lines; API credits remaining: {remaining}")
    if db_path and len(result):
        save_to_sqlite(result, "game_lines", key_columns=["retrieved_at", "event_id", "bookmaker", "market", "outcome"],
                       db_path=db_path)
    return result


# ===========================================================================
# SLEEPER PROJECTIONS -- an outside projection to blend with
# ===========================================================================
# Sleeper's public, unauthenticated (and undocumented) projections endpoint; the
# numbers are Rotowire's. Stat keys mapped to this project's column names.
SLEEPER_STATS = {
    "pts_ppr": "fantasy_points_ppr", "pass_cmp": "completions", "pass_att": "attempts",
    "pass_yd": "passing_yards", "pass_td": "passing_tds", "pass_int": "passing_interceptions",
    "rush_att": "carries", "rush_yd": "rushing_yards", "rush_td": "rushing_tds",
    "rec": "receptions", "rec_tgt": "targets", "rec_yd": "receiving_yards", "rec_td": "receiving_tds",
}


def parse_sleeper_projections(records: list, sleeper_to_gsis: Dict[str, str], season: int, week: int,
                              retrieved_at: str) -> pd.DataFrame:
    """
    Sleeper's projection records -> one row per (season, week, player_id) with
    sleeper_<stat> columns, keyed by nflverse's gsis id. Records with no PPR
    projection, or whose player has no gsis id, are dropped. A stat Sleeper
    leaves out is 0 (it omits zeros).
    """
    rows = []
    for rec in records:
        stats = rec.get("stats") or {}
        gsis = sleeper_to_gsis.get(str(rec.get("player_id")))
        if stats.get("pts_ppr") is None or not gsis:
            continue
        row = {"season": season, "week": week, "player_id": gsis,
               "position": (rec.get("player") or {}).get("position"),
               "last_modified": datetime.utcfromtimestamp(rec["last_modified"] / 1000).isoformat() + "+00:00"
               if rec.get("last_modified") else None,
               "retrieved_at": retrieved_at}
        row.update({f"sleeper_{ours}": float(stats.get(theirs) or 0.0) for theirs, ours in SLEEPER_STATS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _sleeper_crosswalk() -> Dict[str, str]:
    """Sleeper player id -> nflverse gsis id, from nflverse's ID crosswalk (Sleeper's own gsis_id is mostly blank)."""
    ids = nfl.import_ids()[["sleeper_id", "gsis_id"]].dropna()
    return dict(zip(ids["sleeper_id"].astype(float).astype(int).astype(str), ids["gsis_id"]))


def get_sleeper_injuries(db_path: Optional[str] = "data/nfl_stats.db") -> pd.DataFrame:
    """
    Sleeper's current injury status for every QB/RB/WR/TE (player_id = gsis id,
    injury_status e.g. Questionable/Doubtful/Out/IR/PUP; None when healthy), with
    our retrieval time. A snapshot of right now -- appended to "sleeper_injuries"
    keyed on retrieved_at + player_id, so past snapshots stay as they were.
    """
    import json
    import urllib.request

    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")
    retrieved_at = datetime.utcnow().isoformat(timespec="seconds") + "+00:00"
    with urllib.request.urlopen("https://api.sleeper.app/v1/players/nfl", timeout=120) as response:
        players = json.load(response)
    crosswalk = _sleeper_crosswalk()
    rows = [{"player_id": crosswalk[pid], "injury_status": p.get("injury_status") or None,
             "injury_body_part": p.get("injury_body_part"), "retrieved_at": retrieved_at}
            for pid, p in players.items() if p.get("position") in ("QB", "RB", "WR", "TE") and pid in crosswalk]
    result = pd.DataFrame(rows)
    if db_path and len(result):
        save_to_sqlite(result, "sleeper_injuries", key_columns=["retrieved_at", "player_id"], db_path=db_path)
    return result


def get_sleeper_projections(season: int, week: int, db_path: Optional[str] = "data/nfl_stats.db") -> pd.DataFrame:
    """
    Pull Sleeper's QB/RB/WR/TE projections for one regular-season week and
    upsert them into "sleeper_projections" (keyed on season/week/player_id).

    Pregame only if pulled before kickoff: `retrieved_at` records when this
    call ran, and `last_modified` is Sleeper's own stamp. Historical weeks
    pulled after the fact were last modified the day after their games, so
    they may hold post-kickoff edits -- fine for a rough comparison, not proof.
    Players map to nflverse ids through nflverse's ID crosswalk, because
    Sleeper leaves its own gsis_id blank for most players.
    """
    import json
    import urllib.request

    if not _HAS_NFL_DATA_PY:
        raise ImportError("Install with: pip install nfl_data_py")
    url = (f"https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular"
           "&position%5B%5D=QB&position%5B%5D=RB&position%5B%5D=WR&position%5B%5D=TE")
    retrieved_at = datetime.utcnow().isoformat(timespec="seconds") + "+00:00"
    with urllib.request.urlopen(url, timeout=60) as response:
        records = json.load(response)
    result = parse_sleeper_projections(records, _sleeper_crosswalk(), season, week, retrieved_at)
    if db_path and len(result):
        save_to_sqlite(result, "sleeper_projections", key_columns=["season", "week", "player_id"], db_path=db_path)
    return result


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    # Both calls default to the last 3 seasons when `years` is omitted
    # (see default_recent_seasons()) -- pass an explicit years list to
    # override, e.g. get_player_weekly_stats_nflverse([2022, 2023]).
    weekly = get_player_weekly_stats_nflverse(save_path="player_weekly_recent.csv")
    get_team_season_stats_nflverse(save_path="team_season_recent.csv")

    # Print each position's current-season leaderboard as a sanity check
    # that the pull came back looking right, e.g. "top 10 QBs by passing
    # yards this season" -- see get_position_leaders().
    latest_season = int(weekly["season"].max())
    for position in list(POSITION_RANKING_STAT) + ["CB"]:
        leaders = get_position_leaders(weekly, position, season=latest_season)
        print(f"\n=== {latest_season} {position} leaders ===")
        print(leaders.to_string(index=False))
