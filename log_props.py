"""
Log this week's player prop lines and freeze the projections they will be judged
against, both before kickoff -- the record a paper-trading test needs.

    ODDS_API_KEY=... python3 log_props.py 2026 6 [days_ahead]

1. Pulls current lines for games in the next days_ahead days (default 7) into data/nfl_stats.db
   (table prop_lines, one snapshot per run: run it again closer to kickoff to see
   lines move).
2. Copies projections/<season>_w<NN>_*.csv -- run project_week() for that week
   first -- to projections/snapshots/<season>_w<NN>_<retrieved>/, so a later
   re-run of project_week() cannot rewrite what the projection said at the time.

3. Pulls moneyline, spread and total for every game not yet started (~3 credits)
   into the game_lines table -- a failure here is reported and does not stop the rest.
4. Writes each player's consensus line -- the median Over line across books --
   into the week's projection CSVs as line_<stat> (top-level and snapshot), so the
   proj-charts pane can show it. Sportsbook names are matched to nflverse ids by
   normalized display name.

Nothing here bets or decides anything; scoring the log comes later, once a few
hundred lines have been played.
"""
import glob
import os
import re
import shutil
import sys

import pandas as pd

from StatScraper import get_game_lines, get_prop_lines, query_sqlite

DB_PATH = "data/nfl_stats.db"


def norm_name(name: str) -> str:
    """'Marvin Harrison Jr.' and 'marvin harrison' -> 'marvin harrison'."""
    words = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " ")).split()
    return " ".join(w for w in words if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def name_to_id(players: pd.DataFrame) -> dict:
    """normalized display name -> player_id; names two players share are left out rather than guessed."""
    keys = players.assign(k=players.player_display_name.map(norm_name)).drop_duplicates(["k", "player_id"])
    keys = keys[~keys.k.duplicated(keep=False)]
    return dict(zip(keys.k, keys.player_id))


def consensus_lines(lines: pd.DataFrame, ids: dict) -> pd.DataFrame:
    """Latest pull only: one row per player_id, line_<stat> = median Over line across books."""
    latest = lines[(lines.retrieved_at == lines.retrieved_at.max()) & (lines.side == "Over")]
    latest = latest.assign(player_id=latest.player_name.map(norm_name).map(ids)).dropna(subset=["player_id"])
    wide = latest.groupby(["player_id", "stat"]).line.median().unstack("stat")
    return wide.add_prefix("line_").reset_index()


def week_only(lines: pd.DataFrame, game_dates: set) -> pd.DataFrame:
    """Lines whose kickoff falls on one of `game_dates` (US Eastern 'YYYY-MM-DD'), i.e.
    the week being logged -- a pull's window can also reach the next week's games."""
    kickoff = pd.to_datetime(lines["commence_time"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return lines[kickoff.isin(game_dates)]


def attach_lines(files: list, consensus: pd.DataFrame) -> None:
    """Replace any line_ columns in each projection CSV with `consensus`."""
    for f in files:
        d = pd.read_csv(f)
        d = d.drop(columns=[c for c in d if c.startswith("line_")]).merge(consensus, on="player_id", how="left")
        d.to_csv(f, index=False)


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: ODDS_API_KEY=... python3 log_props.py <season> <week> [days_ahead]")
    season, week = int(sys.argv[1]), int(sys.argv[2])
    days_ahead = float(sys.argv[3]) if len(sys.argv) == 4 else 7.0
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY (from the-odds-api.com) in the environment; it is never written anywhere.")
    files = sorted(glob.glob(f"projections/{season}_w{week:02d}_*.csv"))
    if not files:
        sys.exit(f"No projections for {season} week {week}: run project_week() for it first.")
    try:
        get_game_lines(key)
    except Exception as ex:
        print(f"Game lines not pulled ({type(ex).__name__}: {ex}).")
    lines = get_prop_lines(key, days_ahead=days_ahead)
    if lines.empty:
        sys.exit(f"No prop lines came back (no games in the next {days_ahead:g} days, or the plan does not include props).")
    stamp = lines["retrieved_at"].iloc[0].replace(":", "")
    snap = f"projections/snapshots/{season}_w{week:02d}_{stamp}"
    os.makedirs(snap, exist_ok=True)
    for f in files:
        shutil.copy2(f, snap)
    players = query_sqlite("SELECT DISTINCT player_id, player_display_name FROM player_weekly_stats "
                           "WHERE position_group IN ('QB', 'RB', 'WR', 'TE')", db_path=DB_PATH)
    import nfl_data_py as nfl
    sched = nfl.import_schedules([season])
    dates = set(sched.loc[(sched.game_type == "REG") & (sched.week == week), "gameday"].astype(str))
    week_lines = week_only(lines, dates)
    print(f"{len(week_lines)} of {len(lines)} lines are for week {week}'s games.")
    consensus = consensus_lines(week_lines, name_to_id(players))
    attach_lines(files + sorted(glob.glob(f"{snap}/*.csv")), consensus)
    print(f"Consensus lines for {len(consensus)} players written into the projection CSVs.")
    print(f"Logged {len(lines)} lines ({lines.player_name.nunique()} players, {lines.bookmaker.nunique()} books); "
          f"projections frozen in {snap}")


if __name__ == "__main__":
    main()
