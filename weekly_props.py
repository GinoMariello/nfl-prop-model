"""
Thursday- and Saturday-morning job: project the upcoming week, then log its prop lines.

    python3 weekly_props.py            # what the scheduled launchd job runs

1. Picks the regular-season week whose games have not all been played yet
   (nflverse schedule).
2. project_week() (skipping players listed Out/IR/PUP right now) and rest_of_season()
   for QB/RB/WR/TE -- a position that fails is reported and skipped.
3. log_props.py's logging for that week (needs the API key, see below). On a
   Thursday only the next 24 hours (Thursday night's game); otherwise 7 days.
   Saturday's run then covers Sunday and Monday. Yardage and receptions markets (4 credits a
   game) plus ~3 for game lines: ~50 credits a week, plus ~45 for log_closing.py.
4. Paper trades: settle last week's open bets, place this week's (pregame only,
   so Thursday bets Thursday's game and Saturday the rest) -- the main rule (55%
   filtered, since week 6) and the plain 55% rule; the retired base rule only settles.
5. Refresh the game model's win chances for the season (game_picks.py, ~2 min).
6. Rebuild the Prop Board data, site/data.json. Publishing the page itself is a
   scheduled publish task (publish-prop-board).

The Odds API key comes from env ODDS_API_KEY, else the file
~/.config/nfl-claude/odds_api_key (create it yourself, chmod 600). Output goes to
projections/logs/weekly_props.log via the launchd job.
"""
import os
import socket
import subprocess
import sys
import time
from datetime import date

import nfl_data_py as nfl

import game_picks
import paper_trades
import PlayerProjections as pp
import site_data

KEY_FILE = os.path.expanduser("~/.config/nfl-claude/odds_api_key")


# nflverse/Sleeper downloads have no timeout of their own: a stalled GitHub download hung the
# 2026-10-24 run for six hours. With this, a stall raises and the per-position retry takes over.
socket.setdefaulttimeout(120)


def upcoming_week(today: date) -> tuple:
    """(season, week) of the first regular-season week with a game on or after today."""
    season = today.year if today.month >= 3 else today.year - 1
    sched = nfl.import_schedules([season])
    sched = sched[(sched.game_type == "REG") & (sched.gameday >= today.isoformat())]
    if sched.empty:
        sys.exit(f"No regular-season games left in {season}.")
    return season, int(sched.week.min())


def main() -> None:
    season, week = upcoming_week(date.today())
    print(f"== {date.today()}: {season} week {week}", flush=True)
    for pos in ("QB", "RB", "WR", "TE"):
        for attempt in (1, 2):   # one retry: a dropped connection skipped QB on 2026-10-22
            try:
                pp.project_week(pos, season, week, skip_injured=True)
                pp.rest_of_season(pos, season, week)
                print(f"{pos}: projected (+ rest of season)", flush=True)
                break
            except Exception as ex:
                print(f"{pos}: FAILED (attempt {attempt}) {type(ex).__name__}: {ex}", flush=True)
                if attempt == 1:
                    time.sleep(60)
    env = dict(os.environ)
    if not env.get("ODDS_API_KEY") and os.path.exists(KEY_FILE):
        env["ODDS_API_KEY"] = open(KEY_FILE).read().strip()
    days_ahead = "1" if date.today().weekday() == 3 else "7"  # Thursday: tonight's game only
    logged = subprocess.call([sys.executable, "log_props.py", str(season), str(week), days_ahead], env=env)
    # Main rule since 2026 week 6: hit55f (55% filtered). The original rule ("base") only
    # settles now, so its last bets (week 5) get scored; the plain 55% rule runs alongside.
    for step, run in [("settle last week (main: 55% filtered)", lambda: paper_trades.settle(season, week - 1, "hit55f")),
                      ("place bets (main: 55% filtered)", lambda: paper_trades.place(season, week, "hit55f")),
                      ("settle last week (55% rule)", lambda: paper_trades.settle(season, week - 1, "hit55")),
                      ("place bets (55% rule)", lambda: paper_trades.place(season, week, "hit55")),
                      ("settle last week (receptions rule)", lambda: paper_trades.settle(season, week - 1, "rec55")),
                      ("place bets (receptions rule)", lambda: paper_trades.place(season, week, "rec55")),
                      ("settle last week (retired base rule)", lambda: paper_trades.settle(season, week - 1)),
                      ("game model picks", lambda: game_picks.main(season)),
                      ("prop board data", lambda: site_data.main(season, week))]:
        try:
            out = run()
            print(f"{step}: {'done' if out is None else f'{len(out)} bets'}", flush=True)
        except Exception as ex:
            print(f"{step}: FAILED {type(ex).__name__}: {ex}", flush=True)
    sys.exit(logged)


if __name__ == "__main__":
    main()
