"""
Regression checks for bugs that reached the scheduled jobs -- each test is named for the failure it
pins, and each one failed against the code before its fix. Synthetic data, no network, no database:
the SQLite, schedule, id-crosswalk and ESPN calls are swapped for stand-ins.

Run: .venv/bin/python tests/test_regressions.py   (the scheduled jobs' interpreter: pandas 3)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import tempfile
import urllib.error
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import log_closing
import paper_trades as pt
import site_data


def swapped(module, **attrs):
    """Set module attributes; returns a function that restores them."""
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    return lambda: [setattr(module, k, v) for k, v in old.items()]


def demo_settle_all_blank_ledger():
    """2026-10-15 job: 'Invalid value 'win' for dtype float64' -- a ledger with no settled bets yet
    reads its result column as numbers, and writing 'win' into it raised."""
    path = os.path.join(tempfile.mkdtemp(), "ledger.csv")
    pd.DataFrame({"season": 2026, "week": 6, "player_id": ["a", "b", "c"], "stat": "receiving_yards",
                  "side": ["Over", "Under", "Over"], "line": 40.5, "price": -110.0, "stake": 10.0,
                  "actual": np.nan, "result": np.nan, "profit": np.nan}).to_csv(path, index=False)
    stats = pd.DataFrame({"player_id": ["a", "b"], "passing_yards": 0, "rushing_yards": 0,
                          "receiving_yards": [55.0, 60.0], "receptions": 4})
    restore = swapped(pt, LEDGERS={"tmp": path}, query_sqlite=lambda *a, **k: stats.copy(),
                      closing=lambda led: pd.DataFrame(index=led.index, columns=["close_line", "close_price", "clv_yards"]))
    try:
        with warnings.catch_warnings():   # pandas 2 only warns where pandas 3 (the job's .venv) raises
            warnings.simplefilter("error", FutureWarning)
            out = pt.settle(2026, 6, "tmp")
    finally:
        restore()
    assert out.result.tolist() == ["win", "loss", "void"], out.result.tolist()
    assert round(out.profit.iloc[0], 2) == 9.09 and out.profit.iloc[1] == -10 and out.profit.iloc[2] == 0
    print("OK -- settle writes results into a ledger that has never been settled.")


def demo_settle_closing_none_into_float():
    """2026-10-24 job: 'Invalid value [None None ...]' -- re-settling a week whose close_price column
    was already numeric, with closing() answering None for a price ESPN never gives."""
    path = os.path.join(tempfile.mkdtemp(), "ledger.csv")
    pd.DataFrame({"season": 2026, "week": 6, "player_id": ["a", "b"], "stat": "receiving_yards", "side": "Under",
                  "line": 40.5, "price": -110.0, "stake": 10.0, "actual": [30.0, 50.0], "result": ["win", "loss"],
                  "profit": [9.09, -10.0], "close_line": [41.5, 39.5], "close_price": np.nan, "clv_yards": [-1.0, 1.0]}).to_csv(path, index=False)
    cl = lambda led: pd.DataFrame({"close_line": [42.5, None], "close_price": [None, None], "clv_yards": [-2.0, None]}, index=led.index)
    restore = swapped(pt, LEDGERS={"tmp": path}, query_sqlite=lambda *a, **k: pd.DataFrame(columns=["player_id"]), closing=cl)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = pt.settle(2026, 6, "tmp")
    finally:
        restore()
    assert out.close_line.tolist()[0] == 42.5 and np.isnan(out.close_line.tolist()[1]) and out.close_price.isna().all()
    print("OK -- re-settling writes missing closing prices as NaN.")


def demo_median_price_missing_prices():
    """2026-10-17 job: both settles crashed ('<' not supported between NoneType and int) -- ESPN's
    closing rows carry price=None, and median_price passed them to payout()."""
    assert round(pt.median_price([None, -110, -120]), 1) == -114.8
    assert np.isnan(pt.median_price([None])) and np.isnan(pt.median_price([np.nan]))
    print("OK -- median_price skips missing prices.")


def demo_logged_actuals_non_yardage_lines():
    """2026-10-15 job: board data failed with KeyError 'passing_tds' -- older pulls logged TD lines,
    and logged_actuals looked each logged stat up in the yardage columns."""
    lines = pd.DataFrame({"player_name": "Patrick Mahomes", "commence_time": "2026-10-11T17:00:00Z",
                          "stat": ["passing_yards", "passing_tds", "receptions"]})
    stats = pd.DataFrame({"player_id": ["00-1"], "week": [5], "passing_yards": [280.0], "rushing_yards": [12.0],
                          "receiving_yards": [0.0], "receptions": [0.0]})
    sched = pd.DataFrame({"game_type": "REG", "week": [5], "gameday": ["2026-10-11"]})
    sql = lambda q, **k: (lines if "prop_lines" in q else stats).copy()
    restore = swapped(site_data, query_sqlite=sql)
    restore_nfl = swapped(site_data.nfl, import_schedules=lambda seasons: sched.copy())
    try:
        out = site_data.logged_actuals(2026, 6, pd.DataFrame({"player_id": ["00-1"], "player_display_name": ["Patrick Mahomes"]}))
    finally:
        restore()
        restore_nfl()
    assert out == {"2026:5:00-1:passing_yards": 280.0, "2026:5:00-1:receptions": 0.0}, out
    print("OK -- logged_actuals skips stats it doesn't settle and handles receptions.")


def demo_closing_lines_skip_unposted_game():
    """2026-10-17 job: board data failed with HTTP Error 404 -- one game's ESPN prop feed wasn't
    posted yet, and the whole pull died with it."""
    soon = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ")
    events = [{"id": i, "date": soon, "competitions": [{"competitors": []}]} for i in ("e1", "e2")]
    item = {"type": {"name": "Total Receptions (incl. overtime)"}, "athlete": {"$ref": "x/athletes/77?lang=en"},
            "current": {"target": {"value": 4.5}}, "open": {"target": {"value": 4.5}}}

    def get_json(url):
        if "scoreboard" in url:
            return {"events": events}
        if "propBets" in url:
            if "/events/e1/" in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            return {"items": [item]}
        return {}   # game odds: no DraftKings item

    restore = swapped(log_closing, get_json=get_json)
    restore_nfl = swapped(log_closing.nfl, import_ids=lambda: pd.DataFrame({"espn_id": [77.0], "gsis_id": ["00-7"]}))
    try:
        out = log_closing.main(hours_ahead=1.5, db_path=None)
    finally:
        restore()
        restore_nfl()
    assert len(out) == 1 and out.event_id.iloc[0] == "e2" and out.player_id.iloc[0] == "00-7" and out.stat.iloc[0] == "receptions"
    print("OK -- a game whose prop feed 404s is skipped, the rest still load.")


if __name__ == "__main__":
    demo_settle_all_blank_ledger()
    demo_settle_closing_none_into_float()
    demo_median_price_missing_prices()
    demo_logged_actuals_non_yardage_lines()
    demo_closing_lines_skip_unposted_game()
