"""
Log closing lines for games kicking off in the next few hours, from ESPN's public
DraftKings odds feed (free, no API credits):
  - prop_lines_close: every player prop market with a line -- yardage (scored for CLV by
    paper_trades.settle), receptions, carries, attempts, completions, passing TDs,
    interceptions, combined yards and milestones (e.g. "15+"). Touchdown-scorer markets are
    skipped: ESPN lists them with no line or odds.
  - game_lines_close: spread, total and moneyline with prices, current and opening.

    python3 log_closing.py [hours_ahead]     # default 1.5; launchd (com.ginom.nfl-closing-lines) runs it before each kickoff window

ESPN gives one book (DraftKings); props come with the line only -- no prices -- plus the opening line. Players come keyed by ESPN athlete id, mapped to nflverse player_id with nflverse's
id crosswalk. Rows go to the prop_lines_close table, never prop_lines.
"""
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import nfl_data_py as nfl
import pandas as pd

from StatScraper import save_to_sqlite

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
PROPS = ("https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}"
         "/odds/100/propBets?lang=en&region=us&limit=1000")   # provider 100 = DraftKings
ODDS = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}/odds"
STATS = {"Total Passing Yards (incl. overtime)": "passing_yards", "Total Rushing Yards (incl. overtime)": "rushing_yards",
         "Total Receiving Yards (incl. overtime)": "receiving_yards"}


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def stat_key(market: str) -> str:
    """ESPN market name -> stat key: the yardage names paper_trades uses, else a slug
    ("Total Receptions (incl. overtime)" -> "receptions", "Receiving Yards Milestones" -> "receiving_yards_milestones")."""
    if market in STATS:
        return STATS[market]
    return re.sub(r"[^a-z0-9]+", "_", market.lower().replace("(incl. overtime)", "").replace("total ", "")).strip("_")


def parse_props(reply: dict, event: dict, espn_to_gsis: dict, retrieved_at: str) -> pd.DataFrame:
    """ESPN propBets reply -> one row per player, market and line (ESPN lists each prop twice,
    once per side; milestone markets list several lines per player)."""
    rows = {}
    for it in reply.get("items", []):
        market = (it.get("type") or {}).get("name") or ""
        m = re.search(r"/athletes/(\d+)", (it.get("athlete") or {}).get("$ref", ""))
        line = ((it.get("current") or {}).get("target") or {}).get("value")
        if not m or line is None:   # touchdown-scorer markets come with no line
            continue
        stat = stat_key(market)
        rows[(m.group(1), stat, line)] = {
            "retrieved_at": retrieved_at, "event_id": event["id"], "commence_time": event["date"],
            "bookmaker": "draftkings (espn)", "market": market, "stat": stat, "espn_id": m.group(1), "player_id": espn_to_gsis.get(m.group(1)),
            "side": "Over", "line": float(line), "price": None,
            "open_line": ((it.get("open") or {}).get("target") or {}).get("value"),
        }
    return pd.DataFrame(rows.values())


def parse_game(odds: dict, event: dict, retrieved_at: str) -> dict:
    """ESPN competition odds (DraftKings item) -> one row: home spread, total, moneylines and prices, current and opening."""
    g = next((i for i in odds.get("items", []) if (i.get("provider") or {}).get("id") == "100"), None)
    if g is None:
        return {}
    side = {c["homeAway"]: c["team"]["displayName"] for c in event["competitions"][0]["competitors"]}
    am = lambda d, k: float((d.get(k) or {}).get("american")) if (d.get(k) or {}).get("american") not in (None, "") else None
    o = g.get("open") or {}
    return {"retrieved_at": retrieved_at, "event_id": event["id"], "commence_time": event["date"],
            "home_team": side.get("home"), "away_team": side.get("away"), "bookmaker": "draftkings (espn)",
            "spread": g.get("spread"), "total": g.get("overUnder"), "over_odds": g.get("overOdds"), "under_odds": g.get("underOdds"),
            "ml_home": (g.get("homeTeamOdds") or {}).get("moneyLine"), "ml_away": (g.get("awayTeamOdds") or {}).get("moneyLine"),
            "spread_odds_home": (g.get("homeTeamOdds") or {}).get("spreadOdds"), "spread_odds_away": (g.get("awayTeamOdds") or {}).get("spreadOdds"),
            "open_total": am(o, "total"), "open_over_odds": am(o, "over"), "open_under_odds": am(o, "under")}


def main(hours_ahead: float = 1.5, db_path: Optional[str] = "data/nfl_stats.db") -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    retrieved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    # The default scoreboard lags a week behind until it rolls over, so ask for each US-Eastern date in the window.
    et = ZoneInfo("America/New_York")
    days = {(now + timedelta(hours=h)).astimezone(et).strftime("%Y%m%d") for h in range(0, int(hours_ahead) + 2)}
    events = {e["id"]: e for d in sorted(days) for e in get_json(f"{SCOREBOARD}?dates={d}")["events"]}.values()
    events = [e for e in events if now < datetime.fromisoformat(e["date"].replace("Z", "+00:00")) <= now + timedelta(hours=hours_ahead)]
    if not events:
        print(f"{retrieved_at}: no games in the next {hours_ahead:g} hours")
        return pd.DataFrame()
    ids = nfl.import_ids()[["espn_id", "gsis_id"]].dropna()
    espn_to_gsis = dict(zip(ids.espn_id.astype(float).astype(int).astype(str), ids.gsis_id))
    def props(e):
        try:
            return parse_props(get_json(PROPS.format(id=e["id"])), e, espn_to_gsis, retrieved_at)
        except urllib.error.HTTPError as err:   # a game whose props aren't posted yet
            if err.code != 404:
                raise
            return pd.DataFrame()
    out = pd.concat([props(e) for e in events], ignore_index=True)
    games = pd.DataFrame([r for r in (parse_game(get_json(ODDS.format(id=e["id"])), e, retrieved_at) for e in events) if r])
    print(f"{retrieved_at}: {len(events)} games, {len(out)} closing prop lines in {out.stat.nunique() if len(out) else 0} markets "
          f"({out.player_id.notna().sum() if len(out) else 0} matched to nflverse ids), {len(games)} game lines")
    if db_path and len(out):
        save_to_sqlite(out, "prop_lines_close", key_columns=["retrieved_at", "event_id", "espn_id", "stat", "line"], db_path=db_path)
    if db_path and len(games):
        save_to_sqlite(games, "game_lines_close", key_columns=["retrieved_at", "event_id"], db_path=db_path)
    return out


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.5)
