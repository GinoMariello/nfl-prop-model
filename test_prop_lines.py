"""
Check for parse_prop_lines(): an Odds API event reply becomes one row per book,
market, player and side, dropping unknown markets and outcomes without a line.

Synthetic, no network.

Run: python3 test_prop_lines.py
"""
import pandas as pd

from StatScraper import parse_game_lines, parse_prop_lines
from log_props import consensus_lines, name_to_id, norm_name, week_only


def demo():
    event = {"id": "e1", "commence_time": "2026-10-18T17:00:00Z", "home_team": "Kansas City Chiefs", "away_team": "Las Vegas Raiders",
             "bookmakers": [{"key": "draftkings", "markets": [
                 {"key": "player_pass_yds", "last_update": "2026-10-16T12:00:00Z", "outcomes": [
                     {"name": "Over", "description": "Patrick Mahomes", "price": -115, "point": 265.5},
                     {"name": "Under", "description": "Patrick Mahomes", "price": -105, "point": 265.5},
                     {"name": "Over", "description": "No Line", "price": -110}]},
                 {"key": "player_anytime_td", "outcomes": [{"name": "Yes", "description": "Travis Kelce", "price": 120}]}]}]}
    d = parse_prop_lines(event, "2026-10-16T13:00:00Z")
    assert len(d) == 2 and set(d.side) == {"Over", "Under"}
    assert d.stat.iloc[0] == "passing_yards" and d.line.iloc[0] == 265.5 and d.price.tolist() == [-115.0, -105.0]
    assert d.commence_time.iloc[0] == "2026-10-18T17:00:00Z" and d.book_last_update.iloc[0] == "2026-10-16T12:00:00Z"
    assert parse_prop_lines({"id": "e2"}, "x").empty
    print("OK -- prop replies parse to one row per book/market/player/side; unknown markets and lineless outcomes drop.")



def demo_consensus():
    assert norm_name("Marvin Harrison Jr.") == norm_name("marvin harrison") == "marvin harrison"
    assert norm_name("Jaxon Smith-Njigba") == "jaxon smith njigba"
    players = pd.DataFrame({"player_id": ["p1", "p2", "p3"], "player_display_name": ["Puka Nacua", "Josh Allen", "Josh Allen"]})
    ids = name_to_id(players)
    assert ids == {"puka nacua": "p1"}, "a name two players share must not be guessed"
    lines = pd.DataFrame({
        "retrieved_at": ["t1", "t2", "t2", "t2", "t2"], "side": ["Over", "Over", "Over", "Over", "Under"],
        "player_name": ["Puka Nacua"] * 5, "stat": "receiving_yards", "line": [80.5, 90.5, 93.5, 94.5, 99.5]})
    c = consensus_lines(lines, ids)
    assert c.to_dict("records") == [{"player_id": "p1", "line_receiving_yards": 93.5}], c.to_dict("records")
    # Monday night at 8:15 PM ET is 00:15 UTC the next day: still Monday's game.
    pull = pd.DataFrame({"commence_time": ["2026-10-11T17:00:00Z", "2026-10-13T00:15:00Z", "2026-10-16T00:15:00Z"]})
    assert week_only(pull, {"2026-10-11", "2026-10-12"}).index.tolist() == [0, 1], "next Thursday's game is not this week"
    print("OK -- consensus is the latest pull's median Over line for the week's own games, keyed by an unambiguous name match.")



def demo_paper_trades():
    from paper_trades import UNDER_CAP, cap_unders, clv_yards, median_price, payout, settle_one, stake_for
    bets = [{"side": "Under", "game": "A @ B", "chance": c} for c in (0.56, 0.70, 0.60, 0.58, 0.65)]
    kept = cap_unders(bets + [{"side": "Over", "game": "A @ B", "chance": 0.55}], {})
    assert UNDER_CAP == 3 and sum(b["side"] == "Under" for b in kept) == 3
    assert cap_unders(bets, {}, seed=7) == cap_unders(bets, {}, seed=7)   # seeded: a re-run keeps the same picks
    assert any(b["side"] == "Over" for b in kept)                      # overs never capped
    assert len(cap_unders(bets, {"A @ B": 2})) == 1                      # earlier unders count toward the cap
    # books straddling even money: a plain median gives -2 (a 51x payout); this stays near -102
    assert -103 < median_price([-105, 101]) < -101 and median_price([150, 120, 130]) == 130
    assert median_price([-110, -110, -120]) == -110 and median_price([-200, 200]) == 125
    assert clv_yards("Over", 50.5, 53.5) == 3 and clv_yards("Under", 50.5, 53.5) == -3
    assert clv_yards("Under", 50.5, 47.5) == 3
    assert [stake_for(g) for g in (0.4, -9.9, 10.0, -19.9, 20.0, 32.7)] == [5, 5, 10, 10, 15, 20]
    assert round(payout(10, -110), 2) == 9.09 and payout(10, 150) == 15.0
    assert settle_one("Over", 80.5, 81) == "win" and settle_one("Under", 80.5, 81) == "loss"
    assert settle_one("Over", 80.0, 80) == "push" and settle_one("Over", 80.5, None) == "void"
    print("OK -- paper trades stake $5 + $5 per 10 yards of gap, pay American odds, and settle win/loss/push/void.")



def demo_game_lines():
    events = [{"id": "g1", "commence_time": "2026-10-18T17:00:00Z", "home_team": "Kansas City Chiefs", "away_team": "Las Vegas Raiders",
               "bookmakers": [{"key": "fanduel", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": -240}, {"name": "Las Vegas Raiders", "price": 198}]},
                   {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "price": -110, "point": -6.5},
                                                   {"name": "Las Vegas Raiders", "price": -110, "point": 6.5}]},
                   {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 45.5}, {"name": "Under", "price": -112, "point": 45.5}]},
                   {"key": "player_pass_yds", "outcomes": [{"name": "Over", "description": "X", "price": -110, "point": 250.5}]}]}]}]
    d = parse_game_lines(events, "t")
    assert len(d) == 6 and set(d.market) == {"h2h", "spreads", "totals"}, "player markets are not game lines"
    assert d[d.market == "h2h"].point.isna().all() and d[d.market == "spreads"].point.tolist() == [-6.5, 6.5]
    print("OK -- game line replies parse to one row per book/market/outcome with spread/total points.")


def demo_consistency_is_pregame():
    """site_data.consistency must not see the week it rates: the current week's games are its outcome."""
    import site_data
    from StatScraper import query_sqlite
    form = site_data.consistency(2026, 6)
    pid = query_sqlite("SELECT player_id FROM player_weekly_stats WHERE season = 2026 AND week = 6 "
                       "AND season_type = 'REG' AND receiving_yards > 40 LIMIT 1").player_id[0]
    h = query_sqlite(f"SELECT season, week, receiving_yards FROM player_weekly_stats WHERE player_id = '{pid}' "
                     "AND season_type = 'REG' AND season >= 2025 AND season * 100 + week < 202606")
    h = h.sort_values(["season", "week"]).tail(site_data.FORM_GAMES).receiving_yards.dropna()
    got = form[(pid, "receiving_yards")]
    assert got["n"] == len(h) and abs(got["cv"] - round(h.std() / max(h.mean(), 1), 2)) < 0.011, (got, len(h))
    assert abs(got["sd"] - round(h.std(), 1)) < 0.051, got
    print("OK -- consistency uses only the games before the week it rates.")


def demo_edge_tier_matches_board():
    """site_data.edge_tier must rate exactly as the board's edgeScore (site/app/src/lib/model.test.ts cases)."""
    import site_data
    t = lambda stat, side, sl, g, sd=20: site_data.edge_tier(stat, side, 50, 40, g, sl, sd)
    assert t("receiving_yards", "Under", 45, 44) == 3      # gbm, sleeper, |40-50|/20 = 0.5 > EDGE_Z
    assert t("receiving_yards", "Over", 45, 44) == 0       # the other side gets none of them
    assert t("receiving_yards", "Under", 55, 44) == 2
    assert t("receptions", "Under", 45, 44) == 3
    assert t("receptions", "Under", 55, 55) == 1           # 1 + 0.5
    assert t("rushing_yards", "Under", 45, 44) == 1        # 3 - 1.5
    assert t("rushing_yards", "Under", 55, 44) == 0        # 2 - 1.5
    assert t("receiving_yards", "Under", None, None, None) == 0
    print("OK -- edge_tier matches the board's confidence tiers.")


def demo_prop_sids():
    """parse_prop_sids keeps DraftKings outcomes that carry a sid and a line, and nothing from other books."""
    from StatScraper import parse_prop_sids
    out = lambda name, sid=None, point=5.5: {"name": name, "description": "A Player", "point": point, "price": -110, **({"sid": sid} if sid else {})}
    event = {"id": "e1", "bookmakers": [
        {"key": "draftkings", "markets": [{"key": "player_receptions", "outcomes": [out("Over", "0QA1#2_X"), out("Under", "0QA1#3_X"), out("Over", None)]},
                                          {"key": "player_anytime_td", "outcomes": [out("Yes", "0QA9#9")]}]},
        {"key": "fanduel", "markets": [{"key": "player_receptions", "outcomes": [out("Over", "FD1")]}]}]}
    d = parse_prop_sids(event, "2026-10-29T16:20:00Z")
    assert list(d.sid) == ["0QA1#2_X", "0QA1#3_X"] and set(d.stat) == {"receptions"} and list(d.line) == [5.5, 5.5], d
    print("OK -- DraftKings sids parse per outcome; other books, other markets and sid-less outcomes drop.")


def demo_our_margins():
    """From week 9 the blowout check reads our game model's margin, keyed like game_board's "Away @ Home"."""
    from paper_trades import BLOWOUT, our_margins
    m = our_margins(2026, 8)
    assert len(m) >= 13 and all(" @ " in k for k in m), m
    assert abs(m["Tennessee Titans @ Cincinnati Bengals"]) >= BLOWOUT > abs(m["Kansas City Chiefs @ Denver Broncos"])
    print("OK -- our game-model margins key by full team names; CIN-TEN is a blowout, KC-DEN is not.")


if __name__ == "__main__":
    demo_our_margins()
    demo()
    demo_consensus()
    demo_paper_trades()
    demo_game_lines()
    demo_consistency_is_pregame()
    demo_edge_tier_matches_board()
    demo_prop_sids()
