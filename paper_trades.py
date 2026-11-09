"""
Paper trading against logged prop lines -- no real money, a record to judge the
projections by.

    python3 paper_trades.py place 2026 5     # before kickoff: one bet per yardage prop
    python3 paper_trades.py settle 2026 5    # after the games: win / loss / push / void
    python3 paper_trades.py report           # running totals

CLV: settle also records each bet's closing line from log_closing.py's pulls just before
kickoff (DraftKings via ESPN, line only), and how many yards it moved in the bet's favor (clv_yards).

place: for every passing/rushing/receiving-yards prop with a consensus line in
projections/<season>_w<NN>_<POS>.csv (log_props.py writes them), bet the side the
blended projection (pred_) points to: over when it is above the line, under below.
Stake = $5 + $5 per full 10 yards of gap (gap 0-9.9 -> $5, 10-19.9 -> $10, ...).
Odds = median American price for that side across books in the latest pull. Only
games not yet started are bet. Each bet is appended to projections/paper_trades.csv
with the projection, line and odds as they were, so later re-runs cannot rewrite them.

Rules (each with its own ledger, placed and settled the same way):
  base  -- the above, projections/paper_trades.csv.
  hit55f -- hit55, minus games with a 7+ point spread (likely blowouts). Until week 7 it
           also skipped lines under 10 yards and QB rushing; the weeks 5-6 replay showed those
           filters dropping winners, so from week 7 only the Prop Board's parlays skip them.
           (Old wording: hit55, minus lines under 10 yards, QB rushing and games with a 7+ point
           spread (likely blowouts). projections/paper_trades_hit55f.csv.
           With rec55 it is the main rule. Each bet records its game and opp_allow (see WEAK_D).
  rec55 -- receptions (RB/WR/TE): 55%+ on the receptions ladder (site_data REC_W/REC_K) that
           beats the price's break-even, $10 flat, no 7+ point spreads, never the under for
           NO_UNDER. Kept apart from the yardage rules until it proves out (started 2026 week 6).
           projections/paper_trades_rec55.csv.
  hit55 -- only props whose hit-rate ladder (site_data.OverModel.ladder: centered 70% on
           the line, 30% on our projection) gives the likelier side >= 55% at the posted
           line, and only where that chance beats the break-even of the price (positive expected
           value); flat $10 on that side; lines of 1.5 or less skipped; never the under
           for the players in NO_UNDER.
           projections/paper_trades_hit55.csv, with the chance. (Started
           2026 week 5 as a 65% rule; lowered to 55% once the chances were shrunk 15%.)

    python3 paper_trades.py place 2026 6 hit55

settle: actual stat from player_weekly_stats. Over the line wins for an over bet;
exactly on the line is a push (stake back); a player with no game row is void.
"""
import os
import sys
from datetime import datetime, timezone

import nfl_data_py as nfl
import pandas as pd

from log_props import DB_PATH, name_to_id, norm_name, week_only
from StatScraper import query_sqlite

LEDGERS = {"base": "projections/paper_trades.csv", "hit55": "projections/paper_trades_hit55.csv",
           "hit55f": "projections/paper_trades_hit55f.csv", "rec55": "projections/paper_trades_rec55.csv"}
FILTER_MIN_LINE, BLOWOUT = 10.0, 7.0   # hit55f: BLOWOUT; FILTER_MIN_LINE is the board's parlay filter only (from week 7)
# Players the 55% rules never bet the under on (manual exclusion, 2026-10-14). Keyed by nflverse player_id.
NO_UNDER = {"00-0039139": "Jahmyr Gibbs", "00-0038542": "Bijan Robinson", "00-0039075": "Puka Nacua",
             "00-0038543": "Jaxon Smith-Njigba"}
LEDGER = LEDGERS["base"]
HIT_MIN, HIT_STAKE = 0.55, 10.0
# Unders-per-game cap: NOT applied (tried for week 7, dropped 2026-10-19). Replaying weeks 2-6, no
# cap made the most money every time; capping kept ~40% of bets for no reliable gain. Kept for analysis.
UNDER_CAP = 3
# Side test "oppf" (from 2026 week 7, nothing placed separately): the main rule minus unders whose
# opponent allows >= WEAK_D x the league average of that stat to that position (last 8 games) --
# the weakest third of defenses in the weeks 2-6 replay, where dropping them added ~0.2-0.8 points
# of return. Small and unproven; judged on weeks 7+. Each main bet records opp_allow.
WEAK_D = 1.08
YARDS = {"QB": ["passing_yards", "rushing_yards"], "RB": ["rushing_yards", "receiving_yards"],
         "WR": ["receiving_yards"], "TE": ["receiving_yards"]}
RECEPTIONS = {"RB": ["receptions"], "WR": ["receptions"], "TE": ["receptions"]}   # rule rec55
# From 2026 week 9 a blowout is |our game model's margin| >= BLOWOUT, not the market spread (round 7 of
# docs/edge-experiments-protocol.md: passed on weeks 6-7, thinly). Margin = MARGIN_SD * inverse-normal(p_home).
OUR_MARGIN_FROM, MARGIN_SD = 202609, 13.5


def our_margins(season: int, week: int) -> dict:
    """"Away @ Home" (full team names) -> our team_only game model's home margin, for the week's games."""
    from scipy.stats import norm
    path = f"projections/game_model/{season}_team_only.csv"
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path)
    m = m[m.game_id.str.startswith(f"{season}_{week:02d}_")]
    full = dict(zip(*nfl.import_team_desc()[["team_abbr", "team_name"]].values.T))
    out = {}
    for gid, p in zip(m.game_id, m.p_home.clip(0.01, 0.99)):
        _, _, a, h = gid.split("_")
        out[f"{full.get(a, a)} @ {full.get(h, h)}"] = MARGIN_SD * norm.ppf(p)
    return out


def stake_for(gap: float) -> float:
    """$5 plus $5 per full 10 yards of |gap|."""
    return 5.0 + 5.0 * int(abs(gap) // 10)


def payout(stake: float, price: float) -> float:
    """Profit on a winning bet at American odds."""
    return stake * (100.0 / abs(price) if price < 0 else price / 100.0)


def median_price(prices) -> float:
    """Median American price across books, taken on payouts: a plain median of American odds
    breaks when books straddle even money (median of -105 and +101 is -2, read as a 51x payout)."""
    profit = pd.Series([payout(1.0, p) for p in prices if p is not None and not pd.isna(p)], dtype=float).median()   # NaN if none (ESPN rows have no price)
    return profit * 100 if profit >= 1 else -100 / profit


def opponent_allowance(season: int, week: int):
    """f(player_id, pos, stat) -> the player's opponent's last-8-game average of `stat` allowed to
    `pos`, over the league average (1.0 = average, higher = weaker defense). Games before `week` only."""
    p = query_sqlite(f"SELECT season, week, player_id, team, opponent_team, position_group, passing_yards, rushing_yards, "
                     f"receiving_yards, receptions FROM player_weekly_stats WHERE season IN ({season - 1}, {season}) "
                     f"AND season_type = 'REG'", db_path=DB_PATH)
    p = p[p.season * 100 + p.week < season * 100 + week].assign(t=lambda d: d.season * 100 + d.week)
    team = p.sort_values("t").groupby("player_id").team.last()
    sched = nfl.import_schedules([season])
    wk = sched[(sched.game_type == "REG") & (sched.week == week)]
    opp = {**dict(zip(wk.home_team, wk.away_team)), **dict(zip(wk.away_team, wk.home_team))}
    cache = {}

    def f(pid, pos, stat):
        if (pos, stat) not in cache:
            a = p[p.position_group == pos].groupby(["opponent_team", "t"])[stat].sum().reset_index()
            m = a.sort_values("t").groupby("opponent_team").tail(8).groupby("opponent_team")[stat].mean()
            cache[(pos, stat)] = m / m.mean()
        v = cache[(pos, stat)].get(opp.get(team.get(pid)))
        return None if v is None or pd.isna(v) else round(float(v), 3)
    return f


def cap_unders(bets: list, taken: dict, seed: int = 0) -> list:
    """Drop unders past UNDER_CAP per game, keeping a random set (seeded, so a re-run keeps the same
    picks); `taken` = unders already placed per game (updated in place)."""
    import random
    out = []
    for b in random.Random(seed).sample(bets, len(bets)):
        if b["side"] == "Under" and b.get("game"):
            if taken.get(b["game"], 0) >= UNDER_CAP:
                continue
            taken[b["game"]] = taken.get(b["game"], 0) + 1
        out.append(b)
    return out


def settle_one(side: str, line: float, actual) -> str:
    if actual is None or pd.isna(actual):
        return "void"
    if actual == line:
        return "push"
    return "win" if (actual > line) == (side == "Over") else "loss"


def place(season: int, week: int, rule: str = "base") -> pd.DataFrame:
    lines = query_sqlite("SELECT * FROM prop_lines", db_path=DB_PATH)
    lines = lines[lines.retrieved_at == lines.retrieved_at.max()]
    sched = nfl.import_schedules([season])
    lines = week_only(lines, set(sched.loc[(sched.game_type == "REG") & (sched.week == week), "gameday"].astype(str)))
    now = pd.Timestamp.now(tz="UTC")
    lines = lines[pd.to_datetime(lines.commence_time, utc=True) > now]  # pregame only
    players = query_sqlite("SELECT DISTINCT player_id, player_display_name FROM player_weekly_stats "
                           "WHERE position_group IN ('QB', 'RB', 'WR', 'TE')", db_path=DB_PATH)
    lines = lines.assign(player_id=lines.player_name.map(norm_name).map(name_to_id(players))).dropna(subset=["player_id"])
    if rule == "rec55":   # books post receptions at different lines (1.5 vs 2.5), so price only at the consensus line
        lines = lines[lines.stat == "receptions"]
        cons = lines[lines.side == "Over"].groupby("player_id").line.median()
        lines = lines[lines.line == lines.player_id.map(cons)]
    prices = lines.groupby(["player_id", "stat", "side"]).agg(price=("price", median_price), kickoff=("commence_time", "first"))

    ledger = LEDGERS[rule]
    done = pd.read_csv(ledger) if os.path.exists(ledger) else pd.DataFrame()
    hit = rule.startswith("hit55") or rule == "rec55"
    markets = RECEPTIONS if rule == "rec55" else YARDS
    if hit:   # unshrunk misses; the ladder leans on the line instead
        from site_data import REC_K, REC_W, OverModel, blended_2025, game_board
        ladders = {(pos, stat): OverModel(blended_2025(pos), stat) for pos, stats in markets.items() for stat in stats}
        weights = {"w": REC_W, "k": REC_K} if rule == "rec55" else {}
        allow = opponent_allowance(season, week)
        spreads = {f"{g['away']} @ {g['home']}": g["spread"] for g in game_board()}
        if season * 100 + week >= OUR_MARGIN_FROM:   # our margin where the model has the game, else the spread
            spreads.update(our_margins(season, week))
        games = lines.groupby("player_id").agg(home=("home_team", "first"), away=("away_team", "first"))
    already = set(zip(done.season, done.week, done.player_id, done.stat)) if len(done) else set()
    placed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bets = []
    consensus = lines[lines.side == "Over"].groupby(["player_id", "stat"]).line.median()
    for pos, stats in markets.items():
        path = f"projections/{season}_w{week:02d}_{pos}.csv"
        if not os.path.exists(path):
            continue
        proj = pd.read_csv(path)
        if rule == "rec55":   # the CSVs only carry lines from the main pull; take receptions straight from it
            proj["line_receptions"] = [consensus.get((p, "receptions")) for p in proj.player_id]
        for stat in stats:
            if f"line_{stat}" not in proj:
                continue
            for _, r in proj.dropna(subset=[f"line_{stat}", f"pred_{stat}"]).iterrows():
                if (season, week, r.player_id, stat) in already:
                    continue
                gap = r[f"pred_{stat}"] - r[f"line_{stat}"]
                chance = None
                if hit:
                    if rule != "rec55" and r[f"line_{stat}"] <= 1.5:
                        continue   # kneel-downs and zero-carry games swing these; the Prop Board's parlays skip them too
                    if rule in ("hit55f", "rec55"):
                        g = games.loc[r.player_id] if r.player_id in games.index else None
                        sp = spreads.get(f"{g.away} @ {g.home}") if g is not None else None
                        if sp is not None and abs(sp) >= BLOWOUT:
                            continue
                    po = ladders[(pos, stat)].ladder(r[f"pred_{stat}"], r[f"line_{stat}"], **weights)["p_over"]
                    side, chance = ("Over", po) if po >= 0.5 else ("Under", 1 - po)
                    if chance < HIT_MIN or (side == "Under" and r.player_id in NO_UNDER):
                        continue
                elif gap == 0:
                    continue
                else:
                    side = "Over" if gap > 0 else "Under"
                if (r.player_id, stat, side) not in prices.index:
                    continue  # game started, or no price for that side
                price, kickoff = prices.loc[(r.player_id, stat, side)]
                if chance is not None and chance * payout(1.0, price) - (1 - chance) <= 0:
                    continue   # 55% loses at a steep price: the chance must beat the price's break-even
                bets.append({"placed_at": placed_at, "season": season, "week": week, "position": pos,
                             "player_id": r.player_id, "player_name": r.player_name, "stat": stat, "kickoff": kickoff,
                             "projection": round(r[f"pred_{stat}"], 2), "line": r[f"line_{stat}"], "gap": round(gap, 2),
                             "game": f"{games.loc[r.player_id].away} @ {games.loc[r.player_id].home}" if hit and r.player_id in games.index else None,
                             "side": side, "price": price,
                             **({"opp_allow": allow(r.player_id, pos, stat)} if hit else {}), "stake": HIT_STAKE if hit else stake_for(gap),
                             **({"chance": round(chance, 3)} if hit else {}),
                             "actual": None, "result": None, "profit": None})
    new = pd.DataFrame(bets)
    if len(new):
        (pd.concat([done, new], ignore_index=True) if len(done) else new).to_csv(ledger, index=False)
    return new


def clv_yards(side: str, line: float, close_line: float) -> float:
    """Yards the line moved in the bet's favor by kickoff: an Over at 50.5 that closed at 53.5 got +3."""
    return close_line - line if side == "Over" else line - close_line


def closing(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per bet, the closing line from log_closing.py's last pull before kickoff (table
    prop_lines_close: DraftKings via ESPN, keyed by player_id): close_line, close_price (None --
    ESPN has no prices) and clv_yards = how many yards the line moved in the bet's favor."""
    try:
        close = query_sqlite("SELECT * FROM prop_lines_close", db_path=DB_PATH)
    except Exception:
        return pd.DataFrame(index=ledger.index, columns=["close_line", "close_price", "clv_yards"])
    close = close.assign(t=pd.to_datetime(close.retrieved_at, utc=True), ko=pd.to_datetime(close.commence_time, utc=True))
    close = close[close.t < close.ko].dropna(subset=["player_id"])
    close = close[close.t == close.groupby(["player_id", "stat"]).t.transform("max")]   # each prop's last pregame pull
    lines = close[close.side == "Over"].groupby(["player_id", "stat"]).line.median()
    prices = close.groupby(["player_id", "stat", "side"]).price.agg(median_price)
    out = []
    for _, r in ledger.iterrows():
        cl = lines.get((r.player_id, r.stat))
        cp = prices.get((r.player_id, r.stat, r.side))
        clv = None if cl is None else clv_yards(r.side, r.line, cl)
        out.append({"close_line": cl, "close_price": cp, "clv_yards": clv})
    return pd.DataFrame(out, index=ledger.index)


def settle(season: int, week: int, rule: str = "base") -> pd.DataFrame:
    path = LEDGERS[rule]
    if not os.path.exists(path):
        return pd.DataFrame()
    ledger = pd.read_csv(path)
    ledger["result"] = ledger["result"].astype(object)   # all-blank until the first settle, so pandas reads it as numbers
    stats = query_sqlite(f"SELECT player_id, passing_yards, rushing_yards, receiving_yards, receptions FROM player_weekly_stats "
                         f"WHERE season = {int(season)} AND week = {int(week)}", db_path=DB_PATH).set_index("player_id")
    todo = (ledger.season == season) & (ledger.week == week) & ledger.result.isna()
    for i in ledger.index[todo]:
        r = ledger.loc[i]
        actual = stats.at[r.player_id, r.stat] if r.player_id in stats.index else None
        result = settle_one(r.side, r.line, actual)
        ledger.loc[i, ["actual", "result"]] = [actual, result]
        ledger.loc[i, "profit"] = payout(r.stake, r.price) if result == "win" else (-r.stake if result == "loss" else 0.0)
    wk = (ledger.season == season) & (ledger.week == week)
    if wk.any():
        cl = closing(ledger[wk])
        for c in cl.columns:
            ledger.loc[wk, c] = pd.to_numeric(cl[c], errors="coerce").values   # None (no closing price) -> NaN: pandas 3 won't put None in a float column
    ledger.to_csv(path, index=False)
    return ledger[(ledger.season == season) & (ledger.week == week)]


def report(rule: str = "base") -> None:
    ledger = pd.read_csv(LEDGERS[rule])
    settled = ledger[ledger.result.isin(["win", "loss", "push"])]
    wins, losses = (settled.result == "win").sum(), (settled.result == "loss").sum()
    staked = settled.stake.sum()
    print(f"{len(ledger)} bets, {len(settled)} settled: {wins}-{losses} ({100 * wins / max(1, wins + losses):.1f}%; "
          f"52.4% breaks even at -110), staked ${staked:.0f}, profit ${settled.profit.sum():+.2f} "
          f"(ROI {100 * settled.profit.sum() / max(1, staked):+.1f}%). Open: ${ledger[ledger.result.isna()].stake.sum():.0f}.")
    if "clv_yards" in settled and settled.clv_yards.notna().any():
        c = settled.clv_yards.dropna()
        print(f"  closing line value: {len(c)} bets with a closing line, average {c.mean():+.2f} yds, "
              f"{100 * (c > 0).mean():.0f}% beat the close, {100 * (c < 0).mean():.0f}% worse than it")
    for (pos, stat), part in settled.groupby(["position", "stat"]):
        print(f"  {pos} {stat:16s} {(part.result == 'win').sum()}-{(part.result == 'loss').sum()}  profit ${part.profit.sum():+.2f}")


if __name__ == "__main__":
    if len(sys.argv) in (2, 3) and sys.argv[1] == "report":
        report(*sys.argv[2:])
    elif len(sys.argv) in (4, 5) and sys.argv[1] in ("place", "settle"):
        season, week, rule = int(sys.argv[2]), int(sys.argv[3]), (sys.argv[4] if len(sys.argv) == 5 else "base")
        out = place(season, week, rule) if sys.argv[1] == "place" else settle(season, week, rule)
        print(f"{sys.argv[1]} ({rule}): {len(out)} bets for {season} week {week}")
    else:
        sys.exit(__doc__)
