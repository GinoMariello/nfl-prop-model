"""Round 7 (docs/edge-experiments-protocol.md): replay the main rule (hit55f + rec55) for 2026 weeks 6-7 with the
7+ point blowout filter off, then judge Q (market-spread blowouts) and R (blowouts by our own game model).

Each placement run is rebuilt from the prop_lines / game_lines snapshot taken just before its placed_at, with
the same ladder, chance and break-even rules as paper_trades.place (copied, minus the blowout skip).

Run: .venv/bin/python experiments/blowout_replay.py
"""
import os
import sys

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from log_props import DB_PATH, name_to_id, norm_name  # noqa: E402
from paper_trades import (BLOWOUT, HIT_MIN, HIT_STAKE, NO_UNDER, RECEPTIONS, YARDS, median_price,  # noqa: E402
                          payout, settle_one)
from site_data import REC_K, REC_W, OverModel, blended_2025  # noqa: E402
from StatScraper import query_sqlite  # noqa: E402

SEASON, WEEKS, SIGMA = 2026, (6, 7), 13.5


def snapshot(table, t):
    d = query_sqlite(f"SELECT * FROM {table}", db_path=DB_PATH)
    return d[d.retrieved_at == d.retrieved_at[d.retrieved_at <= t].max()]


def spreads_at(t):
    g = snapshot("game_lines", t)
    g = g[(g.market == "spreads") & (g.outcome == g.home_team)]
    return g.groupby(["away_team", "home_team"]).point.median().to_dict()


def our_margins(week):
    """|home margin| from our team_only game model, keyed by (away full name, home full name)."""
    m = pd.read_csv("projections/game_model/2026_team_only.csv")
    m = m[m.game_id.str.startswith(f"{SEASON}_{week:02d}_")]
    teams = nfl.import_team_desc()
    full = dict(zip(teams.team_abbr, teams.team_name))
    out = {}
    for r in m.itertuples():
        _, _, a, h = r.game_id.split("_")
        out[(full.get(a, a), full.get(h, h))] = SIGMA * norm.ppf(np.clip(r.p_home, 0.01, 0.99))
    return out


def replay_run(week, rule, t, done):
    lines = snapshot("prop_lines", t)
    lines = lines[pd.to_datetime(lines.commence_time, utc=True) > pd.Timestamp(t)]
    players = query_sqlite("SELECT DISTINCT player_id, player_display_name FROM player_weekly_stats "
                           "WHERE position_group IN ('QB', 'RB', 'WR', 'TE')", db_path=DB_PATH)
    lines = lines.assign(player_id=lines.player_name.map(norm_name).map(name_to_id(players))).dropna(subset=["player_id"])
    if rule == "rec55":
        lines = lines[lines.stat == "receptions"]
        cons = lines[lines.side == "Over"].groupby("player_id").line.median()
        lines = lines[lines.line == lines.player_id.map(cons)]
    prices = lines.groupby(["player_id", "stat", "side"]).price.agg(median_price)
    games = lines.groupby("player_id").agg(home=("home_team", "first"), away=("away_team", "first"))
    consensus = lines[lines.side == "Over"].groupby(["player_id", "stat"]).line.median()
    markets = RECEPTIONS if rule == "rec55" else YARDS
    weights = {"w": REC_W, "k": REC_K} if rule == "rec55" else {}
    bets = []
    for pos, stats in markets.items():
        proj = pd.read_csv(f"projections/{SEASON}_w{week:02d}_{pos}.csv")
        if rule == "rec55":
            proj["line_receptions"] = [consensus.get((p, "receptions")) for p in proj.player_id]
        for stat in stats:
            lad = OverModel(blended_2025(pos), stat)
            for _, r in proj.dropna(subset=[f"line_{stat}", f"pred_{stat}"]).iterrows():
                if (r.player_id, stat) in done or r.player_id not in games.index:
                    continue
                if rule != "rec55" and r[f"line_{stat}"] <= 1.5:
                    continue
                po = lad.ladder(r[f"pred_{stat}"], r[f"line_{stat}"], **weights)["p_over"]
                side, ch = ("Over", po) if po >= 0.5 else ("Under", 1 - po)
                if ch < HIT_MIN or (side == "Under" and r.player_id in NO_UNDER) or (r.player_id, stat, side) not in prices.index:
                    continue
                price = prices.loc[(r.player_id, stat, side)]
                if ch * payout(1.0, price) - (1 - ch) <= 0:
                    continue
                done.add((r.player_id, stat))
                bets.append({"week": week, "rule": rule, "player_id": r.player_id, "player_name": r.player_name, "stat": stat,
                             "side": side, "line": r[f"line_{stat}"], "price": price, "chance": ch,
                             "away": games.loc[r.player_id].away, "home": games.loc[r.player_id].home, "t": t})
    return bets


def main():
    stats = query_sqlite(f"SELECT player_id, week, receptions, receiving_yards, rushing_yards, passing_yards FROM player_weekly_stats "
                         f"WHERE season = {SEASON} AND season_type = 'REG'", db_path=DB_PATH)
    stats = stats.set_index(["player_id", "week"])
    rows = []
    for rule in ("hit55f", "rec55"):
        led = pd.read_csv(f"projections/paper_trades_{rule}.csv")
        for week in WEEKS:
            done = set()
            for t in sorted(led[(led.season == SEASON) & (led.week == week)].placed_at.unique()):
                rows += replay_run(week, rule, t, done)
    d = pd.DataFrame(rows)
    d["actual"] = [stats.at[(p, w), s] if (p, w) in stats.index else None for p, w, s in zip(d.player_id, d.week, d.stat)]
    d["result"] = [settle_one(s, l, a) for s, l, a in zip(d.side, d.line, d.actual)]
    d = d[d.result != "void"].copy()
    d["stake"] = HIT_STAKE
    d["profit"] = [payout(HIT_STAKE, p) if r == "win" else (-HIT_STAKE if r == "loss" else 0) for r, p in zip(d.result, d.price)]
    sp = {t: spreads_at(t) for t in d.t.unique()}
    d["spread"] = [sp[t].get((a, h)) for t, a, h in zip(d.t, d.away, d.home)]
    marg = {w: our_margins(w) for w in WEEKS}
    d["ours"] = [marg[w].get((a, h)) for w, a, h in zip(d.week, d.away, d.home)]
    d["mkt_blow"] = d.spread.abs() >= BLOWOUT
    d["our_blow"] = d.ours.abs() >= BLOWOUT

    roi = lambda x: x.profit.sum() / x.stake.sum() if len(x) else float("nan")
    show = lambda lab, x: print(f"  {lab:34s} {len(x):4d} bets {(x.result == 'win').sum():3d}-{(x.result == 'loss').sum():<3d} "
                                f"${x.profit.sum():+8.2f}  ROI {100 * roi(x):+6.1f}%")

    # sanity: non-blowout replay vs the real ledger
    real = pd.concat([pd.read_csv(f"projections/paper_trades_{r}.csv") for r in ("hit55f", "rec55")])
    real = real[(real.season == SEASON) & real.week.isin(WEEKS)]
    real = set(zip(real.player_id, real.stat, real.week))
    keep = d[~d.mkt_blow]
    mine = set(zip(keep.player_id, keep.stat, keep.week))
    match, back = np.mean([k in real for k in mine]), np.mean([k in mine for k in real])
    print(f"sanity: {100 * match:.0f}% of replayed non-blowout bets are in the ledger, {100 * back:.0f}% of ledger bets replayed "
          f"({'OK' if min(match, back) >= 0.9 else 'VOID: under 90%'})")
    print(f"games flagged: market {d[d.mkt_blow].groupby(['week', 'home']).ngroups}, ours {d[d.our_blow].groupby(['week', 'home']).ngroups}, "
          f"both {d[d.mkt_blow & d.our_blow].groupby(['week', 'home']).ngroups}; spread missing on {d.spread.isna().sum()} bets, ours on {d.ours.isna().sum()}")

    print("Q  market blowouts (|spread| >= 7)")
    q_ok = True
    for w in WEEKS:
        x = d[d.week == w]
        show(f"week {w} blowout-game bets", x[x.mkt_blow]); show(f"week {w} other bets", x[~x.mkt_blow])
        q_ok &= x[x.mkt_blow].profit.sum() > 0
    show("pooled blowout-game bets", d[d.mkt_blow]); show("pooled other bets", d[~d.mkt_blow])
    q_ok &= roi(d[d.mkt_blow]) >= roi(d[~d.mkt_blow])
    print(f"  Q: {'PASS (lift the filter)' if q_ok else 'FAIL (keep the filter)'}")

    print("R  our game model's margin (|margin| >= 7) vs the market filter")
    r_ok = True
    for w in WEEKS:
        x = d[d.week == w]
        show(f"week {w} kept by our filter", x[~x.our_blow]); show(f"week {w} kept by market filter", x[~x.mkt_blow])
        r_ok &= roi(x[~x.our_blow]) > roi(x[~x.mkt_blow])
    show("only ours flags (market keeps)", d[d.our_blow & ~d.mkt_blow]); show("only market flags (ours keeps)", d[d.mkt_blow & ~d.our_blow])
    print(f"  R: {'PASS (use our margin)' if r_ok else 'FAIL (keep the market spread)'}")
    print(d[d.mkt_blow | d.our_blow].groupby(["week", "away", "home"]).agg(spread=("spread", "first"), ours=("ours", "first"),
          bets=("profit", "size"), profit=("profit", "sum")).round(1).to_string())


if __name__ == "__main__":
    main()
