"""Round 6 (docs/edge-experiments-protocol.md): CLV report (M), line-moved-against filter (N), confidence
tiers (O) and line shopping (P) on the main rule's settled bets, 2026 weeks 6-7. Gates are frozen in the doc.

Run: .venv/bin/python experiments/bet_review.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from log_props import DB_PATH, name_to_id, norm_name  # noqa: E402
from paper_trades import payout  # noqa: E402
from site_data import bet_tiers  # noqa: E402
from StatScraper import query_sqlite  # noqa: E402

WEEKS = (6, 7)


def roi(d):
    return d.profit.sum() / d.stake.sum() if len(d) else float("nan")


def line(name, d):
    w, l = (d.result == "win").sum(), (d.result == "loss").sum()
    return f"  {name:28s} {len(d):4d} bets {w:3d}-{l:<3d} ${d.profit.sum():+8.2f}  ROI {100 * roi(d):+6.1f}%"


def load():
    d = pd.concat([pd.read_csv(f"{ROOT}/projections/paper_trades_{r}.csv") for r in ("hit55f", "rec55")], ignore_index=True)
    return d[(d.season == 2026) & d.week.isin(WEEKS) & d.result.isin(["win", "loss", "push"])].reset_index(drop=True)


def clv_report(d):
    print("M  CLV (line moved toward us by the close = beat the close)")
    c = d.dropna(subset=["clv_yards"])
    for k, g in [("all", c)] + list(c.groupby("stat")):
        print(f"  {k:16s} n={len(g):3d}  beat {100 * (g.clv_yards > 0).mean():3.0f}%  flat {100 * (g.clv_yards == 0).mean():3.0f}%"
              f"  against {100 * (g.clv_yards < 0).mean():3.0f}%  mean {g.clv_yards.mean():+.2f}")
    for k, g in c.groupby(pd.cut(c.clv_yards, [-999, -0.01, 0.01, 999], labels=["against", "flat", "toward"]), observed=True):
        print(line(f"close moved {k}", g))


def adverse(d):
    print("N  drop bets whose close moved against us (>= 2 yd, >= 1 catch)")
    # paper_trades.closing() keeps only pulls before kickoff, so every close on file was knowable in time
    print(f"  pregame close on file for {100 * d.clv_yards.notna().mean():.0f}% of bets (void under 90%)")
    bad = d.clv_yards <= d.stat.map(lambda s: -1 if s == "receptions" else -2)
    ok = d.clv_yards.notna().mean() >= 0.9
    for w in WEEKS:
        a, k = d[d.week == w], d[(d.week == w) & ~bad]
        ok &= roi(k) > roi(a)
        print(line(f"week {w} all", a)); print(line(f"week {w} kept", k)); print(line(f"week {w} dropped", a[bad[a.index]]))
    ok &= roi(d[bad]) < 0
    print(line("pooled dropped", d[bad]))
    print(f"  N: {'PASS' if ok else 'FAIL'}")


def tiers(d):
    print("O  confidence tiers (0 Avoid, 1 Low, 2 Medium, 3 High)")
    os.chdir(ROOT)   # bet_tiers reads projections/ relative paths
    d = d.assign(tier=bet_tiers(d))
    for k, g in d.groupby("tier"):
        print(line(f"tier {k}", g))
    ok = roi(d[d.tier == 3]) > roi(d[d.tier == 1])
    for w in WEEKS:
        x = d[d.week == w]
        ok &= roi(x[x.tier == 3]) > roi(x[x.tier == 1])
        print(f"  week {w}: High {100 * roi(x[x.tier == 3]):+.1f}% ({(x.tier == 3).sum()}), Low {100 * roi(x[x.tier == 1]):+.1f}% ({(x.tier == 1).sum()})")
    print(f"  O: {'PASS' if ok else 'FAIL'}")


def shopping(d):
    print("P  line shopping: best price on the same side and line, last snapshot at or before the bet")
    players = query_sqlite("SELECT DISTINCT player_id, player_display_name FROM player_weekly_stats WHERE season >= 2025")
    ids = name_to_id(players)
    lines = query_sqlite("SELECT retrieved_at, bookmaker, player_name, stat, side, line, price FROM prop_lines", db_path=DB_PATH)
    lines = lines.assign(player_id=lines.player_name.map(norm_name).map(ids)).dropna(subset=["player_id", "price"])
    best, books = [], []
    for b in d.itertuples():
        x = lines[(lines.player_id == b.player_id) & (lines.stat == b.stat) & (lines.side == b.side)
                  & (lines.line == b.line) & (lines.retrieved_at <= b.placed_at)]
        x = x[x.retrieved_at == x.retrieved_at.max()]
        best.append(max([b.price, *x.price], key=lambda p: payout(1, p)))
        books.append(x.bookmaker.nunique())
    d = d.assign(best=best, books=books)
    d = d.assign(profit_best=[payout(r.stake, r.best) if r.result == "win" else (-r.stake if r.result == "loss" else 0)
                              for r in d.itertuples()])
    found = d.books > 0
    print(f"  snapshot found for {found.sum()}/{len(d)} bets, {d.books[found].mean():.1f} books on the same line on average;"
          f" best beat the ledger price on {(d.best != d.price).sum()} bets")
    print(f"  ledger price ${d.profit.sum():+8.2f}  ROI {100 * roi(d):+6.1f}%")
    print(f"  best price   ${d.profit_best.sum():+8.2f}  ROI {100 * d.profit_best.sum() / d.stake.sum():+6.1f}%")


if __name__ == "__main__":
    d = load()
    print(line("main rule, weeks 6-7", d))
    clv_report(d); adverse(d); tiers(d); shopping(d)
