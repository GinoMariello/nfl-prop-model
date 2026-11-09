"""Exploratory: ways to pick the best few of the main rule's bets, backtested on 2026 weeks 6, 7 and 8 (week 8 =
Thursday only, settled here from player_weekly_stats; the ledgers are not touched). Many methods on three weeks
means some will look good by luck: a method worth using goes to a frozen forward test (weeks 9-10) first.

Run: .venv/bin/python experiments/pick_select.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from paper_trades import payout, settle_one  # noqa: E402
from StatScraper import query_sqlite  # noqa: E402

WEEKS = (6, 7, 8)


def load():
    d = pd.concat([pd.read_csv(f"projections/paper_trades_{r}.csv") for r in ("hit55f", "rec55")], ignore_index=True)
    d = d[(d.season == 2026) & d.week.isin(WEEKS)].copy()
    st = query_sqlite("SELECT * FROM player_weekly_stats WHERE season = 2026 AND season_type = 'REG'").set_index(["player_id", "week"])
    todo = d.result.isna()
    d.loc[todo, "actual"] = pd.Series([float(st.at[(p, w), s]) if (p, w) in st.index else float("nan")
                                       for p, w, s in zip(d.player_id[todo], d.week[todo], d.stat[todo])], index=d.index[todo])
    d.loc[todo, "result"] = [settle_one(s, l, a) for s, l, a in zip(d.side[todo], d.line[todo], d.actual[todo])]
    d.loc[todo, "profit"] = [payout(st_, p) if r == "win" else (-st_ if r == "loss" else 0)
                             for r, p, st_ in zip(d.result[todo], d.price[todo], d.stake[todo])]
    # week 8: only Thursday has been played; its other bets have no stats row yet and come out void
    d = d[d.result.isin(["win", "loss", "push"])]
    src = []
    for b in d.itertuples():
        f = f"projections/2026_w{b.week:02d}_{b.position}.csv"
        p = pd.read_csv(f).drop_duplicates("player_id").set_index("player_id") if os.path.exists(f) else None
        row = p.loc[b.player_id] if p is not None and b.player_id in p.index else None
        g = lambda k: row.get(f"{k}_{b.stat}") if row is not None else None
        src.append({"own": g("own"), "gbm": g("gbm"), "sleeper": g("sleeper"), "l5": g("l5")})
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(src).astype(float)], axis=1)
    s = d.side.map({"Over": 1, "Under": -1})
    on = lambda c: (s * (d[c] - d.line) > 0) & d[c].notna()
    d["agree"] = on("own").astype(int) + on("gbm").astype(int) + on("sleeper").astype(int)
    d["form_ok"] = on("l5")
    dec = d.price.map(lambda p: 1 + payout(1, p))
    d["cushion"] = d.chance - 1 / dec
    d["ev"] = d.chance * dec - 1
    d["matchup_ok"] = ((d.side == "Under") & (d.opp_allow < 1)) | ((d.side == "Over") & (d.opp_allow > 1))
    d["score"] = d.agree + d.form_ok.astype(int) + d.matchup_ok.astype(int) + (d.stat == "receptions").astype(int)
    return d


def summary(x):
    w, l = (x.result == "win").sum(), (x.result == "loss").sum()
    return len(x), w, l, x.profit.sum(), x.profit.sum() / x.stake.sum() if len(x) else float("nan")


def main():
    d = load()
    print(f"settled bets: {len(d)} ({', '.join(f'w{w} {n}' for w, n in d.groupby('week').size().items())})\n")
    top = lambda key, n, filt=None: (lambda x: (x if filt is None else x[filt(x)]).sort_values(key, ascending=False).groupby("week").head(n))
    methods = {
        "all bets (the rule)": lambda x: x,
        "top 8 by chance": top("chance", 8),
        "top 8 by cushion": top("cushion", 8),
        "top 8 by EV": top("ev", 8),
        "top 15 by chance": top("chance", 15),
        "top 15 by cushion": top("cushion", 15),
        "receptions only": lambda x: x[x.stat == "receptions"],
        "receptions, top 8 by chance": top("chance", 8, lambda x: x.stat == "receptions"),
        "no rushing": lambda x: x[x.stat != "rushing_yards"],
        "all 3 sources agree": lambda x: x[x.agree == 3],
        "2+ sources agree": lambda x: x[x.agree >= 2],
        "recent form (last 5) agrees": lambda x: x[x.form_ok],
        "matchup agrees (opp_allow)": lambda x: x[x.matchup_ok],
        "3 sources + form agree": lambda x: x[(x.agree == 3) & x.form_ok],
        "score >= 4 (sources+form+matchup+rec)": lambda x: x[x.score >= 4],
        "score, top 8": top("score", 8),
        "line not moved 2+/1+ against (wk 6-7)": lambda x: x[(x.week < 8) & (x.clv_yards > x.stat.map(lambda t: -1 if t == "receptions" else -2))],
    }
    rows = []
    for name, f in methods.items():
        x = f(d)
        n, w, l, p, r = summary(x)
        per = {f"w{wk}": summary(x[x.week == wk])[4] for wk in WEEKS}
        rows.append({"method": name, "bets": n, "record": f"{w}-{l}", "profit": round(p, 2), "ROI%": round(100 * r, 1),
                     **{k: round(100 * v, 1) for k, v in per.items()}, "weeks up": sum(v > 0 for v in per.values() if v == v)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
