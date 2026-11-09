"""
Harness for docs/edge-experiments-protocol.md.

    .venv/bin/python experiments/edge_exp.py teammate   # A: vacated-targets feature, WR/TE, 15 weeks
    .venv/bin/python experiments/edge_exp.py line       # B: beat-the-line logistic model, weeks 5-7
    .venv/bin/python experiments/edge_exp.py stakes     # C: skip-rushing / quarter-Kelly replay
    .venv/bin/python experiments/edge_exp.py clv        # D: results by closing-line move
    .venv/bin/python experiments/edge_exp.py formgap    # K: drop unders projected far below recent form
    .venv/bin/python experiments/edge_exp.py style      # L: schedule-adjusted defense, pass rate, drive length
    .venv/bin/python experiments/edge_exp.py style 2026:9,2026:10 QB   # L shadow: adj_d on QB passing, weeks 9-10
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import GbmProjections as G
import PlayerProjections as pp
from paper_trades import payout

WEEKS = [(2025, w) for w in range(10, 19)] + [(2026, w) for w in range(2, 8)]
YEARS = list(range(2022, 2027))
GBM = dict(loss="absolute_error", max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=0)


def load():
    return pp.load_weekly_stats(years=YEARS), pp.load_schedule_context(years=YEARS)


# ---------------------------------------------------------------- A. teammate-out
def vacated_table(df: pd.DataFrame) -> pd.DataFrame:
    """Every WR/TE/RB listed Out or Doubtful, with his target share (ewm3) from games strictly before that week."""
    rec = df[df.position_group.isin(["WR", "TE", "RB"])].assign(t=lambda d: d.season * 100 + d.week).sort_values("t")
    rec["share_after"] = rec.groupby("player_id").target_share.transform(lambda s: s.fillna(0).ewm(halflife=3).mean())
    inj = pp.load_injury_reports(years=YEARS)
    inj = inj[inj.report_status.isin(["Out", "Doubtful"])].assign(t=lambda d: d.season * 100 + d.week).sort_values("t")
    inj = pd.merge_asof(inj, rec[["player_id", "t", "share_after"]].rename(columns={"t": "t_game"}).sort_values("t_game"),
                        left_on="t", right_on="t_game", by="player_id", allow_exact_matches=False)
    return inj.dropna(subset=["share_after"])[["season", "week", "team", "player_id", "share_after"]]


def add_vacated(rows: pd.DataFrame, vac: pd.DataFrame, report_seasons: set) -> pd.DataFrame:
    tot = vac.groupby(["season", "week", "team"]).share_after.sum().rename("vac_total").reset_index()
    rows = rows.merge(tot, on=["season", "week", "team"], how="left")
    own = vac.rename(columns={"share_after": "own_listed"})[["season", "week", "player_id", "own_listed"]]
    rows = rows.merge(own, on=["season", "week", "player_id"], how="left")
    rows["vacated_targets"] = rows.vac_total.fillna(0) - rows.own_listed.fillna(0)   # never his own share
    rows.loc[~rows.season.isin(report_seasons), "vacated_targets"] = np.nan   # no reports that season: unknown, not 0
    return rows


def teammate():
    df, sched = load()
    vac = vacated_table(df)
    seasons = set(vac.season.unique())
    print("injury-report seasons:", sorted(seasons))
    out = []
    for pos in ("WR", "TE"):
        for season, week in WEEKS:
            cut = season * 100 + week
            hist = df[(df.season * 100 + df.week < cut) & (df.position_group == pos)]
            wk = df[(df.season == season) & (df.week == week) & (df.position_group == pos)]
            last = hist.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
            wk = wk[wk.player_id.isin(last.index)]
            targets = pd.DataFrame({"season": season, "week": week, "position_group": pos, "player_id": wk.player_id.values,
                                    "opponent_team": wk.opponent_team.values, "team": wk.team.values,
                                    "player_name": wk.player_name.values})
            rows = add_vacated(G.build_features(hist, sched, targets), vac, seasons)
            train = rows[~rows.is_target & (rows.games_before >= 1)]
            test = rows[rows.is_target].set_index("player_id")
            o = pd.DataFrame({"pos": pos, "season": season, "week": week, "vacated": test.vacated_targets}, index=test.index)
            act = wk.set_index("player_id")
            for stat in ("receiving_yards", "receptions"):
                y = train[stat].fillna(0.0)
                o[f"act_{stat}"] = act[stat].astype(float)
                for name, feats in (("base", G.FEATURES), ("vac", G.FEATURES + ["vacated_targets"])):
                    o[f"{name}_{stat}"] = np.clip(HistGradientBoostingRegressor(**GBM).fit(train[feats], y).predict(test[feats]), 0, None)
            out.append(o.reset_index())
        print(pos, "done", flush=True)
    r = pd.concat(out).dropna(subset=["act_receiving_yards"])
    r.to_csv("experiments/edge_teammate_out.csv", index=False)
    for pos, g in r.groupby("pos"):
        m = lambda c, s, sub: (sub[f"{c}_{s}"] - sub[f"act_{s}"]).abs().mean()
        hi = g[g.vacated > 0.05]
        res = {k: (m("base", "receiving_yards", sub), m("vac", "receiving_yards", sub))
               for k, sub in (("all", g), ("2025", g[g.season == 2025]), ("2026", g[g.season == 2026]), ("vacated>0.05", hi))}
        rec = (m("base", "receptions", g), m("vac", "receptions", g))
        ok = res["all"][1] < res["all"][0] and res["2025"][1] <= res["2025"][0] and res["2026"][1] <= res["2026"][0] and rec[1] <= rec[0]
        print(f"\n{pos} ({len(g)} player-games, {len(hi)} with vacated > 0.05): receiving-yards MAE base -> with feature")
        for k, (b, v) in res.items():
            print(f"  {k:13s} {b:6.2f} -> {v:6.2f}")
        print(f"  receptions    {rec[0]:6.2f} -> {rec[1]:6.2f}   gate: {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- B. beat-the-line
def all_lines(df):
    w = df.assign(t=df.season * 100 + df.week)
    rows = []
    for f in glob.glob("projections/2026_w0[5-7]_*.csv"):
        wk = int(f.split("_w0")[1][0]); p = pd.read_csv(f)
        act = w[(w.season == 2026) & (w.week == wk)].drop_duplicates("player_id").set_index("player_id")
        for stat in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
            if f"line_{stat}" not in p or f"pred_{stat}" not in p:
                continue
            for r in p[p[f"line_{stat}"].notna()].itertuples():
                line, pred = getattr(r, f"line_{stat}"), getattr(r, f"pred_{stat}")
                if pd.isna(pred) or pred == line or r.player_id not in act.index:
                    continue
                a = act.at[r.player_id, stat]
                if pd.isna(a) or a == line:
                    continue
                side = 1 if pred > line else -1
                h = w[(w.player_id == r.player_id) & (w.t < 202600 + wk)].sort_values("t").tail(8)[stat].dropna()
                sd = h.std() if len(h) >= 4 else np.nan
                g, s = getattr(r, f"gbm_{stat}", np.nan), getattr(r, f"sleeper_{stat}", np.nan)
                z = abs(pred - line) / max(sd, 1) if sd == sd else np.nan
                gbm_ok = float(pd.notna(g) and (g - line) * side > 0); slp_ok = float(pd.notna(s) and (s - line) * side > 0)
                pts = gbm_ok + slp_ok + float(z == z and z > 0.17) + 0.5 * (stat == "receptions") - 1.5 * (stat == "rushing_yards")
                rows.append(dict(week=wk, stat=stat, win=float((a - line) * side > 0), z=z, gap=abs(pred - line) / max(line, 1),
                                 gbm=gbm_ok, slp=slp_ok, logline=np.log1p(line), high=pts >= 3))
    return pd.DataFrame(rows)


def line():
    df, _ = load()
    d = all_lines(df)
    d["z"] = d.z.fillna(d.z.median())
    X = pd.get_dummies(d[["z", "gap", "gbm", "slp", "logline", "stat"]], columns=["stat"], dtype=float)
    ll = lambda y, p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = []
    for te in (6, 7):
        tr, ts = d.week < te, d.week == te
        p = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], d.win[tr]).predict_proba(X[ts])[:, 1]
        res.append(pd.DataFrame({"win": d.win[ts], "p": p, "base": d.win[tr].mean(), "high": d.high[ts]}))
        print(f"train before week {te} (n={tr.sum()}) -> test week {te} (n={ts.sum()})")
    r = pd.concat(res)
    top = r[r.p >= r.p.quantile(2 / 3)]
    print(f"\nlog loss: model {ll(r.win, r.p):.4f} vs constant {ll(r.win, r.base):.4f}")
    print(f"top third by model: n={len(top)} win {100*top.win.mean():.1f}% | edge-score High: n={int(r.high.sum())} "
          f"win {100*r[r.high].win.mean():.1f}% | all {100*r.win.mean():.1f}%")
    print("gate:", "PASS" if ll(r.win, r.p) < ll(r.win, r.base) and top.win.mean() > r[r.high].win.mean() else "FAIL")


# ---------------------------------------------------------------- C/D. settled bets
def settled_bets(df):
    L = pd.concat([pd.read_csv(f"projections/paper_trades_{r}.csv") for r in ("hit55f", "rec55")])
    L = L[(L.season == 2026) & L.week.isin([6, 7])]
    act = df[df.season == 2026].drop_duplicates(["player_id", "week"]).set_index(["player_id", "week"])
    out = []
    for r in L.itertuples(index=False):
        if isinstance(r.result, str) and r.result in ("win", "loss"):
            won = r.result == "win"
        else:
            if (r.player_id, r.week) not in act.index:
                continue   # didn't play: void
            a = act.at[(r.player_id, r.week), r.stat]
            if pd.isna(a) or a == r.line:
                continue
            won = bool((a > r.line) == (r.side == "Over"))
        out.append({**r._asdict(), "won": won})
    return pd.DataFrame(out)


def show(name, g, stake):
    prof = np.where(g.won, [payout(s, p) for s, p in zip(stake, g.price)], -stake)
    print(f"  {name:18s} bets {len(g):3d}  {int(g.won.sum())}-{int((~g.won).sum())}  staked ${stake.sum():8.2f}  "
          f"profit ${prof.sum():+8.2f}  ROI {100*prof.sum()/max(stake.sum(), 1e-9):+5.1f}%")


def stakes():
    df, _ = load()
    b = settled_bets(df)
    dec = b.price.map(lambda p: 1 + (100 / -p if p < 0 else p / 100))
    kelly = ((b.chance * (dec - 1) - (1 - b.chance)) / (dec - 1)).clip(lower=0) * 0.25
    b["k_stake"] = (kelly * 1000).round(2)   # quarter-Kelly on a $1,000 notional bankroll
    for wk, g in [("weeks 6-7", b), ("week 6", b[b.week == 6]), ("week 7", b[b.week == 7])]:
        print(wk)
        show("flat (as placed)", g, g.stake)
        nr = g[g.stat != "rushing_yards"]; show("skip rushing", nr, nr.stake)
        show("quarter-Kelly", g, g.k_stake)


def clv():
    df, _ = load()
    b = settled_bets(df).dropna(subset=["close_line"])
    b["clv"] = np.where(b.side == "Over", 1, -1) * (b.close_line - b.line)   # + = we beat the close
    b["grp"] = pd.cut(b.clv, [-99, -0.51, 0.5, 99], labels=["close moved past us", "flat", "we beat the close"])
    b["profit"] = np.where(b.won, [payout(s, p) for s, p in zip(b.stake, b.price)], -b.stake)
    print(b.groupby("grp", observed=True).agg(bets=("won", "size"), win=("won", "mean"), profit=("profit", "sum"), staked=("stake", "sum"))
          .assign(win=lambda t: (100 * t.win).round(1), ROI=lambda t: (100 * t.profit / t.staked).round(1)).to_string())
    print("weeks with closing lines:", sorted(b.week.unique()))


# ---------------------------------------------------------------- shared walk-forward rows (F, G)
def week_rows(df, sched, pos, season, week):
    """GbmProjections feature rows for one walk-forward week: history strictly before it, targets = who played
    (QB: listed starters)."""
    cut = season * 100 + week
    hist = df[(df.season * 100 + df.week < cut) & (df.position_group == pos)]
    wk = df[(df.season == season) & (df.week == week) & (df.position_group == pos)]
    if pos == "QB":
        wk = pp.qb_starter_rows(wk, sched)
    last = hist.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
    wk = wk[wk.player_id.isin(last.index)]
    targets = pd.DataFrame({"season": season, "week": week, "position_group": pos, "player_id": wk.player_id.values,
                            "opponent_team": wk.opponent_team.values, "team": wk.team.values, "player_name": wk.player_name.values})
    return G.build_features(hist, sched, targets), wk.set_index("player_id")


def gate_report(r: pd.DataFrame, label: str, base: str, cand: str, stat: str) -> None:
    m = lambda c, sub: (sub[f"{c}_{stat}"] - sub[f"act_{stat}"]).abs().mean()
    s = {k: (m(base, sub), m(cand, sub)) for k, sub in (("all", r), ("2025", r[r.season == 2025]), ("2026", r[r.season == 2026]))}
    ok = s["all"][1] < s["all"][0] and s["2025"][1] <= s["2025"][0] and s["2026"][1] <= s["2026"][0]
    print(f"{label:28s} " + "  ".join(f"{k} {b:6.2f} -> {v:6.2f}" for k, (b, v) in s.items()) + f"   {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- F. volume x efficiency
def volume():
    df, sched = load()
    spec = [("WR", "receiving_yards", "targets"), ("TE", "receiving_yards", "targets"), ("RB", "rushing_yards", "carries")]
    for pos, stat, vol in spec:
        out = []
        for season, week in WEEKS:
            rows, act = week_rows(df, sched, pos, season, week)
            g = rows.groupby("player_id")
            sy = g[stat].shift(1).fillna(0).groupby(rows.player_id).transform(lambda s: s.rolling(16, min_periods=1).sum())
            sv = g[vol].shift(1).fillna(0).groupby(rows.player_id).transform(lambda s: s.rolling(16, min_periods=1).sum())
            tr = ~rows.is_target & (rows.games_before >= 1)
            mu = rows.loc[tr, stat].fillna(0).sum() / max(rows.loc[tr, vol].fillna(0).sum(), 1)
            eff = lambda k: (sy + k * mu) / (sv + k)
            k = min([5, 10, 20, 40, 80], key=lambda k: (rows.loc[tr, vol].fillna(0) * eff(k)[tr] - rows.loc[tr, stat].fillna(0)).abs().mean())
            rows["eff"] = eff(k)
            train, test = rows[tr], rows[rows.is_target].set_index("player_id")
            direct = HistGradientBoostingRegressor(**GBM).fit(train[G.FEATURES], train[stat].fillna(0)).predict(test[G.FEATURES])
            vpred = HistGradientBoostingRegressor(**GBM).fit(train[G.FEATURES], train[vol].fillna(0)).predict(test[G.FEATURES])
            out.append(pd.DataFrame({"season": season, f"act_{stat}": act.loc[test.index, stat].astype(float).values,
                                     f"direct_{stat}": np.clip(direct, 0, None), f"volxeff_{stat}": np.clip(vpred, 0, None) * test.eff.values}))
        gate_report(pd.concat(out).dropna(), f"F {pos} {stat}", "direct", "volxeff", stat)


# ---------------------------------------------------------------- G. weather + backup QB
def context():
    df, sched = load()
    s = sched.assign(t=sched.season * 100 + sched.week).sort_values("t")
    s["prev_qb"] = s.groupby("team").starting_qb_id.shift(1)
    s["qb_changed"] = ((s.starting_qb_id != s.prev_qb) & s.prev_qb.notna() & s.starting_qb_id.notna()).astype(float)
    s["dome"] = s.roof.isin(["dome", "closed"]).astype(float)
    ctx = s[["season", "week", "team", "temp", "wind", "dome", "qb_changed"]]
    for pos, stat in (("WR", "receiving_yards"), ("TE", "receiving_yards"), ("QB", "passing_yards")):
        extra = ["temp", "wind", "dome"] + (["qb_changed"] if pos != "QB" else [])
        out = []
        for season, week in WEEKS:
            rows, act = week_rows(df, sched, pos, season, week)
            rows = rows.merge(ctx, on=["season", "week", "team"], how="left")
            train, test = rows[~rows.is_target & (rows.games_before >= 1)], rows[rows.is_target].set_index("player_id")
            y = train[stat].fillna(0)
            o = {"season": season, f"act_{stat}": act.loc[test.index, stat].astype(float).values}
            for name, feats in (("base", G.FEATURES), ("ctx", G.FEATURES + extra)):
                o[f"{name}_{stat}"] = np.clip(HistGradientBoostingRegressor(**GBM).fit(train[feats], y).predict(test[feats]), 0, None)
            out.append(pd.DataFrame(o))
        gate_report(pd.concat(out).dropna(), f"G {pos} {stat}", "base", "ctx", stat)


# ---------------------------------------------------------------- H. ladder refit
def ladder():
    import copy, site_data as sd
    df, _ = load()
    act = df[df.season == 2026].drop_duplicates(["player_id", "week"]).set_index(["player_id", "week"])
    yards = {"QB": ["passing_yards", "rushing_yards"], "RB": ["rushing_yards", "receiving_yards"],
             "WR": ["receiving_yards"], "TE": ["receiving_yards"]}
    rows = []
    for pos, stats in yards.items():
        hist = sd.blended_2025(pos)
        for stat in stats:
            m = sd.OverModel(hist, stat, 1.0)
            for f in glob.glob(f"projections/2026_w0[5-7]_{pos}.csv"):
                wk = int(f.split("_w0")[1][0]); p = pd.read_csv(f)
                if f"line_{stat}" not in p:
                    continue
                for r in p.dropna(subset=[f"line_{stat}", f"pred_{stat}"]).itertuples():
                    if (r.player_id, wk) not in act.index:
                        continue
                    a, line, proj = act.at[(r.player_id, wk), stat], getattr(r, f"line_{stat}"), getattr(r, f"pred_{stat}")
                    if pd.isna(a) or a == line:
                        continue
                    raw = {}
                    for w in np.round(np.arange(0, 1.01, 0.1), 1):
                        c = w * proj + (1 - w) * line
                        raw[w] = float(np.mean(c + m.misses[int(np.digitize(c, m.edges))] > line))
                    rows.append({"week": wk, "over": float(a > line), **{f"raw_{w}": v for w, v in raw.items()}})
    d = pd.DataFrame(rows)
    ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)))
    pk = lambda sub, w, k: 0.5 + k * (sub[f"raw_{w}"] - 0.5)
    tr, te = d[d.week < 7], d[d.week == 7]
    grid = [(w, k) for w in np.round(np.arange(0, 1.01, 0.1), 1) for k in np.round(np.arange(0.6, 1.001, 0.05), 2)]
    best = min(grid, key=lambda wk_: ll(tr.over, pk(tr, *wk_)))
    def bets(sub, w, k):
        p = pk(sub, w, k); ch = np.maximum(p, 1 - p); b = ch >= 0.55
        won = np.where(p >= 0.5, sub.over == 1, sub.over == 0)
        return int(b.sum()), float(won[b].mean()) if b.any() else float("nan")
    print(f"{len(d)} logged yardage lines (weeks 5-6 train {len(tr)}, week 7 test {len(te)})")
    for name, (w, k) in (("current", (0.3, 0.85)), ("refit on wk5-6", best)):
        n, wr = bets(te, w, k)
        print(f"  {name:15s} W={w} K={k}: week-7 log loss {ll(te.over, pk(te, w, k)):.4f}, train log loss {ll(tr.over, pk(tr, w, k)):.4f}, "
              f"week-7 bets at 55%+: {n} won {100*wr:.1f}%")
    cur_n, cur_w = bets(te, 0.3, 0.85); new_n, new_w = bets(te, *best)
    print("gate:", "PASS (propose to user)" if ll(te.over, pk(te, *best)) < ll(te.over, pk(te, 0.3, 0.85)) and new_w > cur_w else "FAIL")


# ---------------------------------------------------------------- I/J. combinations
def edge_level(b: pd.Series, proj_files: dict, w: pd.DataFrame) -> int:
    """The board's edgeScore tier (site/app/src/lib/model.ts) for one placed bet, on the bet's own side."""
    p = proj_files.get((b.week, b.position))
    s = 1 if b.side == "Over" else -1
    on = lambda v: float(pd.notna(v) and s * (v - b.line) > 0)   # float: numpy bools add as OR
    row = p[p.player_id == b.player_id] if p is not None else None
    g = row[f"gbm_{b.stat}"].iloc[0] if row is not None and len(row) and f"gbm_{b.stat}" in row else np.nan
    sl = row[f"sleeper_{b.stat}"].iloc[0] if row is not None and len(row) and f"sleeper_{b.stat}" in row else np.nan
    h = w[(w.player_id == b.player_id) & (w.t < 202600 + b.week)].sort_values("t").tail(8)[b.stat].dropna()
    sd = h.std() if len(h) >= 2 else np.nan
    pts = on(g) + on(sl) + float(sd == sd and sd > 0 and on(b.projection) and abs(b.projection - b.line) / max(sd, 1) > 0.17)
    pts += 0.5 * (b.stat == "receptions") - 1.5 * (b.stat == "rushing_yards")
    return 0 if pts < 1 else 1 if pts < 2 else 2 if pts < 3 else 3


def combo():
    import model_alt as MA
    df, sched = load()
    s = sched.assign(t=sched.season * 100 + sched.week).sort_values("t")
    s["prev_qb"] = s.groupby("team").starting_qb_id.shift(1)
    s["qb_changed"] = ((s.starting_qb_id != s.prev_qb) & s.prev_qb.notna() & s.starting_qb_id.notna()).astype(float)
    s["dome"] = s.roof.isin(["dome", "closed"]).astype(float)
    ctx = s[["season", "week", "team", "temp", "wind", "dome", "qb_changed"]]
    vac = vacated_table(df); seasons = set(vac.season.unique())
    print("I. model combo")
    for pos, stat in (("QB", "passing_yards"), ("WR", "receiving_yards"), ("TE", "receiving_yards")):
        extra = ["shrink", "temp", "wind", "dome"] + ([] if pos == "QB" else ["vacated_targets", "qb_changed"])
        out = []
        for season, week in WEEKS:
            rows, act = week_rows(df, sched, pos, season, week)
            rows = add_vacated(rows.merge(ctx, on=["season", "week", "team"], how="left"), vac, seasons)
            tr = ~rows.is_target & (rows.games_before >= 1)
            lg = pd.Series(rows.loc[tr, stat].fillna(0).mean(), index=rows.index)   # position mean from training rows
            k = min([1, 2, 4, 8, 16, 32], key=lambda k: (MA.shrink_feature(rows, stat, lg, k)[tr] - rows.loc[tr, stat].fillna(0)).abs().mean())
            rows["shrink"] = MA.shrink_feature(rows, stat, lg, k)
            train, test = rows[tr], rows[rows.is_target].set_index("player_id")
            y = train[stat].fillna(0)
            o = {"season": season, f"act_{stat}": act.loc[test.index, stat].astype(float).values}
            for name, feats in (("base", G.FEATURES), ("combo", G.FEATURES + extra)):
                o[f"{name}_{stat}"] = np.clip(HistGradientBoostingRegressor(**GBM).fit(train[feats], y).predict(test[feats]), 0, None)
            out.append(pd.DataFrame(o))
        r = pd.concat(out).dropna()
        m = lambda c, sub: (sub[f"{c}_{stat}"] - sub[f"act_{stat}"]).abs().mean()
        st = {k: (m("base", sub), m("combo", sub)) for k, sub in (("all", r), ("2025", r[r.season == 2025]), ("2026", r[r.season == 2026]))}
        ok = st["all"][1] <= st["all"][0] - 0.5 and st["2025"][1] <= st["2025"][0] and st["2026"][1] <= st["2026"][0]
        print(f"  {pos} {stat:16s} " + "  ".join(f"{k} {b:6.2f} -> {v:6.2f}" for k, (b, v) in st.items()) + f"   {'PASS' if ok else 'FAIL'}")

    print("\nJ. bet combos, settled main-rule bets weeks 6-7 (in-sample, report only)")
    b = settled_bets(df)
    w = df.assign(t=df.season * 100 + df.week)
    pf = {(wk, pos): pd.read_csv(f"projections/2026_w0{wk}_{pos}.csv") for wk in (6, 7) for pos in ("QB", "RB", "WR", "TE")}
    b["tier"] = [edge_level(r, pf, w) for r in b.itertuples(index=False)]
    dec = b.price.map(lambda p: 1 + (100 / -p if p < 0 else p / 100))
    b["k_stake"] = (((b.chance * (dec - 1) - (1 - b.chance)) / (dec - 1)).clip(lower=0) * 0.25 * 1000).round(2)
    nr = b[b.stat != "rushing_yards"]
    show("all bets", b, b.stake)
    for lo, name in ((0, "skip rushing"), (1, "+ tier Low or up"), (2, "+ tier Medium or up"), (3, "+ tier High only")):
        g = nr[nr.tier >= lo]; show(name, g, g.stake)
    show("skip rushing + 1/4 Kelly", nr, nr.k_stake)


# ---------------------------------------------------------------- K. form gap
def formgap():
    df, _ = load()
    b = settled_bets(df)
    w = df.assign(t=df.season * 100 + df.week)
    pf = {(wk, pos): pd.read_csv(f"projections/2026_w0{wk}_{pos}.csv") for wk in (6, 7) for pos in ("QB", "RB", "WR", "TE")}
    l5 = {(wk, r.player_id, c[3:]): getattr(r, c) for (wk, _), p in pf.items() for r in p.itertuples(index=False) for c in p.columns if c.startswith("l5_")}
    b["l5"] = [l5.get((r.week, r.player_id, r.stat)) for r in b.itertuples(index=False)]
    b = b.dropna(subset=["l5"])
    b["gap"] = b.projection / b.l5.where(b.l5 > 0) - 1
    b["tier"] = [edge_level(r, pf, w) for r in b.itertuples(index=False)]
    roi = lambda g: (np.where(g.won, [payout(s, p) for s, p in zip(g.stake, g.price)], -g.stake).sum() / max(g.stake.sum(), 1e-9))
    drop = lambda g, t: (g.side == "Under") & (g.gap < t)
    print(f"{len(b)} settled bets with an l5 average (weeks 6-7)")
    for wk, g in [("weeks 6-7", b), ("week 6", b[b.week == 6]), ("week 7", b[b.week == 7])]:
        print(wk); show("all bets", g, g.stake)
        k, d = g[~drop(g, -0.15)], g[drop(g, -0.15)]
        show("keep (primary)", k, k.stake); show("dropped unders", d, d.stake)
    ok = all(roi(g[~drop(g, -0.15)]) > roi(g) for g in (b[b.week == 6], b[b.week == 7])) and roi(b[drop(b, -0.15)]) < 0
    print("gate:", "PASS" if ok else "FAIL")
    print("\nreport only, weeks 6-7")
    for t in (-0.10, -0.20, -0.25):
        k = b[~drop(b, t)]; show(f"drop unders < {int(100*t)}%", k, k.stake)
    k = b[~((b.side == "Over") & (b.gap > 0.15))]; show("mirror: drop overs > +15%", k, k.stake)
    base = b[(b.stat != "rushing_yards") & (b.tier >= 1)]; show("skip rush + Avoid", base, base.stake)
    k = base[~drop(base, -0.15)]; show("  + drop unders < -15%", k, k.stake)
    print("\ndropped unders by stat (weeks 6-7)")
    d = b[drop(b, -0.15)]
    for st, g in d.groupby("stat"): show(st, g, g.stake)


# ---------------------------------------------------------------- L. schedule-adjusted defense + offense style
def trail(f: pd.DataFrame, key: str, cols: list) -> pd.DataFrame:
    """Trailing 8-game mean per `key`, shifted one game: a row never sees its own game."""
    f = f.sort_values(["season", "week"]).copy()
    f[cols] = f.groupby(key)[cols].transform(lambda s: s.rolling(8, min_periods=1).mean().shift(1))
    return f


def style_features(df: pd.DataFrame):
    import sqlite3
    c = sqlite3.connect("data/nfl_games.db")
    p = pd.read_sql('SELECT season, week, posteam, defteam, drive, "pass", rush, qb_kneel, qb_spike, special_teams_play '
                    "FROM plays WHERE posteam IS NOT NULL AND season >= 2021", c)
    p = p[((p["pass"] == 1) | (p.rush == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1) & (p.special_teams_play != 1)]
    g = p.groupby(["season", "week", "posteam", "defteam"]).agg(passes=("pass", "sum"), plays=("pass", "size"), drives=("drive", "nunique")).reset_index()
    g["rate"], g["ppd"] = g.passes / g.plays, g.plays / g.drives
    off = trail(g.rename(columns={"posteam": "team", "rate": "off_pass_rate", "ppd": "off_plays_per_drive", "drives": "off_drives"}),
                "team", ["off_pass_rate", "off_plays_per_drive", "off_drives"])[["season", "week", "team", "off_pass_rate", "off_plays_per_drive", "off_drives"]]
    de = trail(g.rename(columns={"defteam": "opponent_team", "rate": "def_pass_rate_faced", "ppd": "def_plays_per_drive"}),
               "opponent_team", ["def_pass_rate_faced", "def_plays_per_drive"])[["season", "week", "opponent_team", "def_pass_rate_faced", "def_plays_per_drive"]]
    # schedule-adjusted defense: allowed / what that offense usually scores to the position, before that game
    a = df.groupby(["season", "week", "team", "opponent_team", "position_group"], as_index=False).fantasy_points_ppr.sum()
    a["expected"] = a.fantasy_points_ppr
    a = a.sort_values(["season", "week"])
    a["expected"] = a.groupby(["team", "position_group"]).expected.transform(lambda s: s.rolling(8, min_periods=3).mean().shift(1))
    a["ratio"] = a.fantasy_points_ppr / a.expected.where(a.expected > 1)
    a["adj_d"] = a.groupby(["opponent_team", "position_group"]).ratio.transform(lambda s: s.rolling(8, min_periods=1).mean().shift(1))
    return off, de, a[["season", "week", "opponent_team", "position_group", "adj_d"]]


def style(weeks=None, only=None):
    df, sched = load()
    off, de, adj = style_features(df)
    variants = {"adj_d": ["adj_d"], "off_style": ["off_pass_rate", "off_plays_per_drive", "off_drives"],
                "def_style": ["def_pass_rate_faced", "def_plays_per_drive"]}
    variants["all"] = sum(variants.values(), [])
    spec = [("QB", "passing_yards"), ("WR", "receiving_yards"), ("TE", "receiving_yards"),
            ("WR", "receptions"), ("TE", "receptions"), ("RB", "rushing_yards")]
    for pos, stat in [x for x in spec if only is None or x[0] == only]:
        out = []
        for season, week in weeks or WEEKS:
            rows, act = week_rows(df, sched, pos, season, week)
            rows = (rows.merge(off, on=["season", "week", "team"], how="left")
                        .merge(de, on=["season", "week", "opponent_team"], how="left")
                        .merge(adj, on=["season", "week", "opponent_team", "position_group"], how="left"))
            train, test = rows[~rows.is_target & (rows.games_before >= 1)], rows[rows.is_target].set_index("player_id")
            if season == 2026 and week == 7 and pos == "QB":
                print(f"coverage, 2026 w7 targets: off {test.off_pass_rate.notna().mean():.0%}  def {test.def_plays_per_drive.notna().mean():.0%}  adj {test.adj_d.notna().mean():.0%}")
            y = train[stat].fillna(0)
            o = {"season": season, "week": week, f"act_{stat}": act.loc[test.index, stat].astype(float).values}
            for name, feats in [("base", [])] + list(variants.items()):
                o[f"{name}_{stat}"] = np.clip(HistGradientBoostingRegressor(**GBM).fit(train[G.FEATURES + feats], y).predict(test[G.FEATURES + feats]), 0, None)
            out.append(pd.DataFrame(o))
        r = pd.concat(out).dropna()
        bar = 0.05 if stat == "receptions" else 0.5
        m = lambda c, sub: (sub[f"{c}_{stat}"] - sub[f"act_{stat}"]).abs().mean()
        print(f"\n{pos} {stat}  ({len(r)} player-games; bar {bar})")
        for name in variants:
            st = {k: (m("base", sub), m(name, sub)) for k, sub in (("all", r), ("2025", r[r.season == 2025]), ("2026 w2-7", r[r.season == 2026]))}
            ok = st["all"][1] <= st["all"][0] - bar and st["2025"][1] <= st["2025"][0] and st["2026 w2-7"][1] <= st["2026 w2-7"][0]
            print(f"  {name:10s} " + "  ".join(f"{k} {b:6.2f} -> {v:6.2f}" for k, (b, v) in st.items()) + f"   {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    if sys.argv[1] == "style" and len(sys.argv) > 2:   # shadow weeks, e.g. 2026:9,2026:10 QB
        style([tuple(int(x) for x in w.split(":")) for w in sys.argv[2].split(",")], sys.argv[3] if len(sys.argv) > 3 else None)
        sys.exit()
    {"teammate": teammate, "line": line, "stakes": stakes, "clv": clv,
     "volume": volume, "context": context, "ladder": ladder, "combo": combo, "formgap": formgap, "style": style}[sys.argv[1]]()
