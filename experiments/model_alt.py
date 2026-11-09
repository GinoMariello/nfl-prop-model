"""
Harness for docs/model-alternatives-protocol.md: QB starters, 15 walk-forward weeks, every candidate
refit per week on games strictly before it, scored against the frozen gates.

    .venv/bin/python experiments/model_alt.py
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import GbmProjections as G
import PlayerProjections as pp

WEEKS = [(2025, w) for w in range(10, 19)] + [(2026, w) for w in range(2, 8)]
STATS = ["passing_yards", "rushing_yards"]
LEAGUE = ["passing_yards", "attempts", "fantasy_points_ppr", "rushing_yards"]
QS = np.round(np.arange(0.1, 0.91, 0.1), 1)
GBM = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=0)

def gbm(loss="absolute_error", **kw):
    return HistGradientBoostingRegressor(loss=loss, **GBM, **kw)


def league_series(starts: pd.DataFrame) -> pd.DataFrame:
    """Per-week league mean per QB start, indexed by t = season*100+week (that week's own games)."""
    return starts.groupby("t")[LEAGUE].mean().sort_index()


def league_ewm(series: pd.DataFrame) -> pd.DataFrame:
    """Value at week t uses weeks strictly before t."""
    return series.ewm(halflife=3).mean().shift(1).add_prefix("lg_")


def league_lstm(series: pd.DataFrame, window: int = 6, seed: int = 0) -> pd.DataFrame:
    """One-step forecasts of the league series from a small LSTM trained on `series` (all strictly before the
    cutoff). Row i = forecast of series week window+i from the `window` weeks before it; the final row forecasts
    the week after the last one in `series` (the target week)."""
    torch.manual_seed(seed)
    v = series.to_numpy(np.float32); mu, sd = v.mean(0), v.std(0) + 1e-6; z = (v - mu) / sd
    X = np.stack([z[i - window:i] for i in range(window, len(z))]); Y = z[window:]
    net = torch.nn.LSTM(z.shape[1], 16, batch_first=True); head = torch.nn.Linear(16, z.shape[1])
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-2)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(200):
        opt.zero_grad(); loss = ((head(net(Xt)[1][0][-1]) - Yt) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        allX = torch.from_numpy(np.stack([z[i - window:i] for i in range(window, len(z) + 1)]))
        f = head(net(allX)[1][0][-1]).numpy() * sd + mu
    return pd.DataFrame(f, columns=[f"lstm_{c}" for c in LEAGUE])


def shrink_feature(rows: pd.DataFrame, stat: str, lg: pd.Series, k: float) -> pd.Series:
    """(n*mean_last16 + k*league)/(n+k) from the player's strictly earlier games."""
    prev = rows.groupby("player_id")[stat].shift(1)
    m = prev.groupby(rows.player_id).transform(lambda s: s.rolling(16, min_periods=1).mean())
    n = prev.groupby(rows.player_id).transform(lambda s: s.rolling(16, min_periods=1).count())
    return ((n * m).fillna(0) + k * lg) / (n.fillna(0) + k)


def one_week(df, sched, starts_all, season, week):
    cut = season * 100 + week
    hist = df[(df.season * 100 + df.week < cut) & (df.position_group == "QB")]
    wk = pp.qb_starter_rows(df[(df.season == season) & (df.week == week) & (df.position_group == "QB")], sched)
    last = hist.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
    wk = wk[wk.player_id.isin(last.index)]
    targets = pd.DataFrame({"season": season, "week": week, "position_group": "QB", "player_id": wk.player_id.values,
                            "opponent_team": wk.opponent_team.values, "team": [last.at[p, "team"] for p in wk.player_id],
                            "player_name": [last.at[p, "player_name"] for p in wk.player_id]})
    rows = G.build_features(hist, sched, targets)

    series = league_series(starts_all[starts_all.t < cut])
    ew = league_ewm(pd.concat([series, pd.DataFrame(index=[cut], columns=LEAGUE, dtype=float)]))
    rows = rows.merge(ew, left_on="t", right_index=True, how="left")
    rows["rel_pass"] = rows.ewm3_passing_yards / rows.lg_passing_yards
    fc = league_lstm(series)
    fc.index = list(series.index[6:]) + [cut]
    rows = rows.merge(fc, left_on="t", right_index=True, how="left")

    tr_mask = ~rows.is_target & (rows.games_before >= 1)
    act = wk.set_index("player_id")
    out = pd.DataFrame(index=rows.loc[rows.is_target, "player_id"])
    for stat in STATS:
        lg = rows[f"lg_{stat}"]
        k = min([1, 2, 4, 8, 16, 32],   # picked on training rows only
                key=lambda k: (shrink_feature(rows, stat, lg, k)[tr_mask] - rows.loc[tr_mask, stat].fillna(0)).abs().mean())
        rows[f"shrink_{stat}"] = shrink_feature(rows, stat, lg, k)
        train = rows[tr_mask].sort_values("t"); test = rows[rows.is_target].set_index("player_id")
        y = train[stat].fillna(0.0)
        out[f"act_{stat}"] = act[stat].astype(float)
        sets = {"gbm_base": G.FEATURES,
                "gbm_league": G.FEATURES + [f"lg_{c}" for c in LEAGUE] + ["rel_pass"],
                "gbm_league_lstm": G.FEATURES + [f"lstm_{c}" for c in LEAGUE],
                "gbm_shrink": G.FEATURES + [f"shrink_{stat}"]}
        for name, feats in sets.items():
            out[f"{name}_{stat}"] = np.clip(gbm().fit(train[feats], y).predict(test[feats]), 0, None)
        out[f"shrink_{stat}"] = test[f"shrink_{stat}"]
        if stat == "passing_yards":   # distribution candidate vs point + held-out residual quantiles
            qp = np.sort(np.column_stack([gbm("quantile", quantile=q).fit(train[G.FEATURES], y).predict(test[G.FEATURES]) for q in QS]), axis=1)
            c85 = int(len(train) * 0.85)
            g85 = gbm().fit(train[G.FEATURES].iloc[:c85], y.iloc[:c85])
            res = y.iloc[c85:].to_numpy() - g85.predict(train[G.FEATURES].iloc[c85:])
            qb = out["gbm_base_passing_yards"].to_numpy()[:, None] + np.quantile(res, QS)[None, :]
            for i, q in enumerate(QS):
                out[f"qgbm_{q}"], out[f"qbase_{q}"] = qp[:, i], qb[:, i]
    out["season"], out["week"] = season, week
    print(f"{season} w{week}: {len(out)} starters", flush=True)
    return out.reset_index()


def pinball(qcols, a):
    return np.mean([np.maximum(q * (a - qcols[i]), (q - 1) * (a - qcols[i])).mean() for i, q in enumerate(QS)])


def main():
    years = list(range(2022, 2027))
    df, sched = pp.load_weekly_stats(years=years), pp.load_schedule_context(years=years)
    starts_all = pp.qb_starter_rows(df[df.position_group == "QB"], sched).assign(t=lambda d: d.season * 100 + d.week)
    res = pd.concat([one_week(df, sched, starts_all, s, w) for s, w in WEEKS]).dropna(subset=["act_passing_yards"])
    res.to_csv("experiments/model_alt_out.csv", index=False)
    s25, s26 = (res.season == 2025).to_numpy(), (res.season == 2026).to_numpy()
    stored = []
    for s, w in WEEKS:   # sanity: the refit baseline vs the stored production GBM
        f = f"projections/gbm/2025_w{w:02d}_QB.csv" if s == 2025 else f"projections/2026_w{w:02d}_QB.csv"
        col = "pred_passing_yards" if s == 2025 else "gbm_passing_yards"
        stored.append(pd.read_csv(f)[["player_id", col]].rename(columns={col: "stored"}).assign(season=s, week=w))
    st = res.merge(pd.concat(stored), on=["player_id", "season", "week"])
    print(f"\n{len(res)} QB starts | refit gbm_base vs stored GBM on {len(st)} shared: "
          f"{(st.gbm_base_passing_yards - st.act_passing_yards).abs().mean():.2f} vs {(st.stored - st.act_passing_yards).abs().mean():.2f}")
    mae = lambda c, s_, m=slice(None): (res.loc[m, c] - res.loc[m, f"act_{s_}"]).abs().mean()
    b = [mae("gbm_base_passing_yards", "passing_yards", m) for m in (slice(None), s25, s26)]
    rb = mae("gbm_base_rushing_yards", "rushing_yards")
    print(f"\n{'candidate':16s} {'pass':>7s} {'2025':>7s} {'2026':>7s} {'rush':>7s}  gate")
    for name in ["gbm_base", "gbm_league", "gbm_league_lstm", "shrink", "gbm_shrink"]:
        p = [mae(f"{name}_passing_yards", "passing_yards", m) for m in (slice(None), s25, s26)]
        r = mae(f"{name}_rushing_yards", "rushing_yards")
        ok = p[0] <= b[0] - 0.5 and p[1] <= b[1] and p[2] <= b[2] and r <= rb + 0.3
        print(f"{name:16s} {p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f} {r:7.2f}  {'' if name == 'gbm_base' else ('PASS' if ok else 'FAIL')}")
    a = res.act_passing_yards.to_numpy()
    print("\ndistribution (passing yards): mean pinball over 9 quantiles, 10-90% coverage")
    out = {}
    for pre in ("qbase", "qgbm"):
        pin = [pinball([res.loc[m, f"{pre}_{q}"].to_numpy() for q in QS], res.loc[m, "act_passing_yards"].to_numpy())
               for m in (slice(None), s25, s26)]
        cov = ((a >= res[f"{pre}_0.1"]) & (a <= res[f"{pre}_0.9"])).mean()
        out[pre] = (pin, cov)
        print(f"  {pre:6s} pinball {pin[0]:6.2f} (2025 {pin[1]:6.2f}, 2026 {pin[2]:6.2f})  coverage {100*cov:4.1f}%")
    (pq, cq), (pb, _) = out["qgbm"], out["qbase"]
    print("  gbm_quantile", "PASS" if all(x < y for x, y in zip(pq, pb)) and 0.72 <= cq <= 0.88 else "FAIL")


if __name__ == "__main__":
    if len(sys.argv) > 1:   # e.g. 2026:8,2026:9 -- the shadow weeks after the frozen run
        WEEKS = [tuple(int(x) for x in w.split(":")) for w in sys.argv[1].split(",")]
    main()
