"""
Harness for docs/lstm-fix-protocol.md: walk-forward QB projections per variant, then the frozen pass/fail.

    .venv/bin/python experiments/lstm_fix.py run current          # one variant, all 15 weeks (resumes)
    .venv/bin/python experiments/lstm_fix.py run yards_only --seeds 0,1,2,3,4
    .venv/bin/python experiments/lstm_fix.py score                # every variant run so far vs gbm/last5

Every week retrains from scratch on games before it (walk_forward_ensemble_predict, the production recipe:
starters only, lookback 3, 30 epochs). Predictions land in experiments/lstm_fix_out/<variant>_s<seeds>/.
"""
import argparse, glob, os, sys
import pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import PlayerProjections as pp

WEEKS = [(2025, w) for w in range(10, 19)] + [(2026, w) for w in range(2, 8)]
VARIANTS = {
    "current": {},
    "yards_only": {"loss_stats": ["passing_yards"]},
    "regularized": {"weight_decay": 1e-3, "grad_clip": 1.0},
    "median": {"quantile": 0.5},
}
OUT = os.path.join(ROOT, "experiments", "lstm_fix_out")


def run(variant: str, seeds: tuple) -> None:
    torch.set_num_threads(2)
    out_dir = os.path.join(OUT, f"{variant}_s{''.join(map(str, seeds))}")
    os.makedirs(out_dir, exist_ok=True)
    years = list(range(2022, 2027))
    df, sched = pp.load_weekly_stats(years=years), pp.load_schedule_context(years=years)
    dts, bio, age, inj = (pp.load_defensive_target_stats(years=years), pp.load_player_bio(),
                          pp.load_player_age(years=years), pp.load_injury_reports(years=years))
    for season, week in WEEKS:
        path = os.path.join(out_dir, f"{season}_w{week:02d}.csv")
        if os.path.exists(path):
            continue
        out = pp.walk_forward_ensemble_predict(df, dts, bio, sched, age, inj, "QB", season, week, seeds=seeds,
                                               starters_only=True, epochs=30, **VARIANTS[variant])
        out.to_csv(path, index=False)
        print(f"{variant} {season} w{week}: {len(out)} QBs", flush=True)


def baselines() -> pd.DataFrame:
    """gbm and last-5-starts projections plus actuals, one row per (season, week, player_id)."""
    years = list(range(2022, 2027))
    df, sched = pp.load_weekly_stats(years=years), pp.load_schedule_context(years=years)
    starts = pp.qb_starter_rows(df, sched).sort_values(["season", "week"])
    rows = []
    for season, week in WEEKS:
        if season == 2025:
            g = pd.read_csv(f"projections/gbm/2025_w{week:02d}_QB.csv").rename(columns=lambda c: c.replace("pred_", "gbm_"))
        else:
            g = pd.read_csv(f"projections/2026_w{week:02d}_QB.csv")
        before = starts[(starts.season < season) | ((starts.season == season) & (starts.week < week))]
        l5 = before.groupby("player_id").tail(5).groupby("player_id")[["passing_yards", "rushing_yards"]].mean().add_prefix("last5_")
        act = df[(df.season == season) & (df.week == week)].set_index("player_id")[["passing_yards", "rushing_yards"]].add_prefix("act_")
        b = g.set_index("player_id")[["gbm_passing_yards", "gbm_rushing_yards"]].join(l5, how="inner").join(act, how="inner")
        rows.append(b.assign(season=season, week=week).reset_index())
    return pd.concat(rows)


def score() -> None:
    base = baselines()
    runs = {os.path.basename(d): d for d in sorted(glob.glob(os.path.join(OUT, "*_s*")))}
    preds = {}
    for name, d in runs.items():
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        if len(files) < len(WEEKS):
            print(f"{name}: only {len(files)}/{len(WEEKS)} weeks, skipped")
            continue
        preds[name] = pd.concat(pd.read_csv(f).assign(season=int(os.path.basename(f)[:4]), week=int(f[-6:-4]))
                                for f in files).set_index(["season", "week", "player_id"])[["pred_passing_yards", "pred_rushing_yards"]]
    if not preds:
        return
    t = base.set_index(["season", "week", "player_id"])
    for name, p in preds.items():   # only players every model projects
        t = t.join(p.add_prefix(f"{name}:"), how="inner")
    mae = lambda col, stat, m=slice(None): (t.loc[m, col] - t.loc[m, f"act_{stat}"]).abs().mean()
    s25, s26 = t.index.get_level_values("season") == 2025, t.index.get_level_values("season") == 2026
    print(f"\nQB starters scored: {len(t)} games over {len(WEEKS)} weeks\n")
    print(f"{'model':22s} {'pass MAE':>9s} {'2025':>7s} {'2026':>7s} {'rush MAE':>9s}")
    for name, col in [("gbm", "gbm"), ("last5", "last5")] + [(n, f"{n}:pred") for n in preds]:
        print(f"{name:22s} {mae(f'{col}_passing_yards', 'passing_yards'):9.2f} {mae(f'{col}_passing_yards', 'passing_yards', s25):7.2f} "
              f"{mae(f'{col}_passing_yards', 'passing_yards', s26):7.2f} {mae(f'{col}_rushing_yards', 'rushing_yards'):9.2f}")
    cur = next((n for n in preds if n.startswith("current_")), None)
    if cur is None:
        print("\nno `current` run yet: gates need it")
        return
    c = lambda stat, m=slice(None): mae(f"{cur}:pred_{stat}", stat, m)
    print(f"\ngates vs {cur} (docs/lstm-fix-protocol.md):")
    for n in preds:
        if n == cur:
            continue
        v = lambda stat, m=slice(None): mae(f"{n}:pred_{stat}", stat, m)
        g = {"1 beats gbm": v("passing_yards") < mae("gbm_passing_yards", "passing_yards"),
             "2 >=1.0 yd better than current": v("passing_yards") <= c("passing_yards") - 1.0,
             "3 no worse either season": v("passing_yards", s25) <= c("passing_yards", s25) and v("passing_yards", s26) <= c("passing_yards", s26),
             "4 rush guardrail +0.3": v("rushing_yards") <= c("rushing_yards") + 0.3}
        print(f"  {n:20s} {'PASS' if all(g.values()) else 'FAIL'}  " + "  ".join(f"{k}: {'y' if ok else 'n'}" for k, ok in g.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("variant", nargs="?", choices=list(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    run(a.variant, tuple(int(s) for s in a.seeds.split(","))) if a.cmd == "run" else score()
