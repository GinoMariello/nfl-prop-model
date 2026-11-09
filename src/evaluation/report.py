"""
Reproducible development report (docs/next-implementations.md Stage 5;
protocol: docs/experiment-protocol.md "Uncertainty rules").

Reads a Stage 4 run's saved `predictions.csv` and reports it. Nothing here
fits, refits, or touches a database -- that is what makes the report
reproducible bit-for-bit from the artifact, and it is why a report can be
regenerated years later from a directory alone.

Three separate aggregations, never blended (model-plan.md section 7):

  * per-season scores,
  * pooled game-weighted scores (every game counts once),
  * the equal-season mean (every season counts once).

A season with 272 games and one with 240 pull the pooled number differently
than they pull the equal-season mean; reporting only one hides which.

Uncertainty is a **paired** season-block bootstrap: 2,000 fixed-seed
replicates, 95% percentile intervals, whole seasons resampled with
replacement, and the same sampled blocks reused for every compared model.
Games are not independent draws -- shared rosters, schedule and era effects
correlate them within a season -- so the block is the season, not the game.

Run: python3 -m src.evaluation.report --run-dir models_game/home_win/<run_id>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import (
    FULL_COVERAGE_MODELS, MATCHED_MODELS, comparison_seasons,
)
from src.evaluation.metrics import (
    N_CALIBRATION_BINS, align_by_game_id, calibration_bins, log_loss_terms,
)
from src.models.train import RANDOM_SEED

N_REPLICATES = 2000                 # experiment-protocol.md, frozen
INTERVAL = 95                       # percentile interval, both tails
BASELINE_MODEL = "home_frequency"   # bootstrap_paired_differences' default baseline

LIMITATIONS = """\
- Retrospective, corrected source data. Upstream values may have been revised
  after the fact, so this is a historical snapshot study, not a verified
  as-of replay of what was knowable before each kickoff.
- Completion is inferred from finite non-negative integer scores, a proxy --
  the ingestion schema carries no explicit status column.
- Market inputs are devigged CLOSING lines. Every market comparison is
  therefore a closing-line study; live inference could only use a quote
  actually received before the cutoff.
- No untouched holdout exists. 2025 was inspected during development, and
  2026 is partial, so every number here is a development number and none of
  it is a final evaluation.
- Intervals are approximate. They do not remove selection bias, distribution
  shift, or the cost of repeated inspection.
- Nothing here supports a betting-profitability claim. A classification
  improvement is not an edge after vigorish, limits, and execution."""


def load_run(run_dir) -> dict:
    """A Stage 4 run directory: its predictions, metrics and manifest."""
    run_dir = Path(run_dir)
    missing = [name for name in ("predictions.csv", "metrics.csv", "manifest.json")
               if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is not a Stage 4 run directory (missing {missing}); "
            f"run `python3 -m src.evaluation.backtest --snapshot <dir>` first")
    return {
        "run_dir": run_dir,
        "predictions": pd.read_csv(run_dir / "predictions.csv"),
        "metrics": pd.read_csv(run_dir / "metrics.csv"),
        "manifest": json.loads((run_dir / "manifest.json").read_text()),
    }


def declared_models(manifest: dict) -> dict:
    """The models a run declared per population, from its own manifest.

    A report describes the run it was given, not the code reading it: a run
    saved under an older protocol declares fewer models than the code does now,
    and the code's lists would demand models that run never fitted. Falls back
    to the code's lists only for a manifest that predates those keys.
    """
    return {"full": tuple(manifest.get("full_coverage_models", FULL_COVERAGE_MODELS)),
            "matched": tuple(manifest.get("matched_models", MATCHED_MODELS))}


def _require_declared_present(predictions: pd.DataFrame, declared: dict) -> None:
    """A declared model with no predictions is a broken artifact. Left alone it
    empties the comparison window and surfaces deep in the bootstrap as "need at
    least one array to concatenate", naming nothing."""
    present = set(predictions["model"])
    for population, models in declared.items():
        missing = [m for m in models if m not in present]
        if missing:
            raise ValueError(
                f"the run manifest declares {missing} for {population} coverage, but "
                f"predictions.csv has no rows for them -- the artifact is incomplete, "
                f"or the manifest and predictions come from different runs")


def _comparisons(declared: dict) -> list:
    """(population, baseline, models), one paired bootstrap each, all on the
    population's shared season blocks -- restricted to what this run declared.
    A run from before a model existed simply has no comparison involving it."""
    full, matched = declared["full"], declared["matched"]
    wanted = [
        ("full", BASELINE_MODEL, full),
        ("matched", BASELINE_MODEL, matched),
        # Does any fitted model beat the raw devigged quote it could simply have used?
        # (home_frequency is left out: that row would only mirror the one above.)
        ("matched", "market", tuple(m for m in matched if m != BASELINE_MODEL)),
        # The control model-plan.md section 6 requires before claiming team statistics
        # add anything beyond fitting the market: same rows, same market column, same
        # preprocessing -- the only difference is the team features.
        ("matched", "market_only", ("team_market",)),
        # Protocol v5's primary question: do the starting-QB and rest features add
        # anything to the residual model, i.e. beyond the closing line?
        ("matched", "team_residual", ("team_residual_plus",)),
    ]
    out = []
    for population, baseline, models in wanted:
        kept = tuple(m for m in models if m in declared[population] and m != baseline)
        if baseline in declared[population] and kept:
            out.append((population, baseline, kept))
    return out


def _population(predictions: pd.DataFrame, population: str, declared=None) -> tuple:
    declared = declared or {"full": FULL_COVERAGE_MODELS, "matched": MATCHED_MODELS}
    if population == "matched":
        return predictions.loc[predictions["paired"]], declared["matched"]
    if population == "full":
        return predictions, declared["full"]
    raise ValueError(f"unknown population {population!r}; expected 'full' or 'matched'")


def per_game_losses(predictions: pd.DataFrame, population: str, declared=None) -> pd.DataFrame:
    """Per-game log loss for every declared model, on identical game IDs.

    One row per game, one column per model, plus `season`. Built by aligning
    each model to the same label series through `align_by_game_id`, so a model
    that does not cover the population raises here rather than contributing a
    shorter column that would silently mis-pair under resampling.

    Restricted to the population's comparison seasons. `team_market` has no
    predictions before 2008 -- its training folds hold no paired quotes -- so
    without this the matched game set would include 464 games three of the
    four models scored and the fourth did not, and pairing them would be
    exactly the mismatched comparison model-plan.md section 6 forbids. The
    restriction is derived from coverage, never a hardcoded season range.
    """
    subset, models = _population(predictions, population, declared)
    window = comparison_seasons(predictions, population, models)
    subset = subset.loc[subset["season"].isin(window)]
    scored = subset.loc[subset["p_home"].notna()]
    games = scored.drop_duplicates("game_id").set_index("game_id").sort_index()
    labels = pd.Series(games["label"].to_numpy(dtype=float), index=games.index)

    table = pd.DataFrame({"season": games["season"]}, index=games.index)
    for model in models:
        rows = scored.loc[scored["model"] == model].set_index("game_id").sort_index()
        y, p = align_by_game_id(labels, pd.Series(rows["p_home"].to_numpy(dtype=float),
                                                  index=rows.index))
        table[model] = log_loss_terms(y, p)
    return table


def _draw_blocks(seasons, n_replicates: int, seed: int) -> list:
    """Season draws, generated once and reused for every compared model.

    The protocol requires the *same* sampled blocks across models: an
    independent draw per model would make the paired difference's interval
    reflect two unrelated resamplings rather than the shared one, and would
    widen it for no reason. Returned as a list so the caller cannot
    accidentally advance the generator between models.
    """
    rng = np.random.RandomState(seed)
    seasons = np.asarray(sorted(seasons))
    return [rng.choice(seasons, size=len(seasons), replace=True) for _ in range(n_replicates)]


def bootstrap_paired_differences(losses: pd.DataFrame, models, baseline=BASELINE_MODEL,
                                 n_replicates: int = N_REPLICATES,
                                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Paired model-minus-baseline log-loss differences, season-block bootstrap.

    Paired means the difference is taken per game and then averaged, never two
    independently resampled means subtracted -- the two models score the exact
    same games, and pairing is what removes the game-to-game variance they
    share.

    A negative difference favours the model; the interval excludes 0 when the
    sign is stable across replicates. That is not a significance test and does
    not survive repeated inspection (experiment-protocol.md).
    """
    if baseline not in losses:
        raise KeyError(f"baseline {baseline!r} is not in the loss table: {list(losses.columns)}")
    if not n_replicates > 0:
        raise ValueError("n_replicates must be positive")

    by_season = {season: rows.index for season, rows in losses.groupby("season")}
    blocks = _draw_blocks(by_season, n_replicates, seed)
    low, high = (100 - INTERVAL) / 2, 100 - (100 - INTERVAL) / 2

    rows = []
    for model in models:
        if model == baseline:
            continue
        difference = (losses[model] - losses[baseline])
        replicates = np.array([difference.loc[np.concatenate([by_season[s] for s in draw])].mean()
                               for draw in blocks])
        rows.append({
            "model": model, "baseline": baseline, "n_games": int(len(difference)),
            "n_seasons": int(len(by_season)), "n_replicates": n_replicates,
            "mean_difference": float(difference.mean()),
            "ci_low": float(np.percentile(replicates, low)),
            "ci_high": float(np.percentile(replicates, high)),
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["favours_model"] = frame["ci_high"] < 0
    return frame


def equal_season_mean(metrics: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Every season counts once, against the pooled score's every game once.

    Reported separately, never instead: a 272-game season and a 240-game
    season move the two numbers differently, and showing one hides which.

    Restricted to each population's comparison window, for the same reason
    the bootstrap is. Averaging `team_market` over its 17 seasons beside
    `market` over 19 compares two different season sets and reads as though
    they were the same one -- the mismatched comparison model-plan.md section
    6 forbids, arriving through the back door of an average rather than a
    pooled score.
    """
    per_season = metrics.loc[metrics["season"].ne("pooled")].copy()
    per_season["season"] = pd.to_numeric(per_season["season"], errors="coerce")
    kept = pd.concat([per_season.loc[per_season["population"].eq(population)
                                     & per_season["season"].isin(window)]
                      for population, window in windows.items()], ignore_index=True)
    return kept.groupby(["population", "model"], as_index=False).agg(
        n_seasons=("season", "count"),
        log_loss=("log_loss", "mean"), brier=("brier", "mean"),
        accuracy=("accuracy", "mean"), roc_auc=("roc_auc", "mean"))


def calibration_table(predictions: pd.DataFrame, population: str,
                      n_bins: int = N_CALIBRATION_BINS, declared=None) -> pd.DataFrame:
    """Ten equal-width bins per model, with counts. Empty bins are reported as
    count 0 with NaN rates rather than dropped -- an absent bin and an empty
    bin mean different things."""
    subset, models = _population(predictions, population, declared)
    frames = []
    for model in models:
        rows = subset.loc[subset["model"].eq(model) & subset["p_home"].notna()]
        if rows.empty:
            continue
        bins = calibration_bins(rows["label"].to_numpy(dtype=float),
                                rows["p_home"].to_numpy(dtype=float), n_bins=n_bins)
        frames.append(bins.assign(population=population, model=model))
    if not frames:
        return pd.DataFrame(columns=["population", "model", "bin_low", "bin_high", "count",
                                     "mean_predicted", "observed_frequency"])
    out = pd.concat(frames, ignore_index=True)
    return out[["population", "model", "bin_low", "bin_high", "count",
                "mean_predicted", "observed_frequency"]]


def build_report(run: dict, n_replicates: int = N_REPLICATES) -> dict:
    """Every table the written report renders."""
    predictions, metrics = run["predictions"], run["metrics"]
    declared = declared_models(run["manifest"])
    _require_declared_present(predictions, declared)
    calibrations, windows, losses = [], {}, {}
    for population in ("full", "matched"):
        windows[population] = comparison_seasons(predictions, population, declared[population])
        losses[population] = per_game_losses(predictions, population, declared)
        calibrations.append(calibration_table(predictions, population, declared=declared))
    bootstraps = [bootstrap_paired_differences(losses[population], models, baseline=baseline,
                                               n_replicates=n_replicates).assign(population=population)
                  for population, baseline, models in _comparisons(declared)]
    return {
        "per_season": metrics.loc[metrics["season"].ne("pooled")],
        "pooled": metrics.loc[metrics["season"].eq("pooled")],
        "equal_season": equal_season_mean(metrics, windows),
        "windows": windows,
        "bootstrap": pd.concat(bootstraps, ignore_index=True),
        "calibration": pd.concat(calibrations, ignore_index=True),
        "comparison_seasons": comparison_seasons(predictions, declared=declared["matched"]),
        "manifest": run["manifest"],
    }


def _window_text(window) -> str:
    """`2008-2024 (17 seasons)`, or the explicit list when there is a gap --
    a contiguous-looking range that is actually missing a season would
    otherwise read as complete."""
    if not window:
        return "none"
    contiguous = list(window) == list(range(min(window), max(window) + 1))
    span = f"{min(window)}-{max(window)}" if contiguous else ", ".join(str(s) for s in window)
    return f"{span} ({len(window)} seasons)"


def render_report(report: dict) -> str:
    """The readable report. Plain text; no plotting dependency for v1."""
    manifest = report["manifest"]
    fmt = lambda v: f"{v:.6f}"
    out = [
        "=" * 78,
        "NFL game-outcome development report",
        "=" * 78,
        f"run_id            {manifest.get('run_id')}",
        f"snapshot          {manifest.get('snapshot_dir')}",
        f"snapshot sha256   {str(manifest.get('snapshot_sha256'))[:16]}...",
        f"code revision     {manifest.get('git', {}).get('commit', 'unknown')}"
        f"{' (DIRTY)' if manifest.get('git', {}).get('dirty') else ''}",
        f"target            {manifest.get('target')}",
        f"study type        {manifest.get('study_type')}",
        "",
        "Comparison windows -- every table below is restricted to these, so no two",
        "models are ever scored on different season sets:",
        f"  full     {_window_text(report['windows']['full'])}",
        f"  matched  {_window_text(report['windows']['matched'])}",
        "",
        "-- pooled, game-weighted (every game counts once) --",
        report["pooled"][["population", "model", "n", "log_loss", "brier",
                          "accuracy", "roc_auc"]].to_string(index=False, float_format=fmt),
        "",
        "-- equal-season mean (every season counts once) --",
        report["equal_season"].to_string(index=False, float_format=fmt),
        "",
        "-- paired log-loss differences (model minus baseline), season-block bootstrap --",
        f"   {N_REPLICATES} replicates, seed {RANDOM_SEED}, {INTERVAL}% percentile interval,",
        "   whole seasons resampled with replacement, same blocks across models.",
        "   Negative favours the model. Not a significance test.",
        report["bootstrap"][["population", "baseline", "model", "n_games", "n_seasons",
                             "mean_difference", "ci_low", "ci_high", "favours_model"]]
            .to_string(index=False, float_format=fmt),
        "",
        "-- calibration, ten equal-width bins --",
        report["calibration"].to_string(index=False, float_format=fmt),
        "",
    ]
    failures = manifest.get("failures") or []
    out += [f"-- folds that could not be fitted ({len(failures)}) --"]
    out += [f"   {f['season']} {f['variant']}: {f['error']}" for f in failures] or ["   none"]
    out += ["", "-- limitations --", LIMITATIONS, "", "=" * 78]
    return "\n".join(out)


def save_report(report: dict, run_dir) -> dict:
    """Write beside the run's own artifacts, never over them."""
    run_dir = Path(run_dir)
    written = {}
    for name in ("bootstrap", "calibration", "equal_season"):
        path = run_dir / f"{name}.csv"
        if path.exists():
            raise FileExistsError(f"{path} already exists; a report is never overwritten")
        report[name].to_csv(path, index=False)
        written[name] = path
    path = run_dir / "report.txt"
    if path.exists():
        raise FileExistsError(f"{path} already exists; a report is never overwritten")
    path.write_text(render_report(report))
    written["report"] = path
    return written


def main(argv=None) -> None:
    """CLI entry point over a saved Stage 4 run directory. Reads only that
    directory -- no snapshot, no database, no refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="a Stage 4 run directory")
    parser.add_argument("--replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(load_run(args.run_dir), n_replicates=args.replicates)
    print(render_report(report))
    if not args.no_save:
        written = save_report(report, args.run_dir)
        print(f"written: {', '.join(str(p) for p in written.values())}")


if __name__ == "__main__":
    main()
