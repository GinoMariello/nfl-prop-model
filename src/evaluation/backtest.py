"""
Chronological backtest (docs/next-implementations.md Stage 4; design:
docs/stage4-design.md).

For each outer season T: fit candidates on 1999..T-2, select C by validation
log loss on T-1, refit preprocessing and model on 1999..T-1, score T with that
refit and nothing else (docs/model-plan.md section 5).

Two invariants this module exists to hold:

  * An outer-test outcome can never reach a fitted parameter or a selected
    hyperparameter. `select_c` takes the training and validation frames, not
    the matchups plus a fold, so the test rows are not in scope at all -- the
    isolation is structural, not a convention to remember.

  * Two models are never compared across different populations
    (model-plan.md section 6). Population membership is declared up front;
    `align_by_game_id` then raises if a declared model turns out not to cover
    it, rather than quietly scoring the smaller sample.

Run: python3 -m src.evaluation.backtest --snapshot snapshots/<id>
"""
import argparse
import json
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.snapshot import (
    content_hash, open_snapshot, snapshot_db_path, verify_snapshot, _library_versions,
)
from src.evaluation.diagnose import EXCLUDED_SEASONS, _git_info, _sha256_file
from src.evaluation.metrics import (
    accuracy, align_by_game_id, brier_score, log_loss, predictions_for_scoring, roc_auc,
)
from src.features.rolling_features import (
    STAT_COLUMNS, add_rolling_features, build_matchup_table, exclusion_report,
    paired_quote_mask, _dates,
)
from src.features.team_game_stats import build_team_games
from src.models.baselines import always_home_accuracy, fit_home_frequency, predict_home_frequency
from src.models.train import (
    MAX_ITER, NO_QUOTE, RANDOM_SEED, SOLVER, TOL, fit_model, predict_model, save_model,
)

TRAIN_START = 1999                                  # model-plan.md section 5
OUTER_SEASONS = tuple(range(2005, 2025))            # development window, 2005-2024
C_GRID = (0.01, 0.1, 1.0, 10.0)                     # ascending; see select_c
# Protocol v4 amendment. team_residual is meant to fall back to the market as
# C -> 0, but under C_GRID validation chose the floor, 0.01, in all 17 folds:
# the market-equivalent end of its path was off the grid. Extended downward
# for this variant only; every other variant keeps C_GRID unchanged.
VARIANT_C_GRID = {"team_residual": (1e-5, 1e-4, 1e-3, *C_GRID),
                  "team_residual_plus": (1e-5, 1e-4, 1e-3, *C_GRID)}  # protocol v5: same as its base
VARIANTS = ("team_only", "team_market", "team_spread", "market_only", "team_residual",
            "team_residual_plus")

# Declared, never derived from observed coverage. A season in which every game
# happened to be quoted would otherwise promote `market` to full coverage,
# which is the comparison model-plan.md section 6 forbids.
#
# `team_spread` is full-coverage and `team_market` is not, though both use a
# market input. The difference is the input, not the model: a devigged
# probability needs BOTH moneylines, which are absent before 2006 and missing
# for 390 later games, while the closing spread is present for every REG game
# from 1999 on. So team_spread is the only market-informed model that can be
# scored on the same games team_only is.
FULL_COVERAGE_MODELS = ("home_frequency", "team_only", "team_spread")
# `market_only` is matched-only for the same reason team_market is: it needs both
# moneylines.
MATCHED_MODELS = ("home_frequency", "team_only", "team_market", "team_spread",
                  "market_only", "team_residual", "team_residual_plus", "market")

TARGET = "home_win"
ARTIFACT_ROOT = "models_game"
MARKET_MODEL = "market"
HOME_FREQUENCY_MODEL = "home_frequency"
PREDICTION_COLUMNS = ["game_id", "p_home", "abstention_reason"]
REQUIRED_MATCHUP_COLUMNS = ("game_id", "home_season", "home_week", "home_gameday",
                            "home_win", "market_home_probability", "spread_line",
                            "neutral_site")


def load_matchups(snapshot_dir) -> tuple:
    """`(matchups, games)` rebuilt in memory from a snapshot's raw tables.

    Deliberately does not read the snapshot's own `matchup_features`. Stage 1
    says derived tables are non-authoritative, and measured, that one is stale:
    62 columns against this builder's 64 (measured 2026-10-01), carrying `point_diff` and raw
    moneyline columns and missing `neutral_site`. Nothing would raise -- every
    allowlisted feature is present -- so reading it would fit on old rolling
    values and silently lose the neutral-site flag.
    """
    snapshot_dir = Path(snapshot_dir)
    verify_snapshot(snapshot_dir)                   # tamper check, once per run
    with closing(open_snapshot(snapshot_dir)) as conn:
        games = pd.read_sql_query("SELECT * FROM games", conn)
        stats = pd.read_sql_query("SELECT * FROM team_game_stats", conn)
    return matchups_from_tables(games, stats), games


def matchups_from_tables(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """The matchup table rebuilt from raw `games` and `team_game_stats` rows --
    load_matchups()'s builder, split out so a live (non-snapshot) caller such as
    game_picks.py scores with exactly the features the backtest fitted on."""
    skeleton = build_team_games(games)
    metrics = [s for s in STAT_COLUMNS if s not in ("points_for", "points_against")]
    stats = skeleton.merge(stats[["game_id", "team", *metrics]],
                           on=["game_id", "team"], how="left", validate="one_to_one")
    matchups = build_matchup_table(add_rolling_features(stats), games)

    missing = [c for c in REQUIRED_MATCHUP_COLUMNS if c not in matchups]
    if missing:
        raise ValueError(
            f"rebuilt matchup table is missing {missing}; the backtest rebuilds from raw "
            f"games/team_game_stats and never reads the cached matchup_features table, "
            f"which is non-authoritative (docs/next-implementations.md Stage 1)"
        )
    return matchups


def _eligible(matchups: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable label. Variant filtering stays in select_features."""
    return matchups.loc[matchups[TARGET].notna()]


def _season_slice(eligible: pd.DataFrame, low: int, high: int) -> pd.DataFrame:
    """Eligible rows in [low, high], reindexed.

    `reset_index(drop=True)` is required, not tidiness: train.py's
    `_require_unique_index` rejects a repeated index, because label alignment
    is `matchups.loc[X.index]` and a repeated label fans rows out.
    """
    window = eligible.loc[eligible["home_season"].between(low, high)]
    return window.reset_index(drop=True)


def _assert_partitions_ordered(fold: dict) -> None:
    """Disjoint AND strictly chronologically ordered, on parsed dates.

    Season numbers are not evidence: they order the partitions only because
    this snapshot happens to separate every consecutive season by at least 241
    days. next-implementations.md strengthened this from "disjoint partitions"
    to "disjoint AND strictly chronologically ordered" precisely because
    disjointness alone permits an out-of-order partition.
    """
    names = ("train", "valid", "test")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(fold[f"{a}_ids"]) & set(fold[f"{b}_ids"]))
            if shared:
                raise ValueError(
                    f"fold {fold['outer_season']}: {a} and {b} partitions share "
                    f"game(s) {shared[:3]}")
    for a, b in (("train", "valid"), ("valid", "test")):
        if not fold[f"{a}_dates"][1] < fold[f"{b}_dates"][0]:
            raise ValueError(
                f"fold {fold['outer_season']}: partitions are not strictly chronologically "
                f"ordered -- {a} ends {fold[f'{a}_dates'][1]} but {b} starts "
                f"{fold[f'{b}_dates'][0]}")


def make_folds(matchups: pd.DataFrame, outer_seasons=OUTER_SEASONS) -> list:
    """One fold per outer season, validated before it is returned."""
    missing = [c for c in REQUIRED_MATCHUP_COLUMNS if c not in matchups]
    if missing:
        raise KeyError(f"matchups missing columns required to build folds: {missing}")

    ids = pd.Index(matchups["game_id"])
    if ids.hasnans or ids.has_duplicates:
        repeated = ids[ids.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"game_id must be non-null and unique to build folds (repeated: {repeated})")

    outer_seasons = list(outer_seasons)
    if not outer_seasons:
        raise ValueError("outer_seasons must not be empty")
    excluded = sorted(set(outer_seasons) & set(EXCLUDED_SEASONS))
    if excluded:
        raise ValueError(
            f"outer seasons {excluded} are excluded from this release "
            f"(docs/experiment-protocol.md, 'Holdout designation'); the development "
            f"window is 2005-2024")

    eligible = _eligible(matchups).copy()
    eligible["_date"] = _dates(eligible["home_gameday"])
    undated = eligible.loc[eligible["_date"].isna(), "game_id"].tolist()
    if undated:
        raise ValueError(
            f"eligible game(s) have an unparseable gameday and cannot be ordered "
            f"chronologically: {undated[:3]}")

    folds = []
    for season in sorted(outer_seasons):
        parts = {"train": (TRAIN_START, season - 2),
                 "valid": (season - 1, season - 1),
                 "test": (season, season)}
        fold = {"outer_season": int(season),
                "train_seasons": (TRAIN_START, season - 2),
                "valid_season": season - 1,
                "test_season": season}
        for name, (low, high) in parts.items():
            rows = eligible.loc[eligible["home_season"].between(low, high)]
            if rows.empty:
                raise ValueError(
                    f"fold {season}: {name} partition (seasons {low}-{high}) has no "
                    f"eligible games")
            fold[f"{name}_ids"] = tuple(sorted(rows["game_id"]))
            fold[f"{name}_dates"] = (rows["_date"].min().date().isoformat(),
                                     rows["_date"].max().date().isoformat())
            fold[f"n_{name}"] = int(len(rows))
            fold[f"n_{name}_paired"] = int(paired_quote_mask(rows).sum())
        _assert_partitions_ordered(fold)
        folds.append(fold)
    return folds


def select_c(train: pd.DataFrame, valid: pd.DataFrame, variant: str, c_grid=C_GRID) -> dict:
    """Pick C by validation log loss. Takes frames, never the test rows.

    That signature is the isolation argument: there is no parameter through
    which an outer-test outcome could reach this function, so model-plan.md
    section 9 gate 4 holds by construction and the test only has to confirm it.

    `c_grid` ascending plus a strict `<` keeps the smaller C on a tie, which is
    the protocol's rule with no tie-breaking branch to get wrong.
    """
    grid = [float(c) for c in c_grid]
    if not grid or not all(np.isfinite(grid)) or any(c <= 0 for c in grid):
        raise ValueError("c_grid values must be finite and positive")

    candidates, best, chosen = [], np.inf, None
    for candidate_c in sorted(grid):
        bundle = fit_model(train, variant=variant, C=candidate_c)
        predicted = predictions_for_scoring(predict_model(bundle, valid))
        labels = valid.set_index("game_id")[TARGET].loc[predicted.index]
        y, p = align_by_game_id(labels.astype(float), predicted)
        score = log_loss(y, p)
        candidates.append({
            "variant": variant, "C": candidate_c, "valid_log_loss": score,
            "n_train": bundle["n_train"], "n_valid_eligible": int(len(valid)),
            "n_valid_scored": int(len(predicted)),
            "converged": bundle["converged"], "n_iter": bundle["n_iter"],
        })
        if np.isfinite(score) and score < best:
            best, chosen = score, candidate_c

    if chosen is None:
        scored = candidates[-1]["n_valid_scored"] if candidates else 0
        raise ValueError(
            f"cannot select C for {variant}: no candidate produced a finite validation "
            f"log loss on {scored} scorable row(s)")
    return {"variant": variant, "C": chosen, "candidates": candidates}


def _as_prediction_frame(game_ids, p_home) -> pd.DataFrame:
    """The three-column shape predict_model returns, for the baselines.

    Both baselines reach the metrics through exactly the path a fitted model
    does -- one funnel, so Stage 5 never needs a reshape per model type.
    """
    values = pd.Series(p_home, dtype=float)
    return pd.DataFrame({
        "game_id": list(game_ids),
        "p_home": values.to_numpy(),
        "abstention_reason": np.where(values.isna(), NO_QUOTE, None),
    })


def run_backtest(matchups: pd.DataFrame, outer_seasons=OUTER_SEASONS) -> dict:
    """Walk the folds. Writes nothing; `save_backtest` owns persistence."""
    folds = make_folds(matchups, outer_seasons)
    eligible = _eligible(matchups)

    predictions, candidates, selection, failures, fitted, home_freq = [], [], [], [], {}, {}

    for fold in folds:
        season = fold["outer_season"]
        refit_rows = _season_slice(eligible, TRAIN_START, season - 1)
        test = _season_slice(eligible, season, season)
        paired = paired_quote_mask(test)
        context = test[["game_id", "home_season", "home_week", "home_gameday",
                        TARGET, "neutral_site"]].rename(columns={
                            "home_season": "season", "home_week": "week",
                            "home_gameday": "gameday", TARGET: "label"})
        context = context.assign(paired=paired.to_numpy())

        def emit(model, frame, variant=None, selected_c=np.nan):
            merged = context.merge(frame[PREDICTION_COLUMNS], on="game_id", validate="one_to_one")
            predictions.append(merged.assign(model=model, variant=variant, selected_C=selected_c))

        # Baseline 1: one constant per fold, fitted on the same rows the final
        # logistic refit uses, so the two see identical information. Fitted
        # once; scoring a subset later never refits it.
        fitted_baseline = fit_home_frequency(refit_rows[TARGET])
        home_freq[season] = fitted_baseline
        emit(HOME_FREQUENCY_MODEL,
             _as_prediction_frame(test["game_id"],
                                  predict_home_frequency(fitted_baseline, test["game_id"])))

        # Baseline 2: the market's own quote. Never fitted, abstains unpaired.
        emit(MARKET_MODEL,
             _as_prediction_frame(test["game_id"],
                                  test["market_home_probability"].where(paired)))

        for variant in VARIANTS:
            # Only ValueError: that is what train.py raises for an unfittable
            # fold. A KeyError or TypeError here is a bug and must crash.
            # Separate blocks so `stage` records which call actually failed --
            # one combined block cannot tell them apart after the fact.
            def note(stage, exc):
                failures.append({
                    "season": season, "variant": variant, "stage": stage, "error": str(exc),
                    "n_train_eligible": int(len(refit_rows)),
                    "n_train_paired": int(paired_quote_mask(refit_rows).sum()),
                })

            try:
                picked = select_c(_season_slice(eligible, TRAIN_START, season - 2),
                                  _season_slice(eligible, season - 1, season - 1), variant,
                                  c_grid=VARIANT_C_GRID.get(variant, C_GRID))
            except ValueError as exc:
                note("select_c", exc)
                continue
            candidates.extend({"season": season, **row} for row in picked["candidates"])
            try:
                bundle = fit_model(refit_rows, variant=variant, C=picked["C"])
            except ValueError as exc:
                note("refit", exc)
                continue
            fitted[(season, variant)] = bundle
            selection.append({"season": season, "variant": variant, "C": picked["C"],
                              "n_train": bundle["n_train"], "converged": bundle["converged"],
                              "n_iter": bundle["n_iter"]})
            emit(variant, predict_model(bundle, test), variant=variant, selected_c=picked["C"])

    frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return {
        "predictions": frame,
        "folds": folds,
        "candidates": pd.DataFrame(candidates),
        "selection": pd.DataFrame(selection),
        "failures": pd.DataFrame(failures, columns=["season", "variant", "stage", "error",
                                                    "n_train_eligible", "n_train_paired"]),
        "home_frequency": home_freq,
        "fitted": fitted,
        "metrics": score_predictions(frame),
    }


def _score_one(rows: pd.DataFrame) -> dict:
    """Every model reaches the metrics through this one funnel."""
    predicted = predictions_for_scoring(rows[PREDICTION_COLUMNS])
    labels = pd.Series(rows["label"].to_numpy(dtype=float),
                       index=pd.Index(rows["game_id"]))
    y, p = align_by_game_id(labels, predicted)
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier_score(y, p),
            "accuracy": accuracy(y, p), "roc_auc": roc_auc(y, p)}


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-season and pooled scores, per declared population.

    Labels are NOT restricted to what each model happened to predict. The
    population is fixed before the model is consulted, so a model declared for
    a population but abstaining inside it makes `align_by_game_id` raise --
    which is the point. That is what stops a full-coverage model from being
    quietly scored on a subset (model-plan.md section 6).
    """
    if predictions.empty:
        return pd.DataFrame(columns=["season", "population", "model", "n", "log_loss",
                                     "brier", "accuracy", "roc_auc"])

    rows = []
    populations = (("full", FULL_COVERAGE_MODELS, predictions),
                   ("matched", MATCHED_MODELS, predictions.loc[predictions["paired"]]))
    for population, declared, subset in populations:
        covered = {}
        for model in declared:
            model_rows = subset.loc[subset["model"] == model]
            seasons = sorted(model_rows["season"].unique())
            covered[model] = set(seasons)
            for season in seasons:
                rows.append({"season": season, "population": population, "model": model,
                             **_score_one(model_rows.loc[model_rows["season"] == season])})

        # Always-home is accuracy only, never a probability (model-plan.md
        # section 6 / B7), so it cannot traverse the probability path above.
        # Emitted once per population from the game set, outside the model
        # loop -- gating it on a particular model would move it silently the
        # day someone reorders the declared tuple.
        for season in sorted(subset["season"].unique()):
            games = subset.loc[subset["season"] == season].drop_duplicates("game_id")
            rows.append({"season": season, "population": population, "model": "always_home",
                         "n": int(len(games)),
                         "accuracy": always_home_accuracy(games["label"])["accuracy"],
                         "log_loss": np.nan, "brier": np.nan, "roc_auc": np.nan})

        window = sorted(set.intersection(*covered.values())) if covered else []
        for model in declared:
            pooled = subset.loc[subset["model"].eq(model) & subset["season"].isin(window)]
            if len(pooled):
                rows.append({"season": "pooled", "population": population, "model": model,
                             **_score_one(pooled)})

    return pd.DataFrame(rows)[["season", "population", "model", "n", "log_loss",
                               "brier", "accuracy", "roc_auc"]]


def comparison_seasons(predictions: pd.DataFrame, population: str = "matched",
                       declared=None) -> list:
    """Seasons where every model declared for `population` produced predictions.

    `declared` defaults to this code's model lists; a report on a saved run
    passes the lists that run's manifest recorded instead.

    Derived from coverage rather than hardcoded, so it cannot drift from the
    coverage it describes. The only window in which a model-vs-model claim is
    permitted -- Stage 5 restricts its paired bootstrap to it, because a
    variant that could not be fitted for some seasons (team_market before
    2008) otherwise leaves the compared models on different game sets.
    """
    if population == "matched":
        paired, default = predictions.loc[predictions["paired"]], MATCHED_MODELS
    elif population == "full":
        paired, default = predictions, FULL_COVERAGE_MODELS
    else:
        raise ValueError(f"unknown population {population!r}; expected 'full' or 'matched'")
    declared = default if declared is None else declared
    covered = [set(paired.loc[paired["model"] == m, "season"].unique()) for m in declared]
    # int(), not the numpy scalars a pandas unique() yields -- these go into
    # manifest.json, and json.dumps cannot serialize np.int64.
    return sorted(int(season) for season in set.intersection(*covered)) if covered else []


def save_backtest(result: dict, matchups: pd.DataFrame, games: pd.DataFrame,
                  snapshot_dir, out_dir=None) -> dict:
    """Write every artifact. Refuses an existing directory (model-plan.md
    section 8: "never overwriting earlier runs")."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(out_dir) if out_dir else Path(ARTIFACT_ROOT) / TARGET / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    fold_rows = [{"outer_season": f["outer_season"], "partition": part, "game_id": gid}
                 for f in result["folds"] for part in ("train", "valid", "test")
                 for gid in f[f"{part}_ids"]]
    summary = [{k: v for k, v in f.items() if not k.endswith("_ids")} for f in result["folds"]]

    written = {}
    for name, frame in (("folds", pd.DataFrame(fold_rows)),
                        ("fold_summary", pd.DataFrame(summary)),
                        ("candidates", result["candidates"]),
                        ("selection", result["selection"]),
                        ("predictions", result["predictions"]),
                        ("metrics", result["metrics"]),
                        ("failures", result["failures"])):
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path

    # Every model file is a pickle, so src/models/predict.py refuses to load one
    # whose checksum this manifest does not vouch for.
    model_sha256 = {}
    for (season, variant), bundle in result["fitted"].items():
        path = save_model(bundle, out_dir / variant / "logistic" / f"season_{season}.joblib")
        model_sha256[path.relative_to(out_dir).as_posix()] = _sha256_file(path)

    eligible = _eligible(matchups)
    manifest = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage4_backtest",
        "target": TARGET,
        "study_type": "development backtest -- NOT a final evaluation; no untouched "
                      "holdout exists (docs/experiment-protocol.md)",
        "closing_line_caveat": "market probabilities are devigged closing lines, so every "
                               "market comparison is a closing-line study",
        "snapshot_dir": str(snapshot_dir),
        "snapshot_sha256": _sha256_file(snapshot_db_path(snapshot_dir)),
        "matchup_content_hash": content_hash(
            eligible[["game_id", TARGET, "market_home_probability", "neutral_site"]]),
        "protocol": {"train_start": TRAIN_START, "outer_seasons": list(OUTER_SEASONS),
                     "c_grid": list(C_GRID),
                     "c_grid_overrides": {k: list(v) for k, v in VARIANT_C_GRID.items()},
                     "random_seed": RANDOM_SEED, "solver": SOLVER,
                     "tol": TOL, "max_iter": MAX_ITER},
        "full_coverage_models": list(FULL_COVERAGE_MODELS),
        "matched_models": list(MATCHED_MODELS),
        "comparison_seasons": comparison_seasons(result["predictions"]),
        "neutral_site_policy": "scored in the standard population; flagged in predictions.csv "
                               "for separate reporting, never excluded (docs/stage4-design.md)",
        "home_frequency_fitted": {str(k): v for k, v in result["home_frequency"].items()},
        "model_sha256": model_sha256,
        "failures": result["failures"].to_dict("records"),
        "eligibility_counts": exclusion_report(games).to_dict("records"),
        "git": _git_info(),
        "library_versions": _library_versions(),
    }
    manifest.update({f"{name}_sha256": _sha256_file(path) for name, path in written.items()})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                                      default=str))
    return {"out_dir": out_dir, "manifest": manifest, **written}


def print_report(result: dict) -> None:
    """Pooled scores per population, plus any fold that could not be fitted.
    Display only -- computes nothing the result does not already hold."""
    metrics = result["metrics"]
    pooled = metrics.loc[metrics["season"].eq("pooled")]
    print(f"\nComparison seasons (matched): {comparison_seasons(result['predictions'])}")
    for population in ("full", "matched"):
        block = pooled.loc[pooled["population"] == population]
        if block.empty:
            continue
        print(f"\n-- pooled, {population} coverage --")
        print(block[["model", "n", "log_loss", "brier", "accuracy", "roc_auc"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    failures = result["failures"]
    if len(failures):
        print(f"\n-- SKIPPED FOLDS ({len(failures)}) --")
        for row in failures.itertuples():
            print(f"  {row.season} {row.variant}: {row.error}")


def main(argv=None) -> None:
    """CLI entry point. `--no-save` runs the walk without writing artifacts;
    the module docstring is what `--help` prints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="snapshot directory to read")
    parser.add_argument("--no-save", action="store_true", help="run without writing artifacts")
    args = parser.parse_args(argv)

    matchups, games = load_matchups(args.snapshot)
    result = run_backtest(matchups)
    print_report(result)
    if not args.no_save:
        saved = save_backtest(result, matchups, games, args.snapshot)
        print(f"\nartifacts: {saved['out_dir']}")


if __name__ == "__main__":
    main()
