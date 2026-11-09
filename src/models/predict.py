"""
Batch prediction from a saved Stage 4 artifact: load one fold's fitted
pipeline + schema, score eligible pregame rows, return probabilities or an
explicit abstention, each row stamped with the run and model that produced it.

This is the Stage 6 prediction interface (docs/next-implementations.md). It
deliberately adds no modelling: `src/models/train.py` already owns the schema
check, the abstention contract and the save/load format, so this module is the
artifact-path resolution and identity stamping around it. Nothing here selects
a model, a threshold or a C -- re-selection against a final result would make
the result a development experiment (model-plan.md section 5).

Labels are never read. A row scores identically whether or not its outcome is
known, which is what makes the same code path usable before kickoff and in
replay -- `tests/test_predict.py` proves it rather than asserting it.

A Stage 4 run is a development backtest, not a final validated evaluation: no
untouched holdout exists (docs/experiment-protocol.md, "Holdout designation").
`load_saved_model` carries the run manifest's `study_type` forward and the CLI
prints it, so a prediction cannot be presented as a validated number by
accident.

Run: python3 -m src.models.predict --run-dir models_game/home_win/<run_id> \
         --variant team_only --season 2024 --snapshot snapshots/<sha>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import ARTIFACT_ROOT, TARGET
from src.evaluation.diagnose import _sha256_file
from src.models.train import load_model, predict_model

DEFAULT_MODEL = "logistic"  # the only v1 family (model-plan.md section 3)
MANIFEST_FILENAME = "manifest.json"


def saved_model_path(run_dir, variant: str, season: int, model: str = DEFAULT_MODEL) -> Path:
    """`<run_dir>/<variant>/<model>/season_<T>.joblib`, as `save_backtest` wrote it."""
    return Path(run_dir) / variant / model / f"season_{int(season)}.joblib"


def _available_seasons(run_dir, variant: str, model: str) -> list:
    directory = Path(run_dir) / variant / model
    return sorted(int(p.stem.removeprefix("season_")) for p in directory.glob("season_*.joblib"))


def load_saved_model(run_dir, variant: str, season: int, model: str = DEFAULT_MODEL) -> dict:
    """The fitted bundle plus the identity every prediction from it must carry.

    A missing fold is reported with the seasons the run actually holds:
    `team_market` legitimately starts at 2008 while `team_only` starts at 2005
    (docs/next-implementations.md Stage 4), so "no such season" is an ordinary
    coverage fact here, not necessarily a broken artifact.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{run_dir} is not a run directory: no {MANIFEST_FILENAME}")
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("target") != TARGET:
        raise ValueError(
            f"run {run_dir} targets {manifest.get('target')!r}, but this interface "
            f"predicts {TARGET!r}; probabilities from the two are not the same quantity"
        )

    path = saved_model_path(run_dir, variant, season, model)
    if not path.exists():
        raise FileNotFoundError(
            f"no saved {variant}/{model} model for season {season} in {run_dir} "
            f"(seasons present: {_available_seasons(run_dir, variant, model)})"
        )

    # joblib is pickle: loading a file runs whatever code it carries. Check the
    # bytes against the checksum save_backtest recorded BEFORE unpickling, and
    # refuse a file the manifest does not vouch for at all -- that also covers a
    # --variant/--model value that resolves outside the run directory.
    # Known limit: the manifest sits beside the models, so a writer who can replace
    # both defeats this; it stops a swapped or dropped-in file, not a compromised
    # run directory. Pin the manifest's own hash out-of-band if that matters.
    key = path.relative_to(run_dir).as_posix()
    expected = (manifest.get("model_sha256") or {}).get(key)
    if expected is None:
        raise ValueError(
            f"run manifest records no checksum for {key}; refusing to unpickle an unverified "
            f"model (runs saved before model checksums were recorded must be re-run)"
        )
    if _sha256_file(path) != expected:
        raise ValueError(
            f"{path} does not match the checksum its run manifest recorded; refusing to unpickle it"
        )

    bundle = load_model(path)
    if bundle["variant"] != variant:
        raise ValueError(
            f"{path} holds a {bundle['variant']!r} model but was loaded as {variant!r}; "
            f"its feature schema is not the one the caller expects"
        )

    return {
        "bundle": bundle,
        "run_id": manifest.get("run_id"),
        "target": TARGET,
        "variant": variant,
        "model": model,
        "fitted_season": int(season),
        "study_type": manifest.get("study_type"),
        "path": path,
    }


def predict_games(saved: dict, matchups: pd.DataFrame) -> pd.DataFrame:
    """One row per input game: the probability or the abstention, plus identity.

    Never a filtered subset, and never a label read -- `predict_model` selects
    only allowlisted feature columns, so an outcome column present in
    `matchups` is ignored and an absent one changes nothing.
    """
    predictions = predict_model(saved["bundle"], matchups)
    for column in ("run_id", "target", "variant", "model", "fitted_season"):
        predictions[column] = saved[column]
    return predictions


def main(argv=None) -> None:
    """CLI entry point; the module docstring is what `--help` prints."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help=f"a {ARTIFACT_ROOT}/{TARGET}/<run_id> directory")
    parser.add_argument("--variant", required=True, help="team_only, team_market or team_spread")
    parser.add_argument("--season", required=True, type=int, help="which fold's fitted model to load")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--snapshot", required=True, help="snapshot directory supplying the games to score")
    parser.add_argument("--score-season", type=int, default=None,
                        help="season to score (default: the fitted season, i.e. that fold's test year)")
    parser.add_argument("--out", default=None, help="write predictions to this CSV instead of printing them")
    args = parser.parse_args(argv)

    from src.evaluation.backtest import load_matchups  # heavy; only the CLI needs it

    saved = load_saved_model(args.run_dir, args.variant, args.season, args.model)
    matchups, _ = load_matchups(args.snapshot)
    score_season = args.season if args.score_season is None else args.score_season
    matchups = matchups.loc[matchups["home_season"].eq(score_season)]
    if matchups.empty:
        raise SystemExit(f"no games in season {score_season} in {args.snapshot}")

    predictions = predict_games(saved, matchups)
    scored = int(predictions["p_home"].notna().sum())
    print(f"study type: {saved['study_type']}")
    print(f"run {saved['run_id']} / {args.variant} / {args.model} / fitted on season {args.season}")
    print(f"scoring season {score_season}: {scored} scored, {len(predictions) - scored} abstained")
    if args.out:
        predictions.to_csv(args.out, index=False)
        print(f"predictions: {args.out}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
