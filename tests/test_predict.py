"""
Tests for src/models/predict.py (docs/next-implementations.md Stage 6).

The one that matters is demo_a_removed_label_changes_nothing: the whole point
of this interface is that it runs identically before kickoff and in replay, so
a label column leaking into the feature path would be invisible in the output
and would quietly invalidate every pregame use. It is proven by scoring the
same rows twice -- once with the outcome present, once with every outcome
column deleted -- and asserting bit-identical probabilities.

Synthetic and offline; no database, no network.

Run: python3 tests/test_predict.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.rolling_features import TEAM_FEATURE_COLUMNS
from src.models.predict import (
    DEFAULT_MODEL, load_saved_model, predict_games, saved_model_path,
)
from src.evaluation.diagnose import _sha256_file
from src.models.train import NO_QUOTE, fit_model, load_model, save_model

SEASON = 2024


def _matchups(n=120, seed=0):
    """Feature frame with a learnable signal, a label, and a market column."""
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in TEAM_FEATURE_COLUMNS})
    frame.insert(0, "game_id", [f"g{i:04d}" for i in range(n)])
    signal = frame[TEAM_FEATURE_COLUMNS[0]].to_numpy()
    frame["home_win"] = (signal + rng.normal(scale=0.5, size=n) > 0).astype(float)
    frame["market_home_probability"] = np.clip(0.5 + 0.2 * signal, 0.02, 0.98)
    frame["home_score"] = rng.randint(0, 40, size=n)
    frame["away_score"] = rng.randint(0, 40, size=n)
    return frame


def _run_dir(root, matchups, variant="team_only", target="home_win", season=SEASON):
    """A minimal stand-in for what `save_backtest` writes: manifest + one fold."""
    run_dir = Path(root) / "run_0001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_0001",
        "target": target,
        "study_type": "development backtest -- NOT a final evaluation",
        "model_sha256": {},
    }))
    bundle = fit_model(matchups, variant=variant)
    _vouch(run_dir, save_model(bundle, saved_model_path(run_dir, variant, season)))
    return run_dir, bundle


def _vouch(run_dir, path):
    """Record `path`'s checksum in the run manifest, as save_backtest does."""
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"][Path(path).relative_to(run_dir).as_posix()] = _sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))


def demo_load_and_predict_stamps_run_and_model_identity():
    """Every row names the run, variant, model and fold that produced it."""
    matchups = _matchups()
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)
        saved = load_saved_model(run_dir, "team_only", SEASON)
        assert saved["run_id"] == "run_0001"
        assert saved["fitted_season"] == SEASON
        assert "NOT a final evaluation" in saved["study_type"], \
            "the development-only caveat must survive the load"

        predictions = predict_games(saved, matchups)

    assert list(predictions["game_id"]) == list(matchups["game_id"]), \
        "one output row per input game, in input order"
    assert predictions["p_home"].between(0, 1).all()
    assert predictions["abstention_reason"].isna().all()
    for column, value in (("run_id", "run_0001"), ("variant", "team_only"),
                          ("model", DEFAULT_MODEL), ("fitted_season", SEASON),
                          ("target", "home_win")):
        assert (predictions[column] == value).all(), f"{column} not stamped on every row"
    print("OK -- probabilities carry the run, variant, model and fold identity.")


def demo_a_removed_label_changes_nothing():
    """A completed row and the same pregame row without its outcome score alike.

    Deleting every outcome column must not move a single probability, and must
    not raise: that is the property that lets one code path serve replay and
    pregame use.
    """
    matchups = _matchups()
    pregame = matchups.drop(columns=["home_win", "home_score", "away_score"])

    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)
        saved = load_saved_model(run_dir, "team_only", SEASON)
        with_outcome = predict_games(saved, matchups)
        without_outcome = predict_games(saved, pregame)

    assert np.array_equal(with_outcome["p_home"].to_numpy(),
                          without_outcome["p_home"].to_numpy()), \
        "the outcome columns changed a probability -- a label is reaching the features"
    # Not vacuous: the model does respond to its actual inputs.
    moved = matchups.assign(**{TEAM_FEATURE_COLUMNS[0]: matchups[TEAM_FEATURE_COLUMNS[0]] + 5.0})
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)
        saved = load_saved_model(run_dir, "team_only", SEASON)
        assert not np.allclose(predict_games(saved, moved)["p_home"].to_numpy(),
                               with_outcome["p_home"].to_numpy()), \
            "a feature change did not move the probability -- the equality above proves nothing"
    print("OK -- a missing outcome neither blocks prediction nor changes one.")


def demo_abstention_is_explicit_not_a_dropped_row():
    """A market row without its quote comes back abstained, not missing."""
    matchups = _matchups()
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups, variant="team_market")
        saved = load_saved_model(run_dir, "team_market", SEASON)

        unquoted = matchups.copy()
        unquoted.loc[unquoted.index[:10], "market_home_probability"] = np.nan
        predictions = predict_games(saved, unquoted)

    assert len(predictions) == len(unquoted), "abstaining rows must not be dropped"
    abstained = predictions.loc[predictions["abstention_reason"].notna()]
    assert len(abstained) == 10
    assert (abstained["abstention_reason"] == NO_QUOTE).all()
    assert abstained["p_home"].isna().all(), "an abstention must not carry a probability"
    assert (abstained["run_id"] == "run_0001").all(), "abstentions are stamped too"
    print("OK -- an ineligible row abstains explicitly and keeps its identity.")


def demo_mismatched_artifacts_fail_loudly():
    """A wrong target, a wrong variant, a missing fold and a non-run directory."""
    matchups = _matchups()
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)

        try:
            load_saved_model(Path(root) / "not_a_run", "team_only", SEASON)
            assert False, "a directory without a manifest must not load"
        except FileNotFoundError as exc:
            assert "manifest.json" in str(exc)

        try:
            load_saved_model(run_dir, "team_only", 1999)
            assert False, "a fold that was never fitted must not load"
        except FileNotFoundError as exc:
            assert str(SEASON) in str(exc), "the error should name the seasons that do exist"

        # A bundle filed under the wrong variant's path: the schema it holds is
        # not the one the caller is about to hand it.
        mislabelled = fit_model(matchups, variant="team_market")
        _vouch(run_dir, save_model(mislabelled, saved_model_path(run_dir, "team_only", 2023)))
        try:
            load_saved_model(run_dir, "team_only", 2023)
            assert False, "a bundle whose variant contradicts its path must not load"
        except ValueError as exc:
            assert "team_market" in str(exc)

    with tempfile.TemporaryDirectory() as root:
        other_target, _ = _run_dir(root, matchups, target="point_diff")
        try:
            load_saved_model(other_target, "team_only", SEASON)
            assert False, "a run predicting a different target must not load"
        except ValueError as exc:
            assert "point_diff" in str(exc)
    print("OK -- wrong target, wrong variant, missing fold and non-run directory all raise.")


class _Payload:
    """A pickle that runs code on load: it creates `sentinel` as a side effect."""

    def __init__(self, sentinel):
        self.sentinel = str(sentinel)

    def __reduce__(self):
        return (os.mkdir, (self.sentinel,))


def demo_unverified_or_swapped_models_are_never_unpickled():
    """The checksum gate runs BEFORE joblib.load, so a hostile file never executes.

    Proven with a live payload rather than a corrupted byte: a model file
    replaced by a pickle that creates a directory when loaded. Loading through
    the interface must refuse it and the directory must not appear. Loading the
    same file directly with load_model DOES create it, so the payload is real
    and the refusal is what stopped it.
    """
    matchups = _matchups(n=40)
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)
        path = saved_model_path(run_dir, "team_only", SEASON)
        sentinel = Path(root) / "payload_ran"

        import joblib
        joblib.dump(_Payload(sentinel), path)  # swapped in; manifest still holds the real hash
        try:
            load_saved_model(run_dir, "team_only", SEASON)
            assert False, "a file that does not match its recorded checksum must not load"
        except ValueError as exc:
            assert "does not match the checksum" in str(exc)
        assert not sentinel.exists(), "the payload executed -- the file was unpickled before the check"

        load_model(path)  # control: bypassing the gate really does run the payload
        assert sentinel.exists(), "the payload is inert, so the refusal above proved nothing"
        sentinel.rmdir()

        # A run whose manifest vouches for nothing cannot be verified, so it is refused
        # outright -- the shape of every run saved before checksums were recorded.
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["model_sha256"]
        manifest_path.write_text(json.dumps(manifest))
        try:
            load_saved_model(run_dir, "team_only", SEASON)
            assert False, "a model with no recorded checksum must not load"
        except ValueError as exc:
            assert "no checksum" in str(exc)
        assert not sentinel.exists()
    print("OK -- a swapped or unvouched model file is refused before it is ever unpickled.")


def demo_duplicate_or_null_game_ids_are_rejected():
    """Identity stamping is worthless if the key it is stamped against repeats."""
    matchups = _matchups(n=40)
    with tempfile.TemporaryDirectory() as root:
        run_dir, _ = _run_dir(root, matchups)
        saved = load_saved_model(run_dir, "team_only", SEASON)

        duplicated = pd.concat([matchups, matchups.iloc[:1]], ignore_index=True)
        try:
            predict_games(saved, duplicated)
            assert False, "a repeated game_id must not be scored twice"
        except ValueError as exc:
            assert "unique" in str(exc)

        nulled = matchups.copy()
        nulled.loc[nulled.index[0], "game_id"] = None
        try:
            predict_games(saved, nulled)
            assert False, "a null game_id must not be scored"
        except ValueError as exc:
            assert "non-null" in str(exc)
    print("OK -- duplicate and null game IDs are rejected before anything is scored.")


if __name__ == "__main__":
    demo_load_and_predict_stamps_run_and_model_identity()
    demo_a_removed_label_changes_nothing()
    demo_abstention_is_explicit_not_a_dropped_row()
    demo_mismatched_artifacts_fail_loudly()
    demo_unverified_or_swapped_models_are_never_unpickled()
    demo_duplicate_or_null_game_ids_are_rejected()
    print("\nAll src/models/predict.py checks passed.")
