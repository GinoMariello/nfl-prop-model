"""
Hand-computed checks for src/models/baselines.py (docs/next-implementations.md
Stage 0 and Stage 2 acceptance criteria; design: docs/stage2-design.md).

Stage 0: always-home exposes accuracy only, matches the observed home-win
rate, and handles an empty input without raising.

Stage 2: home-frequency fits on training-fold rows only and stays fixed,
one-class folds are not smoothed, and scoring pairs labels to predictions
by game ID rather than by position.

Run: python3 tests/test_baselines.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.baselines import always_home_accuracy, fit_home_frequency, predict_home_frequency
from src.evaluation.metrics import log_loss, align_by_game_id


def demo():
    result = always_home_accuracy(pd.Series([True, True, False, True, False, False, True]))
    assert result == {"accuracy": 4 / 7, "n": 7}

    empty = always_home_accuracy(pd.Series([], dtype=bool))
    assert empty["n"] == 0 and np.isnan(empty["accuracy"])

    all_home = always_home_accuracy(pd.Series([True, True, True]))
    assert all_home == {"accuracy": 1.0, "n": 3}

    all_away = always_home_accuracy(pd.Series([False, False]))
    assert all_away == {"accuracy": 0.0, "n": 2}

    assert set(result.keys()) == {"accuracy", "n"}, \
        "always-home must expose accuracy only, never a probability (model-plan.md B7)"

    print("OK -- always_home_accuracy matches hand computation, exposes accuracy only, and handles the empty case.")


def demo_edge_cases_and_outliers():
    # A caller passing an unfiltered label column (still containing a null
    # from an undecided/tied/unplayed game) must fail loudly, not silently
    # treat the missing value as a win or a loss -- this is the contract
    # documented on always_home_accuracy ("already restricted to eligible,
    # non-null labels").
    try:
        always_home_accuracy(pd.Series([True, False, pd.NA], dtype="boolean"))
        assert False, "expected an error for a Series still containing a null label"
    except (ValueError, TypeError):
        pass

    # Single-observation extremes: a lone game shouldn't be treated as
    # "too small to score" -- it's a real (if statistically useless) 0%/100%.
    assert always_home_accuracy(pd.Series([True])) == {"accuracy": 1.0, "n": 1}
    assert always_home_accuracy(pd.Series([False])) == {"accuracy": 0.0, "n": 1}

    # Integer-coded 0/1 labels (not Python bool) must be accepted the same
    # way -- a season pulled straight from SQLite round-trips booleans as
    # int64 (see rolling_features.py's is_home cast for the same reason).
    assert always_home_accuracy(pd.Series([1, 0, 1, 1])) == {"accuracy": 0.75, "n": 4}

    # A large, heavily lopsided outlier population (e.g. a hypothetical
    # season where the home team almost never lost) is just the observed
    # rate -- no clamping or smoothing toward 0.5.
    lopsided = pd.Series([True] * 999 + [False])
    result = always_home_accuracy(lopsided)
    assert result["n"] == 1000 and np.isclose(result["accuracy"], 0.999)

    print("OK -- always_home_accuracy rejects a still-null label column, handles single-game "
          "and heavily lopsided populations, and accepts int-coded 0/1 labels.")


def demo_null_labels_are_rejected_in_every_dtype():
    """Regression for a silent miscount (docs/stage2-design.md section 2).

    `.astype(bool)` maps `np.nan` to True, and `home_win` comes back from
    SQLite as float64 rather than Int64 -- so its nulls are `np.nan`, and an
    unfiltered column counted every unplayed or tied game as a home win.
    Measured on the first 5000 real matchup rows: n=5000 / 0.5726 reported
    against a true n=4991 / 0.5718, nine games miscounted.

    The old test only covered `dtype="boolean"` carrying `pd.NA`, which did
    raise. The float64 path -- the one the real database actually produces,
    and the one a saved scored_rows.csv round-trips into -- was uncovered.
    """
    nulls_by_dtype = {
        "float64 np.nan (what SQLite yields)": pd.Series([1.0, 0.0, np.nan]),
        "boolean pd.NA": pd.Series([True, False, pd.NA], dtype="boolean"),
        "Int64 pd.NA": pd.Series([1, 0, None], dtype="Int64"),
        "object None": pd.Series([True, False, None], dtype="object"),
    }
    for label, series in nulls_by_dtype.items():
        try:
            result = always_home_accuracy(series)
            assert False, f"{label}: expected ValueError, got {result}"
        except ValueError as exc:
            assert "null" in str(exc).lower(), f"{label}: unhelpful message {exc!r}"

    # The specific shape of the old bug: three rows in, one of them null.
    # Silently returning n == 3 is the failure, not merely "did not raise".
    try:
        counted = always_home_accuracy(pd.Series([1.0, 0.0, np.nan]))
        assert False, f"a float64 null must never be counted as a game, got {counted}"
    except ValueError:
        pass

    # Non-labels must not slip through as though they were 0/1.
    for bad in (pd.Series([0.5, 1.0]), pd.Series([2, 0]), pd.Series([-1, 1]),
                pd.Series(["2020-09-10", "2020-09-11"])):
        try:
            always_home_accuracy(bad)
            assert False, f"expected ValueError for non-0/1 labels: {list(bad)}"
        except ValueError as exc:
            assert "0/1" in str(exc), f"unhelpful message {exc!r}"

    # Valid inputs are untouched by the tightened gate, including the
    # documented empty-input contract.
    assert always_home_accuracy(pd.Series([1.0, 0.0, 1.0])) == {"accuracy": 2 / 3, "n": 3}
    empty = always_home_accuracy(pd.Series([], dtype=bool))
    assert empty["n"] == 0 and np.isnan(empty["accuracy"])

    print("OK -- null labels are rejected in float64/boolean/Int64/object dtypes alike, "
          "non-0/1 values are rejected, and valid and empty inputs are unaffected.")


def demo_fit_home_frequency_matches_hand_computation():
    assert fit_home_frequency(pd.Series([True, True, False, True, False, False, True])) == \
        {"p_home": 4 / 7, "n": 7}
    assert 4 / 7 == 0.5714285714285714  # pins the fixture's expected value

    assert fit_home_frequency(pd.Series([1, 0, 1, 1])) == {"p_home": 0.75, "n": 4}
    assert fit_home_frequency(pd.Series([1.0, 0.0])) == {"p_home": 0.5, "n": 2}

    # The fitted baseline exposes a probability; always-home must never
    # (model-plan.md B7). Assert both contracts together so the distinction
    # between the two baselines cannot silently erode.
    assert set(fit_home_frequency(pd.Series([1, 0]))) == {"p_home", "n"}
    assert set(always_home_accuracy(pd.Series([1, 0]))) == {"accuracy", "n"}

    # An empty training sample fails explicitly -- unlike always_home_accuracy,
    # which documents a NaN return. A fitted parameter with no training data is
    # not a model, and a quiet NaN would propagate into predictions and scores.
    try:
        fit_home_frequency(pd.Series([], dtype=float))
        assert False, "expected ValueError for an empty training sample"
    except ValueError as exc:
        assert "empty" in str(exc).lower(), f"unhelpful message {exc!r}"
    assert always_home_accuracy(pd.Series([], dtype=bool))["n"] == 0  # contrast, still NaN

    # The shared label gate applies here too -- no second validation path.
    for bad in (pd.Series([1.0, np.nan]), pd.Series([0.5, 1.0]), pd.Series([2, 0])):
        try:
            fit_home_frequency(bad)
            assert False, f"expected ValueError for {list(bad)}"
        except ValueError:
            pass

    print("OK -- fit_home_frequency matches hand computation, exposes a probability, "
          "fails explicitly on an empty sample, and shares the label gate.")


def demo_one_class_training_fold_is_not_smoothed():
    all_home = fit_home_frequency(pd.Series([True, True, True]))
    all_away = fit_home_frequency(pd.Series([False, False]))
    # Exact equality, not isclose: a smoothed value would still be "close".
    assert all_home["p_home"] == 1.0
    assert all_away["p_home"] == 0.0

    # Scored against a fold containing one away win, p=1.0 is penalized by the
    # documented eps=1e-15 clipping in metrics.log_loss -- large but finite,
    # never inf, and never softened toward 0.5 because it scored badly.
    p = predict_home_frequency(all_home, ["g1", "g2", "g3"])
    loss = log_loss(np.array([1.0, 1.0, 0.0]), p.to_numpy())
    assert np.isfinite(loss) and loss > 11.0
    assert np.isclose(loss, 11.51319199744696)
    assert np.isclose(-np.log(1e-15), 34.538776394910684)  # the clipped per-row penalty

    print("OK -- a one-class fold fits exactly 0.0/1.0 with no smoothing, and scores "
          "through log loss's documented clipping as a large finite penalty.")


def demo_fitted_value_cannot_be_changed_by_later_labels():
    # Train and test folds deliberately have DIFFERENT home-win rates, so a
    # leak changes the number instead of coincidentally matching it.
    train = pd.Series([True, True, True, False])   # 0.75
    test_labels = pd.Series([False, False, False, False])  # 0.0
    fitted = fit_home_frequency(train)
    assert fitted == {"p_home": 0.75, "n": 4}

    # Mutating the training input in place afterwards cannot reach the fit.
    train.iloc[0] = False
    assert fitted == {"p_home": 0.75, "n": 4}
    train = pd.Series([False] * 100)  # rebinding the name changes nothing either
    assert fitted == {"p_home": 0.75, "n": 4}

    # Predictions for the test fold carry the TRAINING rate, not the test rate.
    p = predict_home_frequency(fitted, [f"g{i}" for i in range(len(test_labels))])
    assert (p == 0.75).all(), "predictions must use the fitted training-fold rate"
    assert p.mean() != test_labels.mean(), "a leak would have produced the test-fold rate"

    print("OK -- an already-fitted value is immune to later mutation of the training "
          "labels and never reflects validation/test outcomes.")


def demo_predictions_are_constant_and_reject_bad_ids():
    fitted = fit_home_frequency(pd.Series([True, True, False, True]))
    p = predict_home_frequency(fitted, ["g1", "g2", "g3"])
    assert len(p) == 3 and (p == fitted["p_home"]).all()
    assert list(p.index) == ["g1", "g2", "g3"]
    assert p.name == "p_home" and p.dtype == np.float64

    # A duplicate ID would fan out in a later join and double-count that game.
    try:
        predict_home_frequency(fitted, ["g1", "g1"])
        assert False, "expected ValueError for duplicate game_ids"
    except ValueError as exc:
        assert "unique" in str(exc)

    for bad_ids in (["g1", None], ["g1", np.nan]):
        try:
            predict_home_frequency(fitted, bad_ids)
            assert False, f"expected ValueError for {bad_ids}"
        except ValueError as exc:
            assert "non-null" in str(exc)

    # An empty scoring subset is legitimate (e.g. a fold with no paired
    # quotes) and must stay float64, not become an object Series.
    empty = predict_home_frequency(fitted, [])
    assert len(empty) == 0 and empty.dtype == np.float64

    # Scoring two different subsets reuses one fitted value -- never refits.
    full = predict_home_frequency(fitted, ["g1", "g2", "g3", "g4"])
    subset = predict_home_frequency(fitted, ["g2", "g4"])
    assert set(full.unique()) == set(subset.unique()) == {fitted["p_home"]}

    print("OK -- predictions are the constant fitted value, reject duplicate/null game "
          "IDs, keep float64 when empty, and never refit when scoring a subset.")


def demo_end_to_end_fold_scoring_is_id_aligned():
    """The Stage 4 shape: fit on a training fold, score a test fold and a
    paired-quote subset of it, all matched by game ID rather than position."""
    train = pd.Series([1, 1, 1, 0], index=["t1", "t2", "t3", "t4"])
    fitted = fit_home_frequency(train)
    assert fitted == {"p_home": 0.75, "n": 4}

    test_labels = pd.Series([1.0, 0.0, 1.0], index=["g1", "g2", "g3"])
    preds = predict_home_frequency(fitted, test_labels.index)
    y, p = align_by_game_id(test_labels, preds)
    assert p.tolist() == [0.75, 0.75, 0.75]
    full_loss = log_loss(y, p)
    assert np.isfinite(full_loss)

    # The paired-quote subset: a second scoring population off the SAME fitted
    # value. Scoring a subset must never refit (next-implementations.md Stage 4).
    paired = test_labels.loc[["g1", "g3"]]
    subset_preds = predict_home_frequency(fitted, paired.index)
    y_s, p_s = align_by_game_id(paired, subset_preds)
    assert y_s.tolist() == [1.0, 1.0] and set(p_s) == {0.75}, \
        "the subset must reuse the fitted value, not a refit one"
    assert fit_home_frequency(paired)["p_home"] == 1.0, \
        "refitting on this subset would give 1.0 -- proving the subset score is not a refit"

    # Test-fold rows arriving in a different order than the predictions still
    # pair correctly, because alignment is by ID.
    shuffled = test_labels.reindex(["g3", "g1", "g2"])
    y_sh, p_sh = align_by_game_id(shuffled, preds)
    assert np.isclose(log_loss(y_sh, p_sh), full_loss)

    print("OK -- end-to-end fold scoring aligns by game ID, and a subset score reuses "
          "the fitted value rather than refitting.")


if __name__ == "__main__":
    demo()
    demo_edge_cases_and_outliers()
    demo_null_labels_are_rejected_in_every_dtype()
    demo_fit_home_frequency_matches_hand_computation()
    demo_one_class_training_fold_is_not_smoothed()
    demo_fitted_value_cannot_be_changed_by_later_labels()
    demo_predictions_are_constant_and_reject_bad_ids()
    demo_end_to_end_fold_scoring_is_id_aligned()
