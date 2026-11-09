"""
Hand-computed checks for src/evaluation/metrics.py (docs/next-implementations.md
Stage 0 acceptance criteria): log loss/Brier/accuracy/ROC AUC/calibration
bins against known values, invalid-probability rejection, and documented
behavior on an empty sample, an empty bin, one-class labels, and
probabilities at exactly 0/1.

Run: python3 tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import (
    log_loss, brier_score, accuracy, roc_auc, calibration_bins,
    validate_probabilities, align_by_game_id, predictions_for_scoring, LOG_LOSS_EPS,
)


def _assert_raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        assert False, f"expected ValueError from {fn.__name__}{args}"
    except ValueError:
        pass


def demo():
    # --- log loss: hand-computed, clipping, validation ---------------------
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.1, 0.5, 0.5])
    want = -np.mean([np.log(0.9), np.log(0.9), np.log(0.5), np.log(0.5)])
    assert np.isclose(log_loss(y, p), want)

    # exactly 0/1 probabilities: clipped, finite, not rejected
    ll = log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert np.isfinite(ll) and np.isclose(ll, -np.log(1 - LOG_LOSS_EPS))

    for bad in (np.array([1.5]), np.array([-0.1]), np.array([np.nan]), np.array([np.inf])):
        try:
            validate_probabilities(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass

    assert np.isnan(log_loss(np.array([]), np.array([])))

    # one-class labels: log loss/Brier/accuracy still computable
    y_one = np.array([1.0, 1.0, 1.0])
    p_one = np.array([0.6, 0.7, 0.8])
    assert np.isfinite(log_loss(y_one, p_one))
    assert np.isfinite(brier_score(y_one, p_one))
    assert np.isfinite(accuracy(y_one, p_one))

    # --- Brier ---------------------------------------------------------------
    assert np.isclose(brier_score(y, p), np.mean((p - y) ** 2))
    assert np.isnan(brier_score(np.array([]), np.array([])))

    # --- accuracy: threshold boundary, >= counts as a home-win prediction ---
    assert accuracy(np.array([1.0]), np.array([0.5]), threshold=0.5) == 1.0
    assert accuracy(np.array([0.0]), np.array([0.4999]), threshold=0.5) == 1.0
    assert np.isclose(accuracy(np.array([1, 0, 1, 1]), np.array([0.6, 0.6, 0.4, 0.9])), 0.5)
    assert np.isnan(accuracy(np.array([]), np.array([])))

    # --- ROC AUC: perfect separation, one-class undefined, empty undefined --
    y_sep = np.array([0, 0, 1, 1])
    p_sep = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y_sep, p_sep) == 1.0
    assert roc_auc(np.array([0, 1]), np.array([0.9, 0.1])) == 0.0  # perfectly wrong ordering
    assert np.isnan(roc_auc(y_one, p_one))  # one-class
    assert np.isnan(roc_auc(np.array([]), np.array([])))

    # --- calibration bins: edges, empty bin, empty sample --------------------
    y_cal = np.array([1.0, 0.0, 1.0, 1.0])
    p_cal = np.array([0.0, 1.0, 0.35, 0.29999999999999999])
    bins = calibration_bins(y_cal, p_cal, n_bins=10)
    assert len(bins) == 10
    assert bins.loc[0, "count"] == 1 and bins.loc[0, "mean_predicted"] == 0.0  # p=0.0 -> bin 0
    assert bins.loc[9, "count"] == 1 and bins.loc[9, "mean_predicted"] == 1.0  # p=1.0 -> last bin
    assert bins.loc[3, "count"] == 2  # both 0.35 and the float-noisy 0.3 land in [0.3, 0.4)
    assert np.isclose(bins.loc[3, "observed_frequency"], 1.0)
    empty_bins = [i for i in range(10) if i not in (0, 3, 9)]
    for i in empty_bins:
        assert bins.loc[i, "count"] == 0
        assert np.isnan(bins.loc[i, "mean_predicted"])
        assert np.isnan(bins.loc[i, "observed_frequency"])

    empty = calibration_bins(np.array([]), np.array([]))
    assert (empty["count"] == 0).all()
    assert empty["mean_predicted"].isna().all()

    print("OK -- log loss/Brier/accuracy/ROC AUC/calibration bins match hand computation, "
          "reject invalid probabilities, and handle empty/one-class inputs as documented.")


def demo_edge_cases_and_outliers():
    # --- length mismatch: must raise, never silently broadcast --------------
    # A length-1 probability array against a length-3 label array is valid
    # numpy broadcasting, so without an explicit shape check this would
    # silently produce a number instead of failing -- exactly the kind of
    # caller alignment bug (e.g. scoring against the wrong subset) that
    # must be loud, not quietly wrong.
    y3, p1 = np.array([1.0, 0.0, 1.0]), np.array([0.5])
    _assert_raises(log_loss, y3, p1)
    _assert_raises(brier_score, y3, p1)
    _assert_raises(accuracy, y3, p1)
    _assert_raises(roc_auc, y3, p1)
    _assert_raises(calibration_bins, y3, p1)
    # a same-length-but-still-wrong pairing (2 vs 3) must also raise
    _assert_raises(log_loss, y3, np.array([0.5, 0.5]))

    # --- invalid labels: not 0/1, or non-finite -----------------------------
    # A stray 0.5/2/-1/NaN in y_true usually means an unfiltered or
    # mislabeled column reached the metric -- it must fail loudly, the same
    # way an invalid probability does, not get treated as an ordinary float.
    for bad_label in (np.array([2.0]), np.array([-1.0]), np.array([0.5]), np.array([np.nan]), np.array([np.inf])):
        for fn in (log_loss, brier_score, accuracy, roc_auc, calibration_bins):
            _assert_raises(fn, bad_label, np.array([0.5]))

    # boolean-coded labels (0/1 as bool, not float) are still accepted
    assert np.isfinite(log_loss(np.array([True, False]), np.array([0.6, 0.4])))

    # --- outlier probabilities: near float precision limits -----------------
    tiny, huge = 1e-300, 1 - 1e-16
    assert np.isfinite(log_loss(np.array([1.0, 0.0]), np.array([huge, tiny])))
    assert np.isclose(brier_score(np.array([1.0]), np.array([tiny])), 1.0, atol=1e-6)

    # eps=0 is a legitimate (if extreme) caller choice: an exact-0
    # probability on a positive label is genuinely infinite surprise, not
    # silently clamped away.
    assert log_loss(np.array([1.0]), np.array([0.0]), eps=0.0) == float("inf")

    # A *correct* boundary prediction (y=0,p=0 or y=1,p=1) must score exactly
    # 0, not NaN. The naive `y*log(p) + (1-y)*log(1-p)` formula fails this:
    # the losing term's coefficient is 0, but IEEE 0 * log(0) = 0 * -inf =
    # nan, not 0 -- select the observed-class term instead of computing both.
    assert log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]), eps=0.0) == 0.0
    assert log_loss(np.array([0.0]), np.array([0.0]), eps=0.0) == 0.0
    assert log_loss(np.array([1.0]), np.array([1.0]), eps=0.0) == 0.0

    # eps must be validated, not silently accepted -- a negative eps or one
    # >= 0.5 would clip a probability past its own decision boundary.
    for bad_eps in (-0.1, 0.5, float("nan"), float("inf")):
        _assert_raises(log_loss, np.array([1.0]), np.array([0.5]), eps=bad_eps)

    # --- ROC AUC: every probability identical (no ranking signal at all) ----
    # A completely uninformative model scores exactly 0.5, the same as a
    # coin flip -- not NaN (both classes are present) and not 1.0/0.0.
    assert roc_auc(np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5

    # --- accuracy: threshold outside [0, 1] is permissive by design --------
    # threshold is a caller-supplied config constant (ACCURACY_THRESHOLD),
    # not adversarial input data, so it is intentionally not range-checked:
    # threshold <= 0 predicts "home" unconditionally, threshold > 1 predicts
    # "away" unconditionally.
    assert accuracy(np.array([1.0, 0.0]), np.array([0.1, 0.9]), threshold=-1.0) == 0.5
    assert accuracy(np.array([1.0, 0.0]), np.array([0.1, 0.9]), threshold=2.0) == 0.5

    # --- calibration bins: n_bins=1 collapses to a single full-range bin ----
    single = calibration_bins(np.array([1.0, 0.0]), np.array([0.2, 0.8]), n_bins=1)
    assert len(single) == 1
    assert single.loc[0, "bin_low"] == 0.0 and single.loc[0, "bin_high"] == 1.0
    assert single.loc[0, "count"] == 2

    # a large outlier cluster of many rows at the exact same boundary value
    # (e.g. repeated market quotes) lands together, not split across bins
    clustered_p = np.full(50, 0.7)
    clustered_y = np.array([1.0] * 35 + [0.0] * 15)
    bins = calibration_bins(clustered_y, clustered_p)
    assert bins.loc[7, "count"] == 50, "all-identical 0.7 probabilities must land in one bin ([0.7, 0.8))"
    assert np.isclose(bins.loc[7, "observed_frequency"], 0.7)

    print("OK -- length-mismatch and invalid-label inputs raise, outlier probabilities "
          "(near 0/1, eps=0, uninformative/tied AUC, single-bin and clustered calibration) "
          "behave as documented.")


def demo_align_by_game_id_rejects_misaligned_populations():
    """Scoring two equal-length vectors that describe *different games* is a
    silent wrong number -- `_validate` only checks shape. This is the
    "duplicate/misaligned game IDs are rejected; paired comparisons use
    identical IDs" requirement."""
    labels = pd.Series([1.0, 0.0, 1.0], index=["g1", "g2", "g3"])
    preds = pd.Series([0.6, 0.4, 0.7], index=["g1", "g2", "g3"])

    y, p = align_by_game_id(labels, preds)
    assert y.tolist() == [1.0, 0.0, 1.0] and p.tolist() == [0.6, 0.4, 0.7]
    assert isinstance(y, np.ndarray) and y.dtype == np.float64

    # Order must not matter: same games, shuffled, align correctly rather than
    # zipping positionally into the wrong pairs.
    shuffled = preds.reindex(["g3", "g1", "g2"])
    y2, p2 = align_by_game_id(labels, shuffled)
    assert p2.tolist() == [0.6, 0.4, 0.7], "predictions must be matched by ID, not position"
    assert y2.tolist() == y.tolist()

    # Entirely different populations of the same length: the exact case that
    # previously scored 1.3862943611198906 without complaint.
    other = pd.Series([0.6, 0.4, 0.7], index=["gX", "gY", "gZ"])
    try:
        align_by_game_id(labels, other)
        assert False, "expected ValueError for predictions covering different games"
    except ValueError as exc:
        assert "different games" in str(exc)

    # Partial overlap is just as wrong as none.
    partial = pd.Series([0.6, 0.4], index=["g1", "g2"])
    try:
        align_by_game_id(labels, partial)
        assert False, "expected ValueError for a missing prediction"
    except ValueError as exc:
        assert "no prediction" in str(exc) and "g3" in str(exc)

    # Duplicate and null IDs on either side.
    dup = pd.Series([0.6, 0.4, 0.7], index=["g1", "g1", "g3"])
    _assert_raises(align_by_game_id, labels, dup)
    _assert_raises(align_by_game_id, pd.Series([1.0, 0.0], index=["g1", None]),
                   pd.Series([0.6, 0.4], index=["g1", None]))

    # An empty pairing is legitimate (a fold with no eligible rows) and the
    # scalar metrics already return NaN for it.
    y3, p3 = align_by_game_id(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert len(y3) == 0 and np.isnan(log_loss(y3, p3))

    print("OK -- align_by_game_id matches by ID not position, rejects different or "
          "partially overlapping populations, and allows an empty pairing.")


def demo_predictions_for_scoring_bridges_the_two_prediction_shapes():
    """Stage 2 returns a game-ID-indexed Series; Stage 3 returns a frame with
    abstentions. Both must reach the metrics through one conversion."""
    frame = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "p_home": [0.6, np.nan, 0.7],          # g2 abstained
        "abstention_reason": [None, "no_paired_market_quote", None],
    })
    scorable = predictions_for_scoring(frame)
    assert list(scorable.index) == ["g1", "g3"], "abstaining rows carry no probability to score"
    assert scorable.tolist() == [0.6, 0.7]
    assert scorable.name == "p_home" and scorable.dtype == np.float64

    # The dropped row must not pass unnoticed: labels still covering g2 raise,
    # rather than quietly scoring a smaller sample than the caller believes.
    labels = pd.Series([1.0, 0.0, 1.0], index=["g1", "g2", "g3"])
    try:
        align_by_game_id(labels, scorable)
        assert False, "expected the shrunken population to be caught"
    except ValueError as exc:
        assert "different games" in str(exc) and "g2" in str(exc)

    # Restricting the labels deliberately is what makes it scoreable.
    y, p = align_by_game_id(labels.drop("g2"), scorable)
    assert np.isfinite(log_loss(y, p))

    # Left in, a NaN would clear alignment and fail later inside log_loss with
    # a message about finiteness that names nothing about abstention.
    raw = pd.Series(frame.p_home.to_numpy(), index=pd.Index(frame.game_id))
    y_raw, p_raw = align_by_game_id(labels, raw)
    _assert_raises(log_loss, y_raw, p_raw)

    # A frame with no abstentions passes through whole.
    clean = frame.assign(p_home=[0.6, 0.5, 0.7], abstention_reason=None)
    assert len(predictions_for_scoring(clean)) == 3

    for missing in ("game_id", "p_home"):
        try:
            predictions_for_scoring(frame.drop(columns=[missing]))
            assert False, f"expected KeyError without {missing}"
        except KeyError:
            pass

    print("OK -- predictions_for_scoring converts a prediction frame to the scorer's "
          "Series, drops abstentions, and leaves the coverage change impossible to miss.")


if __name__ == "__main__":
    demo()
    demo_edge_cases_and_outliers()
    demo_align_by_game_id_rejects_misaligned_populations()
    demo_predictions_for_scoring_bridges_the_two_prediction_shapes()
