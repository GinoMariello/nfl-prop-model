"""
Scoring functions shared by every stage of docs/next-implementations.md
(Stage 0 diagnostic, Stage 2+ fitted baselines, Stage 4 backtest, Stage 5
report). Log loss is the primary metric (docs/model-plan.md §7); Brier,
accuracy, ROC AUC, and calibration bins are supporting diagnostics.

Every function validates both inputs the same way (`_validate`):
probabilities must be finite and in [0, 1] (`validate_probabilities`),
labels must be finite and strictly 0/1 -- not 0.5, 2, or a stray NaN
smuggled in from an unfiltered label column -- and the two must be the
same length. A length mismatch (e.g. a length-1 probability array against
a length-3 label array) silently broadcasts under plain numpy arithmetic
instead of raising; `_validate` catches that before it can produce a
meaningless number. Log loss additionally clips probabilities to
`[eps, 1-eps]` only to keep log(0)/log(1) finite -- that clip happens
after validation, never as a substitute for it, so a legitimate
probability of exactly 0 or 1 is scored (with documented clipping), while
an invalid one (e.g. 1.5, -0.1, nan) is rejected outright.

An empty sample returns NaN from every scalar metric rather than raising
or emitting a numpy warning from an empty mean. A one-class label vector
is a normal input for log loss/Brier/accuracy but leaves ROC AUC
undefined (no positive-negative pair exists to rank) -- it returns NaN.

`_validate` guards the *shape* of a comparison, not its identity: two
equal-length vectors describing different games score without complaint.
Callers holding game IDs should pair inputs through `align_by_game_id`
before scoring.

Run: python3 tests/test_metrics.py
"""
import numpy as np
import pandas as pd

N_CALIBRATION_BINS = 10
LOG_LOSS_EPS = 1e-15
ACCURACY_THRESHOLD = 0.5


def validate_probabilities(y_prob) -> np.ndarray:
    """Reject non-finite or out-of-[0,1] probabilities. Called by every
    metric below before any computation -- an invalid input must fail
    loudly here, not get silently clamped into range by a downstream
    clip."""
    p = np.asarray(y_prob, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite (no NaN/inf)")
    if p.size and ((p < 0).any() or (p > 1).any()):
        raise ValueError("probabilities must be within [0, 1]")
    return p


def _validate(y_true, y_prob) -> tuple:
    """Shared precondition for every metric below: `y_prob` passes
    `validate_probabilities`, `y_true` is finite and strictly 0/1 (a
    decided label, not a probability -- a stray 0.5/2/NaN usually means an
    unfiltered or mislabeled column reached here and should fail loudly,
    not get treated as an ordinary float), and both arrays are the same
    length. Plain numpy arithmetic on mismatched lengths silently
    broadcasts a length-1 array against a longer one instead of raising,
    which would otherwise turn a caller's alignment bug into a
    plausible-looking wrong number."""
    p = validate_probabilities(y_prob)
    y = np.asarray(y_true, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob must have the same shape, got {y.shape} and {p.shape}")
    if y.size and not np.isfinite(y).all():
        raise ValueError("y_true must be finite (no NaN/inf)")
    if y.size and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0/1 (or boolean) values")
    return y, p


def log_loss_terms(y_true, y_prob, eps: float = LOG_LOSS_EPS) -> np.ndarray:
    """Per-game log loss, before averaging.

    `log_loss` is the mean of this. Stage 5's paired bootstrap needs the
    individual terms -- a paired difference is the mean of per-game
    differences, and resampling seasons means re-averaging a subset of these
    thousands of times -- so they are computed once, here, rather than the
    formula being written a second time in the report.
    """
    y, p = _validate(y_true, y_prob)
    if not np.isfinite(eps) or not 0 <= eps < 0.5:
        raise ValueError("eps must be finite and in [0, 0.5)")
    if y.size == 0:
        return np.array([], dtype=float)
    p = np.clip(p, eps, 1 - eps)
    # Select the observed class's term before logging: computing both
    # y*log(p) and (1-y)*log(1-p) unconditionally hits 0*log(0)=nan at a
    # perfect prediction (e.g. y=0, p=0) even though that term's
    # coefficient is 0 -- IEEE 0*-inf is nan, not 0.
    with np.errstate(divide="ignore"):
        return np.where(y == 1, -np.log(p), -np.log1p(-p))


def log_loss(y_true, y_prob, eps: float = LOG_LOSS_EPS) -> float:
    """Mean binary log loss. Probabilities are clipped to `[eps, 1-eps]`
    only to keep log(0)/log(1) finite -- a documented numerical
    accommodation for an already-valid probability, not a fix for an
    invalid one (see `validate_probabilities`)."""
    losses = log_loss_terms(y_true, y_prob, eps)
    if losses.size == 0:
        return float("nan")
    return float(np.mean(losses))


def brier_score(y_true, y_prob) -> float:
    """Mean squared error between probability and outcome. No clipping --
    already validated to lie in [0, 1], and unlike log loss it doesn't
    blow up at the boundary."""
    y, p = _validate(y_true, y_prob)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def accuracy(y_true, y_prob, threshold: float = ACCURACY_THRESHOLD) -> float:
    """Fraction correctly classified at `threshold` (default 0.5, home-win
    predicted when `p >= threshold`)."""
    y, p = _validate(y_true, y_prob)
    if y.size == 0:
        return float("nan")
    predicted = (p >= threshold).astype(float)
    return float(np.mean(predicted == y))


def roc_auc(y_true, y_prob) -> float:
    """Rank-based AUC (Mann-Whitney U / (n_pos * n_neg)), computed without
    sklearn -- Stage 0 is fitting-free and dependency resolution is
    Stage 1's job (docs/next-implementations.md). Ties get the average
    rank via `pandas.Series.rank`. Undefined (NaN) when either class is
    absent -- there is no positive/negative pair to rank."""
    y, p = _validate(y_true, y_prob)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def predictions_for_scoring(frame: pd.DataFrame) -> pd.Series:
    """Turn a prediction frame into the game-ID-indexed Series the metrics take.

    Stage 2's `predict_home_frequency` already returns that Series, but Stage
    3's `predict_model` returns a frame -- `game_id`, `p_home`,
    `abstention_reason` -- because it must report abstentions alongside scores.
    Two shapes reaching one scorer means every caller writes its own reshape,
    and `align_by_game_id` rejects the frame outright (`TypeError: predictions
    must be a pandas Series indexed by game ID`). This is the single conversion.

    Abstaining rows (null `p_home`) are dropped, because they have no
    probability to score -- left in, they pass alignment and then fail inside
    `log_loss` with "probabilities must be finite", which points at the wrong
    thing entirely.

    Dropping them shrinks the population, and that must not pass unnoticed
    (`model-plan.md` section 6: "Do not compare one model's full sample against
    another's restricted sample"). It doesn't: `align_by_game_id` demands an
    identical ID set, so labels still covering an abstained game raise rather
    than silently scoring a smaller sample. Restrict the labels deliberately.
    """
    scorable = frame.loc[frame["p_home"].notna()]
    return pd.Series(scorable["p_home"].to_numpy(dtype=float),
                     index=pd.Index(scorable["game_id"]), name="p_home")


def align_by_game_id(labels: pd.Series, predictions: pd.Series) -> tuple:
    """Match labels to predictions by game ID and return `(y, p)` arrays.

    `_validate` catches a length mismatch, but two equal-length vectors
    describing *different games* score perfectly happily and produce a
    plausible wrong number -- predictions for one set of games against
    labels for another is a silent failure, not a loud one. This is the
    "duplicate/misaligned game IDs are rejected; paired comparisons use
    identical IDs" requirement in docs/next-implementations.md (Stage 0
    acceptance, restated for Stage 2); positional zipping does not satisfy
    it.

    Both arguments are Series indexed by game ID. The ID sets must be
    identical -- a missing or extra game means the caller built two
    different populations and the comparison is not the one they think it
    is. Order is irrelevant: predictions are reindexed onto the labels'
    order, so two correctly-populated inputs sorted differently still
    align. Returns plain numpy arrays, ready for the metrics above.
    """
    for name, series in (("labels", labels), ("predictions", predictions)):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series indexed by game ID")
        if series.index.hasnans:
            raise ValueError(f"{name} game IDs must be non-null")
        if series.index.has_duplicates:
            raise ValueError(f"{name} game IDs must be unique -- duplicates double-count games")

    if not labels.index.equals(predictions.index):
        missing = labels.index.difference(predictions.index)
        extra = predictions.index.difference(labels.index)
        if len(missing) or len(extra):
            raise ValueError(
                f"labels and predictions cover different games: "
                f"{len(missing)} labelled game(s) have no prediction "
                f"(e.g. {list(missing[:3])}), "
                f"{len(extra)} predicted game(s) have no label (e.g. {list(extra[:3])})"
            )
        predictions = predictions.reindex(labels.index)  # same games, different order

    return labels.to_numpy(dtype=float), predictions.to_numpy(dtype=float)


def calibration_bins(y_true, y_prob, n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    """Ten equal-width bins over [0, 1] (bin b covers `[b/n_bins,
    (b+1)/n_bins)`, except the last bin also includes exactly 1.0).
    Membership is `floor(round(p * n_bins, 9))` clipped to
    `[0, n_bins - 1]` -- the round-then-floor avoids a probability that is
    mathematically exactly on a bin edge (e.g. 0.3) landing in the wrong
    bin because of ordinary float representation error (e.g.
    0.30000000000000004). An empty bin gets `count=0` and NaN for both
    mean-predicted and observed-frequency rather than a divide-by-zero."""
    y, p = _validate(y_true, y_prob)
    if p.size:
        idx = np.clip(np.floor(np.round(p * n_bins, 9)), 0, n_bins - 1).astype(int)
    else:
        idx = np.array([], dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        rows.append({
            "bin_low": edges[b],
            "bin_high": edges[b + 1],
            "count": count,
            "mean_predicted": float(p[mask].mean()) if count else float("nan"),
            "observed_frequency": float(y[mask].mean()) if count else float("nan"),
        })
    return pd.DataFrame(rows)
