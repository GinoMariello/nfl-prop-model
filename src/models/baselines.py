"""
Baselines for docs/next-implementations.md (design: docs/stage2-design.md).

Stage 0's always-home is fitting-free by construction (it never looks at
the data beyond the label column) and is scored as accuracy only, never as
a probability -- model-plan.md B7: a constant `P(home_win)=1` has unbounded
log loss the moment it's wrong once.

Stage 2's home-frequency IS a probability, so it is scored with log loss
like any model -- which is exactly why it must be fitted on training-fold
rows only and then held fixed. Fit with `fit_home_frequency`, apply with
`predict_home_frequency`; there is no path that refits at scoring time.

Neither function filters its input. Both validate it (`_validated_labels`)
and reject anything that isn't an eligible 0/1 label.

Run: python3 tests/test_baselines.py
"""
import numpy as np
import pandas as pd


def _validated_labels(home_win) -> np.ndarray:
    """Every baseline's label gate: non-null, strictly 0/1, any dtype.

    The null check is `pd.isna`, not `.astype(bool)`. `.astype(bool)` maps
    `np.nan` to True, and `home_win` comes back from SQLite as float64
    rather than Int64 -- so its nulls are `np.nan`, and an unfiltered
    column silently scored every unplayed or tied game as a home win
    (measured: n=5000/0.5726 against a true n=4991/0.5718 on 5000 real
    matchup rows). The contract has always been "caller supplies eligible
    rows"; this enforces it instead of assuming it.
    """
    values = pd.Series(home_win)
    if pd.isna(values).any():
        raise ValueError("home_win must not contain nulls -- filter to eligible, non-tied games first")
    try:
        y = values.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("home_win must contain only 0/1 (or boolean) values") from exc
    if y.size and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("home_win must contain only 0/1 (or boolean) values")
    return y


def always_home_accuracy(home_win: pd.Series) -> dict:
    """`home_win`: boolean-like Series already restricted to eligible,
    non-null labels (e.g. `matchup["home_win"].notna()`) -- this function
    doesn't filter anything itself, it rejects (see `_validated_labels`).
    Always predicting the home team wins, so accuracy is just the observed
    home-win rate. Returns `{"accuracy": float, "n": int}`; an empty input
    returns `{"accuracy": nan, "n": 0}` rather than raising."""
    y = _validated_labels(home_win)
    n = len(y)
    if n == 0:
        return {"accuracy": float("nan"), "n": 0}
    return {"accuracy": float(y.mean()), "n": n}


def fit_home_frequency(home_win: pd.Series) -> dict:
    """Fit the constant home-win probability on TRAINING-FOLD rows only
    (model-plan.md section 6: "training-fold non-tie home-win rate as a
    constant probability, fit per fold").

    `home_win`: eligible, non-tied, non-null training labels -- the caller
    selects the fold, this function only validates what it is handed.
    Returns `{"p_home": float, "n": int}`.

    The return is plain scalars in a fresh dict, holding no reference to
    the input. That is what makes "changing validation/test labels cannot
    change an already-fitted baseline" a property of the type rather than
    a promise in a docstring -- there is nothing here for a later mutation
    to reach.

    Unlike `always_home_accuracy`, an empty input raises. A descriptive
    statistic may answer "no games, no rate"; a fitted parameter with no
    training data is not a model, and a quiet NaN would propagate into
    predictions and scores as a silent hole.

    A one-class fold fits exactly 0.0 or 1.0. Nothing is smoothed -- see
    docs/stage2-design.md section 5.
    """
    y = _validated_labels(home_win)
    if y.size == 0:
        raise ValueError("cannot fit home-frequency on an empty training sample")
    return {"p_home": float(y.mean()), "n": int(y.size)}


def predict_home_frequency(fitted: dict, game_ids) -> pd.Series:
    """The fitted constant, one row per game ID, indexed by game ID.

    `fitted`: the dict from `fit_home_frequency`. Taking it whole rather
    than a bare float keeps fit and predict visibly paired, and makes it
    awkward to pass a label column where a fitted value belongs.

    Scoring a subset never refits (next-implementations.md Stage 4): fit
    once per fold, then call this once per scoring population -- full
    eligible coverage, and again on the paired-quote subset for the market
    comparison. There is no code path here that refits.

    Game IDs must be unique and non-null. A duplicate would fan out in a
    later join against labels and double-count that game in the score,
    which is why this is rejected before any comparison rather than after.
    An empty `game_ids` is allowed: a fold can legitimately have an empty
    scoring subset (no paired quotes, say), and the metrics already return
    NaN for an empty sample.
    """
    index = pd.Index(game_ids)
    if index.has_duplicates:
        raise ValueError("game_ids must be unique -- duplicates would double-count games when joined to labels")
    if index.hasnans:
        raise ValueError("game_ids must be non-null")
    return pd.Series(fitted["p_home"], index=index, dtype=float, name="p_home")
