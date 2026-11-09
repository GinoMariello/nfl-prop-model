"""
Leakage-safe preprocessing + L2 logistic regression
(docs/next-implementations.md Stage 3; design: docs/stage3-design.md).

Preprocessing *learns* -- medians and scales are fitted parameters. Every
one of them is fitted inside `fit_model` on training rows only, and
`predict_model` calls `transform`, never `fit_transform`. There is no code
path here that fits on data handed to a predict call. That is the single
invariant this module exists to enforce.

Features enter only through `select_features()`'s allowlist, so no target,
score, ID, or date column can reach `X`. Preprocessing and estimator are
saved and loaded as one bundle: a scaler without its model, or a model
without its scaler, is a silent wrong answer.

`C` is configuration here. Tuning it is Stage 4's job, and nothing in this
module reads a validation score.

Run: python3 tests/test_train.py
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features.rolling_features import TEAM_FEATURE_COLUMNS, VARIANT_COLUMNS, select_features
from src.models.baselines import _validated_labels

RANDOM_SEED = 20260915  # docs/experiment-protocol.md, "Seeds"
DEFAULT_C = 1.0  # Stage 4 tunes over {0.01, 0.1, 1, 10}
SOLVER = "lbfgs"
MAX_ITER = 1000
TOL = 1e-4

MARKET_COLUMN = "market_home_probability"
SPREAD_COLUMN = "spread_line"
# The extra, already-filtered column each market variant adds on top of the
# team features. Scaled, never imputed: select_features has restricted the
# rows to those where it is usable, so a missing one here would mean the
# filter failed, not that a value needs inventing.
VARIANT_EXTRA_COLUMN = {"team_market": MARKET_COLUMN, "team_spread": SPREAD_COLUMN,
                        "market_only": MARKET_COLUMN}
# Protocol v4: the column a variant takes as a FIXED offset on the logit scale
# instead of as a fitted feature. Never scaled, never imputed, never given a
# coefficient -- see OffsetLogisticRegression.
VARIANT_OFFSET_COLUMN = {"team_residual": MARKET_COLUMN, "team_residual_plus": MARKET_COLUMN}
NO_QUOTE = "no_paired_market_quote"
NO_SPREAD = "no_closing_spread"
ABSTENTION_REASON = {"team_market": NO_QUOTE, "team_spread": NO_SPREAD, "market_only": NO_QUOTE,
                     "team_residual": NO_QUOTE, "team_residual_plus": NO_QUOTE}


class OffsetLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2 logistic regression around a fixed offset: p = sigmoid(offset + b + w.x).

    The LAST input column is the offset (already on the logit scale); every
    other column is a feature. scikit-learn's LogisticRegression has no offset
    parameter, hence this class.

    Unlike scikit-learn, the intercept is penalized along with the weights. That
    is the point of the model: as C -> 0 every coefficient shrinks to zero and
    the prediction becomes exactly sigmoid(offset) -- the market's own
    probability -- so a deviation from the market has to earn its way through
    the training data. Otherwise the objective matches scikit-learn's,
    0.5*||theta||^2 + C * sum(log loss), so C means the same thing on the same
    grid.
    """

    solver = "scipy L-BFGS-B"

    def __init__(self, C=1.0, max_iter=MAX_ITER, tol=TOL):
        self.C, self.max_iter, self.tol = C, max_iter, tol

    @staticmethod
    def _split(X):
        X = np.asarray(X, dtype=float)
        design = np.hstack([np.ones((len(X), 1)), X[:, :-1]])  # leading column = intercept
        return design, X[:, -1]

    def fit(self, X, y):
        design, offset = self._split(X)
        y = np.asarray(y, dtype=float)
        C = self.C

        def objective(theta):
            eta = offset + design @ theta
            loss = np.sum(np.logaddexp(0.0, eta) - y * eta)
            return 0.5 * theta @ theta + C * loss, theta + C * design.T @ (expit(eta) - y)

        result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "gtol": self.tol})
        if not result.success:
            warnings.warn(f"OffsetLogisticRegression did not converge: {result.message}",
                          ConvergenceWarning)
        self.intercept_ = np.array([result.x[0]])
        self.coef_ = result.x[1:][None, :]
        self.n_iter_ = np.array([result.nit])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        design, offset = self._split(X)
        p = expit(offset + design @ np.r_[self.intercept_, self.coef_.ravel()])
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _build_pipeline(variant: str, C: float) -> Pipeline:
    """Unfitted pipeline. Two scikit-learn flags here are load-bearing:

    `MissingIndicator(features="all")` rather than
    `SimpleImputer(add_indicator=True)` -- the latter emits indicators only
    for columns missing *in training*, so a column complete in training but
    missing at prediction time would be filled with the training median and
    carry no flag, and the transformed width would depend on the fold.

    `keep_empty_features=True` -- an all-missing training column would
    otherwise be dropped, again changing the width per fold. Instead it
    survives, filled with 0.0 (the documented zero fallback), with its
    indicator still marking every row so "zero because unknown" stays
    distinguishable from a real zero.
    """
    # Every allowlisted column except the variant's market/spread/offset column is
    # imputed, flagged and scaled. Derived from the allowlist, never from
    # TEAM_FEATURE_COLUMNS: ColumnTransformer silently DROPS any column no block
    # names, so a new allowlisted feature would otherwise never reach the model.
    # For every pre-v5 variant this is exactly its team columns, as before --
    # `market_only` still gets no team blocks, which is what makes it a control.
    special = {VARIANT_EXTRA_COLUMN.get(variant), VARIANT_OFFSET_COLUMN.get(variant)}
    team_columns = [c for c in VARIANT_COLUMNS[variant] if c not in special]
    blocks = []
    if team_columns:
        blocks += [
            ("team_imputed", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), team_columns),
            ("team_missing", MissingIndicator(features="all"), team_columns),
        ]
    extra = VARIANT_EXTRA_COLUMN.get(variant)
    if extra:
        # Scaled, never imputed: select_features has already restricted the
        # rows to those where this column is usable. Scaling matters because
        # L2 penalizes coefficients on whatever scale their feature uses, so
        # mixed scales mean mixed regularization strength -- and a raw spread
        # runs -19 to +27 against team features already near unit scale.
        blocks.append(("market", StandardScaler(), [extra]))
    offset = VARIANT_OFFSET_COLUMN.get(variant)
    if offset:
        # Must stay the LAST block: OffsetLogisticRegression reads the last
        # column as its offset. select_features has restricted the rows to
        # probabilities strictly inside (0, 1), so the logit is finite.
        blocks.append(("offset", FunctionTransformer(logit), [offset]))
        return Pipeline([("preprocess", ColumnTransformer(blocks)),
                         ("model", OffsetLogisticRegression(C=C))])

    # L2 is LogisticRegression's default and the only penalty this project
    # uses (docs/model-plan.md). Passing penalty="l2" explicitly would be a
    # no-op that raises a FutureWarning on scikit-learn >= 1.8 and stops
    # working in 1.10.
    return Pipeline([
        ("preprocess", ColumnTransformer(blocks)),
        ("model", LogisticRegression(C=C, solver=SOLVER, max_iter=MAX_ITER,
                                     tol=TOL, random_state=RANDOM_SEED)),
    ])


def _require_unique_index(matchups: pd.DataFrame) -> None:
    """Label alignment is `matchups.loc[X.index]`, so a repeated index label
    silently fans rows out instead of selecting them.

    `pd.concat([...])` without `ignore_index=True` produces exactly that, and
    it is how a fold is naturally assembled from several seasons -- which
    makes this Stage 4's most likely footgun. Left unguarded it surfaces as
    `IndexError: indices are out-of-bounds` from deep inside sklearn, naming
    nothing useful.
    """
    if not matchups.index.is_unique:
        repeated = matchups.index[matchups.index.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"matchups index must be unique for label alignment (repeated: {repeated}) -- "
            f"call reset_index(drop=True) after concatenating seasons or folds"
        )


def _training_rows(matchups: pd.DataFrame, variant: str) -> tuple:
    """`(X, y, n_selected)` for training: allowlisted features, labels aligned
    by the index `select_features` preserves, rows with no usable label
    dropped. `n_selected` is the row count before the label filter, so the
    caller can tell an empty fold caused by missing quotes from one caused by
    missing labels."""
    _require_unique_index(matchups)
    if "home_win" not in matchups:
        raise KeyError("matchups must carry a home_win column to fit against")
    X = select_features(matchups, variant)
    n_selected = len(X)
    # Stage 3's deliverable says "align labels by game ID". This aligns by the
    # frame index instead, which is the same alignment: select_features filters
    # rows without reindexing, so X.index still points at the rows that carry
    # those game IDs. The index is the safer key of the two here -- it is
    # guaranteed present, while game_id is only needed at predict time -- and
    # _require_unique_index above is what makes it trustworthy.
    y = matchups.loc[X.index, "home_win"]
    labelled = y.notna()
    X, y = X.loc[labelled], y.loc[labelled]
    return X, pd.Series(_validated_labels(y), index=y.index), n_selected


def fit_model(matchups: pd.DataFrame, variant: str = "team_only", C: float = DEFAULT_C) -> dict:
    """Fit preprocessing + model on TRAINING rows only, returning a bundle.

    An unfittable fold raises rather than producing a fallback model: the
    caller has to see it (docs/stage3-design.md section 5). Note the
    deliberate contrast with Stage 2's `fit_home_frequency`, which fits a
    perfectly good 0.0/1.0 on a one-class fold -- a constant baseline can
    be fitted from one class, a logistic regression cannot.
    """
    X, y, n_selected = _training_rows(matchups, variant)

    if n_selected == 0 and len(matchups):
        # Distinguish "the quote filter emptied it" from "the labels did".
        # A fold of unquoted games has perfectly good labels, and saying
        # otherwise sends the caller hunting the wrong problem.
        raise ValueError(f"cannot fit {variant}: no rows with a paired market quote")
    if len(X) == 0:
        raise ValueError(f"cannot fit {variant}: no training rows with a usable label")
    if len(X) < 2:
        raise ValueError(f"cannot fit {variant}: only {len(X)} labelled training row(s), need at least 2")
    if y.nunique() < 2:
        raise ValueError(f"cannot fit {variant}: training labels are one-class (all 0 or all 1)")

    pipeline = _build_pipeline(variant, C)
    # "Record solver, seed, tolerance, iteration limit, convergence status"
    # (docs/next-implementations.md Stage 3). NOTE: record=True captures every
    # warning fit() emits, not only ConvergenceWarning, so a deprecation from
    # scikit-learn lands here silently -- it once hid the penalty="l2" removal
    # notice. Re-run a fit without this block after any scikit-learn upgrade.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X, y.to_numpy())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    return {
        "pipeline": pipeline,
        "variant": variant,
        "feature_schema": list(X.columns),
        "config": {"C": C, "solver": pipeline.named_steps["model"].solver, "random_state": RANDOM_SEED,
                   "tol": TOL, "max_iter": MAX_ITER},
        "n_train": int(len(X)),
        "n_positive": int(y.sum()),
        "converged": bool(converged),
        "n_iter": int(np.max(pipeline.named_steps["model"].n_iter_)),
    }


def predict_model(bundle: dict, matchups: pd.DataFrame) -> pd.DataFrame:
    """One row per input game: `game_id`, `p_home`, `abstention_reason`.

    Never a filtered subset. A row a market variant cannot use -- no paired
    quote for `team_market`, no closing spread for `team_spread` -- abstains
    with the reason naming which -- dropping it would
    silently change the population the caller believes it scored, the
    coverage confusion model-plan.md section 6 warns against. Missing
    *team* features are not an abstention: they are imputed and flagged.
    """
    if "game_id" not in matchups:
        raise KeyError("matchups must carry a game_id column to label predictions")
    _require_unique_index(matchups)

    # Same discipline as Stage 2's predict_home_frequency: the output is keyed
    # by game_id, so a duplicate would double-count that game once joined to
    # labels, and a null one cannot be matched to anything at all.
    game_ids = pd.Index(matchups["game_id"])
    if game_ids.hasnans:
        raise ValueError("game_id must be non-null to label predictions")
    if game_ids.has_duplicates:
        repeated = game_ids[game_ids.duplicated()].unique()[:3].tolist()
        raise ValueError(
            f"game_id must be unique (repeated: {repeated}) -- "
            f"duplicates would double-count games when joined to labels"
        )

    variant = bundle["variant"]
    X = select_features(matchups, variant)

    schema = list(X.columns)
    if schema != bundle["feature_schema"]:
        raise ValueError(
            f"feature schema does not match the fitted model "
            f"({schema} != {bundle['feature_schema']}); "
            f"the saved bundle was fitted against a different feature contract"
        )

    out = pd.DataFrame({
        "game_id": matchups["game_id"].to_numpy(),
        "p_home": np.nan,
        "abstention_reason": None,
    }, index=matchups.index)

    if len(X):
        out.loc[X.index, "p_home"] = bundle["pipeline"].predict_proba(X)[:, 1]
    abstained = out.index.difference(X.index)
    out.loc[abstained, "abstention_reason"] = ABSTENTION_REASON.get(variant, NO_QUOTE)
    return out.reset_index(drop=True)


def save_model(bundle: dict, path, replace: bool = False) -> Path:
    """Write preprocessing and estimator as one file.

    Refuses to overwrite: docs/model-plan.md section 8 puts artifacts under
    `models_game/<target>/<run_id>/<feature_set>/<model>/`, "never overwriting
    earlier runs". Stage 4 fits one model per fold per C value, so a run_id that
    repeats -- a re-run, a crashed run restarted, a fold loop that forgets to
    vary the path -- would silently replace the earlier run's model and leave
    its metrics pointing at a file that no longer holds the model that
    produced them. Pass `replace=True` to mean it.

    An interrupted write leaves a truncated pickle, which `load_model` raises
    on -- the model is re-fittable, so recovery is a refit, not a rescue.
    """
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(
            f"a model already exists at {path}; earlier runs are never overwritten "
            f"(docs/model-plan.md section 8) -- use a distinct run_id, or pass replace=True"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path) -> dict:
    """The bundle `save_model` wrote: preprocessing and estimator together.

    A truncated file from an interrupted write raises here rather than
    loading partially, and `predict_model` rejects a bundle whose
    `feature_schema` no longer matches the code.
    """
    return joblib.load(Path(path))
