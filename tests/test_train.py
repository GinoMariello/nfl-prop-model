"""
Tests for src/models/train.py (docs/next-implementations.md Stage 3
acceptance criteria; design: docs/stage3-design.md).

The one that matters is demo_validation_values_cannot_move_fitted_parameters:
preprocessing learns medians and scales, so a leak here would be invisible
in the output and fatal to every downstream number. It is proven by
corruption injection, not by a value check -- fit, wreck the held-out rows,
and assert every fitted array is bit-identical, then show that training on
those same corrupted rows DOES move the coefficients so the assertion is
not vacuous.

Synthetic and offline; no database, no network.

Run: python3 tests/test_train.py
"""
import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.rolling_features import (
    TEAM_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, SPREAD_FEATURE_COLUMNS, QB_REST_FEATURE_COLUMNS,
)
from src.models.train import (
    fit_model, predict_model, save_model, load_model, OffsetLogisticRegression,
    RANDOM_SEED, DEFAULT_C, SOLVER, MAX_ITER, TOL, NO_QUOTE, NO_SPREAD,
)


def _matchups(n=200, seed=0, missing_columns=(), cold_start_rows=0):
    """Matchup frame with the full ordered feature set and a learnable signal.

    The label depends on the first feature, so coefficients are non-trivial
    and a changed fit is visible.
    """
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in TEAM_FEATURE_COLUMNS})
    frame.insert(0, "game_id", [f"g{i:04d}" for i in range(n)])
    signal = frame[TEAM_FEATURE_COLUMNS[0]].to_numpy()
    frame["home_win"] = (signal + rng.normal(scale=0.5, size=n) > 0).astype(float)
    frame["market_home_probability"] = np.clip(0.5 + 0.2 * signal, 0.02, 0.98)

    for column in missing_columns:
        frame[column] = np.nan
    if cold_start_rows:
        frame.loc[frame.index[:cold_start_rows], list(TEAM_FEATURE_COLUMNS)] = np.nan
    return frame


def _fitted_arrays(bundle):
    """The learned parameters that a leak would disturb."""
    pre = bundle["pipeline"].named_steps["preprocess"]
    model = bundle["pipeline"].named_steps["model"]
    team = pre.named_transformers_["team_imputed"]
    return {
        "medians": team.named_steps["impute"].statistics_.copy(),
        "scale_mean": team.named_steps["scale"].mean_.copy(),
        "scale_scale": team.named_steps["scale"].scale_.copy(),
        "coef": model.coef_.copy(),
        "intercept": model.intercept_.copy(),
    }


def demo_fit_predict_round_trip_and_schema():
    for variant, width in (("team_only", 76), ("team_market", 77)):
        bundle = fit_model(_matchups(), variant=variant)

        assert bundle["variant"] == variant
        expected = list(TEAM_FEATURE_COLUMNS if variant == "team_only" else MARKET_FEATURE_COLUMNS)
        assert bundle["feature_schema"] == expected
        assert bundle["config"] == {"C": DEFAULT_C, "solver": SOLVER,
                                    "random_state": RANDOM_SEED, "tol": TOL,
                                    "max_iter": MAX_ITER}
        assert bundle["converged"] is True and bundle["n_iter"] >= 1
        assert bundle["n_train"] == 200 and 0 < bundle["n_positive"] < 200

        # 76 = imputed + indicator for each team feature; 77 adds the quote.
        X = _matchups()[expected]
        assert bundle["pipeline"].named_steps["preprocess"].transform(X).shape[1] == width

        out = predict_model(bundle, _matchups(seed=1))
        assert len(out) == 200, "one output row per input game"
        assert list(out.columns) == ["game_id", "p_home", "abstention_reason"]
        assert out.p_home.between(0, 1, inclusive="neither").all()

    print("OK -- both variants fit, emit a 76/77-wide design matrix, record their "
          "config, and predict one row per input game.")


def demo_no_target_id_or_date_column_can_enter_X():
    frame = _matchups()
    for leaky in ("home_score", "away_score", "point_diff", "gameday", "season", "week"):
        frame[leaky] = 1.0
    bundle = fit_model(frame, variant="team_market")

    banned = {"home_score", "away_score", "point_diff", "home_win", "game_id",
              "gameday", "season", "week"}
    assert not (banned & set(bundle["feature_schema"])), \
        "a target/ID/date column reached X"
    assert bundle["feature_schema"] == list(MARKET_FEATURE_COLUMNS), \
        "the fitted schema must be exactly the variant's allowlist"

    print("OK -- target, score, ID, date and unrelated columns are all kept out of X "
          "by select_features' allowlist.")


def demo_validation_values_cannot_move_fitted_parameters():
    train = _matchups(n=200, seed=0)
    holdout = _matchups(n=80, seed=7)

    bundle = fit_model(train, variant="team_market")
    before = _fitted_arrays(bundle)

    # Wreck the held-out rows, then use them for prediction only.
    corrupted = holdout.copy()
    corrupted[list(TEAM_FEATURE_COLUMNS)] = 1e9
    corrupted["home_win"] = 1.0 - corrupted["home_win"]
    predict_model(bundle, corrupted)

    after = _fitted_arrays(bundle)
    for name in before:
        assert np.array_equal(before[name], after[name]), \
            f"{name} changed after predicting on corrupted held-out rows"

    # Not vacuous: training ON those rows does move the coefficients.
    polluted = fit_model(pd.concat([train, corrupted], ignore_index=True), variant="team_market")
    moved = _fitted_arrays(polluted)
    assert not np.array_equal(before["coef"], moved["coef"]), \
        "corrupted rows must change the fit when they are actually trained on"
    assert not np.array_equal(before["medians"], moved["medians"])

    # model-plan.md section 4: "Refit every learned preprocessing step with each
    # fit/refit"; section 5: a fitted model is "held fixed" through its
    # evaluation season. Both fail the moment a pipeline is built once and
    # reused, so the second fit must leave the first bundle untouched -- which
    # the assertions above cannot see, having copied their arrays beforehand.
    still = _fitted_arrays(bundle)
    for name in before:
        assert np.array_equal(before[name], still[name]), \
            f"{name} changed when a SECOND model was fitted -- the pipeline is being reused"

    print("OK -- validation/test values cannot move training-fold medians, scales or "
          "coefficients, and the same rows demonstrably do move them when trained on.")


def demo_missingness_cases():
    # Cold starts: a team's first games have no history at all.
    cold = _matchups(n=120, cold_start_rows=10)
    bundle = fit_model(cold, variant="team_only")
    out = predict_model(bundle, cold)
    assert out.p_home.notna().all(), "cold-start rows are imputed and scored, never dropped"
    assert out.abstention_reason.isna().all(), "a missing team feature is not an abstention"

    # A column all-missing in training: documented 0.0 fallback, schema width
    # unchanged, indicator still firing.
    dead_column = TEAM_FEATURE_COLUMNS[3]
    frame = _matchups(n=120, missing_columns=[dead_column])
    bundle = fit_model(frame, variant="team_only")
    medians = _fitted_arrays(bundle)["medians"]
    position = list(TEAM_FEATURE_COLUMNS).index(dead_column)
    assert medians[position] == 0.0, "an all-missing training column falls back to 0.0"
    transformed = bundle["pipeline"].named_steps["preprocess"].transform(frame[list(TEAM_FEATURE_COLUMNS)])
    assert transformed.shape[1] == 76, "width must not depend on the fold"
    assert transformed[:, len(TEAM_FEATURE_COLUMNS) + position].all(), \
        "the indicator must mark every row of an all-missing column"

    # Missingness that appears only at prediction time -- the case
    # SimpleImputer(add_indicator=True) would silently miss.
    train = _matchups(n=120, seed=2)
    bundle = fit_model(train, variant="team_only")
    late = _matchups(n=20, seed=3)
    surprise = TEAM_FEATURE_COLUMNS[5]
    late.loc[late.index[:5], surprise] = np.nan
    transformed = bundle["pipeline"].named_steps["preprocess"].transform(late[list(TEAM_FEATURE_COLUMNS)])
    indicator = transformed[:, len(TEAM_FEATURE_COLUMNS) + list(TEAM_FEATURE_COLUMNS).index(surprise)]
    assert indicator[:5].all() and not indicator[5:].any(), \
        "test-time-only missingness must still raise its fixed indicator"
    assert transformed.shape[1] == 76
    assert predict_model(bundle, late).p_home.notna().all()

    print("OK -- cold starts are imputed and scored, an all-missing training column falls "
          "back to 0.0 with its indicator, and test-time-only missingness is flagged.")


def demo_team_market_abstains_without_a_paired_quote():
    frame = _matchups(n=100)
    # Three ways a quote fails to be usable; select_features excludes all.
    frame.loc[0, "market_home_probability"] = np.nan
    frame.loc[1, "market_home_probability"] = 0.0
    frame.loc[2, "market_home_probability"] = 1.0
    unquoted = set(frame.loc[[0, 1, 2], "game_id"])

    bundle = fit_model(frame, variant="team_market")
    assert bundle["n_train"] == 97, "unquoted rows must not be fitted on"

    out = predict_model(bundle, frame)
    assert len(out) == 100, "abstaining rows still appear in the output"
    abstained = out[out.abstention_reason.notna()]
    assert set(abstained.game_id) == unquoted
    assert (abstained.abstention_reason == NO_QUOTE).all()
    assert abstained.p_home.isna().all(), "an abstention must not carry a probability"
    assert out[out.abstention_reason.isna()].p_home.notna().all()

    # The market column is never imputed -- team_only scores those same rows,
    # because it needs no quote at all.
    team_only = fit_model(frame, variant="team_only")
    assert team_only["n_train"] == 100
    assert predict_model(team_only, frame).p_home.notna().all()

    print("OK -- team_market excludes unquoted rows from fitting and abstains on them with "
          "a reason, while team_only scores the same rows normally.")


def demo_insufficient_data_fails_loudly():
    frame = _matchups(n=10)

    empty = frame.iloc[0:0]
    try:
        fit_model(empty, variant="team_only")
        assert False, "expected ValueError for an empty training set"
    except ValueError as exc:
        assert "no training rows" in str(exc)

    unlabelled = frame.copy()
    unlabelled["home_win"] = np.nan
    try:
        fit_model(unlabelled, variant="team_only")
        assert False, "expected ValueError when every label is missing"
    except ValueError as exc:
        assert "no training rows" in str(exc)

    try:
        fit_model(frame.iloc[:1], variant="team_only")
        assert False, "expected ValueError for a single-row training set"
    except ValueError as exc:
        assert "at least 2" in str(exc)

    one_class = frame.copy()
    one_class["home_win"] = 1.0
    try:
        fit_model(one_class, variant="team_only")
        assert False, "expected ValueError for one-class labels"
    except ValueError as exc:
        assert "one-class" in str(exc)

    print("OK -- empty, all-unlabelled, single-row and one-class training sets each fail "
          "loudly instead of producing a fallback model.")


def demo_malformed_frames_fail_with_a_useful_message():
    """Guards aimed squarely at Stage 4, which assembles folds from several
    seasons and keys every comparison on game_id."""
    bundle = fit_model(_matchups(n=100), variant="team_only")

    # pd.concat without ignore_index is the natural way to build a multi-season
    # fold, and it repeats index labels. Label alignment is matchups.loc[X.index],
    # so that silently fans rows out; unguarded it surfaced as
    # "IndexError: indices are out-of-bounds" from inside sklearn.
    duplicated_index = pd.concat([_matchups(n=50, seed=1), _matchups(n=50, seed=2)])
    assert not duplicated_index.index.is_unique, "fixture must actually repeat index labels"
    for call, what in ((lambda: fit_model(duplicated_index, variant="team_only"), "fit"),
                       (lambda: predict_model(bundle, duplicated_index), "predict")):
        try:
            call()
            assert False, f"expected ValueError on {what} with a repeated index"
        except ValueError as exc:
            assert "index must be unique" in str(exc) and "reset_index" in str(exc), \
                f"{what}: message must name the fix, got {exc!r}"

    # Duplicate and null game_id, matching Stage 2's predict_home_frequency.
    repeated_ids = pd.concat([_matchups(n=3), _matchups(n=3)], ignore_index=True)
    try:
        predict_model(bundle, repeated_ids)
        assert False, "expected ValueError for duplicate game_id"
    except ValueError as exc:
        assert "game_id must be unique" in str(exc)

    null_ids = _matchups(n=3)
    null_ids.loc[0, "game_id"] = None
    try:
        predict_model(bundle, null_ids)
        assert False, "expected ValueError for a null game_id"
    except ValueError as exc:
        assert "game_id must be non-null" in str(exc)

    # An empty frame is not malformed -- it yields an empty prediction frame.
    empty = predict_model(bundle, _matchups(n=10).iloc[0:0])
    assert len(empty) == 0 and list(empty.columns) == ["game_id", "p_home", "abstention_reason"]

    # A team_market fold with no quotes at all says so, rather than blaming
    # the labels, which are perfectly fine.
    unquoted = _matchups(n=50)
    unquoted["market_home_probability"] = np.nan
    try:
        fit_model(unquoted, variant="team_market")
        assert False, "expected ValueError for a fold with no paired quotes"
    except ValueError as exc:
        assert "no rows with a paired market quote" in str(exc)

    print("OK -- repeated index labels, duplicate/null game IDs and a quote-less "
          "team_market fold each fail with a message naming the real problem.")


def demo_team_spread_uses_the_raw_spread_and_never_imputes_it():
    """Protocol v2. The closing spread is fed raw, not converted to a win
    probability: a conversion needs an assumed margin distribution whose
    standard deviation is itself estimated from NFL history, including the
    seasons being scored. As a coefficient it is fitted per fold instead."""
    frame = _matchups(n=200)
    frame["spread_line"] = np.round(6.0 * frame[TEAM_FEATURE_COLUMNS[0]], 1)

    bundle = fit_model(frame, variant="team_spread")
    assert bundle["feature_schema"] == list(SPREAD_FEATURE_COLUMNS)
    assert bundle["feature_schema"][-1] == "spread_line"
    assert bundle["n_train"] == 200, "a 100%-covered spread must not drop rows"

    width = bundle["pipeline"].named_steps["preprocess"].transform(
        frame[list(SPREAD_FEATURE_COLUMNS)]).shape[1]
    assert width == 77, f"38 imputed + 38 indicators + 1 spread, got {width}"

    # The spread block scales and does nothing else -- no imputer to invent a
    # spread for a row select_features already excluded.
    blocks = dict((name, step) for name, step, _ in
                  bundle["pipeline"].named_steps["preprocess"].transformers)
    assert type(blocks["market"]).__name__ == "StandardScaler"

    # A missing spread abstains, with a reason naming the spread rather than
    # the moneyline.
    gapped = frame.copy()
    gapped.loc[gapped.index[:5], "spread_line"] = np.nan
    fitted = fit_model(gapped, variant="team_spread")
    assert fitted["n_train"] == 195, "rows with no spread must not be fitted on"
    out = predict_model(fitted, gapped)
    abstained = out[out.abstention_reason.notna()]
    assert len(abstained) == 5 and (abstained.abstention_reason == NO_SPREAD).all()
    assert abstained.p_home.isna().all()

    # The spread carries real signal: it must beat team_only on the same rows.
    from src.evaluation.metrics import align_by_game_id, log_loss, predictions_for_scoring
    labels = frame.set_index("game_id")["home_win"]
    scores = {}
    for variant in ("team_only", "team_spread"):
        p = predictions_for_scoring(predict_model(fit_model(frame, variant=variant), frame))
        y, prob = align_by_game_id(labels.loc[p.index], p)
        scores[variant] = log_loss(y, prob)
    assert scores["team_spread"] < scores["team_only"], \
        f"the spread must add information: {scores}"

    print("OK -- team_spread feeds the raw spread, emits 77 columns, never imputes it, "
          "abstains with its own reason, and beats team_only on the same rows.")


def demo_training_cannot_read_a_score():
    """model-plan.md section 5: "Test outcomes cannot select parameters", and
    next-implementations.md Stage 3: "Expose C as configuration; tuning is
    Stage 4's job, and nothing in this module reads a validation score."

    Stage 4 is where that gets tempting -- picking C inside the fit, using a
    metric imported right here, is a two-line change that reads as a tidy
    refactor and quietly makes the selection score its own training data. The
    invariant is structural, so check it structurally: `train.py` imports
    nothing from the scoring layer. C arrives as an argument or not at all.
    """
    source = Path(__file__).resolve().parents[1] / "src" / "models" / "train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    assert imported, "no imports found -- is this parsing the right file?"

    scoring = sorted(m for m in imported if "evaluation" in m or "metrics" in m)
    assert not scoring, (
        f"train.py imports the scoring layer ({scoring}). Tuning belongs to Stage 4's "
        f"fold engine, which passes C in; a metric reachable from inside fit_model is "
        f"how test outcomes start selecting parameters."
    )

    # C really is just an argument: two values, two different fits.
    frame = _matchups(n=120)
    loose = fit_model(frame, variant="team_only", C=10.0)
    tight = fit_model(frame, variant="team_only", C=0.01)
    assert loose["config"]["C"] == 10.0 and tight["config"]["C"] == 0.01
    assert not np.array_equal(loose["pipeline"].named_steps["model"].coef_,
                              tight["pipeline"].named_steps["model"].coef_), \
        "C must actually reach the estimator"
    assert np.abs(tight["pipeline"].named_steps["model"].coef_).sum() < \
           np.abs(loose["pipeline"].named_steps["model"].coef_).sum(), \
        "smaller C must shrink the coefficients -- otherwise the penalty is not wired up"

    print("OK -- train.py cannot reach a score, and C is configuration that visibly "
          "changes the fit.")


def demo_save_load_prediction_equivalence():
    train = _matchups(n=150)
    score = _matchups(n=40, seed=9)
    bundle = fit_model(train, variant="team_market")
    expected = predict_model(bundle, score)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_model(bundle, Path(tmp) / "nested" / "model.joblib")
        assert path.is_file(), "save_model must create parent directories"
        reloaded = load_model(path)

        pd.testing.assert_frame_equal(expected, predict_model(reloaded, score))
        assert reloaded["config"] == bundle["config"]

        # model-plan.md section 8: artifacts "never overwriting earlier runs".
        # Stage 4 fits one model per fold per C, so a repeated run_id must not
        # quietly replace an earlier run's model and orphan its metrics.
        second = fit_model(train, variant="team_only")
        try:
            save_model(second, path)
            assert False, "expected FileExistsError rather than a silent overwrite"
        except FileExistsError as exc:
            assert "never overwritten" in str(exc) and "replace=True" in str(exc)
        assert load_model(path)["variant"] == "team_market", "the earlier run must survive"

        save_model(second, path, replace=True)
        assert load_model(path)["variant"] == "team_only", "replace=True must mean it"
        save_model(bundle, path, replace=True)  # restore for the checks below

        # A bundle whose feature contract no longer matches the code is
        # rejected rather than silently mapping coefficients onto the wrong
        # features.
        stale = load_model(path)
        stale["feature_schema"] = ["renamed_feature"] + stale["feature_schema"][1:]
        try:
            predict_model(stale, score)
            assert False, "expected ValueError for a mismatched feature schema"
        except ValueError as exc:
            assert "feature schema does not match" in str(exc)

    print("OK -- a saved model reloads to bit-identical predictions, and a stale feature "
          "schema is rejected instead of silently misaligning coefficients.")


def demo_model_and_baseline_score_through_one_path():
    """Cross-stage seam: Stage 2's baseline and Stage 3's model must reach the
    metrics the same way, so Stage 4 does not need a reshape per model type."""
    from src.models.baselines import fit_home_frequency, predict_home_frequency
    from src.evaluation.metrics import align_by_game_id, predictions_for_scoring, log_loss

    frame = _matchups(n=60)
    frame.loc[0, "market_home_probability"] = np.nan  # one team_market abstention
    labels = frame.set_index("game_id")["home_win"]

    # Stage 2: already a game-ID-indexed Series.
    fitted = fit_home_frequency(frame["home_win"])
    baseline = predict_home_frequency(fitted, frame["game_id"])
    y_b, p_b = align_by_game_id(labels, baseline)
    assert np.isfinite(log_loss(y_b, p_b))

    # Stage 3: a frame, converted by the one shared helper.
    bundle = fit_model(frame, variant="team_market")
    model = predictions_for_scoring(predict_model(bundle, frame))
    assert len(model) == 59, "the abstaining row carries no probability to score"

    # The shrunken population cannot slip past: labels must be restricted to
    # match, deliberately, before anything scores.
    try:
        align_by_game_id(labels, model)
        assert False, "expected the coverage difference to be caught"
    except ValueError as exc:
        assert "different games" in str(exc)

    y_m, p_m = align_by_game_id(labels.loc[model.index], model)
    assert np.isfinite(log_loss(y_m, p_m))

    # And the matched comparison model-plan.md section 6 actually asks for:
    # both scored on identical IDs.
    y_pair, p_pair = align_by_game_id(labels.loc[model.index], baseline.loc[model.index])
    assert np.isfinite(log_loss(y_pair, p_pair))
    assert len(y_pair) == len(y_m) == 59

    print("OK -- baseline and model both reach the metrics through align_by_game_id, "
          "and a matched-ID comparison scores on identical populations.")


def demo_market_only_differs_from_team_market_by_the_team_features_alone():
    """market_only is the control for team_market, so it must match it in every
    respect except the team features: the same rows (the paired-quote filter),
    the same market column preprocessed the same way, the same abstention.
    Any other difference would leak into `team_market - market_only` and be
    misread as what the team statistics add.
    """
    frame = _matchups(n=200)
    frame.loc[frame.index[:15], "market_home_probability"] = np.nan

    control = fit_model(frame, variant="market_only")
    full = fit_model(frame, variant="team_market")

    assert control["feature_schema"] == ["market_home_probability"], \
        "nothing but the market probability may enter the control"
    assert control["n_train"] == full["n_train"] == 185, "both must fit on exactly the paired rows"

    pre_control = control["pipeline"].named_steps["preprocess"]
    pre_full = full["pipeline"].named_steps["preprocess"]
    assert [name for name, _, _ in pre_control.transformers] == ["market"], \
        "the control must carry no team imputation or missingness blocks"
    scale_control = pre_control.named_transformers_["market"]
    scale_full = pre_full.named_transformers_["market"]
    assert np.array_equal(scale_control.mean_, scale_full.mean_) \
        and np.array_equal(scale_control.scale_, scale_full.scale_), \
        "the market column must be preprocessed identically in both"

    predictions = predict_model(control, frame)
    assert len(predictions) == len(frame), "an unquoted row abstains, it is not dropped"
    abstained = predictions.loc[predictions["p_home"].isna()]
    assert len(abstained) == 15 and (abstained["abstention_reason"] == NO_QUOTE).all()

    # Non-vacuous: sharing everything but the team features, the two are still
    # different models -- otherwise the comparison measures nothing.
    assert not np.allclose(predictions["p_home"].dropna(),
                           predict_model(full, frame)["p_home"].dropna())
    print("OK -- market_only shares team_market's rows, market preprocessing and abstention, "
          "and differs only by the team features.")


def demo_residual_model_starts_from_the_market():
    """team_residual: p = sigmoid(logit(p_market) + b + w.x), every coefficient
    penalized. Three properties carry the design, each checked directly.

    1. The optimizer is right: with a zero offset it is plain L2 logistic
       regression, so it must match scikit-learn's own fit of the same objective
       (penalized intercept == fit_intercept=False plus a constant column).
    2. It defaults to the market: as C -> 0 it must return the market's own
       probability on every row, not merely something close to it.
    3. It is not inert: with real regularization it does move away from the
       market, so property 2 is not a model that ignores its features.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(1)
    Z = rng.normal(size=(400, 5))
    y = (Z[:, 0] + rng.normal(size=400) > 0).astype(int)
    for C in (0.01, 1.0, 10.0):
        mine = OffsetLogisticRegression(C=C, tol=1e-10).fit(np.column_stack([Z, np.zeros(400)]), y)
        ref = LogisticRegression(C=C, fit_intercept=False, tol=1e-10, max_iter=10000) \
            .fit(np.column_stack([np.ones(400), Z]), y)
        np.testing.assert_allclose(np.r_[mine.intercept_, mine.coef_.ravel()], ref.coef_.ravel(),
                                   atol=1e-4, err_msg=f"optimizer disagrees with scikit-learn at C={C}")

    frame = _matchups(n=200)
    frame.loc[frame.index[:15], "market_home_probability"] = np.nan

    shrunk = fit_model(frame, variant="team_residual", C=1e-9)
    assert shrunk["feature_schema"] == fit_model(frame, variant="team_market")["feature_schema"], \
        "team_residual must take exactly team_market's columns"
    assert shrunk["n_train"] == 185, "paired-quote rows only, like team_market"
    predictions = predict_model(shrunk, frame)
    quoted = predictions["p_home"].notna()
    np.testing.assert_allclose(predictions.loc[quoted, "p_home"],
                               frame.loc[quoted, "market_home_probability"], atol=1e-9,
                               err_msg="fully regularized, the model must return the market exactly")
    assert (predictions.loc[~quoted, "abstention_reason"] == NO_QUOTE).all() and (~quoted).sum() == 15

    loose = predict_model(fit_model(frame, variant="team_residual", C=10.0), frame)
    assert np.abs(loose.loc[quoted, "p_home"] - frame.loc[quoted, "market_home_probability"]).max() > 0.01, \
        "with real signal and little regularization, the model must move off the market"

    with tempfile.TemporaryDirectory() as tmp:
        bundle = fit_model(frame, variant="team_residual", C=1.0)
        path = save_model(bundle, Path(tmp) / "residual.joblib")
        np.testing.assert_array_equal(predict_model(load_model(path), frame)["p_home"],
                                      predict_model(bundle, frame)["p_home"])
    print("OK -- the residual model matches scikit-learn's objective, returns the market "
          "exactly when fully regularized, moves off it when allowed, and survives save/load.")


def demo_residual_plus_feeds_every_new_column_to_the_model():
    """team_residual_plus must actually USE the QB/rest columns. ColumnTransformer
    drops any column no block names, without a warning, so a pipeline built only
    from TEAM_FEATURE_COLUMNS would accept the new allowlist, report the right
    schema, and fit a model that never saw a single new value."""
    frame = _matchups(n=300)
    rng = np.random.RandomState(7)
    for column in QB_REST_FEATURE_COLUMNS:
        frame[column] = rng.normal(size=len(frame))
    frame.loc[frame.index[:5], "home_qb_changed"] = np.nan  # cold starts get imputed and flagged

    plus = fit_model(frame, variant="team_residual_plus", C=10.0)
    base = fit_model(frame, variant="team_residual", C=10.0)
    assert plus["feature_schema"] == [*MARKET_FEATURE_COLUMNS, *QB_REST_FEATURE_COLUMNS]
    n_team, n_new = len(TEAM_FEATURE_COLUMNS), len(QB_REST_FEATURE_COLUMNS)
    assert base["pipeline"].named_steps["model"].coef_.shape[1] == 2 * n_team
    assert plus["pipeline"].named_steps["model"].coef_.shape[1] == 2 * (n_team + n_new), \
        "each new column needs an imputed value and a missingness flag in the model"

    # Moving only a new column moves the prediction: it reaches the model.
    moved = frame.assign(rest_diff=frame["rest_diff"] + 5.0)
    assert not np.allclose(predict_model(plus, moved)["p_home"], predict_model(plus, frame)["p_home"]), \
        "rest_diff changed and the prediction did not -- the column never reached the model"

    # And it still starts from the market: fully regularized, it returns the market.
    shrunk = predict_model(fit_model(frame, variant="team_residual_plus", C=1e-9), frame)
    np.testing.assert_allclose(shrunk["p_home"], frame["market_home_probability"], atol=1e-9)
    print("OK -- team_residual_plus feeds every new column to the model, imputed and flagged, "
          "and still returns the market when fully regularized.")


if __name__ == "__main__":
    demo_fit_predict_round_trip_and_schema()
    demo_no_target_id_or_date_column_can_enter_X()
    demo_validation_values_cannot_move_fitted_parameters()
    demo_missingness_cases()
    demo_team_market_abstains_without_a_paired_quote()
    demo_insufficient_data_fails_loudly()
    demo_malformed_frames_fail_with_a_useful_message()
    demo_team_spread_uses_the_raw_spread_and_never_imputes_it()
    demo_training_cannot_read_a_score()
    demo_save_load_prediction_equivalence()
    demo_model_and_baseline_score_through_one_path()
    demo_market_only_differs_from_team_market_by_the_team_features_alone()
    demo_residual_model_starts_from_the_market()
    demo_residual_plus_feeds_every_new_column_to_the_model()
