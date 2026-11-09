"""
Tests for src/evaluation/backtest.py (docs/next-implementations.md Stage 4
acceptance criteria; design: docs/stage4-design.md).

The one that matters is demo_outer_test_outcomes_cannot_reach_selection_or_fit:
an outer-test outcome influencing the selected C or a fitted parameter is
invisible in the output and invalidates every number downstream. It is proven
by corruption injection -- wreck the test season, assert nothing moved, then
wreck the validation and training rows and assert things DO move, so the first
assertion is not vacuous.

Synthetic and offline. The real-data window check needs a snapshot:

Run: python3 tests/test_backtest.py
     python3 tests/test_backtest.py --snapshot snapshots/<id>
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.backtest import (
    C_GRID, FULL_COVERAGE_MODELS, MATCHED_MODELS, OUTER_SEASONS, TRAIN_START, VARIANT_C_GRID,
    comparison_seasons, load_matchups, make_folds, run_backtest, save_backtest,
    score_predictions, select_c, _assert_partitions_ordered, _season_slice, _eligible,
)
from src.evaluation.diagnose import _sha256_file
from src.features.rolling_features import TEAM_FEATURE_COLUMNS
from src.models.baselines import fit_home_frequency


def _matchups(seasons=range(1999, 2009), per_season=24, seed=0, quotes_from=None,
              neutral_every=None):
    """Matchup frame spanning several seasons, with a learnable signal.

    `quotes_from` mimics the real snapshot: no paired market quote before that
    season. Dates increase strictly within and across seasons, so a season
    partition is also a chronological one.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for season in seasons:
        for i in range(per_season):
            rows.append({"game_id": f"{season}_{i:03d}", "home_season": season,
                         "home_week": i + 1,
                         "home_gameday": f"{season}-09-{1 + i % 28:02d}"
                                         if i < 28 else f"{season}-11-{1 + (i - 28) % 28:02d}"})
    frame = pd.DataFrame(rows)
    for column in TEAM_FEATURE_COLUMNS:
        frame[column] = rng.normal(size=len(frame))
    signal = frame[TEAM_FEATURE_COLUMNS[0]].to_numpy()
    frame["home_win"] = (signal + rng.normal(scale=0.5, size=len(frame)) > 0).astype(float)
    quote = np.clip(0.5 + 0.2 * signal, 0.02, 0.98)
    if quotes_from is not None:
        quote = np.where(frame["home_season"] >= quotes_from, quote, np.nan)
    frame["market_home_probability"] = quote
    # Present for every row, as it is in the real snapshot back to 1999, so
    # team_spread is full-coverage and never abstains.
    frame["spread_line"] = np.round(6.0 * signal, 1)
    frame["neutral_site"] = pd.Series(
        [bool(neutral_every and i % neutral_every == 0) for i in range(len(frame))],
        dtype="boolean")
    # Protocol v5 inputs, from their own generator so every draw above is unchanged.
    extra = np.random.RandomState(seed + 1)
    for side in ("home", "away"):
        frame[f"{side}_qb_changed"] = (extra.uniform(size=len(frame)) < 0.12).astype(float)
        frame[f"{side}_qb_prior_starts"] = extra.randint(0, 200, size=len(frame)).astype(float)
    frame["rest_diff"] = extra.randint(-3, 4, size=len(frame)).astype(float)
    return frame


def _fitted_arrays(bundle):
    pre = bundle["pipeline"].named_steps["preprocess"]
    team = pre.named_transformers_["team_imputed"]
    return {"medians": team.named_steps["impute"].statistics_.copy(),
            "scale_mean": team.named_steps["scale"].mean_.copy(),
            "coef": bundle["pipeline"].named_steps["model"].coef_.copy()}


def demo_folds_partition_exactly_and_in_order():
    frame = _matchups(seasons=range(1999, 2008))
    folds = make_folds(frame, outer_seasons=(2005, 2006, 2007))
    assert [f["outer_season"] for f in folds] == [2005, 2006, 2007]

    eligible = _eligible(frame)
    for fold in folds:
        season = fold["outer_season"]
        assert fold["train_seasons"] == (TRAIN_START, season - 2)
        assert fold["valid_season"] == season - 1 and fold["test_season"] == season

        train, valid, test = (set(fold[f"{p}_ids"]) for p in ("train", "valid", "test"))
        assert not (train & valid) and not (train & test) and not (valid & test)
        expected = set(eligible.loc[eligible["home_season"].le(season), "game_id"])
        assert train | valid | test == expected, "partitions must exhaust 1999..T"

        assert fold["train_dates"][1] < fold["valid_dates"][0] < fold["valid_dates"][1] \
            < fold["test_dates"][0], "partitions must be strictly ordered by date"

    # Non-vacuous: a training game played during the test season must be caught
    # by the DATE check, even though its season number still sorts correctly.
    late = frame.copy()
    late.loc[late["game_id"].eq("2003_000"), "home_gameday"] = "2005-10-01"
    try:
        make_folds(late, outer_seasons=(2005,))
        assert False, "expected ValueError for a chronologically out-of-order partition"
    except ValueError as exc:
        assert "not strictly chronologically ordered" in str(exc)

    # ...but a reschedule inside a season is legitimate and must not fire.
    moved = frame.copy()
    moved.loc[moved["game_id"].eq("2003_000"), "home_gameday"] = "2003-12-20"
    make_folds(moved, outer_seasons=(2005,))

    # Overlap is caught even when the dates are fine.
    fold = dict(folds[0])
    fold["valid_ids"] = fold["valid_ids"] + (fold["train_ids"][0],)
    try:
        _assert_partitions_ordered(fold)
        assert False, "expected ValueError for overlapping partitions"
    except ValueError as exc:
        assert "share game(s)" in str(exc)

    print("OK -- folds are disjoint, exhaust 1999..T, and are strictly ordered by real "
          "dates; an out-of-order game is caught and an in-season reschedule is not.")


def demo_outer_test_outcomes_cannot_reach_selection_or_fit():
    """model-plan.md section 9 gate 4."""
    frame = _matchups(seasons=range(1999, 2008))
    eligible = _eligible(frame)
    train = _season_slice(eligible, TRAIN_START, 2005)
    valid = _season_slice(eligible, 2006, 2006)

    picked = select_c(train, valid, "team_only")
    before_scores = [c["valid_log_loss"] for c in picked["candidates"]]
    from src.models.train import fit_model
    before = _fitted_arrays(fit_model(train, variant="team_only", C=picked["C"]))

    # Wreck the outer-test season beyond recognition.
    corrupted = frame.copy()
    test_rows = corrupted["home_season"].eq(2007)
    corrupted.loc[test_rows, list(TEAM_FEATURE_COLUMNS)] = 1e9
    corrupted.loc[test_rows, "home_win"] = 1.0 - corrupted.loc[test_rows, "home_win"]

    corrupted_eligible = _eligible(corrupted)
    after = select_c(_season_slice(corrupted_eligible, TRAIN_START, 2005),
                     _season_slice(corrupted_eligible, 2006, 2006), "team_only")
    assert after["C"] == picked["C"], "the selected C moved when only test rows changed"
    assert [c["valid_log_loss"] for c in after["candidates"]] == before_scores
    after_arrays = _fitted_arrays(
        fit_model(_season_slice(corrupted_eligible, TRAIN_START, 2005),
                  variant="team_only", C=after["C"]))
    for name in before:
        assert np.array_equal(before[name], after_arrays[name]), \
            f"{name} moved when only outer-test rows changed"

    # Non-vacuous, twice. Validation labels MAY legitimately change selection
    # (gate 7), so the candidate scores must move when they are corrupted...
    v = frame.copy()
    v.loc[v["home_season"].eq(2006), "home_win"] = 1.0 - v.loc[v["home_season"].eq(2006), "home_win"]
    v_scores = [c["valid_log_loss"] for c in
                select_c(train, _season_slice(_eligible(v), 2006, 2006), "team_only")["candidates"]]
    assert v_scores != before_scores, "corrupting validation labels must change candidate scores"

    # ...and training features must move the coefficients.
    t = frame.copy()
    t.loc[t["home_season"].le(2005), list(TEAM_FEATURE_COLUMNS)] = 1e3
    moved = _fitted_arrays(fit_model(_season_slice(_eligible(t), TRAIN_START, 2005),
                                     variant="team_only", C=picked["C"]))
    assert not np.array_equal(before["coef"], moved["coef"]), \
        "corrupting training features must move the fit"

    print("OK -- outer-test outcomes cannot move the selected C or any fitted parameter, "
          "while validation labels and training features demonstrably can.")


def demo_unfittable_fold_is_reported_not_crashed():
    frame = _matchups(seasons=range(1999, 2008), quotes_from=2005)
    result = run_backtest(frame, outer_seasons=(2005, 2006, 2007))

    failures = result["failures"]
    assert len(failures), "an unfittable fold must be recorded, not silently skipped"
    assert set(failures["variant"]) == {"team_market", "market_only", "team_residual",
                                        "team_residual_plus"}, \
        "both moneyline variants need paired quotes, so both fail on the same folds"
    assert set(failures["season"]) == {2005, 2006}, \
        "team_market is unfittable exactly while its training fold holds no quotes"
    assert failures["error"].str.contains("no rows with a paired market quote").all()
    assert set(failures["stage"]) <= {"select_c", "refit"}

    predictions = result["predictions"]
    for season in (2005, 2006):
        models = set(predictions.loc[predictions["season"] == season, "model"])
        assert not {"team_market", "market_only", "team_residual", "team_residual_plus"} & models, \
            "a failed variant must emit no predictions"
        assert {"team_only", "home_frequency", "market"} <= models, \
            "the other models must still cover a season team_market could not"

    # Non-vacuous: quote every season and the failures disappear.
    full = run_backtest(_matchups(seasons=range(1999, 2008)), outer_seasons=(2005, 2006, 2007))
    assert len(full["failures"]) == 0
    assert set(full["predictions"].loc[full["predictions"]["model"] == "team_market",
                                       "season"]) == {2005, 2006, 2007}

    print("OK -- an unfittable fold is reported per variant and the other models still "
          "cover that season; filling the quote gap removes the failures entirely.")


def demo_populations_never_mix_coverage():
    frame = _matchups(seasons=range(1999, 2008), quotes_from=2000)
    result = run_backtest(frame, outer_seasons=(2005, 2006, 2007))
    predictions, metrics = result["predictions"], result["metrics"]

    for population, declared in (("full", FULL_COVERAGE_MODELS), ("matched", MATCHED_MODELS)):
        subset = predictions.loc[predictions["paired"]] if population == "matched" else predictions
        for season in sorted(subset["season"].unique()):
            id_sets = [set(subset.loc[subset["model"].eq(m) & subset["season"].eq(season), "game_id"])
                       for m in declared]
            assert all(s == id_sets[0] for s in id_sets), \
                f"{population}/{season}: declared models cover different games"

    scored = metrics.loc[metrics["population"].eq("full"), "model"].unique()
    assert not {"market", "team_market", "market_only", "team_residual",
                "team_residual_plus"} & set(scored), \
        "a restricted-coverage model must never receive a full-coverage score"

    assert comparison_seasons(predictions) == [2005, 2006, 2007]

    # Non-vacuous: blank one full-coverage prediction and the mismatch must
    # raise rather than quietly scoring the smaller sample.
    holed = predictions.copy()
    mask = holed["model"].eq("team_only") & holed["season"].eq(2006)
    holed.loc[holed.index[mask][0], "p_home"] = np.nan
    try:
        score_predictions(holed)
        assert False, "expected the coverage difference to be caught"
    except ValueError as exc:
        assert "different games" in str(exc)

    print("OK -- declared models cover identical IDs within a population, restricted "
          "models get no full-coverage score, and a coverage hole raises.")


def demo_home_frequency_is_fitted_once_and_never_refit():
    # 12 training games, 7 home wins -> 7/12 exactly.
    labels = pd.Series([1.0] * 7 + [0.0] * 5)
    assert fit_home_frequency(labels)["p_home"] == 0.5833333333333334

    frame = _matchups(seasons=range(1999, 2007), quotes_from=2000)
    result = run_backtest(frame, outer_seasons=(2005, 2006))
    predictions = result["predictions"]

    for season, fitted in result["home_frequency"].items():
        rows = predictions.loc[predictions["model"].eq("home_frequency")
                               & predictions["season"].eq(season)]
        assert (rows["p_home"] == fitted["p_home"]).all(), "the constant must not vary by row"
        # The same float on both populations: scoring a subset never refits.
        assert rows.loc[rows["paired"], "p_home"].nunique() == 1
        assert rows.loc[rows["paired"], "p_home"].iloc[0] == fitted["p_home"]

    # Non-vacuous: mutating TEST labels leaves it alone, TRAINING labels move it.
    mutated_test = frame.copy()
    test_rows = mutated_test["home_season"].eq(2006)
    mutated_test.loc[test_rows, "home_win"] = 1.0 - mutated_test.loc[test_rows, "home_win"]
    assert run_backtest(mutated_test, outer_seasons=(2006,))["home_frequency"][2006] \
        == result["home_frequency"][2006]

    mutated_train = frame.copy()
    train_rows = mutated_train["home_season"].le(2005)
    mutated_train.loc[train_rows, "home_win"] = 1.0
    assert run_backtest(mutated_train, outer_seasons=(2006,))["home_frequency"][2006]["p_home"] \
        != result["home_frequency"][2006]["p_home"]

    print("OK -- home-frequency is fitted once per fold on 1999..T-1, is identical across "
          "both populations, and only training labels can move it.")


def demo_malformed_input_fails_with_a_useful_message():
    frame = _matchups(seasons=range(1999, 2008))
    cases = [
        (frame.drop(columns=["neutral_site"]), OUTER_SEASONS[:1], KeyError, "missing columns"),
        (frame, (2025,), ValueError, "excluded from this release"),
        (frame, (), ValueError, "must not be empty"),
    ]
    for data, seasons, kind, fragment in cases:
        try:
            make_folds(data, outer_seasons=seasons)
            assert False, f"expected {kind.__name__} for {fragment}"
        except kind as exc:
            assert fragment in str(exc), f"message must say {fragment!r}, got {exc!r}"

    duplicated = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    try:
        make_folds(duplicated, outer_seasons=(2005,))
        assert False, "expected ValueError for a duplicate game_id"
    except ValueError as exc:
        assert "non-null and unique" in str(exc)

    undated = frame.copy()
    undated.loc[0, "home_gameday"] = "09/10/1999"
    try:
        make_folds(undated, outer_seasons=(2005,))
        assert False, "expected ValueError for an unparseable gameday"
    except ValueError as exc:
        assert "unparseable gameday" in str(exc)

    empty = frame.loc[frame["home_season"].ge(2004)]
    try:
        make_folds(empty, outer_seasons=(2005,))
        assert False, "expected ValueError for an empty partition"
    except ValueError as exc:
        assert "has no eligible games" in str(exc)

    try:
        select_c(_season_slice(_eligible(frame), TRAIN_START, 2005),
                 _season_slice(_eligible(frame), 2006, 2006), "team_only", c_grid=(0.0,))
        assert False, "expected ValueError for a non-positive C"
    except ValueError as exc:
        assert "finite and positive" in str(exc)

    messages = {"missing columns", "excluded from this release", "must not be empty",
                "non-null and unique", "unparseable gameday", "has no eligible games",
                "finite and positive"}
    assert len(messages) == 7, "each failure must have its own message"

    print("OK -- missing columns, excluded seasons, empty input, duplicate IDs, bad dates, "
          "empty partitions and a bad C grid each fail with their own message.")


def demo_artifacts_round_trip_and_never_overwrite():
    frame = _matchups(seasons=range(1999, 2007), quotes_from=2000, neutral_every=17)
    result = run_backtest(frame, outer_seasons=(2005, 2006))

    with tempfile.TemporaryDirectory() as tmp:
        # A real snapshot, not a bare directory: the manifest records the
        # snapshot's checksum as provenance, so a fixture with no database
        # would only prove that the hash was skipped.
        snapshot_dir = _tiny_snapshot(Path(tmp))
        out = Path(tmp) / "run"
        save_backtest(result, frame, _games_stub(frame), snapshot_dir, out_dir=out)

        for name in ("folds", "fold_summary", "candidates", "selection", "predictions",
                     "metrics", "failures"):
            assert (out / f"{name}.csv").is_file(), f"{name}.csv missing"

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["comparison_seasons"] == [2005, 2006]
        assert "never excluded" in manifest["neutral_site_policy"]
        assert manifest["protocol"]["c_grid"] == list(C_GRID)
        assert manifest["protocol"]["c_grid_overrides"] == {k: list(v) for k, v in VARIANT_C_GRID.items()}
        tried = pd.read_csv(out / "candidates.csv").groupby("variant")["C"].apply(lambda c: sorted(set(c)))
        for variant in ("team_residual", "team_residual_plus"):
            assert tried[variant] == [1e-5, 1e-4, 1e-3, *C_GRID], f"{variant}: the extended grid must be searched"
        assert all(tried[v] == list(C_GRID) for v in tried.index if v not in VARIANT_C_GRID), \
            "every other variant must keep the frozen grid"

        # Replay: the saved predictions must reproduce every metric.
        replayed = score_predictions(pd.read_csv(out / "predictions.csv"))
        original = result["metrics"]
        merged = original.merge(replayed, on=["season", "population", "model"],
                                suffixes=("_orig", "_replay"))
        assert len(merged) == len(original)
        for column in ("log_loss", "brier", "accuracy"):
            np.testing.assert_allclose(merged[f"{column}_orig"], merged[f"{column}_replay"],
                                       atol=1e-8, equal_nan=True)

        # Neutral-site rows survive the round trip rather than being dropped.
        assert pd.read_csv(out / "predictions.csv")["neutral_site"].any()

        assert manifest["snapshot_sha256"], "the manifest must carry the snapshot's checksum"

        saved_models = {f.relative_to(out).as_posix(): f for f in out.rglob("*.joblib")}
        assert saved_models, "the run must save its fitted models"
        assert set(manifest["model_sha256"]) == set(saved_models), \
            "every saved model, and nothing else, must have a recorded checksum"
        for key, path in saved_models.items():
            assert manifest["model_sha256"][key] == _sha256_file(path), f"{key} checksum is wrong"

        try:
            save_backtest(result, frame, _games_stub(frame), snapshot_dir, out_dir=out)
            assert False, "expected FileExistsError rather than overwriting an earlier run"
        except FileExistsError:
            pass

    print("OK -- every artifact is written, predictions.csv replays every metric within "
          "1e-8, neutral-site rows survive, and a second save refuses to overwrite.")


def _tiny_snapshot(root: Path) -> Path:
    """A real captured snapshot over a two-row database, so save_backtest's
    provenance hashing is exercised rather than skipped."""
    import sqlite3
    from src.data.snapshot import create_snapshot, snapshot_dir

    source = root / "games.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE games (game_id TEXT, season INT)")
        conn.execute("INSERT INTO games VALUES ('g1', 2005)")
        conn.execute("CREATE TABLE plays (game_id TEXT, play_id INT)")
        conn.execute("INSERT INTO plays VALUES ('g1', 1)")
    manifest = create_snapshot(db_path=str(source), snapshots_dir=str(root / "snapshots"))
    return snapshot_dir(root / "snapshots", manifest["snapshot_sha256"])


def _games_stub(matchups):
    """Minimal `games` frame for exclusion_report inside save_backtest."""
    return pd.DataFrame({
        "game_id": matchups["game_id"], "season": matchups["home_season"],
        "week": matchups["home_week"], "game_type": "REG",
        "gameday": matchups["home_gameday"],
        "home_score": 20.0, "away_score": 17.0,
        "home_team": "H", "away_team": "A",
    })


def check_real_backtest_matches_the_measured_window(snapshot_dir):
    """Real data. Pins the window Stage 4 can actually cover."""
    matchups, _ = load_matchups(snapshot_dir)
    result = run_backtest(matchups)
    predictions, metrics = result["predictions"], result["metrics"]

    failures = result["failures"]
    assert len(failures) == 12, f"expected exactly 12 unfittable folds, got {len(failures)}"
    assert sorted(failures["season"]) == [2005] * 4 + [2006] * 4 + [2007] * 4
    assert set(failures["variant"]) == {"team_market", "market_only", "team_residual",
                                        "team_residual_plus"}
    assert failures["error"].str.contains("no rows with a paired market quote").all()

    for model, expected in (("team_only", set(OUTER_SEASONS)),
                            ("home_frequency", set(OUTER_SEASONS)),
                            ("market", set(OUTER_SEASONS)),
                            ("team_market", set(range(2008, 2025))),
                            ("market_only", set(range(2008, 2025))),
                            ("team_residual", set(range(2008, 2025))),
                            ("team_residual_plus", set(range(2008, 2025)))):
        covered = set(predictions.loc[predictions["model"] == model, "season"])
        assert covered == expected, f"{model} covers {sorted(covered)[:3]}..., expected otherwise"

    per_season = metrics.loc[metrics["season"].ne("pooled")]
    full = per_season.loc[per_season["population"].eq("full") & per_season["model"].eq("team_only")]
    matched = per_season.loc[per_season["population"].eq("matched")
                             & per_season["model"].eq("market")]
    assert int(full["n"].sum()) == 5170, f"full population sums to {int(full['n'].sum())}, not 5170"
    assert int(matched["n"].sum()) == 4780, \
        f"matched population sums to {int(matched['n'].sum())}, not 4780"

    assert comparison_seasons(predictions) == list(range(2008, 2025))
    pooled = metrics.loc[metrics["season"].eq("pooled") & metrics["population"].eq("matched")]
    assert set(pooled["n"]) == {4316}, f"matched pooled n must be 4316 each, got {set(pooled['n'])}"

    market = float(pooled.loc[pooled["model"].eq("market"), "log_loss"].iloc[0])
    assert abs(market - 0.606980) < 5e-7, f"market pooled log loss {market:.6f}, expected 0.606980"

    team_only = float(pooled.loc[pooled["model"].eq("team_only"), "log_loss"].iloc[0])
    home_freq = float(pooled.loc[pooled["model"].eq("home_frequency"), "log_loss"].iloc[0])
    assert market < team_only < home_freq, \
        "the market should beat team_only, which should beat a constant"

    neutral = int(predictions.loc[predictions["model"].eq("team_only"), "neutral_site"].sum())
    assert neutral == 57, f"expected 57 eligible neutral-site games scored, got {neutral}"

    print(f"OK -- 12 unfittable folds (the four moneyline variants, 2005-2007); full sums to 5170 and matched "
          f"to 4780; comparison window 2008-2024 with n=4316 each; market pooled log loss "
          f"{market:.6f}; {neutral} neutral-site games scored and flagged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", help="run the real-data window check against this snapshot")
    args = parser.parse_args()

    demo_folds_partition_exactly_and_in_order()
    demo_outer_test_outcomes_cannot_reach_selection_or_fit()
    demo_unfittable_fold_is_reported_not_crashed()
    demo_populations_never_mix_coverage()
    demo_home_frequency_is_fitted_once_and_never_refit()
    demo_malformed_input_fails_with_a_useful_message()
    demo_artifacts_round_trip_and_never_overwrite()
    if args.snapshot:
        check_real_backtest_matches_the_measured_window(args.snapshot)
