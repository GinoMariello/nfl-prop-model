"""
Tests for src/evaluation/report.py (docs/next-implementations.md Stage 5
acceptance criteria; protocol: docs/experiment-protocol.md).

The ones that matter are the bootstrap's two contracts: the same sampled
season blocks are reused for every compared model, and the differences are
paired per game rather than two independently resampled means subtracted.
Both are invisible in the output -- a report built the wrong way still prints
plausible intervals -- so each is proven by construction rather than by a
value check.

Synthetic and offline; no snapshot, no database, no network.

Run: python3 tests/test_report.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.backtest import FULL_COVERAGE_MODELS, MATCHED_MODELS, comparison_seasons
from src.evaluation.report import (
    BASELINE_MODEL, INTERVAL, N_REPLICATES, _draw_blocks, _window_text,
    bootstrap_paired_differences, build_report, calibration_table, equal_season_mean,
    load_run, per_game_losses, render_report, save_report,
)
from src.models.train import RANDOM_SEED


ALL_MODELS = ("home_frequency", "team_only", "team_market", "team_spread",
              "market_only", "team_residual", "team_residual_plus", "market")


def _predictions(seasons=range(2005, 2011), per_season=20, seed=0, quotes_from=None,
                 fitted_from=None, season_sizes=None, models=ALL_MODELS):
    """A Stage 4 predictions frame: one row per (model, game).

    `quotes_from` and `fitted_from` are separate on purpose, because they are
    separate on real data: quotes appear in 2006, but team_market's training
    fold does not reach them until 2008, so there are seasons that ARE in the
    matched population and which team_market nonetheless did not score. A
    fixture that conflates the two hides exactly the mismatch the comparison
    window exists to prevent.
    """
    rng = np.random.RandomState(seed)
    fitted_from = fitted_from if fitted_from is not None else quotes_from
    rows = []
    for season in seasons:
        n = (season_sizes or {}).get(season, per_season)
        for i in range(n):
            label = float(rng.randint(2))
            game = f"{season}_{i:03d}"
            paired = quotes_from is None or season >= quotes_from
            for model in models:
                if model == "market" and not paired:
                    continue
                if model in ("team_market", "market_only", "team_residual", "team_residual_plus") \
                        and (fitted_from is None or season < fitted_from):
                    if fitted_from is not None:
                        continue
                skill = {"home_frequency": 0.0, "team_only": 0.25,
                         "team_market": 0.35, "team_spread": 0.38,
                         "market_only": 0.39, "team_residual": 0.395, "team_residual_plus": 0.396,
                         "market": 0.40}[model]
                p = 0.5 + skill * (label - 0.5) * 2 + rng.normal(scale=0.05)
                rows.append({"season": season, "week": i + 1,
                             "gameday": f"{season}-09-{1 + i % 28:02d}", "game_id": game,
                             "model": model, "variant": None, "label": label,
                             "p_home": float(np.clip(p, 0.02, 0.98)),
                             "abstention_reason": None, "paired": paired,
                             "neutral_site": False, "selected_C": np.nan})
    return pd.DataFrame(rows)


def _metrics_stub(predictions):
    from src.evaluation.backtest import score_predictions
    return score_predictions(predictions)


def demo_blocks_are_drawn_once_and_shared_by_every_model():
    """experiment-protocol.md: "reuse the same sampled blocks for every
    compared model". An independent draw per model would make a paired
    interval reflect two unrelated resamplings."""
    seasons = range(2005, 2011)
    first = _draw_blocks(seasons, n_replicates=50, seed=RANDOM_SEED)
    again = _draw_blocks(seasons, n_replicates=50, seed=RANDOM_SEED)
    assert all(np.array_equal(a, b) for a, b in zip(first, again)), \
        "the same seed must produce the same blocks"
    assert not all(np.array_equal(a, b) for a, b in
                   zip(first, _draw_blocks(seasons, 50, seed=RANDOM_SEED + 1))), \
        "a different seed must produce different blocks -- otherwise the seed does nothing"

    # Whole seasons, with replacement, as many blocks as seasons.
    for draw in first:
        assert len(draw) == len(list(seasons))
        assert set(draw) <= set(seasons)
    assert any(len(set(draw)) < len(list(seasons)) for draw in first), \
        "resampling with replacement must sometimes repeat a season"

    losses = pd.DataFrame({"season": [2005] * 10 + [2006] * 10,
                           "home_frequency": np.linspace(0.1, 2.0, 20)})
    losses["twin"] = losses["home_frequency"]
    losses["other"] = np.linspace(2.0, 0.1, 20)

    # The requirement is structural -- drawn once, reused -- so assert exactly
    # that. A per-model draw produces intervals that look entirely plausible,
    # which is why a value check cannot detect it: an earlier version of this
    # test compared only a model against its own twin, and that passes under
    # per-model draws because the difference is already zero per game.
    import src.evaluation.report as report_module
    draws = []
    real = report_module._draw_blocks

    def counting(*args, **kwargs):
        draws.append(1)
        return real(*args, **kwargs)

    report_module._draw_blocks = counting
    try:
        bootstrap_paired_differences(losses, ["twin", "other"], n_replicates=50)
    finally:
        report_module._draw_blocks = real
    assert len(draws) == 1, \
        f"blocks must be drawn once and shared across all compared models, drawn {len(draws)}x"

    out = bootstrap_paired_differences(losses, ["twin"], n_replicates=100)
    assert out["mean_difference"].iloc[0] == 0.0
    assert out["ci_low"].iloc[0] == 0.0 and out["ci_high"].iloc[0] == 0.0, \
        "identical models must have a zero-width interval"

    print("OK -- blocks are seeded, whole seasons, drawn with replacement, and shared "
          "across models; identical models give a zero-width interval.")


def demo_differences_are_paired_per_game():
    """A paired difference is the mean of per-game differences. Subtracting two
    separately resampled means is a different, wider statistic."""
    # One season: every bootstrap draw is that same season, so the interval
    # collapses onto the point estimate and the arithmetic is checkable by hand.
    losses = pd.DataFrame({"season": [2005] * 4,
                           "home_frequency": [1.0, 1.0, 1.0, 1.0],
                           "model_a": [0.5, 0.5, 0.5, 0.5]})
    out = bootstrap_paired_differences(losses, ["model_a"], n_replicates=25)
    assert out["mean_difference"].iloc[0] == -0.5
    assert out["ci_low"].iloc[0] == -0.5 and out["ci_high"].iloc[0] == -0.5, \
        "a single-season bootstrap can only ever draw that season"
    assert bool(out["favours_model"].iloc[0]) is True
    assert int(out["n_games"].iloc[0]) == 4 and int(out["n_seasons"].iloc[0]) == 1

    # A model that is worse must not be flagged as favoured.
    losses["model_b"] = [2.0, 2.0, 2.0, 2.0]
    worse = bootstrap_paired_differences(losses, ["model_b"], n_replicates=25)
    assert worse["mean_difference"].iloc[0] == 1.0
    assert bool(worse["favours_model"].iloc[0]) is False

    try:
        bootstrap_paired_differences(losses, ["model_a"], baseline="absent")
        assert False, "expected KeyError for a missing baseline"
    except KeyError as exc:
        assert "absent" in str(exc)

    print("OK -- differences are paired per game, a single-season bootstrap collapses to "
          "the point estimate, and a worse model is not flagged as favoured.")


def demo_models_are_never_compared_across_different_season_sets():
    """team_market has no predictions before its first fittable season. Pairing
    it against models that do cover those seasons is the mismatched comparison
    model-plan.md section 6 forbids."""
    predictions = _predictions(seasons=range(2005, 2011), quotes_from=2006, fitted_from=2008)

    # Quotes exist from 2006, so 2006 and 2007 ARE in the matched population --
    # but team_market did not score them, so the window must still start at 2008.
    assert set(predictions.loc[predictions["paired"], "season"]) == set(range(2006, 2011))
    assert comparison_seasons(predictions, "matched") == [2008, 2009, 2010]
    assert comparison_seasons(predictions, "full") == list(range(2005, 2011))

    losses = per_game_losses(predictions, "matched")
    assert set(losses["season"]) == {2008, 2009, 2010}, \
        "the matched loss table must be restricted to the shared window"
    assert not losses.isna().any().any(), "a restricted table must have no gaps"
    for model in MATCHED_MODELS:
        assert model in losses.columns

    full = per_game_losses(predictions, "full")
    assert set(full["season"]) == set(range(2005, 2011))
    assert list(full.columns) == ["season", *FULL_COVERAGE_MODELS]

    # The equal-season mean is restricted the same way: averaging team_market
    # over 3 seasons beside market over 6 compares different season sets.
    metrics = _metrics_stub(predictions)
    windows = {p: comparison_seasons(predictions, p) for p in ("full", "matched")}
    equal = equal_season_mean(metrics, windows)
    matched = equal.loc[equal["population"].eq("matched")]
    assert set(matched["n_seasons"]) == {3}, \
        f"every matched model must average over the same 3 seasons, got {set(matched['n_seasons'])}"
    assert set(equal.loc[equal["population"].eq("full"), "n_seasons"]) == {6}

    print("OK -- the loss table, the bootstrap and the equal-season mean are all restricted "
          "to the window every compared model covers.")


def demo_equal_season_mean_differs_from_the_pooled_score():
    """Reported separately because they answer different questions -- a fixture
    with uneven seasons makes them disagree."""
    predictions = _predictions(seasons=(2005, 2006), season_sizes={2005: 8, 2006: 60}, seed=3)
    metrics = _metrics_stub(predictions)
    windows = {p: comparison_seasons(predictions, p) for p in ("full", "matched")}
    equal = equal_season_mean(metrics, windows)

    pooled = metrics.loc[metrics["season"].eq("pooled") & metrics["population"].eq("full")
                         & metrics["model"].eq("team_only"), "log_loss"].iloc[0]
    mean_of_seasons = equal.loc[equal["population"].eq("full")
                                & equal["model"].eq("team_only"), "log_loss"].iloc[0]
    assert pooled != mean_of_seasons, \
        "with a 8-game and a 60-game season these must differ; reporting one hides which"

    # The equal-season mean really is the unweighted mean of the per-season scores.
    per_season = metrics.loc[metrics["season"].ne("pooled") & metrics["population"].eq("full")
                             & metrics["model"].eq("team_only"), "log_loss"]
    assert abs(mean_of_seasons - per_season.mean()) < 1e-12

    print("OK -- the equal-season mean is the unweighted mean of per-season scores and "
          "differs from the game-weighted pool when seasons are uneven.")


def demo_calibration_reports_empty_bins():
    predictions = _predictions(seasons=(2005, 2006))
    table = calibration_table(predictions, "full")

    for model in FULL_COVERAGE_MODELS:
        rows = table.loc[table["model"] == model]
        assert len(rows) == 10, "all ten bins must be present, occupied or not"
        assert rows["count"].sum() == int(
            (predictions["model"].eq(model) & predictions["p_home"].notna()).sum())
        empty = rows.loc[rows["count"] == 0]
        assert empty["mean_predicted"].isna().all(), \
            "an empty bin reports NaN rates, never 0 -- an absent bin and an empty bin differ"
        assert rows["bin_low"].is_monotonic_increasing

    print("OK -- calibration keeps all ten bins with counts, and an empty bin reports NaN "
          "rather than a fabricated zero.")


def demo_report_is_reproducible_and_never_overwrites():
    predictions = _predictions(seasons=range(2005, 2011), quotes_from=2006, fitted_from=2008)
    metrics = _metrics_stub(predictions)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        (run_dir / "manifest.json").write_text(json.dumps(
            {"run_id": "test", "target": "home_win", "failures": [],
             "git": {"commit": "abc123", "dirty": False}}))

        run = load_run(run_dir)
        first = build_report(run, n_replicates=200)
        second = build_report(load_run(run_dir), n_replicates=200)

        compared = set(zip(first["bootstrap"]["population"], first["bootstrap"]["baseline"],
                           first["bootstrap"]["model"]))
        assert ("matched", "team_residual", "team_residual_plus") in compared, \
            "protocol v5's primary comparison must be reported"
        assert ("matched", "market_only", "team_market") in compared, \
            "the market-only control comparison must be reported"
        assert {("matched", "market", m) for m in MATCHED_MODELS
                if m not in ("market", "home_frequency")} <= compared, \
            "every fitted matched model must also be differenced against the raw market"
        assert ("matched", "market", "home_frequency") not in compared, \
            "a mirror of the home_frequency-baseline row adds nothing"
        assert ("full", "home_frequency", "team_spread") in compared

        # Same predictions and same seed must reproduce the intervals exactly.
        pd.testing.assert_frame_equal(first["bootstrap"], second["bootstrap"])
        pd.testing.assert_frame_equal(first["calibration"], second["calibration"])
        assert render_report(first) == render_report(second)

        text = render_report(first)
        for expected in ("equal-season mean", "season-block bootstrap", "limitations",
                         "closing-line", "betting-profitability", "2008-2010 (3 seasons)",
                         "2005-2010 (6 seasons)"):
            assert expected in text, f"the report must state {expected!r}"

        save_report(first, run_dir)
        for name in ("report.txt", "bootstrap.csv", "calibration.csv", "equal_season.csv"):
            assert (run_dir / name).is_file()
        try:
            save_report(first, run_dir)
            assert False, "expected FileExistsError rather than overwriting a report"
        except FileExistsError:
            pass

        try:
            load_run(Path(tmp))
            assert False, "expected FileNotFoundError outside a run directory"
        except FileNotFoundError as exc:
            assert "not a Stage 4 run directory" in str(exc)

    print("OK -- the report reproduces exactly from saved predictions, states its "
          "limitations, and refuses to overwrite an earlier one.")


def demo_window_text_does_not_imply_a_gap_is_contiguous():
    assert _window_text([2008, 2009, 2010]) == "2008-2010 (3 seasons)"
    assert _window_text([2008, 2010]) == "2008, 2010 (2 seasons)", \
        "a gap must be spelled out, not rendered as a range"
    assert _window_text([]) == "none"
    print("OK -- a window with a missing season is listed rather than shown as a range.")


def _run_dir_with(root, predictions, manifest_extra):
    run_dir = Path(root) / "run"
    run_dir.mkdir()
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    _metrics_stub(predictions).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "old", "target": "home_win", "failures": [],
         "git": {"commit": "abc123", "dirty": False}, **manifest_extra}))
    return run_dir


def demo_report_uses_the_models_its_run_declared():
    """A run saved under an older protocol declares fewer models than the code
    does now. Its report must be built from what the RUN declared -- the code's
    current lists would demand models the run never fitted. Before this, an
    older run failed with "need at least one array to concatenate", naming
    nothing.
    """
    old_models = ("home_frequency", "team_only", "team_market", "market")
    predictions = _predictions(seasons=range(2005, 2011), quotes_from=2006, fitted_from=2008,
                               models=old_models)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir_with(tmp, predictions, {
            "full_coverage_models": ["home_frequency", "team_only"],
            "matched_models": list(old_models)})
        report = build_report(load_run(run_dir), n_replicates=100)

        compared = set(zip(report["bootstrap"]["baseline"], report["bootstrap"]["model"]))
        assert ("market", "team_market") in compared and ("market", "team_only") in compared
        assert not {m for _, m in compared} & {"team_spread", "market_only", "team_residual"}, \
            "a model the run never declared must not appear in its report"
        assert "market_only" not in set(report["bootstrap"]["baseline"]), \
            "a comparison whose baseline the run lacks does not apply to it"
        assert report["windows"] == {"full": list(range(2005, 2011)), "matched": [2008, 2009, 2010]}
        assert "2008-2010 (3 seasons)" in render_report(report)

    # A declared model with no predictions is a broken artifact, and must say so.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _run_dir_with(tmp, predictions, {
            "full_coverage_models": ["home_frequency", "team_only", "team_spread"],
            "matched_models": list(old_models)})
        try:
            build_report(load_run(run_dir), n_replicates=100)
            assert False, "a declared model missing from predictions.csv must raise"
        except ValueError as exc:
            assert "team_spread" in str(exc) and "predictions.csv" in str(exc), str(exc)
    print("OK -- a report follows the models its run declared, and a declared model "
          "with no predictions raises by name.")


if __name__ == "__main__":
    demo_blocks_are_drawn_once_and_shared_by_every_model()
    demo_differences_are_paired_per_game()
    demo_models_are_never_compared_across_different_season_sets()
    demo_equal_season_mean_differs_from_the_pooled_score()
    demo_calibration_reports_empty_bins()
    demo_report_is_reproducible_and_never_overwrites()
    demo_window_text_does_not_imply_a_gap_is_contiguous()
    demo_report_uses_the_models_its_run_declared()
