"""
Tests for src/evaluation/diagnose.py (docs/next-implementations.md Stage 0
acceptance criteria): unconditional 2025/2026 exclusion, exact scored-row
population, save/reload reproducibility, and one smoke run against the
real cached database (read-only, no network, no mutation of
data/nfl_games.db).

Run: python3 tests/test_diagnose.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.diagnose import (
    select_window, scored_rows, per_season_report, save_diagnostic, run_diagnostic,
    DEFAULT_SEASONS, EXCLUDED_SEASONS, GAMES_DB_PATH,
)


def _matchup(rows):
    """rows: list of dicts. Missing keys default to a decided, paired-quote
    2020 home win so each test only has to spell out what it cares about."""
    defaults = {
        "home_season": 2020, "home_week": 1, "home_team": "H", "away_team": "A",
        "home_win": True, "market_home_probability": 0.6,
        "market_home_moneyline": -150, "market_away_moneyline": 130,
    }
    out = pd.DataFrame([{**defaults, **r} for r in rows])
    out.insert(0, "game_id", [f"g{i}" for i in range(len(out))])
    out["away_season"] = out["home_season"]
    return out


def demo_select_window_excludes_2025_2026_unconditionally():
    matchup = _matchup([
        {"home_season": 2020}, {"home_season": 2024}, {"home_season": 2025}, {"home_season": 2026},
    ])
    windowed = select_window(matchup, seasons=range(2018, 2027))  # explicitly asks for 2025/2026 too
    assert set(windowed["home_season"]) == {2020, 2024}, \
        "2025/2026 must never survive select_window, even when explicitly requested"

    # Mutating an excluded row cannot change anything derived from the window.
    mutated = matchup.copy()
    mutated.loc[mutated["home_season"] == 2025, "home_win"] = False
    windowed_after = select_window(mutated, seasons=range(2018, 2027))
    assert windowed.reset_index(drop=True).equals(windowed_after.reset_index(drop=True))

    print("OK -- select_window drops 2025/2026 unconditionally and excluded rows can't affect the result.")


def demo_scored_rows_excludes_unplayed_and_tied():
    matchup = _matchup([
        {"home_win": True, "market_home_probability": 0.6},   # decided, paired -> scored
        {"home_win": pd.NA, "market_home_probability": 0.55},  # unplayed/tied but has a quote -> excluded
        {"home_win": False, "market_home_probability": np.nan},  # decided, unpaired -> scored, no quote
    ])
    windowed = select_window(matchup, seasons=[2020])
    rows = scored_rows(windowed)
    assert len(rows) == 2, "a valid market quote on an unplayed/tied game must never become a scored row"
    assert set(rows["home_win"]) == {True, False}
    assert rows["market_home_probability"].isna().sum() == 1

    print("OK -- scored_rows keeps only non-null-label games regardless of market-quote presence.")


def demo_per_season_report_denominators():
    matchup = _matchup([
        {"home_season": 2020, "home_win": True, "market_home_probability": 0.6},
        {"home_season": 2020, "home_win": False, "market_home_probability": 0.4},
        {"home_season": 2020, "home_win": True, "market_home_probability": np.nan},  # unpaired
        {"home_season": 2021, "home_win": False, "market_home_probability": 0.5},
    ])
    rows = scored_rows(select_window(matchup, seasons=[2020, 2021]))
    report = per_season_report(rows).set_index("season")

    assert report.loc[2020, "n_full"] == 3 and report.loc[2020, "n_paired"] == 2
    assert report.loc[2020, "always_home_accuracy_full"] == 2 / 3
    assert report.loc[2020, "always_home_accuracy_paired"] == 0.5
    assert report.loc[2021, "n_full"] == 1 and report.loc[2021, "n_paired"] == 1
    assert report.loc["pooled", "n_full"] == 4 and report.loc["pooled", "n_paired"] == 3
    assert report.loc["pooled", "always_home_accuracy_full"] == 2 / 4

    print("OK -- per_season_report denominators (full vs. paired, per-season vs. pooled) match hand computation.")


def demo_save_and_reload_reproduces_metrics():
    matchup = _matchup([
        {"home_win": True, "market_home_probability": 0.7},
        {"home_win": False, "market_home_probability": 0.3},
        {"home_win": True, "market_home_probability": 0.55},
    ])
    rows = scored_rows(select_window(matchup, seasons=[2020]))
    original_report = per_season_report(rows)

    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d) / "run"
        exclusions = pd.DataFrame([{"season": 2020, "reason": "history_eligible", "count": 3}])
        paths = save_diagnostic(rows, exclusions, {"diagnostic": "stage0"}, out_dir)

        # A later mutation of the "live" in-memory rows must not affect what's on disk.
        rows.loc[0, "home_win"] = False

        reloaded = pd.read_csv(paths["rows_path"])
        reloaded_report = per_season_report(reloaded)
        pd.testing.assert_frame_equal(
            original_report.reset_index(drop=True), reloaded_report.reset_index(drop=True),
        )

        import json
        manifest = json.loads(paths["manifest_path"].read_text())
        assert manifest["scored_rows_count"] == 3

    print("OK -- reloading saved scored rows reproduces every printed metric; a later live mutation doesn't.")


def demo_full_smoke_against_real_db():
    with tempfile.TemporaryDirectory() as d:
        result = run_diagnostic(seasons=DEFAULT_SEASONS, db_path=GAMES_DB_PATH, out_dir=Path(d) / "run")

    seasons_scored = result["manifest"]["seasons_scored"]
    assert set(seasons_scored).isdisjoint(EXCLUDED_SEASONS)
    assert seasons_scored == sorted(seasons_scored)
    assert min(seasons_scored) >= 2005 and max(seasons_scored) <= 2024

    pooled = result["report"].set_index("season").loc["pooled"]
    assert pooled["n_full"] > 0 and pooled["n_paired"] > 0
    assert 0 <= pooled["always_home_accuracy_full"] <= 1
    assert np.isfinite(pooled["market_log_loss"])

    print(f"OK -- Stage 0 diagnostic ran against the real database: "
          f"{int(pooled['n_full'])} scored games, "
          f"{int(pooled['n_paired'])} with a paired market quote, "
          f"pooled market log loss {pooled['market_log_loss']:.4f}.")


def demo_fully_excluded_window_degrades_gracefully():
    # A plausible misuse: someone runs the diagnostic asking only for the
    # unconditionally-excluded seasons (e.g. --seasons 2025-2026). This must
    # not crash -- it should degrade to a single "pooled" row reporting
    # zero games, the same shape callers already handle for an empty season.
    matchup = _matchup([{"home_season": 2025}, {"home_season": 2026}])
    rows = scored_rows(select_window(matchup, seasons=[2025, 2026]))
    assert len(rows) == 0

    report = per_season_report(rows)
    assert len(report) == 1 and report.loc[0, "season"] == "pooled"
    assert report.loc[0, "n_full"] == 0 and report.loc[0, "n_paired"] == 0
    assert np.isnan(report.loc[0, "always_home_accuracy_full"])
    assert np.isnan(report.loc[0, "market_log_loss"])

    print("OK -- a window that excludes every requested season degrades to a zero-game "
          "pooled report instead of crashing.")


def demo_single_game_and_one_class_season_outliers():
    # A season with exactly one scored game: real, not "too small to
    # report." log loss/Brier/accuracy are still defined; ROC AUC is
    # undefined (only one class present) and must read NaN, not 0 or 1.
    matchup = _matchup([{"home_season": 2020, "home_win": True, "market_home_probability": 0.9}])
    rows = scored_rows(select_window(matchup, seasons=[2020]))
    report = per_season_report(rows).set_index("season")
    assert report.loc[2020, "n_paired"] == 1
    assert np.isfinite(report.loc[2020, "market_log_loss"])
    assert np.isnan(report.loc[2020, "market_roc_auc"])

    # A season where every game has the same outcome (a real if extreme
    # population, e.g. a small division that never saw a road win that
    # year) is one-class for the whole season, not just one game.
    matchup_one_class = _matchup([
        {"home_season": 2021, "home_win": True, "market_home_probability": 0.8},
        {"home_season": 2021, "home_win": True, "market_home_probability": 0.55},
        {"home_season": 2021, "home_win": True, "market_home_probability": 0.65},
    ])
    rows_oc = scored_rows(select_window(matchup_one_class, seasons=[2021]))
    report_oc = per_season_report(rows_oc).set_index("season")
    assert report_oc.loc[2021, "always_home_accuracy_full"] == 1.0
    assert np.isnan(report_oc.loc[2021, "market_roc_auc"])
    assert np.isfinite(report_oc.loc[2021, "market_log_loss"])

    print("OK -- a single-game season and an all-one-outcome season score log loss/Brier/"
          "accuracy normally and read ROC AUC as NaN (undefined), not 0 or 1.")


def demo_extreme_market_quote_outliers():
    # A near-certain moneyline (e.g. -10000) devigs to a probability very
    # close to the 0/1 boundary -- must be scored (with log loss's
    # documented clipping), not rejected as "too extreme," and a wrong
    # near-certain prediction must dominate pooled log loss the way a real
    # blowout-favorite upset does.
    matchup = _matchup([
        {"home_season": 2020, "home_win": False, "market_home_probability": 0.999,
         "market_home_moneyline": -10000, "market_away_moneyline": 5000},
        {"home_season": 2020, "home_win": True, "market_home_probability": 0.5},
    ])
    rows = scored_rows(select_window(matchup, seasons=[2020]))
    report = per_season_report(rows).set_index("season")
    assert report.loc[2020, "n_paired"] == 2
    assert report.loc[2020, "market_log_loss"] > 3, \
        "a confident-and-wrong probability must dominate pooled log loss, not get clipped away"
    assert np.isfinite(report.loc[2020, "market_log_loss"])

    print("OK -- a near-certain (0.999) market probability that turns out wrong is scored "
          "in full (large but finite log loss), not rejected or silently softened.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=GAMES_DB_PATH)
    GAMES_DB_PATH = parser.parse_args().db_path
    demo_select_window_excludes_2025_2026_unconditionally()
    demo_scored_rows_excludes_unplayed_and_tied()
    demo_per_season_report_denominators()
    demo_save_and_reload_reproduces_metrics()
    demo_full_smoke_against_real_db()
    demo_fully_excluded_window_degrades_gracefully()
    demo_single_game_and_one_class_season_outliers()
    demo_extreme_market_quote_outliers()
