"""
Sanity check for src/data/ingest.py. Pulls one real season (small/fast) and
checks the raw data is structurally sound before anything downstream builds
on it: unique keys, expected non-null pre-game columns, and that every play
belongs to a game that's actually in the schedule.

This is NOT a leakage test -- there are no computed features yet for
leakage to hide in. It only guards the ingestion step itself. The real
leakage tests (rolling stats using only strictly-earlier games) belong with
the feature-engineering step, once that exists.

Run: python3 tests/test_ingest.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ingest import (
    fetch_schedules, fetch_pbp, _select_pbp_columns, PBP_COLUMNS, PBP_REQUIRED_COLUMNS,
)

SEASON = 2023


def demo_required_columns_are_enforced():
    """Offline; no network. The column trim is an intersection, so a column
    that disappears upstream is dropped from the curated list too -- and any
    check comparing the result against that same filtered list passes while
    the data is gone. Required columns must raise instead."""
    assert set(PBP_REQUIRED_COLUMNS) <= set(PBP_COLUMNS), \
        "a required column must also be one of the curated columns"

    complete = pd.DataFrame({c: [0] for c in PBP_COLUMNS})
    assert list(_select_pbp_columns(complete).columns) == PBP_COLUMNS

    # An optional column may vanish: nflverse has no air_yards for 1999/2005,
    # and nothing downstream reads it yet.
    optional_gone = complete.drop(columns=["air_yards", "special_teams_play"])
    kept = _select_pbp_columns(optional_gone)
    assert "air_yards" not in kept.columns
    assert list(kept.columns) == [c for c in PBP_COLUMNS if c in kept.columns], \
        "surviving columns must keep PBP_COLUMNS order"

    # A required one may not, and the error has to name it.
    for column in ("success", "epa", "posteam", "play_id"):
        try:
            _select_pbp_columns(complete.drop(columns=[column]))
            assert False, f"expected ValueError when {column} is missing"
        except ValueError as exc:
            assert column in str(exc), f"the error must name {column}, got: {exc}"

    # Extra source columns are trimmed away, not passed through.
    assert "clock_running" not in _select_pbp_columns(complete.assign(clock_running=1)).columns

    print(f"OK -- all {len(PBP_REQUIRED_COLUMNS)} required pbp columns raise when absent; "
          f"optional ones are still allowed to be missing.")


def demo():
    games = fetch_schedules([SEASON])
    assert not games["game_id"].duplicated().any(), "duplicate game_id in schedules"
    assert (games["season"] == SEASON).all(), "schedules returned rows outside the requested season"
    # 2023 is a completed season -- every game should have a final score and a closing line.
    assert games["home_score"].notna().all(), "completed season has games with no home_score"
    assert games["spread_line"].notna().all(), "completed season has games with no spread_line"

    plays = fetch_pbp([SEASON])
    # Not `[c for c in PBP_COLUMNS if c in plays.columns]` -- that filters the
    # expectation by the result, so a column missing from both sides compares
    # equal and the loss goes unreported. Name the required set outright.
    absent = [c for c in PBP_REQUIRED_COLUMNS if c not in plays.columns]
    assert not absent, f"required pbp columns missing from a completed season: {absent}"
    assert set(plays.columns) <= set(PBP_COLUMNS), "fetch_pbp returned an uncurated column"
    assert list(plays.columns) == [c for c in PBP_COLUMNS if c in set(plays.columns)], \
        "pbp columns are out of PBP_COLUMNS order"
    assert not plays.duplicated(subset=["game_id", "play_id"]).any(), "duplicate (game_id, play_id) in plays"
    assert plays["game_id"].isin(games["game_id"]).all(), "a play references a game_id missing from schedules"
    assert plays["epa"].notna().sum() > 0, "epa column is entirely null -- pbp fetch likely broken"

    print(f"OK -- {len(games)} {SEASON} games, {len(plays)} plays, all keys unique and cross-referenced.")


if __name__ == "__main__":
    demo_required_columns_are_enforced()   # offline
    demo()                                 # pulls one real season
