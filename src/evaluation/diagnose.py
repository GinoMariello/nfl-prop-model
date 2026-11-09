"""
Stage 0 exploratory diagnostic (docs/next-implementations.md Stage 0):
fitting-free always-home accuracy and devigged-market probability metrics
on development seasons only. This is NOT the Stage 1 experiment harness
(no frozen snapshot, no manifest of dependency versions beyond what's
printed here) and NOT a final evaluation -- see the printed header.

Every run reads the selected raw schedule rows once, in a read-only SQLite
transaction, and reuses the feature layer's label and market definitions.
No team statistics or rolling features are needed by these two baselines.
Development seasons default to 2005-2024; 2025 and 2026 are
excluded unconditionally, even if `--seasons` is given explicitly
including them (any inspection of that data already used it -- see
"Review decisions" in docs/next-implementations.md). Every run saves its
exact scored rows and eligibility/exclusion counts plus a checksummed
manifest to a unique output directory, so the printed numbers can be
reproduced later without rebuilding the feature pipeline.

Run: python3 -m src.evaluation.diagnose
"""
import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.rolling_features import _compute_labels, normalized_home_probability, exclusion_report
from src.features.team_game_stats import build_team_games
from src.evaluation.metrics import log_loss, brier_score, accuracy, roc_auc, calibration_bins
from src.models.baselines import always_home_accuracy

GAMES_DB_PATH = "data/nfl_games.db"
DEFAULT_SEASONS = tuple(range(2005, 2025))  # 2005-2024 inclusive
EXCLUDED_SEASONS = (2025, 2026)
DIAGNOSTICS_DIR = "diagnostics"

SCORED_ROW_COLUMNS = [
    "game_id", "season", "week", "home_team", "away_team", "home_win",
    "market_home_probability", "market_home_moneyline", "market_away_moneyline",
]


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_root=None) -> dict:
    """Best-effort -- a diagnostic run outside a git checkout (e.g. a
    packaged copy) still runs, just without provenance."""
    try:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE)
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
        diff = git("diff", "--no-ext-diff", "--binary", "HEAD", "--")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
        sources = {p: (root / p).read_text() for p in untracked
                   if p and not (root / p).is_symlink()
                   and (Path(p).suffix == ".py" or p == "requirements.txt")}
        return {"commit": commit, "dirty": dirty, "diff": diff, "untracked_sources": sources}
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        return {"commit": None, "dirty": None, "provenance_error": str(exc)}


def build_matchup(games: pd.DataFrame) -> pd.DataFrame:
    """Build only baseline scoring inputs, with the existing label/odds rules."""
    build_team_games(games)  # Reuse schedule key and team validation.
    labels = _compute_labels(games)
    metadata = games.reindex(columns=["game_id", "season", "week", "home_team", "away_team",
                                     "home_moneyline", "away_moneyline"]).rename(columns={
        "season": "home_season", "week": "home_week",
        "home_moneyline": "market_home_moneyline", "away_moneyline": "market_away_moneyline"})
    metadata["market_home_probability"] = normalized_home_probability(games)
    return labels.merge(metadata, on="game_id", validate="one_to_one")


def select_window(matchup: pd.DataFrame, seasons=DEFAULT_SEASONS) -> pd.DataFrame:
    """Restrict to `seasons`, after unconditionally dropping 2025/2026 --
    even if `seasons` explicitly names them. `home_season`/`away_season`
    are always identical for one game (both derived from the same row of
    `games`), so filtering on `home_season` alone is exact."""
    kept = sorted(set(seasons) - set(EXCLUDED_SEASONS))
    return matchup[matchup["home_season"].isin(kept)].copy()


def scored_rows(windowed: pd.DataFrame) -> pd.DataFrame:
    """The exact population Stage 0 scores: supervised-eligible rows
    (non-null `home_win`) in the already-window-filtered matchup. A valid
    market quote on an unplayed or tied game can never appear here --
    `home_win` is null for both (see `_compute_labels`), so the
    `notna()` filter drops it regardless of the market columns."""
    eligible = windowed[windowed["home_win"].notna()].copy()
    if eligible.game_id.isna().any() or eligible.game_id.duplicated().any():
        raise ValueError("Scored game IDs must be non-null and unique")
    renamed = eligible.rename(columns={"home_season": "season", "home_week": "week"})
    return renamed[SCORED_ROW_COLUMNS].sort_values("game_id").reset_index(drop=True)


def _score_group(label, rows: pd.DataFrame) -> dict:
    paired = rows[rows["market_home_probability"].notna()]
    y_paired = paired["home_win"].astype(float).to_numpy()
    p_paired = paired["market_home_probability"].to_numpy()
    return {
        "season": label,
        "n_full": len(rows),
        "always_home_accuracy_full": always_home_accuracy(rows["home_win"])["accuracy"],
        "n_paired": len(paired),
        "always_home_accuracy_paired": always_home_accuracy(paired["home_win"])["accuracy"],
        "market_log_loss": log_loss(y_paired, p_paired),
        "market_brier": brier_score(y_paired, p_paired),
        "market_accuracy": accuracy(y_paired, p_paired),
        "market_roc_auc": roc_auc(y_paired, p_paired),
    }


def per_season_report(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-season rows plus a final `season="pooled"` row over every
    scored game, each with its own denominators -- a bare accuracy number
    without its denominator/ID set isn't a valid comparison
    (docs/next-implementations.md Stage 0)."""
    records = [_score_group(season, group) for season, group in rows.groupby("season", sort=True)]
    records.append(_score_group("pooled", rows))
    return pd.DataFrame(records)


def save_diagnostic(rows: pd.DataFrame, exclusions: pd.DataFrame, manifest: dict, out_dir: Path) -> dict:
    """Writes scored rows, eligibility/exclusion counts, and a manifest
    (with both files' checksums) to `out_dir`. Reloading `scored_rows.csv`
    and recomputing metrics from it must reproduce every printed number
    exactly -- it is the frozen input, independent of whatever the live
    database looks like afterward."""
    out_dir.mkdir(parents=True, exist_ok=False)
    rows_path = out_dir / "scored_rows.csv"
    exclusions_path = out_dir / "eligibility_counts.csv"
    rows.to_csv(rows_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    manifest = dict(manifest)
    manifest["scored_rows_sha256"] = _sha256_file(rows_path)
    manifest["scored_rows_count"] = len(rows)
    manifest["eligibility_counts_sha256"] = _sha256_file(exclusions_path)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return {"rows_path": rows_path, "exclusions_path": exclusions_path, "manifest_path": manifest_path}


def run_diagnostic(seasons=DEFAULT_SEASONS, db_path: str = GAMES_DB_PATH, out_dir: Path = None) -> dict:
    """The Stage 0 diagnostic: per-season and pooled scores with no fitting.

    2025 and 2026 are dropped unconditionally, even when `seasons` names
    them explicitly -- the caller cannot opt back into the excluded window.
    Reads the database read-only and writes scored rows plus a manifest to
    `out_dir`, so the printed numbers can be reproduced later from the saved
    CSV rather than from whatever the live database holds by then.
    """
    seasons = tuple(seasons)
    is_default_window = tuple(sorted(seasons)) == tuple(sorted(DEFAULT_SEASONS))
    requested_excluded = sorted(set(seasons) & set(EXCLUDED_SEASONS))

    kept_seasons = sorted(set(seasons) - set(EXCLUDED_SEASONS))
    db_path = Path(db_path).resolve()
    placeholders = ",".join("?" for _ in kept_seasons) or "NULL"
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        games = pd.read_sql_query(f"SELECT * FROM games WHERE season IN ({placeholders}) ORDER BY game_id",
                                  conn, params=kept_seasons)
    input_hash = hashlib.sha256(games.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    matchup = build_matchup(games)
    windowed = select_window(matchup, seasons)
    rows = scored_rows(windowed)

    exclusions = exclusion_report(games)

    report = per_season_report(rows)
    paired_all = rows[rows["market_home_probability"].notna()]
    calib = calibration_bins(
        paired_all["home_win"].astype(float).to_numpy(),
        paired_all["market_home_probability"].to_numpy(),
    )

    manifest = {
        "diagnostic": "stage0",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "input_sha256": input_hash,
        "input_hash_encoding": "Selected games ordered by game_id; pandas CSV, UTF-8, LF, empty nulls; pandas version below",
        "git": _git_info(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "seasons_requested": sorted(seasons),
        "seasons_scored": kept_seasons,
        "seasons_unconditionally_excluded": list(EXCLUDED_SEASONS),
        "is_default_window": is_default_window,
        "requested_seasons_that_were_excluded_anyway": requested_excluded,
        "target_rule": (
            "home_win: REG season, dated, completed (explicit `completed` column "
            "when present, else a scores-present proxy), non-tied games only; "
            "see src/features/rolling_features.py::_compute_labels"
        ),
        "metric_settings": {
            "log_loss_eps": 1e-15,
            "accuracy_threshold": 0.5,
            "calibration_bins": 10,
        },
    }

    if out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(DIAGNOSTICS_DIR) / run_id
    paths = save_diagnostic(rows, exclusions, manifest, out_dir)

    return {
        "manifest": manifest,
        "report": report,
        "calibration": calib,
        "paths": paths,
    }


def print_report(result: dict) -> None:
    """Per-season and pooled table as printed to the terminal. Display only;
    every number comes from the saved scored rows, not recomputed here."""
    manifest = result["manifest"]
    print("=== Stage 0 exploratory diagnostic on development data -- NOT final evaluation ===")
    seasons = manifest["seasons_scored"]
    print(f"selected seasons: {', '.join(map(str, seasons)) or 'none'} ({len(seasons)} seasons)")
    if not manifest["is_default_window"]:
        print("ALTERNATIVE WINDOW -- not the 2005-2024 default")
    if manifest["requested_seasons_that_were_excluded_anyway"]:
        print(f"WARNING: requested seasons {manifest['requested_seasons_that_were_excluded_anyway']} "
              f"were dropped -- 2025/2026 are excluded unconditionally")
    print(f"db: {manifest['db_path']} (selected-input sha256 {manifest['input_sha256'][:12]}...)")
    print(f"git commit: {manifest['git']['commit']} (dirty={manifest['git']['dirty']})")
    print()
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(result["report"].to_string(index=False))
    print()
    print("calibration (market_home_probability, pooled paired rows):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(result["calibration"].to_string(index=False))
    print()
    print(f"scored rows saved to: {result['paths']['rows_path']}")
    print(f"eligibility counts saved to: {result['paths']['exclusions_path']}")
    print(f"manifest saved to: {result['paths']['manifest_path']}")


def main(argv=None) -> None:
    """CLI entry point. `--db-path` points the diagnostic at another cache;
    the module docstring is what `--help` prints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default=None,
                         help="Season range 'START-END' (inclusive). Default: 2005-2024. "
                              "2025/2026 are excluded unconditionally regardless of this flag.")
    parser.add_argument("--db-path", default=GAMES_DB_PATH)
    args = parser.parse_args(argv)

    if args.seasons:
        start, end = (int(x) for x in args.seasons.split("-"))
        seasons = tuple(range(start, end + 1))
    else:
        seasons = DEFAULT_SEASONS

    result = run_diagnostic(seasons=seasons, db_path=args.db_path)
    print_report(result)


if __name__ == "__main__":
    main()
