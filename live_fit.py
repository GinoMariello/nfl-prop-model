"""
Fit the game model for live picks in one season -- not an evaluation.

    python3 live_fit.py 2026        # writes models_game/home_win/live_<stamp>/

The existing fold rule, for T = the season being picked (the same rule
docs/experiment-protocol.md fixes for T = 2027): select C by validation log loss on
T-1 with candidates fitted on 1999..T-2, then refit on 1999..T-1. Nothing from season
T is read -- not a feature row, not an outcome -- so no development metric on T exists
or is implied, and partial 2026 stays out of the development window as the protocol
requires. The run is saved in the backtest's artifact layout (manifest with
checksums) so src.models.predict and game_picks.py load it unchanged.
"""
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.ingest import GAMES_DB_PATH
from src.evaluation.backtest import (C_GRID, TARGET, TRAIN_START, VARIANT_C_GRID, _eligible,
                                     _season_slice, matchups_from_tables, select_c)
from src.evaluation.diagnose import _sha256_file
from src.models.train import fit_model, save_model

VARIANTS = ["team_only", "team_residual"]


def main(season: int) -> Path:
    with closing(sqlite3.connect(GAMES_DB_PATH)) as conn:
        games = pd.read_sql_query("SELECT * FROM games", conn)
        stats = pd.read_sql_query("SELECT * FROM team_game_stats", conn)
    eligible = _eligible(matchups_from_tables(games, stats))
    eligible = eligible.loc[eligible["home_season"] < season].reset_index(drop=True)  # never season T itself
    train, valid = _season_slice(eligible, TRAIN_START, season - 2), _season_slice(eligible, season - 1, season - 1)
    refit = _season_slice(eligible, TRAIN_START, season - 1)

    run_id = "live_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("models_game") / TARGET / run_id
    manifest = {
        "run_id": run_id, "target": TARGET,
        "study_type": f"live fit for {season} picks -- NOT an evaluation; no {season} row was read",
        "fold_rule": f"C selected on {season - 1} (candidates fit {TRAIN_START}-{season - 2}); refit {TRAIN_START}-{season - 1}",
        "games_db_sha256": _sha256_file(GAMES_DB_PATH),
        "selection": {}, "model_sha256": {},
    }
    for variant in VARIANTS:
        picked = select_c(train, valid, variant, c_grid=VARIANT_C_GRID.get(variant, C_GRID))
        bundle = fit_model(refit, variant=variant, C=picked["C"])
        path = save_model(bundle, run_dir / variant / "logistic" / f"season_{season}.joblib")
        manifest["model_sha256"][path.relative_to(run_dir).as_posix()] = _sha256_file(path)
        best = min(picked["candidates"], key=lambda c: c["valid_log_loss"])
        manifest["selection"][variant] = {"C": picked["C"], "valid_log_loss": best["valid_log_loss"],
                                          "n_train": bundle["n_train"], "converged": bundle["converged"]}
        print(f"{variant}: C={picked['C']} (validation log loss {best['valid_log_loss']:.4f} on {season - 1}), "
              f"refit on {bundle['n_train']} games {TRAIN_START}-{season - 1}")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {run_dir}")
    return run_dir


if __name__ == "__main__":
    main(int(sys.argv[1]))
