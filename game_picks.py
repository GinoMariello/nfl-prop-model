"""
Refresh the game-outcome model's picks for the current season, for the Prop Board.

    python3 game_picks.py 2026      # weekly_props.py runs this

1. Re-pulls the season's schedule and play-by-play (src.data.ingest) and rebuilds
   team_game_stats -- so finished games feed the rolling features.
2. Rebuilds the matchup table from the live data/nfl_games.db with the backtest's own
   builder (src.evaluation.backtest.matchups_from_tables). No snapshot: snapshots are
   ~330 MB each and are the backtests' permanent records, not a weekly cache.
3. Scores every game of the season with the pinned run's models and writes
   projections/game_model/<season>_<variant>.csv, which site_data.py reads.

Pinned to one run and fold so the picks do not silently change model. Change them
deliberately after a retrain (live_fit.py prints the new run directory).
"""
import sqlite3
import sys
from contextlib import closing

import pandas as pd

from src.data.ingest import GAMES_DB_PATH, run_ingestion
from src.evaluation.backtest import matchups_from_tables
from src.features import team_game_stats
from src.models.predict import load_saved_model, predict_games

# Live fit (live_fit.py): C chosen on 2025, refit on 1999-2025, no 2026 row read.
RUN = "models_game/home_win/live_20261011T022832Z"
FOLD = 2026
VARIANTS = ["team_only", "team_residual"]


def main(season: int) -> None:
    run_ingestion([season])
    team_game_stats.run()
    with closing(sqlite3.connect(GAMES_DB_PATH)) as conn:
        games = pd.read_sql_query("SELECT * FROM games", conn)
        stats = pd.read_sql_query("SELECT * FROM team_game_stats", conn)
    matchups = matchups_from_tables(games, stats)
    matchups = matchups.loc[matchups["home_season"].eq(season)]
    for variant in VARIANTS:
        preds = predict_games(load_saved_model(RUN, variant, FOLD), matchups)
        out = f"projections/game_model/{season}_{variant}.csv"
        preds.to_csv(out, index=False)
        print(f"game model {variant}: {int(preds.p_home.notna().sum())} of {len(preds)} games scored -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
