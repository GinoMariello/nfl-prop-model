"""Offline ingestion compatibility check: python tests/test_dependency_smoke.py.

Exercise the pinned adapter and parquet readers with local fixtures in place
of remote downloads. This checks the project's two ingestion paths, not every
nfl_data_py API or upstream network availability.
"""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import fastparquet
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ingest import fetch_schedules, fetch_pbp, PBP_REQUIRED_COLUMNS


def demo():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        schedule_path = root / "games.csv"
        pd.DataFrame({"game_id": ["g1", "g2"], "season": [2020, 2021]}).to_csv(schedule_path, index=False)
        plays_path = root / "plays.parquet"
        # Carry every column the feature layer reads, not only the three this
        # check asserts on. fetch_pbp now refuses a source missing one of them,
        # and a fixture standing in for play-by-play data should meet the same
        # contract real play-by-play data does.
        play = {column: 0 for column in PBP_REQUIRED_COLUMNS}
        play.update({"game_id": "g1", "play_id": 1, "season": 2020, "epa": 0.25,
                     "posteam": "A", "defteam": "B", "play_type": "pass"})
        pd.DataFrame([play]).to_parquet(plays_path)
        participation_path = root / "participation.parquet"
        pd.DataFrame({"nflverse_game_id": ["g1"], "play_id": [1]}).to_parquet(participation_path)
        read_csv, read_parquet = pd.read_csv, pd.read_parquet

        def local_schedule(url, *args, **kwargs):
            assert url == "http://www.habitatring.com/games.csv"
            return read_csv(schedule_path, *args, **kwargs)

        def local_plays(path, *args, **kwargs):
            if str(path).endswith("/pbp/play_by_play_2020.parquet"):
                source = plays_path
            elif str(path).endswith("/pbp_participation/pbp_participation_2020.parquet"):
                source = participation_path
            else:
                raise AssertionError(f"Unexpected download: {path}")
            return read_parquet(source, *args, **kwargs)

        with patch.object(pd, "read_csv", side_effect=local_schedule):
            assert fetch_schedules([2020]).game_id.tolist() == ["g1"]
        with patch.object(pd, "read_parquet", side_effect=local_plays):
            plays = fetch_pbp([2020])
        assert plays.game_id.tolist() == ["g1"]
        assert plays.season.tolist() == [2020]
        assert plays.epa.dtype == np.float32 and plays.epa.iloc[0] == 0.25
        assert fastparquet.ParquetFile(plays_path).to_pandas().epa.iloc[0] == 0.25
    print("OK -- local schedule/parquet ingestion, participation merge, downcast, and modeling imports work.")


def demo_pinned_packages_are_usable_not_merely_installed():
    """A package can install cleanly and still be dead. `pip install --no-deps`
    applied to fastparquet (rather than to nfl_data_py alone, the only package
    needing the override) left it importable-looking but broken at first use:
    "ModuleNotFoundError: No module named 'fsspec'". The ingestion path above
    went through pyarrow, so nothing caught it."""
    frame = pd.DataFrame({
        "game_id": ["2023_01_DET_KC", "2023_01_CAR_ATL"],
        "epa": [0.1234567890123456, -0.5],
        "home_win": [1, 0],
    })
    with tempfile.TemporaryDirectory() as tmp:
        for engine in ("pyarrow", "fastparquet"):
            path = Path(tmp) / f"{engine}.parquet"
            frame.to_parquet(path, engine=engine)
            back = pd.read_parquet(path, engine=engine)
            assert back.game_id.tolist() == frame.game_id.tolist(), f"{engine} round-trip corrupted data"
            assert back.epa[0] == frame.epa[0], f"{engine} round-trip lost float precision"

    # Fit, don't just import -- importing alone missed a missing scipy once.
    model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
    assert np.isfinite(model.coef_).all()
    assert model.predict_proba(np.array([[1.5]])).shape == (1, 2)

    # nfl_data_py is installed with --no-deps against a pandas it declares it
    # does not support, so confirm its two entry points still resolve.
    import nfl_data_py
    for name in ("import_schedules", "import_pbp_data"):
        assert callable(getattr(nfl_data_py, name)), f"nfl_data_py.{name} missing"

    print("OK -- both parquet engines round-trip, sklearn fits, and nfl_data_py's entry points resolve.")


if __name__ == "__main__":
    demo()
    demo_pinned_packages_are_usable_not_merely_installed()
