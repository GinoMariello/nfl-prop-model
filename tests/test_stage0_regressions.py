"""Offline Stage 0 regressions: python -m unittest discover -s tests -p test_stage0_regressions.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.diagnose import run_diagnostic, print_report, _git_info
from src.evaluation.metrics import log_loss


class Stage0Regressions(unittest.TestCase):
    def test_schedule_only_database_and_saved_replay(self):
        # No derived tables: stale/missing team aggregates must not select the population.
        games = pd.DataFrame([
            dict(game_id="g1", season=2020, week=1, game_type="REG",
                 gameday="2020-09-10", home_team="KC", away_team="HOU",
                 home_score=34, away_score=20, home_moneyline=-200, away_moneyline=200),
            dict(game_id="g2", season=2025, week=1, game_type="REG",
                 gameday="2025-09-10", home_team="KC", away_team="HOU",
                 home_score=20, away_score=34, home_moneyline=-200, away_moneyline=200),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "games.db"
            with sqlite3.connect(db) as conn:
                games.to_sql("games", conn, index=False)
            before = db.read_bytes()
            result = run_diagnostic(seasons=[2020, 2025], db_path=str(db), out_dir=Path(tmp) / "run")
            self.assertEqual(db.read_bytes(), before)
            rows = pd.read_csv(result["paths"]["rows_path"], float_precision="round_trip")
            self.assertEqual(rows.game_id.tolist(), ["g1"])
            self.assertAlmostEqual(rows.market_home_probability.iloc[0], 2 / 3)
            self.assertEqual(result["report"].iloc[-1].n_full, 1)
            counts = pd.read_csv(result["paths"]["exclusions_path"])
            self.assertEqual(counts.query("reason == 'supervised_eligible'")["count"].sum(), 1)
            manifest = json.loads(result["paths"]["manifest_path"].read_text())
            self.assertIn("input_sha256", manifest)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE games SET home_score = 99 WHERE season = 2025")
            repeat = run_diagnostic(seasons=[2020, 2025], db_path=str(db), out_dir=Path(tmp) / "repeat")
            pd.testing.assert_frame_equal(result["report"], repeat["report"])
            self.assertEqual(result["manifest"]["input_sha256"], repeat["manifest"]["input_sha256"])
            empty = run_diagnostic(seasons=[2025, 2026], db_path=str(db), out_dir=Path(tmp) / "empty")
            self.assertEqual(empty["report"].iloc[-1].n_full, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                print_report(empty)

    def test_empty_window_prints_without_crashing(self):
        result = {"manifest": {"seasons_scored": [], "is_default_window": False,
                  "requested_seasons_that_were_excluded_anyway": [2025, 2026],
                  "db_path": "test.db", "db_sha256": "abc", "input_sha256": "abc",
                  "git": {"commit": "test", "dirty": False}},
                  "report": pd.DataFrame([{"season": "pooled", "n_full": 0}]),
                  "calibration": pd.DataFrame(),
                  "paths": dict(rows_path="rows", exclusions_path="counts", manifest_path="manifest")}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_report(result)
        self.assertIn("pooled", output.getvalue())
        self.assertIn("0 seasons", output.getvalue())

    def test_zero_epsilon_boundaries(self):
        with np.errstate(divide="raise", invalid="raise"):
            self.assertEqual(log_loss([1, 0], [1, 0], eps=0), 0)
            self.assertEqual(log_loss([1, 0], [0, 1], eps=0), float("inf"))

    def test_git_provenance_preserves_staged_unstaged_and_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def git(*args):
                return subprocess.run(["git", "-C", tmp, *args], check=True, capture_output=True, text=True)
            git("init")
            source = repo / "example.py"
            source.write_text("value = 1\n")
            git("add", ".")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial")
            source.write_text("value = 2\n")
            git("add", ".")
            source.write_text("value = 3\n")
            (repo / "new.py").write_text("new_value = 4\n")
            info = _git_info(repo)
            self.assertTrue(info["dirty"])
            self.assertIn("+value = 3", info["diff"])
            self.assertEqual(info["untracked_sources"]["new.py"], "new_value = 4\n")


if __name__ == "__main__":
    unittest.main()
