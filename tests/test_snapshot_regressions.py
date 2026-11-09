"""Offline snapshot regressions: unittest discover -s tests -p test_snapshot_regressions.py."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.snapshot import create_snapshot, content_hash, verify_snapshot


class SnapshotRegressions(unittest.TestCase):
    def test_committed_wal_rows_are_captured_and_corrections_get_new_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "source.db"
            snapshots = Path(tmp) / "snapshots"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE games (game_id TEXT, season INTEGER)")
                conn.commit()
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA wal_autocheckpoint=0")
                conn.execute("INSERT INTO games VALUES ('g1', 2020)")
                conn.commit()
                first = create_snapshot(db, snapshots)
                self.assertEqual(first["table_counts"]["games"], 1)
                self.assertEqual(create_snapshot(db, snapshots), first)
                conn.execute("INSERT INTO games VALUES ('g2', 2021)")
                conn.commit()
                second = create_snapshot(db, snapshots)
                self.assertEqual(second["table_counts"]["games"], 2)
                self.assertNotEqual(first["snapshot_sha256"], second["snapshot_sha256"])
                for folder in snapshots.iterdir():
                    verify_snapshot(folder)
            finally:
                conn.close()

    def test_hash_preserves_float_precision(self):
        def frame(value):
            return pd.DataFrame({"game_id": ["g1"], "p": [value]})
        self.assertNotEqual(content_hash(frame(0.6000000000001)), content_hash(frame(0.6000000000002)))
        self.assertEqual(content_hash(frame(1)), content_hash(frame(1.0)))

    def test_null_markers_and_separator_text_cannot_collide(self):
        def frame(a, b="z"):
            return pd.DataFrame({"game_id": ["g1"], "a": [a], "b": [b]})
        self.assertNotEqual(content_hash(frame(None)), content_hash(frame("<NULL>")))
        self.assertNotEqual(content_hash(frame("x\x1fy")), content_hash(frame("x", "y\x1fz")))
        self.assertNotEqual(content_hash(frame(1)), content_hash(frame("1")))

    def test_numeric_id_does_not_coerce_large_integer_cells_to_float(self):
        a = pd.DataFrame({"game_id": [1], "value": [2**53], "p": [0.5]})
        b = a.copy()
        b["value"] = np.array([2**53 + 1], dtype=np.int64)
        self.assertNotEqual(content_hash(a), content_hash(b))


if __name__ == "__main__":
    unittest.main()
