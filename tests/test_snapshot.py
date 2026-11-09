"""
Offline tests for src/data/snapshot.py (docs/next-implementations.md Stage 1
acceptance criteria): snapshot immutability, checksum tamper detection,
idempotent re-capture, and content_hash's order-independence / value-
sensitivity. No network; builds tiny synthetic SQLite databases.

Run: python3 tests/test_snapshot.py
"""
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.snapshot import (
    create_snapshot, verify_snapshot, content_hash, open_snapshot,
    snapshot_db_path, snapshot_dir, PIPELINE_PACKAGES,
)


def _make_db(path, home_scores=(20, 17)):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE games (game_id TEXT, season INT, home_score INT)")
        conn.executemany("INSERT INTO games VALUES (?, ?, ?)",
                          [(f"g{i}", 2020 + i, s) for i, s in enumerate(home_scores)])
        conn.execute("CREATE TABLE plays (game_id TEXT, play_id INT)")
        conn.executemany("INSERT INTO plays VALUES (?, ?)", [("g0", 1), ("g0", 2)])


def demo_create_snapshot_captures_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        _make_db(db)
        manifest = create_snapshot(db_path=str(db), snapshots_dir=str(Path(tmp) / "snapshots"))

        assert manifest["table_counts"] == {"games": 2, "plays": 2}
        assert manifest["season_coverage"]["games"] == {"min_season": 2020, "max_season": 2021, "distinct_seasons": 2}
        assert "plays" not in manifest["season_coverage"]  # no season column there
        assert manifest["source_sha256"] == manifest["snapshot_sha256"]
        assert manifest["python_version"] and manifest["platform"]

        # model-plan.md section 8: "Python/library versions". Every package the
        # pipeline runs on, including the modelling library the models that
        # read this snapshot will be fitted with.
        versions = manifest["library_versions"]
        assert set(versions) == set(PIPELINE_PACKAGES), \
            "the manifest must record a version slot for every pipeline package"
        for required in ("pandas", "numpy", "scikit-learn"):
            assert versions[required], f"{required} version missing from the manifest"
        # An absent package records None rather than vanishing -- the manifest
        # must keep distinguishing "known absent" from "never recorded".
        assert all(v is None or isinstance(v, str) for v in versions.values())

        # model-plan.md section 5: the manifest must label what kind of study
        # this is, so nothing downstream reports it as a verified as-of replay.
        assert manifest["study_type"] == "historical snapshot study"
        assert "as-of replay" in manifest["study_type_basis"]
        assert manifest["historical_retrieved_utc"] is None, \
            "retrieval time is unknown and must stay explicitly unknown"

        print("OK -- create_snapshot captures correct table counts, season coverage, "
              "library versions, and provenance.")


def demo_repeat_capture_is_idempotent_and_source_correction_does_not_change_it():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        snapshots_dir = str(Path(tmp) / "snapshots")
        _make_db(db, home_scores=(20, 17))
        first = create_snapshot(db_path=str(db), snapshots_dir=snapshots_dir)

        # Re-running against the unchanged source returns the same snapshot, no error.
        second = create_snapshot(db_path=str(db), snapshots_dir=snapshots_dir)
        assert first == second
        assert len(list(Path(snapshots_dir).iterdir())) == 1

        # "Correcting" the live source (a later score fix) must not alter the
        # already-captured snapshot -- it must produce a *new*, separate one.
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE games SET home_score = 99 WHERE game_id = 'g0'")
        corrected = verify_snapshot(Path(snapshots_dir) / list(Path(snapshots_dir).iterdir())[0].name)
        assert corrected["table_counts"] == first["table_counts"]  # untouched

        third = create_snapshot(db_path=str(db), snapshots_dir=snapshots_dir)
        assert third["source_sha256"] != first["source_sha256"]
        assert len(list(Path(snapshots_dir).iterdir())) == 2  # old snapshot preserved alongside new one

        print("OK -- unchanged source re-capture is idempotent; a source correction leaves the "
              "original snapshot untouched and creates a separate new one.")


def demo_verify_snapshot_detects_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        snapshots_dir = str(Path(tmp) / "snapshots")
        _make_db(db)
        manifest = create_snapshot(db_path=str(db), snapshots_dir=snapshots_dir)
        out_dir = snapshot_dir(snapshots_dir, manifest["source_sha256"])

        verify_snapshot(out_dir)  # untouched: passes

        db_copy = out_dir / "nfl_games.db"
        db_copy.chmod(0o644)
        with open(db_copy, "ab") as f:
            f.write(b"tampered")
        try:
            verify_snapshot(out_dir)
            assert False, "expected ValueError for a modified snapshot file"
        except ValueError as exc:
            assert "tampering" in str(exc).lower()

        print("OK -- verify_snapshot detects a modified snapshot file as tampering.")


def demo_wal_committed_data_is_captured():
    # The bug this replaced: shutil.copy2 of the main .db file alone drops
    # everything still living in the -wal sidecar, even though it is
    # committed. A snapshot would then "verify" perfectly against its own
    # checksum while silently missing data -- the exact failure Stage 1
    # exists to prevent.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        snapshots_dir = Path(tmp) / "snapshots"
        with sqlite3.connect(db) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE games (game_id TEXT, season INT)")
            conn.execute("INSERT INTO games VALUES ('g1', 2020)")
            conn.execute("CREATE TABLE plays (game_id TEXT, play_id INT)")
            conn.execute("INSERT INTO plays VALUES ('g1', 1)")
            conn.commit()  # committed, but still in games.db-wal

        assert (Path(tmp) / "games.db-wal").exists(), "test needs the data to actually be in the WAL"

        manifest = create_snapshot(db_path=str(db), snapshots_dir=str(snapshots_dir))
        assert manifest["table_counts"] == {"games": 1, "plays": 1}, \
            f"committed WAL data missing from snapshot: {manifest['table_counts']}"
        assert manifest["season_coverage"]["games"]["min_season"] == 2020

    print("OK -- committed WAL data is captured; a snapshot never silently drops it.")


def demo_failed_capture_leaves_no_stuck_directory():
    # A crash mid-capture (here: disk-full during the copy) must not leave a
    # half-written directory at the real content-addressed path. If it did,
    # every later attempt for that same content would short-circuit into
    # verify_snapshot and fail forever on a missing manifest, with no way to
    # recover except deleting the directory by hand.
    import src.data.snapshot as snapshot_module

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        snapshots_dir = Path(tmp) / "snapshots"
        _make_db(db)

        # Fail right after the backup lands in the temp dir, so there is a
        # full-looking db sitting there when the error hits.
        real_sha256 = snapshot_module._sha256_file

        def exploding_sha256(path):
            raise OSError(28, "No space left on device")

        snapshot_module._sha256_file = exploding_sha256
        try:
            create_snapshot(db_path=str(db), snapshots_dir=str(snapshots_dir))
            assert False, "expected the simulated disk-full error to propagate"
        except OSError:
            pass
        finally:
            snapshot_module._sha256_file = real_sha256

        leftovers = list(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
        assert leftovers == [], f"failed capture left directories behind: {leftovers}"

        # And the retry, once the failure is gone, still works normally.
        manifest = create_snapshot(db_path=str(db), snapshots_dir=str(snapshots_dir))
        assert manifest["table_counts"] == {"games": 2, "plays": 2}

    print("OK -- a failed capture leaves no stuck snapshot directory, and a later retry succeeds.")


def demo_odd_databases_and_corrupt_manifests():
    with tempfile.TemporaryDirectory() as tmp:
        snapshots_dir = str(Path(tmp) / "snapshots")

        # Table names are read back out of sqlite_master and pasted into
        # COUNT(*)/PRAGMA. A name containing a quote used to produce a syntax
        # error, so identifiers must be quoted and escaped, not interpolated.
        weird = Path(tmp) / "weird.db"
        with sqlite3.connect(weird) as conn:
            conn.execute("CREATE TABLE \"it's a table\" (season INT)")
            conn.execute("INSERT INTO \"it's a table\" VALUES (2020)")
            conn.execute('CREATE TABLE "quo""ted" (season INT)')
            conn.execute('INSERT INTO "quo""ted" VALUES (2021)')
        manifest = create_snapshot(db_path=str(weird), snapshots_dir=snapshots_dir)
        assert manifest["table_counts"] == {"it's a table": 1, 'quo"ted': 1}
        assert manifest["season_coverage"]["it's a table"]["min_season"] == 2020

        # A database with no tables at all, and a table with no rows, are both
        # legitimate empty states, not errors.
        empty = Path(tmp) / "empty.db"
        sqlite3.connect(empty).close()
        assert create_snapshot(db_path=str(empty), snapshots_dir=snapshots_dir)["table_counts"] == {}

        norows = Path(tmp) / "norows.db"
        with sqlite3.connect(norows) as conn:
            conn.execute("CREATE TABLE games (game_id TEXT, season INT)")
        coverage = create_snapshot(db_path=str(norows), snapshots_dir=snapshots_dir)["season_coverage"]
        assert coverage["games"] == {"min_season": None, "max_season": None, "distinct_seasons": 0}

        # A corrupt manifest is a corruption report, not a raw parser error --
        # detecting exactly this is verify_snapshot's job.
        db = Path(tmp) / "ok.db"
        _make_db(db)
        good = create_snapshot(db_path=str(db), snapshots_dir=snapshots_dir)
        out_dir = snapshot_dir(snapshots_dir, good["snapshot_sha256"])
        (out_dir / "manifest.json").chmod(0o644)
        (out_dir / "manifest.json").write_text("{not json")
        try:
            verify_snapshot(out_dir)
            assert False, "expected ValueError for a corrupt manifest"
        except ValueError as exc:
            assert "corrupt" in str(exc).lower()

    print("OK -- quoted/odd table names, empty databases, and corrupt manifests all handled explicitly.")


def demo_concurrent_publish_does_not_clobber():
    # Two captures of identical content race to publish the same
    # content-addressed directory. The loser must dedup onto the winner, never
    # overwrite it -- simulated deterministically by publishing a competing
    # snapshot midway through this capture, after its dedup pre-check.
    import src.data.snapshot as snapshot_module

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        snapshots_dir = Path(tmp) / "snapshots"
        _make_db(db)

        real_table_counts = snapshot_module._table_counts
        competitor = {}
        raced = []

        def publish_competitor_first(path):
            counts = real_table_counts(path)
            if not raced:
                raced.append(True)  # set before recursing, or the inner call re-triggers this
                competitor["manifest"] = snapshot_module.create_snapshot(
                    db_path=str(db), snapshots_dir=str(snapshots_dir))
            return counts

        snapshot_module._table_counts = publish_competitor_first
        try:
            manifest = create_snapshot(db_path=str(db), snapshots_dir=str(snapshots_dir))
        finally:
            snapshot_module._table_counts = real_table_counts

        assert competitor, "test did not actually simulate a competing publish"
        assert manifest["snapshot_sha256"] == competitor["manifest"]["snapshot_sha256"]
        assert manifest["created_utc"] == competitor["manifest"]["created_utc"], \
            "loser must return the winner's manifest, not overwrite it with its own"
        assert len(list(snapshots_dir.iterdir())) == 1, "no temp or duplicate directories left behind"

    print("OK -- a losing concurrent capture dedups onto the published snapshot instead of clobbering it.")


def demo_open_snapshot_cannot_rebuild_a_table():
    """Stage 1: "Training/evaluation must never call a table-replacement entry
    point against the selected snapshot." `open_snapshot` makes that a SQLite
    error rather than a convention.

    Two independent mechanisms reject a write here -- the 0444 file mode and
    the `mode=ro` URI -- so the second half of this test deliberately removes
    the first. Without that, dropping `mode=ro` entirely would leave the test
    green (verified: it does).
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "games.db"
        _make_db(db)
        manifest = create_snapshot(db_path=str(db), snapshots_dir=str(Path(tmp) / "snapshots"))
        out_dir = snapshot_dir(Path(tmp) / "snapshots", manifest["snapshot_sha256"])

        with closing(open_snapshot(out_dir)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2, \
                "reads must work through the read-only door"

            # Every shape of table replacement the plan names.
            for statement in ("DROP TABLE games",
                              "CREATE TABLE rolling_features (game_id TEXT)",
                              "INSERT INTO games VALUES ('g9', 2030, 0)",
                              "DELETE FROM games"):
                try:
                    conn.execute(statement)
                    assert False, f"a snapshot must reject: {statement}"
                except sqlite3.OperationalError as exc:
                    assert "readonly" in str(exc), f"unexpected error for {statement}: {exc}"

        # The file is unchanged, so the checksum still verifies.
        verify_snapshot(out_dir)

        # Now take the file mode out of the picture: a snapshot restored from a
        # backup, or copied with a tool that does not preserve permissions,
        # arrives writable. The connection must still refuse.
        writable = snapshot_db_path(out_dir)
        writable.chmod(0o644)
        with closing(sqlite3.connect(str(writable))) as plain:
            plain.execute("INSERT INTO games VALUES ('g9', 2030, 0)")   # proves it IS writable now
            plain.rollback()
        with closing(open_snapshot(out_dir)) as conn:
            try:
                conn.execute("DROP TABLE games")
                assert False, "open_snapshot must reject a write even on a writable file"
            except sqlite3.OperationalError as exc:
                assert "readonly" in str(exc), f"unexpected error: {exc}"
        writable.chmod(0o444)
        verify_snapshot(out_dir)    # and nothing above actually changed a byte

        # And it refuses to open something that is not a snapshot.
        try:
            open_snapshot(Path(tmp))
            assert False, "expected FileNotFoundError outside a snapshot directory"
        except FileNotFoundError:
            pass
        assert snapshot_db_path(out_dir).is_file()

    print("OK -- open_snapshot reads fine but rejects DROP/CREATE/INSERT/DELETE, on a "
          "read-only file and on a writable one alike.")


def demo_content_hash_ignores_order_and_detects_value_changes():
    base = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_win": [1, 0, np.nan],
        "market_home_probability": [0.6, 0.4, 0.5],
    })
    shuffled = base.sample(frac=1, random_state=0)[["market_home_probability", "game_id", "home_win"]]
    assert content_hash(base) == content_hash(shuffled), "row/column order must not affect the hash"

    changed_value = base.copy()
    changed_value.loc[0, "home_win"] = 0
    assert content_hash(changed_value) != content_hash(base)

    changed_missing = base.copy()
    changed_missing.loc[2, "home_win"] = 1  # was NaN, now a real 1 -- must differ from both a NaN and a 0
    assert content_hash(changed_missing) not in (content_hash(base), content_hash(changed_value))

    # A missing value must never collide with a real 0.
    zero_not_missing = base.copy()
    zero_not_missing.loc[2, "home_win"] = 0
    assert content_hash(zero_not_missing) != content_hash(base)

    try:
        content_hash(pd.DataFrame({"game_id": ["g1", "g1"], "x": [1, 2]}))
        assert False, "expected ValueError for duplicated id_columns"
    except ValueError:
        pass

    try:
        content_hash(pd.DataFrame({"x": [1]}), id_columns=("game_id",))
        assert False, "expected KeyError for a missing id column"
    except KeyError:
        pass

    # Delimiter safety: a value containing a separator character must not let
    # two different tables hash identically.
    split_left = pd.DataFrame({"game_id": ["g1"], "a": ["x,y"], "b": ["z"]})
    split_right = pd.DataFrame({"game_id": ["g1"], "a": ["x"], "b": ["y,z"]})
    assert content_hash(split_left) != content_hash(split_right)

    # Float precision must be lossless. Fixed-precision formatting (e.g.
    # "%.12g") renders these two distinct float64s identically, silently
    # reporting "unchanged" for data that did change.
    near_a = pd.DataFrame({"game_id": ["g1"], "epa": [0.1234567890123456]})
    near_b = pd.DataFrame({"game_id": ["g1"], "epa": [0.1234567890123999]})
    assert near_a.epa[0] != near_b.epa[0]
    assert content_hash(near_a) != content_hash(near_b), \
        "floats differing beyond 12 significant digits must not collide"

    one_ulp = pd.DataFrame({"game_id": ["g1"], "epa": [np.nextafter(0.1234567890123456, 1)]})
    assert content_hash(one_ulp) != content_hash(near_a), "one-ULP difference must change the hash"

    # A missing value must not collide with a string cell whose text happens
    # to match whatever marker the serializer uses for null.
    for marker in ("<NULL>", "null", "None", "nan"):
        literal = pd.DataFrame({"game_id": ["g1"], "x": [marker]})
        real_null = pd.DataFrame({"game_id": ["g1"], "x": [None]})
        assert content_hash(literal) != content_hash(real_null), \
            f"string {marker!r} must not hash the same as a real missing value"

    # A number and its string form are different data, so different hashes.
    assert content_hash(pd.DataFrame({"game_id": ["g1"], "x": [1]})) != \
           content_hash(pd.DataFrame({"game_id": ["g1"], "x": ["1"]}))

    # Numerically equal int/float/bool share an encoding -- 1, 1.0 and True
    # are the same value, and a dtype change alone is not a data change.
    as_int = pd.DataFrame({"game_id": ["g1"], "x": [1]})
    as_float = pd.DataFrame({"game_id": ["g1"], "x": [1.0]})
    assert content_hash(as_int) == content_hash(as_float)

    # Large integers stay exact (beyond float64's 2**53 contiguous range).
    big = 9007199254740993
    assert content_hash(pd.DataFrame({"game_id": ["g1"], "n": [big], "f": [1.5]})) != \
           content_hash(pd.DataFrame({"game_id": ["g1"], "n": [big + 1], "f": [1.5]}))

    # An unsupported cell type fails loudly rather than being str()'d into
    # something that might collide with a real string.
    try:
        content_hash(pd.DataFrame({"game_id": ["g1"], "x": [{"a": 1}]}))
        assert False, "expected TypeError for an unsupported cell type"
    except TypeError:
        pass

    # A bare string id column is the natural call; it must mean one column,
    # not a list of its characters.
    single = pd.DataFrame({"game_id": ["g1"], "x": [1]})
    assert content_hash(single, id_columns="game_id") == content_hash(single, id_columns=("game_id",))

    # Dates have to hash: this repo's feature/label tables carry gameday.
    # Exact nanoseconds, so a one-second change is a change, and a timestamp
    # never collides with its own string rendering.
    day = pd.DataFrame({"game_id": ["g1"], "gameday": [pd.Timestamp("2020-09-10")]})
    next_day = pd.DataFrame({"game_id": ["g1"], "gameday": [pd.Timestamp("2020-09-11")]})
    one_second = pd.DataFrame({"game_id": ["g1"], "gameday": [pd.Timestamp("2020-09-10 00:00:01")]})
    as_text = pd.DataFrame({"game_id": ["g1"], "gameday": ["2020-09-10"]})
    assert content_hash(day) != content_hash(next_day)
    assert content_hash(day) != content_hash(one_second)
    assert content_hash(day) != content_hash(as_text)

    # NaT is missing, like NaN/None -- not a date value.
    assert content_hash(pd.DataFrame({"game_id": ["g1"], "gameday": [pd.NaT]})) == \
           content_hash(pd.DataFrame({"game_id": ["g1"], "gameday": [None]}))

    print("OK -- content_hash is order-independent, sensitive to real value changes, and "
          "never confuses a missing value with a real zero.")


if __name__ == "__main__":
    demo_create_snapshot_captures_manifest()
    demo_repeat_capture_is_idempotent_and_source_correction_does_not_change_it()
    demo_verify_snapshot_detects_tampering()
    demo_wal_committed_data_is_captured()
    demo_failed_capture_leaves_no_stuck_directory()
    demo_odd_databases_and_corrupt_manifests()
    demo_concurrent_publish_does_not_clobber()
    demo_open_snapshot_cannot_rebuild_a_table()
    demo_content_hash_ignores_order_and_detects_value_changes()
