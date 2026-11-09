"""
Immutable raw-data snapshots (docs/next-implementations.md Stage 1).

A snapshot is a consistent SQLite backup of the raw `games`/`plays`
database (online backup API, so committed WAL data is included -- a plain
file copy silently drops it), named by the sha256 of its own content, plus a manifest
recording table counts, season coverage, and environment provenance. Once
written, both files are chmod'd read-only and never rewritten -- a later
correction to the live database produces a *new* snapshot (new content ->
new hash -> new directory); it cannot alter one already captured.
Re-running `create_snapshot` against unchanged source content returns the
existing snapshot instead of copying again (`verify_snapshot`'s checksum
check doubles as tamper detection either way).

Content-hashing a feature/label table (`content_hash`) is a separate,
smaller concern: a deterministic fingerprint for reproducibility checks
(e.g. "did this backtest run see the same rows as last time"), independent
of row order and of dtype/pandas-version formatting differences.

This module only captures/reads bytes -- it computes no features and
enforces no leakage rules itself. Per Stage 1, training/evaluation code
must read a snapshot's `games`/`plays` tables directly (read-only) and
must never call a table-replacement entry point
(`team_game_stats.run()`/`rolling_features.run()`) against it; those
functions upsert into the *live* `GAMES_DB_PATH`, not a snapshot file.
`open_snapshot()` is the read path, and it enforces that rather than
asking callers to remember it: the connection is `mode=ro`, so any write
raises. `tests/test_db_path_discipline.py` covers the live database,
which has no such protection.

Run: python3 -m src.data.snapshot
"""
import errno
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

GAMES_DB_PATH = "data/nfl_games.db"
SNAPSHOTS_DIR = "snapshots"
SNAPSHOT_DB_FILENAME = "nfl_games.db"
MANIFEST_FILENAME = "manifest.json"
DATA_SOURCE = "nflverse via nfl_data_py (import_schedules/import_pbp_data)"
MIN_FREE_SPACE_MULTIPLE = 1.5  # require this many times the source db size free before copying
# Hex characters of the content hash used as the snapshot's directory name. 16
# is 64 bits -- far past collision range for a handful of snapshots per season,
# and short enough to type. Go through `snapshot_dir()` rather than slicing a
# hash at the call site, so this stays one decision rather than four.
SNAPSHOT_ID_LENGTH = 16

# Every package the game pipeline runs on, by distribution name. Listed rather
# than hand-written into the manifest one key at a time: the modelling library
# was missing for three stages that way.
PIPELINE_PACKAGES = ("pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "nfl_data_py")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quote_identifier(name: str) -> str:
    """SQLite identifier quoting. Table names come from sqlite_master, not a
    caller, but they still can't be pasted into SQL raw: a name containing a
    quote (`it's a table`) produced a syntax error, and identifier
    interpolation is the wrong shape regardless of who supplies it."""
    return '"' + name.replace('"', '""') + '"'


def _table_names(db_path) -> list:
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


def _table_counts(db_path) -> dict:
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        return {name: conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(name)}").fetchone()[0]
                for name in _table_names(db_path)}


def _season_coverage(db_path) -> dict:
    """Min/max/distinct-count season per table that has a `season` column.
    Missing on purpose for a table without one (e.g. a future non-seasonal
    table) rather than raising."""
    coverage = {}
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        for name in _table_names(db_path):
            quoted = _quote_identifier(name)
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")}
            if "season" not in columns:
                continue
            lo, hi, n = conn.execute(
                f"SELECT MIN(season), MAX(season), COUNT(DISTINCT season) FROM {quoted}").fetchone()
            coverage[name] = {"min_season": lo, "max_season": hi, "distinct_seasons": n}
    return coverage


def _library_versions() -> dict:
    """Versions of the packages the pipeline runs on (model-plan.md section 8:
    "Python/library versions").

    Read from installed distribution metadata rather than by importing each
    package -- this module has no other reason to import scikit-learn, and a
    manifest should not pay a second of import time per capture.

    A package that is not installed records None rather than being left out,
    so the manifest keeps distinguishing "known to be absent" from "never
    recorded" (next-implementations.md Stage 1 acceptance).
    """
    versions = {}
    for name in PIPELINE_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot_dir(snapshots_dir, sha256: str) -> Path:
    """Content-addressed directory a snapshot with this hash lives in.

    Identical content therefore lands on the identical path, which is what
    makes re-capture a no-op and makes a concurrent capture dedup instead of
    clobber. Callers that hold a manifest use
    `snapshot_dir(snapshots_dir, manifest["snapshot_sha256"])` rather than
    slicing the hash themselves.
    """
    return Path(snapshots_dir) / sha256[:SNAPSHOT_ID_LENGTH]


def _check_free_space(db_path, snapshots_dir) -> None:
    # page_count * page_size, not the main file's size on disk: in WAL mode
    # committed pages can still live in the -wal file, so st_size understates
    # what the backup will actually write.
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        needed = (conn.execute("PRAGMA page_count").fetchone()[0]
                  * conn.execute("PRAGMA page_size").fetchone()[0] * MIN_FREE_SPACE_MULTIPLE)
    probe = Path(snapshots_dir).resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed:
        raise OSError(
            f"Not enough free disk space for a snapshot copy: need ~{needed / 1e6:.0f}MB, "
            f"have {free / 1e6:.0f}MB free. Free up space before snapshotting."
        )


def create_snapshot(db_path=GAMES_DB_PATH, snapshots_dir=SNAPSHOTS_DIR) -> dict:
    """Copy `db_path` into a content-addressed snapshot directory and return
    its manifest. Idempotent: unchanged source content returns the existing
    snapshot (verified against its recorded checksum) instead of copying
    again. Raises if a same-named snapshot exists but its file has been
    altered since capture (tampering) -- see `verify_snapshot`.

    Built in a temporary sibling directory and moved into place only once
    the copy is verified and the manifest is written -- a crash or disk-full
    error mid-copy must never leave a permanent, unrecoverable half-written
    snapshot directory sitting at the real content-addressed path.

    Captured with SQLite's online backup API, never a file copy: in WAL mode
    committed transactions can still live in the `-wal` sidecar, so copying
    the main file alone silently drops them (verified: a plain copy of a
    one-row WAL database produced a snapshot with no table at all). The
    snapshot is therefore hashed *after* the backup completes -- the backup's
    bytes, not the live main file's, are what gets content-addressed."""
    resolved_db_path = Path(db_path).resolve()
    if not resolved_db_path.is_file():
        raise FileNotFoundError(f"No database at {resolved_db_path}")

    _check_free_space(resolved_db_path, snapshots_dir)

    Path(snapshots_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".tmp_snapshot_", dir=snapshots_dir))
    try:
        db_copy = tmp_dir / SNAPSHOT_DB_FILENAME
        with closing(sqlite3.connect(f"{resolved_db_path.as_uri()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(db_copy)) as destination:
                source.backup(destination)
                destination.execute("PRAGMA journal_mode=DELETE")
        copy_sha256 = _sha256_file(db_copy)

        # Only knowable now: the live main file's bytes aren't the logical
        # content, so a re-capture must back up first, then dedup on the result.
        out_dir = snapshot_dir(snapshots_dir, copy_sha256)
        if out_dir.exists():
            return verify_snapshot(out_dir)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_db_path": str(resolved_db_path),
            "source_sha256": copy_sha256,
            "source_hash_scope": "consistent SQLite backup including committed WAL data; not the live main file",
            "snapshot_sha256": copy_sha256,
            "snapshot_db_bytes": db_copy.stat().st_size,
            "table_counts": _table_counts(db_copy),
            "season_coverage": _season_coverage(db_copy),
            "data_source": DATA_SOURCE,
            "source_urls": ["https://github.com/nflverse/nflverse-data"],
            "source_release": None,
            "historical_retrieved_utc": None,  # unknown: capture time is not retrieval time
            # model-plan.md section 5: "Label these as historical snapshot
            # studies, not exact as-of replays." The label belongs in the
            # manifest, not only in a design doc -- this file is what travels
            # with the data and what the Stage 5 report reads.
            "study_type": "historical snapshot study",
            "study_type_basis": (
                "Upstream values may have been corrected after the fact, which does not "
                "establish what was published before each kickoff. A verified as-of replay "
                "additionally requires recorded publication/receipt times, which this "
                "snapshot does not carry (model-plan.md section 5)."
            ),
            "upstream_field_model_versions": None,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "library_versions": _library_versions(),
        }
        manifest_path = tmp_dir / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        db_copy.chmod(0o444)
        manifest_path.chmod(0o444)
        try:
            os.rename(tmp_dir, out_dir)  # atomic on the same filesystem
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            # Another capture of identical content published first. rename
            # refuses to replace a non-empty directory, so the existing
            # snapshot is never clobbered -- dedup onto it instead of failing.
            return verify_snapshot(out_dir)
    finally:
        # Also covers the dedup early-return, where tmp_dir is a throwaway backup.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return manifest


def verify_snapshot(out_dir) -> dict:
    """Recompute the snapshot copy's checksum and compare to its manifest.
    Raises ValueError on mismatch (the copy was modified after capture --
    tampering, not a legitimate new snapshot, which would live in a
    different content-addressed directory)."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_FILENAME
    db_copy = out_dir / SNAPSHOT_DB_FILENAME
    if not manifest_path.is_file() or not db_copy.is_file():
        raise FileNotFoundError(f"Not a snapshot directory (missing manifest/db): {out_dir}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        # Corruption detection is this function's job -- report it as such
        # rather than leaking a bare parser error from an unreadable manifest.
        raise ValueError(f"Snapshot manifest at {manifest_path} is corrupt: {exc}") from exc
    if "snapshot_sha256" not in manifest:
        raise ValueError(f"Snapshot manifest at {manifest_path} has no snapshot_sha256")
    current_sha256 = _sha256_file(db_copy)
    if current_sha256 != manifest["snapshot_sha256"]:
        raise ValueError(
            f"Snapshot tampering detected at {out_dir}: recorded checksum "
            f"{manifest['snapshot_sha256'][:12]}... but file now hashes to {current_sha256[:12]}..."
        )
    return manifest


def snapshot_db_path(out_dir) -> Path:
    """Path to a snapshot's own database file, for a read-only connection.
    Never open it for writing -- Stage 1 forbids running
    `team_game_stats.run()`/`rolling_features.run()` (or any other
    table-replacement entry point) against a snapshot."""
    return Path(out_dir) / SNAPSHOT_DB_FILENAME


def open_snapshot(out_dir):
    """Read-only connection to a snapshot's database -- the one door in.

    Stage 1 requires that "training/evaluation must never call a
    table-replacement entry point against the selected snapshot"
    (next-implementations.md). A `mode=ro` URI connection makes that a SQLite
    error instead of a convention: any CREATE, DROP or INSERT raises
    `sqlite3.OperationalError: attempt to write a readonly database`.

    Does not re-verify the checksum: that is a full re-read of the file, too
    expensive to repeat per connection. Call `verify_snapshot(out_dir)` when
    you want the tamper check -- once per run, not once per query.
    """
    db = snapshot_db_path(out_dir)
    if not db.is_file():
        raise FileNotFoundError(f"Not a snapshot directory (no {SNAPSHOT_DB_FILENAME}): {out_dir}")
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def content_hash(rows: pd.DataFrame, id_columns=("game_id",)) -> str:
    """Deterministic sha256 over a feature/label table: order-independent
    (sorted by `id_columns` then by column name) and canonicalized per cell
    (exact numeric ratios, typed strings, JSON null) so identical values hash
    identically regardless of row order, column order, dtype, or
    pandas/numpy version. Any real value change, or any value
    becoming/un-becoming missing, changes the hash.

    Encoding is lossless and type-tagged on purpose. Formatting floats to
    fixed precision instead collides genuinely different values (two float64s
    one ULP apart round to the same text), and an in-band null marker
    collides with a string cell that happens to contain that same marker
    text. `as_integer_ratio()` keeps a float's exact binary value, and the
    ["number"|"string", ...] tag keeps 1 distinct from "1"."""
    # A bare string is the natural way to pass one id column; without this it
    # silently becomes a list of its characters and fails with nonsense.
    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    missing = [c for c in id_columns if c not in rows.columns]
    if missing:
        raise KeyError(f"content_hash id_columns not in frame: {missing}")
    if rows[id_columns].isna().any().any() or rows.duplicated(id_columns).any():
        raise ValueError("content_hash id_columns must be non-null and unique")
    if not rows.columns.is_unique or not all(isinstance(c, str) for c in rows.columns):
        raise ValueError("content_hash columns must have unique string names")

    ordered = rows.sort_values(id_columns).reset_index(drop=True)
    columns = sorted(ordered.columns)
    ordered = ordered[columns]

    def cell(value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_, int, np.integer)):
            return ["number", int(value), 1]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return ["number", str(float(value))]
            numerator, denominator = float(value).as_integer_ratio()
            return ["number", numerator, denominator]
        if isinstance(value, str):
            return ["string", value]
        # Feature/label tables in this repo carry gameday, so dates have to
        # hash rather than fail. Exact integer nanoseconds, no formatting.
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            stamp = pd.Timestamp(value)
            return ["datetime", int(stamp.value), str(stamp.tz)]
        raise TypeError(f"Unsupported content-hash cell type: {type(value).__name__}")

    # JSON structure, so no delimiter can appear inside a value and merge two
    # cells. itertuples, not iterrows: iterrows can coerce a mixed-dtype row
    # to a common float and round large integers.
    payload = [columns, [[cell(v) for v in row] for row in ordered.itertuples(index=False, name=None)]]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    manifest = create_snapshot()
    print(json.dumps(manifest, indent=2, sort_keys=True))
