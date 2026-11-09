"""
Two static guards over `src/`'s database discipline.

**1. Every call to StatScraper's shared SQLite helpers names its database.**

`save_to_sqlite`, `query_sqlite` and `drop_table_if_exists` all default
`db_path` to "data/nfl_stats.db" -- the PLAYER pipeline's database. The game
pipeline works only because every call site overrides it with
"data/nfl_games.db". Omit that once and the call silently reads, or worse
writes, the wrong database: no exception, no dtype mismatch, just numbers from
the other project. `save_to_sqlite` would happily CREATE the table there.

The two pipelines are separate by convention, not by construction (project notes:
"the SQLite helpers are shared infrastructure even though the two pipelines'
databases are separate"). This test makes the convention enforced rather than
remembered.

Deliberately a static check. Requiring `db_path` in the helpers themselves
would edit StatScraper.py and 6 player-side call sites, and
docs/next-implementations.md defers "any change to the root-level player-model
pipeline".

**2. Training and evaluation never replace a table.**

Stage 1: "Training/evaluation must never call a table-replacement entry point
(`team_game_stats.run()`/`rolling_features.run()`) against the selected
snapshot... Rebuild into memory or scratch storage."

`snapshot.open_snapshot()` enforces that for a snapshot directory by opening
`mode=ro`, so a write raises. It cannot enforce it for `data/nfl_games.db`
itself, which is writable and is what every current test points at. Stage 4
adds the first real fold loop over that database; this keeps the rule from
being rediscovered the hard way.

Run: python3 tests/test_db_path_discipline.py
"""
import ast
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Entry points that DROP and re-CREATE a table. Reading (`query_sqlite`) is
# fine anywhere; these three are not, inside the packages below.
TABLE_REPLACING = ("run", "save_to_sqlite", "drop_table_if_exists")

# Packages that consume the data and must never rebuild it.
READ_ONLY_PACKAGES = ("evaluation", "models")

# Helper -> zero-based position of `db_path` in its signature, so a positional
# call is accepted as readily as a keyword one. See StatScraper.py.
DB_PATH_POSITION = {
    "save_to_sqlite": 3,        # (df, table_name, key_columns, db_path)
    "query_sqlite": 1,          # (sql, db_path, params)
    "drop_table_if_exists": 1,  # (table_name, db_path)
}


def _called_name(node: ast.Call) -> str:
    """`query_sqlite(...)` and `StatScraper.query_sqlite(...)` alike."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _scan(path: Path) -> tuple:
    """`(violations, checked)` for one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations, checked = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name not in DB_PATH_POSITION:
            continue
        checked += 1
        by_keyword = any(kw.arg == "db_path" for kw in node.keywords)
        by_position = len(node.args) > DB_PATH_POSITION[name]
        # **kwargs forwarding (kw.arg is None) could carry db_path; don't
        # flag what this check cannot see.
        forwarded = any(kw.arg is None for kw in node.keywords)
        if not (by_keyword or by_position or forwarded):
            violations.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}() without db_path")
    return violations, checked


def demo_every_src_call_names_its_database():
    sources = sorted((REPO / "src").rglob("*.py"))
    assert sources, "no src/*.py files found -- test is pointing at the wrong tree"

    violations, checked = [], 0
    for path in sources:
        found, n = _scan(path)
        violations += found
        checked += n

    # Non-vacuity: a scanner that matches nothing passes trivially. If the
    # helpers are renamed or the imports move, this fires instead of going
    # quietly green.
    assert checked >= 10, (
        f"only {checked} call(s) to the shared SQLite helpers found in src/ -- "
        f"expected at least 10. Were they renamed, or is DB_PATH_POSITION stale?"
    )

    assert not violations, (
        "these calls fall back to the player pipeline's data/nfl_stats.db:\n  "
        + "\n  ".join(violations)
    )

    print(f"OK -- all {checked} src/ calls to save_to_sqlite/query_sqlite/"
          f"drop_table_if_exists name their database explicitly.")


def _scan_replacements(path: Path) -> list:
    """Table-replacing calls found in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{path.relative_to(REPO)}:{node.lineno}: {_called_name(node)}()"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) in TABLE_REPLACING
    ]


def demo_training_and_evaluation_never_replace_a_table():
    violations, scanned = [], 0
    for package in READ_ONLY_PACKAGES:
        sources = sorted((REPO / "src" / package).rglob("*.py"))
        assert sources, f"src/{package}/ has no Python files -- has it moved?"
        for path in sources:
            violations += _scan_replacements(path)
            scanned += 1

    # Non-vacuity: if the packages are renamed, the loop above raises rather
    # than scanning nothing. This catches the subtler case of them emptying out.
    assert scanned >= 4, f"only {scanned} file(s) scanned across {READ_ONLY_PACKAGES}"

    assert not violations, (
        "training/evaluation code must read the data, never rebuild it "
        "(docs/next-implementations.md Stage 1). Rebuild into scratch storage "
        "instead:\n  " + "\n  ".join(violations)
    )

    print(f"OK -- no table-replacing call ({'/'.join(TABLE_REPLACING)}) in any of "
          f"the {scanned} files under src/{', src/'.join(READ_ONLY_PACKAGES)}.")


def demo_the_guard_actually_catches_an_omission():
    """The check above is only worth having if it fails on a bad call -- and it
    has to fail through `_scan` itself. A second copy of the rule written here
    would keep passing after `_scan` stopped working.

    The fixtures live under REPO (outside src/, so the scan above ignores them)
    because `_scan` reports paths relative to it.
    """
    # dir=REPO: _scan's message is path.relative_to(REPO).
    with tempfile.TemporaryDirectory(dir=REPO) as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text(
            "from StatScraper import query_sqlite\n"
            "df = query_sqlite('SELECT * FROM games')\n",
            encoding="utf-8")
        violations, checked = _scan(bad)
        assert checked == 1, "the fixture must reach the db_path check at all"
        assert len(violations) == 1 and "query_sqlite() without db_path" in violations[0], \
            f"the guard must flag a call that omits db_path, got {violations}"

        good = Path(tmp) / "good.py"
        good.write_text(
            "from StatScraper import query_sqlite, save_to_sqlite\n"
            "a = query_sqlite('SELECT 1', db_path='data/nfl_games.db')\n"   # keyword
            "b = query_sqlite('SELECT 1', 'data/nfl_games.db')\n"           # positional
            "c = save_to_sqlite(df, 't', ['k'], db_path=GAMES_DB_PATH)\n",  # keyword, 4-arg
            encoding="utf-8")
        violations, checked = _scan(good)
        assert checked == 3, "all three legitimate forms must be examined"
        assert not violations, f"guard flagged a correct call: {violations}"

        # Same treatment for the second guard: a rebuild call is flagged, a
        # read is not.
        rebuild = Path(tmp) / "rebuild.py"
        rebuild.write_text(
            "from src.features import rolling_features\n"
            "rolling_features.run(db_path='snapshots/abc/nfl_games.db')\n",
            encoding="utf-8")
        flagged = _scan_replacements(rebuild)
        assert len(flagged) == 1 and "run()" in flagged[0], \
            f"a table-replacing call must be flagged, got {flagged}"

        reader = Path(tmp) / "reader.py"
        reader.write_text(
            "from StatScraper import query_sqlite\n"
            "from src.features.rolling_features import build_rolling_features\n"
            "df = query_sqlite('SELECT 1', db_path='data/nfl_games.db')\n"
            "features = build_rolling_features(df)\n",   # builds in memory, writes nothing
            encoding="utf-8")
        assert not _scan_replacements(reader), \
            "reading and in-memory building must not be flagged"

    print("OK -- the guard flags an omitted db_path and a table-replacing call, "
          "and leaves keyword, positional and read-only forms alone.")


if __name__ == "__main__":
    demo_every_src_call_names_its_database()
    demo_training_and_evaluation_never_replace_a_table()
    demo_the_guard_actually_catches_an_omission()
