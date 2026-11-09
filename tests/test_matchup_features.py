"""
Synthetic-data tests for src/features/rolling_features.py's label, allowlist,
and admitted-history logic (docs/model-plan.md §9 gates 1, 2, 3, 6). No
network or pre-built database needed -- every fixture here is constructed
by hand, unlike tests/test_rolling_features.py which reads cached real data.

Run: python3 tests/test_matchup_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.rolling_features import (
    add_rolling_features, build_matchup_table, select_features, _compute_labels,
    normalized_home_probability, _neutral_site, TEAM_FEATURE_COLUMNS, STAT_COLUMNS,
    QB_REST_FEATURE_COLUMNS, qb_rest_features,
)


def _games(rows):
    """rows: list of dicts. Missing keys default to a REG, dated game
    between placeholder teams "H"/"A" with a neutral market quote, so each
    test only has to spell out what it cares about. build_matchup_table()
    cross-checks these rows against the corresponding _team_game_stats()
    fixture's team/opponent/is_home (see its exact-scheduled-pairs check),
    so a test using real team names in _team_game_stats() must pass
    matching home_team/away_team here."""
    defaults = {"game_type": "REG", "gameday": "2023-09-10", "season": 2023, "week": 1,
                "home_team": "H", "away_team": "A", "home_rest": 7, "away_rest": 7,
                "spread_line": 0.0, "total_line": 44.0, "home_moneyline": -110, "away_moneyline": -110}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _team_game_stats(rows):
    """rows: list of dicts. Missing keys default to a decided, admitted,
    dated REG game so each test only has to spell out what it cares about."""
    defaults = {"season": 2023, "week": 1, "game_type": "REG",
                "is_home": True, "opponent": "OPP", "rest_days": 7}
    out = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("points_for", 20)
        row.setdefault("points_against", 10)
        for c in STAT_COLUMNS:
            row.setdefault(c, float(i))  # distinct per row so rolling changes are visible
        out.append(row)
    return pd.DataFrame(out)


# --- Gate 1: home_win missing-score/tie labels --------------------------

def test_home_win_label():
    games = _games([
        {"game_id": "g_decided", "home_score": 20, "away_score": 17},   # home won
        {"game_id": "g_lost", "home_score": 10, "away_score": 24},      # home lost
        {"game_id": "g_tie", "home_score": 21, "away_score": 21},       # tie
        {"game_id": "g_unplayed", "home_score": None, "away_score": None},  # not played
    ])
    out = _compute_labels(games).set_index("game_id")

    assert out.loc["g_decided", "home_win"] == True
    assert out.loc["g_lost", "home_win"] == False
    assert pd.isna(out.loc["g_tie", "home_win"]), "tie must be missing, not False"
    assert pd.isna(out.loc["g_unplayed", "home_win"]), \
        "unplayed game (null score) must be missing, not False -- this is the confirmed B2 bug"


# --- Gate 2: explicit feature allowlist ----------------------------------

def test_select_features_never_returns_target_or_id_columns():
    banned = {"home_score", "away_score", "point_diff", "home_win", "game_id"}
    assert not (banned & set(TEAM_FEATURE_COLUMNS)), \
        "TEAM_FEATURE_COLUMNS must never include a target/ID column by construction"

    matchup = pd.DataFrame({c: [1.0] for c in TEAM_FEATURE_COLUMNS})
    for c in banned:
        matchup[c] = [1.0]
    X = select_features(matchup)
    assert not (banned & set(X.columns)), "select_features() leaked a target/ID column"
    assert list(X.columns) == list(TEAM_FEATURE_COLUMNS)


def test_select_features_raises_on_missing_column():
    matchup = pd.DataFrame({TEAM_FEATURE_COLUMNS[0]: [1.0]})
    try:
        select_features(matchup)
        assert False, "expected KeyError for an incomplete matchup table"
    except KeyError:
        pass


def test_select_features_rejects_unknown_variant():
    matchup = pd.DataFrame({c: [1.0] for c in TEAM_FEATURE_COLUMNS})
    try:
        select_features(matchup, variant="bogus")
        assert False, "expected ValueError for an unknown variant"
    except ValueError:
        pass


def test_team_market_variant_devigs_and_drops_invalid_quotes():
    games = _games([
        {"game_id": "g_even", "home_score": 20, "away_score": 17,
         "home_moneyline": -110, "away_moneyline": -110},
        {"game_id": "g_favorite", "home_score": 27, "away_score": 24,
         "home_moneyline": -200, "away_moneyline": 170},
        {"game_id": "g_no_market", "home_score": 10, "away_score": 9,
         "home_moneyline": None, "away_moneyline": None},
        {"game_id": "g_invalid_market", "home_score": 14, "away_score": 3,
         "home_moneyline": 0, "away_moneyline": -110},  # zero odds -- invalid, must abstain
    ])
    p = normalized_home_probability(games).round(6)
    assert p.iloc[0] == 0.5, "symmetric -110/-110 must devig to exactly 0.5"
    implied_fav = 200 / 300   # -200 -> 200/(200+100)
    implied_dog = 100 / 270   # +170 -> 100/(170+100)
    expected_fav = round(implied_fav / (implied_fav + implied_dog), 6)
    assert p.iloc[1] == expected_fav
    assert pd.isna(p.iloc[2]), "missing quotes must abstain, never be imputed"
    assert pd.isna(p.iloc[3]), "zero/invalid odds must abstain"

    tgs = pd.concat([
        _team_game_stats([{"game_id": gid, "team": "H", "opponent": "A", "gameday": f"2023-09-{10+i:02d}"}])
        for i, gid in enumerate(["g_even", "g_favorite", "g_no_market", "g_invalid_market"])
    ] + [
        _team_game_stats([{"game_id": gid, "team": "A", "opponent": "H", "is_home": False,
                            "gameday": f"2023-09-{10+i:02d}"}])
        for i, gid in enumerate(["g_even", "g_favorite", "g_no_market", "g_invalid_market"])
    ], ignore_index=True)
    matchup = build_matchup_table(add_rolling_features(tgs), games)
    market_only = select_features(matchup, "team_market")
    kept_ids = matchup.loc[market_only.index, "game_id"].tolist()
    assert set(kept_ids) == {"g_even", "g_favorite"}, \
        "team_market must drop rows with no/invalid quote pair, not impute them"


# --- Referential integrity: no silent duplicate/dropped games -----------

def test_add_rolling_features_rejects_duplicate_game_team_rows():
    tgs = _team_game_stats([
        {"game_id": "g1", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g1", "team": "KC", "gameday": "2023-09-10"},  # duplicate key
    ])
    try:
        add_rolling_features(tgs)
        assert False, "expected ValueError for duplicate (game_id, team) rows"
    except ValueError:
        pass


def test_build_matchup_table_rejects_two_home_rows_for_one_game():
    tgs = _team_game_stats([
        {"game_id": "g1", "team": "KC", "is_home": True, "gameday": "2023-09-10"},
        {"game_id": "g1", "team": "DEN", "is_home": True, "gameday": "2023-09-10"},  # bug: two homes
    ])
    games = _games([{"game_id": "g1", "home_score": 20, "away_score": 17}])
    feats = add_rolling_features(tgs)
    try:
        build_matchup_table(feats, games)
        assert False, "expected ValueError instead of a silently duplicated/cartesian matchup row"
    except ValueError:
        pass


def test_add_rolling_features_empty_history_does_not_crash():
    # REG but unplayed (null scores) -- exercises the empty-admitted-history
    # guard branch itself. An all-PRE/POST input no longer reaches this
    # branch at all: it's filtered out before any history computation runs
    # (see test_non_reg_games_excluded_from_output_entirely).
    tgs = _team_game_stats([
        {"game_id": "g_future", "team": "KC", "gameday": "2026-09-10",
         "points_for": None, "points_against": None},
    ])
    feats = add_rolling_features(tgs).set_index("game_id")
    assert np.isnan(feats.loc["g_future", f"roll_{STAT_COLUMNS[0]}"])
    assert feats.loc["g_future", "games_played_before"] == 0


def test_non_reg_games_excluded_from_output_entirely():
    tgs = _team_game_stats([
        {"game_id": "g_reg", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g_post", "team": "KC", "gameday": "2024-01-20", "game_type": "POST"},
        {"game_id": "g_pre", "team": "KC", "gameday": "2023-08-10", "game_type": "PRE"},
    ])
    feats = add_rolling_features(tgs)
    assert set(feats["game_id"]) == {"g_reg"}, \
        "preseason/postseason rows must not appear in the output at all, not just carry a null label"


# --- Eligibility: home_win must match the same admission rule as history ---

def test_postseason_excluded_and_unparseable_date_gets_no_label():
    tgs = _team_game_stats([
        {"game_id": "g_post", "team": "KC", "opponent": "SF", "is_home": True,
         "gameday": "2024-01-20", "game_type": "POST"},
        {"game_id": "g_post", "team": "SF", "opponent": "KC", "is_home": False,
         "gameday": "2024-01-20", "game_type": "POST"},
        {"game_id": "g_bad_date", "team": "KC", "opponent": "SF", "is_home": True, "gameday": None},
        {"game_id": "g_bad_date", "team": "SF", "opponent": "KC", "is_home": False, "gameday": None},
        {"game_id": "g_reg", "team": "KC", "opponent": "SF", "is_home": True, "gameday": "2023-09-10"},
        {"game_id": "g_reg", "team": "SF", "opponent": "KC", "is_home": False, "gameday": "2023-09-10"},
    ])
    games = _games([
        {"game_id": "g_post", "home_score": 24, "away_score": 21, "game_type": "POST", "gameday": "2024-01-20",
         "home_team": "KC", "away_team": "SF"},
        {"game_id": "g_bad_date", "home_score": 24, "away_score": 21, "game_type": "REG", "gameday": None,
         "home_team": "KC", "away_team": "SF"},
        {"game_id": "g_reg", "home_score": 24, "away_score": 21, "game_type": "REG", "gameday": "2023-09-10",
         "home_team": "KC", "away_team": "SF"},
    ])
    matchup = build_matchup_table(add_rolling_features(tgs), games).set_index("game_id")
    assert "g_post" not in matchup.index, \
        "a decided postseason game must not appear in matchup_features at all (model-plan.md §2)"
    assert pd.isna(matchup.loc["g_bad_date", "home_win"]), \
        "a decided REG game with an unparseable date must not get a trainable label"
    assert matchup.loc["g_reg", "home_win"] == True


# --- Same-date games can't order each other (no completion timestamps) ---

def test_same_calendar_date_games_cannot_see_each_other():
    # Two rows for the same team on an identical date can't happen in real
    # NFL data (a team never plays twice in a day) -- this is a defensive
    # case: if it ever did, neither may see the other, since there's no
    # completion timestamp to order them by.
    tgs = _team_game_stats([
        {"game_id": "g_a", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g_b", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g_next_week", "team": "KC", "gameday": "2023-09-17"},
    ])
    feats = add_rolling_features(tgs).set_index("game_id")
    assert feats.loc["g_a", "games_played_before"] == 0
    assert feats.loc["g_b", "games_played_before"] == 0, \
        "a same-date sibling must not be admitted into another same-date game's history"
    assert feats.loc["g_next_week", "games_played_before"] == 2, \
        "both same-day games are still admitted once a strictly later date arrives"


def test_gameday_must_be_strict_iso_date():
    tgs = _team_game_stats([
        {"game_id": "g_ok", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g_timestamp", "team": "KC", "gameday": "2023-09-17 13:00:00"},
        {"game_id": "g_slash", "team": "KC", "gameday": "09/24/2023"},
    ])
    feats = add_rolling_features(tgs).set_index("game_id")
    assert feats.loc["g_ok", "games_played_before"] == 0
    for gid in ("g_timestamp", "g_slash"):
        assert np.isnan(feats.loc[gid, f"roll_{STAT_COLUMNS[0]}"]), \
            f"{gid}: a non-ISO gameday format must be treated as unparseable, not guessed"
        assert pd.isna(feats.loc[gid, "games_played_before"]), \
            f"{gid}: an unparseable date makes games_played_before unknown, not a confirmed 0"


# --- Gate 6: admitted-history rules --------------------------------------

def test_unplayed_and_non_reg_games_do_not_advance_history():
    tgs = _team_game_stats([
        {"game_id": "g1", "team": "KC", "gameday": "2023-09-10"},
        {"game_id": "g2", "team": "KC", "gameday": "2023-09-17"},
        # a future/unplayed game slotted in between by date -- must not count
        {"game_id": "g_future", "team": "KC", "gameday": "2023-09-24",
         "points_for": None, "points_against": None},
        # a playoff game -- excluded from the output entirely (see
        # test_non_reg_games_excluded_from_output_entirely), and must not
        # count toward regular-season history either
        {"game_id": "g_playoff", "team": "KC", "gameday": "2023-09-25", "game_type": "POST"},
        {"game_id": "g3", "team": "KC", "gameday": "2023-10-01"},
    ])
    feats = add_rolling_features(tgs)

    assert "g_playoff" not in set(feats["game_id"])

    counts = feats.set_index("game_id")["games_played_before"]
    assert counts["g1"] == 0
    assert counts["g2"] == 1
    assert counts["g_future"] == 2, "an unplayed game must still see prior admitted history"
    assert counts["g3"] == 2, "the playoff and unplayed games must not have advanced the REG count"

    roll_col = f"roll_{STAT_COLUMNS[0]}"
    row = feats.set_index("game_id")[roll_col]
    assert row["g3"] == row["g_future"], \
        "g_future/g3 both sit after the same admitted history (g1, g2) and must carry the same value"


def test_cold_start_and_unparseable_date():
    tgs = _team_game_stats([
        {"game_id": "g_bad_date", "team": "NE", "gameday": None},
        {"game_id": "g1", "team": "NE", "gameday": "2023-09-10"},
    ])
    feats = add_rolling_features(tgs).set_index("game_id")

    assert feats.loc["g1", "games_played_before"] == 0, \
        "a row with an unparseable date must not count as admitted history for a later game"
    assert np.isnan(feats.loc["g_bad_date", f"roll_{STAT_COLUMNS[0]}"])
    assert pd.isna(feats.loc["g_bad_date", "games_played_before"]), \
        "an unparseable date's own games_played_before is unknown, not a confirmed 0"


def test_cross_season_history_carries_without_reset():
    tgs = _team_game_stats([
        {"game_id": "g_2023_last", "team": "BUF", "season": 2023, "week": 18, "gameday": "2024-01-07"},
        {"game_id": "g_2024_first", "team": "BUF", "season": 2024, "week": 1, "gameday": "2024-09-08"},
    ])
    feats = add_rolling_features(tgs).set_index("game_id")
    assert feats.loc["g_2024_first", "games_played_before"] == 1, \
        "history must carry across the season boundary, not reset to 0"


# --- Gate 3: leak-safety extended to build_matchup_table() ---------------

def test_leak_safety_extends_to_matchup_table():
    tgs = _team_game_stats([
        {"game_id": "g1", "team": "KC", "opponent": "DEN", "gameday": "2023-09-10"},
        {"game_id": "g2", "team": "KC", "opponent": "LAC", "gameday": "2023-09-17"},
        {"game_id": "g1", "team": "DEN", "opponent": "KC", "gameday": "2023-09-10", "is_home": False},
        {"game_id": "g2", "team": "LAC", "opponent": "KC", "gameday": "2023-09-17", "is_home": False},
    ])
    games = _games([
        {"game_id": "g1", "home_score": 20, "away_score": 10, "home_team": "KC", "away_team": "DEN"},
        {"game_id": "g2", "home_score": 24, "away_score": 21, "home_team": "KC", "away_team": "LAC"},
    ])

    baseline_matchup = build_matchup_table(add_rolling_features(tgs), games)

    corrupted = tgs.copy()
    idx = corrupted[(corrupted["game_id"] == "g1") & (corrupted["team"] == "KC")].index[0]
    corrupted.loc[idx, STAT_COLUMNS[0]] = 999999.0
    corrupted_matchup = build_matchup_table(add_rolling_features(corrupted), games)

    col = f"home_roll_{STAT_COLUMNS[0]}"
    g1_before = baseline_matchup.set_index("game_id").loc["g1", col]
    g1_after = corrupted_matchup.set_index("game_id").loc["g1", col]
    assert g1_before == g1_after or (pd.isna(g1_before) and pd.isna(g1_after)), \
        "LEAKAGE: g1's own matchup row changed from a value planted in g1 itself"

    g2_before = baseline_matchup.set_index("game_id").loc["g2", col]
    g2_after = corrupted_matchup.set_index("game_id").loc["g2", col]
    assert g2_before != g2_after, \
        "KC's next game's matchup row never changed -- history isn't flowing through build_matchup_table()"


def test_neutral_site_is_recorded_but_never_becomes_a_feature():
    """A neutral-site game's "home" team has no home field, but the row is
    shaped exactly like every other home/away row and says otherwise.

    model-plan.md §4 excludes `is_home` as "redundant with the home/away row
    structure itself" -- which holds for ordinary games and fails for these.
    The flag is carried as matchup metadata so Stage 4 can mask or report the
    rows; it is deliberately NOT added to the allowlist, because §4 freezes
    that and changing it is a protocol decision.
    """
    games = _games([
        {"game_id": "g_home", "gameday": "2023-09-10", "home_team": "H", "away_team": "A",
         "home_score": 20, "away_score": 17, "location": "Home"},
        {"game_id": "g_neutral", "gameday": "2023-09-17", "home_team": "H", "away_team": "A",
         "home_score": 10, "away_score": 24, "location": "Neutral"},
    ])
    stats = _team_game_stats([
        {"game_id": "g_home", "team": "H", "opponent": "A", "is_home": True,
         "gameday": "2023-09-10"},
        {"game_id": "g_home", "team": "A", "opponent": "H", "is_home": False,
         "gameday": "2023-09-10"},
        {"game_id": "g_neutral", "team": "H", "opponent": "A", "is_home": True,
         "gameday": "2023-09-17"},
        {"game_id": "g_neutral", "team": "A", "opponent": "H", "is_home": False,
         "gameday": "2023-09-17"},
    ])
    matchup = build_matchup_table(add_rolling_features(stats), games).set_index("game_id")

    assert matchup.loc["g_neutral", "neutral_site"] == True
    assert matchup.loc["g_home", "neutral_site"] == False
    # The label is unaffected -- a neutral-site game still has a winner.
    assert matchup.loc["g_neutral", "home_win"] == False

    for variant in ("team_only", "team_market"):
        X = select_features(matchup.reset_index(), variant)
        assert "neutral_site" not in X.columns, \
            f"{variant}: neutral_site is metadata, not an allowlisted feature"
        assert "location" not in X.columns

    # Absent `location` is UNKNOWN, never False -- silently calling an
    # unlabelled game a home game is the exact failure this flag exists for.
    no_column = _games([{"game_id": "g_home", "home_score": 20, "away_score": 17}])
    assert "location" not in no_column.columns
    assert _compute_labels(no_column).shape[0] == 1
    assert pd.isna(_neutral_site(no_column)).all(), "missing location must be NA, not False"

    # A location value nobody has classified must stop the build, not fall
    # through as "home".
    surprise = _games([{"game_id": "g1", "home_score": 20, "away_score": 17,
                        "location": "Overseas"}])
    try:
        _neutral_site(surprise)
        assert False, "expected ValueError for an unclassified location value"
    except ValueError as exc:
        assert "Overseas" in str(exc) and "Home" in str(exc)


# --- Protocol v5 inputs: starting-QB continuity and rest (plan Step 4) -----

def _qb_games(**overrides):
    """KC starts QB "A" three straight weeks. DEN's same-day game in week 2 also
    lists "A" -- impossible in reality, and exactly what proves a same-day game
    cannot feed another's history."""
    rows = [
        {"game_id": "g1", "gameday": "2023-09-10", "home_team": "KC", "away_team": "DEN",
         "home_qb_id": "A", "away_qb_id": "D", "home_score": 20, "away_score": 10},
        {"game_id": "g2", "gameday": "2023-09-17", "home_team": "KC", "away_team": "LAC",
         "home_qb_id": "A", "away_qb_id": "L", "home_score": 24, "away_score": 21, "away_rest": 6},
        {"game_id": "g2b", "gameday": "2023-09-17", "home_team": "DEN", "away_team": "LV",
         "home_qb_id": "A", "away_qb_id": "V", "home_score": 17, "away_score": 14},
        {"game_id": "g3", "gameday": "2023-09-24", "home_team": "KC", "away_team": "LV",
         "home_qb_id": "A", "away_qb_id": "V", "home_score": 30, "away_score": 20},
    ]
    for game_id, changes in overrides.items():
        next(r for r in rows if r["game_id"] == game_id).update(changes)
    return _games(rows)


def _qb_feats(games):
    return qb_rest_features(games).set_index("game_id")


def _same(a, b):
    return a.equals(b) or (a.isna() == b.isna()).all() and (a.dropna() == b.dropna()).all()


def test_qb_rest_features_values():
    f = _qb_feats(_qb_games())
    assert pd.isna(f.loc["g1", "home_qb_changed"]), "a team's first game has no previous starter"
    assert f.loc["g1", "home_qb_prior_starts"] == 0
    assert f.loc["g2", "home_qb_changed"] == 0.0 and f.loc["g2", "home_qb_prior_starts"] == 1, \
        "g2 sees g1 only -- g2b is the same day"
    assert f.loc["g2", "rest_diff"] == 1.0, "home_rest 7 minus away_rest 6"
    assert f.loc["g2b", "home_qb_changed"] == 1.0, "DEN went from D in g1 to A in g2b"
    assert f.loc["g3", "home_qb_prior_starts"] == 3, "g1, g2 and g2b are all on earlier dates"
    assert f.loc["g3", "away_qb_changed"] == 0.0 and f.loc["g3", "away_qb_prior_starts"] == 1
    assert list(f.columns) == list(QB_REST_FEATURE_COLUMNS)


def test_qb_rest_features_are_leak_safe():
    """Corruption injection: plant changes where a leak would carry them, and
    prove they never reach a row they must not -- then prove the same machinery
    does carry an EARLIER change forward, so the checks are not vacuous."""
    base = _qb_feats(_qb_games())

    # A game's own outcome never reaches its own row.
    own = _qb_feats(_qb_games(g2={"home_score": 0, "away_score": 60}))
    assert _same(own.loc["g2"], base.loc["g2"]), "LEAKAGE: g2's own score moved g2's features"

    # Nothing from a later game reaches an earlier one.
    later = _qb_feats(_qb_games(g3={"home_qb_id": "Z", "away_qb_id": "Y", "home_score": 99,
                                    "home_rest": 30}))
    for game in ("g1", "g2", "g2b"):
        assert _same(later.loc[game], base.loc[game]), f"LEAKAGE: g3 changed {game}"

    # A same-day game is invisible.
    same_day = _qb_feats(_qb_games(g2b={"home_qb_id": "Q"}))
    assert _same(same_day.loc["g2"], base.loc["g2"]), "LEAKAGE: same-day g2b reached g2"

    # An unplayed earlier game is not history.
    unplayed = _qb_feats(_qb_games(g1={"home_score": None, "away_score": None}))
    assert unplayed.loc["g2", "home_qb_prior_starts"] == 0
    assert pd.isna(unplayed.loc["g2", "home_qb_changed"]), \
        "with g1 unplayed, KC has no completed previous game"

    # Non-vacuous: an EARLIER change does flow forward.
    earlier = _qb_feats(_qb_games(g1={"home_qb_id": "B"}))
    assert earlier.loc["g2", "home_qb_changed"] == 1.0, "a starter change in g1 must show in g2"
    assert earlier.loc["g2", "home_qb_prior_starts"] == 0, "A had no completed start before g2"


def test_qb_features_unknown_without_source_columns():
    """No QB columns in the source means unknown, never "unchanged" or "0 starts"."""
    f = _qb_feats(_games([{"game_id": "g1", "home_score": 20, "away_score": 17, "home_rest": 10}]))
    for column in ("home_qb_changed", "away_qb_changed", "home_qb_prior_starts", "away_qb_prior_starts"):
        assert pd.isna(f.loc["g1", column]), f"{column} must be NaN when the starter is unknown"
    assert f.loc["g1", "rest_diff"] == 3.0, "rest is independent of the QB columns"


def test_qb_rest_allowlist_holds_no_target_score_or_id():
    banned = {"home_score", "away_score", "point_diff", "home_win", "game_id",
              "home_qb_id", "away_qb_id", "gameday"}
    assert not banned & set(QB_REST_FEATURE_COLUMNS), \
        "the QB/rest allowlist must never name a target, score, raw ID or date"
    # Every allowlisted column is actually built into the matchup table.
    tgs = _team_game_stats([
        {"game_id": "g1", "team": "KC", "opponent": "DEN", "gameday": "2023-09-10"},
        {"game_id": "g1", "team": "DEN", "opponent": "KC", "gameday": "2023-09-10", "is_home": False},
    ])
    games = _games([{"game_id": "g1", "home_score": 20, "away_score": 10, "home_team": "KC",
                     "away_team": "DEN", "home_qb_id": "A", "away_qb_id": "D"}])
    matchup = build_matchup_table(add_rolling_features(tgs), games)
    assert set(QB_REST_FEATURE_COLUMNS) <= set(matchup.columns)


if __name__ == "__main__":
    test_home_win_label()
    test_select_features_never_returns_target_or_id_columns()
    test_select_features_raises_on_missing_column()
    test_select_features_rejects_unknown_variant()
    test_team_market_variant_devigs_and_drops_invalid_quotes()
    test_add_rolling_features_rejects_duplicate_game_team_rows()
    test_build_matchup_table_rejects_two_home_rows_for_one_game()
    test_add_rolling_features_empty_history_does_not_crash()
    test_non_reg_games_excluded_from_output_entirely()
    test_postseason_excluded_and_unparseable_date_gets_no_label()
    test_same_calendar_date_games_cannot_see_each_other()
    test_gameday_must_be_strict_iso_date()
    test_unplayed_and_non_reg_games_do_not_advance_history()
    test_cold_start_and_unparseable_date()
    test_cross_season_history_carries_without_reset()
    test_leak_safety_extends_to_matchup_table()
    test_neutral_site_is_recorded_but_never_becomes_a_feature()
    test_qb_rest_features_values()
    test_qb_rest_features_are_leak_safe()
    test_qb_features_unknown_without_source_columns()
    test_qb_rest_allowlist_holds_no_target_score_or_id()
    print("OK -- labels, allowlist, market devig, referential integrity, admitted-history rules, "
          "matchup-table leak-safety, and the QB/rest features' leak-safety all hold.")
