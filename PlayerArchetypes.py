"""
PlayerArchetypes.py
=====================
Classify QB/RB/WR/TE players into play-style archetypes from real
box-score-derived stats, and measure how each defense performs
specifically against each archetype -- a separate, standalone module from
PlayerProjections.py/PlayerProjectionsTransformer.py, reusing their data
loaders rather than re-pulling anything.

WHY NOT REAL MADDEN ARCHETYPES
--------------------------------
Madden's in-game archetypes/ratings are EA's proprietary data with no
legitimate public API -- there's no clean, licensed way to pull them into
this project, and scraping them would mean depending on numbers this
project can't verify or reproduce. This module instead builds the same
KIND of classification from real, verifiable NFL performance data already
in this project's pipeline, calibrated against real box scores rather than
someone's in-game rating guess. Category names are Madden-inspired (QB's
were given explicitly: Scrambler/Improviser/Gunslinger/Field General; RB/WR/
TE's are this project's own, in the same spirit); the numbers underneath
are this project's own throughout.

METHOD
------
Each position has its own small set of style stats (ARCHETYPE_FEATURES),
computed as TRAILING (leak-safe, shifted) averages over ROLLING_DEFENSE_WINDOW
games, from RAW_STAT_COLUMNS summed first and THEN divided (never averaging
already-computed per-game rates -- see build_style_features()'s Notes for
why, same reasoning PlayerProjections.build_top_cb_profile() uses).

Each player's stat vector is Z-SCORED against the qualifying population AT
THEIR OWN POSITION for that same (season, week) -- so "aggressive" or
"high-volume" always means relative to the league at that position that
season, not a fixed historical threshold that drifts meaningless as
league-wide style changes year to year (the same reasoning
PlayerProjections.build_opponent_difficulty_profile() uses for tiering
defenses by quantile rather than a fixed yardage cutoff).

Each archetype is a target vector in this z-scored space (ARCHETYPE_CENTROIDS),
with per-axis WEIGHTS (ARCHETYPE_AXIS_WEIGHTS) so the archetype's actually-
defining stat dominates its distance calculation -- classify_archetype()
assigns each player-week to its nearest weighted-distance archetype. This is
a simple nearest-centroid classifier, not a trained/fit model -- centroids
are hand-specified based on what each archetype means (see
ARCHETYPE_CENTROIDS' comments), not learned from labeled examples (there are
no real labels to learn from). Treat the exact values as a tunable starting
point, not validated ground truth -- see `python PlayerArchetypes.py` for
real current-season examples to sanity-check them against.

USAGE
-----
    from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile
    from PlayerProjections import load_weekly_stats

    df = load_weekly_stats(years=[2023, 2024, 2025])
    style = compute_style_features(df, "QB")
    archetypes = classify_archetype(style, "QB")
    print(archetypes[archetypes["season"] == 2025].groupby("archetype").size())

    defense_profile = build_defense_vs_archetype_profile(df, archetypes, "QB")
    print(defense_profile[defense_profile["team"] == "SF"])
"""

from typing import Dict

import numpy as np
import pandas as pd

from PlayerProjections import POSITION_STAT_COLUMNS, ROLLING_DEFENSE_WINDOW, nfl_passer_rating

# A player needs at least this many trailing "touches" (position-appropriate
# -- see VOLUME_COLUMNS) before their style stats are trusted -- below this,
# a couple of unusual games swing the rates wildly (same "don't trust a tiny
# sample" reasoning as PlayerProjections.MIN_TRAILING_CB_TARGETS).
MIN_TRAILING_VOLUME = {"QB": 40, "RB": 20, "WR": 15, "TE": 10}

# Raw per-game counting stats summed (trailing, shifted) before any ratio is
# computed -- see the module docstring's METHOD section for why sum-then-
# divide, never average-of-rates.
RAW_STAT_COLUMNS = {
    "QB": ["attempts", "completions", "carries", "passing_air_yards", "passing_interceptions"],
    "RB": ["carries", "rushing_yards", "targets", "receiving_yards"],
    "WR": ["targets", "receptions", "receiving_air_yards", "receiving_yards"],
    "TE": ["targets", "receptions", "receiving_air_yards", "receiving_yards"],
}

# Which of RAW_STAT_COLUMNS count as this position's "touches" for
# MIN_TRAILING_VOLUME/qualification purposes.
VOLUME_COLUMNS = {
    "QB": ["attempts"],
    "RB": ["carries", "targets"],
    "WR": ["targets"],
    "TE": ["targets"],
}

ARCHETYPE_FEATURES = {
    "QB": ["rush_rate", "adot", "int_rate", "comp_pct"],
    "RB": ["pass_game_involvement", "yards_per_carry", "yards_per_target"],
    "WR": ["adot", "catch_rate", "yards_per_reception"],
    "TE": ["targets_per_game", "adot", "catch_rate"],
}


def _qb_style(t: dict, games: pd.Series) -> dict:
    return {
        "rush_rate": t["carries"] / (t["attempts"] + t["carries"]),
        "adot": t["passing_air_yards"] / t["attempts"],
        "int_rate": t["passing_interceptions"] / t["attempts"],
        "comp_pct": t["completions"] / t["attempts"],
    }


def _rb_style(t: dict, games: pd.Series) -> dict:
    return {
        "pass_game_involvement": t["targets"] / (t["carries"] + t["targets"]),
        "yards_per_carry": t["rushing_yards"] / t["carries"],
        "yards_per_target": t["receiving_yards"] / t["targets"],
    }


def _wr_style(t: dict, games: pd.Series) -> dict:
    return {
        "adot": t["receiving_air_yards"] / t["targets"],
        "catch_rate": t["receptions"] / t["targets"],
        "yards_per_reception": t["receiving_yards"] / t["receptions"],
    }


def _te_style(t: dict, games: pd.Series) -> dict:
    return {
        "targets_per_game": t["targets"] / games,
        "adot": t["receiving_air_yards"] / t["targets"],
        "catch_rate": t["receptions"] / t["targets"],
    }


_STYLE_FORMULAS = {"QB": _qb_style, "RB": _rb_style, "WR": _wr_style, "TE": _te_style}

# Archetype target vectors, in each position's own ARCHETYPE_FEATURES
# z-score order (see that dict). All hand-specified starting points -- see
# the module docstring's METHOD section.
#
# QB (rush_rate, adot, int_rate, comp_pct):
#   - Scrambler: defined almost entirely by rushing workload -- everything
#     else roughly average. Runs because that's his game, not because a
#     play broke down.
#   - Improviser: real rushing AND above-average efficiency/aDOT -- extends
#     plays outside structure and still makes something of them, distinct
#     from a Scrambler defined by volume alone.
#   - Gunslinger: pocket-based but aggressive downfield and accepts risk
#     for it -- the boom-or-bust pure passer.
#   - Field General: low risk, low rush, checkdown/short game, high
#     accuracy -- the conservative "game manager."
#
# RB (pass_game_involvement, yards_per_carry, yards_per_target):
#   - Workhorse: rush-centric workload, everything else average -- the
#     between-the-tackles feature back.
#   - Receiving Back: pass-game-centric workload -- the third-down/
#     satellite role.
#   - Explosive: big-play ability in EITHER phase (high yards/carry AND
#     yards/target), regardless of role split.
# WR (adot, catch_rate, yards_per_reception):
#   - Deep Threat: aggressive downfield target depth and big plays when it
#     connects, at the cost of a lower catch rate (more contested/
#     lower-percentage looks).
#   - Possession: short/intermediate, high catch rate, high-floor
#     low-ceiling.
#   - Playmaker: shorter average depth but outsized yards-per-catch (gains
#     extra yards after the catch) with a good catch rate -- the YAC-driven
#     profile.
#
# TE (targets_per_game, adot, catch_rate):
#   - Receiving Threat: high-volume, moderate-to-aggressive route depth --
#     used like a big receiver.
#   - Underneath: high-volume but short/checkdown-heavy, high catch rate --
#     the safety-valve role.
#   - In-Line/Low-Usage: defined by LOW target volume -- everything else
#     near average (proprietary blocking data isn't available, so low
#     receiving usage is the best available proxy for an in-line/blocking-
#     first role).
ARCHETYPE_CENTROIDS = {
    "QB": {
        "Scrambler": np.array([1.4, 0.0, 0.0, 0.0]),
        "Improviser": np.array([0.9, 0.6, -0.2, 0.3]),
        "Gunslinger": np.array([-0.3, 1.1, 1.0, -0.3]),
        "Field General": np.array([-0.8, -0.6, -1.0, 0.9]),
    },
    "RB": {
        "Workhorse": np.array([-1.3, 0.2, 0.0]),
        "Receiving Back": np.array([1.3, -0.2, 0.0]),
        "Explosive": np.array([0.0, 1.0, 1.0]),
    },
    "WR": {
        "Deep Threat": np.array([1.3, -0.6, 1.0]),
        "Possession": np.array([-1.0, 1.0, -0.3]),
        "Playmaker": np.array([-0.2, 0.6, 1.0]),
    },
    "TE": {
        "Receiving Threat": np.array([1.2, 0.8, 0.0]),
        "Underneath": np.array([1.0, -0.9, 0.8]),
        "In-Line/Low-Usage": np.array([-1.4, 0.0, 0.0]),
    },
}

# Per-archetype axis weights for classify_archetype()'s distance calc -- NOT
# just "how far is this player from the centroid" on every axis equally, but
# "how far on the axis THIS ARCHETYPE IS ACTUALLY DEFINED BY." Without this,
# a player who's merely average on the other axes can out-"fit" an archetype
# over one with a genuinely extreme value on the axis that actually defines
# it (verified this happening for QB -- Justin Herbert, average elsewhere,
# out-scored Lamar Jackson for closest Scrambler fit under plain unweighted
# distance -- before adding these). Weights are relative, not normalized to
# sum to 1 -- only their ratio within a row matters for nearest-centroid
# ranking.
ARCHETYPE_AXIS_WEIGHTS = {
    "QB": {
        "Scrambler": np.array([2.5, 0.3, 0.3, 0.3]),
        "Improviser": np.array([1.5, 1.2, 0.4, 0.6]),
        "Gunslinger": np.array([0.4, 1.5, 1.5, 0.4]),
        "Field General": np.array([0.6, 0.6, 1.5, 1.5]),
    },
    "RB": {
        "Workhorse": np.array([2.0, 0.5, 0.3]),
        "Receiving Back": np.array([2.0, 0.3, 0.5]),
        "Explosive": np.array([0.3, 1.5, 1.5]),
    },
    "WR": {
        "Deep Threat": np.array([1.8, 0.6, 1.2]),
        "Possession": np.array([1.2, 1.8, 0.5]),
        "Playmaker": np.array([0.5, 1.0, 1.5]),
    },
    "TE": {
        "Receiving Threat": np.array([1.5, 1.3, 0.4]),
        "Underneath": np.array([1.3, 1.3, 1.0]),
        "In-Line/Low-Usage": np.array([2.2, 0.3, 0.3]),
    },
}


def compute_style_features(df: pd.DataFrame, position_group: str, window: int = ROLLING_DEFENSE_WINDOW) -> pd.DataFrame:
    """
    Trailing (leak-safe, shifted) style stats for every player-week at one
    position -- see the module docstring's METHOD section and this
    position's entry in ARCHETYPE_FEATURES for what's computed.

    Args:
        position_group: "QB"/"RB"/"WR"/"TE".
        df: Weekly player stats (e.g. PlayerProjections.load_weekly_stats()'s
            output) -- filtered to `position_group` internally.
        window: Games of trailing history to average over (default
            ROLLING_DEFENSE_WINDOW).

    Returns:
        DataFrame with one row per (player_id, season, week): player_name,
        team, trailing_volume (the sample size these stats are based on --
        see MIN_TRAILING_VOLUME), plus this position's ARCHETYPE_FEATURES
        columns. NaN stats (and trailing_volume below MIN_TRAILING_VOLUME)
        for a player's earliest games, before enough trailing history exists.

    Raises:
        ValueError: if position_group isn't one of QB/RB/WR/TE.

    Notes:
        Leakage prevention: same `.rolling(window).sum().shift(1)` pattern
        used throughout PlayerProjections.py -- each week's style stats are
        computed from STRICTLY EARLIER games' raw counts (summed first,
        THEN divided -- never averaging already-computed per-game rates,
        for the same statistical reason PlayerProjections.build_top_cb_profile()
        sums raw counts before computing a rate stat from them, since
        averaging a rate across games with different volumes is improper),
        so a game's own result never leaks into its own style classification.
    """
    if position_group not in RAW_STAT_COLUMNS:
        raise ValueError(f"Unknown position_group {position_group!r}; expected one of {list(RAW_STAT_COLUMNS)}")

    pos_df = df[df["position_group"] == position_group].sort_values(["player_id", "season", "week"]).copy()

    raw_cols = RAW_STAT_COLUMNS[position_group]
    trailing = {}
    for raw_col in raw_cols:
        trailing[raw_col] = pos_df.groupby("player_id")[raw_col].transform(
            lambda s: s.fillna(0).rolling(window, min_periods=1).sum().shift(1)
        )
    trailing_games = pos_df.groupby("player_id").cumcount()  # number of STRICTLY EARLIER games on record (0 for a player's first)

    volume_cols = VOLUME_COLUMNS[position_group]
    pos_df["trailing_volume"] = sum(trailing[c] for c in volume_cols)

    with np.errstate(divide="ignore", invalid="ignore"):
        style = _STYLE_FORMULAS[position_group](trailing, trailing_games.clip(lower=1))
    for feature_name, values in style.items():
        pos_df[feature_name] = values

    features = ARCHETYPE_FEATURES[position_group]
    pos_df.loc[pos_df["trailing_volume"] < MIN_TRAILING_VOLUME[position_group], features] = np.nan

    return pos_df[["player_id", "player_name", "team", "season", "week", "trailing_volume"] + features].reset_index(drop=True)


def classify_archetype(style_features: pd.DataFrame, position_group: str) -> pd.DataFrame:
    """
    Assign each player-week a play-style archetype via nearest (weighted)
    centroid classification in league-relative z-score space -- see the
    module docstring for the full method and ARCHETYPE_CENTROIDS for what
    each archetype means.

    Args:
        style_features: Output of compute_style_features(df, position_group).
        position_group: "QB"/"RB"/"WR"/"TE" -- must match what
            compute_style_features() was called with (selects which
            ARCHETYPE_CENTROIDS/ARCHETYPE_AXIS_WEIGHTS/ARCHETYPE_FEATURES
            to use).

    Returns:
        DataFrame with one row per (player_id, season, week) that had
        enough trailing volume to classify (rows below
        MIN_TRAILING_VOLUME were already NaN'd out by
        compute_style_features() and are dropped here): player_name, team,
        archetype (one of this position's ARCHETYPE_CENTROIDS keys),
        archetype_distance (weighted distance to that archetype's centroid
        -- smaller means a more clear-cut fit; a player sitting roughly
        equidistant between two archetypes will have a comparatively large
        distance even to their "winning" one).

    Notes:
        Z-scoring is done separately WITHIN each (season, week) -- comparing
        a player's stats to the CURRENT league's players at the SAME
        position that same week, not a fixed historical baseline, so what
        counts as "aggressive" or "high-volume" adapts as league-wide style
        shifts year to year (see the module docstring). A (season, week)
        with fewer than 2 qualifying players at this position can't be
        z-scored meaningfully and is skipped.
    """
    if position_group not in ARCHETYPE_CENTROIDS:
        raise ValueError(f"Unknown position_group {position_group!r}; expected one of {list(ARCHETYPE_CENTROIDS)}")

    features = ARCHETYPE_FEATURES[position_group]
    centroids_by_name = ARCHETYPE_CENTROIDS[position_group]
    weights_by_name = ARCHETYPE_AXIS_WEIGHTS[position_group]
    names = list(centroids_by_name.keys())
    centroids = np.stack([centroids_by_name[n] for n in names])
    weights = np.stack([weights_by_name[n] for n in names])

    df = style_features.dropna(subset=features).copy()

    out_columns = ["player_id", "player_name", "team", "season", "week",
                   "archetype", "archetype_distance"]

    def _zscore_and_classify(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 2:
            return group.reindex(columns=[*group.columns, "archetype", "archetype_distance"]).iloc[0:0]
        z = (group[features] - group[features].mean()) / group[features].std().replace(0, 1)
        z_matrix = z.to_numpy()
        diff = z_matrix[:, None, :] - centroids[None, :, :]
        dists = np.sqrt((weights[None, :, :] * diff ** 2).sum(axis=2))
        best = dists.argmin(axis=1)
        group = group.copy()
        group["archetype"] = [names[i] for i in best]
        group["archetype_distance"] = dists[np.arange(len(dists)), best]
        return group

    # Iterated rather than `.apply()`: pandas 3.0 drops the grouping columns
    # from each sub-frame and removed `include_groups=True`, so an apply here
    # returns rows with no `season`/`week` at all. Iterating keeps every
    # column and behaves identically on pandas 2.x.
    classified = [_zscore_and_classify(group) for _, group in df.groupby(["season", "week"], sort=True)]
    classified = pd.concat(classified) if classified else pd.DataFrame(columns=out_columns)
    return classified[out_columns].reset_index(drop=True)


def build_defense_vs_archetype_profile(
    df: pd.DataFrame, archetypes: pd.DataFrame, position_group: str, window: int = ROLLING_DEFENSE_WINDOW
) -> pd.DataFrame:
    """
    For each (season, week, defense, archetype), the defense's trailing
    performance SPECIFICALLY against players of that archetype at this
    position -- "how does this defense do against Scramblers vs.
    Gunslingers" (QB) or "against Deep Threats vs. Possession receivers"
    (WR), not just against the position in general.

    Args:
        df: Weekly player stats (e.g. load_weekly_stats()'s output).
        archetypes: Output of classify_archetype(..., position_group) --
            each player-week's archetype at the time.
        position_group: "QB"/"RB"/"WR"/"TE".
        window: Games of trailing history per (defense, archetype) pair to
            average over (default ROLLING_DEFENSE_WINDOW). Since a defense
            typically faces any single archetype only a handful of times a
            season, this window will often span multiple seasons in
            practice -- effectively "trailing history," not "this season."

    Returns:
        DataFrame with one row per (season, week, team, archetype): team is
        the DEFENSE, archetype is the opposing player's. For QB, the metric
        column is passer_rating_allowed_vs_archetype (the standard QB
        efficiency metric). For RB/WR/TE, it's
        fantasy_points_allowed_vs_archetype (PPR fantasy points -- passer
        rating has no equivalent for non-QB positions, but fantasy points
        already exist as a common unit across all of them, computed the
        same trailing-sum-then-average way). Rows are only produced for
        (defense, archetype) pairs with at least one prior meeting -- a
        defense that's never faced a Deep Threat yet has no row for that
        combination (left for the caller's fillna/cold-start handling,
        same convention as the rest of this project).

    Notes:
        Leakage prevention: same shifted-rolling pattern as the rest of
        this module/PlayerProjections.py. For the QB/passer-rating path,
        ratings are computed from trailing AGGREGATED counts (summed
        first, then ONE rating computed from the totals), not by averaging
        already-computed per-game ratings -- see
        PlayerProjections.build_top_cb_profile()'s Notes for why that
        matters (averaging a rate stat like passer rating across games
        with different attempt volumes is statistically improper).
    """
    if position_group not in ARCHETYPE_CENTROIDS:
        raise ValueError(f"Unknown position_group {position_group!r}; expected one of {list(ARCHETYPE_CENTROIDS)}")

    if position_group == "QB":
        keep_cols = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions"]
    else:
        keep_cols = ["fantasy_points_ppr"]

    pos_games = df[df["position_group"] == position_group][["player_id", "season", "week", "opponent_team"] + keep_cols]
    tagged = pos_games.merge(
        archetypes[["player_id", "season", "week", "archetype"]], on=["player_id", "season", "week"], how="inner"
    )

    by_defense = tagged.rename(columns={"opponent_team": "team"}).sort_values(["team", "archetype", "season", "week"])

    trailing_cols = [f"trailing_{c}" for c in keep_cols]
    for raw_col, trailing_col in zip(keep_cols, trailing_cols):
        by_defense[trailing_col] = by_defense.groupby(["team", "archetype"])[raw_col].transform(
            lambda s: s.rolling(window, min_periods=1).sum().shift(1)
        )
    by_defense = by_defense.dropna(subset=trailing_cols)

    if position_group == "QB":
        by_defense["passer_rating_allowed_vs_archetype"] = nfl_passer_rating(
            by_defense["trailing_completions"], by_defense["trailing_passing_yards"],
            by_defense["trailing_passing_tds"], by_defense["trailing_passing_interceptions"], by_defense["trailing_attempts"],
        )
        metric_col = "passer_rating_allowed_vs_archetype"
    else:
        # Mean, not the raw trailing SUM -- fantasy_points_ppr doesn't have
        # a natural "per-attempt" denominator the way passer rating does,
        # so this reports it as an average allowed PER GAME instead.
        n_games = by_defense.groupby(["team", "archetype"]).cumcount()
        by_defense["fantasy_points_allowed_vs_archetype"] = by_defense["trailing_fantasy_points_ppr"] / n_games.clip(lower=1)
        metric_col = "fantasy_points_allowed_vs_archetype"

    return by_defense[["season", "week", "team", "archetype", metric_col]].reset_index(drop=True)


# Which build_defense_vs_archetype_profile() column carries the metric, per
# position -- used by PlayerProjections.py's integration so it doesn't need
# to hardcode the QB-vs-everyone-else branching itself.
ARCHETYPE_METRIC_COLUMN = {
    "QB": "passer_rating_allowed_vs_archetype",
    "RB": "fantasy_points_allowed_vs_archetype",
    "WR": "fantasy_points_allowed_vs_archetype",
    "TE": "fantasy_points_allowed_vs_archetype",
}


def compute_all_archetypes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Classify every position's players into archetypes in one call --
    convenience wrapper for PlayerProjections.py's integration, which needs
    all 4 positions' archetypes up front rather than computing QB's
    separately from RB's/WR's/TE's.

    Returns:
        {position_group: classify_archetype(compute_style_features(df, position_group), position_group)}
        for every position_group in POSITION_STAT_COLUMNS.
    """
    return {pos: classify_archetype(compute_style_features(df, pos), pos) for pos in POSITION_STAT_COLUMNS}


def compute_all_defense_vs_archetype_profiles(df: pd.DataFrame, archetypes_by_position: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build every position's defense-vs-archetype profile in one call -- see
    compute_all_archetypes().

    Args:
        archetypes_by_position: Output of compute_all_archetypes().

    Returns:
        {position_group: build_defense_vs_archetype_profile(df, archetypes_by_position[position_group], position_group)}
        for every position_group in POSITION_STAT_COLUMNS.
    """
    return {
        pos: build_defense_vs_archetype_profile(df, archetypes_by_position[pos], pos)
        for pos in POSITION_STAT_COLUMNS
    }


if __name__ == "__main__":
    from PlayerProjections import load_weekly_stats

    df = load_weekly_stats(years=[2023, 2024, 2025])
    for position_group in POSITION_STAT_COLUMNS:
        style = compute_style_features(df, position_group)
        archetypes = classify_archetype(style, position_group)
        latest = archetypes[archetypes["season"] == 2025].sort_values(["player_id", "week"]).groupby("player_id").tail(1)
        latest = latest.merge(
            df[df["season"] == 2025].groupby("player_id")[VOLUME_COLUMNS[position_group]].sum().sum(axis=1).reset_index(name="season_volume"),
            on="player_id",
        )
        starters = latest[latest["season_volume"] > MIN_TRAILING_VOLUME[position_group] * 3]
        print(f"\n=== 2025 {position_group} archetypes (season_volume > {MIN_TRAILING_VOLUME[position_group] * 3}) ===")
        print(starters[["player_name", "team", "archetype", "archetype_distance"]].sort_values(["archetype", "archetype_distance"]).to_string(index=False))
