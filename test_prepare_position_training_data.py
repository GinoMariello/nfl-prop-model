"""
Regression check for prepare_position_training_data() -- the data-prep block
extracted out of train_position_model(), train_position_model_ga(), and
PlayerProjectionsTransformer.train_position_model_hf() (previously
copy-pasted identically in all three).

Loads real cached rows directly from data/nfl_stats.db (bypassing the
load_*() wrappers, which refresh over the network) and asserts the helper's
output matches the same steps run inline by hand -- i.e. the extraction
didn't silently change anything (wrong arg order, dropped line, etc.).

Run: python3 test_prepare_position_training_data.py
"""
import numpy as np

from PlayerProjections import (
    DB_PATH, POSITION_STAT_COLUMNS, POSITION_RANKING_STAT, TEAM_CONTEXT_COLUMNS, WINDOW_SIZE,
    query_sqlite, build_top_cb_profile, build_matchup_context,
    build_player_team_rank_profile, build_size_diff_profile, build_age_profile,
    build_injury_profile, build_defense_profile, build_team_game_log,
    build_opponent_difficulty_profile, build_archetype_context, build_sequences,
    chronological_split, fit_normalizer, compute_recency_weights, fillna_context,
    prepare_position_training_data,
)
from PlayerArchetypes import compute_all_archetypes, compute_all_defense_vs_archetype_profiles

YEARS = [2023, 2024, 2025]
POSITION_GROUP = "QB"


def load_cached(table, years=None):
    if years is None:
        return query_sqlite(f"SELECT * FROM {table}", db_path=DB_PATH)
    placeholders = ", ".join("?" * len(years))
    return query_sqlite(f"SELECT * FROM {table} WHERE season IN ({placeholders})", db_path=DB_PATH, params=years)


def reference_prepare(position_group, df, context_df, top_cb_profile, player_bio, player_age, injury_reports,
                       archetypes, defense_vs_archetype, window_size, val_frac):
    """Hand-written copy of the pre-refactor inline block -- ground truth."""
    stat_columns = POSITION_STAT_COLUMNS[position_group]
    filled_context_df, context_fill_values = fillna_context(context_df)
    context_lookup = {
        (int(row.season), int(row.week), row.team): np.array([getattr(row, c) for c in TEAM_CONTEXT_COLUMNS], dtype=np.float32)
        for row in filled_context_df.itertuples()
    }
    pos_df = df[df["position_group"] == position_group]
    rank_stat = POSITION_RANKING_STAT[position_group]
    rank_profile = build_player_team_rank_profile(pos_df, rank_stat)
    pos_df = pos_df.merge(rank_profile, on=["player_id", "season", "week"], how="left")
    size_diff_profile = build_size_diff_profile(pos_df, top_cb_profile, player_bio)
    pos_df = pos_df.merge(size_diff_profile, on=["player_id", "season", "week"], how="left")
    age_profile = build_age_profile(pos_df, player_age)
    pos_df = pos_df.merge(age_profile, on=["player_id", "season", "week"], how="left")
    injury_profile = build_injury_profile(pos_df, injury_reports)
    pos_df = pos_df.merge(injury_profile, on=["player_id", "season", "week"], how="left")
    defense_profile = build_defense_profile(build_team_game_log(df))
    difficulty_profile = build_opponent_difficulty_profile(pos_df, defense_profile, position_group, rank_stat)
    pos_df = pos_df.merge(difficulty_profile, on=["player_id", "season", "week"], how="left")
    pos_df["player_vs_opponent_difficulty"] = pos_df["player_vs_opponent_difficulty"].fillna(pos_df[rank_stat].mean())
    archetype_context = build_archetype_context(pos_df, archetypes, defense_vs_archetype, position_group)
    pos_df = pos_df.merge(archetype_context, on=["player_id", "season", "week"], how="left")
    X, C, y, meta = build_sequences(pos_df, stat_columns, context_lookup, window_size)
    train_idx, val_idx = chronological_split(meta, val_frac)
    X_train, C_train, y_train = X[train_idx], C[train_idx], y[train_idx]
    X_val, C_val, y_val = X[val_idx], C[val_idx], y[val_idx]
    normalizer = fit_normalizer(np.concatenate([X_train.reshape(-1, X_train.shape[-1]), y_train], axis=0))
    context_normalizer = fit_normalizer(C_train)
    recency_weights = compute_recency_weights(meta)
    weights_train, weights_val = recency_weights[train_idx], recency_weights[val_idx]
    return (stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
            normalizer, context_normalizer, context_fill_values, weights_train, weights_val)


def demo():
    df = load_cached("player_weekly_stats", YEARS)
    df = df[df["season_type"] == "REG"].reset_index(drop=True)
    player_bio = load_cached("player_bio")
    player_age = load_cached("weekly_player_age", YEARS)
    injury_reports = load_cached("weekly_injury_reports", YEARS)
    schedule_context = load_cached("schedule_context", YEARS)
    schedule_context = schedule_context[schedule_context["game_type"] == "REG"].reset_index(drop=True)
    defensive_target_stats = load_cached("weekly_defensive_target_stats", YEARS)

    top_cb_profile = build_top_cb_profile(defensive_target_stats, player_bio)
    context_df = build_matchup_context(df, top_cb_profile, schedule_context, injury_reports, player_bio)
    archetypes = compute_all_archetypes(df)
    defense_vs_archetype = compute_all_defense_vs_archetype_profiles(df, archetypes)
    archetypes_pos = archetypes[POSITION_GROUP]
    defense_vs_archetype_pos = defense_vs_archetype[POSITION_GROUP]

    args = (POSITION_GROUP, df, context_df, top_cb_profile, player_bio, player_age,
            injury_reports, archetypes_pos, defense_vs_archetype_pos, WINDOW_SIZE, 0.15)

    got = prepare_position_training_data(*args)
    want = reference_prepare(*args)

    names = ["stat_columns", "X_train", "C_train", "y_train", "X_val", "C_val", "y_val",
             "normalizer", "context_normalizer", "context_fill_values", "weights_train", "weights_val"]
    for name, g, w in zip(names, got, want):
        if name in ("normalizer", "context_normalizer"):
            assert np.allclose(g.mean, w.mean) and np.allclose(g.std, w.std), f"{name} mismatch"
        elif name == "context_fill_values":
            assert g == w, f"{name} mismatch"
        elif name == "stat_columns":
            assert g == w, f"{name} mismatch"
        else:
            assert np.array_equal(g, w), f"{name} mismatch"

    assert len(got[1]) + len(got[4]) >= 20, "sanity: too few sequences to be a meaningful test"
    print(f"OK -- prepare_position_training_data() matches reference on {len(got[1]) + len(got[4])} real {POSITION_GROUP} sequences "
          f"({len(got[1])} train / {len(got[4])} val).")


if __name__ == "__main__":
    demo()
