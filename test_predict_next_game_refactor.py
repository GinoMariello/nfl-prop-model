"""
Regression check for the predict_next_game()/predict_next_game_hf()
inference refactor: both used to copy-paste an identical ~45-line
feature-build+forward+inverse-transform block; it's now shared via
_predict_next_game_core(). This loads a real saved model and real cached
data (no network) and checks predict_next_game()'s output is byte-identical
to a hand-written copy of the pre-refactor inline logic.

Run: python3 test_predict_next_game_refactor.py
"""
import numpy as np
import torch

from PlayerProjections import (
    CONTEXT_COLUMNS, DB_PATH, MODEL_DIR, VEGAS_CONTEXT_COLUMNS, query_sqlite,
    build_team_game_log, current_matchup_context, load_model, predict_next_game,
    _predict_next_game_core, qb_starter_rows,
)
from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

YEARS = [2023, 2024, 2025]
PLAYER_ID = "00-0034857"  # J.Allen, 66 cached REG games
NEXT_OPPONENT = "MIA"
POSITION_GROUP = "QB"


def load_cached(table, years=None, type_col=None, type_val=None):
    if years is None:
        df = query_sqlite(f"SELECT * FROM {table}", db_path=DB_PATH)
    else:
        placeholders = ", ".join("?" * len(years))
        df = query_sqlite(f"SELECT * FROM {table} WHERE season IN ({placeholders})", db_path=DB_PATH, params=years)
    if type_col:
        df = df[df[type_col] == type_val].reset_index(drop=True)
    return df


def reference_predict_next_game(player_id, next_opponent, df, defensive_target_stats, player_bio, schedule_context,
                                 player_age, injury_reports, position_group, model_dir, device):
    """Hand-written copy of the pre-refactor inline body -- ground truth."""
    player_rows = df[df["player_id"] == player_id].sort_values(["season", "week"])
    archetypes = classify_archetype(compute_style_features(df, position_group), position_group)
    defense_vs_archetype = build_defense_vs_archetype_profile(df, archetypes, position_group)

    model, normalizer, context_normalizer, context_fill_values, stat_columns, window_size = load_model(
        position_group, model_dir, device, model_name=None
    )
    recent = player_rows.tail(window_size)[stat_columns].fillna(0.0).to_numpy(dtype=np.float32)
    x = normalizer.transform(recent)

    team = player_rows["team"].iloc[-1]
    team_games = build_team_game_log(df)
    context_vec = current_matchup_context(
        df, team_games, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports,
        archetypes, defense_vs_archetype,
        player_id, position_group, team, next_opponent, context_fill_values, injury_status=None,
    )
    if len(context_normalizer.mean) == len(CONTEXT_COLUMNS) - len(VEGAS_CONTEXT_COLUMNS):
        # Added with the Vegas features: a model saved before them gets the inputs it was trained on.
        context_vec = context_vec[[i for i, col in enumerate(CONTEXT_COLUMNS) if col not in VEGAS_CONTEXT_COLUMNS]]
    c = context_normalizer.transform(context_vec.reshape(1, -1))[0]

    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
    c_tensor = torch.from_numpy(c).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_normalized = model(x_tensor, c_tensor).cpu().numpy()[0]
    pred = normalizer.inverse_transform(pred_normalized)
    pred = np.clip(pred, a_min=0, a_max=None)
    return {col: float(val) for col, val in zip(stat_columns, pred)}


def demo():
    device = "cpu"
    df = load_cached("player_weekly_stats", YEARS, "season_type", "REG")
    defensive_target_stats = load_cached("weekly_defensive_target_stats", YEARS)
    player_bio = load_cached("player_bio")
    schedule_context = load_cached("schedule_context", YEARS, "game_type", "REG")
    player_age = load_cached("weekly_player_age", YEARS)
    injury_reports = load_cached("weekly_injury_reports", YEARS)

    got = predict_next_game(
        PLAYER_ID, NEXT_OPPONENT, df=df, defensive_target_stats=defensive_target_stats, player_bio=player_bio,
        schedule_context=schedule_context, player_age=player_age, injury_reports=injury_reports,
        position_group=POSITION_GROUP, model_dir=MODEL_DIR, device=device,
    )
    want = reference_predict_next_game(
        PLAYER_ID, NEXT_OPPONENT, df, defensive_target_stats, player_bio, schedule_context,
        player_age, injury_reports, POSITION_GROUP, MODEL_DIR, device,
    )

    assert got.keys() == want.keys(), "stat_columns mismatch"
    for k in got:
        assert got[k] == want[k], f"{k}: refactored={got[k]!r} reference={want[k]!r}"

    print(f"OK -- predict_next_game() matches reference on real J.Allen -> MIA prediction ({len(got)} stats): {got}")


def demo_starters_only_model_is_fed_starts():
    """A QB model trained on starts must predict from the player's last starts, not
    his last appearances; a model trained on every row keeps the old behaviour.
    Uses a real QB whose recent appearances include a non-start, and a loader that
    reports either training rule, recording the exact input sequence the model sees."""
    df = load_cached("player_weekly_stats", YEARS, "season_type", "REG")
    schedule_context = load_cached("schedule_context", YEARS, "game_type", "REG")
    starts = qb_starter_rows(df, schedule_context)
    qbs = df[df["position_group"] == "QB"].sort_values(["season", "week"])
    candidate = None
    for player_id, rows in qbs.groupby("player_id"):
        recent = rows.tail(5)
        player_starts = starts[starts["player_id"] == player_id]
        if len(player_starts) >= 5 and len(recent.merge(player_starts, on=["season", "week"])) < 5:
            candidate = player_id
            break
    assert candidate is not None, "need a QB whose last five appearances include a non-start"

    model, normalizer, context_normalizer, fill_values, stat_columns, window_size = load_model("QB", MODEL_DIR, "cpu")
    common = dict(df=df, defensive_target_stats=load_cached("weekly_defensive_target_stats", YEARS),
                  player_bio=load_cached("player_bio"), schedule_context=schedule_context,
                  player_age=load_cached("weekly_player_age", YEARS),
                  injury_reports=load_cached("weekly_injury_reports", YEARS), position_group="QB", device="cpu")
    for flag in (True, False):
        fed = []

        def forward(m, x, c):
            fed.append(x[0].numpy())
            return m(x, c)

        loader = lambda pg: (model, stat_columns, window_size, normalizer, context_normalizer, fill_values, forward, flag)
        _predict_next_game_core(candidate, "MIA", loader, **common)
        history = starts if flag else df
        rows = history[history["player_id"] == candidate].sort_values(["season", "week"]).tail(window_size)
        expected = normalizer.transform(rows[stat_columns].fillna(0.0).to_numpy(dtype=np.float32))
        np.testing.assert_allclose(fed[0], expected, rtol=1e-6, err_msg=f"starters_only={flag}: wrong input sequence")

    print(f"OK -- a starters-only QB model is fed {candidate}'s last {window_size} starts; "
          f"an all-rows model is fed his last {window_size} appearances.")


if __name__ == "__main__":
    demo()
    demo_starters_only_model_is_fed_starts()
