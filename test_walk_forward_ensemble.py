"""
Check for walk_forward_ensemble_predict() in PlayerProjections.py: the seed
ensemble behind live projections.

Synthetic, no network, no database, no training.

Run: python3 test_walk_forward_ensemble.py
"""
import numpy as np
import pandas as pd
import torch

import PlayerProjections as pp


def demo_ensemble_seeds_each_run_and_averages():
    """Stub the trainer so each seed's 'prediction' is the seed itself: the
    ensemble must seed every run, average them, and leave actual_* alone."""
    real = pp.walk_forward_train_and_predict
    seen = []

    def stub(*args, **kwargs):
        seed = torch.initial_seed()
        seen.append(seed)
        return pd.DataFrame({"player_id": ["a", "b"], "pred_passing_yards": [float(seed), 10.0 + seed],
                             "actual_passing_yards": [7.0, 8.0]})

    pp.walk_forward_train_and_predict = stub
    try:
        out = pp.walk_forward_ensemble_predict(*[None] * 6, "QB", 2025, 3, seeds=(1, 2, 3))
        assert seen == [1, 2, 3], f"each run must be seeded, got {seen}"
        assert out["pred_passing_yards"].tolist() == [2.0, 12.0], "mean of seeds 1, 2, 3"
        assert out["actual_passing_yards"].tolist() == [7.0, 8.0], "outcomes are never averaged"

        # Different players across seeds would silently misalign the average.
        pp.walk_forward_train_and_predict = lambda *a, **k: pd.DataFrame(
            {"player_id": ["a"] if torch.initial_seed() == 1 else ["b"], "pred_passing_yards": [1.0]})
        try:
            pp.walk_forward_ensemble_predict(*[None] * 6, "QB", 2025, 3, seeds=(1, 2))
            assert False, "mismatched player sets must raise"
        except ValueError:
            pass
        try:
            pp.walk_forward_ensemble_predict(*[None] * 6, "QB", 2025, 3, seeds=())
            assert False, "an empty seed list must raise"
        except ValueError:
            pass
    finally:
        pp.walk_forward_train_and_predict = real
    print("OK -- the ensemble seeds each run, averages predictions only, and refuses mismatched or empty runs.")


def demo_live_targets_use_only_pregame_facts():
    """QB targets come from the listed starter; other groups from each team's
    last game before the week, never from the week itself."""
    schedule = pd.DataFrame({"season": [2026] * 3, "week": [6] * 3, "team": ["KC", "LV", "NYJ"],
                             "opponent": ["LV", "KC", "BYE?"], "starting_qb_id": ["qb1", "qb2", None]})
    stats = pd.DataFrame({
        "season": [2026] * 5, "week": [4, 5, 5, 4, 6], "team": ["KC", "KC", "KC", "LV", "LV"],
        "position_group": ["WR"] * 5, "player_id": ["old", "w1", "w2", "w3", "future"]})
    assert pp.live_targets(stats, schedule, "QB", 2026, 6) == [("qb1", "LV"), ("qb2", "KC")], \
        "unlisted starters are skipped, not guessed"
    assert pp.live_targets(stats, schedule, "WR", 2026, 6) == [("w1", "LV"), ("w2", "LV"), ("w3", "KC")], \
        "last game before the week only; week 6 rows must not leak in"
    print("OK -- live targets come from listed starters and each team's last completed game.")


def demo_attach_outcomes_scores_without_leaking():
    """l5_ averages the last 5 games strictly before the week; actual_ is that
    week's line; a week not yet played gets no actual_ columns."""
    games = [(2026, w) for w in range(1, 8)]
    df = pd.DataFrame({"season": [s for s, _ in games], "week": [w for _, w in games], "team": "KC",
                       "position_group": "WR", "player_id": "p", "receiving_yards": [10, 20, 30, 40, 50, 60, 999]})
    out = pd.DataFrame({"player_id": ["p"], "pred_receiving_yards": [55.0]})
    scored = pp.attach_outcomes(out, df, pd.DataFrame(), "WR", 2026, 7)
    assert scored["l5_receiving_yards"].tolist() == [40.0], "weeks 2-6 only; week 7 must not leak in"
    assert scored["actual_receiving_yards"].tolist() == [999]
    future = pp.attach_outcomes(out, df, pd.DataFrame(), "WR", 2026, 8)
    assert "actual_receiving_yards" not in future and future["l5_receiving_yards"].tolist() == [235.8]
    print("OK -- outcomes attach the week's actuals and a strictly-earlier last-5 baseline.")


def demo_blend_with_last5():
    """pred_ becomes the weighted mix, the model's number survives as model_, and
    a player with no last-5 history keeps the model's projection."""
    out = pd.DataFrame({"player_id": ["a", "b"], "pred_receptions": [4.0, 6.0], "l5_receptions": [8.0, None]})
    blended = pp.blend_with_last5(out, 0.75)
    assert blended["pred_receptions"].tolist() == [7.0, 6.0]
    assert blended["model_receptions"].tolist() == [4.0, 6.0]
    assert out["pred_receptions"].tolist() == [4.0, 6.0], "input left untouched"
    print("OK -- the last-5 blend mixes pred_, keeps the model's number, and falls back without history.")



def demo_rest_of_season():
    """Per-game mean_ x expected games (team games left past byes x availability, adjusted by the
    player's prior games-played share) + actual before it."""
    import tempfile
    sched = pd.DataFrame({"season": 2026, "week": [5, 6, 7, 5, 7], "team": ["KC", "KC", "KC", "LV", "LV"]})  # LV bye in 6
    hist = pd.DataFrame({"season": 2026, "week": [3, 4, 4], "position_group": "WR", "player_id": ["a", "a", "b"],
                         "team": ["KC", "KC", "LV"], **{c: [5.0, 7.0, 2.0] for c in pp.POSITION_STAT_COLUMNS["WR"]}})
    # Prior seasons: a played every 2025 game (rate 1.0); b has no history (rookie).
    prior_sched = pd.DataFrame({"season": 2025, "week": [1, 2], "team": ["KC", "KC"]})
    prior_hist = pd.DataFrame({"season": 2025, "week": [1, 2], "player_id": "a", "team": "KC", "position_group": "WR"})
    real = pp.load_schedule_context, pp.load_weekly_stats, pp.get_sleeper_injuries
    pp.get_sleeper_injuries = lambda db_path=None: pd.DataFrame({"player_id": ["a"], "injury_status": ["Out"]})
    pp.load_schedule_context = lambda years: sched if 2026 in years else prior_sched
    pp.load_weekly_stats = lambda years: hist if 2026 in years else prior_hist
    try:
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"player_id": ["a", "b"], "player_name": ["A", "B"],
                          **{f"pred_{c}": [1.0, 1.0] for c in pp.POSITION_STAT_COLUMNS["WR"]},
                          "pred_fantasy_points_ppr": [10.0, 4.0],  # a: top half, b: bottom half
                          "mean_receptions": [2.0, 3.0]}).to_csv(f"{d}/2026_w05_WR.csv", index=False)
            r = pp.rest_of_season("WR", 2026, 5, save_dir=d).set_index("player_id")
    finally:
        pp.load_schedule_context, pp.load_weekly_stats, pp.get_sleeper_injuries = real
    assert r.games_left.to_dict() == {"a": 3, "b": 2}, "LV's bye in week 6 is skipped"
    top, bottom = pp.AVAILABILITY["WR"]
    h = pp.AVAILABILITY_HISTORY
    m = pp.INJURY_MULTIPLIER  # a is listed Out right now; b is off the report
    rate_a = (top + h["intercept"] + h["rate_coef"] * 1.0) * m["Out"]
    rate_b = min(1.0, (bottom + h["intercept"] + h["no_history"]) * m["healthy"])
    assert r.injury_status.tolist() == ["Out", "healthy"]
    assert r.prior_games_share.iloc[0] == 1.0 and pd.isna(r.prior_games_share.iloc[1])
    assert np.allclose(r.availability, [rate_a, rate_b]), "tier rate x own history x current injury news"
    assert np.allclose(r.expected_games, [3 * rate_a, 2 * rate_b])
    assert np.allclose(r.ros_receptions, [2.0 * 3 * rate_a, 3.0 * 2 * rate_b]), "mean_ is the per-game number when present"
    assert np.allclose(r.season_receptions, [12.0 + 6 * rate_a, 2.0 + 6 * rate_b]), "actual so far is added"
    assert np.allclose(r.season_healthy_fantasy_points_ppr, [12.0 + 30.0, 2.0 + 8.0]), "healthy total plays every game"
    assert "ros_wopr" not in r
    print("OK -- rest-of-season totals discount games left by tier, own history and current injury news, prefer mean_, and add the actual so far.")



def demo_drop_injured():
    targets = [("a", "KC"), ("b", "KC"), ("c", "LV"), ("d", "LV")]
    news = pd.DataFrame({"player_id": ["a", "b", "c", "e"], "injury_status": ["Out", "Questionable", "IR", "PUP"]})
    kept, dropped = pp.drop_injured(targets, news)
    assert kept == [("b", "KC"), ("d", "LV")] and dropped == ["a", "c"], "Out/IR leave; Questionable and unlisted stay"
    print("OK -- live targets drop players listed Out/IR/PUP and keep Questionable ones.")


if __name__ == "__main__":
    demo_ensemble_seeds_each_run_and_averages()
    demo_live_targets_use_only_pregame_facts()
    demo_attach_outcomes_scores_without_leaking()
    demo_blend_with_last5()
    demo_rest_of_season()
    demo_drop_injured()
