"""
Checks for PlayerProjections.qb_starter_rows() and position_training_rows(),
the filter behind every starters_only option: walk_forward_train_and_predict()
and the four train_all_* functions that save models.

Synthetic, no network, no database.

Run: python3 test_qb_starter_rows.py
"""
import pandas as pd

from PlayerProjections import POSITION_STAT_COLUMNS, position_training_rows, qb_starter_rows


def _fixture():
    frame = pd.DataFrame({
        "player_id": ["qb_start", "qb_backup", "wr_1", "qb_start", "qb_other_team"],
        "season": [2024, 2024, 2024, 2024, 2024],
        "week": [1, 1, 1, 2, 1],
        "team": ["AAA", "AAA", "AAA", "AAA", "BBB"],
        "position_group": ["QB", "QB", "WR", "QB", "QB"],
    }, index=[10, 11, 12, 13, 14])
    schedule = pd.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [1, 2, 1],
        "team": ["AAA", "AAA", "BBB"],
        "starting_qb_id": ["qb_start", "qb_backup", "qb_other_team"],  # backup starts week 2
    })
    return frame, schedule


def demo():
    frame, schedule = _fixture()
    kept = qb_starter_rows(frame, schedule)
    assert list(kept.index) == [10, 12, 14], \
        f"expected the week-1 starter, the WR and BBB's starter only, got {list(kept.index)}"
    # qb_start's week-2 row goes: the backup was that week's listed starter.
    assert 13 not in kept.index, "a QB is a starter per week, not per season"
    assert "starting_qb_id" not in kept.columns, "the filter must not add columns"

    # Two listed starters for one team-week is a data error, not something to guess around.
    doubled = pd.concat([schedule, schedule.iloc[[0]].assign(starting_qb_id="qb_backup")])
    try:
        qb_starter_rows(frame, doubled)
        assert False, "expected a duplicate team-week starter to raise"
    except pd.errors.MergeError:
        pass
    print("OK -- non-starter QB rows are dropped per week, other positions are kept, "
          "and a duplicate listed starter raises.")


def demo_position_training_rows():
    frame, schedule = _fixture()
    assert position_training_rows(frame, schedule, "QB") is frame, "off by default"
    assert list(position_training_rows(frame, schedule, "QB", starters_only=True).index) == [10, 12, 14]
    assert position_training_rows(frame, schedule, "WR", starters_only=True) is frame, \
        "only QB has a listed starter; every other position trains on all its rows"
    print("OK -- position_training_rows filters QB only, and only when asked.")


def demo_train_all_functions_wire_the_flag():
    """Each train_all_* function must hand its trainer the filtered rows for QB
    when starters_only=True, and the full frame otherwise. Loaders, context
    builders, trainers and savers are stubbed, so this checks the wiring alone:
    no network, no training, nothing written."""
    import PlayerArchetypes as pa
    import PlayerProjections as pp
    import PlayerProjectionsTransformer as ppt

    from types import SimpleNamespace

    frame, schedule = _fixture()
    empty = pd.DataFrame()
    seen, saved = [], []

    def trainer(result):
        def train(position_group, df, *args, **kwargs):
            seen.append((position_group, list(df.index)))
            return result
        return train

    loaders = {"load_weekly_stats": lambda *a, **k: frame, "load_schedule_context": lambda *a, **k: schedule,
               "load_defensive_target_stats": lambda *a, **k: empty, "load_player_bio": lambda *a, **k: empty,
               "load_player_age": lambda *a, **k: empty, "load_injury_reports": lambda *a, **k: empty,
               "build_top_cb_profile": lambda *a, **k: None, "build_matchup_context": lambda *a, **k: None}
    stubs = [(pa, "compute_all_archetypes", lambda df: {g: None for g in POSITION_STAT_COLUMNS}),
             (pa, "compute_all_defense_vs_archetype_profiles", lambda df, a: {g: None for g in POSITION_STAT_COLUMNS}),
             (pp, "train_position_model", trainer((None, None, None, {}, [], []))),
             (pp, "train_position_model_ga", trainer(([], [], None, None, {}, [], []))),
             (pp, "save_model", lambda *a, **k: saved.append((a[5], k.get("starters_only", False)))),
             (pp, "save_population", lambda *a, **k: None),
             (ppt, "train_position_model_hf", lambda *a, **k: (seen.append((a[0], list(a[1].index))),
                                                              (SimpleNamespace(config=SimpleNamespace()), []))[1]),
             (ppt, "save_model_hf", lambda model, pg, *a, **k: saved.append((pg, model.config.starters_only)))]
    stubs += [(module, name, fn) for module in (pp, ppt) for name, fn in loaders.items()]
    originals = [(module, name, getattr(module, name)) for module, name, _ in stubs]
    try:
        for module, name, fn in stubs:
            setattr(module, name, fn)
        for train_all in (pp.train_all_position_models, pp.train_all_quantile_models,
                          pp.train_all_position_models_ga, ppt.train_all_position_models_hf):
            for flag, qb_rows in ((False, [10, 11, 12, 13, 14]), (True, [10, 12, 14])):
                seen.clear(); saved.clear()
                train_all(starters_only=flag)
                for position_group, recorded in saved:
                    assert recorded == (flag and position_group == "QB"), \
                        f"{train_all.__name__}(starters_only={flag}) saved {position_group} with starters_only={recorded}"
                assert seen, f"{train_all.__name__} never called its trainer"
                for position_group, rows in seen:
                    expected = qb_rows if position_group == "QB" else [10, 11, 12, 13, 14]
                    assert rows == expected, \
                        f"{train_all.__name__}(starters_only={flag}) gave {position_group} rows {rows}, expected {expected}"
    finally:
        for module, name, fn in originals:
            setattr(module, name, fn)
    print("OK -- all four train_all_* functions pass QB starters-only rows when asked, full rows "
          "otherwise, and record on the saved model which rule it was trained with.")


def demo_saved_model_records_its_training_rule():
    """save_model writes starters_only into the meta, trained_on_starters reads it,
    and a model saved before the flag existed reads False -- it was trained on every
    row, so prediction must keep feeding it every row."""
    import json, os, tempfile
    import numpy as np
    from PlayerProjections import Normalizer, PlayerStatLSTM, load_model, save_model, trained_on_starters

    model = PlayerStatLSTM(num_features=2, context_size=3, hidden_size=4, num_layers=1)
    norm = Normalizer(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
    context_norm = Normalizer(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        save_model(model, norm, context_norm, {}, ["a", "b"], "QB", 5, model_dir=tmp, starters_only=True)
        save_model(model, norm, context_norm, {}, ["a", "b"], "QB", 5, model_dir=tmp, model_name="qb_low")
        assert trained_on_starters("QB", tmp) is True
        assert trained_on_starters("QB", tmp, model_name="qb_low") is False, "off by default"
        load_model("QB", tmp, "cpu")  # the extra meta key does not break loading

        meta_path = os.path.join(tmp, "qb_meta.json")
        meta = json.load(open(meta_path))
        del meta["starters_only"]
        json.dump(meta, open(meta_path, "w"))
        assert trained_on_starters("QB", tmp) is False, "a pre-flag model must read as trained on every row"
    print("OK -- a saved model records its training rule, and a pre-flag model reads as all rows.")


if __name__ == "__main__":
    demo()
    demo_position_training_rows()
    demo_train_all_functions_wire_the_flag()
    demo_saved_model_records_its_training_rule()
