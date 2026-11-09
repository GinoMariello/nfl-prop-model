# Stage 4 design: chronological backtest

Settles every decision `src/evaluation/backtest.py` and `tests/test_backtest.py`
need, so implementing them is transcription. Requirements:
`docs/next-implementations.md` §4; `docs/model-plan.md` §§5-9;
`docs/experiment-protocol.md`.

## 1. What Stage 4 is, and is not

It walks the frozen fold rule over the development window, selects `C` on
validation log loss, refits, scores the outer season, and saves every
prediction with its provenance.

It is **not** the report. Calibration plots, the equal-season mean, the
2,000-replicate season-block bootstrap, and any cross-model claim beyond the
matched window belong to Stage 5, computed from `predictions.csv`.

It is **not** a final evaluation. There is no untouched holdout
(`experiment-protocol.md`, "Holdout designation"), so every number here is a
development number and the manifest says so.

## 2. Constants

```python
TRAIN_START     = 1999                       # model-plan.md §5
OUTER_SEASONS   = tuple(range(2005, 2025))   # 2005-2024
C_GRID          = (0.01, 0.1, 1.0, 10.0)     # ascending; experiment-protocol.md
VARIANTS        = ("team_only", "team_market")
FULL_COVERAGE_MODELS = ("home_frequency", "team_only")
MATCHED_MODELS       = ("home_frequency", "team_only", "team_market", "market")
TARGET          = "home_win"
ARTIFACT_ROOT   = "models_game"
```

`FULL_COVERAGE_MODELS` is **declared, not derived**. Deriving it from "which
models happened to have no abstentions" would promote `market` to full coverage
in any season where every game was quoted — exactly what §6 forbids.

## 3. `load_matchups(snapshot_dir)`

`verify_snapshot` once, then read `games` and `team_game_stats` through
`open_snapshot`'s `mode=ro` connection and **rebuild the matchup table in
memory**, mirroring `rolling_features.run()` minus its writes.

The snapshot's own `matchup_features` table is **not** read. Stage 1 already
says derived tables are non-authoritative; measured, this one is also stale —
62 columns against the current builder's 58, carrying `point_diff`,
`home_rest_days` and raw moneyline columns, and missing `neutral_site`
entirely. Nothing would raise: every allowlisted feature is present, so reading
it would train on old rolling values and silently drop the neutral-site flag.
A guard raises if the rebuild lacks `neutral_site` or `home_gameday`.

## 4. `make_folds(matchups, outer_seasons)`

Population is eligible rows only (`home_win.notna()`); variant filtering stays
in `select_features`, so one fold serves both variants.

For outer season T: train `1999..T-2`, validate `T-1`, test `T`
(`model-plan.md` §5).

Validation order, each message distinct: missing columns → null/duplicate
`game_id` → empty `outer_seasons` → excluded season named → unparseable
`gameday` → empty partition → overlapping partitions → not strictly ordered.

`_assert_partitions_ordered` is called unconditionally — there is no flag to
skip it. It checks disjointness **and** `max(train) < min(valid)` and
`max(valid) < min(test)` on **parsed dates**, never season numbers
(`next-implementations.md` L60-62). Seasons in this snapshot happen to be
date-separated by at least 241 days, which is why the assertion passes today —
not a reason to assume it.

Fold fields are tuples: immutable after the assertion passed, and
JSON-serializable.

## 5. `select_c(train, valid, variant, c_grid)`

**Takes frames, not `(matchups, fold)`.** The test rows are never in scope, so
no code path exists by which an outer-test outcome reaches `C`. That is the
structural half of §9 gate 4; `tests/test_backtest.py` verifies it empirically
on top.

Per candidate: `fit_model` → `predict_model` → `predictions_for_scoring` →
restrict labels to the scorable index → `align_by_game_id` → `log_loss`.
Abstention shrinkage is recorded (`n_valid_eligible` vs `n_valid_scored`), not
hidden.

Grid ascending with strict `<` keeps the smaller `C` on a tie — the protocol's
rule, with no tie-breaking branch.

## 6. Unfittable folds

Each `(T, variant)` is wrapped in `except ValueError` — only `ValueError`,
which is what `train.py` raises for an unfittable fold. A `KeyError` or
`TypeError` is a bug and must crash.

On catch: record `(season, variant, stage, error, n_train_eligible,
n_train_paired)` and continue. The skip is per-variant, so `team_only`,
`home_frequency` and `market` still produce predictions for that season.

Measured: exactly three failures, `team_market` at 2005/2006/2007, message
`cannot fit team_market: no rows with a paired market quote` — there are no
paired moneylines before 2006, so those folds train on zero quoted rows.
`run_backtest` does not assert that (it is data-dependent); the real-data check
pins it.

## 7. One scoring path, and the coverage rule

Two populations per season: `full` (all eligible test rows) and `matched`
(paired rows). Declared membership per §2.

Every model — fitted, constant, or market quote — reaches the metrics through
the identical funnel: Stage 3's three-column frame → `predictions_for_scoring`
→ `align_by_game_id`. `_as_prediction_frame` builds that shape for the
baselines; `predict_model` already returns it.

Labels are **not** restricted here, unlike validation scoring: the population
is fixed before the model is consulted, so a declared model that abstains
inside its own population makes `align_by_game_id` raise. That is the guard
that stops a full-coverage model from being quietly scored on a subset
(`model-plan.md` §6: "Do not compare one model's full sample against another's
restricted sample").

`comparison_seasons` — the seasons where every model declared for `matched`
produced predictions — is **derived from coverage, not hardcoded**, and is the
only window in which a model-vs-model claim is permitted. Measured: 2008-2024.

`always_home` gets one row per (season, population) carrying `accuracy` only,
with NaN in every probability column: §6 calls it "accuracy only, not a
probability baseline (B7)", and a NaN is how that stays visible in the data.

## 8. Neutral sites

**Scored normally, flag carried to `predictions.csv`.** Not excluded.

Excluding 57 of 5170 eligible games would make the backtest score a different
population than Stage 0 measured, breaking the 5170/4780 cross-check, to chase
an effect `_neutral_site`'s own docstring records as "a direction, not a
measured effect" at n=58. The flag costs nothing to carry and lets Stage 5
slice it if the count keeps growing.

## 9. Artifacts

`models_game/home_win/<run_id>/`, per the adjudication in
`stage3-design.md` §2.3: the roadmap's later decision includes `<target>`.
`mkdir(exist_ok=False)` is §8's "never overwriting earlier runs".

```
manifest.json      snapshot sha256, matchup content_hash, git, versions,
                   protocol block, population membership, comparison_seasons,
                   neutral_site_policy, per-season counts, failures,
                   development-not-final label
folds.csv          outer_season, partition, game_id, gameday
fold_summary.csv   per fold: seasons, date boundaries, counts
candidates.csv     season, variant, C, valid_log_loss, counts, converged, n_iter
selection.csv      the chosen C per (season, variant)
predictions.csv    one row per (model, eligible test game)
metrics.csv        season, population, model, n, log_loss, brier, accuracy, roc_auc
failures.csv       the unfittable folds
<variant>/logistic/season_<T>.joblib
```

The baselines get no model file: home-frequency is one float and the market is
not fitted, so both live in the manifest. A `none/` path component for a model
with no feature set would be a directory nothing reads.

## 10. Tests

`tests/test_backtest.py`, plain `assert` + `demo_*` + `__main__`.

| Test | Proves | Non-vacuity |
|---|---|---|
| `demo_folds_partition_exactly_and_in_order` | disjoint, exhaustive, strictly ordered | move a training game into the test season → must raise; move one within its own season → must not |
| `demo_outer_test_outcomes_cannot_reach_selection_or_the_fit` | §9 gate 4 | corrupt validation labels → candidate scores *must* move; corrupt training features → `coef_` must move |
| `demo_unfittable_fold_is_reported_not_crashed` | §6 handling | fill the quote gap → failures empty, coverage widens |
| `demo_populations_never_mix_coverage` | the §6 rule | NaN one full-coverage prediction → `align_by_game_id` must raise |
| `demo_home_frequency_is_fitted_once` | fitted once, never refit on a subset | mutate test labels → unchanged; mutate training labels → changes |
| `demo_malformed_input_fails_with_a_useful_message` | each guard in §4 | every message distinct |
| `demo_artifacts_round_trip_and_never_overwrite` | replay within 1e-8 | second save to the same dir raises |

Real-data check (`--snapshot`): exactly 3 failures at 2005/2006/2007; summed
`full` n == 5170 and `matched` n == 4780; `comparison_seasons` == 2008-2024;
matched n == 4316 per model with identical ID sets; 57 neutral-site games
present and flagged.

## 11. Out of scope

Stage 5 owns the report, bootstrap intervals, and calibration plots. Stage 6
owns final evaluation and the prediction interface, and is blocked on a holdout
that does not exist. The C grid, feature allowlist, and calibration policy are
frozen — changing any is a new protocol version.

No fold-strategy abstraction (one rule, frozen), no configurable
`TRAIN_START`, no per-fold seed (the protocol specifies one global seed).
