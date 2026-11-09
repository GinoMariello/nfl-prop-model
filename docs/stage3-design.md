# Stage 3 design: leakage-safe preprocessing and logistic regression

Implementation spec for `docs/next-implementations.md` Stage 3. Decisions are
settled here so implementation is transcription. Cited where a rule comes from
an earlier agreed document; marked explicitly where this document decides.

**Files touched:** `src/models/train.py` (new), `tests/test_train.py` (new).
Nothing else.

## 1. What Stage 3 is, and what it deliberately is not

Stage 3 builds the fitted model: preprocessing plus L2 logistic regression over
`select_features()`'s output, for both feature variants. It does **not** choose
`C`, build folds, or score anything — Stage 4 owns tuning and the chronological
backtest, Stage 5 owns reporting. `C` is exposed as configuration and nothing
in this stage reads a validation score.

The whole difficulty is that preprocessing *learns* from data — medians and
scales are fitted parameters. Every one of them must come from training rows
only, or the model has seen its own test set. That is the single invariant this
stage exists to enforce.

## 2. Public API

```python
RANDOM_SEED = 20260915      # docs/experiment-protocol.md, "Seeds"
DEFAULT_C   = 1.0           # Stage 4 tunes over {0.01, 0.1, 1, 10}
SOLVER      = "lbfgs"
MAX_ITER    = 1000
TOL         = 1e-4

def fit_model(matchups, variant="team_only", C=DEFAULT_C) -> dict
def predict_model(bundle, matchups) -> pd.DataFrame
def save_model(bundle, path) -> Path
def load_model(path) -> dict
```

A "bundle" is a plain dict holding the fitted pipeline together with everything
needed to reproduce and validate its use. Preprocessing and model are saved and
loaded as one object — never separately, since a scaler without its model, or a
model without its scaler, is a silent wrong answer waiting to happen.

### 2.1 `fit_model(matchups, variant, C) -> dict`

`matchups` is the matchup table (must carry `game_id`, `home_win`, and the
variant's feature columns). Steps, in order:

1. `X = select_features(matchups, variant)` — the allowlist is the only way
   features enter. For `team_market` this already drops rows without a paired
   quote, since `select_features` filters on
   `market_home_probability.between(0, 1, inclusive="neither")`.
2. Align labels by the index `select_features` preserves:
   `y = matchups.loc[X.index, "home_win"]`.
3. Drop rows whose label is null (unplayed, tied, undated), then validate the
   remainder through the shared label gate.
4. Fail loudly if what remains is not trainable (§5).
5. Fit the pipeline (§3) on those rows only.

Returns:

```python
{
  "pipeline": Pipeline,              # fitted preprocessing + estimator
  "variant": "team_only" | "team_market",
  "feature_schema": [...],           # ordered input columns, from select_features
  "config": {"C":…, "solver":…, "random_state":…, "tol":…, "max_iter":…},
  "n_train": int,
  "n_positive": int,
  "converged": bool,
  "n_iter": int,
}
```

`converged`/`n_iter` come from the estimator's `n_iter_` and from catching
`ConvergenceWarning` during fit — Stage 3's deliverable says "record solver,
seed, tolerance, iteration limit, convergence status", so a non-converged fit
is recorded rather than silently accepted. It is **not** an error: Stage 5
reports it.

### 2.2 `predict_model(bundle, matchups) -> pd.DataFrame`

Returns **one row per input game**, never a filtered subset:

| column | meaning |
|---|---|
| `game_id` | as given |
| `p_home` | predicted `P(home_win)`, or `NaN` when abstaining |
| `abstention_reason` | `None` when scored; a string when not |

The only abstention in v1 is `"no_paired_market_quote"`, for `team_market` rows
`select_features` drops. Missing *team* features are never an abstention — they
are imputed and flagged (§3). This satisfies "preserve the abstention reason
alongside every output row"; returning only scored rows would silently change
the population a caller thinks it is scoring, which is exactly the coverage
confusion `model-plan.md` §6 warns against.

Schema is checked before predicting (§6).

### 2.3 `save_model` / `load_model`

`joblib.dump`/`load` of the whole bundle dict (joblib is already a resolved
dependency via scikit-learn; no new requirement), after `mkdir(parents=True)`.

`save_model` refuses an existing path unless called with `replace=True`.
The plan puts artifacts under
`models_game/<target>/<run_id>/<feature_set>/<model>/`, "never overwriting
earlier runs" — and Stage 4 fits one model per fold per C value, so a run_id that
repeats (a re-run, a crashed run restarted, a fold loop that forgets to vary
the path) would otherwise replace an earlier run's model in place and leave
its recorded metrics pointing at a file that no longer holds the model that
produced them.

**Which path, exactly** — the two planning documents disagree, so Stage 4
should not have to rediscover this. `model-plan.md` §8 writes
`models_game/<run_id>/<feature_set>/<model>/` and says "add a target path
component before introducing additional targets".
`next-implementations.md` §55 supersedes it: the round-2 review explicitly
accepted `models_game/<target>/<run_id>/<feature_set>/<model>/`, and §328
restates it. The roadmap's later decision wins, and including `<target>`
(`home_win` in v1) now satisfies model-plan's rule too — it asks for the
component *before* a second target exists, not after. `save_model` itself
takes whatever path it is given; this is the convention its caller builds.

No atomic temp-and-rename here, unlike the snapshot module. A snapshot is the
irreplaceable record of a database at a moment; a model is re-derivable from
its training fold in seconds. An interrupted write leaves a truncated pickle,
which `joblib.load` raises on rather than returning silently wrong, so the
recovery is a refit either way.

## 3. The pipeline

```
ColumnTransformer(
  ("team_imputed",   Pipeline(SimpleImputer(strategy="median",
                                            keep_empty_features=True),
                              StandardScaler()),          TEAM_FEATURE_COLUMNS),
  ("team_missing",   MissingIndicator(features="all"),    TEAM_FEATURE_COLUMNS),
  ("market",         StandardScaler(),                    ["market_home_probability"]),
)
→ LogisticRegression(C=C, solver=SOLVER,
                     max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_SEED)
```

The `market` block exists only for `team_market`.

L2 is not passed explicitly: it is `LogisticRegression`'s default, and
`penalty="l2"` raises a `FutureWarning` on scikit-learn 1.8+ (removed in
1.10). Verified identical coefficients and intercept with and without the
argument on 1.9.1.

Three choices worth their reasoning, each verified against scikit-learn 1.9.1:

**`MissingIndicator(features="all")`, not `SimpleImputer(add_indicator=True)`.**
`add_indicator` only emits indicators for columns that happened to be missing
*in training*. A column complete in training but missing at prediction time
would then be filled with the training median and carry no flag — invisible
imputation, and a transformed schema whose width depends on which fold you
fitted. `features="all"` gives one fixed indicator per team feature, so the
schema is stable across folds and test-time-only missingness is visible.
Verified: a column complete in training and missing at predict time yields
`[False, True, False]`.

**`keep_empty_features=True`.** A column that is all-missing in training would
otherwise be dropped, changing the schema width per fold. With this flag the
column survives and is filled with `0.0`, which is the "documented zero
fallback plus their indicator, preserving schema order" the deliverable asks
for. Verified: `statistics_` records `0.0` for the all-NaN column and the
output keeps all three columns. The indicator still marks every row, so the
model can tell "zero because unknown" from a real zero.

**The market probability is scaled but never imputed.** `select_features` has
already guaranteed it is present and strictly inside `(0, 1)` for every
surviving row, so there is nothing to impute; the deliverable's "never impute a
market probability" is satisfied structurally rather than by a flag. It is
standardized along with everything else because L2 penalizes coefficients on
whatever scale their feature happens to use, so mixed scales silently mean
mixed regularization strength.

### 3.1 Transformed schema

The `ColumnTransformer` emits blocks in the order declared above:

```
[imputed + scaled TEAM_FEATURE_COLUMNS]   # 38
+ [missingness indicator per column]      # 38
+ [scaled market_home_probability]        # team_market only
```

76 columns for `team_only`, 77 for `team_market`. This width must not vary with
the fold — that is the point of the two flags above, and the tests assert it
against the real `transform()` output rather than against a list of names
maintained alongside it.

No `transformed_schema` list is stored in the bundle: nothing reads one. Stage
5 can build the names from `feature_schema` when it labels coefficients.

## 4. The leakage invariant

Everything learned — medians, means, scales, coefficients — is fitted inside
`fit_model` on training rows only. `predict_model` calls `transform`, never
`fit_transform`. There is no code path in this module that fits on data handed
to a predict call.

The acceptance criterion "changing validation/test feature values cannot change
training-fold medians, scales, coefficients, or the fixed transformed feature
schema" is tested directly in §7.3 by fitting, perturbing a held-out frame
wildly, and asserting every fitted array is bit-identical.

## 5. Insufficient data fails loudly

Checked in `fit_model` before touching sklearn, so the error names the real
problem:

| Condition | Error |
|---|---|
| `team_market` fold with no paired quote at all | `ValueError("cannot fit <variant>: no rows with a paired market quote")` |
| no rows survive label filtering | `ValueError("cannot fit <variant>: no training rows with a usable label")` |
| repeated index labels on `matchups` | `ValueError` naming the repeats and `reset_index(drop=True)` |
| duplicate or null `game_id` (predict) | `ValueError` naming the repeats |
| fewer than 2 rows | `ValueError` naming the count |
| labels are one-class | `ValueError("cannot fit <variant>: training labels are one-class (all 0 or all 1)")` |

sklearn would raise for the one-class case anyway ("This solver needs samples
of at least 2 classes"), but the pre-check names the fold rather than the
solver. No fallback model is ever produced — an unfittable fold is an error the
caller must see, per the deliverable.

Note the deliberate contrast with Stage 2: `fit_home_frequency` fits a
perfectly good `0.0`/`1.0` on a one-class fold, because a constant baseline
*can* be fitted from one class. A logistic regression cannot.

The last three rows of that table are aimed at Stage 4 specifically. Label
alignment is `matchups.loc[X.index]`, so a repeated index label fans rows out
rather than selecting them — and `pd.concat([...])` without
`ignore_index=True`, the natural way to assemble a fold from several seasons,
produces exactly that. Unguarded it surfaced as `IndexError: indices are
out-of-bounds` from inside sklearn, naming nothing useful. The `game_id` checks
mirror Stage 2's `predict_home_frequency`, since the prediction frame is keyed
on that column. And a quote-less `team_market` fold is reported separately from
a label-less one because its labels are perfectly good; blaming them would send
the caller hunting the wrong problem.

## 6. Schema rejection on load and predict

`predict_model` rebuilds `X` through `select_features(matchups, bundle["variant"])`
and compares `list(X.columns)` to `bundle["feature_schema"]`. A mismatch raises
`ValueError` naming the first difference.

This is not redundant with `select_features`'s own `KeyError`: that catches a
matchup table missing columns *today*, while this catches a **saved model whose
feature contract no longer matches the code** — a bundle fitted before a change
to `TEAM_FEATURE_COLUMNS`, loaded after. Silent misalignment there would map
coefficients onto the wrong features and produce confident nonsense, the same
failure class as the branch-divergence risk that motivated the Stage 1
reconciliation.

## 7. Test plan (`tests/test_train.py`)

Synthetic and offline, in the project's `demo_*` + plain-`assert` style. A
shared fixture builds a matchup frame with the full ordered feature set,
controllable missingness, and a learnable signal.

**§7.1 `demo_fit_predict_round_trip_and_schema()`** — fit both variants;
assert `transform()` emits the §3.1 width (76 / 77); `p_home` within
`(0, 1)`; one output row per input game; config records C, solver, seed, tol,
max_iter; `converged` is True on easy data.

**§7.2 `demo_no_target_id_or_date_column_can_enter_X()`** — reuse the allowlist
pattern from `tests/test_matchup_features.py`: add `home_score`, `away_score`,
`point_diff`, `home_win`, `game_id`, `gameday` to the input and assert none
reaches `feature_schema`, and that the schema equals the variant's constant.

**§7.3 `demo_validation_values_cannot_move_fitted_parameters()`** — the core
leak test. Fit, snapshot the imputer `statistics_`, scaler `mean_`/`scale_`,
and `coef_`/`intercept_`; then mutate held-out rows to absurd values, refit
*nothing*, call `predict_model`, and assert every fitted array is unchanged
via `np.array_equal`. Then fit a second model on train-plus-corrupted-holdout
and assert the coefficients **do** move, so the first assertion is not vacuous.

**§7.4 `demo_missingness_cases()`** — cold-start rows (all team features NaN);
a column all-missing in training (asserts the `0.0` fallback in `statistics_`,
the surviving schema width, and the indicator firing); missingness appearing
only at prediction time (asserts the indicator flips without refitting and the
schema width is unchanged).

**§7.5 `demo_team_market_abstains_without_a_paired_quote()`** — rows with a
null/0/1 market probability are excluded from fitting, appear in the prediction
output with `p_home` NaN and `abstention_reason == "no_paired_market_quote"`,
and the market column is never imputed. Also assert `team_only` scores those
same rows normally, since it does not need a quote.

**§7.6 `demo_insufficient_data_fails_loudly()`** — empty, single-row, and
one-class training sets each raise with the §5 message; no bundle is returned.

**§7.7 `demo_save_load_prediction_equivalence()`** — save, load, predict;
assert predictions are bit-identical (`pd.testing.assert_frame_equal`), and
that a bundle whose `feature_schema` is tampered with is rejected by
`predict_model` with a schema error.

## 8. Out of scope

No `C` tuning, no folds, no chronological logic (Stage 4). No class weighting,
feature selection, or post-hoc calibration — `model-plan.md` defers calibration
until chronological calibration evaluation is designed. No scoring or metrics
here beyond what the tests need. No real-data run: Stage 3's tests are
synthetic, and the real backtest arrives with Stage 4.

## 9. Verification

```bash
.venv/bin/python tests/test_train.py
.venv/bin/python tests/test_baselines.py
.venv/bin/python tests/test_metrics.py
.venv/bin/python tests/test_matchup_features.py
.venv/bin/python -m unittest discover -s tests -p test_rolling_features.py
.venv/bin/python -m src.evaluation.diagnose --db-path data/nfl_games.db
```

Stage 0 must still report **5170 scored games, 4780 paired, pooled market log
loss 0.608600**. Stage 3 adds a model; it touches no Stage 0 path, so any
movement there means something unintended was changed.

`tests/test_train.py` is a new test file, so it gets a documented run command in
the project notes (the repo's rule, `next-implementations.md` Stage 5).
