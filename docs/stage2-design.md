# Stage 2 design: fitted baselines

Implementation spec for `docs/next-implementations.md` Stage 2. Every decision
is settled here so implementation is transcription, not design. Where a rule
comes from an earlier agreed document it is cited; where this document decides
something new it says so explicitly.

**Files touched:** `src/models/baselines.py` (extend), `tests/test_baselines.py`
(extend). Nothing else.

## 1. What Stage 2 adds, and why it is not Stage 0's number

Stage 0 already reports the home-win *rate* over a fixed window, as an accuracy
figure for the fitting-free always-home baseline. Stage 2 adds a different
thing that happens to be computed the same way: a **fitted constant
probability**.

The distinction is the whole point. `docs/model-plan.md` §6:

| Baseline | Definition | Fit rule |
|---|---|---|
| Always-home | Predict `home_win = 1` | No fitting; accuracy only, not a probability baseline (B7) |
| Home-frequency | Training-fold non-tie home-win rate as a constant probability | Fit per fold |

Always-home is scored as accuracy only, never as a probability, because a
constant `P(home_win)=1` has unbounded log loss the moment it is wrong once
(`model-plan.md` B7). Home-frequency *is* a probability, so it is scored with
log loss like any model — and therefore it must be fitted on training rows only
and held fixed, or it leaks.

Stage 4 supplies real folds. Stage 2 accepts an explicit training sample and
stops there (`next-implementations.md` Stage 2: "Accept an explicit training
sample here; Stage 4 will supply folds").

## 2. Defect in existing code — FIXED, ahead of the rest of Stage 2

**Status: done.** Landed before the rest of this stage, as commit `17418bc` on
`the main branch` and `1dc5183` on `the review branch`. `_validated_labels` (§4.1) exists
and is in use; the regression test in §7.1 is written and passing. Recorded
here because it motivates the shared label gate the new functions reuse.

`always_home_accuracy()` silently miscounted a float-dtype null as a home win.
`.astype(bool)` maps `np.nan → True`.

Verified against the real cached database. `home_win` comes back from SQLite as
**float64**, not `Int64`, so its nulls are `np.nan`, not `pd.NA`:

```
home_win dtype from SQLite: float64 | nulls: 9   (first 5000 matchup rows)
always_home_accuracy(unfiltered column) -> {'accuracy': 0.5726, 'n': 5000}
true rate among non-null                -> 0.5718292927269084, n = 4991
```

Nine unplayed or tied games were counted as home wins. The existing test only
covers `pd.Series([...], dtype="boolean")` carrying `pd.NA`, which *does* raise
`ValueError: cannot convert float NaN to bool`. The float64 path — the one the
real database actually produces, and the one a `scored_rows.csv` round-trip
produces — is uncovered and silently wrong.

Current Stage 0 results are unaffected, because `diagnose.py` filters with
`.notna()` before calling. The bug is latent, not active. It is fixed here
because Stage 2 adds a second function to the same file under the same "caller
supplies eligible rows" contract, and that contract must be enforced rather
than assumed.

**Fix:** both functions validate their label input through one shared helper
(§4.1) that rejects nulls of any dtype explicitly. `always_home_accuracy`'s
documented behavior is otherwise unchanged, including returning
`{"accuracy": nan, "n": 0}` for a genuinely empty input.

## 3. Public API

Two new functions in `src/models/baselines.py`. No classes, no state, no
persistence — a fitted constant is a float, and Stage 5 owns artifact writing.

```python
def fit_home_frequency(home_win: pd.Series) -> dict:
    """Fit the constant home-win probability on TRAINING-FOLD rows only."""

def predict_home_frequency(fitted: dict, game_ids) -> pd.Series:
    """Constant fitted probability, one row per game ID."""
```

### 3.1 `fit_home_frequency(home_win)`

- **Input**: a `pd.Series` of boolean-like labels, already restricted by the
  caller to eligible, non-tied, non-null rows. Same contract as
  `always_home_accuracy`: this function does not filter, it validates.
  Accepted dtypes: `bool`, `boolean`, `int64`, `Int64`, `float64` carrying only
  `0.0`/`1.0`. The index is ignored.
- **Returns**: `{"p_home": float, "n": int}` — a plain dict of plain scalars.
- `p_home` is `float(y.mean())` over the validated labels; `n` is `len(y)`.

Returning plain scalars is what structurally satisfies the acceptance criterion
"changing validation/test labels cannot change an already-fitted
home-frequency baseline": the returned dict holds no reference to the input, so
there is nothing for a later mutation to reach. This is a property of the
return type, not a promise in a docstring.

### 3.2 `predict_home_frequency(fitted, game_ids)`

- **Input**: `fitted` as returned by `fit_home_frequency`; `game_ids` as any
  `pd.Index`, `pd.Series`, or sequence of game IDs to predict for.
- **Returns**: `pd.Series` of length `len(game_ids)`, every value `p_home`,
  indexed by the game IDs, `name="p_home"`.
- Validates that the game IDs are non-null and unique before returning
  anything: a duplicated ID would silently fan out during a later join with
  labels and double-count games in the score.

This covers only the **duplicate** half of the acceptance criterion "reject
duplicate/misaligned game IDs before computing any comparison". An earlier
revision of this document claimed the whole criterion was satisfied here; that
was wrong. `predict_home_frequency` never sees the labels, so it cannot detect
a *misaligned* population — predictions for one set of games scored against
labels for another produced a plausible number with nothing objecting
(measured: 1.3862943611198906 for three predicted games sharing no ID with the
three labelled ones). The misalignment half is enforced by
`align_by_game_id` — see §3.3.

Taking the `fitted` dict rather than a bare float is deliberate: it keeps fit
and predict visibly paired, and makes it awkward to pass a label column where a
fitted value belongs.

### 3.3 `align_by_game_id(labels, predictions)` — in `metrics.py`, not here

```python
def align_by_game_id(labels: pd.Series, predictions: pd.Series) -> tuple:
```

Both Series are indexed by game ID. Requires unique, non-null IDs on each side
and an **identical ID set**; reindexes predictions onto the labels' order and
returns plain `(y, p)` float arrays for the metrics. Order is therefore
irrelevant — two correctly-populated inputs sorted differently still align,
while two different populations raise instead of scoring.

This lives in `src/evaluation/metrics.py`, a deliberate deviation from Stage 2's
stated file list. It is a scoring precondition, not a baseline: it sits beside
`_validate` (which guards a comparison's *shape* while this guards its
*identity*), and Stage 4's paired model-versus-market comparison on identical
game IDs needs the same function. Putting it in `baselines.py` would mean moving
it two stages later.

**Scoring a subset never refits.** `next-implementations.md` Stage 4 requires
that "the home-frequency probability stays fitted on its full training set —
scoring a subset never refits it." With this API that is automatic: call
`fit_home_frequency` once per fold, then call `predict_home_frequency` as many
times as there are scoring populations (full eligible coverage, and the
paired-quote subset for market comparison). There is no code path that refits.

## 4. Validation rules and exact errors

### 4.1 Shared label validation

Both functions' label handling goes through one private helper. Implement it
once; do not duplicate the checks.

```python
def _validated_labels(home_win: pd.Series) -> np.ndarray:
```

In order:

| Condition | Error | Message |
|---|---|---|
| any null (`pd.NA`, `np.nan`, `None`, `NaT`) | `ValueError` | `"home_win must not contain nulls -- filter to eligible, non-tied games first"` |
| any value not in `{0, 1}` after numeric coercion | `ValueError` | `"home_win must contain only 0/1 (or boolean) values"` |
| non-numeric, non-boolean dtype (e.g. strings) | `ValueError` | same as above |

Use `pd.isna(...).any()` for the null check — it covers every null flavor
across dtypes, unlike `.astype(bool)`. Reuse the 0/1 check shape already in
`src/evaluation/metrics.py::_validate` (`np.isin(y, (0.0, 1.0)).all()`) so the
two modules agree on what a label is.

Returns a `float64` numpy array of 0.0/1.0.

### 4.2 `fit_home_frequency` specifics

| Condition | Behavior |
|---|---|
| empty input | `ValueError("cannot fit home-frequency on an empty training sample")` |
| all labels 1 | `p_home = 1.0` exactly. No smoothing. |
| all labels 0 | `p_home = 0.0` exactly. No smoothing. |
| otherwise | `p_home = mean`, `n = len` |

The empty case **raises**, unlike `always_home_accuracy`, which returns
`{"accuracy": nan, "n": 0}`. This asymmetry is intentional and required by the
spec ("An empty training sample fails explicitly"). The reason: always-home is
a descriptive statistic, and "no games, no rate" is a fine answer; a fitted
parameter with no training data is not a model, and quietly returning NaN would
propagate into predictions and scores as a silent hole.

### 4.3 `predict_home_frequency` specifics

| Condition | Error |
|---|---|
| duplicate game IDs | `ValueError("game_ids must be unique -- duplicates would double-count games when joined to labels")` |
| null game ID | `ValueError("game_ids must be non-null")` |
| `fitted` missing `p_home` | `KeyError` from the plain lookup; no custom handling |
| empty `game_ids` | returns an empty `pd.Series` with `dtype=float64`. Not an error — a fold can legitimately have an empty scoring subset (e.g. no paired quotes), and the metrics already return NaN for an empty sample. |

## 5. One-class training folds, and why nothing is smoothed

A training fold in which every game was a home win fits `p_home = 1.0`. Scored
against any fold containing a single away win, `log_loss` clips to `eps=1e-15`
and yields a large but finite penalty rather than `inf`:

```
-log(1e-15) = 34.538776394910684
log_loss([1,1,0], [1.0,1.0,1.0]) = 11.51319199744696
```

That is the documented, intended outcome (`next-implementations.md` Stage 2:
"probabilities 0/1 from a one-class training sample use the documented metric
clipping, without evaluation-based smoothing or replacement").

Do **not** add Laplace smoothing, a floor/ceiling, or any adjustment. Choosing
a smoothing constant because the unsmoothed value scored badly on evaluation
data is fitting to the evaluation set — the precise failure this whole staged
protocol exists to prevent. A one-class fold is a real, if degenerate,
training set, and its honest fitted value is 0 or 1.

## 6. Interaction with metrics

Scoring is `src/evaluation/metrics.py`'s job; do not recompute anything in
`baselines.py`. A caller scores the baseline exactly as it scores a model:

```python
fitted = fit_home_frequency(train["home_win"])
p = predict_home_frequency(fitted, test["game_id"])
log_loss(test["home_win"].to_numpy(float), p.to_numpy())
```

`metrics._validate` already rejects mismatched lengths, non-finite
probabilities, and non-0/1 labels, so no additional guarding belongs at the
call site.

Per `model-plan.md` §6, home-frequency is reported on **all eligible games**,
and separately on the paired-quote subset when compared against the market —
"Do not compare one model's full sample against another's restricted sample."

## 7. Exact test plan

Extend `tests/test_baselines.py`, following its existing style: module-level
`demo_*` functions with plain `assert`, called from `__main__`, each printing
one `OK -- ...` line. No framework.

### 7.1 Regression for the §2 defect — DONE

`demo_null_labels_are_rejected_in_every_dtype()` — written and passing; landed
with the §2 fix.

- `pd.Series([1.0, 0.0, np.nan])` → both functions raise `ValueError`
- `pd.Series([True, False, pd.NA], dtype="boolean")` → raises `ValueError`
- `pd.Series([1, 0, None], dtype="Int64")` → raises `ValueError`
- `pd.Series(["2020-01-01", None])` → raises `ValueError`
- Assert the float64 case specifically does **not** return `n == 3`; that was
  the bug, and a test that only checks "raises" on the `boolean` dtype would
  have missed it.

### 7.2 Fitting

`demo_fit_home_frequency_matches_hand_computation()`

- `[True, True, False, True, False, False, True]` → `{"p_home": 4/7, "n": 7}`;
  `4/7 == 0.5714285714285714`
- `[1, 0, 1, 1]` (int-coded) → `{"p_home": 0.75, "n": 4}`
- `[1.0, 0.0]` (float-coded) → `{"p_home": 0.5, "n": 2}`
- `set(result) == {"p_home", "n"}` — the fitted baseline exposes a probability,
  where always-home must never (B7). Assert both contracts in the same test so
  the distinction cannot silently erode.
- Empty → `ValueError`, and the message mentions "empty"

`demo_one_class_training_fold_is_not_smoothed()`

- all-True → `p_home == 1.0` exactly (`assert fitted["p_home"] == 1.0`, not
  `isclose`)
- all-False → `p_home == 0.0` exactly
- `log_loss([1,1,0], predict(...))` is finite and `> 11.0`, confirming clipping
  rather than `inf` and rather than a smoothed value

### 7.3 The leak-safety criterion

`demo_fitted_value_cannot_be_changed_by_later_labels()`

- Fit on a training Series, capture `p_home`
- Mutate the original Series in place (`train.iloc[0] = False`) and assert the
  fitted dict is unchanged
- Rebind/extend the Series and assert again
- Fit on train, predict for *test* game IDs, and assert every predicted value
  equals the training-fold rate — not the test-fold rate. Build the fixture so
  the two rates differ (e.g. train 3/4 home wins, test 0/4), so a leak would
  change the number rather than coincidentally matching.

This is the acceptance criterion "changing validation/test labels cannot change
an already-fitted home-frequency baseline", made observable.

### 7.4 Prediction and ID discipline

`demo_predictions_are_constant_and_reject_bad_ids()`

- `predict_home_frequency(fitted, ["g1","g2","g3"])` → length 3, all equal
  `p_home`, index equals the game IDs, `name == "p_home"`
- duplicate IDs `["g1","g1"]` → `ValueError` mentioning "unique"
- null ID `["g1", None]` → `ValueError`
- empty ID list → empty Series, `dtype == float64`, no raise
- scoring two different subsets from one fitted value returns the same constant
  — the "scoring a subset never refits" guarantee

## 8. Out of scope

Do not, in this stage:

- implement folds, splits, or any chronological logic (Stage 4)
- touch `src/features/` or the feature contract
- persist anything to disk or to a database (Stage 5)
- add the fitted market-only control (deferred; `model-plan.md` §6)
- score against real data in the tests — Stage 2's tests are synthetic and
  offline; the real-data path arrives with Stage 4's backtest
- change `always_home_accuracy`'s public return shape or its empty-input
  behavior; only its null handling tightens

## 9. Verification

```bash
.venv/bin/python tests/test_baselines.py
```

Then confirm nothing upstream shifted, since `baselines.py` is on Stage 0's
path:

```bash
.venv/bin/python tests/test_metrics.py
.venv/bin/python tests/test_diagnose.py --db-path data/nfl_games.db
.venv/bin/python -m unittest discover -s tests -p test_stage0_regressions.py
.venv/bin/python -m src.evaluation.diagnose --db-path data/nfl_games.db
```

The Stage 0 diagnostic must still report **5170 scored games, 4780 paired,
pooled market log loss 0.608600**. Stage 2 adds a baseline; it does not move a
Stage 0 number. If that line changes, the null-handling fix in §2 altered the
scored population and the change must be explained before it is committed.

Add the new run command to the project notes only if a new test *file* appears; this
stage extends an existing one, which is already listed.
