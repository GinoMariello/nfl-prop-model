# Next implementations: reproducible NFL win-probability baseline

Status: implementation plan, 2026-09-15. No model implementation or training
is authorized by this document alone. This roadmap follows `model-plan.md`
(including its 2026-09-15 historical completion-proxy amendment) and the
feature contract implemented in `src/features/rolling_features.py`
(`TEAM_FEATURE_COLUMNS`/`MARKET_FEATURE_COLUMNS`, `select_features()`,
`exclusion_report()`).

## Review decisions (round 2, responding to the review branch `a99e04f`)

the reviewer's review of the round-1 version of this document (`ee22b08`) agreed
with the core sequencing call — a small Stage 0 before the full experiment
harness — and made several corrections I'm accepting, plus one I'm only
partially accepting. In order of importance:

- **Accepted, and this is the important one: I have already inspected 2025
  and partial-2026 data informally this session, without excluding them,
  and cited those numbers as justification in round 1 of this document.**
  Earlier this session I ran baseline accuracy/log-loss numbers against
  the full 1999-2026 range and reported them, then quoted
  those same numbers in round 1's "Where this disagrees" section as
  evidence for Stage 0. model-plan.md §5 is explicit: "Reserve 2025 for
  final evaluation only after documenting that its model results have not
  guided decisions. If already used, mark it development data and reserve
  an untouched future block... Freeze the protocol before inspecting final
  results; repeated inspection makes a holdout part of development." That
  rule doesn't carve out an exception for "just a descriptive check" — I
  looked at 2025 and let it shape what I wrote. Per the rule as stated,
  2025 should be treated as **already-used development data, not a clean
  holdout**, and Stage 1 needs to identify an untouched future block
  instead. There isn't an obvious clean one: 2026 is partial/in-progress
  (a season boundary itself doesn't resolve this the way it would for a
  completed season), so this is a genuinely open question Stage 1 must
  resolve and document, not something this roadmap can wave through.
  Stage 0 (below) is revised to default to 2005-2024 and exclude
  2025/2026 by construction, so this doesn't recur.
- **Accepted: Stage 0 needs real discipline, not just "no fitting
  required."** the reviewer is right that absence of fitting doesn't remove
  selection bias, snapshot drift, or holdout exposure — a diagnostic still
  needs a defined population, matched comparison IDs, saved scored rows
  with a checksum (so the printed numbers can be reproduced later even
  though the full Stage 1 snapshot doesn't exist yet), and explicit
  source/version tracking. Folded into Stage 0 below.
- **Accepted: "~340MB" was the wrong way to size the snapshot strategy.**
  Round 1 anchored Stage 1's storage decision to a database size measured
  once, this session — that number is already stale by the time anyone
  implements this (the database grows every season). Stage 1 now says to
  measure at implementation time. Also accepted: a checksum only verifies
  an *already-immutable* artifact hasn't changed — it isn't itself a
  snapshot mechanism, and can't restore a value that got overwritten. My
  round-1 phrasing floated "checksummed reference" as if it were a
  standalone alternative to "full copy"; it isn't.
- **Accepted: artifact path ordering.** Switched to the reviewer's
  `models_game/<target>/<run_id>/<feature_set>/<model>/` from round 1's
  `models_game/<target>/<feature_set>/<model>/<run_id>/`. A single
  protocol run (Stage 5) produces multiple feature-set/model combinations
  together; grouping by `run_id` right after `target` keeps one run's
  outputs co-located, which matters more than grouping by model family.
- **Accepted: strengthen Stage 4's chronological-ordering assertion** from
  "disjoint partitions" to "disjoint AND strictly chronologically ordered"
  — disjointness alone doesn't rule out an out-of-order partition.
- **Partially accepted: test conventions.** Round 1 said every new test in
  this roadmap should use this repo's plain-assert/`demo()` convention,
  not `unittest`. the reviewer's pushback: the review branch already has both discoverable
  `unittest` tests and a standalone cached-data check, and converting
  existing tests just to impose one style isn't warranted — plus, whatever
  style is used, `unittest discover` doesn't pick up a free-function
  `demo()` body, so any non-`unittest` test needs its own documented run
  command regardless. I'm keeping the underlying instruction for
  **the main branch's own new tests** (plain assert + `demo()`/`__main__` is
  what the project notes document for *this* repo, and every test
  file added by this roadmap in this worktree should match the ones
  already here: `tests/test_matchup_features.py`, `tests/test_rolling_features.py`).
  But I'm dropping the implication that the reviewer's worktree needs to convert
  anything — that's a call for whoever owns that worktree, not something
  this document should mandate. Documenting an explicit run command per
  test file (not just "pytest picks it up") applies regardless of style
  and is now called out explicitly below.

Everything else (stages 2-6, deferred work) is unchanged from round 1 and
close to the reviewer's draft — sound, spec-faithful, not rewritten.

## Objective

Produce a reproducible, chronologically evaluated `home_win` probability
model and an honest comparison against simple and market baselines.
Success means trustworthy evaluation and replay; beating the market is
not a release gate.

The existing feature builder is the starting point and is not touched by
any of this: ordered allowlists, eight-game REG history across seasons,
strict earlier-date rule, nullable labels, market abstention,
explicit-status precedence, score-based completion proxy. No new reshape
or feature pipeline is needed — `src/features/rolling_features.py` is
done for v1. Reopen it only for a demonstrated correctness issue with its
own regression test, not to expand the feature set as part of this roadmap.

## Implementation sequence

Each numbered stage is a separate reviewable commit. Run its offline
tests before proceeding, with an explicit documented command for every
new test file. Do not introduce new features or tune models while
implementing the evaluation machinery.

### 0. Small, traceable baseline diagnostic on development data only

**Files:** `src/models/baselines.py`, `src/evaluation/metrics.py`,
`src/evaluation/diagnose.py`, needed package `__init__.py` files,
`tests/test_baselines.py`, `tests/test_metrics.py`, `tests/test_diagnose.py`
(plain `assert` + `demo()`/`__main__`, matching this repo's existing test
files — see "Review decisions" above).

This stage is exploratory and doesn't need the full Stage 1 harness, but
it still has to define and preserve exactly what it scored.

Deliverables:

- **Default to development seasons 2005-2024. Exclude 2025 and 2026
  unconditionally** — not "unless a model has been fit," any inspection
  counts (see "Review decisions"). Any alternative window must be
  explicit in the diagnostic's own printed output, never a silent default.
- Record the diagnostic's run date, source database identity and
  revision/dirty diff, Python/package versions, selected seasons,
  target/eligibility rules, and metric settings.
- Read a consistent view of inputs: either recompute derived rows in
  memory/a scratch database from the current builder, or verify that a
  cached `matchup_features` table actually came from the current
  ingestion+feature code before trusting it.
- Save the exact scored rows (game ID, season/week, label, market
  probability, the raw quotes behind it) plus eligibility/exclusion
  counts, with a checksum, in a unique diagnostic output directory — this
  reproduces the printed calculation later; it is not a substitute for
  Stage 1's retrievable raw snapshot.
- Always-home: report accuracy only, on non-null eligible labels in the
  selected window. Never score it as a probability (model-plan.md B7:
  unbounded log loss the moment it's wrong once).
- Devigged market: reuse `normalized_home_probability()` /
  `select_features(matchup, "team_market")` unchanged — do not
  re-implement the devig formula here. Intersect the paired-quote mask
  with non-null labels and the selected seasons before scoring.
- Metrics (`src/evaluation/metrics.py`, reused by every later stage): log
  loss (primary), Brier score, accuracy at a documented 0.5 threshold,
  ROC AUC, ten equal-width calibration bins with counts, mean predicted
  probability, and observed win frequency per bin. Document numerical
  clipping (log loss only) and behavior on an empty bin, an empty sample,
  or one-class labels. Reject non-finite or out-of-[0,1] probabilities
  before clipping, don't silently clip them into range.
- Report always-home accuracy on both full eligible coverage AND the
  exact paired-quote game IDs used for the market baseline — a bare
  accuracy number without its denominator/ID set isn't a valid comparison.
- A CLI prints per-season and pooled metrics with denominators, labeled
  "exploratory diagnostic on 2005-2024 development data; not final
  evaluation." No superiority claim from this stage alone.

Tests / acceptance:

- Hand-computed log loss/Brier/accuracy/calibration-bin membership,
  including probabilities at exactly 0/1, an empty bin, an empty sample,
  and one-class labels.
- Always-home exposes accuracy only; the shared log-loss function still
  scores a legitimate 0/1-containing probability vector under documented
  clipping (that's a different code path, not the always-home baseline).
- A valid market quote on an unplayed or tied game never becomes a scored
  observation.
- Duplicate/misaligned game IDs are rejected; paired comparisons use
  identical IDs; full and matched coverage are reported separately.
- The diagnostic excludes 2025/2026 even if those rows already have
  final-looking scores. Mutating an excluded row cannot change any
  printed output.
- Reloading the saved scored rows reproduces every printed metric
  exactly; a later mutation of the live cache doesn't change that saved
  calculation.

### 1. Freeze the experiment contract and input data

**Files:** `src/data/snapshot.py`, `tests/test_snapshot.py`,
`docs/experiment-protocol.md`, `requirements.txt`, a resolved dependency
file.

Deliverables:

- A compatible `.venv` dependency set for ingestion, pandas/numpy, and
  scikit-learn. Verify dependency resolution; do not blindly widen the
  current unverified version ranges. Record Python and platform versions.
- Measure the actual database size and available storage **at
  implementation time** (not the number quoted in this document, which
  will already be stale). Capture one consistent SQLite backup per
  selected dataset revision and reuse it across every run — a checksum
  alone is not a snapshot; it only verifies an already-immutable artifact
  hasn't changed. Defer deduplication or an archive service until
  measured storage actually justifies it.
- Preserve raw `games`/`plays` untouched; treat any already-derived
  tables as non-authoritative. Rebuild into memory or scratch storage.
  Training/evaluation must never call a table-replacement entry point
  (`team_game_stats.run()`/`rolling_features.run()`) against the selected
  snapshot.
- A manifest: snapshot creation time, checksum, table counts, season
  coverage, source URLs/releases, field/model versions where known.
  Preserve unknown historical retrieval times as unknown — creation time
  is not historical data availability.
- Deterministic content hashing for feature/label rows: ordered game IDs
  and columns, canonical numeric encoding, explicit missing-value marker.
- Freeze the `home_win` definition, completion assumption, cutoff
  convention, model variants, C grid, scoring metrics, uncertainty rules,
  and seeds in the protocol doc.
- **Resolve the holdout question this document could not close**: 2025 is
  development data as of this session (see "Review decisions"), not an
  untouched holdout. Document that explicitly, and either identify a
  genuinely untouched future block or state plainly that v1 ships without
  one until a full future season is available. Keep partial 2026 out of
  the initial full-season backtest regardless.

Tests / acceptance:

- An existing snapshot cannot be overwritten; a later source correction
  does not change a captured snapshot. Checksum verification detects
  tampering.
- Content hashes ignore input row order but change when a feature or
  label value changes.
- Manifest distinguishes missing provenance from known values. A fresh
  environment can import ingestion/feature/modeling dependencies and pass
  the existing feature tests plus this stage's own.

### 2. Fitted baselines

**Files:** `src/models/baselines.py` (extend), `tests/test_baselines.py`
(extend).

- Home-frequency: fit a constant probability from eligible non-tied
  *training-fold* outcomes only; hold that fitted value fixed through
  validation/test predictions in that fold. This needs Stage 4's fold
  engine to supply real folds — accept an explicit training sample here
  and wire it to Stage 4 later. Stage 0's home-win rate was accuracy on a
  fixed window, not this fitted baseline.

Tests / acceptance:

- Changing validation/test labels cannot change an already-fitted
  home-frequency baseline.
- Reject duplicate/misaligned game IDs before computing any comparison.
- An empty training sample fails explicitly; a one-class training sample
  produces a 0/1 probability handled by the documented metric clipping,
  never smoothed or replaced based on evaluation outcomes.

### 3. Leakage-safe preprocessing and logistic regression

**Files:** `src/models/train.py`, `tests/test_train.py`.

Deliverables:

- Consume `select_features()` with an explicit variant; persist the exact
  ordered schema. Align labels by game ID; exclude missing labels before
  fitting.
- sklearn pipeline: training-only medians for team features, fixed
  missingness indicators per team feature, standardization, L2 logistic
  regression. All-missing training columns get a documented zero
  fallback plus their indicator, preserving schema order.
- `team_market`: filter to paired-quote rows before fitting and
  prediction; never impute a market probability. Preserve the abstention
  reason alongside every output row.
- Expose `C` as configuration; tuning is Stage 4's job, not this one.
  Record solver, seed, tolerance, iteration limit, convergence status. No
  class weighting, feature selection, or post-hoc calibration in v1.
- Save/load the fitted preprocessing and model together; reject a
  mismatched inference schema. An insufficient-data fold fails loudly, it
  doesn't silently produce a fallback model.

Tests / acceptance:

- Changing validation/test feature values cannot change training-fold
  medians, scales, coefficients, or the fixed transformed feature schema.
- Cold starts, all-missing training columns, missingness that only
  appears at prediction time, missing market quotes, save/load prediction
  equivalence.
- No target, raw score, ID, date, or unrelated column ever enters `X`
  (reuse the existing `select_features()` allowlist test pattern from
  `tests/test_matchup_features.py`).

### 4. Chronological backtest

**Files:** `src/evaluation/backtest.py`, `tests/test_backtest.py`.

Deliverables:

- Outer development seasons: 2005-2024 (subject to actual snapshot
  coverage — see Stage 1's holdout resolution). For outer season T: fit
  candidates on 1999..T-2, select by T-1 validation log loss, refit all
  learned preprocessing+model steps on 1999..T-1, score T only with that
  final refit.
- **`team_market` starts at T=2008, not 2005 — measured, not assumed.**
  The snapshot carries no moneylines at all before 2006 (1999-2005 is
  0/1768 paired quotes; 2006 is 209/256, 2007 is 255/256, 2010 onward is
  complete). Under the fold rule above, outer seasons 2005/2006/2007 each
  train on 1999..T-2, which holds zero quoted games, so `fit_model` raises
  `cannot fit team_market: no rows with a paired market quote` — correctly,
  by design. T=2008 is the first outer season whose training fold reaches
  2006. So `team_only` and home-frequency get 20 outer seasons and
  `team_market` gets 17, and §6's "do not compare one model's full sample
  against another's restricted sample" puts every model-vs-model comparison
  on 2008-2024. The fold loop must skip those three seasons for that
  variant deliberately and report the skip, not crash and not silently
  widen. `tests/test_cached_features.py` pins all of this against the real
  database, so a re-ingest that backfills early odds fails the test instead
  of quietly moving the window.
- **Neutral-site games carry `neutral_site`; decide what the backtest does
  with them.** 58 REG games in 2005-2024 are played at neutral sites, where
  the nominal home team has no home field — the one place §4's "is_home is
  redundant with the home/away row structure" does not hold. The flag is
  built (`rolling_features._neutral_site`) and is matchup metadata, *not* an
  allowlisted feature, because §4 freezes the allowlist. Stage 4 must pick
  one and record it: score them normally, exclude them from the eligible
  population, or report them as a separate slice. Their home-win rate is
  0.4561 against 0.5596 elsewhere, but at n=57 decided games that is a
  direction, not a measured effect — and it is growing (5-6 per season
  since 2022 against ~1 per season through 2011).
- `C ∈ {0.01, 0.1, 1, 10}`, ties broken toward smaller C. Select
  separately for `team_only` and `team_market` on their own eligible rows.
- Persist exact split game IDs and date boundaries; assert both
  disjointness AND strict chronological ordering across partitions, using
  actual dates, not season numbers alone.
- Freeze fitted parameters during T; earlier-completed games within T may
  still update later-game history (sequential pregame forecasting, not a
  preseason forecast of the whole season).
- Score `team_only`/home-frequency on full eligible coverage; separately
  compare every probability model against the market baseline on
  identical paired-quote game IDs only. The home-frequency probability
  stays fitted on its full training set — scoring a subset never refits it.
- Save every prediction: game ID, season, week, label, probability,
  model, variant, fold, eligibility, abstention reason. Report a failed
  fold plainly, don't drop it silently.

Tests / acceptance:

- Synthetic multi-season end-to-end test proves exact partitions and refits.
- Mutating an outer-test season's outcomes cannot change the selected C
  or the fitted preprocessing for that season. Mutating validation
  outcomes may legitimately change selection.
- Earlier test-game results may affect later features (the sequential
  contract); current/future results cannot affect earlier ones. Cover
  rescheduled dates and season boundaries.
- Coverage comparisons always use identical IDs; missing market data
  cannot silently change the team-only full-coverage score.

### 5. Reproducible development report

**Files:** `src/evaluation/report.py`, `tests/test_report.py`; extend
`metrics.py`/`backtest.py` rather than duplicating them.

Deliverables:

- One command runs the frozen protocol from a selected snapshot. Offline
  synthetic smoke run first, then the real development data.
- Artifacts write to `models_game/<target>/<run_id>/<feature_set>/<model>/`
  (run grouped under target, so one protocol run's full set of
  feature-set/model outputs stays co-located), never overwriting a prior
  run: run manifest, source revision/dirty diff, resolved dependencies,
  configuration, snapshot+content hashes, splits, exclusion counts,
  candidate validation scores, fitted models, predictions.
- A readable report: per-season metrics, pooled game-weighted metrics,
  and the equal-season mean reported separately; calibration plots+bin
  counts, full vs. matched coverage, failures, missing-data counts.
- Paired model-minus-baseline differences via 2,000 fixed-seed
  season-block bootstrap replicates, 95% intervals, blocks shared across
  compared models. Final single-season stage uses week blocks and states
  that between-season uncertainty is omitted.
- Record limitations plainly: retrospective corrected data, score-based
  completion proxy, closing-line market inputs, no betting-profitability
  claim.

Tests / acceptance:

- Repeat runs reproduce splits exactly and probabilities within 1e-8
  absolute in the same environment; manifests distinguish run identities.
- Block-resampling determinism, paired alignment, report aggregates.
- Every feature/probability artifact traces back to its exact source
  snapshot and code revision.
- Existing feature checks still pass unmodified (this stage never
  touches the feature layer):

  ```sh
  python3 tests/test_matchup_features.py
  python3 tests/test_rolling_features.py
  python3 tests/test_team_game_stats.py
  ```

  Every new test file in this stage gets its own documented run command
  here too — don't rely on implicit discovery.

### 6. Final evaluation and a saved-model prediction interface

**Status (2026-09-18): shipped in part.** The prediction interface
(`src/models/predict.py`, `tests/test_predict.py`) is built and verified —
loading fold `season_2024` of run `20260918T214515Z_1a1885fc` and scoring that
season reproduces the backtest's own 272 recorded probabilities to 0.0 absolute
difference. **The final evaluation is not built and must not be**: the frozen
protocol records that no untouched holdout exists (`experiment-protocol.md`,
"Holdout designation"), so there is nothing a final evaluation could honestly
be run against. Predictions therefore carry the run manifest's `study_type`
forward, which names the run a development backtest rather than a validated
result. Unblocking the rest of this stage means predeclaring an untouched
future block and documenting its exposure history — not re-scoring 2025 or
partial 2026.

**Files:** `src/models/predict.py`, `tests/test_predict.py`, final
evaluation report.

Start only after Stage 1's protocol is frozen and the holdout question is
actually resolved (see Stage 1) — not merely "no model has been fit yet."
No model/threshold re-selection based on the final result — any later
refinement is a new development experiment, not a tweak to this one.

A small batch prediction interface: loads a saved pipeline+schema,
accepts eligible pregame features, returns probabilities or explicit
abstentions with run/model identifiers. Missing outcomes never block
prediction. Test that scoring a completed row and the same pregame
feature row with its label removed yields identical probabilities.

This interface does not make historical closing quotes or the score
proxy safe for live use. Live deployment is a separate milestone needing
verified final status, quote receipt times, feature-availability
cutoffs, and operational tests — not scoped here.

## Work deliberately deferred

- XGBoost, neural networks, additional rolling windows, player-level
  inputs, weather/rest/injuries, team identity, additional outcome
  targets (`point_diff`, `spread_cover`).
- Post-hoc calibration until chronological calibration evaluation is
  designed.
- Betting strategies, staking, dashboards, live automation.
- Any change to the root-level player-model pipeline (`PlayerProjections.py`
  et al.) — separate pipeline per the project notes.

After the baseline report, choose improvements from a concrete
validation finding, not speculation. Before claiming team statistics add
information beyond market recalibration, introduce a fitted market-only
control.

**Status (2026-10-01): market-only control built (protocol v3) and run.**
Development data, matched coverage, 2008-2024, n=4316, run
`20261002T012354Z_a419db68`; paired season-block bootstrap, 2000 replicates:

| comparison | mean log-loss difference | 95% interval |
|---|---|---|
| `team_market − market_only` | +0.007786 | [+0.003788, +0.011686] |
| `market_only − market` | +0.001826 | [+0.001076, +0.002668] |
| `team_spread − market` | +0.003523 | [+0.001140, +0.005823] |
| `team_market − market` | +0.009612 | [+0.005490, +0.013635] |

The team statistics do not add information beyond a fitted market: adding the
76 team features to `market_only` makes it *worse*, with an interval that
excludes zero. Every fitted model loses to the raw devigged quote, and even
the market alone loses a little once it is refitted per fold. So no model here
adds information to the closing line. That is a development-data finding with
no holdout behind it, and per §6 it is not a claim about betting value either
way.

**Status (2026-10-01): residual model built (protocol v4) and run.** Same
window. Under the frozen C grid (run `20261002T014047Z_5186cdf2`) validation
pinned `C = 0.01`, the floor, in all 17 folds, and `team_residual` trailed the
market by +0.003732 [+0.000716, +0.006614]. With the grid extended downward for
this variant (v4 amendment, run `20261002T014705Z_a7524baa`):

| model | pooled log loss | vs `market` |
|---|---|---|
| `market` | 0.606980 | — |
| `team_residual` | 0.607388 | +0.000408 [−0.001303, +0.002202] |

**Tied with the closing line, not ahead of it.** It is the first model whose
interval against the market includes zero, but the point estimate is still on
the market's side. Starting from the market and letting the team statistics
adjust it recovers the market; the team statistics add nothing measurable to
it. Development data, no holdout, no betting claim. New feature sets or model families earn their complexity on
development data, never the final holdout.

## Recommended first implementation request

Implement Stage 0 only: fitting-free baseline and metric calculations,
restricted to 2005-2024, with saved scored rows and a checksum, and
offline tests with documented run commands. This produces a real,
checkable number — and a working `metrics.py` every later stage reuses —
without touching 2025/2026 or building reproducibility infrastructure
Stage 0 doesn't need yet. Then Stage 1, which now also has to resolve the
holdout-season question this round of review surfaced.

## Agreement record — 2026-09-15

the review branch's `a99e04f` reviewed this document at `5dd9f54` and reported no
substantive roadmap disagreement remaining. I accept that: the two
worktrees' documents describe the same sequencing, the same Stage 0/1
scope, and the same conservative treatment of 2025 as used development
data, in independently-worded but non-contradictory form. One addition
from the review branch's side is worth folding in explicitly rather than leaving
implicit: **Stages 0-5 may proceed while the untouched-holdout question
stays open** — only Stage 6 (final evaluation) is blocked on it, and a
development release without a resolved holdout must say so rather than
imply final validation.

Test commands and file names remain worktree-specific by design (see
"Review decisions" above); neither worktree needs to convert its tests to
match the other's. The agreed next task in this worktree is Stage 0 as
scoped above.
