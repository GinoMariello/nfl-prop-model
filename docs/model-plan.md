# NFL Game Prediction — Final Architecture

Status: design only. The adjudication records decisions on both archived
reviews; the architecture below is the current implementation specification.
Documentation follow-up: corrected validation/test metric separation, causal
history rules, market coverage, and reproducibility requirements. No model or
feature implementation has been changed by this documentation revision.

## 1. Adjudication of prior reviews

Severity/labels as originally assigned. "Verified" means I ran my own
read-only SQLite query or read the cited source line myself this turn,
separate from any evidence already in the archived text.

### 1.1 First review (my own, A1-A14)

| # | Verdict | Reasoning |
|---|---|---|
| A1 spread sign unverified | **Accept, resolved** | Verified directly: `2023_01_DET_KC` has home=KC, home_score=20, away_score=21, `spread_line=4.0`, `home_moneyline=-198` (KC favored). `point_diff = -1`. The flagged formula (`point_diff > -spread_line` → `-1 > -4` → True) would call a 4-point favorite that lost outright a "cover." It's wrong. Correct formula: **`point_diff > spread_line`** (`-1 > 4` → False, correctly "no cover"). `spread_line` is stored as the home team's expected margin, not a signed handicap. |
| A2 missing push case | **Accept** | `spread_cover` is deferred (§2), but the push-as-missing rule stands for whenever it ships. |
| A3 weather provenance | **Accept the provenance concern; reject the literal source claim** | `src/data/ingest.py` explicitly lists weather as pregame-safe, but does not establish forecast semantics or publication timestamps. A local assertion is insufficient evidence of historical availability. Weather remains excluded from v1 pending source verification. |
| A4 rest cold starts | **Accept** | Spec gap, not demonstrated leakage. Deferred with the feature (§4/§10). |
| A5 tuning-split wording | **Accept** | My own original §5 text really did conflate three different fits under one window label. The underlying design (tune on a held-out last season, refit including it, test on the next season) is standard and fine — the prose was the bug, not the method. Rewritten unambiguously in §5 below. |
| A6 season count | **Accept** | 1999-2025 inclusive = 27 seasons. Arithmetic, not debatable. |
| A7 vig/devig | **Accept, qualified** | Use a documented two-sided normalization convention for a comparable probability baseline. Proportional normalization is not a uniquely correct recovery of fair probabilities, and it does not guarantee improved empirical calibration. Validate quote format and paired availability. |
| A8 target-specific market baseline | **Accept** | Moot for v1 since only `home_win` ships (§2), but the underlying point is correct and recorded for when `spread_cover` ships: a closing line's own honest baseline is ~50%, not a devigged probability computed the same way as the moneyline one. |
| A9 missing point_diff baseline | **Accept, deferred** | Not needed to ship a `home_win` classifier; recorded for when `point_diff` ships (§10). |
| A10 no uncertainty reporting | **Accept** | Point estimates alone, across ~20 seasons of ~260 games, aren't enough to trust a small edge over market. Needs season-block resampling (§7). |
| A11 prefer one margin regressor | **Partially reject** | The complexity concern (three parallel target pipelines in v1) is valid and is why §2 scopes v1 to `home_win` only. But my original *recommendation* — lead with a point-differential regressor and derive win/cover probabilities from its residual distribution — adds a second free design decision (which residual distribution, fit how) instead of removing one. A directly-fit classifier trained on log loss is the simpler starting point for the actual primary target. Reversing my own earlier call here. |
| A12 artifact naming collision | **Accept** | `<target>_<model>` alone can't distinguish a market-included vs. market-excluded variant of the same model type. §8 fixes the naming scheme. |
| A13 metadata fingerprint | **Accept, confirmed** | Verified `StatScraper.py:81`'s `save_to_sqlite()` docstring: it's a key-based upsert with no ingestion-timestamp column added anywhere. Row count and season range genuinely cannot detect a same-shape, different-values re-pull after an nflverse correction. §8 requires a content hash instead. |
| A14 need a new reshape module + its test | **Reject the module, accept the test** | Verified `src/features/rolling_features.py` already contains `build_matchup_table()` (lines 56-75) and `run()` (writes `team_rolling_features` and `matchup_features` to `data/nfl_games.db`) — this already exists in the checked-out code. My original plan proposing a new `game_features.py` was solving an already-solved problem. What's still missing is a leak-safety test for that existing reshape (§9). |

### 1.2 Second review (B1-B11)

This review cites specific row counts and source lines. I re-ran the
underlying checks myself rather than trusting the citations at face value.

| # | Verdict | Reasoning |
|---|---|---|
| B1 wrong spread sign | **Accept, independently confirmed** | Same KC/DET row, same conclusion as A1 — I queried it myself before reading this finding's claimed evidence, not after. |
| B2 unplayed games mislabeled | **Accept, independently confirmed** | Queried `data/nfl_games.db` directly: **257 rows** have a null score. Read `build_matchup_table()`: `home_win = point_diff > 0` then `.where(point_diff != 0, other=NA)`. For a NaN `point_diff` (missing score), `NaN > 0` evaluates to Python/NumPy `False` (not NaN), and `NaN != 0` evaluates to `True` — so the `.where()` mask *keeps* that `False` instead of replacing it with `NA`. Unplayed games are silently labeled `home_win = 0`. This is a live bug in code that exists right now, not a hypothetical. Must fix before any label is trusted (§9, gate 1). |
| B3 target columns present in matchup output | **Accept** | Verified: `build_matchup_table()` merges `home_score`, `away_score`, `point_diff`, `home_win` into the same table it returns/saves alongside the rolling features. Agree with the hedge — this is a hazard for a future "select all columns" mistake, not proof anything has leaked yet, since no model consumes this table today. §3/§9 require an explicit allowlist. |
| B4 availability assumptions | **Accept, unresolved** | Can't independently verify nflverse's own publication timestamps from inside this repo. Treated as an open gate, not a proven leak (§4). |
| B5 cutoff / final-evaluation framing | **Accept** | Sound distinction between "sequential pregame forecasting" (features may use earlier-in-season results) and "preseason forecast of a whole season" (they may not). §5 states this explicitly. |
| B6 no reproducibility timestamps | **Accept, independently confirmed** | Read `StatScraper.py:81` (`save_to_sqlite`) myself: upserts on `key_columns`, no timestamp column written anywhere in the function. |
| B7 bad baseline definitions | **Accept** | Correct on the math: a constant `P(home_win)=1` baseline is fine for accuracy but has unbounded log loss the moment it's wrong once. A probability baseline needs an actual probability (§6). |
| B8 incomplete market coverage | **Accept, independently confirmed** | Queried it myself: 7,548 total games, **2,221 null `home_moneyline`**, **2,221 null `away_moneyline`** — exact match to the cited figures, confirmed before I read them. §6 requires matched-coverage comparison. |
| B9 metrics/uncertainty policy | **Accept** | Consistent with A7/A10; folded into §7. |
| B10 duplicate reshape | **Accept, independently confirmed** | Same file/line evidence as A14 — `build_matchup_table()`/`run()` already exist and already write `team_rolling_features` + `matchup_features`. No second reshape module needed. |
| B11 four methodological corrections | **Accept, all four, independently reasoned** | (1) Team ID: correct — a raw team-ID feature doesn't violate chronological ordering by itself if splits stay strictly time-based; the real argument against it is that a static per-team effect goes stale as rosters turn over, which is a generalization argument, not a leakage one. My original doc conflated the two. (2) Within-fold shuffling (e.g. minibatch order) isn't the same thing as shuffling across the train/test boundary — fair, and my "no shuffling, ever" phrasing didn't make that distinction. (3) Expanding training history doesn't change the rolling-feature window itself (fixed at 8 games) and older eras may reflect a different game (rule changes) rather than strictly helping — expanding-vs-recent-window should be an empirical comparison, not an assumption. (4) 1999-2004 is six seasons, so 2005 is the seventh, not the sixth — off-by-one, confirmed by counting. |

### 1.3 Player-pipeline findings (root-level `PlayerProjections.py` etc.)

Out of scope for this architecture (separate pipeline per the project notes), but
independently verified since they were raised, and worth a follow-up:

| Finding | Verdict | Verification |
|---|---|---|
| Future-roster lookup in backtesting | **Accept** | Read `PlayerProjections.py:1687`: `player_age_rows = player_age[player_age["player_id"] == player_id].sort_values(...)` then takes `.iloc[-1]` — the *latest row in whatever table is passed in*, with no cutoff filter inside the function itself. Whether this leaks depends entirely on caller discipline; not proven to have fired in practice, but the function offers no protection. |
| Pre-split imputation uses full-table stats | **Accept** | Read `PlayerProjections.py:2275`: `pos_df["player_vs_opponent_difficulty"].fillna(pos_df[rank_stat].mean())` runs on the full `pos_df`, and `chronological_split()` (which produces `train_idx`/`val_idx`) isn't called until line 2287 — twelve lines later. The fill value is computed from data that includes the eventual validation rows. |
| Chronological split cuts by row count, not time groups | **Accept** | Read `PlayerProjections.py:1930-1949`: `chronological_split()` sorts by `["season","week"]` then slices by `int(len(ordered) * (1 - val_frac))` — a fixed-fraction row-count cutoff, which can fall inside a season/week that has more than one row (e.g. multiple players sharing that week), splitting it across train and validation. |
| GA "validation" is actually a selection signal | **Accept** | Read `PlayerProjections.py:3868-3888` (`train_position_model_ga`): `val_loader` is scored every generation via `_evaluate_fitness`, and that score drives `ranked = sorted(...)` which selects survivors/parents each generation. The same set used for reported "val_loss" is the set the GA is actively selecting against — it's an in-loop optimization signal, not a held-out check. |

## 2. Scope and targets (v1)

**Implementation amendment (2026-09-15):** the agreed historical-snapshot
pipeline uses valid scores as a completion proxy when ingestion supplies no
independent status. An optional explicit completion flag takes precedence;
false/unknown status is excluded. Proxy usage is reported by season. This is
an exception to the independent-status requirement below for historical studies,
not verified live completion or as-of availability. See `feature-pipeline.md`
for the implemented contract and test commands.

**First release covers regular-season games and ships `home_win` only** — binary, conditional on a
non-tied result (per A11's resolution: one directly-fit classifier, not
three parallel target pipelines). `point_diff` and `spread_cover` are
deferred to §10, not built now.

- `home_win = 1` if `home_score > away_score`, `0` if lower, **missing**
  (not 0, not 1) if either score is null or the game is a tie. Games with a
  missing label are excluded from training/evaluation, not silently
  coerced (fixes the confirmed bug in B2 — see §9, gate 1, for the required
  fix and test).
- `spread_cover` (deferred): `point_diff > spread_line` (corrected sign
  per A1/B1 — **not** the original plan's `point_diff > -spread_line`),
  missing on a push (`point_diff == spread_line`) or a missing score/line.
- `point_diff` (deferred): `home_score - away_score`, missing if either
  score is missing.

Only completed regular-season games enter supervised fitting and scoring.
Preseason and postseason are excluded from v1 training, scoring, and rolling
history; their raw records remain intact. A tied completed game may contribute
historical statistics even though it has no binary label. Completion status and
score completeness must both be checked; an unfinished score is not a final
outcome. All exclusions are counted by season and reason.

## 3. Components and data ownership

Extend the existing pipeline; no new reshape module (A14/B10).

| Component | Status | Responsibility |
|---|---|---|
| `src/data/ingest.py` | exists | Raw `games`/`plays` fetch into `data/nfl_games.db` |
| `src/features/team_game_stats.py` | exists | Per-game box scores, not leak-safe alone |
| `src/features/rolling_features.py` | exists, **needs revision** | Preserve shift-before-roll and the single `build_matchup_table()` reshape; correct missing-outcome labels, chronological ordering, and completed-history counts/windows (§4). |
| `src/data/snapshot.py` | new | Immutable raw snapshot + checksum/source manifest (§8) |
| `src/models/baselines.py` | new | The baselines in §6 |
| `src/models/train.py` | new | Explicit feature allowlist, sklearn preprocessing + logistic regression, fit/predict |
| `src/evaluation/backtest.py` | new | Season-expanding-window harness (§5) shared by baselines and trained models |
| `src/evaluation/metrics.py` | new | Metrics/uncertainty in §7 |

`matchup_features` (existing table) is never handed to a model as-is —
`src/models/train.py` selects an explicit column allowlist before fitting
anything, so `home_score`/`away_score`/`point_diff`/`home_win` (and any
future target column) can never leak in by omission (B3).

## 4. Feature and availability rules

v1 uses home/away rolling metrics named explicitly in the existing
`STAT_COLUMNS`, plus prior-completed-game counts. Store the exact ordered
allowlist in the run configuration. Preserve the eight-game window and
shift-before-roll principle, but do not leave the current history builder
unchanged:

- Admit only prior completed regular-season games. Unplayed schedule rows do
  not advance the rolling window or `games_played_before`.
- Order by actual game date/time and stable game ID, not just season/week.
  Without reliable completion times, conservatively admit only strictly earlier
  game dates; ambiguous dates are excluded and reported. In a verified replay,
  require data availability before the prediction cutoff as well.
- Keep history across seasons. A completed game missing play statistics counts
  as completed history, with missing values for its unavailable metrics.
- First-history features stay missing until training-only imputation. No target
  scores, current-game aggregates, labels, IDs, or timestamps enter X directly.

Two feature variants:

- `team_only`: rolling stats and completed-history counts, trained and evaluated
  on all eligible games.
- `team_market`: the same features plus normalized home moneyline probability,
  trained and evaluated only where both valid quotes exist. It abstains on other
  rows. Market probabilities are never fabricated by imputation.

For numeric team features, use a sklearn pipeline with training-fold median
imputation, fixed missingness indicators, standardization, and L2 logistic
regression. All-missing training columns use a documented zero fallback with
an indicator, retaining schema order. No class weighting or feature selection
in v1. Refit every learned preprocessing step with each fit/refit.

The historical all-game snapshot had 2,221 missing entries on each moneyline
column among 7,548 games. Equal separate counts alone do not establish that
the same rows are missing; compute the paired eligibility mask directly for
each regular-season fold.

Devig: convert each American odds value `a` to `q = 100/(a+100)` (a>0) or
`-a/(-a+100)` (a<0), then `p_home = q_home / (q_home + q_away)`. This is a
normalization convention, not a claim of the bookmaker's "true" probability
(A7).

**Excluded from v1, pending the provenance gate in §9:** weather
(`temp`/`wind`), rest days, starters/injuries, `is_home` as an explicit
column (it's redundant with the home/away row structure itself — A-series
already flagged the earlier "constant 1" version of this as adding no
information). Team ID is excluded from v1 not because it leaks (B11
corrects that framing) but because static per-team identity goes stale
across a 27-season span and isn't worth the added feature-engineering
surface for a first release; revisit once `team_only`/`team_market` have a
working baseline to compare against.

## 5. Chronological validation protocol

**Prediction cutoff:** immediately before each game's kickoff. Models are fitted
before the evaluation season and held fixed through it. Earlier completed games
in that season may update later feature history. This is sequential pregame
forecasting, not a preseason forecast of the full season.

Historical experiments use immutable source snapshots with causal outcome
ordering. Corrected historical values do not prove historical publication
availability. Label these as **historical snapshot studies**, not exact as-of
replays. Document upstream field/model versions, including EPA where available.
A verified replay additionally requires recorded publication/receipt times.
Closing-line studies must be labeled as such; live inference can use only quotes
actually received before cutoff, never a closing quote identified afterward.

Initial development outer seasons: 2005–2024. 1999–2025 is 27 seasons;
1999–2004 supplies six seasons before the first outer evaluation in 2005.
For each outer season T:

1. Fit candidate pipelines on 1999 through T-2.
2. Select regularization using **validation log loss on T-1**. Initial C grid:
   0.01, 0.1, 1, 10; break ties toward stronger regularization. Record solver,
   tolerance, iteration limit, and seed. Report convergence failures explicitly.
3. Refit the selected pipeline, including preprocessing, on 1999 through T-1.
4. Freeze fitted parameters and score T. Test outcomes cannot select parameters.

There are candidate fits and a final refit, not necessarily exactly three fits.
All baselines use the same chronological eligibility rules. Never randomly split
time groups. Within-training row shuffling is not itself leakage.

Use expanding history for the initial baseline. A bounded recent window can be
a later validation-selected experiment; it is not guaranteed better or worse.
Reserve 2025 for final evaluation only after documenting that its model results
have not guided decisions. If already used, mark it development data and reserve
an untouched future block. 2026 is partial in the current snapshot. Freeze the
protocol before inspecting final results; repeated inspection makes a holdout
part of development.

No post-hoc probability calibrator in v1; measure calibration directly. A later
calibrator must be fit/selected using chronological pre-test data, with its
reported evaluation outcomes held out from that fitting.

## 6. Baselines

| Baseline | Definition | Fit rule |
|---|---|---|
| Always-home | Predict `home_win = 1` | No fitting; accuracy only, not a probability baseline (B7) |
| Home-frequency | Training-fold non-tie home-win rate as a constant probability | Fit per fold |
| Devigged moneyline | §4's `p_home` | No outcome fitting; requires both moneylines present, scored only on that matched subset, with a separately reported full-coverage model score for comparison (B8) |
| `team_only` logistic | Rolling stats, no market | Fit per fold |
| `team_market` logistic | Rolling stats + normalized market feature | Fit per fold on paired-quote rows only; impute team metrics only |

Report `team_only` and home-frequency results on all eligible games. Compare
all probability models and the market baseline separately on the exact same
paired-quote game IDs. `team_market` and the direct market baseline have no
full-coverage score. Save masks and exclusions; reject zero/invalid American
quotes. Do not compare one model's full sample against another's restricted sample.
A gain from `team_market` alone does not prove team statistics add value beyond
market recalibration; a later market-only fitted control is required for that
stronger claim.

## 7. Metrics and decision rules

Primary tuning objective: **validation-season log loss**. Outer-test log loss
is a reporting metric, never a tuning signal. Supporting metrics: Brier score,
accuracy at 0.5, ROC AUC, and a ten-bin equal-width reliability diagram on [0,1]
with bin counts. ECE is secondary and bin-dependent. Report empty bins and
undefined metrics explicitly; validate probability ranges and record numerical
clipping used for log loss.

Report per-season scores, pooled game-weighted scores, and the equal-season
mean separately. For market comparisons compute paired differences on identical
games. Use 2,000 fixed-seed season-block bootstrap replicates across development
seasons and 95% intervals. Resample the same blocks for each model. For a single
final season use week blocks and state that uncertainty omits between-season
variation. These are approximate intervals, not protection against repeated
tuning or distribution shift.

A completed baseline need not beat the market. Any superiority claim must show
matched coverage, effect size, uncertainty, and untouched final evaluation.
Classification improvements alone do not establish betting profitability.

## 8. Reproducibility and artifacts

`save_to_sqlite()` adds no ingestion timestamp. Each experiment requires an
immutable, retrievable raw snapshot, a checksum, source/release identifiers and
URLs, field semantics, and retrieval time. Legacy retrieval times stay unknown
unless evidence exists. A checksum detects change but cannot reconstruct lost
content. Use a consistent SQLite backup when capturing a live database; never
modify the selected snapshot during training or evaluation.

Each run saves:

- Snapshot identity/checksum and canonical hashes of the exact feature/label
  content, using stable ordering and a documented missing-value representation.
- Ordered feature schema, eligibility policy, target definition, exact split
  game IDs, cutoff rules, market masks, and exclusion counts.
- Seeds, candidate scores, selected hyperparameters, preprocessing/model
  artifacts, per-game predictions, metrics, and uncertainty outputs.
- Python/library versions, a resolved compatible dependency specification,
  platform, source revision, and dirty diff/source archive as needed.

Use `.venv`; specify compatible ingestion/modeling dependencies during
implementation rather than assuming the existing requirements cover modeling.
Do not retrain player models as part of environment verification.

Artifacts write to `models_game/<run_id>/<feature_set>/<model>/`, never
overwriting earlier runs. Target is explicitly recorded as `home_win` in v1
metadata; add a target path component before introducing additional targets.
Replay must reproduce splits exactly and predictions within a documented
same-environment tolerance (initial absolute probability tolerance 1e-8).

## 9. Validation gates (before this is trusted)

1. **Fix and test the `home_win` bug.** `build_matchup_table()` must
   produce a missing label (not `False`) when `point_diff` is null.
   Use synthetic missing-score/tie cases and verify all currently incomplete
   rows. The prior 257-row count is an observation, not a permanent test fixture.
2. **Allowlist test.** Assert that the model's input `X` never contains
   `home_score`, `away_score`, `point_diff`, `home_win`, or any future
   target column, by construction, not by convention (B3).
3. **Leak-safety test on the existing reshape**, same spirit as
   `tests/test_rolling_features.py`'s corruption test: plant a value in
   one team's stats for one game and confirm it's invisible to that same
   game's `matchup_features` row but visible in that team's *next* game's
   row (A14/B10 — extends the existing test to `build_matchup_table()`
   rather than creating a parallel one).
4. **Split-boundary test**: mutating an outer-test season's outcomes must
   not change any earlier season's fitted preprocessing or chosen
   hyperparameters.
5. **Spread-sign regression test**: lock `point_diff > spread_line` against
   the verified `2023_01_DET_KC` example (and a favorite that *did* cover,
   for the other direction) so this can't silently flip again.

6. **Completed-history and availability tests:** future schedule placeholders
   must not change completed counts/windows. Cover both teams, season boundaries,
   franchise mappings, reschedules, unknown dates, missing metrics, and cold starts.
7. **Preprocessing/tuning isolation:** changes to validation/test values cannot
   affect inner-training preprocessing. Validation labels may legitimately change
   candidate selection; outer-test labels cannot. Earlier test-game outcomes may
   affect later test-game features under the sequential contract in §5.
8. **Metric and replay tests:** known probability/metric examples, invalid odds,
   paired-coverage masks, convergence reporting, and snapshot-preserving replay.
9. **Provenance gate:** document source definitions, quote semantics, and the
   evidence mode for every enabled feature. Exclude optional unverified inputs;
   never label a historical-snapshot study as verified as-of performance.

Use small offline tests for these invariants. Existing `demo()` scripts must be
invoked explicitly; ordinary discovery does not run them. Ingestion tests need
network access, while cached-data checks do not. Feature module entry points
rebuild database tables and are not read-only smoke tests. No test/training run
is needed solely for this documentation revision.

## 10. Deferred to later releases

- `point_diff` regression and `spread_cover` classification (A9/A11), each
  with their own market-free baseline once they're scoped.
- Weather, rest days, starters/injuries as features, once their
  pre-cutoff availability is independently confirmed against nflverse's
  own field documentation (A3/B4) — not assumed from this project's
  ingestion docstring alone.
- A revisit of team-identity features (A11/B11), once `team_only`/
  `team_market` baselines exist to compare against.
- A follow-up on the four identified root-level
  player-pipeline issues in §1.3 (future-roster lookup, pre-split
  imputation, row-count chronological split, GA validation-as-selection) —
  a separate pipeline, out of scope for this document, but real and worth
  its own ticket.

## Archived reviews

The text below is preserved for traceability. Its references to original §§1–9
refer to the superseded plan, and its recommendations are governed by the
adjudication above. It does not describe implemented fixes.

## 10. Independent Review

Independent review of §§1-9 above, before any implementation. Findings
below, grouped by the categories requested; severity is High/Medium/Low.
Nothing here has been fixed in the sections above — this is a review, not
an edit.

### Target leakage

1. **(High)** §1's `spread_cover` definition (`point_diff > -spread_line`)
   bakes in a specific sign convention for `spread_line` ("negative = home
   favored") that is asserted, not verified against nflverse's actual
   schema. If the convention is reversed, the label is silently inverted
   for every game — the walk-forward backtest would still run cleanly and
   produce plausible-looking numbers with no error to catch the mistake.
   Before implementing: hand-check the sign against a few real games with
   well-known spreads (e.g. a double-digit playoff favorite) and confirm
   which direction `spread_line` points.
2. **(Low)** The same definition has no push case
   (`point_diff == -spread_line`) — a push currently falls into the "not
   cover" class. Sportsbooks treat a push as a third outcome (stake
   returned, not a loss); folding it into "not cover" penalizes the model
   for correctly predicting a genuine coin-flip result. Push rows should
   likely be excluded from `spread_cover` training and evaluation, not
   relabeled.

### Future information leaking into features

3. **(Medium)** §8 states that `game_temp`/`game_wind` are confirmed
   pre-game "by the existing `src/data/ingest.py` docstring." That
   docstring's leakage classification (quoted earlier in this project)
   explicitly covers `spread_line`/`total_line`/moneylines and play-level
   stats — it does not, in what's been reviewed, make any claim about
   whether temp/wind are a pre-kickoff forecast or a conditions-at-kickoff
   (or later) reading. §8 is citing a source for a claim the source hasn't
   actually been shown to make. This needs an explicit check against
   nflverse's own field documentation before the plan can call weather
   "confirmed" pre-game-safe.
4. **(Low)** `rest_days` is described as computed from "each team's prior
   game date," but the boundary case — a team's first game in the whole
   1999-2025 table, or the first game after an offseason — isn't given an
   explicit null/imputation rule. Left implicit, this is the kind of gap
   where an implementer might default a missing prior-game gap to 0 (a
   short-rest short week) instead of a neutral/typical value, quietly
   biasing every season-opener row.

### Incorrect train/test splitting

5. **(High)** §5's hyperparameter-tuning bullet is internally
   inconsistent. It calls "1999-2010" the training window, says the *last*
   season (2010) is "held out" for tuning, and then says "refit on
   1999-2010" — three different implied definitions of what's actually in
   the fit at tuning time. As worded, an implementer could easily fit once
   on the full 1999-2010 window and then score that same fit on 2010 to
   "pick" hyperparameters — which is in-sample and leaks the validation
   season into the model that's supposedly being validated against it.
   Needs to be stated unambiguously as three distinct fits: fit on
   1999-2009 → select hyperparameters by scoring on 2010 → refit on
   1999-2010 with the winning hyperparameters → score the true 2011 test
   season only with that final refit.
6. **(Low)** §5 says "26 seasons" for 1999-2025 inclusive; that's 27. Purely
   arithmetic, but it's the number used to justify the walk-forward loop
   being "cheap," so worth fixing.

### Poor evaluation metrics

7. **(High)** Baseline 2 (market-implied probability) doesn't mention
   removing the bookmaker's vig/overround before converting moneylines to
   probabilities. Raw implied probabilities from both sides of a moneyline
   sum to more than 100%; scoring that directly with log loss/Brier/
   calibration makes the market look artificially worse-calibrated than it
   actually is, which undercuts the entire reason for using it as the
   hardest baseline to beat (§6's own stated purpose). Needs devigging
   (proportional or logit-based normalization) before any baseline-2 metric
   is computed.
8. **(Medium)** §6 doesn't say what "market-implied" means for
   `spread_cover`. For `home_win`/`point_diff`, it's the devigged moneyline
   probability; for `spread_cover`, the line itself defines the outcome
   being predicted, so a well-set closing spread makes the honest market
   baseline ~50%, not a devigged probability computed the same way as
   baseline 2. As written it's ambiguous which of the three targets each
   baseline actually applies to.
9. **(Medium)** Baseline 4 (market-free logistic regression) only covers
   the classification targets. There's no market-free baseline defined for
   `point_diff` (e.g. OLS on the same rolling-feature set) — so the
   "how much of a fancier model's performance is market vs. real skill"
   comparison that §6 argues for is missing for the one target named after
   the exact thing (point differential) that comparison would be most
   useful on.
10. **(Medium)** All comparisons in §7 are point estimates against
    baselines. With ~20 walk-forward test seasons of ~260 games each,
    season-to-season metric noise is real (rule changes, an unusually wild
    season) — §7 should report per-season variance or a paired
    significance check across seasons alongside the pooled metric, not
    just the mean. Otherwise a genuine-looking improvement over the market
    baseline could just as easily be noise concentrated in one or two
    seasons.

### Unnecessary complexity

11. **(Medium)** §1 commits to training `home_win`, `point_diff`, and
    `spread_cover` as three fully independent models (× every model type
    in §6/§7 × every walk-forward window in §5), but `home_win` and
    `spread_cover` are both deterministic functions of `point_diff` (sign,
    and comparison to the spread) plus an assumption about the residual
    distribution. Before committing to three parallel training/evaluation
    pipelines, it's worth checking whether a single point-differential
    regressor with a fitted residual distribution produces competitive
    `home_win`/`spread_cover` probabilities on its own — and only adding
    dedicated classifiers if they measurably beat the derived versions.
    As scoped now, every future model type triples in both code and
    compute for a distinction that hasn't been shown to earn its keep.
12. **(Low)** §2's artifact naming
    (`models_game/<target>_<model>.<ext>`) has no room for the
    market-included vs. market-excluded feature variant from §4/§6 —
    baseline 4's logistic regression and a hypothetical market-including
    logistic regression would both resolve to the same filename. Needs a
    feature-set tag in the naming scheme before any training code is
    written against it.

### Reproducibility problems

13. **(Medium)** §9's per-run metadata records row count and season range,
    but not a content fingerprint of `game_features`. §3 already
    acknowledges nflverse revises data in place — two ingests with
    identical row counts and season ranges can still contain different
    values after a corrections update, and row count alone won't detect
    that. Metadata should include a hash of the actual `game_features`
    rows used to fit each model, not just a count.
14. **(Low)** AGENTS.md requires tests for feature generation, and the
    codebase already sets a concrete precedent for what that looks like
    here — `tests/test_rolling_features.py`'s corruption-injection
    leak-safety proof. §2's file list for the new `game_features.py` has no
    corresponding test planned; it should get an equivalent proof (plant a
    value that a given game's row shouldn't be able to see, rebuild, and
    assert it didn't leak through the home/away reshape), not just inherit
    `rolling_features.py`'s existing test by assumption.


## Independent Review

Independent ML/code review, 2026-09-14. Reviewed the project notes,
this plan, the existing game feature pipeline, shared SQLite storage, and
relevant player-model evaluation paths. Database checks used SQLite read-only
mode. No implementation, training, ingestion, or test scripts were run.

**Assessment:** the plan has a sound starting direction: separate raw and derived
outputs, chronological evaluation, simple baselines, and probability scoring.
However, target definitions, feature availability, and experiment provenance
need correction before results can be trusted. Findings below distinguish
confirmed code/data defects from unresolved design requirements.

### High priority — resolve before training

1. **The spread-cover target has the wrong sign (§1).** The stored
   `spread_line` represents the home team's expected point advantage, rather
   than the signed handicap assumed by `point_diff > -spread_line`. Local
   example: `2023_01_DET_KC` has KC home, score 20–21, spread `4.0`, and home
   moneyline `-198`. The proposed formula labels KC as covering because
   `-1 > -4`; KC lost while favored by four. For this stored convention,
   compare `point_diff > spread_line`. Specify pushes separately when the
   difference equals the line; missing lines must produce missing cover
   labels. Add favorite, underdog, pick'em, and push examples as tests.

2. **Unplayed games are already labeled as losses, and other outcomes lack
   an explicit policy (§1).** `src/features/rolling_features.py:67` computes
   `home_win` from a comparison against a potentially missing point
   differential, then only masks ties. Read-only inspection found **257 rows
   with missing point differentials and `home_win = 0`**. Exclude incomplete
   games from supervised fitting/evaluation and retain missing outcomes for
   inference. Explicitly define whether ties are excluded from binary win
   evaluation or modeled separately, and report counts. Do the same for
   spread pushes. A simple `home_win.notna()` filter is currently insufficient.

3. **The existing matchup table contains direct target-leakage columns
   (§§3–4, 8).** `build_matchup_table()` returns `home_score`, `away_score`,
   `point_diff`, and `home_win` together with features. Selecting all numeric
   columns or dropping only the chosen target would leak the answer. Require
   an explicit feature allowlist for each model variant; exclude every
   outcome-derived column for every target. Raw/derived table separation
   alone does not enforce this. In particular, raw scores are targets even
   though no feature engineering was required to produce them.

4. **Availability before prediction is asserted, not established (§§4, 8).**
   Define the prediction cutoff: immediately before each kickoff, a fixed
   earlier time, or a weekly batch. Closing lines cannot support an earlier
   prediction claim. A local ingestion docstring is not evidence that
   historical weather fields were published before that cutoff, nor that
   recorded starters match announcements available then. Record source field
   definitions and publication/snapshot times; exclude unverified fields
   from the first model. Historical corrected data can support a retrospective
   study, but should not be described as an exact replay of available data
   without that provenance. External source semantics were not independently
   verified in this review.

5. **The split plan needs a prediction-time contract and an untouched final
   evaluation (§5).** Holding out the last training season for tuning is a
   reasonable design. Explicitly fit imputation, scaling, feature selection,
   and any probability calibration using only appropriate pre-test data.
   Freeze model choices before the outer test; reserve a final season/block
   if outer-fold results will guide subsequent development. Specify whether
   test-season features update after earlier games finish: this is legitimate
   for sequential pregame predictions, but leaks unavailable information into
   a preseason forecast of the entire season. Assert each contributing game
   finished before the prediction cutoff, including rescheduled games.

6. **Reproducibility claims exceed the implementation (§§3, 9).**
   `StatScraper.py:81` does not add ingestion timestamps; it upserts supplied
   columns. Existing `games` has no ingestion timestamp column. Row counts
   and season ranges cannot detect changed values, and a timestamp cannot
   reconstruct overwritten data. Preserve a retrievable immutable input
   snapshot, its checksum, source/release identifiers, and retrieval time.
   Record resolved dependencies, run configuration, seeds, exact split game
   IDs, feature schema, predictions, and code version/dirty diff. Use unique
   run directories rather than overwriting `<target>_<model>` artifacts.
   `requirements.txt` is incomplete for the planned modeling stack and has
   no reproducible resolved dependency set.

### Medium priority — evaluation and unnecessary complexity

7. **Baselines need task-specific outputs and fitting rules (§§6–7, 9).**
   Always predicting home victory is an accuracy baseline, not a sensible
   probability baseline: probability 1 incurs extreme log loss on losses.
   Add a training-fold home-win frequency baseline for probability metrics.
   Logistic regression does require fitting, contrary to §9. Define moneyline
   conversion and removal of the bookmaker margin using both teams' implied
   probabilities; any learned spread-to-win mapping must fit on training
   data only. Moneyline win probability is not spread-cover probability.
   Use target-appropriate baselines: e.g. constant cover probability for
   cover classification, and zero plus market-implied margin for regression.
   Do not apply all four baselines to every target/metric indiscriminately.

8. **Market coverage is not complete (§4).** In the current 7,548-game table,
   both home and away moneylines are missing on **2,221 rows each**. The
   ingestion docstring's completeness statement about spread/total does not
   establish moneyline completeness. Define missing-value handling inside
   each fold. Compare model and market on identical eligible game IDs,
   alongside full-coverage model results and exclusion counts; otherwise
   apparent gains can reflect different eras or samples.

9. **Metrics are mostly appropriate but need a selection and uncertainty
   policy (§7).** Predeclare one tuning objective, such as home-win log
   loss, with Brier score and calibration diagnostics as supporting measures.
   Fix calibration-bin rules and show bin counts; per-season ECE can be
   unstable and depends on binning. Report paired model-minus-baseline
   differences with uncertainty accounting for temporal clustering, such as
   week/season block resampling. Record resampling seeds. Distinguish pooled
   game-weighted scores from an equal-season average. Never fit a probability
   calibrator on the outcomes used to report its performance.

10. **A new reshape stage duplicates existing code (§§2–3).**
    `rolling_features.py:56` already joins home/away rows and market columns,
    and `run()` writes `team_rolling_features` and `matchup_features`.
    The plan calls the existing table `rolling_features` and proposes this
    reshape as new work. Extend or extract the existing builder rather than
    maintaining two definitions of a matchup. Start with the primary win
    target, constant/market baselines, and market-free/market-inclusive
    logistic regression. Defer auxiliary targets and elaborate model families
    until an evaluated baseline and a concrete performance gap justify them.

11. **Several methodological explanations are incorrect (§§4–5, 8).**
    Team IDs do not inherently leak future seasons if training is correctly
    isolated; excluding them is a modeling choice. Shuffling independent
    training examples within an already isolated historical fold is not
    equivalent to randomly splitting time. Expanding training history does
    not make a fixed eight-game feature window longer or necessarily improve
    accuracy under changing conditions. Compare expanding versus recent
    training windows on validation data if needed. Also, 1999–2004 is six
    training seasons, so 2005 is the seventh season, not the sixth.

### Existing player pipeline — caution when borrowing its design

These findings concern the separate root-level player models; they are not
claims that the proposed game model already executes these paths.

- **Future roster information in backtesting:**
  `walk_forward_train_and_predict()` passes unfiltered `player_age` to
  `current_matchup_context()`. At `PlayerProjections.py:1687`, that helper
  selects the player's latest row across all supplied seasons, so a historical
  prediction can use later age/experience. Pass an explicit target game and
  as-of cutoff to every lookup. The schedule lookup also infers the next
  meeting after the player's last appearance rather than accepting the exact
  evaluated game, risking wrong-game weather/injury context after absences
  or team changes. This is a context-alignment risk; future injury leakage
  was not independently demonstrated.
- **Pre-split imputation:** `fillna_context()` uses full-table means;
  `prepare_position_training_data():2275` fills opponent-difficulty features
  using the position's full target-stat mean before splitting. This includes
  validation outcomes. Training-only normalization later does not undo it.
- **A chronological sort is insufficient:** `chronological_split():1930`
  cuts by row count after sorting season/week, potentially dividing the same
  week and even the same game's players across training and validation.
  Split on complete time groups with strict boundary assertions.
- **GA validation is an optimization set:** `train_position_model_ga():3868`
  repeatedly evaluates and selects populations using `val_loader`. Those
  fitness scores are not held-out performance; an independent later test
  is required. Player MAE alone also does not validate the coverage of
  quantile or Monte Carlo intervals.

### Required validation before accepting a model

Retain the existing shift-before-rolling approach and corruption test, but
extend coverage to the complete feature and evaluation path:

- Changing current/future outcomes must not change an earlier prediction's
  features or fitted preprocessing; verify both teams and season boundaries.
- Enforce unique game keys, correct home/away joins, and an explicit feature
  allowlist that rejects every target-derived column.
- Test unplayed games, ties, spread signs/pushes, missing lines/weather,
  rescheduled games, and future schedule placeholders. Currently rolling
  counts/windows advance over unplayed scheduled rows as well as completed
  games; they must not be reported as completed history.
- Assert disjoint chronological train/validation/test groups and timestamp
  availability. Test that mutating validation/test values cannot change
  training-fitted preprocessing or hyperparameter selection from earlier data.
- Verify probability ranges, known metric examples, eligible-row alignment,
  and that calibration fitting never sees its evaluation outcomes.
- Reproduce identical splits and predictions from the saved data snapshot,
  configuration, and environment within a documented numerical tolerance.

The existing tests are useful but limited: their assertions live in manually
invoked `demo()` functions, and the rolling corruption test exercises selected
KC rows rather than all ingestion, preprocessing, and prediction paths.
Document a reliable test command. the project notes also overstates network use:
`tests/test_team_game_stats.py` and `tests/test_rolling_features.py` read cached
SQLite data, while `tests/test_ingest.py` downloads data. Feature module entry
points rebuild database tables and should not be treated as read-only checks.

**Recommended next step:** revise the plan around the high-priority findings,
then implement the smallest game-level baseline with those validation gates.
No predictive-performance conclusion is supported by this review alone.
