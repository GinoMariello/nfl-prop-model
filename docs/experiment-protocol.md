# Experiment protocol (frozen at Stage 1, 2026-09-15)

Frozen per `docs/next-implementations.md` Stage 1: these definitions,
conventions, and constants do not change for the life of this development
release. A change to any of them is a new protocol version, not a tweak —
re-run affected stages under the new version and say so explicitly; never
edit a past result to match a later protocol.

## `home_win` definition

From `src/features/rolling_features.py::_compute_labels`: a game gets a
label only if all of

- `game_type == "REG"` (no preseason/playoffs),
- completed (see "Completion assumption" below),
- `gameday` parses as an unambiguous `YYYY-MM-DD` date,
- `home_score != away_score` (ties are excluded, not labeled 0.5).

`home_win = home_score > away_score` (0/1) for exactly those rows;
everything else gets a null label and is never scored.

## Completion assumption

From `src/features/rolling_features.py::_completed`: an explicit
`completed` column (boolean/0/1/null only) wins when present. Without one,
a game is assumed completed if both `home_score`/`away_score` are finite,
non-negative integers — a historical-snapshot proxy, not a live
final-status detector. This proxy is why 2025/2026 need their own
handling (see "Holdout designation") rather than being trusted as
"complete" just because they have scores.

## Cutoff convention (pregame-safe data)

- Rolling team features (`add_rolling_features`) use `.shift(1)` before
  `.rolling().mean()`: a game's trailing window is built only from that
  team's strictly-earlier-dated REG games. A game never contributes to its
  own feature row.
- `market_home_probability` (`normalized_home_probability`) is the closing
  moneyline line nflverse publishes after ingestion, devigged. This is a
  **closing**-line study — nothing here verifies these odds were actually
  observable at any specific pre-kickoff moment, only that they describe
  the market's final state for a completed game. Do not describe backtest
  results as "what a bettor could have gotten."
- `src/data/ingest.py`'s docstring is the authoritative pregame-safe /
  post-game-only column split for anything not covered above; consult it
  before adding a new feature column.

## Model variants (Stage 2-4)

- `always_home` (Stage 0): fitting-free, accuracy only, never scored as a
  probability.
- `market` (Stage 0): the devigged closing line itself, scored as a
  probability baseline, paired-quote rows only.
- `home_frequency` (Stage 2): a single constant probability fit from
  training-fold non-tied outcomes only, held fixed through validation/test.
- `team_only` / `team_market` (Stage 3, `select_features()` variants):
  L2 logistic regression over `TEAM_FEATURE_COLUMNS` alone, or
  `TEAM_FEATURE_COLUMNS + market_home_probability`. `team_market` fits and
  predicts on paired-quote rows only — never an imputed market value.
- `team_spread` (**protocol v2**, added after v1's backtest): the same
  estimator over `TEAM_FEATURE_COLUMNS + spread_line`. Fits and predicts on
  rows with a finite closing spread — never an imputed one.
- `market_only` (**protocol v3**): the same estimator over
  `market_home_probability` alone. A **control**, not a candidate model —
  see below. Paired-quote rows only, exactly like `team_market`.
- `team_residual` (**protocol v4**): `team_market`'s exact columns and rows,
  but the market enters as a fixed `logit` offset rather than a fitted
  feature, and the team features model only the deviation from it — see below.
- `team_residual_plus` (**protocol v5**): `team_residual` plus the
  starting-QB and rest features (`QB_REST_FEATURE_COLUMNS`) — see below.

### Protocol v2 — why `team_spread` was added

v1 had no market-informed model that could be scored on the full sample.
`team_market` needs *both* moneylines, which do not exist before 2006 and are
missing for 390 later games, so §6's rule against comparing a full sample
against a restricted one confines it to 4316 games across 2008–2024. The
closing spread is present for **every** REG game from 1999 through 2024
(measured: 5183/5183 in-window, 1512/1512 for 1999–2004), so `team_spread`
fits every outer season and abstains on nothing.

**The spread is used raw, not converted to a win probability.** A conversion
needs an assumed margin distribution — conventionally normal with a standard
deviation near 13.5 — and that number is itself estimated from NFL history,
including the seasons a fold is meant to be scored on. Handing the raw spread
to the estimator makes the mapping a coefficient fitted per fold on training
rows only. The moneyline devig stays a probability because it is arithmetic
over two quotes with no fitted parameter; the spread conversion is not.

Unchanged by v2: the target, the feature window and rolling rules, the C grid,
the seed, the fold rule, the metrics, the uncertainty rules, and the holdout
position. v2 adds one variant and changes nothing else. Sign convention:
positive `spread_line` means the home team is favoured (measured 2005–2024,
home win rate 0.6765 when positive against 0.3430 when negative).

Still excluded, and still pending the §9 provenance gate: `total_line`,
`div_game`, weather, rest days, starters/injuries, team identity. `team_spread`
cleared that bar because the spread was already documented pre-game-safe in
`src/data/ingest.py` and is 100% covered; the others have not been assessed.
The closing-line caveat applies to it exactly as to the moneyline — this is a
closing-line study, not a claim about quotes receivable before kickoff.

### Protocol v3 — the fitted market-only control

model-plan.md §6: "A gain from `team_market` alone does not prove team
statistics add value beyond market recalibration; a later market-only fitted
control is required for that stronger claim." v3 adds that control.

`market_only` differs from `team_market` **by the team features and nothing
else**: the same paired-quote rows (one `paired_quote_mask`), the same market
column, the same preprocessing (`StandardScaler` on the raw devigged
probability, never imputed), the same estimator, the same C grid selected the
same way, the same abstention reason. The pipeline derives its team blocks
from the variant's allowlist, so the control simply has none. That sameness is
the whole design: `team_market − market_only`, paired on identical games, is
then attributable to the team statistics alone.

What the comparisons license, and what they do not:

- `team_market − market_only` < 0, interval excluding 0: the team statistics
  add information beyond a fitted recalibration of the market, *in this
  representation*. ≥ 0: no evidence they do.
- `market_only − market`: what fitting a recalibration costs or gains against
  the raw devigged quote. Fitting is not free — it estimates an intercept and
  slope per fold from training seasons whose market may differ from the test
  season's.
- `market_only` is deliberately **not** the best available recalibration. It
  is linear in the probability, matching `team_market`'s representation so the
  control isolates the team features. A logit-scale recalibration might track
  the raw quote more closely, but it would change the representation and stop
  being `team_market`'s control. Beating `market_only` is therefore not the
  same claim as beating the market.

No new input: `market_home_probability` was already allowlisted for
`team_market`, so v3 adds no provenance-gate question. The closing-line caveat
applies unchanged. Unchanged by v3: the target, features, windows, C grid,
seed, fold rule, metrics, uncertainty rules, and holdout position. The report
additionally differences every fitted matched model against `market`, and
`team_market` against `market_only`, on the same season blocks.

### Protocol v4 — the residual model

`p = sigmoid(logit(p_market) + b + w·x)`: the devigged closing probability is a
fixed offset with no coefficient, and L2 logistic regression over the team
features (same preprocessing as every other variant) models only the
deviation. **The intercept is penalized with the weights**, unlike
scikit-learn's default, so as `C → 0` every coefficient shrinks to zero and the
model returns the market's probability exactly. A deviation from the market
has to earn its way through the training data. Otherwise the objective matches
scikit-learn's (`0.5‖θ‖² + C·Σ log loss`), so C means the same thing — verified
against scikit-learn to ~5e-6 in `tests/test_train.py`.

scikit-learn has no offset parameter, so `train.OffsetLogisticRegression` fits
it with scipy's L-BFGS-B; `scipy` was already installed, so no dependency was
added. The offset is the pipeline's last column by construction.

Same columns and rows as `team_market` (one `paired_quote_mask`), so
`team_residual − team_market` isolates how the market enters, and
`team_residual − market` asks the question v4 exists for: starting from the
market, do the team statistics improve on it? No new input, no new leakage
surface, closing-line caveat unchanged. Unchanged by v4: target, features,
windows, seed, fold rule, metrics, uncertainty rules, holdout — and the C grid
for every variant except this one (see the amendment below).

**v4 amendment — extended C grid for `team_residual` only.** Under the frozen
grid, validation selected `C = 0.01`, the floor, in all 17 outer folds: with 76
features and thousands of training rows, `C = 0.01` still lets the team
features move the prediction, so the market-equivalent end of the path was not
on the grid. `team_residual` now searches `{1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10}`
(`backtest.VARIANT_C_GRID`, recorded in the manifest as `c_grid_overrides`);
every other variant keeps the frozen grid. Declared as motivated by that
boundary result, which is a validation-side fact, with the outer-test numbers
also already seen by then. Selection under the extended grid is interior in
most folds (`1e-3` ×9, `1e-5` ×5, `1e-4` ×1, `0.01` ×2), so the new floor no
longer pins it.

### Protocol v5 — `team_residual_plus`, written before its first run

Step 5 of `game-model-improvement-plan.md`. This section was written before the
variant was ever fitted or scored; its result is recorded afterwards, below the
line marked "Result".

**Definition.** `team_residual` with the five `QB_REST_FEATURE_COLUMNS` added to
its allowlist: per side, `qb_changed` and `qb_prior_starts`, plus `rest_diff`.
The new columns go through the same preprocessing as the team features —
training-fold median imputation, a missingness indicator per column, and
standardization — and the same penalized coefficients. Everything else is
`team_residual` unchanged: the same paired-quote rows, the same fixed
`logit(p_market)` offset with no coefficient, the same penalized intercept,
the same extended C grid `{1e-5, ..., 10}`.

**Question and decision rule, fixed now.** Do the starting-QB and rest
features add anything the closing line does not already price? Primary
comparison: `team_residual_plus − team_residual`, paired on identical matched
games, 2008-2024, with the usual 2,000-replicate season-block bootstrap.
"Adds information beyond the market" requires a negative mean difference
with a 95% interval entirely below zero. Secondary:
`team_residual_plus − market`. Either way this is development data: no result
here is a holdout claim (see "2027 holdout").

**Leakage check.**

- *New inputs:* built by `qb_rest_features()` (Step 4) from inputs that passed
  the provenance audit. History is admitted from completed REG games on
  strictly earlier dates only; corruption-injection tests prove a game cannot
  see its own outcome, a later game or a same-day game, and a mutation check
  confirmed those tests catch an injected leak. The current game contributes
  only its recorded starter's identity and the scheduled rest.
- *Offset:* the devigged closing moneyline `team_residual` already uses. The
  closing-line caveat applies unchanged.
- *Fitting:* every learned parameter — medians, scales, coefficients — is
  fitted per fold on training rows (1999 through T-1), C is selected on T-1,
  and outer seasons are 2005-2024 only. No 2025, 2026 or 2027 row enters.
- *Disclosed prior look:* while checking Step 4's features on real 2005-2024
  data, I saw the raw association — home teams won 44.6% after a home starter
  change against 57.4% without one. That is development data and inside the
  protocol, but it was seen before this rule was written, so it is recorded here.

**Result** (recorded after the run; artifacts `20261002T024745Z_7a4f50cc`).
Matched coverage, 2008-2024, n=4316, 2,000 season-block replicates:

| comparison | mean log-loss difference | 95% interval |
|---|---|---|
| `team_residual_plus − team_residual` (primary) | +0.000646 | [−0.000173, +0.001635] |
| `team_residual_plus − market` | +0.001054 | [−0.000238, +0.002662] |

By the rule fixed above, **the starting-QB and rest features do not add
information beyond the market**: the primary difference is positive, not
negative, and its interval does not lie below zero. The raw association that
was visible beforehand — home teams winning 44.6% after a home starter change
against 57.4% without one — is real, but the closing line already prices it.
Validation leaned on heavy shrinkage toward the market (`C = 1e-5` ×6, `1e-4`
×4, `1e-3` ×4, `0.01` ×3), and every fit converged. Existing variants were
unchanged by the v5 code (`team_residual` 0.607388, as before).

## `C` grid

`C ∈ {0.01, 0.1, 1, 10}` (already fixed in `next-implementations.md` Stage
4), selected independently for each fitted variant — `team_only`,
`team_market`, `team_spread`, `market_only`, `team_residual`, `team_residual_plus` — by validation log loss, ties broken toward the
smaller `C`. Not a hyperparameter search
space to widen later without a new protocol version.

## Scoring metrics

Log loss is primary (`src/evaluation/metrics.py`, per `docs/model-plan.md` §7).
Brier score, accuracy (threshold 0.5), ROC AUC, and 10-bin calibration are
supporting diagnostics, never the selection criterion. `eps=1e-15` clipping
for log loss, as already defaulted in `metrics.py`.

## Uncertainty rules

Already agreed in `docs/model-plan.md` §7 (and restated in
`next-implementations.md` Stage 5). Recorded here, not decided here:

- Report **paired** model-minus-baseline differences on identical eligible
  game IDs — not independent per-model intervals.
- 2,000 fixed-seed season-block bootstrap replicates, 95% percentile
  intervals. Resample whole seasons with replacement and reuse the same
  sampled blocks for every compared model. Keep all games within a sampled
  block together.
- Games are **not** independent draws. Within-season correlation (shared
  rosters, schedule, era effects) is why blocks are seasons rather than
  games.
- A final single-season evaluation uses week blocks instead, and must state
  that between-season uncertainty is omitted.
- Report per-season, pooled game-weighted, and equal-season means
  separately. These are approximate intervals: they do not remove selection
  bias, distribution shift, or the cost of repeated inspection.
- Seeded (see below) so every report reproduces bit-for-bit from its saved
  predictions.

## Seeds

`RANDOM_SEED = 20260915` (this protocol's freeze date) for every
stochastic step: `sklearn` estimators' `random_state`, and the bootstrap
resampling above. One global seed, not per-fold or per-model — simplicity
over a false sense of extra independence between them.

## Holdout designation

Per Stage 1's mandate to resolve what `next-implementations.md` could not:

- **2025 is already-used development data**, not a clean holdout — 2025
  results were inspected before this protocol was frozen (see
  `next-implementations.md` "Review decisions"). This designation does
  **not** widen the agreed experiment window: Stage 0 and the outer
  development evaluation stay at **2005-2024**, and Stage 0 keeps excluding
  2025/2026 unconditionally. Calling a season "development data" records
  that it was exposed; it does not promote it into the evaluation set.
- **As of this protocol's freeze date (2026-09-15), no genuinely
  untouched future evaluation block exists.** The 2026 season is in
  progress (partial), so it is neither a complete season nor untouched —
  its early weeks are already visible in the same nflverse feed used for
  development. Per Stage 1's deliverable, this ships explicitly **without
  a final holdout result**, rather than retroactively certifying partial
  2026 as one.
- Partial 2026 stays out of the Stage 4 backtest and every development
  metric in this release. A season finishing does not undo prior exposure,
  so 2026 completing is not by itself enough to make it a holdout. Before
  any final evaluation, predeclare an untouched future block and document
  its exposure history. Until then Stage 6 stays pending, and Stages 0-5
  proceed on **2005-2024** — a development-only release must not imply a
  final validated number exists.

### 2027 holdout (predeclared 2026-10-01)

Step 1 of `game-model-improvement-plan.md`. Written before any further
experiment runs, so that a later gain can be claimed against data nothing in
development has seen.

**The block.** The **2027 regular season** (`game_type == "REG"`) is the
untouched final-evaluation block. Preseason and postseason are out, as
everywhere else in this protocol.

**Exposure history: none.** Verified 2026-10-01: neither the live
`data/nfl_games.db` nor snapshot `95e0254680c08783` holds a single 2027 row
(`games` and `plays` both end at 2026). The season has not been played. What
counts as exposure is the same rule that made 2025 development data: *any*
inspection of 2027 outcomes, or of any model's 2027 predictions next to
outcomes, before the freeze below. A descriptive look counts.

**The freeze, before the first 2027 REG kickoff.** Record in this document:

- **Exactly one primary model** — its variant, ordered feature allowlist, C
  grid, and training window — plus the commit that will score it. Only the
  primary model's result can support a "beats the closing line" claim. Any
  other model scored on 2027 is secondary and descriptive; picking the best of
  several after seeing them would turn the holdout back into development data.
- The snapshot that will be scored (see "Data handling").

**Fit rule for T = 2027** — the existing fold rule, unchanged: fit candidates
on 1999-2025, select C by validation log loss on 2026, refit every learned step
on 1999-2026, then hold parameters fixed through 2027. Completed 2027 games may
update later games' feature history (sequential pregame forecasting); they
never refit anything.

**The success rule.** The primary model **beats the closing line** if and only
if both hold:

1. its paired log-loss difference against `market` (model minus market) is
   negative, and
2. the 95% percentile interval of that difference, from **2,000 week-block
   bootstrap replicates** (seed `20260915`; the season's REG weeks resampled
   with replacement; the same sampled blocks for both models), lies entirely
   below zero.

The comparison population is the matched one: 2027 REG games with a paired
quote (`paired_quote_mask`), a non-null `home_win`, and a model prediction,
on identical game IDs for both. Any other outcome, a tie included, reads
"does not beat the closing line." The report must state that week blocks omit
between-season uncertainty. Brier score, accuracy, ROC AUC and calibration are
reported alongside; they cannot change the verdict.

**One shot.** No re-scoring 2027 with another model, feature set, C or
threshold after seeing the result. Any change after that point is a new
development experiment and needs the next untouched season predeclared the
same way. If no primary model is declared by the deadline, 2027 is not used as
a holdout for anything, and the next season must be predeclared instead.

**Data handling.** No development snapshot may contain completed 2027 games.
2027 data enters a new snapshot only after the freeze and after the season
completes, and that snapshot is the one scored. The closing-line caveat
applies unchanged: this remains a closing-line study.

**Prerequisites before scoring — not done by this step**, which changes no
code:

- `report.py` implements season-block bootstrap only. The week-block bootstrap
  this rule specifies has to be implemented and tested before 2027 is scored.
- `diagnose.EXCLUDED_SEASONS` is `(2025, 2026)`. Nothing in code yet stops a
  development run from reading 2027 once it is ingested; add 2027 to that
  exclusion, or an equivalent guard, before any 2027 data is captured.

Stage 6's final evaluation now has a designated block, but stays pending until
2027 completes and the prerequisites above are met.

## Input provenance audit (2026-10-01)

Step 3 of `game-model-improvement-plan.md`, applying model-plan.md §9 gate 9 to
the three candidate inputs. Each verdict rests on nflverse's own definition plus
a check against data, never on this repository's docstrings: `ingest.py`
asserting a column is pregame-safe is not evidence (A3). Upstream definitions
are quoted from nflreadr's data dictionaries
(`nflverse/nflreadr/data-raw/dictionary_schedules.csv`, and the injuries
dictionary at `nflreadr.nflverse.com/articles/dictionary_injuries.html`).
`nflverse/nfldata`'s `DATASETS.md` does not describe the rest, QB or moneyline
fields at all. All checks were read-only; play-by-play and injury data were
fetched into memory with `nfl_data_py` and not stored.

### Rest days — `home_rest` / `away_rest`: **GO**

- **Upstream definition:** "Days of rest that the home team is coming off of."
  It does not say how the value is computed.
- **Check:** recomputed as days since each team's previous game of any type,
  1999-2024. It matches in 99.73% of 12,561 in-season REG team-games. The 34
  mismatches cluster on reschedules and cancellations (for example the 2022
  Week 18 BUF and CIN rows, after their Week 17 game was cancelled). Season
  openers carry a fixed 7 (824 of 829, the rest 14), a convention rather than
  the real offseason gap. Coverage is 100% of REG games back to 1999.
- **Timing:** schedule arithmetic. The schedule is published months ahead, and
  a reschedule is known before the game it moves.
- **Evidence mode:** derived from the published schedule, verified by
  recomputation. Feature constraint: treat the opener value as a convention,
  not a measured offseason length.

### Starting QB — `home_qb_id` / `away_qb_id`: **GO, with a stated limit**

- **Upstream definition:** "GSIS Player ID for home team starting quarterback."
  It does not say whether this is the announced starter or the recorded one.
- **Check:** against nflverse play-by-play for 2010 and 2023 (1,008 REG
  team-games), `qb_id` equals the team's first passer in 99.90% of games and the
  passer with the most attempts in 96.23%. In the 39 games where the QB changed
  mid-game, it followed the **first** passer 38 times and the most-attempts
  passer once. It records who started, not how the game went: an in-game injury
  exit does not change it.
- **Timing:** the starter is fixed by the first snap, inside the
  immediately-before-kickoff cutoff (§ "Cutoff convention").
- **Evidence mode:** historical snapshot of the *recorded* starter, verified
  against play-by-play in two seasons, not all 26. The limit: this is not the
  *announced* starter. A late scratch between the announcement and kickoff
  shows up as the replacement — still known before kickoff, so it is within the
  cutoff, but a live system would need the announced starter and the inactive
  list, which this field cannot supply. Feature constraint: use the current
  game's starter identity and earlier games' history only, never anything from
  inside the current game.

### Injury reports — nflverse `injuries`: **GO on provenance; not ready for Step 4**

- **Upstream definitions:** `report_status` — "Player's status for game on
  official injury report"; `date_modified` — "Date and time that injury
  information was updated"; `week` — "Week that injury occurred".
- **Check:** in 2010, 2016 and 2023 (14,703 REG rows), `date_modified` falls
  before the game's kickoff in 100% of rows, with a median lead of 47-55 hours,
  consistent with the final Friday report. No row was updated after kickoff.
- **Timing:** published before the game, verified by timestamp.
- **Evidence mode:** official pregame report with an update timestamp, verified
  against kickoff times in three seasons.
- **Why not for Step 4:** (1) it is not ingested into `nfl_games.db`, and
  ingestion is out of scope here; (2) coverage starts in 2009 — 2007 and 2008
  are unavailable — so 1999-2008 training history and the 2008 outer season
  would have none; (3) the status taxonomy changed in 2016, when "Probable" was
  dropped, so a status feature needs a definitional decision across that break;
  (4) 34% of rows carry no game status; and (5) a game-level feature needs a
  player-importance mapping (whose absence matters) that does not exist yet.
  These need their own ingestion and design step before any feature work.

### Outcome for the plan

Step 4 may build features from **rest days** and the **starting QB** only.
Injury features stay out of Step 4, as its own out-of-scope line already
requires unless they are ingested, until a separate ingestion-and-design step
resolves points (1)-(5) above.

## Opening-line source evaluation (2026-10-01)

Step 6 of `game-model-improvement-plan.md`. nflverse carries closing lines only.
An opening line is necessarily earlier than the close, so timing cannot leak an
outcome; the questions are coverage, what "opening" actually means, licensing
and cost. The v4/v5 residual models take a moneyline probability as their
offset, so **opening moneylines** are the requirement — opening spreads alone
would need a conversion this protocol avoids (see Protocol v2).

Evidence comes from each provider's public pages, fetched read-only on
2026-10-01. Several refused automated access (noted below); nothing was
downloaded or purchased.

| source | opening ML? | coverage | what "opening" means | licensing | cost | verdict |
|---|---|---|---|---|---|---|
| **Odds Warehouse** | yes — opening and closing moneyline, spread, total | 2009-2026 | not stated; sportsbook not stated | not stated on the product page | $99 one-time | **conditional** — the only candidate with opening moneylines over a usable span |
| **The Odds API** | earliest snapshot only | from 2020-06-06 | timestamped snapshots, every 10 min, every 5 min from Sep 2022 | not stated on the docs page | paid plans only (about $30-$249/month listed) | **no-go historically**; right tool for capturing lines going forward |
| **AusSportsBetting** | not verified | 2006 onward | averaged across a sample of bookmakers, so no single opener | personal use only, no redistribution, "may contain errors" | free | **no-go** as primary source |
| **SportsbookReviewsOnline** | no — opening spreads and totals, single moneyline | archive, no longer updated | not stated | not stated | free | **no-go**; the NFL archive page now returns 404 |

Notes per source:

- **Odds Warehouse.** Princeton DSS's "Historical Sports Odds Database" was
  bought from it, so it is not an independent second source. Its product page
  does not say which book the lines come from or when the opener is taken, and
  its licence is behind a terms page that was not reviewed.
- **The Odds API.** Strongest timestamp semantics of any candidate: every
  snapshot carries its capture time, so "opening" could be defined as the
  earliest snapshot and any pre-kickoff cutoff chosen exactly. But the earliest
  snapshot is not the market's true open, and coverage from mid-2020 leaves
  about five development seasons (2020-2024) — too few for the fold rule, whose
  training folds would also need opening offsets. Going forward it is the
  natural way to capture timestamped lines for the 2027 holdout.
- **AusSportsBetting.** Its page refused automated access, so the opening
  columns could not be confirmed; a closing-odds column is confirmed only via
  third-party code. Its odds are averaged across bookmakers, which has no single
  opening time.
- **Not evaluated in detail:** Sports Insights (opening and closing spreads and
  totals back to 2003, paid), TheFootballLines (from 2007; its page refused
  automated access), Scottfree Analytics (paid). None of their public material
  established opening moneylines with documented timing.

### Verdict

**No free source is usable**: none offers opening moneylines with documented
semantics under terms that fit this project. **Odds Warehouse is the one
conditional candidate**, and the decision to buy it ($99) belongs to the
project owner. If bought, these checks gate Step 7, and any failure makes it
no-go:

1. Its documentation states which sportsbook the lines come from and when the
   opening line is taken.
2. Its licence permits private research use.
3. Its closing moneylines agree with nflverse's devigged closing lines on the
   overlapping games, a consistency check on the file before it is trusted.
4. The file lives only under the gitignored `data/` and is never committed,
   even though this repository is private.

Until then Step 7 is blocked.

### Decision: Step 7 is no-go (2026-10-01)

The project owner chose free sources only and did not buy Odds Warehouse. Both
free routes then failed in practice:

- **AusSportsBetting:** its data pages sit behind a Cloudflare bot check, which
  blocked automated access (not circumvented), and the owner's own browser
  download also failed. No file was obtained.
- **SportsbookReviewsOnline:** the whole site now returns 404, root included.
  The Internet Archive was temporarily offline when checked, so no archived
  copy was retrieved. It never carried opening moneylines anyway.
- **Mirrors of AusSportsBetting data** (for example on Kaggle) were not used:
  AusSportsBetting's terms ask that its data not be made available elsewhere.

**Step 7 is no-go.** No opening-line study runs, and the question "does the
model beat the opening line?" stays untested rather than answered. The routes
back, should that change: buy Odds Warehouse and apply the four checks above;
or capture timestamped lines going forward (for example from an odds API's live
feed during the season), which gives no history to backtest but is the natural
data for the 2027 holdout.
