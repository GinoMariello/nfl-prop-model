# Game model improvement plan: information the closing line lacks

Status: closed, 2026-10-01. Steps 1-6 done; Step 7 no-go. Follows protocol v4
(`experiment-protocol.md`). Outcome in one line: the game model ties the closing
line, the inputs available add nothing to it, and the opening-line route is
untested for want of a usable free source.

## Where the model stands

On development data (matched coverage, 2008-2024, n=4316), every fitted model
loses to or ties the devigged closing moneyline. The v3 control showed the 76
rolling team statistics add nothing beyond a fitted market. The v4 residual
model, which starts from the market's log-odds and lets team features adjust
it, recovers the market: +0.000408 log loss [-0.001303, +0.002202]. More
box-score features in the same model will not change that, because the
closing line already prices them.

The only route past a tie is information the closing line does not have at the
moment the model is scored, or a line that has not yet absorbed it. This plan
pursues both, and changes how success is measured so a gain can eventually be
claimed honestly.

## Ground rules for every step

- **Measure "beyond the market", not "better than team_only".** A new feature
  earns its place only if `team_residual` with it beats `team_residual`
  without it, paired on identical matched games against `market`.
- **Pregame availability is a gate, not an assumption.** model-plan.md §9
  gate 9 applies: a local docstring saying a column is pregame-safe is not
  evidence (A3). Each input needs documented source semantics and a stated
  evidence mode before it enters an allowlist.
- **Every new variant is a protocol version**, documented the way v2-v4 were,
  with the leakage check written down before the backtest runs.
- **No final claim without an untouched holdout.** None exists today
  (`experiment-protocol.md`, "Holdout designation").

## Step 1 — Predeclare an untouched holdout and the success rule

Write into `experiment-protocol.md`, before any further experiment runs: the
2027 regular season is the untouched final-evaluation block; nothing from it
may be inspected until the protocol for the final model is frozen. Define the
success rule in advance: a model "beats the closing line" only if its paired
log-loss difference against `market` on the holdout is negative with a 95%
week-block bootstrap interval excluding zero.

Acceptance: the protocol names 2027 as the holdout and records its exposure
history (none); the success rule is stated with its metric, interval method and
comparison population; no code changes.

Out of scope: any model or feature change; scoring any 2025 or 2026 data.

## Step 2 — Make the report read its model lists from the run manifest

`report.py` takes `FULL_COVERAGE_MODELS`/`MATCHED_MODELS` from the current code,
so it cannot regenerate a report for any run saved before the latest protocol
version (v2 broke pre-`team_spread` runs, v3 and v4 did it again). Read the
declared lists from each run's `manifest.json` instead, falling back to the
code constants only for manifests that predate those keys.

Acceptance: a report regenerates for a run whose manifest declares fewer models
than the current code; a model declared in the manifest but missing from
`predictions.csv` raises with a clear message; existing report tests pass.

Out of scope: changing what any report table computes.

## Step 3 — Provenance audit for starting QB, rest days and injury reports

Candidate inputs, all with 100% REG coverage back to 1999 in `games`:
`home_qb_id`/`away_qb_id` and `home_rest`/`away_rest`. Injury reports are not
yet ingested for this pipeline. For each, document: what the field means
upstream (nflverse's own definition, not this repo's docstring), whether the
historical value is the announced pregame starter or the player recorded as
having started, when it would have been knowable relative to kickoff, and the
evidence mode. Return a go/no-go per input.

Acceptance: a section in `experiment-protocol.md` per input with source
definition, timing and evidence mode, and an explicit go/no-go; any input
without upstream documentation is no-go.

Out of scope: building features; ingesting injury data.

## Step 4 — Leak-safe features for the inputs that passed the gate

For each go input from Step 3, add features in `rolling_features.py` behind a new
allowlist. Starting QB: a "starter differs from the team's previous completed
game" flag per side, plus the starter's own prior-games count, both computed
strictly from earlier completed games (shift before roll, same rule as the
existing rolling stats). Rest: the raw rest-day difference. No feature may read
the current game's outcome or any later game.

Acceptance: a corruption-injection test proves a game's feature row cannot see
its own or a later game's data (same pattern as `tests/test_rolling_features.py`);
the allowlist test proves no target, score or ID enters X; existing feature
tests pass unchanged.

Out of scope: injury features unless Step 3 marked them go and they are ingested.

## Step 5 — Add a residual variant with the new features and run the backtest

Add `team_residual_plus` (protocol v5): `team_residual` with the Step 4 features
added to its allowlist, same offset, same extended C grid. Run the Stage 4
backtest and the Stage 5 report on development data. Report
`team_residual_plus - team_residual` and `team_residual_plus - market`, paired on
identical matched games.

Acceptance: the protocol v5 section records the leakage check before the run;
both paired differences are reported with season-block bootstrap intervals; the
existing variants' pooled log losses are unchanged (team_residual 0.607388).

Out of scope: tuning anything on 2025, 2026 or 2027 data; changing the C grid
for other variants.

**Status (2026-10-01): done, negative result.** `team_residual_plus −
team_residual` = +0.000646 log loss [−0.000173, +0.001635] on 4316 matched
games: the starting-QB and rest features add nothing beyond the closing line,
which already prices them (`experiment-protocol.md`, Protocol v5, "Result").
This strengthens the case for Step 6: of the routes past a tie, information the
line lacks at close has now come up empty for the inputs available, leaving a
line that has not yet absorbed it.

## Step 6 — Opening-line data: find a source or rule it out

Closing lines are the efficient endpoint; an edge is far more plausible against
the opening line. nflverse carries closing lines only. Identify candidate
sources of historical opening moneylines/spreads, and for each record coverage
by season, timestamp semantics (when the opening quote was posted), licensing
for this use, and cost. Return a go/no-go.

Acceptance: a written comparison of at least two candidate sources, or a
documented finding that none is usable; a go/no-go with the reason.

Out of scope: ingesting or scoring any opening-line data.

**Status (2026-10-01): no free source; one conditional candidate.** No free
source offers opening moneylines with documented semantics under usable terms.
Odds Warehouse ($99, 2009-2026, opening and closing moneylines) is the only
candidate; whether to buy it is the owner's call, and four checks gate Step 7
if it is bought (`experiment-protocol.md`, "Opening-line source evaluation").

## Step 7 — If Step 6 is go: an opening-line study

**Status (2026-10-01): no-go.** The owner chose free sources only. AusSportsBetting
is behind a Cloudflare bot check and the owner's own download failed;
SportsbookReviewsOnline is offline entirely and never had opening moneylines.
Not run. Routes back are recorded in `experiment-protocol.md`, "Decision: Step 7
is no-go".

Ingest opening lines into a new raw table without touching `games`, then add an
`opening_market` baseline and a residual variant offset by the opening line, and
measure the model against the opening line as well as the closing one. A model
that beats the opening line but not the closing line is the result a bettor
would care about, and it should be reported that way, with no profitability
claim.

Acceptance: opening lines live in their own raw table with source and
retrieval time recorded; the report shows paired differences against both the
opening and closing lines on identical games.

Out of scope: staking, bankroll or profitability claims; any live use.
