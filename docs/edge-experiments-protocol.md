# Edge experiments protocol (frozen 2026-10-27, before any experiment was run)

After `docs/model-alternatives-protocol.md`: architectures are exhausted (LSTM, GBM, TabPFN make
the same errors). These test new information and betting-side changes instead. Each runs once;
changing anything below is a new protocol version, said explicitly. Nothing here changes the paper
rule or production code; a pass means "propose it for approval", not "ship it".

## A. Teammate-out feature (`experiments/edge_exp.py teammate`)

GBM (GbmProjections features/settings) for WR and TE `receiving_yards` and `receptions`, plus
`vacated_targets`: the summed ewm3 target share of same-team WR/TE/RB teammates listed Out or
Doubtful on that week's official injury report (published before kickoff; teammates' shares from
games strictly before). Same 15 walk-forward weeks (2025 w10-18, 2026 w2-7), every WR/TE who
played.

Pass (per position, `receiving_yards` MAE): below the GBM without the feature pooled, not worse in
either season, and `receptions` MAE not worse pooled. Also reported: the same split on player-weeks
with vacated_targets > 0.05 (where it should matter).

## B. Beat-the-line model (`experiments/edge_exp.py line`)

Logistic regression (sklearn, C=1) predicting "our projection's side wins" on every logged line
(weeks 5-7, rebuilt in-harness from the projection files), features: z, gap, gbm agrees, sleeper
agrees, stat one-hot, log line. Walk-forward: train week 5 -> test 6; train 5-6 -> test 7.

Pass: on the two test weeks pooled, log loss below the constant (training win rate) baseline,
and the top third by predicted probability wins more than the edge score's High tier on the same
rows.

## C. Market selection and stake size (`experiments/edge_exp.py stakes`)

Replay of settled main-rule bets (weeks 6-7, hit55f + rec55, Smith void): (i) flat as placed;
(ii) skip rushing; (iii) quarter-Kelly from the rule's own chance. Report only (two weeks is not a
test), no pass/fail.

## D. Closing-line value (`experiments/edge_exp.py clv`)

Win rate and ROI of settled bets by line move after the bet (toward us / flat / against). Report
only.

## E. Ladder refit

Not run here: refitting LADDER_W/LADDER_K changes the paper rule, which needs approval first.

## Round 2 (frozen 2026-10-27, after A-D ran, before F-H ran; every option was requested)

E is replaced by H (a test only; the rule itself still changes only after approval).

F. Volume x efficiency (`edge_exp.py volume`): for WR/TE receiving yards and RB rushing yards,
   GBM predicts the volume (targets / carries, median loss); efficiency = the player's yards per
   target (carry) over his last 16 games, shrunk toward the position mean with pseudo-count k
   (picked on training rows only); projection = volume x efficiency. Same 15 weeks.
   Pass (per stat): MAE below the direct GBM pooled and not worse in either season.
G. Weather and backup QB (`edge_exp.py context`): GBM + `temp`, `wind`, `roof` (schedule_context)
   for WR/TE receiving yards and QB passing yards; WR/TE also get `qb_changed` (this week's listed
   starting QB differs from the team's starter in his previous game). Same 15 weeks.
   Pass (per stat): MAE below the GBM without them pooled and not worse in either season.
H. Ladder refit (`edge_exp.py ladder`): every logged yardage line weeks 5-7. Grid LADDER_W in
   0..1 step 0.1, LADDER_K in 0.6..1.0 step 0.05, fit by log loss on weeks 5-6, tested on week 7
   against the current (0.3, 0.85). Pass (= propose for approval): lower week-7 log loss AND bets at
   >= 55% chance win more than the current setting's bets on week 7.

## Results (run 2026-10-27)

- A teammate-out: WR receiving MAE 19.43 -> 19.37 (PASS by the letter, 0.06 yd), TE 14.39 -> 14.35 (FAIL, 2025 +0.02);
  on vacated > 0.05 rows WR 19.75 -> 19.72, TE 14.87 -> 14.96. No material effect.
- B beat-the-line: log loss 0.6801 vs constant 0.6979, top third 59.7% vs edge-score High 62.7% -> FAIL.
- C stakes (weeks 6-7, 248 settled): flat +9.3% ROI, skip rushing +13.2%, quarter-Kelly +13.3% (week 6 -9.4%, week 7 +23.4%).
- D CLV (week 6 only, 87 bets): beat the close 65.2% / +19.2% ROI, flat 45.9% / -14.7%, close moved past us 48.1% / -9.5%.
- F volume x efficiency: WR, TE, RB all worse (FAIL).
- G context: TE 14.39 -> 14.34 (PASS, 0.05 yd), WR FAIL, QB passing 61.56 -> 60.59 but 2026 +0.04 (FAIL; weather here is
  observed game weather, so live it could only be weaker).
- H ladder refit: best W=0.1 K=0.6 on weeks 5-6; week 7 log loss 0.6846 vs current 0.6806 -> FAIL (keep 0.3 / 0.85).

## Round 3: combinations (frozen 2026-10-27, before running; user asked whether combos help)

I. Model combo (`edge_exp.py combo`): every feature that passed or came close, together --
   QB passing: GBM + shrink + temp/wind/dome; WR/TE receiving: GBM + shrink + vacated_targets +
   temp/wind/dome + qb_changed. Same 15 weeks. Pass (per stat): >= 0.5 yd below the plain GBM pooled
   and not worse in either season (the same bar as the model-alternatives point candidates).
J. Bet combos (same command, settled main-rule bets weeks 6-7, report only -- in-sample: the edge
   score and skip-rushing were both chosen on these weeks): skip rushing x edge tier floor
   (all / Low+ / Medium+ / High), and skip rushing + quarter-Kelly.

## Round 3 results (run 2026-10-27)

- I model combo: QB passing 61.56 -> 60.13 (2026 +0.12, FAIL), WR 19.43 -> 19.38 (FAIL), TE 14.39 -> 14.44 (FAIL).
- J bet combos (in-sample; first run had a numpy-bool scoring bug, fixed): weeks 6-7 ROI -- all +9.3%, skip rushing
  +13.2%, + Low or up (= also skip Avoid) +16.9% (week 6 +2.9%, week 7 +24.5% -- the only filter positive in both
  weeks), + Medium or up +10.9%, High only +17.5% (week 6 -15.1%, week 7 +36.9%), skip rushing + 1/4 Kelly +16.8%
  (week 6 -6.0%).

## Round 4: regression-to-mean unders (frozen 2026-10-29, before running)

Prompted by week 8 TNF (CAR @ GB): every model put Love, Watson, Golden below both the line and their own
recent form, and the unders lost. Question: are we betting the models' pull toward average rather than
information? Weeks 6-7 never informed this idea, so they are a fair test of it. Week 8 TNF is shown
for context only and is not part of the gate.

K. Form-gap filter (`edge_exp.py formgap`): settled main-rule bets weeks 6-7. For each bet, gap =
   projection / l5 (the player's last-5-game average of that stat, pre-game, from that week's
   projections CSV) - 1. Primary filter: drop unders with gap < -15%.
   Pass: kept bets' ROI beats all bets' ROI in week 6 AND in week 7, and the dropped unders lose money
   over weeks 6-7. Report only (no gate): thresholds -10/-20/-25%; the mirror rule (drop overs with
   gap > +15%); and both rules on top of skip rushing + Avoid.
   If it passes, it becomes a side test on the board from week 9, not a rule change.

## Round 4 results (run 2026-10-29)

- K form gap: FAIL. Weeks 6-7, 248 bets: all +9.3% ROI; keep (drop unders projected >15% below l5) +6.8%;
  the 81 dropped unders went 50-31, +14.6% (week 6 -0.8%, week 7 +22.7%) -- better than the bets kept.
  Every threshold (-10/-20/-25%) lowered ROI; mirror rule on overs +8.5%; on top of skip rushing + Avoid
  +16.9% -> +17.0% on 65 fewer bets (no gain). The pull toward average has been an edge, not a leak;
  TNF week 8 was one game where it lost. No side test added.

## Round 5: schedule-adjusted defense and offense style (frozen 2026-10-30, before running)

Prompted by week 8 TNF: CAR's pass D looked strong (6th fewest pass yards allowed) partly from the offenses it
had faced, and the GBM has no view of how pass-heavy or drive-sustaining an offense is. All features are per
team per game, then a trailing 8-game mean shifted by one game (never sees its own game), across seasons.

Features (`edge_exp.py style`):
- adj_d: schedule-adjusted defense. For each past game, PPR the defense allowed to the position divided by what
  that offense averaged to the position over its 8 games before that one; adj_d = trailing mean of that ratio.
- off_style: own offense's pass rate (pass / (pass + rush) plays, no kneels/spikes/special teams), plays per
  drive, drives per game -- from `plays` in data/nfl_games.db.
- def_style: opponent defense's pass rate faced and plays per drive allowed.
- all: the three together.

Each added to the GBM features separately and together. Same 15 walk-forward weeks (2025 w10-18, 2026 w2-7).
Stats: QB passing yards, WR/TE receiving yards, WR/TE receptions, RB rushing yards.
Pass (per stat and variant): pooled MAE >= 0.5 yd below the plain GBM (>= 0.05 for receptions) and not worse in
either season -- the round 3 bar. Report only: the 2026 weeks 2-7 slice (early season, where schedule strength
should matter most). A pass becomes a proposal, then a shadow run, not a direct model change.

## Round 5 results (run 2026-10-30)

- adj_d (schedule-adjusted defense): QB passing 61.56 -> 61.01 (2025 -0.58, 2026 -0.50) PASS by 0.05 yd over the bar;
  WR, TE receiving and receptions flat (FAIL); RB rushing worse (+0.13, FAIL).
- off_style (pass rate, plays/drive, drives): FAIL everywhere; QB passing better in 2025 but 2026 +0.67.
- def_style: FAIL; RB rushing 17.53 -> 17.43 (2026 w2-7 -0.19) is the best non-pass, under the 0.5 bar.
- all: FAIL everywhere (QB 2026 +0.94).
- Early-season slice (2026 w2-7) did not favour the schedule-adjusted defense outside QB passing.
Proposal: adj_d for QB passing yards only, shadow run next to the gbm_shrink shadow (weeks 9-10), no live change.

## Round 5 shadow run (frozen 2026-10-30, before weeks 9-10 are played)

adj_d runs alongside production for QB passing yards in 2026 weeks 9 and 10 (week 8 prompted the idea, so it is
left out). No production change. Once week 10 is in `player_weekly_stats`:
`.venv/bin/python experiments/edge_exp.py style 2026:9,2026:10 QB`.
Ships to `GbmProjections.py` (QB passing only) only if, on those two weeks pooled, adj_d's QB passing MAE is
below the plain GBM's. Read only the QB passing_yards block and only its "all" column; the per-season columns
and PASS/FAIL tag printed there are the round 5 bar, not this gate. If it ties or loses, it stays an experiment.

## Round 6 (frozen 2026-10-30, before any outcome below is looked at)

Bets: main rule (hit55f + rec55), 2026 weeks 6-7, settled (week 8 has one game, left out). Script:
`.venv/bin/python experiments/bet_review.py`. Paper rule never changes from this round; passes change only the
real-money advice (gameday-picks) and the board's stake sizing.

- M (CLV report, no gate): share of bets whose line moved toward us by the close (`clv_yards` > 0), share
  against, mean `clv_yards` by stat. Read only: beating the close more often than not is the faster sign of a
  real edge than the win/loss record.
- N (line moved against us): drop bets whose close moved against us by >= 2 yards (yardage) or >= 1 catch
  (receptions), `clv_yards` <= -2 / <= -1. Pass only if kept ROI > all-bets ROI in week 6 and in week 7, and the
  dropped bets lose money pooled. The script also reports the share of closing lines retrieved before kickoff;
  under 90% and the test is void (the close would not be knowable in time to skip a bet).
- O (confidence tiers): `site_data.bet_tiers`. Pass only if High (3) ROI > Low (1) ROI pooled and in each of
  week 6 and week 7. Fail means tiers carry no edge and must not raise stakes or real-money picks.
- P (line shopping, measurement, no gate): each bet re-priced at the best price any book offered on the same
  side and the same line in the last `prop_lines` snapshot at or before `placed_at`; ROI at the ledger price vs
  that best price.

## Round 6 results (run 2026-10-30), 248 settled bets, 146-102, +$231.53, ROI +9.3%

- M (CLV): beat the close 22%, flat 50%, against 27%, mean -0.29 yd. Receptions 91% flat (the ESPN close has
  lines, no prices, so receptions CLV says little). Yardage: receiving 30/26/45, rushing 34/36/30. ROI by close
  move: against -13.8% (68), flat +15.3% (125), toward +24.3% (55). The market does not move toward us more
  often than away, so CLV gives no sign of edge yet; the profit is from bets the market left alone or agreed with.
- N (line moved against): PASS. Week 6 all -4.1% / kept -2.4%; week 7 all +16.6% / kept +24.1%; dropped 34 bets
  13-21, -$92.71, ROI -27.3%. 100% of bets have a pregame close. Small sample. Paper rule unchanged; real-money
  advice now skips a bet whose DraftKings line has moved >= 2 yd (>= 1 catch) against it before kickoff.
- O (tiers): FAIL. Avoid -10.3% (65), Low +25.9% (66), Medium -4.9% (40), High +19.1% (77); week 6 High -12.1% vs
  Low +48.0%. No order: tiers must not drive stakes or picks. (Board stakes already use chance, not tier.)
- P (line shopping): best same-line price beat the ledger's median price on 207/248 bets (3.2 books per line);
  +$231.53 -> +$272.08, ROI +9.3% -> +11.0%. Shopping adds about 1.7 points of ROI.

## Round 7 (frozen 2026-11-01, before the replay is run)

Replay of the main rule (hit55f + rec55) for 2026 weeks 6-7 with the 7+ point blowout filter OFF:
`.venv/bin/python experiments/blowout_replay.py`. Each placement run is rebuilt from the `prop_lines` /
`game_lines` snapshot taken just before that run's `placed_at`, the week's projection CSVs and the same
ladders, chance and break-even rules as `paper_trades.place`; settled from `player_weekly_stats` (no row = void).
Sanity check: the replay's bets in non-blowout games must match the real ledger on >= 90% of (player, stat)
rows, else the test is void.

- Q (market blowouts): bets in games whose consensus spread was >= 7 at placement. Lift the filter (main rule,
  from week 9) only if those bets made money in week 6 and in week 7, and their pooled ROI >= the pooled ROI of
  the non-blowout bets. Otherwise keep the filter.
- R (our own score prediction, my idea): our game model (projections/game_model, team_only, p_home
  win) turned into a margin as 13.5 * inverse-normal(p_home); blowout = |margin| >= 7. Replace the market
  definition with R only if, in each of week 6 and week 7, ROI of the bets R keeps > ROI of the bets the market
  filter keeps. Report only: how many games each definition flags, and the overlap.

## Round 7 results (run 2026-11-01)

Sanity OK: 93% of replayed non-blowout bets are in the ledger, 91% of ledger bets replayed.
- Q: FAIL, keep a blowout filter. Bets in 7+ spread games: week 6 36-34 -$44.61 (-6.4%), week 7 10-4 +$51.01
  (+36.4%); pooled 84 bets +0.8% vs +11.3% for the other 252.
- R: PASS on the frozen bar, thinly. Kept by our-margin filter vs market filter: week 6 +4.4% vs +1.3%, week 7
  +17.9% vs +17.8% (a $1.62 difference). Bets only ours drops: 41, +6.1%; bets only the market drops: 27, +18.8%.
  Our model flagged 7 games, the market 6, both 4. Week 8 check: our model flags all four market blowouts plus
  LAC @ LAR (ours LAR by 14.7, market 6.5). Proposed: main rule uses |our margin| >= 7 from week 9.

## Round 8, exploratory (2026-11-01): ways to pick the best bets, weeks 6, 7 and 8 (Thursday)

`.venv/bin/python experiments/pick_select.py`, 17 methods on 270 settled bets, so not a test: anything used
goes to a frozen forward test first. Rule +7.2%. Top 8 a week by chance -0.5% (top 15 -7.6%); by cushion
+10.8%, by EV +12.7% (top 15 by cushion +9.7%). Line not moved against (weeks 6-7) +15.2%. 2+ sources agree
+10.6%, all 3 agree only +3.3%. Receptions only +11.6%, no rushing +9.6%, score >= 4 +10.6%. Every method lost
in week 6; none was up in all three weeks. Ranking by our chance is the weakest way to pick.

## Round 9 (frozen 2026-11-01, before any week 9 bet is placed): forward test of pick selection

Backtest that motivated it (weeks 6-7, exploratory, Round 8): main rule 248 bets +9.3%; line not moved
against 214 bets +15.2%; 2+ sources agree 158 bets +13.1%; both 144 bets +18.9% (week 6 -6.5%, week 7 +32.9%).

- **S (line + agree):** main-rule bets (hit55f + rec55) whose closing line has not moved 2+ yards (1+ catch
  for receptions) against us (`clv_yards`), and where at least 2 of own / gbm / sleeper projections sit on the
  bet's side of its line.
- **T (top 8 by EV):** the 8 main-rule bets per week with the highest `chance * decimal price - 1`.
- **S-NR (line + agree, no rushing):** S without rushing-yards bets. Weeks 6-7 backtest: 126 bets +21.0%
  (S's 18 rushing bets: +4.2%). Added 2026-11-01, still before any week 9 bet is placed.
- **G (one per game):** within S-NR, the single highest-EV bet in each game - the real-money singles rule.
  Weeks 6-7 backtest: top 8 by EV, one per game, +45.2% (random one per game averaged +31.9%) - tiny sample.
- Weeks 9 and 10, official settled results, $10 flat stakes as in the ledger. Ledgers are not changed.
- **Pass:** the subset's ROI beats the full rule's ROI in week 9 AND in week 10, AND its pooled ROI is at
  least 3 points above the full rule's. Anything else is a FAIL, and the subset is not used to pick real-money bets.
- Run: extend `experiments/pick_select.py` with `WEEKS = (9, 10)`. Judge only after week 10 is settled.
