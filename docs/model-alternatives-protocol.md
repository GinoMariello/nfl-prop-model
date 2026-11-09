# Model alternatives protocol (frozen 2026-10-27, before any candidate was run)

Follows `docs/lstm-fix-protocol.md` (whose LSTM variants failed: tuning moved QB passing MAE
by 0.2 yd or less; LSTM and GBM errors correlate 0.94). Same evaluation weeks, same players,
same actuals. Changing anything below is a new protocol version, said explicitly.

## Evaluation

QB starters, 2025 weeks 10-18 and 2026 weeks 2-7 (15 weeks), every model retrained per week
on games strictly before it. Harness: `experiments/model_alt.py`.

## Baseline

`gbm_base`: GbmProjections' features and settings exactly (median loss, 300 iters, lr 0.05,
min_samples_leaf 40), refit in the harness so every candidate shares its rows.

## Point candidates (one change each, against `gbm_base`)

1. `gbm_league`: + league environment -- league-wide passing yards, attempts and PPR per QB
   start, EWM (halflife 3 weeks) over weeks strictly before, plus the player's ewm3 passing
   yards divided by the league figure. Tests "league patterns help".
2. `gbm_league_lstm`: + the same league series forecast one week ahead by a small LSTM trained
   only on weeks before the cutoff. Tests whether an LSTM reads league patterns better than
   a plain EWM. (Training rows get the LSTM's in-sample one-step forecasts; the target week
   gets its true forecast.)
3. `shrink`: empirical-Bayes player mean -- the player's last 16 starts pulled toward the
   league starter mean with pseudo-count k, k picked on training weeks only. Standalone.
4. `gbm_shrink`: `gbm_base` + `shrink` as a feature.

Pass (QB `passing_yards` MAE): at least 0.5 yd below `gbm_base` pooled, not worse in either
season, `rushing_yards` MAE no more than 0.3 yd worse (league/shrink features built for both).

## Distribution candidate

5. `gbm_quantile`: HistGradientBoostingRegressor(loss="quantile") at q = 0.1..0.9, sorted so
   quantiles never cross. Baseline distribution: `gbm_base` + quantiles of held-out residuals
   (a copy of `gbm_base` fit on the first 85% of training rows by time, residuals taken on the
   last 15%) -- the point-plus-miss-distribution idea the paper rule's ladder uses.

Pass: mean pinball loss over the 9 quantiles lower than baseline pooled and in both seasons,
and 10-90% interval coverage within 72-88%.

## Not in this round

TabPFN (needs a package and model-weight download; asked separately). NGBoost/LightGBM not
installed; sklearn's quantile loss covers the distribution question first.

## Result (run 2026-10-27, `experiments/model_alt.py`, 443 QB starts)

| candidate | pass MAE | 2025 | 2026 | rush MAE | gate |
|---|---|---|---|---|---|
| gbm_base | 61.53 | 66.05 | 54.53 | 11.58 | - |
| gbm_league | 61.02 | 64.68 | 55.37 | 11.62 | FAIL (2026 worse) |
| gbm_league_lstm | 62.17 | 65.98 | 56.29 | 11.86 | FAIL |
| shrink | 63.50 | 65.99 | 59.66 | 11.43 | FAIL |
| gbm_shrink | 60.93 | 65.43 | 53.97 | 11.27 | PASS (0.60 yd; 9 of 15 weeks better; week-bootstrap 90% CI -1.36..+0.15) |

Distribution: qgbm pinball 23.96 vs qbase 24.19 (both seasons lower) but 10-90% coverage 65.9%
(too narrow) -> FAIL. qbase coverage 82.2%.

LSTM fix protocol, final: current 60.00, yards_only 59.82, regularized 60.20, median 60.17 -- all FAIL.

## Round 2: TabPFN (frozen 2026-10-27, before it was run)

6. `tabpfn`: TabPFNRegressor (tabpfn 9.2.0, default model, CPU, random_state 0) on exactly
   `gbm_base`'s features and training rows (the most recent 10,000 if more), predicting the
   median (`output_type="median"`, the same target as `gbm_base`). Same 15 weeks, same players.

Pass: the point-candidate gate above (>= 0.5 yd below `gbm_base` pooled passing MAE, not worse
in either season, rushing no more than 0.3 yd worse).

## Shadow run (frozen 2026-10-27, before weeks 8-9 are played)

`gbm_shrink` runs alongside production for 2026 weeks 8 and 9 (no production change):
`.venv/bin/python experiments/model_alt.py 2026:8,2026:9` once week 9 is in `player_weekly_stats`.
Ships to `GbmProjections.py` only if, on those two weeks pooled, its QB passing MAE is no worse
than `gbm_base` AND its rushing MAE no more than 0.3 yd worse. Otherwise it stays an experiment.

## Round 2 result (run 2026-10-27)

`tabpfn`: passing MAE 59.98 (2025 62.87, 2026 55.52), rushing 11.24 vs `gbm_base` 61.53 (66.05, 54.53), 11.58
-> FAIL on gate 3 (worse in 2026). Best pooled number of any candidate; the gain is all in late 2025.
TabPFN removed 2026-10-27 after it failed: package uninstalled, weights and its auth cache moved to ~/.Trash, harness branch deleted (history: commit 9152e10).
