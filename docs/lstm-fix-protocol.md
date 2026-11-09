# LSTM fix protocol (frozen 2026-10-27, before any variant was run)

Frozen before results, in the spirit of `docs/experiment-protocol.md`: the
baselines, evaluation weeks and pass/fail rule below do not change for this
experiment. Changing any of them is a new protocol version, said explicitly.

## Why

The LSTM (`PlayerStatLSTM`, `PlayerProjections.py`) barely beats a last-5
average and loses to the GBM in every market (2026 weeks 2-7). Diagnosis on a
QB walk-forward run (2026 week 6 cutoff, seed 0): 1,174 training examples,
best epoch 9 of 30 (overfits after), passing yards get 13% of a 9-stat loss,
removing the order of the 5 input games changes MAE by 0.3 yds (the recurrence
adds nothing), context is used (shuffling it costs ~3 yds).

## Scope

QB first (the LSTM is 0.7 of the QB blend). Other positions only after QB.

## Evaluation weeks (walk-forward, every model retrained per week on earlier games only)

2025 weeks 10-18 and 2026 weeks 2-7: 15 weeks. QB starters only, as in
production (`project_week`: lookback 3 seasons, 30 epochs, `starters_only`).
Players scored: every starter that every compared model projects that week.
Actuals: `player_weekly_stats` (REG).

## Baselines (fixed)

- `gbm`: GbmProjections' walk-forward projection (`projections/gbm/2025_w*_QB.csv`,
  `gbm_<stat>` in `projections/2026_w*_QB.csv`).
- `last5`: mean of the player's last 5 starts before the week.
- `current`: today's LSTM config, rerun in the same harness and seeds as each
  variant (so seed noise is shared, not compared against a stored number).

## Variants (one change each, against `current`)

1. `yards_only`: loss on `passing_yards` only (all 9 stats still inputs).
2. `regularized`: AdamW weight_decay 1e-3 + grad-norm clip 1.0.
3. `median`: pinball loss at q=0.5 (existing `quantile` option).

Screening uses 3 seeds (0, 1, 2) averaged, as `walk_forward_ensemble_predict`
does. The winner is confirmed with production's 5 seeds before it ships.
Variants combine only after each passes alone.

## Pass/fail (primary stat: QB `passing_yards` MAE, pooled 15 weeks)

A variant ships only if ALL hold:

1. MAE below `gbm`'s.
2. MAE at least 1.0 yd below `current`'s.
3. Not worse than `current` in either season subset (2025 w10-18, 2026 w2-7).
4. Guardrail: `rushing_yards` MAE no more than 0.3 yd worse than `current`.
5. The 5-seed confirmation run still meets 1-4.

Reported, not gating: right side of the posted line (2026 weeks 5-7, ~70 lines),
per-seed spread.

If no variant passes: the LSTM stays as is, and the next step is the
QB blend question (`BLEND_WEIGHTS`), not more tuning.

## After a pass

- `mle-reviewer` pass over the diff for leakage and reproducibility.
- A regression test (`tests/test_lstm_vs_gbm.py`) that fails if the shipped
  config's QB passing MAE on a fixed frozen week set falls behind `gbm`.
- Re-check the paper rule's `LADDER_K` calibration, since QB `pred_` moves.
