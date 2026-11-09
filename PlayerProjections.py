"""
PlayerProjections.py
======================
Next-game player stat projection using an LSTM (PyTorch), matchup-aware.

WHAT THIS DOES
--------------
For each offensive skill position (QB, RB, WR, TE), trains a separate LSTM
regression model that looks at a player's last few games' stat lines PLUS
this week's specific matchup (opponent defensive strength, historical
head-to-head tendency against this exact opponent), and predicts their next
game's stat line. Training data comes from the same weekly stats this
project already scrapes and stores in SQLite (see StatScraper.py) -- this
file imports directly from StatScraper.py rather than re-implementing data
loading.

HOW "REWARDED FOR CLOSENESS" IS IMPLEMENTED
--------------------------------------------
This is supervised regression, not reinforcement learning -- there's no
environment or sequence of actions to reinforce, just "given these past
games and this matchup, predict this next one." The equivalent of a reward
signal here is the loss function: Mean Squared Error (MSE) between the
predicted stat line and the real one. Minimizing MSE during training IS the
model being "rewarded" for closeness -- a perfect prediction contributes
zero loss, and loss grows the further off a prediction is. Validation-set
Mean Absolute Error (MAE), reported in real stat units (yards, TDs, etc.),
is printed during training as the human-readable version of "how close it's
getting."

MATCHUP FEATURES (added on top of the original player-history-only model)
----------------------------------------------------------------------------
Every prediction is conditioned on CONTEXT_COLUMNS (27 numbers, split into
TEAM_CONTEXT_COLUMNS + PLAYER_CONTEXT_COLUMNS) describing the specific game
being predicted, not just the player's own past performance:
  - opp_pass_yards_allowed, opp_pass_tds_allowed, opp_rush_yards_allowed,
    opp_rush_tds_allowed: the upcoming opponent's trailing team defense --
    a tougher pass defense should pull QB/WR/TE passing-game projections
    down, a weak run defense should raise RB projections. See
    build_defense_profile().
  - opp_top_cb_coverage_score: the opponent's best cornerback's trailing
    passer-rating-allowed on real targeted-coverage stats (times targeted,
    completions/yards/TDs allowed -- from Pro-Football-Reference via
    nflverse), not a rough proxy. See build_top_cb_profile().
  - h2h_pass_yards, h2h_pass_tds, h2h_rush_yards, h2h_rush_tds: this
    player's TEAM's own trailing average offensive output specifically in
    past meetings against THIS opponent. See build_head_to_head_profile().
  - is_home, rest_days, opponent_rest_days, is_divisional_game, game_temp,
    game_wind: schedule/situational facts about the game itself (home
    field, bye-week rest or short-week fatigue, division-rivalry
    familiarity, weather) -- known in advance, not leakage-sensitive at
    all. See build_schedule_features().
  - games_with_current_qb: how many of the team's recent games the CURRENT
    starting QB has started -- 0 means a brand-new starter (a backup
    stepping in), catching QB-change situations that would otherwise
    silently confuse the model about why a WR's/RB's production shifted.
    See build_qb_continuity_profile().
  - team_plays_per_game, opponent_plays_per_game: trailing average
    offensive plays (pass attempts + rush carries) per game, for this team
    and the upcoming opponent -- a pace/ball-control proxy (frequent
    3-and-outs, or an opponent that holds the ball a long time, both mean
    fewer plays/opportunities for this team's own players). See
    build_team_pace_profile().
  - opponent_takeaways_per_game: the upcoming opponent's DEFENSE's trailing
    average takeaways (interceptions + recovered fumbles) per game -- a
    real added risk beyond what yards-allowed alone captures. See
    build_team_takeaways_profile().
  - team_ol_injuries: count of this team's own offensive linemen with a
    game-status designation this week -- protection/run-blocking health.
    See build_ol_injury_profile().
  - implied_team_total, team_spread_line: the betting market's expected
    points for this team and the points it is favored by, from the game's
    total and spread. Known before kickoff. See build_schedule_features().
  - player_team_position_rank: is this player their own team's #1 option at
    their position right now? See build_player_team_rank_profile().
  - size_diff_vs_top_cb_height, size_diff_vs_top_cb_weight: how much
    taller/heavier this player is than the opponent's identified top
    cornerback -- see build_size_diff_profile() and the LIMITATION note
    below on what this can and can't represent.
  - player_age: fractional years old as of that week (nflverse's own
    roster-snapshot computation) -- gives the model a direct signal for
    age-related patterns (e.g. younger players trending more
    turnover/mistake-prone, older players trending toward reduced usage or
    injury) to learn from the data, rather than assuming their shape up
    front. See build_age_profile().
  - player_years_experience: seasons played in the NFL (0 = rookie) --
    deliberately separate from player_age (a 24-year-old day-one rookie and
    a 24-year-old entering his 3rd season have the same age but very
    different experience). See build_age_profile().
  - player_injury_status: this week's official game-status designation
    (Questionable/Doubtful/Out), ordinal-encoded via INJURY_STATUS_ORDINAL
    -- a known-in-advance fact, not derived from the game's own result. See
    build_injury_profile() and StatScraper.get_weekly_injury_reports()'s
    Notes for what this can't fully capture (a truly "Out" player mostly
    has no real stat line to train against, since they didn't play; the
    learned signal here comes overwhelmingly from Questionable/Doubtful
    players who did play through a designation).
  - player_vs_opponent_difficulty: this player's own trailing average
    production specifically against defenses as tough as this week's
    opponent -- captures "does this player elevate or struggle against
    tough matchups," independent of opp_*_allowed above (which describes
    the opponent, not how THIS player has historically handled that level
    of competition). See build_opponent_difficulty_profile().
The opp_*/h2h_*/top-CB features are built from trailing/shifted windows so
a game's own result can never leak into its own context (see
build_defense_profile()/build_head_to_head_profile()/build_top_cb_profile()
for the leakage-prevention details); the schedule/QB-continuity features
are direct known-in-advance facts with their own, different leak-safety
reasoning (see build_schedule_features()'s and
build_qb_continuity_profile()'s Notes).

Also: RB/WR/TE's own stat_columns (both the input sequence and the
predicted target) include target_share, air_yards_share, and wopr --
already computed by nflverse, not an extra data pull -- which measure a
pass-catcher's ROLE/OPPORTUNITY within their offense directly, rather than
leaving the model to infer it purely from raw reception/yardage counts.

LIMITATION -- no player-level coverage assignment: "which CB covers which
WR on a given play" is proprietary charting data (e.g. PFF), not something
nflverse's free stats include, so size_diff_vs_top_cb_* and
opp_top_cb_coverage_score describe the opponent's SINGLE best-rated
cornerback overall, not necessarily the specific defender who'll actually
be assigned to this specific player. It's the closest available proxy, not
a substitute for real coverage-assignment data.

TWO WAYS TO TRAIN/USE THIS (both available, pick per use)
------------------------------------------------------------
1. train_all_position_models() -- trains each position's model ONCE on all
   available history, saves it to disk. Fast to reuse; predict_next_game()
   just runs the already-trained model. Good default for "give me a
   projection right now."
2. walk_forward_train_and_predict() / walk_forward_backtest() -- the
   literal step-by-step process: pretend a given (season, week) is "right
   now," train fresh using ONLY the 3 (by default) years immediately before
   it, predict that week, and (for the backtest) fold that week in and move
   to the next one. Slower (retrains from scratch every call) but gives an
   always-current model and, when run across many past weeks, an honest
   backtested accuracy number -- see both functions' docstrings.
   You choose which to use each time: call walk_forward_train_and_predict()
   for a single week when you want a freshly-retrained live projection, or
   walk_forward_backtest() to evaluate a whole season's worth of weeks.

UNCERTAINTY -- MONTE CARLO PROJECTIONS (floor / median / ceiling)
--------------------------------------------------------------------
predict_next_game() gives a single point estimate. For a distribution of
plausible outcomes instead of one number, build_all_residual_pools() runs
walk_forward_backtest() per position and pools the real, leakage-free
(actual - predicted) error VECTORS it produces -- one per backtested
player-week, covering every stat at once so the natural correlation between
stats (a big game is big across stats, not randomly big in just one) is
preserved -- bucketed by predicted fantasy_points_ppr so a confident
projection isn't given the same spread as a volatile one. Once a position's
pool is built and saved (see save_residual_pool()/load_residual_pool()),
predict_next_game_monte_carlo() bootstraps thousands of simulated stat
lines from it around a player's point prediction and reports percentile
summaries (25th/50th/75th by default) -- see build_residual_pool()'s and
predict_next_game_monte_carlo()'s docstrings for the full reasoning.

UNCERTAINTY -- QUANTILE ENSEMBLE (a second, complementary floor/ceiling)
-----------------------------------------------------------------------
predict_next_game_ensemble() takes a different approach to the same
"give me a range, not one number" goal: instead of bootstrapping backtest
error onto a single model's point estimate, it trains 2 EXTRA models per
position with pinball (quantile) loss -- see train_position_model()'s
`quantile` param / train_all_quantile_models() -- directly learning a
floor (low quantile, e.g. 10th) and ceiling (high quantile, e.g. 90th)
conditioned on the FULL feature set for each specific prediction, which can
capture finer-grained heteroscedastic uncertainty than Monte Carlo's
3-bucket bootstrap. The "likely" value it reports is also an ensemble of
this project's 2 different point-estimate architectures -- PlayerStatLSTM
here and the Transformer counterpart in PlayerProjectionsTransformer.py,
both MSE-trained -- averaged together, which predict_next_game_monte_carlo()
alone doesn't get (it's built on the LSTM only). Neither approach makes the
other redundant; see predict_next_game_ensemble()'s docstring.

DESIGN CHOICES (documented so they're easy to revisit later)
--------------------------------------------------------------
- One LSTM per position group (QB/RB/WR/TE), each predicting only the stats
  relevant to that position -- a QB and a WR don't share a meaningful stat
  vector, so mixing them into one model would mostly be modeling on noise
  for the other position's columns.
- Kicking/defense/special-teams/offensive-line positions are out of scope:
  this is meant for fantasy-relevant skill-position projections. Extending
  POSITION_STAT_COLUMNS below to add e.g. "K" is straightforward if needed.
- Recency-weighted training loss (see compute_recency_weights()): every
  training example's loss contribution decays exponentially with how old
  its target game is (half-life RECENCY_HALF_LIFE_WEEKS), rather than a
  game from 3 seasons ago counting the same as one from last week. Applied
  automatically inside train_position_model()/train_position_model_hf(),
  not an opt-in flag -- current-season form is a genuinely different signal
  from a player's whole career average, and should outweigh it.
- Fixed-length input window (WINDOW_SIZE games, default 5): every training
  example uses exactly that many past games, so players with fewer than
  WINDOW_SIZE games on record are skipped entirely rather than zero-padded.
  This keeps the model simple (no padding/packing/masking logic) at the
  cost of losing rookies' and injury-return players' earliest eligible
  games. If you want to include those, the natural extension is
  `torch.nn.utils.rnn.pack_padded_sequence` with left-padding.
- Train/validation split is CHRONOLOGICAL (earliest games train, most
  recent games validate), not a random or per-player split. A random split
  would leak future information into training (the model could "see" a
  player's week 10 performance while learning to predict their week 3), and
  a per-player split would test generalization to entirely unseen players,
  which isn't this tool's use case -- it projects players it already has
  data on.
- Player-history feature set == target set (a purely autoregressive
  sequence): the sequence inputs are the same stat columns being predicted,
  just from earlier games. Matchup context (see above) is the one exception
  -- it's exogenous, known-in-advance information about the upcoming game,
  concatenated onto the LSTM's output rather than fed through the sequence.
  No home/away or weather/injury-report features are included -- those
  would meaningfully improve accuracy but are a separate, larger scope than
  what was asked for here.
- The matchup-context fill value for "no data yet" (a team's first tracked
  game, or two teams that have never met in the data) is estimated from the
  full context table rather than a strict per-split training-only slice --
  see fillna_context()'s docstring for why this is a defensible, minor
  simplification rather than real leakage.

USAGE
-----
    python PlayerProjections.py
        Trains and saves all four position models (into ./models/) using
        all available history.

    from PlayerProjections import predict_next_game
    predict_next_game(player_id="00-0039732", next_opponent="LV")
        # df/defensive_target_stats/player_bio/schedule_context/player_age/
        # injury_reports all default to loading themselves if not passed --
        # pass them explicitly (see load_weekly_stats()/
        # load_defensive_target_stats()/load_player_bio()/
        # load_schedule_context()/load_player_age()/load_injury_reports())
        # when predicting many players in a loop, so each call doesn't
        # reload the same data from scratch. Pass injury_status="Doubtful"
        # (etc.) to override this week's real/missing injury report.

    from PlayerProjections import (
        load_weekly_stats, load_defensive_target_stats, load_player_bio, load_schedule_context,
        load_player_age, load_injury_reports, walk_forward_backtest, summarize_backtest, POSITION_STAT_COLUMNS,
    )
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)
    results = walk_forward_backtest(
        df, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports, "QB",
        eval_weeks=[(2025, w) for w in range(1, 6)],
    )
    print(summarize_backtest(results, POSITION_STAT_COLUMNS["QB"]))
"""

import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from GbmProjections import gbm_project_week
from StatScraper import (
    POSITION_RANKING_STAT,
    default_recent_seasons,
    get_player_bio,
    get_player_weekly_stats_nflverse,
    get_schedule_context,
    get_sleeper_injuries,
    get_sleeper_projections,
    get_weekly_defensive_target_stats,
    get_weekly_injury_reports,
    get_weekly_player_age,
    query_sqlite,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Which stat columns each position group is trained to predict. Column names
# match StatScraper's player_weekly_stats table exactly (see the "Column
# ambiguity warning" note in StatScraper.get_player_weekly_stats_nflverse for
# why e.g. "passing_interceptions" is spelled out rather than "interceptions").
# RB/WR/TE also include target_share, air_yards_share, and wopr (Weighted
# Opportunity Rating) -- these already exist, pre-computed, in nflverse's
# weekly stats (no extra data pull needed) and directly measure a
# pass-catcher's ROLE/OPPORTUNITY within their offense, independent of
# whatever bounced their way in a given box score. As inputs, they give the
# LSTM a much more direct usage signal than trying to infer role purely
# from raw reception/yardage counts; as prediction targets, forecasting
# next week's target share is a reasonable secondary output alongside the
# box-score stats. QB is left out since target/air-yards share describe a
# pass-catcher's slice of their team's passing offense, not a passer's own
# role.
POSITION_STAT_COLUMNS: Dict[str, List[str]] = {
    "QB": ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
           "carries", "rushing_yards", "rushing_tds", "fantasy_points_ppr"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "receptions", "targets",
           "receiving_yards", "receiving_tds", "target_share", "air_yards_share", "wopr", "fantasy_points_ppr"],
    "WR": ["receptions", "targets", "receiving_yards", "receiving_tds",
           "rushing_yards", "rushing_tds", "target_share", "air_yards_share", "wopr", "fantasy_points_ppr"],
    "TE": ["receptions", "targets", "receiving_yards", "receiving_tds",
           "target_share", "air_yards_share", "wopr", "fantasy_points_ppr"],
}

# OL positions used by build_ol_injury_profile()/current_matchup_context()'s
# team_ol_injuries feature -- nflverse spells these a few different ways
# depending on source table (get_player_bio()'s position column).
OL_POSITIONS = {"T", "G", "C", "OL", "OT", "OG"}

# The 22 TEAM-level matchup features -- these describe the upcoming
# game/opponent/situation and are the same for every player on a team that
# week. See build_matchup_context(). Order matters within this list.
#   - opp_*: opponent defense strength + best CB (see build_defense_profile()/
#     build_top_cb_profile()).
#   - h2h_*: historical head-to-head tendency vs this exact opponent (see
#     build_head_to_head_profile()).
#   - is_home, rest_days, opponent_rest_days, is_divisional_game: straight
#     from the schedule -- these are known facts about the upcoming game
#     itself, not something inferred from past performance, so unlike the
#     opp_*/h2h_* features there's no leakage question at all (see
#     get_schedule_context()).
#   - game_temp, game_wind: weather -- filled with a "climate-controlled, no
#     weather effect" constant for dome games rather than a raw average
#     (see build_schedule_features()).
#   - games_with_current_qb: how many of the team's past games (capped at
#     WINDOW_SIZE) were started by the SAME QB starting this week -- 0 means
#     a brand-new starter (backup stepping in), catching QB-change
#     situations that would otherwise silently confuse the model (see
#     build_qb_continuity_profile()).
#   - team_plays_per_game: this team's own trailing average offensive plays
#     (pass attempts + rush carries) -- a pace/ball-control proxy; a team
#     sustaining fewer drives (e.g. frequent 3-and-outs) gives every one of
#     its own players fewer opportunities that game. See build_team_pace_profile().
#   - opponent_plays_per_game: same, for the upcoming OPPONENT -- a
#     ball-control opponent that holds the ball longer leaves this team's
#     own offense fewer possessions/plays.
#   - opponent_takeaways_per_game: the upcoming opponent's DEFENSE's
#     trailing average takeaways (interceptions + recovered fumbles) per
#     game -- a real added risk (and short-field/extra-possession swing)
#     beyond what yards-allowed alone captures. See build_team_takeaways_profile().
#   - implied_team_total, team_spread_line: the betting market's view of
#     this game -- points this team is expected to score ((total + spread) / 2)
#     and points it is favored by. Set before kickoff and carry no result;
#     history uses nflverse's closing line, live projections the current one
#     (see StatScraper.get_schedule_context()). Last in this list so models
#     saved before they existed still load (see _predict_next_game_core()).
#   - team_ol_injuries: count of this team's OWN offensive linemen
#     (T/G/C -- see OL_POSITIONS) with a game-status designation for this
#     upcoming week -- protection/run-blocking health, distinct from any
#     single skill-position player's own injury_status. See
#     build_ol_injury_profile().
TEAM_CONTEXT_COLUMNS: List[str] = [
    "opp_pass_yards_allowed", "opp_pass_tds_allowed", "opp_rush_yards_allowed", "opp_rush_tds_allowed",
    "opp_top_cb_coverage_score",
    "h2h_pass_yards", "h2h_pass_tds", "h2h_rush_yards", "h2h_rush_tds",
    "is_home", "rest_days", "opponent_rest_days", "is_divisional_game", "game_temp", "game_wind",
    "games_with_current_qb",
    "team_plays_per_game", "opponent_plays_per_game", "opponent_takeaways_per_game", "team_ol_injuries",
    "implied_team_total", "team_spread_line",
]
# Added after models were first saved; a model whose context is this much shorter
# predates them (see _predict_next_game_core()).
VEGAS_CONTEXT_COLUMNS: List[str] = ["implied_team_total", "team_spread_line"]

# The 7 PLAYER-level matchup features -- these depend on the specific player
# being predicted, not just their team, so they're computed differently from
# the team-level ones above (see build_player_team_rank_profile(),
# build_size_diff_profile(), build_age_profile(), build_injury_profile(),
# build_opponent_difficulty_profile()):
#   - player_team_position_rank: is this player their own team's #1 option
#     at their position right now?
#   - size_diff_vs_top_cb_height / _weight: how much taller/heavier is this
#     player than the opponent's identified top cornerback (inches / lbs;
#     positive means the offensive player is bigger) -- captures "a big TE
#     can win physical matchups even against a smaller, highly-ranked CB."
#   - player_age: fractional years old as of that week (nflverse's own
#     roster-snapshot computation -- see StatScraper.get_weekly_player_age())
#     -- lets the model learn age-related patterns directly (e.g. younger
#     players being more turnover/mistake-prone, older players trending
#     toward reduced usage) from the data itself, rather than assuming any
#     particular shape for that relationship up front.
#   - player_years_experience: seasons played in the NFL (0 = rookie) --
#     deliberately separate from player_age: a 24-year-old day-one rookie
#     and a 24-year-old entering his 3rd season have the same age but very
#     different experience, which is the more direct signal for
#     "green/mistake-prone" vs. "established."
#   - player_injury_status: this week's official game-status designation,
#     ordinal-encoded via INJURY_STATUS_ORDINAL (0=no report, 1=Questionable,
#     2=Doubtful, 3=Out) -- see StatScraper.get_weekly_injury_reports() and
#     build_injury_profile()'s Notes for what this can and can't capture.
#   - player_vs_opponent_difficulty: this player's own trailing average
#     production specifically against defenses as tough as this week's
#     opponent (see build_opponent_difficulty_profile()) -- captures
#     "does this player elevate or struggle against tough matchups,"
#     independent of the opp_*_allowed team-level features above (which
#     describe the opponent, not how THIS player has historically handled
#     that level of competition).
#   - player_archetype: this player's current play-style archetype
#     (Scrambler/Improviser/Gunslinger/Field General for QB; a different,
#     position-appropriate set for RB/WR/TE), ordinal-encoded per
#     PlayerArchetypes.ARCHETYPE_CENTROIDS[position_group]'s key order --
#     see PlayerArchetypes.py and build_archetype_context().
#   - opponent_defense_vs_archetype: the upcoming opponent's trailing
#     performance SPECIFICALLY against players of this player's own
#     archetype (passer rating allowed for QB, fantasy points allowed for
#     RB/WR/TE) -- see PlayerArchetypes.build_defense_vs_archetype_profile().
PLAYER_CONTEXT_COLUMNS: List[str] = [
    "player_team_position_rank", "size_diff_vs_top_cb_height", "size_diff_vs_top_cb_weight",
    "player_age", "player_years_experience", "player_injury_status", "player_vs_opponent_difficulty",
    "player_archetype", "opponent_defense_vs_archetype",
]

# All 31 matchup-context features every model is conditioned on, team-level
# ones first, matching how build_sequences()/current_matchup_context()
# assemble the combined vector.
CONTEXT_COLUMNS: List[str] = TEAM_CONTEXT_COLUMNS + PLAYER_CONTEXT_COLUMNS

# Ordinal encoding for player_injury_status -- higher means less likely to
# play a full, normal workload. A player with no row in that week's injury
# report (the common case) gets 0, not a missing value -- absence of a
# report means no game-status concern was raised, which IS the meaningful
# "healthy" signal here (see build_injury_profile()'s Notes on what this
# can't distinguish: a truly "Out" player essentially never has a real
# stat line to train the target side of that row against, since they
# didn't play).
INJURY_STATUS_ORDINAL: Dict[str, int] = {"Questionable": 1, "Doubtful": 2, "Out": 3}

WINDOW_SIZE = 5                  # number of past games fed into the LSTM per prediction
ROLLING_DEFENSE_WINDOW = 8       # games of trailing history used for opponent defense-strength and top-CB features
ROLLING_H2H_WINDOW = 10          # meetings of trailing history used for head-to-head features (teams rarely meet more than 2x/season, so this effectively means "all prior meetings")
MIN_TRAILING_CB_TARGETS = 5      # a CB needs at least this many trailing targets before their passer-rating-allowed is trusted as "best CB" (see build_top_cb_profile) -- below this, one lucky/unlucky play swings the rating wildly
DOME_TEMP_FALLBACK = 70.0        # "climate-controlled" fill for temp on dome games (roof is not "outdoors"/"open") -- a known fact, not missing data
DOME_WIND_FALLBACK = 0.0         # same, for wind
MODEL_DIR = "models"             # trained weights + normalization stats are saved here
DB_PATH = "data/nfl_stats.db"    # reuses the SQLite database StatScraper.py builds

POISSON_WEIGHT = 1.0             # weight of the auxiliary Poisson term added onto MSE loss for count-like stat columns (touchdowns, interceptions) -- see train_position_model()'s Notes on why
LOW_QUANTILE = 0.1                # default lower quantile for the floor model -- see train_position_model()'s `quantile` param and predict_next_game_ensemble()
HIGH_QUANTILE = 0.9               # default upper quantile for the ceiling model, same
RECENCY_HALF_LIFE_WEEKS = 52.0    # a training example's loss weight halves every this many weeks older it is than the most recent example -- see compute_recency_weights()

GA_POPULATION_SIZE = 12           # individuals per generation -- see train_position_model_ga()
GA_SURVIVORS = 4                  # top-fitness individuals kept as parents each generation
GA_ENSEMBLE_SIZE = 3              # how many of the final population get saved/used as this position's "3 models" -- see predict_next_game_ga()
GA_MUTATION_STD = 0.05            # std of Gaussian noise added to a mutated weight
GA_MUTATION_RATE = 0.15           # fraction of each parameter tensor's elements mutated per child
GA_DEFAULT_GENERATIONS = 50       # default generations per train_position_model_ga() call -- see save_population()/resume_population for continuing evolution across multiple calls instead of raising this


# ---------------------------------------------------------------------------
# DATA LOADING -- reuses StatScraper.py rather than re-scraping
# ---------------------------------------------------------------------------
def load_weekly_stats(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load regular-season weekly player stats to train on, refreshing the
    shared SQLite database first (via StatScraper's scraper) so this always
    trains on current data.

    Args:
        years: Season years to include, e.g. [2022, 2023, 2024, 2025].
            Defaults to StatScraper's default_recent_seasons() (last 3
            seasons) if not given. Widen this if you plan to use the
            walk-forward functions below -- they need `lookback_years` of
            history BEFORE the earliest week you evaluate, so e.g.
            backtesting all of the 2025 season with lookback_years=3 needs
            years back to 2022.

    Returns:
        DataFrame of regular-season-only rows (season_type == "REG") from
        the player_weekly_stats table -- playoff games are excluded so a
        team's playoff run doesn't get treated as "one more regular game"
        in the training sequence.

    Notes:
        get_player_weekly_stats_nflverse() upserts by (player_id, season,
        week), so calling this repeatedly is cheap after the first run --
        it re-fetches and re-upserts, but won't duplicate rows.

        The underlying SQLite table can end up holding MORE seasons than
        `years` asks for, though -- e.g. an earlier call with years=[2022,
        2023] leaves those rows in the table permanently (upserts only ever
        add/refresh rows, never delete them). The query below explicitly
        filters to `years`, so a later call with a narrower range still
        returns only what was asked for rather than silently including
        whatever history happens to already be sitting in the database.
    """
    years = years if years is not None else default_recent_seasons()

    get_player_weekly_stats_nflverse(years=years, save_path=None, db_path=DB_PATH)

    placeholders = ", ".join("?" * len(years))
    df = query_sqlite(
        f"SELECT * FROM player_weekly_stats WHERE season IN ({placeholders})", db_path=DB_PATH, params=years
    )

    return df[df["season_type"] == "REG"].reset_index(drop=True)


def load_defensive_target_stats(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load real "targeted" defensive coverage stats (times targeted,
    completions/yards/TDs allowed, passer rating allowed) from
    StatScraper's Pro-Football-Reference-backed pull, refreshing the shared
    SQLite database first. Used to identify and rate each team's best
    cornerback -- see build_top_cb_profile().

    Args:
        years: Season years to include. Defaults to
            default_recent_seasons() if not given, same as load_weekly_stats().

    Returns:
        DataFrame of regular-season-only rows -- see
        StatScraper.get_weekly_defensive_target_stats() for columns.
    """
    years = years if years is not None else default_recent_seasons()

    get_weekly_defensive_target_stats(years=years, save_path=None, db_path=DB_PATH)

    placeholders = ", ".join("?" * len(years))
    df = query_sqlite(
        f"SELECT * FROM weekly_defensive_target_stats WHERE season IN ({placeholders})", db_path=DB_PATH, params=years
    )
    return df[df["game_type"] == "REG"].reset_index(drop=True) if "game_type" in df.columns else df


def load_player_bio() -> pd.DataFrame:
    """
    Load player bio/identity data (height, weight, position, and the
    gsis_id/pfr_id crosswalk) from StatScraper's nflverse pull, refreshing
    the shared SQLite database first. Used for the size_diff_vs_top_cb_*
    context features -- see build_size_diff_profile().

    Returns:
        DataFrame with one row per player -- see StatScraper.get_player_bio()
        for columns.
    """
    get_player_bio(save_path=None, db_path=DB_PATH)
    return query_sqlite("SELECT * FROM player_bio", db_path=DB_PATH)


def load_schedule_context(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load per-team schedule context (home/away, rest days, divisional flag,
    weather, starting QB) from StatScraper's nflverse pull, refreshing the
    shared SQLite database first. Used for the is_home/rest_days/
    is_divisional_game/game_temp/game_wind/games_with_current_qb context
    features -- see build_schedule_features()/build_qb_continuity_profile().

    Args:
        years: Season years to include. Defaults to default_recent_seasons()
            if not given, same as load_weekly_stats(). nflverse's schedule
            includes not-yet-played games in the CURRENT season too, so this
            also covers upcoming weeks.

    Returns:
        DataFrame of regular-season-only rows -- see
        StatScraper.get_schedule_context() for columns.
    """
    years = years if years is not None else default_recent_seasons()

    get_schedule_context(years=years, save_path=None, db_path=DB_PATH)

    placeholders = ", ".join("?" * len(years))
    df = query_sqlite(f"SELECT * FROM schedule_context WHERE season IN ({placeholders})", db_path=DB_PATH, params=years)
    return df[df["game_type"] == "REG"].reset_index(drop=True)


def load_player_age(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load each player's fractional age for every week they were on an active
    roster, from StatScraper's nflverse pull, refreshing the shared SQLite
    database first. Used for the player_age context feature -- see
    build_age_profile().

    Args:
        years: Season years to include. Defaults to default_recent_seasons()
            if not given, same as load_weekly_stats().

    Returns:
        DataFrame -- see StatScraper.get_weekly_player_age() for columns.
    """
    years = years if years is not None else default_recent_seasons()

    get_weekly_player_age(years=years, save_path=None, db_path=DB_PATH)

    placeholders = ", ".join("?" * len(years))
    return query_sqlite(f"SELECT * FROM weekly_player_age WHERE season IN ({placeholders})", db_path=DB_PATH, params=years)


def load_injury_reports(years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load the official weekly injury report (game-status designations), from
    StatScraper's nflverse pull, refreshing the shared SQLite database
    first. Used for the player_injury_status context feature -- see
    build_injury_profile().

    Args:
        years: Season years to include. Defaults to default_recent_seasons()
            if not given, same as load_weekly_stats(). Covers the current
            season's already-announced reports too, useful for live
            pregame context once a team's report is out.

    Returns:
        DataFrame -- see StatScraper.get_weekly_injury_reports() for columns.
    """
    years = years if years is not None else default_recent_seasons()

    get_weekly_injury_reports(years=years, save_path=None, db_path=DB_PATH)

    placeholders = ", ".join("?" * len(years))
    return query_sqlite(f"SELECT * FROM weekly_injury_reports WHERE season IN ({placeholders})", db_path=DB_PATH, params=years)


# ---------------------------------------------------------------------------
# MATCHUP CONTEXT FEATURES -- opponent defense strength + head-to-head
# ---------------------------------------------------------------------------
def build_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse player-week rows into one row per TEAM per game, with that
    team's total offensive output -- the building block both
    build_defense_profile() and build_head_to_head_profile() aggregate
    further. Uses ALL position groups (not just one), since a team's real
    offensive output includes every player who touched the ball.

    Args:
        df: Weekly player stats (any subset of columns is fine as long as
            season, week, team, opponent_team, passing_yards, passing_tds,
            rushing_yards, rushing_tds are present -- e.g. load_weekly_stats()'s
            output).

    Returns:
        DataFrame with one row per (season, week, team, opponent_team):
        that team's summed passing_yards, passing_tds, rushing_yards,
        rushing_tds for that game.
    """
    return df.groupby(["season", "week", "team", "opponent_team"], as_index=False).agg(
        passing_yards=("passing_yards", "sum"),
        passing_tds=("passing_tds", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rushing_tds", "sum"),
    )


def build_defense_profile(team_games: pd.DataFrame, window: int = ROLLING_DEFENSE_WINDOW) -> pd.DataFrame:
    """
    For each (season, week, team), compute that team's DEFENSIVE strength
    heading INTO that week's game: a trailing rolling average of yards/TDs
    ALLOWED over their previous `window` games.

    Args:
        team_games: Output of build_team_game_log().
        window: How many past games to average over (default
            ROLLING_DEFENSE_WINDOW).

    Returns:
        DataFrame with one row per (season, week, team): opp_pass_yards_allowed,
        opp_pass_tds_allowed, opp_rush_yards_allowed, opp_rush_tds_allowed.
        NaN for a team's very first tracked game (no prior games to average)
        -- see fillna_context() for how that's handled downstream.

    Notes:
        Leakage prevention: a team's own offensive output IS the opposing
        defense's "yards allowed," so relabeling team_games (offense_team ->
        opponent_team, opponent_team -> team) turns this into a per-defense
        log. `.rolling(window).mean().shift(1)` then means the average
        EXCLUDES the current row -- a defense's strength "entering" week W
        only reflects games strictly before W, never week W's own result or
        anything later. Without the shift, a dominant defensive performance
        in a game would leak into that same game's own "how strong is this
        defense" feature, which is exactly the kind of look-ahead bias that
        would make backtested accuracy numbers meaningless.
    """
    allowed = team_games.rename(columns={
        "team": "offense_team", "opponent_team": "team",
        "passing_yards": "opp_pass_yards_allowed", "passing_tds": "opp_pass_tds_allowed",
        "rushing_yards": "opp_rush_yards_allowed", "rushing_tds": "opp_rush_tds_allowed",
    }).sort_values(["team", "season", "week"])

    cols = ["opp_pass_yards_allowed", "opp_pass_tds_allowed", "opp_rush_yards_allowed", "opp_rush_tds_allowed"]
    allowed[cols] = allowed.groupby("team")[cols].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))

    return allowed[["season", "week", "team"] + cols]


def build_head_to_head_profile(team_games: pd.DataFrame, window: int = ROLLING_H2H_WINDOW) -> pd.DataFrame:
    """
    For each (season, week, team, opponent_team), compute that team's own
    trailing average offensive output specifically in PAST meetings against
    THIS exact opponent.

    Args:
        team_games: Output of build_team_game_log().
        window: How many past meetings to average over (default
            ROLLING_H2H_WINDOW -- effectively "all of them," since two teams
            rarely meet more than twice a season).

    Returns:
        DataFrame with one row per (season, week, team, opponent_team):
        h2h_pass_yards, h2h_pass_tds, h2h_rush_yards, h2h_rush_tds. NaN when
        there's no PRIOR meeting between these two exact teams yet in the
        data -- see fillna_context().

    Notes:
        Same leakage-prevention shape as build_defense_profile(): grouping
        by (team, opponent_team) and shifting by 1 means only meetings
        strictly before this one are averaged.
    """
    h2h = team_games.rename(columns={
        "passing_yards": "h2h_pass_yards", "passing_tds": "h2h_pass_tds",
        "rushing_yards": "h2h_rush_yards", "rushing_tds": "h2h_rush_tds",
    }).sort_values(["team", "opponent_team", "season", "week"])

    cols = ["h2h_pass_yards", "h2h_pass_tds", "h2h_rush_yards", "h2h_rush_tds"]
    h2h[cols] = h2h.groupby(["team", "opponent_team"])[cols].transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))

    return h2h[["season", "week", "team", "opponent_team"] + cols]


def nfl_passer_rating(
    completions: pd.Series, yards: pd.Series, touchdowns: pd.Series, interceptions: pd.Series, attempts: pd.Series
) -> pd.Series:
    """
    The standard NFL passer rating formula -- the same one used for a
    quarterback's own rating, applied here to a DEFENDER's targeted-coverage
    totals ("passer rating allowed"). Lower means tougher to throw at,
    exactly like a QB's own rating measures how well THEY throw. Verified
    to exactly reproduce Pro-Football-Reference's own single-game
    def_passer_rating_allowed values before being used here to aggregate
    across a trailing window (see build_top_cb_profile()).

    Args:
        completions, yards, touchdowns, interceptions, attempts: Equal-length
            Series of counts -- can be single-game raw stats or (as used in
            this file) counts already summed over a trailing window.

    Returns:
        Series of ratings (0-158.3 scale). NaN wherever attempts is 0
        (undefined with no attempts) -- callers should filter those out.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        a = ((completions / attempts) - 0.3) * 5
        b = ((yards / attempts) - 3) * 0.25
        c = (touchdowns / attempts) * 20
        d = 2.375 - ((interceptions / attempts) * 25)
    return ((a.clip(0, 2.375) + b.clip(0, 2.375) + c.clip(0, 2.375) + d.clip(0, 2.375)) / 6) * 100


def build_top_cb_profile(
    defensive_target_stats: pd.DataFrame,
    player_bio: pd.DataFrame,
    window: int = ROLLING_DEFENSE_WINDOW,
    min_targets: int = MIN_TRAILING_CB_TARGETS,
) -> pd.DataFrame:
    """
    For each (season, week, team), identify that team's best cornerback
    heading into that week's game using REAL targeted-coverage stats (see
    StatScraper.get_weekly_defensive_target_stats()) -- how many times each
    CB was actually thrown at and how the opposing offense did against
    them -- rather than the rough interceptions+pass-deflections proxy an
    earlier version of this project used.

    Args:
        defensive_target_stats: Output of load_defensive_target_stats().
        player_bio: Output of load_player_bio() -- used to filter to CBs
            only, since the PFR-sourced defensive_target_stats doesn't
            carry a position column of its own.
        window: Games of trailing history to aggregate each CB's raw
            targeted-coverage counts over before computing their rating
            (default ROLLING_DEFENSE_WINDOW).
        min_targets: A CB's trailing SUM of targets must reach at least this
            many before their rating is trusted as "best CB" (default
            MIN_TRAILING_CB_TARGETS) -- a corner with 1 target and a lucky
            incompletion would otherwise show a perfect 0.0 rating that
            means nothing statistically.

    Returns:
        DataFrame with one row per (season, week, team): opp_top_cb_coverage_score
        (that CB's trailing passer-rating-allowed) and opp_top_cb_player_id
        (their identity, used by build_size_diff_profile() to look up their
        height/weight). Rows where no CB on the team has reached
        min_targets yet are simply absent (left for the caller's
        fillna/live-fallback handling, same as the other cold-start cases).

    Notes:
        Leakage-safe: each CB's raw counts are summed over a trailing
        window and shifted by ONE game BEFORE computing the rating and
        picking the team's best, so neither this game nor any later one
        leaks in. Ratings are computed from trailing AGGREGATED counts, not
        by averaging each game's already-computed rating -- averaging a
        rate stat like passer rating across games with different attempt
        volumes is statistically improper (a 158.3 off 1 target shouldn't
        count the same as a 90 off 10 targets); summing raw counts first
        and computing one rating from the totals is the correct way to
        combine them (see nfl_passer_rating()).
    """
    cb_ids = set(player_bio.loc[player_bio["position"] == "CB", "player_id"])
    cb_df = defensive_target_stats[defensive_target_stats["player_id"].isin(cb_ids)].copy()
    cb_df = cb_df.sort_values(["player_id", "season", "week"])

    raw_cols = ["def_targets", "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed", "def_ints"]
    trailing_cols = [f"trailing_{c}" for c in raw_cols]
    for raw_col, trailing_col in zip(raw_cols, trailing_cols):
        cb_df[trailing_col] = cb_df.groupby("player_id")[raw_col].transform(
            lambda s: s.rolling(window, min_periods=1).sum().shift(1)
        )

    cb_df = cb_df.dropna(subset=trailing_cols)
    cb_df = cb_df[cb_df["trailing_def_targets"] >= min_targets]

    cb_df["trailing_rating"] = nfl_passer_rating(
        cb_df["trailing_def_completions_allowed"], cb_df["trailing_def_yards_allowed"],
        cb_df["trailing_def_receiving_td_allowed"], cb_df["trailing_def_ints"], cb_df["trailing_def_targets"],
    )

    # Team's best CB = LOWEST trailing rating allowed among those who've
    # earned enough trailing targets to trust (idxmin, not idxmax -- lower
    # passer rating allowed is better defense, same convention as a QB's
    # own rating).
    best_idx = cb_df.groupby(["season", "week", "team"])["trailing_rating"].idxmin()
    team_best = cb_df.loc[best_idx, ["season", "week", "team", "player_id", "trailing_rating"]]
    return team_best.rename(columns={"trailing_rating": "opp_top_cb_coverage_score", "player_id": "opp_top_cb_player_id"})


def build_player_team_rank_profile(pos_df: pd.DataFrame, stat_column: str) -> pd.DataFrame:
    """
    For one position group's weekly stats, compute each player's trailing
    rank (1 = current team leader) among their own team's players at that
    position, by season-to-date total of `stat_column` -- "is this player
    their team's clear #1 option at this position, or further down the
    depth chart," heading into each of their games.

    Args:
        pos_df: Weekly stats already filtered to ONE position_group (e.g.
            df[df["position_group"] == "WR"]).
        stat_column: Which stat to rank by -- typically
            StatScraper.POSITION_RANKING_STAT[position_group].

    Returns:
        DataFrame with one row per (player_id, season, week):
        player_team_position_rank (1 = that team's current leader at this
        position heading into this game).

    Notes:
        Leakage-safe: `stat_column`'s cumulative season total is shifted by
        one game before ranking, so a big game doesn't retroactively boost
        that same game's own "how established is this player" feature.
        Ties (e.g. two players both still at 0 cumulative production, as in
        everyone's first game of a season) share the same rank via
        `method="min"`. Unlike the team-level context features, this one
        never needs a "no history" fallback constant -- every row already
        has a real (possibly 0) cumulative stat to rank, both here (training,
        shifted) and in current_matchup_context() (live, unshifted).
    """
    pos_df = pos_df.sort_values(["player_id", "season", "week"]).copy()
    pos_df["trailing_cum_stat"] = pos_df.groupby(["player_id", "season"])[stat_column].transform(lambda s: s.cumsum().shift(1))
    pos_df["trailing_cum_stat"] = pos_df["trailing_cum_stat"].fillna(0.0)
    pos_df["player_team_position_rank"] = pos_df.groupby(["team", "season", "week"])["trailing_cum_stat"].rank(method="min", ascending=False)
    return pos_df[["player_id", "season", "week", "player_team_position_rank"]]


def build_size_diff_profile(pos_df: pd.DataFrame, top_cb_profile: pd.DataFrame, player_bio: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-week in `pos_df`, compute how much taller/heavier they
    are than the opponent's identified top cornerback (see
    build_top_cb_profile()) -- "a big TE might do well against a small but
    highly-ranked corner," made into an explicit feature rather than
    something the model would have to infer indirectly from names alone.

    Args:
        pos_df: Weekly stats for one position group, with at least
            player_id, season, week, team, opponent_team.
        top_cb_profile: Output of build_top_cb_profile() -- needs
            opp_top_cb_player_id, keyed by (season, week, team=that CB's
            OWN team -- which is pos_df's OPPONENT from each row's
            perspective, see the join below).
        player_bio: Output of load_player_bio() -- height/weight lookup.

    Returns:
        DataFrame with one row per (player_id, season, week):
        size_diff_vs_top_cb_height (inches), size_diff_vs_top_cb_weight
        (lbs) -- positive means the offensive player is bigger. 0.0 (a
        neutral "no known mismatch") when no qualifying opponent CB was
        identified that week, or bio data is missing for either player --
        unlike fillna_context()'s league-average approach, 0 is the natural
        "no information" value for a DIFFERENCE, not an average.

    Notes:
        Joins top_cb_profile using the player's OPPONENT team, not their
        own -- getting this backwards (attaching a player's own team's best
        CB to themselves) would be the same class of bug as the opponent-
        defense join issue fixed earlier in this file; see
        build_matchup_context()'s note on that.
    """
    bio = player_bio.drop_duplicates(subset=["player_id"]).set_index("player_id")[["height", "weight"]]

    cb_lookup = top_cb_profile.rename(columns={"team": "opponent_team"})[
        ["season", "week", "opponent_team", "opp_top_cb_player_id"]
    ]
    merged = pos_df[["player_id", "season", "week", "team", "opponent_team"]].merge(
        cb_lookup, on=["season", "week", "opponent_team"], how="left"
    )

    merged["own_height"] = merged["player_id"].map(bio["height"])
    merged["own_weight"] = merged["player_id"].map(bio["weight"])
    merged["cb_height"] = merged["opp_top_cb_player_id"].map(bio["height"])
    merged["cb_weight"] = merged["opp_top_cb_player_id"].map(bio["weight"])

    merged["size_diff_vs_top_cb_height"] = (merged["own_height"] - merged["cb_height"]).fillna(0.0)
    merged["size_diff_vs_top_cb_weight"] = (merged["own_weight"] - merged["cb_weight"]).fillna(0.0)

    return merged[["player_id", "season", "week", "size_diff_vs_top_cb_height", "size_diff_vs_top_cb_weight"]]


def build_age_profile(pos_df: pd.DataFrame, player_age: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each player-week's age AND years of NFL experience as the
    player_age / player_years_experience context features -- both come
    from the same load_player_age() table (see StatScraper.get_weekly_player_age()),
    so one merge produces both rather than two nearly-identical functions.

    Args:
        pos_df: Weekly stats for one position group, with at least
            player_id, season, week.
        player_age: Output of load_player_age().

    Returns:
        DataFrame with one row per (player_id, season, week): player_age
        (fractional years), player_years_experience (integer seasons
        played, 0 for a rookie).

    Notes:
        No leak-safety treatment needed (unlike the trailing/shifted
        opp_*/h2h_* team features) -- like build_schedule_features(), a
        player's age/experience going into a specific week are known,
        fixed facts about that week, not something derived from that
        week's own performance.

        A small fraction of player-weeks have no matching roster record
        (see get_weekly_player_age()'s Notes -- a practice-squad elevation
        or similar not reflected in that week's roster snapshot) -- these
        fall back to this POSITION GROUP's own average age/experience
        within `pos_df` (computed here, not a league-wide average across
        every position) rather than being left NaN, since a rookie WR's
        and a veteran QB's typical age/tenure differ enough that a
        cross-position average would be a worse estimate than no
        adjustment at all.
    """
    merged = pos_df[["player_id", "season", "week"]].merge(player_age, on=["player_id", "season", "week"], how="left")
    merged["player_age"] = merged["age"].fillna(merged["age"].mean())
    merged["player_years_experience"] = merged["years_exp"].fillna(merged["years_exp"].mean())
    return merged[["player_id", "season", "week", "player_age", "player_years_experience"]]


def build_injury_profile(pos_df: pd.DataFrame, injury_reports: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each player-week's official game-status designation as the
    player_injury_status context feature, ordinal-encoded via
    INJURY_STATUS_ORDINAL.

    Args:
        pos_df: Weekly stats for one position group, with at least
            player_id, season, week.
        injury_reports: Output of load_injury_reports().

    Returns:
        DataFrame with one row per (player_id, season, week):
        player_injury_status (0 = no report that week, 1 = Questionable,
        2 = Doubtful, 3 = Out).

    Notes:
        Like build_age_profile()/build_schedule_features(), this is a
        direct known-in-advance fact about the specific game (the report is
        announced days before kickoff), not derived from that game's own
        result -- no leak-safety treatment needed, unlike the trailing
        opp_*/h2h_* team features.

        A "no report" week is encoded as 0, the same value a genuinely
        healthy player gets -- there's no way to distinguish "confirmed
        fine" from "never came up" in this data, but in practice the
        overwhelming majority of no-report player-weeks are simply healthy
        players, so treating them the same is a reasonable simplification.
        See get_weekly_injury_reports()'s Notes for the more important
        asymmetry this can't fully capture: a truly "Out" player mostly
        has no real stat line to train against (they didn't play), so this
        feature's learned signal comes overwhelmingly from Questionable/
        Doubtful players who did play through a designation, not from
        "Out" ever meaningfully predicting a near-zero target in training.
    """
    ordinal = injury_reports.copy()
    ordinal["player_injury_status"] = ordinal["report_status"].map(INJURY_STATUS_ORDINAL)

    merged = pos_df[["player_id", "season", "week"]].merge(
        ordinal[["player_id", "season", "week", "player_injury_status"]],
        on=["player_id", "season", "week"], how="left",
    )
    merged["player_injury_status"] = merged["player_injury_status"].fillna(0).astype(float)
    return merged[["player_id", "season", "week", "player_injury_status"]]


def _opponent_difficulty_raw_column(position_group: str) -> str:
    """Which raw offensive stat a defense's "difficulty" is measured by: rushing yards allowed for RB (run-defense matchups), passing yards allowed for every other position (pass-defense matchups)."""
    return "rushing_yards" if position_group == "RB" else "passing_yards"


def build_opponent_difficulty_profile(
    pos_df: pd.DataFrame,
    defense_profile: pd.DataFrame,
    position_group: str,
    rank_stat: str,
    n_tiers: int = 3,
    window: int = ROLLING_H2H_WINDOW,
) -> pd.DataFrame:
    """
    For each player-week, compute this player's own trailing average
    `rank_stat` production specifically in past games against opponents of
    the SAME difficulty tier as that week's actual opponent -- "does this
    player elevate his game against tough defenses, or struggle against
    them" -- generalizing build_head_to_head_profile()'s exact-opponent
    logic to a TIER of similarly-tough opponents instead of one specific
    team (which a player usually only faces once or twice a season, too
    sparse on its own to say much about how they handle that level of
    competition).

    Args:
        pos_df: Weekly stats for one position group, with at least
            player_id, season, week, opponent_team, and rank_stat.
        defense_profile: Output of build_defense_profile() -- trailing
            opp_pass_yards_allowed/opp_rush_yards_allowed per (season,
            week, team), already leak-safe (shifted).
        position_group: "QB"/"RB"/"WR"/"TE" -- selects which defensive
            metric defines "difficulty" (see _opponent_difficulty_raw_column()).
        rank_stat: Which of the player's own stats to average within each
            tier -- typically POSITION_RANKING_STAT[position_group].
        n_tiers: How many equal-COUNT difficulty tiers to split defenses
            into each week (default 3: toughest/average/easiest).
        window: How many past same-tier meetings to average over (default
            ROLLING_H2H_WINDOW).

    Returns:
        DataFrame with one row per (player_id, season, week):
        player_vs_opponent_difficulty. NaN where a player hasn't yet faced
        this exact tier before, or that week's tiering couldn't be
        computed (e.g. week 1, before any team has defense_profile
        history) -- left for the caller's fillna handling, same as the
        other cold-start cases in this file.

    Notes:
        Tiers are defined by EQUAL COUNT (pd.qcut) of each week's already
        leak-safe trailing defense metric, not a fixed yardage cutoff --
        "toughest tier" always means roughly the toughest third of
        defenses that specific week, so it doesn't drift meaningless as
        league-wide passing/rushing volume shifts year to year.

        Leak-safety: defense_profile's own values are already shifted (a
        defense's tier heading into week W never reflects week W's own
        game), and this player's own trailing rank_stat average within a
        tier is ALSO shifted (only strictly earlier meetings, via the same
        groupby-rolling-shift pattern build_head_to_head_profile() uses,
        just grouped by (player_id, difficulty_tier) instead of (team,
        opponent_team)) -- so nothing about a game's own result leaks into
        its own "how do I play against defenses this tough" feature.
    """
    metric_col = "opp_rush_yards_allowed" if position_group == "RB" else "opp_pass_yards_allowed"

    def _tier(s: pd.Series) -> pd.Series:
        if s.nunique() < 2:
            return pd.Series(0, index=s.index)
        return pd.qcut(s, min(n_tiers, s.nunique()), labels=False, duplicates="drop")

    tiers = defense_profile[["season", "week", "team", metric_col]].dropna(subset=[metric_col]).copy()
    tiers["difficulty_tier"] = tiers.groupby(["season", "week"])[metric_col].transform(_tier)
    tiers = tiers.rename(columns={"team": "opponent_team"})[["season", "week", "opponent_team", "difficulty_tier"]]

    tagged = pos_df[["player_id", "season", "week", "opponent_team", rank_stat]].merge(
        tiers, on=["season", "week", "opponent_team"], how="left"
    )
    tagged = tagged.sort_values(["player_id", "difficulty_tier", "season", "week"])
    tagged["player_vs_opponent_difficulty"] = tagged.groupby(["player_id", "difficulty_tier"])[rank_stat].transform(
        lambda s: s.rolling(window, min_periods=1).mean().shift(1)
    )

    return pos_df[["player_id", "season", "week"]].merge(
        tagged[["player_id", "season", "week", "player_vs_opponent_difficulty"]],
        on=["player_id", "season", "week"], how="left",
    )


def compute_current_opponent_tiers(
    team_games: pd.DataFrame, position_group: str, window: int = ROLLING_DEFENSE_WINDOW, n_tiers: int = 3
) -> pd.Series:
    """
    Rank every team's CURRENT trailing defensive strength into difficulty
    tiers -- the live counterpart to build_opponent_difficulty_profile()'s
    per-week leak-safe tiering, used by current_matchup_context() for the
    live player_vs_opponent_difficulty lookup. There's no "as of week W"
    question here since this is always "right now" -- the same
    relationship the rest of current_matchup_context() already has to
    build_defense_profile() (unshifted "most recent games," not a
    per-historical-week reconstruction).

    Args:
        team_games: Output of build_team_game_log().
        position_group: Selects which raw stat defines "difficulty" (see
            _opponent_difficulty_raw_column()).
        window: Games of trailing history per team (default ROLLING_DEFENSE_WINDOW).
        n_tiers: How many equal-count tiers (default 3).

    Returns:
        Series indexed by team, valued by difficulty tier (0 = toughest).
        Teams with no tracked games yet are simply absent.
    """
    raw_col = _opponent_difficulty_raw_column(position_group)
    current_defense = team_games.groupby("opponent_team").apply(
        lambda g: g.sort_values(["season", "week"]).tail(window)[raw_col].mean()
    ).dropna()

    if current_defense.nunique() < 2:
        return pd.Series(0, index=current_defense.index)
    return pd.qcut(current_defense, min(n_tiers, current_defense.nunique()), labels=False, duplicates="drop")


def build_archetype_context(
    pos_df: pd.DataFrame, archetypes: pd.DataFrame, defense_vs_archetype: pd.DataFrame, position_group: str
) -> pd.DataFrame:
    """
    Attach each player-week's own archetype (ordinal-encoded) AND the
    upcoming opponent's trailing performance against that specific
    archetype, as the player_archetype / opponent_defense_vs_archetype
    context features -- see PlayerArchetypes.py.

    Args:
        pos_df: Weekly stats for one position group, with at least
            player_id, season, week, opponent_team.
        archetypes: Output of PlayerArchetypes.classify_archetype(df, position_group)
            (or compute_all_archetypes(df)[position_group]) -- this
            player's own archetype over time.
        defense_vs_archetype: Output of PlayerArchetypes.build_defense_vs_archetype_profile(
            df, archetypes, position_group) (or compute_all_defense_vs_archetype_profiles(...)[position_group])
            -- the upcoming opponent's trailing performance by archetype.
        position_group: "QB"/"RB"/"WR"/"TE" -- selects the ordinal encoding
            (PlayerArchetypes.ARCHETYPE_CENTROIDS[position_group]'s key
            order) and which defense_vs_archetype metric column to read
            (PlayerArchetypes.ARCHETYPE_METRIC_COLUMN[position_group]).

    Returns:
        DataFrame with one row per (player_id, season, week):
        player_archetype (ordinal), opponent_defense_vs_archetype.

    Notes:
        No leak-safety treatment needed here beyond what archetypes/
        defense_vs_archetype already carry -- both are already trailing/
        shifted (see PlayerArchetypes.compute_style_features()'s and
        build_defense_vs_archetype_profile()'s Notes), so this is a
        straightforward join, same as build_injury_profile()'s pattern.

        Fallback for a player-week with no archetype yet (not enough
        trailing volume -- see PlayerArchetypes.MIN_TRAILING_VOLUME) or an
        opponent with no trailing performance yet against that archetype:
        the position's own mean ordinal / mean metric value within
        `pos_df`'s matched rows, same "league-average, no info yet"
        convention as build_age_profile().
    """
    from PlayerArchetypes import ARCHETYPE_CENTROIDS, ARCHETYPE_METRIC_COLUMN

    ordinal_map = {name: i for i, name in enumerate(ARCHETYPE_CENTROIDS[position_group])}
    metric_col = ARCHETYPE_METRIC_COLUMN[position_group]

    merged = pos_df[["player_id", "season", "week", "opponent_team"]].merge(
        archetypes[["player_id", "season", "week", "archetype"]], on=["player_id", "season", "week"], how="left"
    )
    merged["player_archetype"] = merged["archetype"].map(ordinal_map)
    merged["player_archetype"] = merged["player_archetype"].fillna(merged["player_archetype"].mean())

    merged = merged.merge(
        defense_vs_archetype.rename(columns={"team": "opponent_team"}),
        on=["season", "week", "opponent_team", "archetype"], how="left",
    )
    merged["opponent_defense_vs_archetype"] = merged[metric_col].fillna(merged[metric_col].mean())

    return merged[["player_id", "season", "week", "player_archetype", "opponent_defense_vs_archetype"]]


def build_schedule_features(schedule_context: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the "direct fact" team-level context features from schedule
    data: is_home, rest_days, opponent_rest_days, is_divisional_game,
    game_temp, game_wind.

    Args:
        schedule_context: Output of load_schedule_context().

    Returns:
        DataFrame with one row per (season, week, team): is_home,
        rest_days, opponent_rest_days, is_divisional_game, game_temp,
        game_wind.

    Notes:
        UNLIKE every other *_profile()/*_context() builder in this file,
        this one needs no leak-safety treatment (no rolling/shift) -- these
        columns are already known, fixed facts about the specific upcoming
        game itself (who's home, how much rest each team had, the weather),
        not derived from past performance that could leak a future result.
        The one adjustment made here is filling temp/wind for dome games
        with DOME_TEMP_FALLBACK/DOME_WIND_FALLBACK (a known "no weather
        effect" constant) rather than leaving them NaN or using a raw
        league-average fill -- a dome's condition is known, not missing.
    """
    sc = schedule_context.copy()
    is_dome = ~sc["roof"].isin(["outdoors", "open"])
    sc["game_temp"] = sc["temp"].where(~is_dome, DOME_TEMP_FALLBACK)
    sc["game_wind"] = sc["wind"].where(~is_dome, DOME_WIND_FALLBACK)
    sc["is_divisional_game"] = sc["div_game"].astype(float)
    sc["is_home"] = sc["is_home"].astype(float)
    # NaN when no line is posted yet; fillna_context() fills it like any other gap.
    sc["implied_team_total"] = (sc["total_line"] + sc["team_spread_line"]) / 2

    return sc[["season", "week", "team", "is_home", "rest_days", "opponent_rest_days", "is_divisional_game", "game_temp", "game_wind",
               "implied_team_total", "team_spread_line"]]


def build_qb_continuity_profile(schedule_context: pd.DataFrame, window: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    For each (season, week, team), count how many of that team's PAST games
    (up to `window`) were started by the SAME quarterback who's starting
    this week -- 0 means a brand-new starter (e.g. a backup stepping in for
    an injured starter), up to `window` means a fully stable starter.

    Args:
        schedule_context: Output of load_schedule_context() -- needs
            starting_qb_id.
        window: How many past games to look back over (default WINDOW_SIZE
            -- matches the player-history sequence length, so this feature
            covers the same span of "recent context" the LSTM sequence does).

    Returns:
        DataFrame with one row per (season, week, team): games_with_current_qb.

    Notes:
        Leakage-safe in the relevant sense: this week's OWN starting QB
        identity is legitimate to use directly (it's public knowledge before
        kickoff, not the game's outcome), but the comparison against PAST
        games only ever looks backward (index i-1, i-2, ... within `window`),
        so no future game's starter can affect a game's own continuity count.
    """
    sc = schedule_context.sort_values(["team", "season", "week"]).reset_index(drop=True)

    def _consecutive_same_qb(group: pd.DataFrame) -> pd.Series:
        qb_ids = group["starting_qb_id"].to_numpy()
        counts = np.zeros(len(qb_ids), dtype=np.float32)
        for i in range(len(qb_ids)):
            count = 0
            for j in range(i - 1, max(-1, i - 1 - window), -1):
                if qb_ids[j] == qb_ids[i] and pd.notna(qb_ids[i]):
                    count += 1
                else:
                    break
            counts[i] = count
        return pd.Series(counts, index=group.index)

    sc["games_with_current_qb"] = sc.groupby("team", group_keys=False).apply(_consecutive_same_qb)
    return sc[["season", "week", "team", "games_with_current_qb"]]


def build_team_pace_profile(df: pd.DataFrame, window: int = ROLLING_DEFENSE_WINDOW) -> pd.DataFrame:
    """
    Trailing average offensive plays (pass attempts + rush carries) per
    team per game -- a pace/ball-control proxy: a team with fewer plays per
    game is sustaining fewer drives (e.g. frequent 3-and-outs) or facing
    opponents who eat clock, either way limiting every one of their own
    players' opportunities that game.

    Args:
        df: Weekly player stats (all positions -- attempts/carries are 0
            for players who don't record them, so a plain team sum works).
        window: Games of trailing history to average (default ROLLING_DEFENSE_WINDOW).

    Returns:
        DataFrame with one row per (season, week, team): team_plays_per_game.
        NaN for a team's very first tracked game (no prior games to
        average) -- left for the caller's fillna handling.

    Notes:
        Leakage prevention: same `.rolling(window).mean().shift(1)` pattern
        as build_defense_profile() -- a team's pace heading into week W
        only reflects games strictly before W.

        This is a lighter proxy for "ball control" than literal drive-level
        data (e.g. 3-and-out rate, time of possession) would be -- that
        needs play-by-play/drive data, a much heavier pull (see
        get_team_season_stats_nflverse()'s Notes on nfl.import_pbp_data()'s
        size). Plays run per game is derivable from data this project
        already pulls and correlates strongly with time of possession.
    """
    team_plays = df.groupby(["season", "week", "team"], as_index=False)[["attempts", "carries"]].sum(min_count=1)
    team_plays["plays"] = team_plays["attempts"].fillna(0) + team_plays["carries"].fillna(0)
    team_plays = team_plays.sort_values(["team", "season", "week"])
    team_plays["team_plays_per_game"] = team_plays.groupby("team")["plays"].transform(
        lambda s: s.rolling(window, min_periods=1).mean().shift(1)
    )
    return team_plays[["season", "week", "team", "team_plays_per_game"]]


def build_team_takeaways_profile(df: pd.DataFrame, window: int = ROLLING_DEFENSE_WINDOW) -> pd.DataFrame:
    """
    Trailing average takeaways (interceptions + recovered opponent
    fumbles) generated by a team's DEFENSE per game -- a defense that
    creates a lot of turnovers is a real added risk (and short-field/
    extra-possession swing for them) beyond what yards-allowed alone
    (build_defense_profile()) captures.

    Args:
        df: Weekly player stats (all positions -- def_interceptions/
            fumble_recovery_opp are 0 for offensive players).
        window: Games of trailing history to average (default ROLLING_DEFENSE_WINDOW).

    Returns:
        DataFrame with one row per (season, week, team): team_takeaways_per_game.
        NaN for a team's very first tracked game -- left for the caller's
        fillna handling.

    Notes:
        Same leakage-prevention shape as build_defense_profile()/
        build_team_pace_profile() -- shifted by 1 so a game's own takeaways
        never leak into its own "how turnover-prone is this defense"
        feature heading into that same game.
    """
    takeaways = df.groupby(["season", "week", "team"], as_index=False)[["def_interceptions", "fumble_recovery_opp"]].sum(min_count=1)
    takeaways["takeaways"] = takeaways["def_interceptions"].fillna(0) + takeaways["fumble_recovery_opp"].fillna(0)
    takeaways = takeaways.sort_values(["team", "season", "week"])
    takeaways["team_takeaways_per_game"] = takeaways.groupby("team")["takeaways"].transform(
        lambda s: s.rolling(window, min_periods=1).mean().shift(1)
    )
    return takeaways[["season", "week", "team", "team_takeaways_per_game"]]


def build_ol_injury_profile(injury_reports: pd.DataFrame, player_bio: pd.DataFrame) -> pd.DataFrame:
    """
    Count of a team's own offensive linemen (see OL_POSITIONS) with a
    game-status designation each week -- protection/run-blocking health,
    distinct from any single skill-position player's own player_injury_status.

    Args:
        injury_reports: Output of load_injury_reports().
        player_bio: Output of load_player_bio() -- position lookup.

    Returns:
        DataFrame with one row per (season, week, team) that had at least
        one O-line injury report that week: team_ol_injuries (count).
        Weeks with none are simply absent -- the caller fills those with 0
        (no O-line injury concern that week), same as
        build_injury_profile()'s "no report" convention.

    Notes:
        Like build_injury_profile(), this is a direct known-in-advance fact
        about the specific game (the report is announced days before
        kickoff), not derived from that game's own result -- no
        leak-safety treatment (rolling/shift) needed.
    """
    ol_ids = set(player_bio.loc[player_bio["position"].isin(OL_POSITIONS), "player_id"])
    ol_injuries = injury_reports[injury_reports["player_id"].isin(ol_ids)]
    return ol_injuries.groupby(["season", "week", "team"]).size().reset_index(name="team_ol_injuries")


def build_matchup_context(
    df: pd.DataFrame, top_cb_profile: pd.DataFrame, schedule_context: pd.DataFrame,
    injury_reports: pd.DataFrame, player_bio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full per-(season, week, team) TEAM-level matchup context
    table: opponent defensive strength, opponent's best cornerback's
    trailing rating, historical head-to-head tendency against that week's
    specific opponent, and the schedule/situational facts (home/away, rest,
    divisional game, weather, QB continuity) -- the TEAM_CONTEXT_COLUMNS
    this file conditions predictions on. The PLAYER_CONTEXT_COLUMNS
    (player_team_position_rank, size_diff_vs_top_cb_*) are built separately
    since they vary per player, not per team -- see
    build_player_team_rank_profile()/build_size_diff_profile().

    Args:
        df: Weekly player stats to derive team-vs-team offensive game logs
            from (e.g. load_weekly_stats()'s output, or a chronologically-
            filtered slice of it for walk-forward training -- see
            walk_forward_train_and_predict()).
        top_cb_profile: Output of build_top_cb_profile() -- provides
            opp_top_cb_coverage_score. Built separately by the caller (not
            internally here) because it needs load_defensive_target_stats()/
            load_player_bio() data, not just `df` -- and the caller
            typically also needs top_cb_profile on its own for
            build_size_diff_profile(), so building it once and passing it
            to both avoids redundant computation.
        schedule_context: Output of load_schedule_context() -- provides
            is_home/rest_days/opponent_rest_days/is_divisional_game/
            game_temp/game_wind/games_with_current_qb.
        injury_reports: Output of load_injury_reports() -- provides
            team_ol_injuries (see build_ol_injury_profile()).
        player_bio: Output of load_player_bio() -- position lookup for
            build_ol_injury_profile().

    Returns:
        DataFrame with one row per (season, week, team): all
        TEAM_CONTEXT_COLUMNS, still possibly containing NaN for cold-start
        cases (see fillna_context() to resolve those before use).
    """
    team_games = build_team_game_log(df)
    defense = build_defense_profile(team_games)
    h2h = build_head_to_head_profile(team_games)
    top_cb_score = top_cb_profile[["season", "week", "team", "opp_top_cb_coverage_score"]]
    pace = build_team_pace_profile(df)
    takeaways = build_team_takeaways_profile(df)
    ol_injuries = build_ol_injury_profile(injury_reports, player_bio)

    # IMPORTANT: `defense`, `top_cb_score`, and `takeaways` are each keyed
    # by (season, week, team) representing THAT team's own defense/best-CB/
    # takeaways. A context row for team=X, opponent_team=Y needs Y's
    # defense/CB/takeaways attached, not X's -- so these merges join h2h's
    # opponent_team against the other tables' team column (not team-to-team,
    # which would silently attach each team's own defensive stats to itself
    # instead of their upcoming opponent's -- exactly the bug this project
    # shipped and then caught). `pace` gets merged BOTH ways further down --
    # opponent_plays_per_game here (flipped, describes the opponent), and
    # team_plays_per_game below (direct, describes this row's own team).
    context = h2h.merge(
        defense, left_on=["season", "week", "opponent_team"], right_on=["season", "week", "team"],
        how="left", suffixes=("", "_defense_side"),
    ).drop(columns=["team_defense_side"])
    context = context.merge(
        top_cb_score, left_on=["season", "week", "opponent_team"], right_on=["season", "week", "team"],
        how="left", suffixes=("", "_cb_side"),
    ).drop(columns=["team_cb_side"])
    context = context.merge(
        pace.rename(columns={"team_plays_per_game": "opponent_plays_per_game"}),
        left_on=["season", "week", "opponent_team"], right_on=["season", "week", "team"],
        how="left", suffixes=("", "_pace_side"),
    ).drop(columns=["team_pace_side"])
    context = context.merge(
        takeaways.rename(columns={"team_takeaways_per_game": "opponent_takeaways_per_game"}),
        left_on=["season", "week", "opponent_team"], right_on=["season", "week", "team"],
        how="left", suffixes=("", "_takeaway_side"),
    ).drop(columns=["team_takeaway_side"])

    # UNLIKE the merges above, schedule/QB-continuity/pace(own)/OL-injury
    # features describe the row's OWN team (is THIS team home, how much
    # rest did THIS team get, how many plays does THIS team's own offense
    # run, are THIS team's own linemen hurt), not the opponent -- so these
    # join directly on (season, week, team), no flip. Getting this backwards
    # would be the opposite mistake from the defense/CB bug: it would happen
    # to still "work" in the sense of producing a valid-looking number, just
    # the WRONG team's home/rest/weather/pace/OL-health status, which is a
    # much subtler bug to notice by inspection -- verified explicitly
    # against real data before relying on it.
    schedule_features = build_schedule_features(schedule_context)
    qb_continuity = build_qb_continuity_profile(schedule_context)
    context = context.merge(schedule_features, on=["season", "week", "team"], how="left")
    context = context.merge(qb_continuity, on=["season", "week", "team"], how="left")
    context = context.merge(pace, on=["season", "week", "team"], how="left")
    context = context.merge(ol_injuries, on=["season", "week", "team"], how="left")
    context["team_ol_injuries"] = context["team_ol_injuries"].fillna(0)

    return context[["season", "week", "team", "opponent_team"] + TEAM_CONTEXT_COLUMNS]


def fillna_context(context_df: pd.DataFrame, columns: List[str] = TEAM_CONTEXT_COLUMNS) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fill missing matchup-context values (a team's first tracked game, or a
    pair of teams with no prior meeting in the data) with that column's
    overall average, as a neutral "no information yet, assume league
    average" prior.

    Args:
        context_df: Output of build_matchup_context().
        columns: Which columns to fill (default CONTEXT_COLUMNS).

    Returns:
        (filled_df, fill_values): the context table with NaNs replaced, and
        the dict of fill constants used (saved alongside a trained model so
        the exact same constants can be reused at prediction time for a
        matchup with no history -- see save_model()/current_matchup_context()).

    Notes:
        Minor simplification: the fill CONSTANT is estimated from the full
        context table passed in, not a strict train-only slice (train/val
        split happens later, per-position, inside train_position_model() --
        context is built once up front and shared across positions). This
        only affects rows with zero real signal to begin with (no prior
        games/meetings at all), and what "leaks" is a single coarse
        league-wide average across the whole date range, not any specific
        game's outcome -- a reasonable trade against fitting this twice per
        position group.
    """
    fill_values = context_df[columns].mean().to_dict()
    filled = context_df.copy()
    filled[columns] = filled[columns].fillna(fill_values)
    return filled, fill_values


def current_matchup_context(
    df: pd.DataFrame,
    team_games: pd.DataFrame,
    defensive_target_stats: pd.DataFrame,
    player_bio: pd.DataFrame,
    schedule_context: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    archetypes: pd.DataFrame,
    defense_vs_archetype: pd.DataFrame,
    player_id: str,
    position_group: str,
    team: str,
    opponent: str,
    context_fill_values: Dict[str, float],
    injury_status: Optional[str] = None,
    opponent_tier_labels: Optional[pd.Series] = None,
    defense_window: int = ROLLING_DEFENSE_WINDOW,
    h2h_window: int = ROLLING_H2H_WINDOW,
    min_targets: int = MIN_TRAILING_CB_TARGETS,
    qb_window: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Compute a LIVE matchup-context vector for an upcoming game that isn't in
    the data yet: `team` (with `player_id` on it, playing `position_group`)
    is about to play `opponent`. Used by predict_next_game() and
    walk_forward_train_and_predict().

    Args:
        df: Weekly player stats (needed for this player's own team-position
            rank, which team_games' team-vs-team aggregation doesn't retain).
        team_games: Output of build_team_game_log() -- ideally built only
            from games known to have happened before this prediction point
            (see the callers for how they scope this); should be built from
            the same rows as `df`.
        defensive_target_stats: Output of load_defensive_target_stats() (or
            a chronologically-filtered slice, for walk-forward) -- used to
            identify the opponent's best CB with real targeted-coverage data.
        player_bio: Output of load_player_bio() -- height/weight/position lookup.
        schedule_context: Output of load_schedule_context() (or a
            chronologically-filtered slice) -- used to look up the specific
            upcoming `team` vs `opponent` game's home/away, rest, weather,
            and starting QB, and that team's recent starting-QB history.
        player_age: Output of load_player_age() -- used for player_age and
            player_years_experience (see build_age_profile() for the
            leak-safety/fallback reasoning, which applies the same way here).
        injury_reports: Output of load_injury_reports() -- used for
            player_injury_status when `injury_status` isn't given explicitly.
        archetypes: This position's entry from
            PlayerArchetypes.compute_all_archetypes(df) -- used for
            player_archetype (this player's most recent classification).
        defense_vs_archetype: This position's entry from
            PlayerArchetypes.compute_all_defense_vs_archetype_profiles(df, ...)
            -- used for opponent_defense_vs_archetype.
        player_id: The player being predicted.
        position_group: Their position group ("QB"/"RB"/"WR"/"TE") -- used
            to rank them among their own team's players at that position.
        team: The player's team.
        opponent: The team they're about to face.
        context_fill_values: Fallback constants from fillna_context(), used
            when there's no relevant history at all (a new team's first
            game, two teams that have never played each other before, or an
            opponent with no CB reaching min_targets trailing targets yet).
        injury_status: Optional explicit override for this week's game-status
            designation ("Questionable"/"Doubtful"/"Out"), for when you
            already know it (or know it changed) and don't want to rely on
            whatever's in `injury_reports` -- e.g. the report came out after
            `injury_reports` was loaded. Pass None (default) to look it up
            from `injury_reports` instead; if nothing's found there either
            (no report yet, or a report that predates the game being
            predicted), player_injury_status is 0 ("no reported concern").
        opponent_tier_labels: Precomputed output of
            compute_current_opponent_tiers() for this position_group, used
            for player_vs_opponent_difficulty. Pass this in (rather than
            leaving it None) whenever calling current_matchup_context() in
            a loop over many players for the same week -- it's the same
            league-wide computation for every one of them, so computing it
            once outside the loop avoids redoing it per player. Left as
            None to compute it internally (fine for a single one-off call,
            e.g. predict_next_game()'s normal use).
        defense_window, h2h_window, min_targets: Same meaning as in
            build_defense_profile()/build_head_to_head_profile()/build_top_cb_profile().

    Returns:
        1D float32 array in CONTEXT_COLUMNS order.

    Notes:
        Unlike build_defense_profile()/build_head_to_head_profile()/
        build_top_cb_profile(), the team-level values here do NOT shift(1)
        -- there's no "current game" to exclude here, since the game being
        predicted hasn't happened yet and isn't in `team_games`/`df`/
        `defensive_target_stats` at all. This simply aggregates the most
        recent games directly. player_team_position_rank never needs a
        fallback constant -- see build_player_team_rank_profile()'s Notes.
        size_diff_vs_top_cb_* fall back to 0.0 (no known mismatch) rather
        than context_fill_values, matching build_size_diff_profile()'s
        convention that 0 is the natural "no information" value for a
        difference, not a league average. The schedule/QB-continuity block
        looks up the ACTUAL upcoming team-vs-opponent game in
        schedule_context by identity (not by inferring "next week" from the
        player's last game), so it's correct even across a bye week; if
        that exact matchup isn't found there at all, every schedule-derived
        value falls back to context_fill_values like the other cold-start
        cases. player_age and player_injury_status are looked up
        independently of whether that upcoming-game lookup succeeds -- see
        their own blocks below.
    """
    opp_allowed = team_games[team_games["opponent_team"] == opponent].sort_values(["season", "week"]).tail(defense_window)
    if len(opp_allowed):
        opp_pass_yards_allowed = float(opp_allowed["passing_yards"].mean())
        opp_pass_tds_allowed = float(opp_allowed["passing_tds"].mean())
        opp_rush_yards_allowed = float(opp_allowed["rushing_yards"].mean())
        opp_rush_tds_allowed = float(opp_allowed["rushing_tds"].mean())
    else:
        opp_pass_yards_allowed = context_fill_values["opp_pass_yards_allowed"]
        opp_pass_tds_allowed = context_fill_values["opp_pass_tds_allowed"]
        opp_rush_yards_allowed = context_fill_values["opp_rush_yards_allowed"]
        opp_rush_tds_allowed = context_fill_values["opp_rush_tds_allowed"]

    # Opponent's best CB right now: real targeted-coverage stats, aggregated
    # over their last `defense_window` games, requiring at least min_targets
    # trailing targets to trust -- same logic as build_top_cb_profile(), just
    # without the shift(1) (nothing to exclude; this game hasn't happened).
    cb_ids = set(player_bio.loc[player_bio["position"] == "CB", "player_id"])
    opp_cb_stats = defensive_target_stats[
        (defensive_target_stats["team"] == opponent) & (defensive_target_stats["player_id"].isin(cb_ids))
    ]
    top_cb_player_id = None
    if len(opp_cb_stats):
        recent = opp_cb_stats.sort_values(["player_id", "season", "week"]).groupby("player_id").tail(defense_window)
        totals = recent.groupby("player_id").agg(
            targets=("def_targets", "sum"), completions=("def_completions_allowed", "sum"),
            yards=("def_yards_allowed", "sum"), tds=("def_receiving_td_allowed", "sum"), ints=("def_ints", "sum"),
        )
        totals = totals[totals["targets"] >= min_targets]
        if len(totals):
            totals["rating"] = nfl_passer_rating(totals["completions"], totals["yards"], totals["tds"], totals["ints"], totals["targets"])
            top_cb_player_id = totals["rating"].idxmin()
            opp_top_cb_coverage_score = float(totals.loc[top_cb_player_id, "rating"])
        else:
            opp_top_cb_coverage_score = context_fill_values["opp_top_cb_coverage_score"]
    else:
        opp_top_cb_coverage_score = context_fill_values["opp_top_cb_coverage_score"]

    h2h_games = team_games[(team_games["team"] == team) & (team_games["opponent_team"] == opponent)] \
        .sort_values(["season", "week"]).tail(h2h_window)
    if len(h2h_games):
        h2h_pass_yards = float(h2h_games["passing_yards"].mean())
        h2h_pass_tds = float(h2h_games["passing_tds"].mean())
        h2h_rush_yards = float(h2h_games["rushing_yards"].mean())
        h2h_rush_tds = float(h2h_games["rushing_tds"].mean())
    else:
        h2h_pass_yards = context_fill_values["h2h_pass_yards"]
        h2h_pass_tds = context_fill_values["h2h_pass_tds"]
        h2h_rush_yards = context_fill_values["h2h_rush_yards"]
        h2h_rush_tds = context_fill_values["h2h_rush_tds"]

    # Pace -- this team's own and the opponent's trailing average offensive
    # plays per game (see build_team_pace_profile()). Computed directly
    # from `df` rather than `team_games` since team_games doesn't carry
    # attempts/carries.
    own_plays = df[df["team"] == team].groupby(["season", "week"])[["attempts", "carries"]].sum(min_count=1).sum(axis=1)
    team_plays_per_game = (
        float(own_plays.tail(defense_window).mean()) if len(own_plays) else context_fill_values.get("team_plays_per_game", 0.0)
    )
    opp_plays = df[df["team"] == opponent].groupby(["season", "week"])[["attempts", "carries"]].sum(min_count=1).sum(axis=1)
    opponent_plays_per_game = (
        float(opp_plays.tail(defense_window).mean()) if len(opp_plays) else context_fill_values.get("opponent_plays_per_game", 0.0)
    )

    # Opponent takeaways -- opponent's DEFENSE's trailing average takeaways
    # per game (see build_team_takeaways_profile()).
    opp_takeaways = df[df["team"] == opponent].groupby(["season", "week"])[["def_interceptions", "fumble_recovery_opp"]].sum(min_count=1).sum(axis=1)
    opponent_takeaways_per_game = (
        float(opp_takeaways.tail(defense_window).mean()) if len(opp_takeaways) else context_fill_values.get("opponent_takeaways_per_game", 0.0)
    )

    # This player's rank on their own team at their position, right now:
    # rank teammates in the same position_group by season-to-date total of
    # the position's primary stat, using every game played so far.
    teammates = df[(df["team"] == team) & (df["position_group"] == position_group) & (df["season"] == df["season"].max())]
    rank_stat = POSITION_RANKING_STAT[position_group]
    totals = teammates.groupby("player_id")[rank_stat].sum().sort_values(ascending=False)
    if player_id in totals.index:
        player_team_position_rank = float(totals.index.get_loc(player_id) + 1)
    else:
        player_team_position_rank = float(len(totals) + 1)  # no games yet this season: assume bottom of the depth chart

    # Size differential vs. that identified top CB (0.0 = no known mismatch,
    # if no qualifying CB was found or bio data is missing for either player).
    bio = player_bio.drop_duplicates(subset=["player_id"]).set_index("player_id")[["height", "weight"]]
    size_diff_vs_top_cb_height = 0.0
    size_diff_vs_top_cb_weight = 0.0
    if top_cb_player_id is not None and player_id in bio.index and top_cb_player_id in bio.index:
        own_height, own_weight = bio.loc[player_id, ["height", "weight"]]
        cb_height, cb_weight = bio.loc[top_cb_player_id, ["height", "weight"]]
        if pd.notna(own_height) and pd.notna(cb_height):
            size_diff_vs_top_cb_height = float(own_height - cb_height)
        if pd.notna(own_weight) and pd.notna(cb_weight):
            size_diff_vs_top_cb_weight = float(own_weight - cb_weight)

    # Age / experience -- this player's most recent known roster-snapshot
    # values (off by at most a few weeks vs. literally right now; negligible).
    player_age_rows = player_age[player_age["player_id"] == player_id].sort_values(["season", "week"])
    if len(player_age_rows):
        current_player_age = float(player_age_rows["age"].iloc[-1])
        current_years_experience = float(player_age_rows["years_exp"].iloc[-1])
    else:
        # Cold start: no roster record for this player at all -- fall back
        # to the league-wide average, same spirit as build_age_profile()'s
        # per-position fallback (a single live player lookup has no pos_df
        # to average within, so this uses the whole table instead).
        current_player_age = float(player_age["age"].mean())
        current_years_experience = float(player_age["years_exp"].mean())

    # Opponent difficulty -- this player's own trailing average rank_stat
    # against defenses as tough as `opponent` currently is (see
    # build_opponent_difficulty_profile()'s live counterpart,
    # compute_current_opponent_tiers()).
    if opponent_tier_labels is None:
        opponent_tier_labels = compute_current_opponent_tiers(team_games, position_group, defense_window)
    opponent_tier = opponent_tier_labels.get(opponent)
    if opponent_tier is not None and pd.notna(opponent_tier):
        own_games = df[df["player_id"] == player_id][["season", "week", "opponent_team", rank_stat]].sort_values(["season", "week"])
        own_games_vs_tier = own_games[own_games["opponent_team"].map(opponent_tier_labels) == opponent_tier]
        recent_vs_tier = own_games_vs_tier.tail(h2h_window)
        if len(recent_vs_tier):
            player_vs_opponent_difficulty = float(recent_vs_tier[rank_stat].mean())
        else:
            # This player hasn't yet faced a defense in this tier -- fall
            # back to the league-wide average of rank_stat, same "no
            # history yet" spirit as every other cold-start case here.
            player_vs_opponent_difficulty = float(df[rank_stat].mean())
    else:
        # Opponent's tier couldn't be determined (e.g. too early in the
        # season for any team to have defense history yet).
        player_vs_opponent_difficulty = float(df[rank_stat].mean())

    # Archetype -- this player's most recent classification, and the
    # opponent's trailing performance specifically against that archetype
    # (see PlayerArchetypes.py / build_archetype_context()).
    from PlayerArchetypes import ARCHETYPE_CENTROIDS, ARCHETYPE_METRIC_COLUMN

    ordinal_map = {name: i for i, name in enumerate(ARCHETYPE_CENTROIDS[position_group])}
    own_archetype_rows = archetypes[archetypes["player_id"] == player_id].sort_values(["season", "week"])
    if len(own_archetype_rows):
        current_archetype = own_archetype_rows["archetype"].iloc[-1]
        current_player_archetype = float(ordinal_map[current_archetype])
    else:
        # Cold start: not enough trailing volume yet to classify this
        # player -- fall back to the mean ordinal across archetypes (no
        # particular archetype implied).
        current_archetype = None
        current_player_archetype = float(np.mean(list(ordinal_map.values())))

    metric_col = ARCHETYPE_METRIC_COLUMN[position_group]
    opp_defense_vs_archetype_rows = pd.DataFrame()
    if current_archetype is not None:
        opp_defense_vs_archetype_rows = defense_vs_archetype[
            (defense_vs_archetype["team"] == opponent) & (defense_vs_archetype["archetype"] == current_archetype)
        ].sort_values(["season", "week"])
    if len(opp_defense_vs_archetype_rows):
        opponent_defense_vs_archetype = float(opp_defense_vs_archetype_rows[metric_col].iloc[-1])
    else:
        # No trailing meetings between this opponent and this archetype yet
        # (or the player's own archetype is unknown) -- league-average fallback.
        opponent_defense_vs_archetype = float(defense_vs_archetype[metric_col].mean())

    # Injury status -- an explicit override always wins; otherwise looked up
    # below from injury_reports once the upcoming game's (season, week) is
    # known. Defaults to 0 ("no reported concern") if neither is available.
    current_injury_status = float(INJURY_STATUS_ORDINAL.get(injury_status, 0)) if injury_status is not None else None

    # O-line health -- count of this team's OWN O-line players (T/G/C) with
    # a game-status designation for the same upcoming (season, week) as
    # injury_status above -- resolved once game_row is known, same pattern.
    ol_ids = set(player_bio.loc[player_bio["position"].isin(OL_POSITIONS), "player_id"])
    team_ol_injuries: Optional[float] = None

    # Schedule/situational facts for the SPECIFIC upcoming team-vs-opponent
    # game: find it by (team, opponent), restricted to strictly after this
    # player's most recent known game (so a rematch later in the season
    # doesn't get confused with an earlier meeting already in the data).
    last_known = df[df["player_id"] == player_id][["season", "week"]].sort_values(["season", "week"])
    upcoming = schedule_context[(schedule_context["team"] == team) & (schedule_context["opponent"] == opponent)]
    if len(last_known) and len(upcoming):
        last_season, last_week = last_known.iloc[-1][["season", "week"]]
        upcoming = upcoming[
            (upcoming["season"] > last_season) | ((upcoming["season"] == last_season) & (upcoming["week"] > last_week))
        ].sort_values(["season", "week"])
    else:
        upcoming = upcoming.iloc[0:0]

    if len(upcoming):
        game_row = upcoming.iloc[0]
        is_home = float(game_row["is_home"])
        rest_days = float(game_row["rest_days"]) if pd.notna(game_row["rest_days"]) else context_fill_values["rest_days"]
        opponent_rest_days = float(game_row["opponent_rest_days"]) if pd.notna(game_row["opponent_rest_days"]) else context_fill_values["opponent_rest_days"]
        is_divisional_game = float(game_row["div_game"])
        is_dome = game_row["roof"] not in ("outdoors", "open")
        game_temp = DOME_TEMP_FALLBACK if is_dome else (float(game_row["temp"]) if pd.notna(game_row["temp"]) else context_fill_values["game_temp"])
        game_wind = DOME_WIND_FALLBACK if is_dome else (float(game_row["wind"]) if pd.notna(game_row["wind"]) else context_fill_values["game_wind"])
        # .get: a model saved before these features existed has no fill value for them.
        has_line = pd.notna(game_row.get("total_line")) and pd.notna(game_row.get("team_spread_line"))
        implied_team_total = (float(game_row["total_line"]) + float(game_row["team_spread_line"])) / 2 if has_line \
            else context_fill_values.get("implied_team_total", 0.0)
        team_spread_line = float(game_row["team_spread_line"]) if has_line else context_fill_values.get("team_spread_line", 0.0)

        current_qb = game_row["starting_qb_id"]
        team_history = schedule_context[
            (schedule_context["team"] == team)
            & ((schedule_context["season"] < game_row["season"])
               | ((schedule_context["season"] == game_row["season"]) & (schedule_context["week"] < game_row["week"])))
        ].sort_values(["season", "week"]).tail(qb_window)
        if pd.notna(current_qb) and len(team_history):
            consecutive = 0
            for qb in team_history["starting_qb_id"].to_numpy()[::-1]:
                if qb == current_qb:
                    consecutive += 1
                else:
                    break
            games_with_current_qb = float(consecutive)
        else:
            games_with_current_qb = context_fill_values["games_with_current_qb"]

        if current_injury_status is None:
            report_row = injury_reports[
                (injury_reports["player_id"] == player_id)
                & (injury_reports["season"] == game_row["season"]) & (injury_reports["week"] == game_row["week"])
            ]
            current_injury_status = float(INJURY_STATUS_ORDINAL.get(report_row["report_status"].iloc[0], 0)) if len(report_row) else 0.0

        team_ol_injuries = float(injury_reports[
            (injury_reports["team"] == team) & (injury_reports["player_id"].isin(ol_ids))
            & (injury_reports["season"] == game_row["season"]) & (injury_reports["week"] == game_row["week"])
        ].shape[0])
    else:
        # No matching upcoming game found in schedule_context (e.g. outside
        # the pulled years) -- fall back to league-average constants like
        # every other cold-start case in this file.
        is_home = context_fill_values["is_home"]
        rest_days = context_fill_values["rest_days"]
        opponent_rest_days = context_fill_values["opponent_rest_days"]
        is_divisional_game = context_fill_values["is_divisional_game"]
        game_temp = context_fill_values["game_temp"]
        game_wind = context_fill_values["game_wind"]
        implied_team_total = context_fill_values.get("implied_team_total", 0.0)
        team_spread_line = context_fill_values.get("team_spread_line", 0.0)
        games_with_current_qb = context_fill_values["games_with_current_qb"]
        if current_injury_status is None:
            current_injury_status = 0.0  # no upcoming game identified at all -- nothing sensible to look injury status up against
        team_ol_injuries = 0.0  # same -- no specific upcoming week to check O-line injuries against

    return np.array([
        opp_pass_yards_allowed, opp_pass_tds_allowed, opp_rush_yards_allowed, opp_rush_tds_allowed,
        opp_top_cb_coverage_score,
        h2h_pass_yards, h2h_pass_tds, h2h_rush_yards, h2h_rush_tds,
        is_home, rest_days, opponent_rest_days, is_divisional_game, game_temp, game_wind, games_with_current_qb,
        team_plays_per_game, opponent_plays_per_game, opponent_takeaways_per_game, team_ol_injuries,
        implied_team_total, team_spread_line,
        player_team_position_rank, size_diff_vs_top_cb_height, size_diff_vs_top_cb_weight,
        current_player_age, current_years_experience, current_injury_status, player_vs_opponent_difficulty,
        current_player_archetype, opponent_defense_vs_archetype,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# SEQUENCE CONSTRUCTION
# ---------------------------------------------------------------------------
def build_sequences(
    df: pd.DataFrame,
    stat_columns: List[str],
    context_lookup: Dict[Tuple[int, int, str], np.ndarray],
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turn a position group's game log into sliding-window training examples:
    for every player, every run of (window_size + 1) consecutive games
    becomes one example -- the first window_size games are the input
    sequence, the (window_size + 1)th game is the prediction target, and the
    matchup context for THAT target game is looked up alongside it.

    Args:
        df: Weekly stats for ONE position group (already filtered), with at
            least player_id, player_name, season, week, team, stat_columns,
            AND every column in PLAYER_CONTEXT_COLUMNS already merged in
            (see build_player_team_rank_profile()/build_size_diff_profile())
            -- the CONTEXT_COLUMNS that are per-player rather than per-team,
            so they're read directly off each row instead of coming from
            `context_lookup`.
        stat_columns: Which columns form both the input features and the
            prediction target (see POSITION_STAT_COLUMNS).
        context_lookup: Dict mapping (season, week, team) -> a
            len(TEAM_CONTEXT_COLUMNS) float32 array (see build_matchup_context()
            + fillna_context() to build this, already NaN-free). Missing
            keys fall back to an all-NaN vector filled in by the caller's
            fillna step -- shouldn't happen if context_lookup was built from
            the same df's teams/weeks, but guarded defensively.
        window_size: How many past games make up one input sequence.

    Returns:
        X: float32 array, shape (num_examples, window_size, num_stat_columns).
        C: float32 array, shape (num_examples, len(CONTEXT_COLUMNS)) -- the
           target game's matchup context.
        y: float32 array, shape (num_examples, num_stat_columns) -- the
           target (next game's) stat line.
        meta: DataFrame, one row per example in the same order as X/C/y,
            with player_id, player_name, season, week identifying the
            TARGET game -- used for traceability and the chronological
            train/val split in chronological_split().

    Notes:
        Missing stat values (e.g. a QB who didn't attempt a pass that week)
        are filled with 0.0. Players with fewer than window_size + 1 games
        in `df` contribute no examples at all (see the WINDOW_SIZE design
        note in the module docstring).
    """
    df = df.sort_values(["player_id", "season", "week"])
    X_list: List[np.ndarray] = []
    C_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_rows: List[dict] = []
    team_context_fallback = np.full(len(TEAM_CONTEXT_COLUMNS), np.nan, dtype=np.float32)

    for player_id, g in df.groupby("player_id"):
        g = g.sort_values(["season", "week"]).reset_index(drop=True)
        if len(g) < window_size + 1:
            continue

        stats = g[stat_columns].fillna(0.0).to_numpy(dtype=np.float32)
        for i in range(window_size, len(g)):
            season, week, team = int(g.loc[i, "season"]), int(g.loc[i, "week"]), g.loc[i, "team"]
            X_list.append(stats[i - window_size:i])
            y_list.append(stats[i])

            team_context = context_lookup.get((season, week, team), team_context_fallback)
            player_context = g.loc[i, PLAYER_CONTEXT_COLUMNS].to_numpy(dtype=np.float32)
            C_list.append(np.concatenate([team_context, player_context]))

            meta_rows.append({
                "player_id": player_id,
                "player_name": g.loc[i, "player_name"],
                "season": season,
                "week": week,
            })

    num_features = len(stat_columns)
    X = np.stack(X_list).astype(np.float32) if X_list else np.empty((0, window_size, num_features), dtype=np.float32)
    C = np.stack(C_list).astype(np.float32) if C_list else np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32)
    y = np.stack(y_list).astype(np.float32) if y_list else np.empty((0, num_features), dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, C, y, meta


def chronological_split(meta: pd.DataFrame, val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split example indices into train/validation sets by time rather than
    randomly, so validation only ever contains games later than everything
    the model trained on (see the module docstring's split rationale).

    Args:
        meta: The per-example metadata DataFrame returned by build_sequences()
            (must have season/week columns for the target game).
        val_frac: Fraction of examples (by time, not by player) to hold out
            as the most-recent validation slice. Default 0.15 (last ~15%).

    Returns:
        (train_indices, val_indices): integer arrays indexing into the
        X/C/y arrays build_sequences() returned (meta's row order matches
        them).
    """
    ordered = meta.sort_values(["season", "week"]).index.to_numpy()
    cutoff = int(len(ordered) * (1 - val_frac))
    return ordered[:cutoff], ordered[cutoff:]


def compute_recency_weights(meta: pd.DataFrame, half_life_weeks: float = RECENCY_HALF_LIFE_WEEKS) -> np.ndarray:
    """
    Per-example loss weight that decays exponentially with how old each
    training example's TARGET game is -- current-season and recent games
    count more than older ones, instead of every example (however old)
    pulling the loss equally.

    Args:
        meta: The per-example metadata DataFrame build_sequences() returns
            (needs season/week for the target game). Row order/index must
            match the X/C/y arrays being weighted, same contract as
            chronological_split().
        half_life_weeks: How many weeks back a training example's weight
            halves (default RECENCY_HALF_LIFE_WEEKS). Smaller = more
            aggressively recency-weighted.

    Returns:
        1D float32 array, same length/order as `meta`, in (0, 1] -- 1.0 for
        the single most recent example in `meta`, decaying by half every
        `half_life_weeks` further back.

    Notes:
        "Weeks ago" is approximated as `season * 18 + week` distance (every
        NFL season having 18 weeks since 2021) rather than real calendar
        dates -- close enough for a smooth recency curve; this file has no
        game-date data pulled (see get_schedule_context()'s columns), and
        an off-by-a-few-days approximation doesn't meaningfully change
        which games count as "recent."

        Applies to the TRAINING set's loss (see train_position_model()) --
        the validation set (chronological_split()'s already-most-recent
        slice) doesn't need this weighting for its own sake, but gets it
        anyway for consistency since it's cheap and doesn't change which
        checkpoint looks best (weighting shifts val loss's absolute scale,
        not the ranking across epochs).
    """
    absolute_week = meta["season"].to_numpy() * 18 + meta["week"].to_numpy()
    weeks_ago = absolute_week.max() - absolute_week
    return np.power(0.5, weeks_ago / half_life_weeks).astype(np.float32)


# ---------------------------------------------------------------------------
# NORMALIZATION -- fit on training data only, applied to train + val alike
# ---------------------------------------------------------------------------
@dataclass
class Normalizer:
    """Per-feature z-score (standardize/inverse) normalization stats."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))


def fit_normalizer(values: np.ndarray) -> Normalizer:
    """
    Compute per-column mean/std for z-score normalization.

    Args:
        values: 2D array, shape (num_rows, num_features).

    Returns:
        A Normalizer with mean/std per feature. Columns with zero variance
        (e.g. a stat that's always 0 for every example in a small dataset)
        get std=1.0 instead of 0, to avoid a divide-by-zero -- their
        normalized value is just (x - mean), which is fine since there's no
        real spread to preserve.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))


# ---------------------------------------------------------------------------
# PYTORCH DATASET
# ---------------------------------------------------------------------------
class PlayerGameSequenceDataset(Dataset):
    """
    Thin Dataset wrapper around pre-built (X, C, y) sequence/context/target
    arrays, with an optional per-example loss weight (see
    compute_recency_weights()) -- defaults to all-ones (every example
    counted equally) when not given.
    """

    def __init__(self, X: np.ndarray, C: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.C = torch.from_numpy(C).float()
        self.y = torch.from_numpy(y).float()
        self.weights = torch.from_numpy(weights).float() if weights is not None else torch.ones(len(X), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.C[idx], self.y[idx], self.weights[idx]


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class PlayerStatLSTM(nn.Module):
    """
    LSTM regressor: reads a player's last `window_size` games (one stat
    vector per game), combines that with this week's matchup context, and
    predicts the next game's stat vector directly.

    Architecture: a stacked LSTM processes the game sequence. The matchup
    context (opponent defense strength + head-to-head history for THIS
    specific game -- exogenous, known-in-advance information that doesn't
    belong in the sequence itself) is first passed through its own small
    encoder (Linear + ReLU) rather than concatenated onto the LSTM's hidden
    state raw. That encoder step matters in practice: an 8-number context
    vector concatenated directly onto a 64-number hidden state is only
    ~11% of the combined vector, and empirically the optimizer mostly
    ignored it that way (context weights in the final layer ended up ~4x
    smaller than the hidden-state weights, and predictions barely moved
    between facing the league's best and worst pass defense). Projecting
    context through its own `context_embed_size`-wide layer first gives it
    equal standing in the concatenation and its own learnable
    transformation, rather than competing for a sliver of a much larger
    layer's attention. The combined [hidden_state, context_embedding]
    vector then goes through one linear layer to the prediction -- no
    activation on that final layer, since these are continuous, unbounded
    (in principle) counting stats rather than a classification target.
    """

    def __init__(
        self,
        num_features: int,
        context_size: int = 0,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        context_embed_size: int = 24,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.context_size = context_size
        self.context_embed_size = context_embed_size if context_size > 0 else 0

        if context_size > 0:
            self.context_encoder = nn.Sequential(nn.Linear(context_size, context_embed_size), nn.ReLU())
        else:
            self.context_encoder = None

        self.head = nn.Linear(hidden_size + self.context_embed_size, num_features)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, num_features)
            context: (batch, context_size) -- required whenever
                context_size > 0 (i.e. whenever this model was trained with
                matchup features, which is every model this file produces).

        Returns:
            (batch, num_features) -- predicted next-game stat vector.
        """
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size): final LSTM layer's hidden state at the last time step

        if self.context_encoder is not None:
            if context is None:
                raise ValueError("This model was trained with matchup context features; forward() requires `context`.")
            context_embedding = self.context_encoder(context)  # (batch, context_embed_size)
            last_layer_hidden = torch.cat([last_layer_hidden, context_embedding], dim=1)

        return self.head(last_layer_hidden)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def evaluate_mae(model: nn.Module, data_loader: DataLoader, normalizer: Normalizer, device: str) -> np.ndarray:
    """
    Run a trained model over a DataLoader and compute per-stat Mean Absolute
    Error in real (denormalized) units -- e.g. "off by 14.2 passing yards on
    average," which is far more readable than a normalized MSE number.

    Args:
        model: A trained PlayerStatLSTM in any mode; this function sets it
            to eval() internally.
        data_loader: Yields (normalized) (X, C, y, weight) batches (weight is ignored here).
        normalizer: The Normalizer fit on this position group's training
            stat data (not the context normalizer -- context isn't part of
            the prediction target, so it doesn't need inverse-transforming).
        device: "cpu" or "cuda".

    Returns:
        1D array of length num_features: mean absolute error per stat column,
        in the original (non-normalized) units.
    """
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for xb, cb, yb, _wb in data_loader:
            xb, cb = xb.to(device), cb.to(device)
            preds_all.append(model(xb, cb).cpu().numpy())
            targets_all.append(yb.numpy())

    if not preds_all:
        return np.zeros(normalizer.mean.shape, dtype=np.float32)

    preds = normalizer.inverse_transform(np.concatenate(preds_all))
    targets = normalizer.inverse_transform(np.concatenate(targets_all))
    return np.mean(np.abs(preds - targets), axis=0)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantile: float, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Quantile ("pinball") loss: asymmetrically penalizes over- vs.
    under-prediction so the network learns a specific quantile of the
    target distribution instead of its mean. quantile=0.5 reduces to (half)
    mean absolute error; quantile=0.1 penalizes over-prediction 9x harder
    than under-prediction, pulling the model toward the 10th percentile
    (a "floor"); quantile=0.9 does the reverse (a "ceiling").

    Args:
        preds, target: Equal-shaped 2D tensors (batch, features) --
            normalized or real-unit values both work, since quantile loss
            is equivariant under the strictly increasing affine z-score
            transform this file's Normalizer applies (minimizing it in
            normalized space minimizes it in real space too, so no
            special-casing needed for count vs. continuous columns here,
            unlike the Poisson term in train_position_model()).
        quantile: Target quantile in (0, 1).
        weights: Optional (batch,) per-example weight (see
            compute_recency_weights()) -- weights the per-example mean
            before it's summed, rather than every example counting
            equally. Pass None (default) to weight every example the same.

    Returns:
        Scalar loss.
    """
    diff = target - preds
    per_example = torch.maximum(quantile * diff, (quantile - 1) * diff).mean(dim=1)
    if weights is None:
        return per_example.mean()
    return (per_example * weights).sum() / weights.sum()


def prepare_position_training_data(
    position_group: str,
    df: pd.DataFrame,
    context_df: pd.DataFrame,
    top_cb_profile: pd.DataFrame,
    player_bio: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    archetypes: pd.DataFrame,
    defense_vs_archetype: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    val_frac: float = 0.15,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, "Normalizer", "Normalizer", Dict[str, float], np.ndarray, np.ndarray]:
    """
    Shared leak-safe data prep for train_position_model(),
    train_position_model_ga(), and
    PlayerProjectionsTransformer.train_position_model_hf() -- context
    filling, per-player profile merges, sequence building, chronological
    split, and normalization. Identical across all three training paths;
    only the model architecture and optimization loop differ downstream.
    See train_position_model()'s docstring for the full leak-safety/
    normalization reasoning (unchanged here).

    Args:
        Same meaning as train_position_model()'s matching arguments.

    Returns:
        (stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
        normalizer, context_normalizer, context_fill_values, weights_train,
        weights_val).

    Raises:
        ValueError: if position_group isn't in POSITION_STAT_COLUMNS, or if
            there are too few training examples (fewer than 20) -- widen
            `years` in load_weekly_stats() if you hit this for a
            lower-volume position group.
    """
    if position_group not in POSITION_STAT_COLUMNS:
        raise ValueError(f"Unknown position_group {position_group!r}; expected one of {list(POSITION_STAT_COLUMNS)}")
    stat_columns = POSITION_STAT_COLUMNS[position_group]

    filled_context_df, context_fill_values = fillna_context(context_df)
    context_lookup = {
        (int(row.season), int(row.week), row.team): np.array([getattr(row, c) for c in TEAM_CONTEXT_COLUMNS], dtype=np.float32)
        for row in filled_context_df.itertuples()
    }

    pos_df = df[df["position_group"] == position_group]
    rank_stat = POSITION_RANKING_STAT[position_group]
    rank_profile = build_player_team_rank_profile(pos_df, rank_stat)
    pos_df = pos_df.merge(rank_profile, on=["player_id", "season", "week"], how="left")

    size_diff_profile = build_size_diff_profile(pos_df, top_cb_profile, player_bio)
    pos_df = pos_df.merge(size_diff_profile, on=["player_id", "season", "week"], how="left")

    age_profile = build_age_profile(pos_df, player_age)
    pos_df = pos_df.merge(age_profile, on=["player_id", "season", "week"], how="left")

    injury_profile = build_injury_profile(pos_df, injury_reports)
    pos_df = pos_df.merge(injury_profile, on=["player_id", "season", "week"], how="left")

    defense_profile = build_defense_profile(build_team_game_log(df))
    difficulty_profile = build_opponent_difficulty_profile(pos_df, defense_profile, position_group, rank_stat)
    pos_df = pos_df.merge(difficulty_profile, on=["player_id", "season", "week"], how="left")
    pos_df["player_vs_opponent_difficulty"] = pos_df["player_vs_opponent_difficulty"].fillna(pos_df[rank_stat].mean())

    archetype_context = build_archetype_context(pos_df, archetypes, defense_vs_archetype, position_group)
    pos_df = pos_df.merge(archetype_context, on=["player_id", "season", "week"], how="left")

    X, C, y, meta = build_sequences(pos_df, stat_columns, context_lookup, window_size)
    if len(X) < 20:
        raise ValueError(
            f"Only {len(X)} training examples for {position_group} (need players with "
            f">= {window_size + 1} games each) -- pass a wider `years` range to load_weekly_stats()."
        )

    train_idx, val_idx = chronological_split(meta, val_frac)
    X_train, C_train, y_train = X[train_idx], C[train_idx], y[train_idx]
    X_val, C_val, y_val = X[val_idx], C[val_idx], y[val_idx]

    # Fit normalization on training data only (never let validation-set
    # statistics leak into how inputs/targets are scaled). Stat columns and
    # context columns are on different scales/semantics, so each gets its
    # own Normalizer.
    normalizer = fit_normalizer(np.concatenate([X_train.reshape(-1, X_train.shape[-1]), y_train], axis=0))
    context_normalizer = fit_normalizer(C_train)

    # Recency weights -- current-season/recent games count more toward the
    # loss than older ones (see compute_recency_weights()). Computed once
    # over the full `meta` so train and val share the same "how recent"
    # reference point, then sliced the same way X/C/y already were.
    recency_weights = compute_recency_weights(meta)
    weights_train, weights_val = recency_weights[train_idx], recency_weights[val_idx]

    return (
        stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
        normalizer, context_normalizer, context_fill_values, weights_train, weights_val,
    )


def train_position_model(
    position_group: str,
    df: pd.DataFrame,
    context_df: pd.DataFrame,
    top_cb_profile: pd.DataFrame,
    player_bio: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    archetypes: pd.DataFrame,
    defense_vs_archetype: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    hidden_size: int = 64,
    num_layers: int = 2,
    quantile: Optional[float] = None,
    device: Optional[str] = None,
    loss_stats: Optional[List[str]] = None,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = None,
) -> Tuple[PlayerStatLSTM, Normalizer, Normalizer, Dict[str, float], List[str], List[dict]]:
    """
    Train one PlayerStatLSTM for a single position group, conditioned on
    matchup context.

    Args:
        position_group: One of POSITION_STAT_COLUMNS' keys ("QB"/"RB"/"WR"/"TE").
        df: Weekly stats for ALL positions (e.g. from load_weekly_stats()) --
            this function filters to `position_group` itself.
        context_df: Output of build_matchup_context(df, top_cb_profile) (or
            a subset, for walk-forward training -- see
            walk_forward_train_and_predict()). Built once and passed in
            rather than recomputed per position, so all four position
            models see identical matchup features.
        top_cb_profile: Output of build_top_cb_profile() -- used here for
            build_size_diff_profile() (context_df already has its
            opp_top_cb_coverage_score baked in).
        player_bio: Output of load_player_bio() -- height/weight lookup for
            build_size_diff_profile().
        player_age: Output of load_player_age() -- see build_age_profile().
        injury_reports: Output of load_injury_reports() -- see
            build_injury_profile().
        archetypes: This position's entry from
            PlayerArchetypes.compute_all_archetypes(df) -- see
            build_archetype_context().
        defense_vs_archetype: This position's entry from
            PlayerArchetypes.compute_all_defense_vs_archetype_profiles(df, ...)
            -- see build_archetype_context().
        window_size: Games of history per input sequence.
        epochs: Training epochs (full passes over the training examples).
        batch_size: Examples per gradient step.
        lr: Adam learning rate.
        val_frac: Fraction of (chronologically) most recent examples held
            out for validation -- see chronological_split().
        hidden_size: LSTM hidden state size.
        num_layers: Stacked LSTM layers.
        quantile: If given (0 < quantile < 1), trains with pinball_loss()
            at this quantile instead of MSE -- e.g. quantile=LOW_QUANTILE
            (0.1) trains a "floor" model, quantile=HIGH_QUANTILE (0.9) a
            "ceiling" model. Pass None (default) for the normal MSE-trained
            "likely" point-estimate model (the Poisson-mix described below
            only applies in this default case -- pinball loss already
            handles asymmetric residuals correctly on its own). See
            predict_next_game_ensemble() for how a floor/ceiling pair
            combines with two MSE-trained models into one projection.
        device: "cpu"/"cuda"; autodetects (preferring cuda) if not given.
        loss_stats: If given, only these stat columns enter the loss (every
            stat is still an input and still predicted) -- e.g.
            ["passing_yards"] so the bet stat gets the whole training signal
            instead of 1/9 of it. None (default) = every stat, as before.
        weight_decay: > 0 switches Adam to AdamW with this decoupled decay.
            0 (default) = plain Adam, as before.
        grad_clip: If given, clips the gradient norm to this before each
            step. None (default) = no clipping, as before.
            (These three: docs/lstm-fix-protocol.md, 2026-10-27.)

    Returns:
        (model, normalizer, context_normalizer, context_fill_values,
        stat_columns, history) -- normalizer covers the player-history
        stat columns (inputs+target), context_normalizer separately covers
        the CONTEXT_COLUMNS matchup features (different scale/semantics, so
        they get their own z-score fit), context_fill_values are the "no
        history yet" fallback constants from fillna_context(), and history
        is a list of per-epoch dicts ({"epoch", "train_loss", "val_loss"}).

    Raises:
        ValueError: if position_group isn't in POSITION_STAT_COLUMNS, or if
            there are too few training examples (fewer than 20) to bother
            training on -- widen `years` in load_weekly_stats() if you hit
            this for a lower-volume position group.

    Notes:
        MSE mode (quantile=None) adds an auxiliary Poisson negative-
        log-likelihood term for the count-like stat columns (anything
        ending in "_tds" or "_interceptions") on top of the usual MSE, at
        weight POISSON_WEIGHT. MSE assumes Gaussian-shaped error, which
        fits continuous stats like yardage reasonably well but is a poor
        match for a column that's usually 0 and rarely 2 or 3 -- the 2025
        backtest showed exactly this (touchdown/interception MAE running
        100-240% of their own mean value, far worse than yardage stats).
        The Poisson term is computed on DENORMALIZED predictions (inverse-
        transformed back to real touchdown/interception counts via the
        same affine Normalizer already fit for MSE, so no separate
        normalization path or output-head change is needed) and added to
        the base MSE loss -- still one shared linear output head, still one
        Normalizer, just an extra differentiable term that penalizes count
        columns the way their actual (roughly Poisson) distribution
        warrants rather than treating them like any other continuous stat.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (
        stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
        normalizer, context_normalizer, context_fill_values, weights_train, weights_val,
    ) = prepare_position_training_data(
        position_group, df, context_df, top_cb_profile, player_bio, player_age,
        injury_reports, archetypes, defense_vs_archetype, window_size, val_frac,
    )

    train_ds = PlayerGameSequenceDataset(
        normalizer.transform(X_train), context_normalizer.transform(C_train), normalizer.transform(y_train), weights_train
    )
    val_ds = PlayerGameSequenceDataset(
        normalizer.transform(X_val), context_normalizer.transform(C_val), normalizer.transform(y_val), weights_val
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PlayerStatLSTM(
        num_features=len(stat_columns), context_size=len(CONTEXT_COLUMNS), hidden_size=hidden_size, num_layers=num_layers
    ).to(device)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if weight_decay > 0
                 else torch.optim.Adam(model.parameters(), lr=lr))

    # See this function's Notes for why MSE mode adds a Poisson term for
    # count-like columns, and why quantile mode doesn't need it.
    keep = [i for i, c in enumerate(stat_columns) if loss_stats is None or c in loss_stats]
    if not keep:
        raise ValueError(f"loss_stats {loss_stats} names none of {stat_columns}")
    poisson_idx = [i for i in keep if stat_columns[i].endswith("_tds") or stat_columns[i].endswith("_interceptions")]
    mean_t = torch.tensor(normalizer.mean, device=device)
    std_t = torch.tensor(normalizer.std, device=device)

    def compute_loss(preds: torch.Tensor, yb: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
        if loss_stats is not None:
            preds_k, yb_k = preds[:, keep], yb[:, keep]
        else:
            preds_k, yb_k = preds, yb
        if quantile is not None:
            return pinball_loss(preds_k, yb_k, quantile, weights=wb)
        per_example_mse = ((preds_k - yb_k) ** 2).mean(dim=1)
        loss = (per_example_mse * wb).sum() / wb.sum()
        if poisson_idx:
            pred_real = torch.clamp(preds[:, poisson_idx] * std_t[poisson_idx] + mean_t[poisson_idx], min=1e-3)
            target_real = yb[:, poisson_idx] * std_t[poisson_idx] + mean_t[poisson_idx]
            per_example_poisson = (pred_real - target_real * torch.log(pred_real)).mean(dim=1)
            poisson_nll = (per_example_poisson * wb).sum() / wb.sum()
            loss = loss + POISSON_WEIGHT * poisson_nll
        return loss

    # Track the best-validation-loss checkpoint rather than just keeping
    # whatever the final epoch happens to produce. With this small a dataset
    # (a few thousand examples, single-digit feature counts) the model
    # reliably overfits well before `epochs` is reached -- train_loss keeps
    # dropping while val_loss bottoms out and then climbs back up. Returning
    # the final-epoch weights would silently hand back a model that's worse
    # at "how close it gets to the next game" than one seen earlier in
    # training, which defeats the actual goal.
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, cb, yb, wb in train_loader:
            xb, cb, yb, wb = xb.to(device), cb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            preds = model(xb, cb)
            loss = compute_loss(preds, yb, wb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / max(len(train_ds), 1)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, cb, yb, wb in val_loader:
                xb, cb, yb, wb = xb.to(device), cb.to(device), yb.to(device), wb.to(device)
                val_running_loss += compute_loss(model(xb, cb), yb, wb).item() * len(xb)
        val_loss = val_running_loss / max(len(val_ds), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            mae = evaluate_mae(model, val_loader, normalizer, device)
            mae_str = ", ".join(f"{c}={m:.2f}" for c, m in zip(stat_columns, mae))
            log_tag = f"{position_group}-q{quantile:g}" if quantile is not None else position_group
            loss_name = "pinball" if quantile is not None else "mse+poisson"
            print(f"[{log_tag}] epoch {epoch:3d}/{epochs}  train_{loss_name}={train_loss:.4f}  val_{loss_name}={val_loss:.4f}  val_mae: {mae_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    log_tag = f"{position_group}-q{quantile:g}" if quantile is not None else position_group
    print(f"[{log_tag}] using epoch {best_epoch}'s weights (lowest val_loss={best_val_loss:.4f})")

    return model, normalizer, context_normalizer, context_fill_values, stat_columns, history


# ---------------------------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------------------------
def save_model(
    model: PlayerStatLSTM,
    normalizer: Normalizer,
    context_normalizer: Normalizer,
    context_fill_values: Dict[str, float],
    stat_columns: List[str],
    position_group: str,
    window_size: int,
    model_dir: str = MODEL_DIR,
    model_name: Optional[str] = None,
    starters_only: bool = False,
) -> None:
    """
    Save a trained model's weights (.pt) and everything needed to use it
    again later -- stat columns, window size, architecture sizes, and both
    normalizers -- as a sidecar .json (PyTorch state dicts alone don't
    carry any of that, so without it a saved model would be unusable
    without hardcoding those values back in by hand).

    Args:
        model: A trained PlayerStatLSTM.
        normalizer: The stat-column Normalizer fit during training.
        context_normalizer: The matchup-context Normalizer fit during training.
        context_fill_values: The "no history yet" fallback constants from
            fillna_context(), needed to build live context vectors later.
        stat_columns: Which columns this model predicts, in order.
        position_group: "QB"/"RB"/"WR"/"TE" -- the real position this model
            is for (unrelated to which file it's saved as -- see model_name).
        window_size: Games of history this model expects as input.
        model_dir: Directory to save into (created if missing).
        model_name: Filename prefix to save under. Defaults to
            position_group.lower() (e.g. "qb"). Pass a distinct name (e.g.
            "qb_low", "qb_high") to save a quantile variant (see
            train_position_model()'s `quantile` param) alongside the
            standard model without overwriting it -- load_model() and
            predict_next_game() take the same parameter to load it back.
    """
    os.makedirs(model_dir, exist_ok=True)
    prefix = (model_name or position_group).lower()

    torch.save(model.state_dict(), os.path.join(model_dir, f"{prefix}_lstm.pt"))

    meta = {
        "position_group": position_group,
        "stat_columns": stat_columns,
        "context_columns": CONTEXT_COLUMNS,
        "window_size": window_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "context_size": model.context_size,
        "context_embed_size": model.context_embed_size,
        # The training-row rule, so prediction can feed the model the same kind
        # of history it learned from (see _predict_next_game_core()).
        "starters_only": bool(starters_only),
        "normalizer": normalizer.to_dict(),
        "context_normalizer": context_normalizer.to_dict(),
        "context_fill_values": context_fill_values,
    }
    with open(os.path.join(model_dir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {position_group} model -> {model_dir}/{prefix}_lstm.pt (+ {prefix}_meta.json)")


def load_model(position_group: str, model_dir: str = MODEL_DIR, device: Optional[str] = None, model_name: Optional[str] = None):
    """
    Load a previously saved position model back into memory, ready for
    predict_next_game().

    Args:
        position_group: "QB"/"RB"/"WR"/"TE".
        model_dir: Directory save_model() wrote into.
        device: "cpu"/"cuda"; autodetects if not given.
        model_name: Which saved file to load -- defaults to
            position_group.lower(). Pass "qb_low"/"qb_high" (etc.) to load
            a quantile variant saved under that name (see save_model()).

    Returns:
        (model, normalizer, context_normalizer, context_fill_values,
        stat_columns, window_size) -- model is already in eval() mode and
        on `device`.

    Raises:
        FileNotFoundError: if this position group/model_name hasn't been
            trained/saved yet -- run train_all_position_models() (or
            train_all_quantile_models() for a low/high variant) first.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prefix = (model_name or position_group).lower()
    meta_path = os.path.join(model_dir, f"{prefix}_meta.json")
    weights_path = os.path.join(model_dir, f"{prefix}_lstm.pt")

    if not os.path.exists(meta_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No saved model named {prefix!r} in {model_dir}/ -- run train_all_position_models() "
            f"(or train_all_quantile_models() for a low/high variant) first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    model = PlayerStatLSTM(
        num_features=len(meta["stat_columns"]),
        context_size=meta.get("context_size", 0),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        context_embed_size=meta.get("context_embed_size", 16),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    normalizer = Normalizer.from_dict(meta["normalizer"])
    context_normalizer = Normalizer.from_dict(meta["context_normalizer"])
    return model, normalizer, context_normalizer, meta["context_fill_values"], meta["stat_columns"], meta["window_size"]


# ---------------------------------------------------------------------------
# PREDICTION (using an already-trained/saved model)
# ---------------------------------------------------------------------------
def trained_on_starters(position_group: str, model_dir: str = MODEL_DIR, model_name: Optional[str] = None) -> bool:
    """
    Whether a saved model was trained on listed starters' games only (its meta's
    "starters_only"). A model saved before the flag existed reads False: it was
    trained on every row, so it must keep being fed every row.
    """
    prefix = (model_name or position_group).lower()
    with open(os.path.join(model_dir, f"{prefix}_meta.json")) as f:
        return bool(json.load(f).get("starters_only", False))


def _predict_next_game_core(
    player_id: str,
    next_opponent: str,
    load_for_position,
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    archetypes: Optional[pd.DataFrame] = None,
    defense_vs_archetype: Optional[pd.DataFrame] = None,
    position_group: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Shared inference body behind predict_next_game() and
    PlayerProjectionsTransformer.predict_next_game_hf() -- everything
    except how the model/normalizer are loaded and how the model's forward
    pass is shaped (LSTM returns a raw tensor, the HF transformer returns a
    dict), which `load_for_position` supplies. See predict_next_game()'s
    docstring for the full argument/return/raise contract -- identical here.

    Args:
        load_for_position: position_group -> (model, stat_columns,
            window_size, normalizer, context_normalizer, context_fill_values,
            forward, starters_only), where forward(model, x_tensor, c_tensor)
            returns the raw normalized prediction tensor and starters_only says
            whether the model was trained on listed starters' games only.
        Other args: same meaning as predict_next_game().
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if df is None:
        df = load_weekly_stats()
    if defensive_target_stats is None:
        defensive_target_stats = load_defensive_target_stats()
    if player_bio is None:
        player_bio = load_player_bio()
    if schedule_context is None:
        schedule_context = load_schedule_context()
    if player_age is None:
        player_age = load_player_age()
    if injury_reports is None:
        injury_reports = load_injury_reports()

    player_rows = df[df["player_id"] == player_id].sort_values(["season", "week"])
    if player_rows.empty:
        raise ValueError(f"No games found for player_id={player_id!r}")

    if position_group is None:
        position_group = player_rows["position_group"].iloc[-1]
    if position_group not in POSITION_STAT_COLUMNS:
        raise ValueError(f"No model for position_group={position_group!r}; supported: {list(POSITION_STAT_COLUMNS)}")

    if archetypes is None or defense_vs_archetype is None:
        from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

        if archetypes is None:
            archetypes = classify_archetype(compute_style_features(df, position_group), position_group)
        if defense_vs_archetype is None:
            defense_vs_archetype = build_defense_vs_archetype_profile(df, archetypes, position_group)

    model, stat_columns, window_size, normalizer, context_normalizer, context_fill_values, forward, starters_only = \
        load_for_position(position_group)

    # A QB model trained on starts must be fed starts: the same rule its training
    # rows followed (position_training_rows()). Context below still uses every
    # row, and the player's team is his latest appearance, start or not.
    starts_only = starters_only and position_group == "QB"
    history = qb_starter_rows(player_rows, schedule_context) if starts_only else player_rows
    if len(history) < window_size:
        name = player_rows["player_name"].iloc[-1]
        unit = "start(s) as listed starter" if starts_only else "game(s) on record"
        raise ValueError(f"{name} has only {len(history)} {unit}; need at least {window_size} to predict a next game.")

    recent = history.tail(window_size)[stat_columns].fillna(0.0).to_numpy(dtype=np.float32)
    x = normalizer.transform(recent)

    team = player_rows["team"].iloc[-1]
    team_games = build_team_game_log(df)
    context_vec = current_matchup_context(
        df, team_games, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports,
        archetypes, defense_vs_archetype,
        player_id, position_group, team, next_opponent, context_fill_values, injury_status=injury_status,
    )
    if len(context_normalizer.mean) == len(CONTEXT_COLUMNS) - len(VEGAS_CONTEXT_COLUMNS):
        # Saved before the Vegas features existed: feed it the inputs it was trained on.
        context_vec = context_vec[[i for i, col in enumerate(CONTEXT_COLUMNS) if col not in VEGAS_CONTEXT_COLUMNS]]
    c = context_normalizer.transform(context_vec.reshape(1, -1))[0]

    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1, window_size, num_features)
    c_tensor = torch.from_numpy(c).float().unsqueeze(0).to(device)  # (1, context_size)

    with torch.no_grad():
        pred_normalized = forward(model, x_tensor, c_tensor).cpu().numpy()[0]
    pred = normalizer.inverse_transform(pred_normalized)
    pred = np.clip(pred, a_min=0, a_max=None)  # counting stats can't be negative

    return {col: float(val) for col, val in zip(stat_columns, pred)}


def predict_next_game(
    player_id: str,
    next_opponent: str,
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    archetypes: Optional[pd.DataFrame] = None,
    defense_vs_archetype: Optional[pd.DataFrame] = None,
    position_group: Optional[str] = None,
    model_dir: str = MODEL_DIR,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict one player's next game stat line using their trained
    position-group model, their most recent WINDOW_SIZE games, and this
    week's specific matchup.

    Args:
        player_id: nflverse player_id, e.g. "00-0039732" (Bo Nix). Look these
            up via df[df["player_name"] == "..."] if you don't have one handy.
        next_opponent: The team abbreviation this player's team is about to
            face, e.g. "LV". Required -- matchup context is central to what
            this model conditions its prediction on, so there's no sensible
            "no opponent" default. Use load_schedule_context()'s data to look
            up an upcoming opponent if you don't already know it.
        df: Weekly stats to pull the player's recent game history AND the
            league's team-game log (for matchup context) from. Defaults to
            load_weekly_stats() if not given (that hits the database/
            network, so pass an already-loaded DataFrame if calling this in
            a loop for many players).
        defensive_target_stats: Targeted-coverage stats for identifying the
            opponent's best CB. Defaults to load_defensive_target_stats() if
            not given -- same "pass it in when calling this in a loop"
            advice as `df`.
        player_bio: Height/weight/position lookup. Defaults to
            load_player_bio() if not given -- same advice.
        schedule_context: Schedule/situational data (home/away, rest,
            weather, starting QB) for looking up the specific upcoming
            game. Defaults to load_schedule_context() if not given -- same
            advice.
        player_age: Roster-snapshot age data. Defaults to load_player_age()
            if not given -- same advice.
        injury_reports: Official weekly injury reports. Defaults to
            load_injury_reports() if not given -- same advice.
        injury_status: Optional explicit override for this week's
            game-status designation ("Questionable"/"Doubtful"/"Out") --
            use this once you know the real designation (e.g. it was
            announced after `injury_reports` was loaded) rather than
            relying on whatever's in `injury_reports`. See
            current_matchup_context()'s docstring for the fallback order.
        archetypes, defense_vs_archetype: This player's position's outputs
            from PlayerArchetypes.compute_all_archetypes(df)/
            compute_all_defense_vs_archetype_profiles(df, ...). Defaults to
            computing them (for the resolved position_group) if not given
            -- same "pass them in when calling this in a loop" advice as
            `df`, though computing archetypes is heavier than the other
            defaults, so this matters more here.
        position_group: Which trained model to use. Defaults to the
            player's most recent position_group in `df` if not given.
        model_dir: Directory saved models live in.
        model_name: Which saved model file to load -- defaults to
            position_group.lower(). Pass "qb_low"/"qb_high" (etc., see
            train_position_model()'s `quantile` param) to run this
            player/matchup through a floor/ceiling quantile model instead
            of the standard one -- see predict_next_game_ensemble() for
            the combined version of this.
        device: "cpu"/"cuda"; autodetects if not given.

    Returns:
        Dict mapping each of that position's stat_columns to a predicted
        value for the next game, e.g. {"passing_yards": 241.3, "passing_tds": 1.8, ...}.
        Counting stats are clipped at 0 (the model is an unconstrained
        regressor and can otherwise predict small negative values on a
        quiet-game forecast, which isn't meaningful for a stat that can't
        go below zero in reality).

    Raises:
        ValueError: if the player has no rows in `df`, or has fewer games on
            record than the model's window_size requires, or if
            position_group (given or inferred) has no trained model.
        FileNotFoundError: if the position group's model hasn't been trained
            yet (propagated from load_model()).

    Notes:
        A QB model saved with starters_only=True (see trained_on_starters())
        is fed the player's last window_size games as listed starter, not his
        last appearances: it learned from starts, so a mop-up appearance in
        the input would be out of distribution. Pass the full df regardless;
        the filtering happens here.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(pg):
        model, normalizer, context_normalizer, context_fill_values, stat_columns, window_size = load_model(
            pg, model_dir, device, model_name=model_name
        )
        return (model, stat_columns, window_size, normalizer, context_normalizer, context_fill_values,
                lambda m, x, c: m(x, c), trained_on_starters(pg, model_dir, model_name))

    return _predict_next_game_core(
        player_id, next_opponent, _load,
        df=df, defensive_target_stats=defensive_target_stats, player_bio=player_bio,
        schedule_context=schedule_context, player_age=player_age, injury_reports=injury_reports,
        injury_status=injury_status, archetypes=archetypes, defense_vs_archetype=defense_vs_archetype,
        position_group=position_group, device=device,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT (train once on all history)
# ---------------------------------------------------------------------------
def train_all_position_models(years: Optional[List[int]] = None, starters_only: bool = False,
                              **train_kwargs) -> Dict[str, List[dict]]:
    """
    Train and save a model for every position group in POSITION_STAT_COLUMNS,
    using ALL available history in one shot (not the walk-forward approach
    -- see walk_forward_train_and_predict()/walk_forward_backtest() for that).

    Args:
        years: Passed through to load_weekly_stats() (defaults to the last
            3 seasons if not given).
        starters_only: QB only -- train the QB model on listed starters' games
            (see position_training_rows()); other positions are unaffected.
            Defaults to False, the original behaviour.
        **train_kwargs: Passed through to train_position_model() for every
            position group, e.g. epochs=100, window_size=6.

    Returns:
        Dict mapping position_group -> its training history (see
        train_position_model()'s Returns).
    """
    from PlayerArchetypes import compute_all_archetypes, compute_all_defense_vs_archetype_profiles

    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)
    top_cb_profile = build_top_cb_profile(defensive_target_stats, player_bio)
    context_df = build_matchup_context(df, top_cb_profile, schedule_context, injury_reports, player_bio)
    archetypes_by_position = compute_all_archetypes(df)
    defense_vs_archetype_by_position = compute_all_defense_vs_archetype_profiles(df, archetypes_by_position)

    results = {}
    for position_group in POSITION_STAT_COLUMNS:
        print(f"\n=== Training {position_group} model ===")
        model, normalizer, context_normalizer, context_fill_values, stat_columns, history = train_position_model(
            position_group, position_training_rows(df, schedule_context, position_group, starters_only), context_df, top_cb_profile, player_bio, player_age, injury_reports,
            archetypes_by_position[position_group], defense_vs_archetype_by_position[position_group], **train_kwargs
        )
        window_size = train_kwargs.get("window_size", WINDOW_SIZE)
        save_model(model, normalizer, context_normalizer, context_fill_values, stat_columns, position_group, window_size,
                   starters_only=starters_only and position_group == "QB")
        results[position_group] = history

    return results


def train_all_quantile_models(
    years: Optional[List[int]] = None,
    low_quantile: float = LOW_QUANTILE,
    high_quantile: float = HIGH_QUANTILE,
    model_dir: str = MODEL_DIR,
    starters_only: bool = False,
    **train_kwargs,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    Train and save a LOW and HIGH quantile ("floor"/"ceiling") model for
    every position group -- the pinball-loss counterpart to
    train_all_position_models(), which trains the standard MSE "likely"
    model. See train_position_model()'s `quantile` param and
    predict_next_game_ensemble() for how these combine with the standard
    LSTM and Transformer models into one floor/likely/ceiling projection.

    Args:
        years: Passed through to load_weekly_stats() (defaults to the last
            3 seasons if not given).
        low_quantile, high_quantile: Which quantiles to train (defaults
            LOW_QUANTILE=0.1, HIGH_QUANTILE=0.9).
        model_dir: Directory to save into -- same one train_all_position_models()
            uses; the low/high models are saved under distinct names
            ("{position}_low"/"{position}_high"), so this doesn't overwrite
            the standard models.
        starters_only: QB only -- train the QB model on listed starters' games
            (see position_training_rows()); other positions are unaffected.
            Defaults to False, the original behaviour.
        **train_kwargs: Passed through to train_position_model() for every
            position group and quantile (e.g. epochs=100).

    Returns:
        Dict mapping position_group -> {"low": history, "high": history}.

    Notes:
        This trains 8 models total (4 positions x 2 quantiles), each from
        scratch -- expect roughly double train_all_position_models()'s
        runtime.
    """
    from PlayerArchetypes import compute_all_archetypes, compute_all_defense_vs_archetype_profiles

    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)
    top_cb_profile = build_top_cb_profile(defensive_target_stats, player_bio)
    context_df = build_matchup_context(df, top_cb_profile, schedule_context, injury_reports, player_bio)
    archetypes_by_position = compute_all_archetypes(df)
    defense_vs_archetype_by_position = compute_all_defense_vs_archetype_profiles(df, archetypes_by_position)

    results = {}
    for position_group in POSITION_STAT_COLUMNS:
        results[position_group] = {}
        for tag, q in [("low", low_quantile), ("high", high_quantile)]:
            print(f"\n=== Training {position_group} {tag} (q={q}) model ===")
            model, normalizer, context_normalizer, context_fill_values, stat_columns, history = train_position_model(
                position_group, position_training_rows(df, schedule_context, position_group, starters_only), context_df, top_cb_profile, player_bio, player_age, injury_reports,
                archetypes_by_position[position_group], defense_vs_archetype_by_position[position_group],
                quantile=q, **train_kwargs
            )
            window_size = train_kwargs.get("window_size", WINDOW_SIZE)
            model_name = f"{position_group.lower()}_{tag}"
            save_model(
                model, normalizer, context_normalizer, context_fill_values, stat_columns, position_group,
                window_size, model_dir=model_dir, model_name=model_name,
                starters_only=starters_only and position_group == "QB",
            )
            results[position_group][tag] = history

    return results


# ---------------------------------------------------------------------------
# WALK-FORWARD: retrain fresh on a rolling lookback window, one week at a time
# ---------------------------------------------------------------------------
def qb_starter_rows(frame: pd.DataFrame, schedule_context: pd.DataFrame) -> pd.DataFrame:
    """
    `frame` without the QB rows of players who were not their team's listed
    starter that week (schedule_context.starting_qb_id). Non-QB rows are kept.

    Why: about one QB row in five is a non-starter appearance -- a kneel-down or
    mop-up series, median 3 attempts against 31 for starters (2022-2025). Trained
    on that mix, the QB model shrinks toward a pooled mean: a 2025 walk-forward
    under-projected starters by ~18 passing yards per game and over-projected
    backups by ~36. Who starts is known before kickoff, so filtering on it is
    not a leak.

    Raises if schedule_context lists more than one starter for a team-week,
    rather than silently keeping both.
    """
    keys = schedule_context[["season", "week", "team", "starting_qb_id"]].drop_duplicates()
    merged = frame.merge(keys, on=["season", "week", "team"], how="left", validate="many_to_one")
    keep = (frame["position_group"].to_numpy() != "QB") | (
        merged["player_id"].to_numpy() == merged["starting_qb_id"].to_numpy())
    return frame[keep]


def position_training_rows(df: pd.DataFrame, schedule_context: pd.DataFrame, position_group: str,
                           starters_only: bool = False) -> pd.DataFrame:
    """
    The rows a position's model trains on. With starters_only, QB trains on listed
    starters' games only (qb_starter_rows()); every other position is unchanged,
    since only QB has a listed starter. Callers still build context tables
    (build_matchup_context(), archetypes) from every row, so team and defense
    stats keep backup production -- the same rule
    walk_forward_train_and_predict(starters_only=True) follows.
    """
    return qb_starter_rows(df, schedule_context) if starters_only and position_group == "QB" else df


def walk_forward_train_and_predict(
    df: pd.DataFrame,
    defensive_target_stats: pd.DataFrame,
    player_bio: pd.DataFrame,
    schedule_context: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    position_group: str,
    season: int,
    week: int,
    lookback_years: int = 3,
    window_size: int = WINDOW_SIZE,
    targets: Optional[List[Tuple[str, str]]] = None,
    device: Optional[str] = None,
    starters_only: bool = False,
    **train_kwargs,
) -> pd.DataFrame:
    """
    The literal walk-forward step: pretend (season, week) is "right now."
    Train a fresh model using ONLY the `lookback_years` of data immediately
    before it (never anything from this week or later), then predict every
    targeted player's stat line for this exact week.

    This is the one building block both live, ad-hoc weekly projections AND
    historical backtesting are built from (see walk_forward_backtest(),
    which just calls this repeatedly across many weeks) -- call it yourself
    whenever you want a projection trained specifically on the most recent
    `lookback_years`, rather than reusing a long-lived train_all_position_models()
    model.

    Args:
        df: Full weekly stats (all position groups), covering AT LEAST
            `lookback_years` of seasons before `season` -- e.g.
            load_weekly_stats(years=[season - lookback_years, ..., season]).
        defensive_target_stats: Targeted-coverage stats covering the same
            season range as `df` -- e.g. load_defensive_target_stats(years=[...]).
        player_bio: Output of load_player_bio().
        schedule_context: Schedule/situational data covering the same
            season range as `df` -- e.g. load_schedule_context(years=[...]).
        player_age: Roster-snapshot age data covering the same range --
            e.g. load_player_age(years=[...]).
        injury_reports: Official weekly injury reports covering the same
            range -- e.g. load_injury_reports(years=[...]).
        position_group: "QB"/"RB"/"WR"/"TE".
        season, week: The game(s) being predicted -- "pretend this is now."
        lookback_years: How many years of trailing history to train on.
        window_size: Games of history per player input sequence.
        targets: [(player_id, opponent_team), ...] to predict. If not given,
            auto-derived from df's own rows for (season, week) and this
            position_group -- convenient for backtesting a week you already
            have real results for (also lets the returned DataFrame include
            actual_* columns for scoring). For a genuinely future week (not
            yet in df), pass targets explicitly, since there's no
            roster/schedule data to infer them from here. Each predicted
            player's own game-status is looked up automatically from
            `injury_reports` (no manual override here, unlike
            predict_next_game() -- this predicts many players per call, so
            a single override string wouldn't apply sensibly to all of them).
        device: "cpu"/"cuda"; autodetects if not given.
        starters_only: QB only. Train on, predict from, and (when targets is
            auto-derived) predict only listed starters' games -- see
            qb_starter_rows(). Context tables are still built from every
            row, so team and defense stats keep backup production.
        **train_kwargs: Passed through to train_position_model (epochs,
            hidden_size, etc.) -- consider fewer epochs than the default 60
            here, since this trains from scratch every time it's called
            (see walk_forward_backtest()'s epochs note).

    Returns:
        DataFrame: one row per predicted player, with player_id, player_name,
        pred_<stat> for each of the position's stat_columns, and (when
        targets was auto-derived) actual_<stat> columns too, ready to score.
        Players with fewer than window_size games of history BEFORE
        (season, week) are silently skipped (not enough sequence to predict
        from yet -- same rule as the rest of this file).

    Notes:
        Every call retrains a model from scratch on lookback_years of data
        -- there's no persistence between calls (nothing is saved to
        model_dir/). If you're calling this for the same position group
        across many consecutive weeks (as walk_forward_backtest() does),
        expect it to take roughly (per-week training time) x (number of
        weeks) -- there's no shortcut here, since each week's model must
        legitimately only have seen data from before that week.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    lookback_start_season = season - lookback_years

    def before_cutoff(frame: pd.DataFrame) -> pd.Series:
        return (frame["season"] >= lookback_start_season) & (
            (frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))
        )

    if starters_only and position_group != "QB":
        raise ValueError("starters_only is defined for QB only (schedule_context.starting_qb_id)")

    train_df = df[before_cutoff(df)]
    # What the model learns from and predicts from. Context below keeps train_df.
    model_df = qb_starter_rows(train_df, schedule_context) if starters_only else train_df
    train_target_stats = defensive_target_stats[before_cutoff(defensive_target_stats)]
    # schedule_context/player_age/injury_reports are NOT filtered by
    # before_cutoff -- unlike df/defensive_target_stats (trailing
    # PAST-performance stats that would leak the future if not cut off),
    # these are direct, known-in-advance properties of each specific game/
    # week, joined by exact (player_id, season, week) key -- a training
    # row only ever picks up ITS OWN week's age/injury/schedule facts, never
    # another week's, so nothing about a later week can leak in even
    # unfiltered (build_qb_continuity_profile() carries its own backward-only
    # leak-safety the same way -- see that function's Notes).
    top_cb_profile = build_top_cb_profile(train_target_stats, player_bio)
    context_df = build_matchup_context(train_df, top_cb_profile, schedule_context, injury_reports, player_bio)

    # Archetypes/defense_vs_archetype ARE trailing-performance-derived (like
    # df/defensive_target_stats), so -- UNLIKE schedule_context/player_age/
    # injury_reports above -- these must be built from train_df (the
    # before_cutoff-filtered slice), not the full df, or a player's future
    # games would leak into their own trailing style classification.
    from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

    archetypes = classify_archetype(compute_style_features(train_df, position_group), position_group)
    defense_vs_archetype = build_defense_vs_archetype_profile(train_df, archetypes, position_group)

    model, normalizer, context_normalizer, context_fill_values, stat_columns, _ = train_position_model(
        position_group, model_df, context_df, top_cb_profile, player_bio, player_age, injury_reports,
        archetypes, defense_vs_archetype, window_size=window_size, device=device, **train_kwargs
    )

    team_games = build_team_game_log(train_df)
    # Computed once here rather than inside current_matchup_context()'s
    # per-player call below -- it's the same league-wide computation for
    # every player in `targets` this week, see that function's
    # `opponent_tier_labels` docstring.
    opponent_tier_labels = compute_current_opponent_tiers(team_games, position_group)

    actual_rows = None
    if targets is None:
        week_rows = df[(df["season"] == season) & (df["week"] == week) & (df["position_group"] == position_group)]
        if starters_only:
            week_rows = qb_starter_rows(week_rows, schedule_context)
        targets = list(zip(week_rows["player_id"], week_rows["opponent_team"]))
        actual_rows = week_rows.set_index("player_id")

    results = []
    for player_id, opponent in targets:
        player_history = model_df[model_df["player_id"] == player_id].sort_values(["season", "week"])
        if len(player_history) < window_size:
            continue  # not enough history before this week to form an input sequence for this player

        recent = player_history.tail(window_size)[stat_columns].fillna(0.0).to_numpy(dtype=np.float32)
        x = normalizer.transform(recent)
        team = player_history["team"].iloc[-1]
        context_vec = current_matchup_context(
            train_df, team_games, train_target_stats, player_bio, schedule_context, player_age, injury_reports,
            archetypes, defense_vs_archetype,
            player_id, position_group, team, opponent, context_fill_values, opponent_tier_labels=opponent_tier_labels,
        )
        c = context_normalizer.transform(context_vec.reshape(1, -1))[0]

        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
        c_tensor = torch.from_numpy(c).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_normalized = model(x_tensor, c_tensor).cpu().numpy()[0]
        pred = np.clip(normalizer.inverse_transform(pred_normalized), a_min=0, a_max=None)

        row = {"player_id": player_id, "player_name": player_history["player_name"].iloc[-1]}
        row.update({f"pred_{col}": float(v) for col, v in zip(stat_columns, pred)})
        if actual_rows is not None and player_id in actual_rows.index:
            actual = actual_rows.loc[player_id]
            row.update({f"actual_{col}": float(actual[col]) if pd.notna(actual[col]) else None for col in stat_columns})
        results.append(row)

    return pd.DataFrame(results)


def walk_forward_ensemble_predict(
    df: pd.DataFrame,
    defensive_target_stats: pd.DataFrame,
    player_bio: pd.DataFrame,
    schedule_context: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    position_group: str,
    season: int,
    week: int,
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    progress_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    walk_forward_train_and_predict() averaged over several seeds -- the recipe for
    live projections, because it is the recipe that was validated.

    The once-a-season saved model was a lottery: across five seeds its QB
    projections followed recent form with slope 0.09 to 0.30, and the deployed seed
    was so timid that real outcomes spread ~2.5x wider than its projections
    (calibration slope 2.46 on 2026 weeks 2-4). Retraining weekly on the trailing
    seasons is far steadier (five seeds: 0.35 to 0.38), and averaging the seeds
    removes what luck remains. On the 2025 QB walk-forward (486 starter games)
    the 5-seed ensemble missed passing yards by 59.69 per game against 64.00 for
    the last-5-starts average, and on 2026 weeks 2-4 its calibration slope was
    1.06. A fitted calibration line was tried and dropped: fitted on one half of
    2025 it made the other half worse.

    Same arguments and return shape as walk_forward_train_and_predict(); actual_*
    columns pass through untouched. progress_path, when given, gets a small JSON
    note per seed ({position, season, week, seed, of, done}) for anything
    watching a long run.
    """
    if not seeds:
        raise ValueError("seeds must name at least one seed")
    def note(done_seeds: int) -> None:
        if progress_path:
            os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
            with open(progress_path, "w") as f:
                json.dump({"position": position_group, "season": season, "week": week,
                           "seed": done_seeds, "of": len(seeds), "done": done_seeds == len(seeds)}, f)

    runs = []
    for seed in seeds:
        note(len(runs))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        runs.append(walk_forward_train_and_predict(
            df, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports,
            position_group, season, week, **kwargs).set_index("player_id"))
    if any(not run.index.equals(runs[0].index) for run in runs):
        raise ValueError("seeds predicted different players; the ensemble needs identical rows")
    note(len(runs))
    out = runs[0].copy()
    pred_columns = [c for c in out.columns if c.startswith("pred_")]
    out[pred_columns] = sum(run[pred_columns] for run in runs) / len(runs)
    return out.reset_index()


def live_targets(df: pd.DataFrame, schedule_context: pd.DataFrame, position_group: str,
                 season: int, week: int) -> List[Tuple[str, str]]:
    """
    [(player_id, opponent), ...] for a week that has not been played yet, built
    only from what is known before it: the schedule's matchups and listed
    starting QBs, and who played in each team's most recent game.

    QB: the schedule's starting_qb_id for that week. Teams with no listed
    starter are skipped rather than guessed. Other positions: every player of
    the group who appeared in the team's last game before `week`.
    """
    games = schedule_context[(schedule_context["season"] == season) & (schedule_context["week"] == week)]
    if position_group == "QB":
        listed = games.dropna(subset=["starting_qb_id"])
        return list(zip(listed["starting_qb_id"], listed["opponent"]))
    past = df[(df["position_group"] == position_group) & (
        (df["season"] < season) | ((df["season"] == season) & (df["week"] < week)))]
    targets = []
    for team, opponent in zip(games["team"], games["opponent"]):
        team_rows = past[past["team"] == team]
        if team_rows.empty:
            continue
        last = team_rows.sort_values(["season", "week"]).iloc[-1]
        last_game = team_rows[(team_rows["season"] == last["season"]) & (team_rows["week"] == last["week"])]
        targets += [(pid, opponent) for pid in last_game["player_id"]]
    return targets


def attach_outcomes(out: pd.DataFrame, df: pd.DataFrame, schedule_context: pd.DataFrame,
                    position_group: str, season: int, week: int) -> pd.DataFrame:
    """
    `out` plus, for every pred_<stat>, actual_<stat> (the player's real line that
    week, absent until it is played) and l5_<stat> (mean of the player's last 5
    games before it -- the baseline the model has to beat). QB counts only
    starts, matching how the model is trained. Scoring columns only: nothing
    here feeds back into a model.
    """
    rows = df[df["position_group"] == position_group]
    if position_group == "QB":
        rows = qb_starter_rows(rows, schedule_context)
    stats = [c[len("pred_"):] for c in out.columns if c.startswith("pred_")]
    before = rows[(rows["season"] < season) | ((rows["season"] == season) & (rows["week"] < week))]
    l5 = before.sort_values(["season", "week"]).groupby("player_id").tail(5).groupby("player_id")[stats].mean()
    actual = rows[(rows["season"] == season) & (rows["week"] == week)].set_index("player_id")[stats]
    out = out.drop(columns=[c for c in out.columns if c.startswith(("actual_", "l5_"))])
    out = out.join(l5.add_prefix("l5_"), on="player_id")
    return out.join(actual.add_prefix("actual_"), on="player_id") if len(actual) else out


# Live projection = BLEND_L5 * last-5 average + (1 - BLEND_L5) * model, per position.
# WR: on the 2025 walk-forward (18 weeks, 1967 WR games) the model alone missed PPR
# by 4.775 per game against 4.636 for the last-5 average (week-block 95% CI of the
# gap +0.04 to +0.23): it ranks WRs worse than their recent average. A 0.7 blend,
# chosen on 2025, missed by 4.593 there and held on 2026 weeks 2-4, which the choice
# never saw (4.897 vs 4.940 last-5 alone and 5.114 model alone).
BLEND_L5 = {"WR": 0.7}


# Then live projection = a weighted mix of three independent sources, per position:
#   sleeper -- Sleeper's projection (Rotowire's), StatScraper.get_sleeper_projections()
#   own     -- this file's model (after the BLEND_L5 step)
#   gbm     -- GbmProjections.gbm_project_week(), gradient-boosted trees sharing no code with this file
# Weights fitted on 2025 fantasy points (Sleeper pulled after the fact, so possibly
# flattered by post-kickoff edits) and checked on 2026 weeks 2-5, which the fit never
# saw. 2026 PPR miss, own alone -> Sleeper+own -> all three: QB 6.00 -> 5.97 -> 5.90,
# RB 4.73 -> 4.49 -> 4.26, WR 4.95 -> 4.73 -> 4.70, TE 4.40 -> 4.28 -> 4.22. Fitted on
# PPR only; the same weights apply to every stat. own is 0 for RB/WR (the other two
# beat it there) but still computed: it is the fallback when a source is missing.
BLEND_WEIGHTS = {
    "QB": {"sleeper": 0.2, "own": 0.7, "gbm": 0.1},
    "RB": {"sleeper": 0.6, "own": 0.0, "gbm": 0.4},
    "WR": {"sleeper": 0.5, "own": 0.0, "gbm": 0.5},
    "TE": {"sleeper": 0.5, "own": 0.2, "gbm": 0.3},
}


# The blend aims at the typical game (the GBM's median fit pulls it there), which runs
# below the average for skewed stats -- right for a weekly pick or a prop line, low
# for a season total. mean_<stat> = pred_<stat> * MEAN_RATIOS[pos][stat] corrects for
# totals: each ratio is sum(actual) / sum(blend) over 2025. Only RB/WR/TE, the
# positions whose 2025 blend ran low (PPR bias -0.47/-0.40/-0.59; QB's ran +0.41).
# Share stats (target_share, air_yards_share, wopr) are not summed and are left out.
# 2026 weeks 2-5 season-total check, projected/actual, blend -> mean_: RB PPR 0.97 ->
# 1.03, TE PPR 0.92 -> 1.01, WR PPR 0.81 -> 0.85 (2026 WRs run lower than 2025's did).
MEAN_RATIOS = {
    "RB": {"carries": 1.092, "rushing_yards": 1.073, "rushing_tds": 1.0, "receptions": 1.09, "targets": 1.185,
           "receiving_yards": 1.198, "receiving_tds": 1.298, "fantasy_points_ppr": 1.059},
    "WR": {"receptions": 1.044, "targets": 1.048, "receiving_yards": 1.05, "receiving_tds": 1.79,
           "rushing_yards": 1.971, "rushing_tds": 1.0, "fantasy_points_ppr": 1.055},
    "TE": {"receptions": 1.074, "targets": 1.061, "receiving_yards": 1.082, "receiving_tds": 1.613,
           "fantasy_points_ppr": 1.098},
}


def add_mean_columns(out: pd.DataFrame, position_group: str) -> pd.DataFrame:
    """mean_<stat> -- the average-game version of pred_<stat>, for season totals; see MEAN_RATIOS."""
    out = out.copy()
    for stat, ratio in MEAN_RATIOS.get(position_group, {}).items():
        if f"pred_{stat}" in out:
            out[f"mean_{stat}"] = out[f"pred_{stat}"] * ratio
    return out


def blend_sources(out: pd.DataFrame, sources: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """
    pred_<stat> := weighted mean of own (the current pred_) and each named source's
    <name>_<stat>, for every stat. Each source frame has player_id and
    <name>_<stat> columns. Per player and stat, a source with no value drops out and
    the remaining weights are rescaled; if every weighted source is missing, own
    stands. Ours is kept as own_<stat>.
    """
    for name, frame in sources.items():
        out = out.merge(frame.drop_duplicates("player_id"), on="player_id", how="left")
    for col in [c for c in out.columns if c.startswith("pred_")]:
        stat = col[len("pred_"):]
        out[f"own_{stat}"] = out[col]
        total = pd.Series(0.0, index=out.index)
        weight = pd.Series(0.0, index=out.index)
        for name, w in weights.items():
            values = out[f"own_{stat}"] if name == "own" else out.get(f"{name}_{stat}")
            if values is None or w == 0:
                continue
            present = values.notna()
            total += np.where(present, w * values.fillna(0.0), 0.0)
            weight += np.where(present, w, 0.0)
        out[col] = (total / weight.where(weight > 0)).fillna(out[f"own_{stat}"])
    return out


def blend_with_last5(out: pd.DataFrame, weight: float) -> pd.DataFrame:
    """
    pred_<stat> := weight * l5_<stat> + (1 - weight) * pred_<stat>, keeping the
    model's own number as model_<stat>. A player with no last-5 value keeps the
    model's. Needs attach_outcomes()'s l5_ columns.
    """
    out = out.copy()
    for col in [c for c in out.columns if c.startswith("pred_")]:
        stat = col[len("pred_"):]
        out[f"model_{stat}"] = out[col]
        out[col] = (weight * out[f"l5_{stat}"] + (1 - weight) * out[col]).fillna(out[col])
    return out


# Share of the team's remaining games a player goes on to appear in -- injuries,
# benchings and lost roles all count -- by position and by (top half, bottom half)
# of early-season PPR per game. Measured over 2022-2025 for players with 3+ games in
# weeks 1-4, counting weeks 5-18; from week 9 the top-half rates are within 0.04.
AVAILABILITY = {"QB": (0.824, 0.649), "RB": (0.834, 0.714), "WR": (0.823, 0.717), "TE": (0.808, 0.685)}

# Per-player adjustment to that rate from his own history: share of his teams' games
# he appeared in over the two prior seasons (catches IR stints, which weekly injury
# reports stop listing). adjustment = intercept + rate_coef * prior_rate, plus
# no_history for a player with no prior-season games (rookies). Fitted on 2024
# (history 2022-23) and tested on 2025, which it never saw: miss in share of remaining
# games played 0.207 -> 0.197 for everyone, 0.183 -> 0.172 for top-half players.
# Counts of past Out/Doubtful report weeks added nothing beyond this and are left out.
AVAILABILITY_HISTORY = {"intercept": -0.214, "rate_coef": 0.265, "no_history": 0.290}

# Current injury news: multiplies the rate above by the player's status right now
# (Sleeper's live injury_status). Each is the share of remaining games players with that
# status at a week went on to play, relative to all players at that week -- 2022-2025,
# weeks 5-14, status from nflverse's weekly injury reports and rosters (RES = reserve).
# Top and bottom halves agree within ~0.07. Not on a report counts as "healthy".
INJURY_MULTIPLIER = {"healthy": 1.06, "Questionable": 0.99, "Doubtful": 0.82, "Out": 0.73, "IR": 0.27}
SLEEPER_STATUS = {"Questionable": "Questionable", "Doubtful": "Doubtful", "Out": "Out", "IR": "IR", "PUP": "IR"}


def drop_injured(targets: List[Tuple[str, str]], injuries: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    targets without players whose current status (injuries: player_id,
    injury_status from StatScraper.get_sleeper_injuries()) is Out, IR or PUP --
    they will not play this week. Returns (kept targets, dropped player_ids).
    Questionable/Doubtful players stay: most of them play.
    """
    sitting = set(injuries.loc[injuries["injury_status"].isin(["Out", "IR", "PUP"]), "player_id"])
    return [t for t in targets if t[0] not in sitting], [t[0] for t in targets if t[0] in sitting]


def rest_of_season(position_group: str, season: int, week: int, save_dir: str = "projections") -> pd.DataFrame:
    """
    Rest-of-season totals from week `week` on: each player's per-game projection
    for that week (mean_<stat> where it exists -- the average game, right for a sum
    -- else pred_<stat>) times the games he is expected to play: his team's
    remaining regular-season games (byes skipped) times his expected share of them:
    AVAILABILITY for his position and half (top or bottom half of this file's per-game
    PPR), adjusted by his own games-played history (AVAILABILITY_HISTORY). to_date_<stat> is his
    actual total before `week`; season_ = both. season_healthy_fantasy_points_ppr
    keeps the every-game-played total for comparison.
    Reads <save_dir>/<season>_w<NN>_<POS>.csv (run project_week() first) and writes
    <save_dir>/<season>_ros_<POS>.csv for the proj-charts pane.

    Then INJURY_MULTIPLIER by his current injury status, pulled live from Sleeper
    (StatScraper.get_sleeper_injuries()); if that pull fails, a warning and no
    news adjustment (multiplier 1). injury_status/availability columns show both.
    Share stats (target_share, wopr, ...) are rates, not totals, and are left out.
    """
    week_file = pd.read_csv(os.path.join(save_dir, f"{season}_w{week:02d}_{position_group}.csv"))
    stats = [s for s in POSITION_STAT_COLUMNS[position_group] if "share" not in s and s != "wopr"]
    schedule = load_schedule_context(years=[season])
    history = load_weekly_stats(years=[season])
    history = history[history["position_group"] == position_group]
    played = history[history["week"] < week]
    team = history.sort_values("week").groupby("player_id")["team"].last()
    games_left = schedule[schedule["week"] >= week].groupby("team").size()

    out = week_file[["player_id", "player_name"]].copy()
    out["team"] = out["player_id"].map(team)
    out["games_left"] = out["team"].map(games_left).fillna(0).astype(int)
    ppr = week_file.get("mean_fantasy_points_ppr", week_file["pred_fantasy_points_ppr"])
    top, bottom = AVAILABILITY[position_group]
    base = np.where(ppr.to_numpy() >= ppr.median(), top, bottom)
    prior = load_weekly_stats(years=[season - 2, season - 1])
    prior_sched = load_schedule_context(years=[season - 2, season - 1])
    team_games = prior_sched.groupby(["season", "team"]).size()
    seasons = prior.groupby(["player_id", "season"]).agg(gp=("week", "size"), team=("team", "last")).reset_index()
    seasons["tg"] = [team_games.get((s_, t), 17) for s_, t in zip(seasons["season"], seasons["team"])]
    totals = seasons.groupby("player_id")[["gp", "tg"]].sum()
    prior_rate = out["player_id"].map(totals["gp"] / totals["tg"])
    h = AVAILABILITY_HISTORY
    adjust = np.where(prior_rate.isna(), h["intercept"] + h["no_history"],
                      h["intercept"] + h["rate_coef"] * prior_rate.fillna(0.0))
    out["prior_games_share"] = prior_rate
    try:
        news = get_sleeper_injuries(db_path=DB_PATH).set_index("player_id")["injury_status"]
        out["injury_status"] = out["player_id"].map(news).map(SLEEPER_STATUS).fillna("healthy")
        multiplier = out["injury_status"].map(INJURY_MULTIPLIER).to_numpy()
    except Exception as ex:  # network or an API change: no news adjustment
        print(f"Sleeper injury news unavailable ({type(ex).__name__}: {ex}); no news adjustment.")
        out["injury_status"] = None
        multiplier = 1.0
    out["availability"] = np.clip((base + adjust) * multiplier, 0.0, 1.0)
    out["expected_games"] = out["games_left"] * out["availability"]
    to_date = played.groupby("player_id")[stats].sum()
    for stat in stats:
        per_game = week_file.get(f"mean_{stat}", week_file[f"pred_{stat}"])
        out[f"per_game_{stat}"] = per_game.to_numpy()
        out[f"ros_{stat}"] = out[f"per_game_{stat}"] * out["expected_games"]
        out[f"to_date_{stat}"] = out["player_id"].map(to_date[stat]).fillna(0.0)
        out[f"season_{stat}"] = out[f"to_date_{stat}"] + out[f"ros_{stat}"]
    out["season_healthy_fantasy_points_ppr"] = (out["to_date_fantasy_points_ppr"]
                                                + out["per_game_fantasy_points_ppr"] * out["games_left"])
    out = out.sort_values("ros_fantasy_points_ppr", ascending=False)
    out.to_csv(os.path.join(save_dir, f"{season}_ros_{position_group}.csv"), index=False)
    return out


def project_week(position_group: str, season: int, week: int, lookback_years: int = 3,
                 seeds: Tuple[int, ...] = (0, 1, 2, 3, 4), epochs: int = 30,
                 save_dir: Optional[str] = "projections", blend: bool = True, skip_injured: bool = False,
                 **kwargs) -> pd.DataFrame:
    """
    Live projections for every player of `position_group` in an upcoming
    (season, week): loads the trailing data, picks targets with live_targets(),
    and runs walk_forward_ensemble_predict() -- the validated recipe (QB on
    starters only, 30 epochs, 5 seeds). Use this instead of predict_next_game()
    and its once-a-season saved model, whose QB projections were far too timid.

    Positions in BLEND_L5 get pred_ blended with the last-5 average (the model's
    own number kept as model_<stat>), then every position is mixed with Sleeper's
    projection and the GBM's per BLEND_WEIGHTS (blend_sources()). A source that
    fails (Sleeper's network pull, say) prints a warning and drops out of the mix --
    the projections are never lost to it. blend=False skips that step.

    skip_injured=True (live runs only -- weekly_props.py) leaves out players Sleeper
    lists right now as Out, IR or PUP (drop_injured()). Off by default because a
    past week's backtest must not use today's injury news. For RB/WR/TE
    the blend also gets mean_<stat>, its average-game version for season totals
    (add_mean_columns()); use pred_ for a weekly pick or against a prop line. Adds actual_/l5_ columns (attach_outcomes()) and, unless save_dir is None,
    writes <save_dir>/<season>_w<NN>_<POS>.csv -- what the proj-charts pane reads.
    Re-run a week after it is played to fill in its actuals.
    """
    years = list(range(season - lookback_years, season + 1))
    df = load_weekly_stats(years=years)
    schedule_context = load_schedule_context(years=years)
    targets = live_targets(df, schedule_context, position_group, season, week)
    if skip_injured:
        try:
            targets, dropped = drop_injured(targets, get_sleeper_injuries(db_path=DB_PATH))
            if dropped:
                names = df.drop_duplicates("player_id").set_index("player_id")["player_name"]
                print(f"{position_group}: skipping {len(dropped)} listed Out/IR/PUP: "
                      + ", ".join(str(names.get(p, p)) for p in dropped))
        except Exception as ex:  # network or an API change: project everyone rather than nobody
            print(f"Sleeper injury news unavailable ({type(ex).__name__}: {ex}); not skipping injured players.")
    out = walk_forward_ensemble_predict(
        df, load_defensive_target_stats(years=years), load_player_bio(), schedule_context,
        load_player_age(years=years), load_injury_reports(years=years), position_group, season, week,
        seeds=seeds, lookback_years=lookback_years, targets=targets,
        starters_only=position_group == "QB", epochs=epochs,
        progress_path=os.path.join(save_dir, ".progress.json") if save_dir else None, **kwargs)
    out = attach_outcomes(out, df, schedule_context, position_group, season, week)
    if position_group in BLEND_L5:
        out = blend_with_last5(out, BLEND_L5[position_group])
    if blend and position_group in BLEND_WEIGHTS:
        sources = {}
        try:
            theirs = get_sleeper_projections(season, week, db_path=DB_PATH)
            theirs = theirs[theirs["position"] == position_group]
            sources["sleeper"] = theirs[["player_id"] + [c for c in theirs if c.startswith("sleeper_")]]
        except Exception as ex:  # network or an API change: blend without it
            print(f"Sleeper projections unavailable ({type(ex).__name__}: {ex}); blending without them.")
        try:
            gbm = gbm_project_week(df, schedule_context, position_group, season, week, targets,
                                   POSITION_STAT_COLUMNS[position_group])
            sources["gbm"] = gbm.drop(columns="player_name").rename(columns=lambda c: c.replace("pred_", "gbm_"))
        except Exception as ex:
            print(f"GBM projections failed ({type(ex).__name__}: {ex}); blending without them.")
        out = blend_sources(out, sources, BLEND_WEIGHTS[position_group])
        out = add_mean_columns(out, position_group)
    if save_dir:
        out.to_csv(os.path.join(save_dir, f"{season}_w{week:02d}_{position_group}.csv"), index=False)
    return out


def walk_forward_backtest(
    df: pd.DataFrame,
    defensive_target_stats: pd.DataFrame,
    player_bio: pd.DataFrame,
    schedule_context: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    position_group: str,
    eval_weeks: List[Tuple[int, int]],
    lookback_years: int = 3,
    window_size: int = WINDOW_SIZE,
    epochs: int = 30,
    device: Optional[str] = None,
    **train_kwargs,
) -> pd.DataFrame:
    """
    Run walk_forward_train_and_predict() across many (season, week) pairs IN
    ORDER, retraining from scratch each time on the trailing `lookback_years`
    -- exactly the "start with the oldest week, predict the next one, fold
    it in, repeat" process, automated across a whole evaluation stretch
    instead of one week at a time by hand.

    Args:
        df: Full weekly stats, covering AT LEAST lookback_years before the
            EARLIEST (season, week) in eval_weeks.
        defensive_target_stats: Targeted-coverage stats covering the same
            season range as `df`.
        player_bio: Output of load_player_bio().
        schedule_context: Schedule/situational data covering the same
            season range as `df`.
        player_age: Output of load_player_age(), same range.
        injury_reports: Output of load_injury_reports(), same range.
        position_group: "QB"/"RB"/"WR"/"TE".
        eval_weeks: Ordered [(season, week), ...] to walk forward through --
            e.g. [(2025, w) for w in range(1, 19)] for a full season.
        lookback_years, window_size: See walk_forward_train_and_predict().
        epochs: Training epochs PER STEP -- kept lower than the usual
            default of 60 since this retrains once per entry in eval_weeks
            (e.g. a full 18-week season at 60 epochs/step would be ~18x
            slower than one normal training run); 30 is a reasonable
            speed/quality trade for a backtest -- raise it if you have time
            to spare and want closer-to-production-quality weekly models.
        device: "cpu"/"cuda"; autodetects if not given.
        **train_kwargs: Passed through to train_position_model each step.

    Returns:
        DataFrame with one row per (season, week, player) prediction plus
        actual_* columns (since targets are auto-derived from `df`'s real
        results for each week), concatenated across every step. Pass this
        to summarize_backtest() for a ready-made per-stat MAE report.
    """
    all_results = []
    for season, week in eval_weeks:
        print(f"--- Walk-forward step: {position_group} {season} week {week} ---")
        step_df = walk_forward_train_and_predict(
            df, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports, position_group, season, week,
            lookback_years=lookback_years, window_size=window_size, epochs=epochs, device=device, **train_kwargs
        )
        if not step_df.empty:
            step_df["season"] = season
            step_df["week"] = week
        all_results.append(step_df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def summarize_backtest(results: pd.DataFrame, stat_columns: List[str]) -> pd.Series:
    """
    Turn walk_forward_backtest()'s raw per-player predictions into a
    per-stat Mean Absolute Error summary -- the "how close did it actually
    get" number, in real units, across the whole backtest period. This is
    the honest, no-leakage accuracy estimate: every prediction it's based on
    came from a model that had genuinely never seen that week's (or any
    later week's) data during training.

    Args:
        results: Output of walk_forward_backtest().
        stat_columns: Which stat columns to summarize (e.g.
            POSITION_STAT_COLUMNS["QB"]).

    Returns:
        Series indexed by stat column, mean absolute error in real units.
        Rows with a missing actual_<stat> (e.g. a player who ended up not
        playing that week) are excluded from that column's average.
    """
    maes = {}
    for col in stat_columns:
        pred_col, actual_col = f"pred_{col}", f"actual_{col}"
        if pred_col in results.columns and actual_col in results.columns:
            valid = results[[pred_col, actual_col]].dropna()
            if len(valid):
                maes[col] = (valid[pred_col] - valid[actual_col]).abs().mean()
    return pd.Series(maes)


# ---------------------------------------------------------------------------
# MONTE CARLO -- simulate a distribution of outcomes from real backtested error
# ---------------------------------------------------------------------------
# Which predicted stat to bucket the residual pool by (see
# build_residual_pool()) -- present in every position's POSITION_STAT_COLUMNS,
# so it's a single consistent "how big a game is this projected to be" proxy
# across QB/RB/WR/TE.
RESIDUAL_MAGNITUDE_STAT = "fantasy_points_ppr"


def build_residual_pool(
    results: pd.DataFrame,
    stat_columns: List[str],
    n_buckets: int = 3,
    magnitude_stat: str = RESIDUAL_MAGNITUDE_STAT,
) -> dict:
    """
    Turn walk_forward_backtest()'s raw predictions into a residual pool for
    Monte Carlo simulation: real, leakage-free (actual - predicted) error
    VECTORS -- one per backtested player-week, covering every stat_columns
    entry at once -- bucketed by how big a game was predicted so a confident
    projection doesn't get the same spread as a volatile one.

    Args:
        results: Output of walk_forward_backtest() -- needs pred_<col> and
            actual_<col> for every col in stat_columns AND for
            magnitude_stat (used for bucketing, not necessarily also in
            stat_columns, though normally it is).
        stat_columns: Which stat columns to build residual vectors for
            (e.g. POSITION_STAT_COLUMNS[position_group]) -- same order
            predict_next_game_monte_carlo() will add them back onto a point
            prediction in.
        n_buckets: How many predicted-magnitude buckets to split into
            (default 3: low/mid/high). Buckets are equal-COUNT (split on
            quantiles of magnitude_stat's predicted value), not equal-width,
            so each one has roughly the same amount of data to bootstrap from.
        magnitude_stat: Which predicted stat to bucket by (default
            RESIDUAL_MAGNITUDE_STAT).

    Returns:
        dict: {"stat_columns", "magnitude_stat", "bucket_edges" (n_buckets-1
        quantile cutoffs), "buckets" (list of n_buckets lists, each a list
        of [residual_per_stat_column, ...] vectors)} -- JSON-serializable,
        pass to save_residual_pool() / predict_next_game_monte_carlo().

    Notes:
        Residual VECTORS, not independent per-column samples: a QB's
        yardage error and TD error move together in real games (a big game
        is big across stats, not randomly big in one and flat in another),
        so bootstrapping a whole historical row at once keeps simulated
        stat lines physically consistent -- sampling each column's noise
        independently could hand back e.g. 350 passing yards with 0 TDs, a
        combination that essentially never occurs in the real error data.

        These residuals come specifically from walk_forward_backtest()'s
        output because every prediction in it was produced by a model that
        had genuinely never seen that week (or any later week) during
        training -- the same no-lookahead guarantee summarize_backtest()
        relies on for its MAE numbers. A residual pool built from a single
        train/val split's leftover examples would reflect one model's luck
        on one held-out stretch rather than real out-of-sample error.
    """
    pred_cols = [f"pred_{c}" for c in stat_columns]
    actual_cols = [f"actual_{c}" for c in stat_columns]
    magnitude_col = f"pred_{magnitude_stat}"
    valid = results.dropna(subset=pred_cols + actual_cols + [magnitude_col])

    residuals = valid[actual_cols].to_numpy() - valid[pred_cols].to_numpy()
    magnitude = valid[magnitude_col].to_numpy()

    bucket_edges = np.quantile(magnitude, np.linspace(0, 1, n_buckets + 1)[1:-1]).tolist()
    bucket_idx = np.digitize(magnitude, bucket_edges)
    buckets = [residuals[bucket_idx == b].tolist() for b in range(n_buckets)]

    return {
        "stat_columns": stat_columns,
        "magnitude_stat": magnitude_stat,
        "bucket_edges": bucket_edges,
        "buckets": buckets,
    }


def save_residual_pool(pool: dict, position_group: str, model_dir: str = MODEL_DIR) -> None:
    """
    Save a residual pool (see build_residual_pool()) alongside a position's
    trained model, for predict_next_game_monte_carlo() to load later.

    Args:
        pool: Output of build_residual_pool().
        position_group: "QB"/"RB"/"WR"/"TE" -- used to name the output file.
        model_dir: Directory to save into (same one save_model() uses).
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{position_group.lower()}_residuals.json")
    with open(path, "w") as f:
        json.dump(pool, f)

    total = sum(len(b) for b in pool["buckets"])
    print(f"Saved {position_group} residual pool ({total} vectors across {len(pool['buckets'])} buckets) -> {path}")


def load_residual_pool(position_group: str, model_dir: str = MODEL_DIR) -> dict:
    """
    Load a previously saved residual pool.

    Raises:
        FileNotFoundError: if none has been built for this position yet --
            run walk_forward_backtest() + build_residual_pool() +
            save_residual_pool() first (see build_all_residual_pools()).
    """
    path = os.path.join(model_dir, f"{position_group.lower()}_residuals.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No residual pool for {position_group!r} in {model_dir}/ -- build one with "
            f"walk_forward_backtest() + build_residual_pool() + save_residual_pool() "
            f"(or build_all_residual_pools()) first."
        )
    with open(path) as f:
        return json.load(f)


def build_all_residual_pools(
    years: Optional[List[int]] = None,
    eval_weeks: Optional[List[Tuple[int, int]]] = None,
    lookback_years: int = 3,
    epochs: int = 30,
    model_dir: str = MODEL_DIR,
    **backtest_kwargs,
) -> None:
    """
    Build and save a residual pool for every position in POSITION_STAT_COLUMNS
    -- the walk_forward_backtest() + build_residual_pool() + save_residual_pool()
    pipeline, run once per position. Mirrors train_all_position_models()'s
    "do all four positions" convenience, but for the Monte Carlo residual
    pools instead of the models themselves.

    Args:
        years: Passed to load_weekly_stats()/load_defensive_target_stats()/
            load_schedule_context() -- must cover at least lookback_years
            before the earliest eval_weeks entry. Defaults to the last 3
            seasons if not given (same as those loaders).
        eval_weeks: [(season, week), ...] to backtest over -- more weeks
            means a bigger, more reliable residual pool, at the cost of
            retraining a model per step. Defaults to the last 6 weeks with
            data in `years` if not given.
        lookback_years, epochs: Passed to walk_forward_backtest() -- see
            its docstring (epochs defaults lower than train_position_model's
            60 since this retrains from scratch every step).
        model_dir: Directory to save each position's residual pool into.
        **backtest_kwargs: Passed through to walk_forward_backtest() (e.g.
            window_size, hidden_size).

    Notes:
        This is independent of train_all_position_models()/save_model() --
        it doesn't touch or require an already-saved model on disk, it just
        needs its own transient walk-forward-trained models to generate
        out-of-sample residuals. Run it any time after enough weeks of a
        season have real results to backtest against.
    """
    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)

    if eval_weeks is None:
        recent = df[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        eval_weeks = list(recent.tail(6).itertuples(index=False, name=None))

    for position_group, stat_columns in POSITION_STAT_COLUMNS.items():
        print(f"\n=== Building {position_group} residual pool ({len(eval_weeks)} backtest weeks) ===")
        results = walk_forward_backtest(
            df, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports, position_group,
            eval_weeks=eval_weeks, lookback_years=lookback_years, epochs=epochs, **backtest_kwargs,
        )
        pool = build_residual_pool(results, stat_columns)
        save_residual_pool(pool, position_group, model_dir)


def predict_next_game_monte_carlo(
    player_id: str,
    next_opponent: str,
    n_samples: int = 2000,
    percentiles: Tuple[float, ...] = (25, 50, 75),
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    archetypes: Optional[pd.DataFrame] = None,
    defense_vs_archetype: Optional[pd.DataFrame] = None,
    position_group: Optional[str] = None,
    model_dir: str = MODEL_DIR,
    device: Optional[str] = None,
    min_bucket_size: int = 20,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate a distribution of outcomes for one player's next game: get the
    normal point prediction (see predict_next_game()), then Monte Carlo
    simulate n_samples realistic stat lines by bootstrapping real,
    leakage-free backtested error vectors (see build_residual_pool()) onto
    it, and summarize as percentile stat lines -- floor/median/ceiling by
    default.

    Args:
        player_id, next_opponent: Same as predict_next_game().
        n_samples: How many Monte Carlo draws (default 2000).
        percentiles: Which percentiles to summarize each stat's simulated
            distribution at (default (25, 50, 75) -- floor/median/ceiling).
        df, defensive_target_stats, player_bio, schedule_context, player_age,
            injury_reports, injury_status, position_group, model_dir, device:
            Same as predict_next_game() -- this calls it internally for the
            point prediction, and (unlike predict_next_game() alone)
            resolves/loads the DataFrames here first so both the point
            prediction and the residual-pool lookup share one already-loaded
            position_group/DataFrame set instead of predict_next_game()
            re-deriving its own copy.
        min_bucket_size: If the point prediction's magnitude bucket has
            fewer than this many residual vectors to sample from, fall back
            to the full pool (all buckets combined) instead -- too few
            examples in one bucket means bootstrapping the same handful of
            games over and over, understating the true spread.
        seed: Optional RNG seed for reproducible simulations.

    Returns:
        DataFrame indexed by stat column, with a "point" column (the
        underlying predict_next_game() value) plus one column per requested
        percentile, e.g. "p25"/"p50"/"p75" -- row "passing_yards" might read
        point=241.3, p25=198.1, p50=239.4, p75=281.0.

    Raises:
        ValueError: if the player has no rows in `df` (propagated from
            predict_next_game()), or if the saved residual pool's
            stat_columns don't match the point prediction's (e.g. the model
            was retrained with a different POSITION_STAT_COLUMNS entry
            since the pool was last built).
        FileNotFoundError: if no residual pool has been built for this
            position yet (propagated from load_residual_pool()) -- see
            build_all_residual_pools().
    """
    if df is None:
        df = load_weekly_stats()
    if defensive_target_stats is None:
        defensive_target_stats = load_defensive_target_stats()
    if player_bio is None:
        player_bio = load_player_bio()
    if schedule_context is None:
        schedule_context = load_schedule_context()
    if player_age is None:
        player_age = load_player_age()
    if injury_reports is None:
        injury_reports = load_injury_reports()

    player_rows = df[df["player_id"] == player_id]
    if player_rows.empty:
        raise ValueError(f"No games found for player_id={player_id!r}")
    if position_group is None:
        position_group = player_rows["position_group"].iloc[-1]
    if archetypes is None or defense_vs_archetype is None:
        from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

        if archetypes is None:
            archetypes = classify_archetype(compute_style_features(df, position_group), position_group)
        if defense_vs_archetype is None:
            defense_vs_archetype = build_defense_vs_archetype_profile(df, archetypes, position_group)

    point_pred = predict_next_game(
        player_id, next_opponent, df=df, defensive_target_stats=defensive_target_stats,
        player_bio=player_bio, schedule_context=schedule_context, player_age=player_age,
        injury_reports=injury_reports, injury_status=injury_status, archetypes=archetypes,
        defense_vs_archetype=defense_vs_archetype, position_group=position_group,
        model_dir=model_dir, device=device,
    )

    pool = load_residual_pool(position_group, model_dir)
    stat_columns = pool["stat_columns"]
    if stat_columns != list(point_pred.keys()):
        raise ValueError(
            f"Residual pool stat_columns {stat_columns} don't match predict_next_game()'s "
            f"{list(point_pred.keys())} for {position_group!r} -- rebuild the residual pool "
            f"(build_all_residual_pools()) after retraining this position's model."
        )

    point_vec = np.array([point_pred[c] for c in stat_columns], dtype=np.float64)

    magnitude_idx = stat_columns.index(pool["magnitude_stat"])
    bucket_idx = int(np.digitize(point_vec[magnitude_idx], pool["bucket_edges"]))
    bucket = pool["buckets"][bucket_idx]
    if len(bucket) < min_bucket_size:
        bucket = [row for b in pool["buckets"] for row in b]

    rng = np.random.default_rng(seed)
    residual_samples = np.array(bucket, dtype=np.float64)
    draws = residual_samples[rng.integers(0, len(residual_samples), size=n_samples)]
    simulated = np.clip(point_vec[None, :] + draws, a_min=0, a_max=None)

    summary = {"point": point_vec}
    for p in percentiles:
        summary[f"p{p:g}"] = np.percentile(simulated, p, axis=0)

    return pd.DataFrame(summary, index=stat_columns)


# ---------------------------------------------------------------------------
# ENSEMBLE -- 2 MSE "likely" models (LSTM + Transformer) + 2 quantile models
# (floor/ceiling), combined into one floor/likely/ceiling projection
# ---------------------------------------------------------------------------
def predict_next_game_ensemble(
    player_id: str,
    next_opponent: str,
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    position_group: Optional[str] = None,
    model_dir: str = MODEL_DIR,
    transformer_model_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Combine 4 independently-trained models into one floor/likely/ceiling
    projection -- this project's 2 different point-estimate architectures
    (PlayerStatLSTM and the Transformer counterpart in
    PlayerProjectionsTransformer.py, both trained with ordinary MSE, see
    train_position_model()) averaged into a single "likely" estimate, plus
    2 pinball-loss quantile models (see train_position_model()'s `quantile`
    param / train_all_quantile_models()) providing a directly-learned
    floor and ceiling.

    This is a different, complementary way to get a range of outcomes from
    predict_next_game_monte_carlo(): Monte Carlo bootstraps real backtested
    error onto a single point estimate (bucketed by predicted magnitude into
    just 3 buckets); this ensemble instead trains dedicated models to
    directly learn the floor/ceiling conditioned on the FULL feature set for
    each specific prediction, which can in principle capture finer-grained,
    genuinely heteroscedastic uncertainty than a 3-bucket bootstrap can. The
    "likely" value here also benefits from ensembling two differently-biased
    architectures, which predict_next_game_monte_carlo() alone doesn't get
    (it's built on the LSTM only). Neither approach makes the other
    redundant -- compare them if you're unsure which to trust more.

    Args:
        player_id, next_opponent, df, defensive_target_stats, player_bio,
            schedule_context, player_age, injury_reports, injury_status,
            position_group, device: Same as predict_next_game().
        model_dir: Directory the LSTM and quantile models live in (same
            MODEL_DIR train_all_position_models()/train_all_quantile_models()
            save into).
        transformer_model_dir: Directory the Transformer model lives in.
            Defaults to PlayerProjectionsTransformer.TRANSFORMER_MODEL_DIR
            if not given.

    Returns:
        DataFrame indexed by stat column, columns: "likely_lstm" (the
        PlayerStatLSTM's own prediction), "likely_transformer" (the
        Transformer's), "likely" (their average -- the ensemble estimate),
        "floor" (the low-quantile model's prediction), "ceiling" (the
        high-quantile model's). floor/ceiling are clamped against "likely"
        (floor = min(floor, likely), ceiling = max(ceiling, likely)) to
        guard against quantile crossing -- see Notes.

    Raises:
        ValueError: if the player has no rows in `df` (propagated from
            predict_next_game()).
        FileNotFoundError: if any of the 4 required models (standard LSTM,
            Transformer, low quantile, high quantile) hasn't been trained
            yet for this position -- see train_all_position_models(),
            PlayerProjectionsTransformer.train_all_position_models_hf(),
            and train_all_quantile_models().

    Notes:
        Quantile crossing: the low/high quantile models are trained
        completely independently (separate pinball-loss optimizations), so
        nothing mathematically guarantees floor <= ceiling, or that either
        stays on the correct side of "likely" -- this is a well-known
        quantile-regression phenomenon, not a bug in this implementation.
        The clamp against "likely" above is a practical guard, not a fix
        for miscalibration -- if you see the floor and ceiling clamped to
        the same value often, the quantile models may need more training
        data/epochs, not just a tighter clamp.
    """
    from PlayerProjectionsTransformer import predict_next_game_hf, TRANSFORMER_MODEL_DIR

    if df is None:
        df = load_weekly_stats()
    if defensive_target_stats is None:
        defensive_target_stats = load_defensive_target_stats()
    if player_bio is None:
        player_bio = load_player_bio()
    if schedule_context is None:
        schedule_context = load_schedule_context()
    if player_age is None:
        player_age = load_player_age()
    if injury_reports is None:
        injury_reports = load_injury_reports()
    transformer_model_dir = transformer_model_dir or TRANSFORMER_MODEL_DIR

    player_rows = df[df["player_id"] == player_id]
    if player_rows.empty:
        raise ValueError(f"No games found for player_id={player_id!r}")
    if position_group is None:
        position_group = player_rows["position_group"].iloc[-1]

    from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

    archetypes = classify_archetype(compute_style_features(df, position_group), position_group)
    defense_vs_archetype = build_defense_vs_archetype_profile(df, archetypes, position_group)

    common_kwargs = dict(
        df=df, defensive_target_stats=defensive_target_stats, player_bio=player_bio,
        schedule_context=schedule_context, player_age=player_age, injury_reports=injury_reports,
        injury_status=injury_status, archetypes=archetypes, defense_vs_archetype=defense_vs_archetype,
        position_group=position_group, device=device,
    )

    likely_lstm = predict_next_game(player_id, next_opponent, model_dir=model_dir, **common_kwargs)
    likely_transformer = predict_next_game_hf(player_id, next_opponent, model_dir=transformer_model_dir, **common_kwargs)
    floor = predict_next_game(
        player_id, next_opponent, model_dir=model_dir, model_name=f"{position_group.lower()}_low", **common_kwargs
    )
    ceiling = predict_next_game(
        player_id, next_opponent, model_dir=model_dir, model_name=f"{position_group.lower()}_high", **common_kwargs
    )

    stat_columns = list(likely_lstm.keys())
    likely_lstm_vec = np.array([likely_lstm[c] for c in stat_columns])
    likely_transformer_vec = np.array([likely_transformer[c] for c in stat_columns])
    likely_vec = (likely_lstm_vec + likely_transformer_vec) / 2
    floor_vec = np.minimum(np.array([floor[c] for c in stat_columns]), likely_vec)
    ceiling_vec = np.maximum(np.array([ceiling[c] for c in stat_columns]), likely_vec)

    return pd.DataFrame({
        "likely_lstm": likely_lstm_vec,
        "likely_transformer": likely_transformer_vec,
        "likely": likely_vec,
        "floor": floor_vec,
        "ceiling": ceiling_vec,
    }, index=stat_columns)


# ---------------------------------------------------------------------------
# GENETIC ALGORITHM TRAINING -- an evolved population instead of gradient
# descent, for a 3-model-per-position ensemble whose weights persist and
# can keep evolving across later calls instead of retraining from scratch
# ---------------------------------------------------------------------------
# HONEST TRADEOFF: gradient descent (every other model in this file) uses
# the actual gradient of the loss to know which direction improves the
# model; a genetic algorithm only knows "this individual scored better than
# that one" and searches by random mutation + recombination. For a
# differentiable model this size (tens of thousands of continuous weights),
# GD is essentially always more sample-efficient and reaches a better
# optimum in far fewer iterations. This section exists because it was
# explicitly asked for as a genuine alternative training paradigm -- not
# because it's expected to out-predict train_position_model()'s
# gradient-descent models. Compare its backtested accuracy against theirs
# before trusting it over them.
def _random_individual(num_features: int, context_size: int, hidden_size: int, num_layers: int, device: str) -> PlayerStatLSTM:
    """A freshly, randomly initialized PlayerStatLSTM -- one population member."""
    return PlayerStatLSTM(num_features=num_features, context_size=context_size, hidden_size=hidden_size, num_layers=num_layers).to(device)


def _crossover_models(parent_a: PlayerStatLSTM, parent_b: PlayerStatLSTM) -> PlayerStatLSTM:
    """
    Uniform crossover: the child's architecture is a deep copy of parent_a
    (same shapes), but each individual weight independently comes from
    parent_a or parent_b with 50/50 probability -- not whole layers from
    one parent or the other, so a child can combine fine-grained traits
    from both.
    """
    child = copy.deepcopy(parent_a)
    a_state, b_state = parent_a.state_dict(), parent_b.state_dict()
    child_state = {}
    with torch.no_grad():
        for key in a_state:
            mask = torch.rand_like(a_state[key]) < 0.5
            child_state[key] = torch.where(mask, a_state[key], b_state[key])
    child.load_state_dict(child_state)
    return child


def _mutate_model(model: PlayerStatLSTM, std: float = GA_MUTATION_STD, rate: float = GA_MUTATION_RATE) -> None:
    """In-place: add Gaussian noise (std) to a random `rate` fraction of each parameter tensor's elements."""
    with torch.no_grad():
        for param in model.parameters():
            mask = (torch.rand_like(param) < rate).float()
            param.add_(mask * torch.randn_like(param) * std)


def _evaluate_fitness(model: PlayerStatLSTM, data_loader: DataLoader, device: str) -> float:
    """Fitness = mean recency-weighted MSE over data_loader (lower is better) -- no gradient computed, just a forward pass."""
    model.eval()
    total_loss, total_weight = 0.0, 0.0
    with torch.no_grad():
        for xb, cb, yb, wb in data_loader:
            xb, cb, yb, wb = xb.to(device), cb.to(device), yb.to(device), wb.to(device)
            per_example = ((model(xb, cb) - yb) ** 2).mean(dim=1)
            total_loss += (per_example * wb).sum().item()
            total_weight += wb.sum().item()
    return total_loss / max(total_weight, 1e-8)


def train_position_model_ga(
    position_group: str,
    df: pd.DataFrame,
    context_df: pd.DataFrame,
    top_cb_profile: pd.DataFrame,
    player_bio: pd.DataFrame,
    player_age: pd.DataFrame,
    injury_reports: pd.DataFrame,
    archetypes: pd.DataFrame,
    defense_vs_archetype: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    generations: int = GA_DEFAULT_GENERATIONS,
    population_size: int = GA_POPULATION_SIZE,
    survivors: int = GA_SURVIVORS,
    ensemble_size: int = GA_ENSEMBLE_SIZE,
    mutation_std: float = GA_MUTATION_STD,
    mutation_rate: float = GA_MUTATION_RATE,
    hidden_size: int = 64,
    num_layers: int = 2,
    val_frac: float = 0.15,
    device: Optional[str] = None,
    resume_population: Optional[List[PlayerStatLSTM]] = None,
) -> Tuple[List[PlayerStatLSTM], List[PlayerStatLSTM], Normalizer, Normalizer, Dict[str, float], List[str], List[dict]]:
    """
    Evolve a population of PlayerStatLSTM individuals for one position group
    by selection + crossover + mutation instead of backpropagation -- the
    genetic-algorithm counterpart to train_position_model(). Same data prep
    (leak-safe context building, sequence construction, chronological
    split, normalization, recency weighting) is identical and reused
    directly; only HOW the weights are optimized differs. See this
    section's header Notes for the tradeoff against gradient descent.

    Args:
        position_group, df, context_df, top_cb_profile, player_bio,
            player_age, injury_reports, archetypes, defense_vs_archetype,
            window_size, val_frac: Same meaning as train_position_model().
        generations: How many selection/crossover/mutation rounds to run
            this call (default GA_DEFAULT_GENERATIONS). Pass
            resume_population (below) across multiple calls to keep
            evolving past what one call covers, rather than raising this
            arbitrarily high in one run.
        population_size: Individuals per generation (default GA_POPULATION_SIZE).
        survivors: Top-fitness individuals kept as parents each generation
            (default GA_SURVIVORS).
        ensemble_size: How many of the FINAL population's fittest
            individuals to return as `top_models` (default GA_ENSEMBLE_SIZE
            -- the "3 models" this project's ensemble uses per position).
        mutation_std, mutation_rate: See _mutate_model().
        hidden_size, num_layers: Architecture sizing for each individual
            (only used when NOT resuming -- resumed individuals keep
            whatever architecture they were already built with).
        device: "cpu"/"cuda"; autodetects if not given.
        resume_population: A previously saved population (see
            load_population()) to continue evolving from generation 0 of
            THIS call, instead of starting from random individuals -- this
            is what makes "wait a few games, then keep going" possible:
            each call's generations pick up where the last call's left off,
            rather than re-evolving from scratch every time. Population
            size must match `population_size` if given.

    Returns:
        (top_models, population, normalizer, context_normalizer,
        context_fill_values, stat_columns, history) -- top_models is the
        fittest `ensemble_size` individuals from the FINAL generation
        (what predict_next_game_ga() actually uses); population is the
        FULL final population (save this via save_population() if you want
        to resume evolution later -- top_models alone isn't enough to
        resume from, since it discards the rest of the gene pool).
        history is a list of per-generation dicts ({"generation",
        "best_fitness", "mean_fitness"}), fitness in recency-weighted MSE
        (lower is better, comparable across generations but NOT directly
        to train_position_model()'s MSE+Poisson loss, which includes the
        extra Poisson term GA fitness doesn't).

    Raises:
        ValueError: same conditions as train_position_model() (unknown
            position_group, or fewer than 20 training examples), or if
            resume_population's size doesn't match population_size.
    """
    if resume_population is not None and len(resume_population) != population_size:
        raise ValueError(f"resume_population has {len(resume_population)} individuals, expected population_size={population_size}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (
        stat_columns, X_train, C_train, y_train, X_val, C_val, y_val,
        normalizer, context_normalizer, context_fill_values, weights_train, weights_val,
    ) = prepare_position_training_data(
        position_group, df, context_df, top_cb_profile, player_bio, player_age,
        injury_reports, archetypes, defense_vs_archetype, window_size, val_frac,
    )

    # GA fitness only needs forward passes (no gradients), so batch size can
    # be large -- one batch per split is fine and keeps _evaluate_fitness()
    # fast across population_size individuals x generations calls. (train_ds
    # isn't needed -- GA never trains via backprop mini-batches on it.)
    val_ds = PlayerGameSequenceDataset(
        normalizer.transform(X_val), context_normalizer.transform(C_val), normalizer.transform(y_val), weights_val
    )
    val_loader = DataLoader(val_ds, batch_size=max(len(val_ds), 1), shuffle=False)

    num_features = len(stat_columns)
    context_size = len(CONTEXT_COLUMNS)
    if resume_population is not None:
        population = resume_population
    else:
        population = [_random_individual(num_features, context_size, hidden_size, num_layers, device) for _ in range(population_size)]

    history = []
    for generation in range(1, generations + 1):
        fitness = [_evaluate_fitness(ind, val_loader, device) for ind in population]
        ranked = sorted(zip(population, fitness), key=lambda p: p[1])
        best_fitness, mean_fitness = ranked[0][1], float(np.mean(fitness))
        history.append({"generation": generation, "best_fitness": best_fitness, "mean_fitness": mean_fitness})

        if generation == 1 or generation % 5 == 0 or generation == generations:
            print(f"[{position_group}-GA] generation {generation:3d}/{generations}  best_val_mse={best_fitness:.4f}  mean_val_mse={mean_fitness:.4f}")

        if generation == generations:
            population = [ind for ind, _ in ranked]  # keep final ranking order, don't breed a generation that's never evaluated
            break

        parents = [ind for ind, _ in ranked[:survivors]]
        children = []
        while len(children) < population_size - len(parents):
            a, b = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            child = _crossover_models(a, b)
            _mutate_model(child, mutation_std, mutation_rate)
            children.append(child)
        population = parents + children

    final_fitness = [_evaluate_fitness(ind, val_loader, device) for ind in population]
    ranked_final = sorted(zip(population, final_fitness), key=lambda p: p[1])
    top_models = [ind for ind, _ in ranked_final[:ensemble_size]]
    final_population = [ind for ind, _ in ranked_final]

    fitness_str = ", ".join(f"{f:.4f}" for _, f in ranked_final[:ensemble_size])
    print(f"[{position_group}-GA] done -- top {ensemble_size} final fitness: [{fitness_str}]")

    return top_models, final_population, normalizer, context_normalizer, context_fill_values, stat_columns, history


def save_population(
    population: List[PlayerStatLSTM],
    normalizer: Normalizer,
    context_normalizer: Normalizer,
    context_fill_values: Dict[str, float],
    stat_columns: List[str],
    position_group: str,
    window_size: int,
    model_dir: str = MODEL_DIR,
) -> str:
    """
    Save an ENTIRE GA population (not just the top ensemble_size survivors)
    so evolution can resume later via train_position_model_ga(...,
    resume_population=load_population(...)[0]) instead of restarting from
    generation 0 -- the top survivors alone aren't enough to resume from,
    since resuming needs the full gene pool (diversity among the
    non-survivors is what crossover/mutation draws on for the next round).

    Args:
        population: Full population (see train_position_model_ga()'s
            `population` return value).
        normalizer, context_normalizer, context_fill_values, stat_columns,
            window_size: Same as save_model() -- shared across every
            individual in the population (they were all trained on the
            same data/features).
        position_group: "QB"/"RB"/"WR"/"TE".
        model_dir: Directory to save into.

    Returns:
        The directory saved into (<model_dir>/<position_group.lower()>_ga_population).
    """
    path = os.path.join(model_dir, f"{position_group.lower()}_ga_population")
    os.makedirs(path, exist_ok=True)

    for i, model in enumerate(population):
        torch.save(model.state_dict(), os.path.join(path, f"individual_{i}.pt"))

    meta = {
        "position_group": position_group,
        "population_size": len(population),
        "stat_columns": stat_columns,
        "context_columns": CONTEXT_COLUMNS,
        "window_size": window_size,
        "hidden_size": population[0].lstm.hidden_size,
        "num_layers": population[0].lstm.num_layers,
        "context_size": population[0].context_size,
        "context_embed_size": population[0].context_embed_size,
        "normalizer": normalizer.to_dict(),
        "context_normalizer": context_normalizer.to_dict(),
        "context_fill_values": context_fill_values,
    }
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {position_group} GA population ({len(population)} individuals) -> {path}/")
    return path


def load_population(
    position_group: str, model_dir: str = MODEL_DIR, device: Optional[str] = None
) -> Tuple[List[PlayerStatLSTM], Normalizer, Normalizer, Dict[str, float], List[str], int]:
    """
    Load a previously saved GA population, ready to resume evolution (pass
    the returned list as train_position_model_ga()'s `resume_population`)
    or to inspect/evaluate directly.

    Args:
        position_group: "QB"/"RB"/"WR"/"TE".
        model_dir: Directory save_population() wrote into.
        device: "cpu"/"cuda"; autodetects if not given.

    Returns:
        (population, normalizer, context_normalizer, context_fill_values,
        stat_columns, window_size) -- population is in eval() mode, on
        `device`; order matches how save_population() wrote it (not
        necessarily fitness-ranked, since train_position_model_ga() may
        have already bred a new unranked generation before this was saved).

    Raises:
        FileNotFoundError: if no GA population has been saved for this
            position yet.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(model_dir, f"{position_group.lower()}_ga_population")
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No saved GA population for {position_group!r} in {path}/ -- run "
            f"train_all_position_models_ga() first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    population = []
    for i in range(meta["population_size"]):
        model = PlayerStatLSTM(
            num_features=len(meta["stat_columns"]), context_size=meta["context_size"],
            hidden_size=meta["hidden_size"], num_layers=meta["num_layers"], context_embed_size=meta["context_embed_size"],
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(path, f"individual_{i}.pt"), map_location=device, weights_only=True))
        model.eval()
        population.append(model)

    normalizer = Normalizer.from_dict(meta["normalizer"])
    context_normalizer = Normalizer.from_dict(meta["context_normalizer"])
    return population, normalizer, context_normalizer, meta["context_fill_values"], meta["stat_columns"], meta["window_size"]


def train_all_position_models_ga(
    years: Optional[List[int]] = None, model_dir: str = MODEL_DIR, resume: bool = False,
    starters_only: bool = False, **train_kwargs
) -> Dict[str, List[dict]]:
    """
    Evolve (or resume-evolving) a 3-model GA ensemble for every position in
    POSITION_STAT_COLUMNS -- the GA counterpart to train_all_position_models().
    Saves both the usable top-ensemble_size models (standard save_model()
    format, for predict_next_game_ga()) and the full final population
    (save_population(), for resuming evolution in a later call).

    Args:
        years: Passed through to load_weekly_stats() (defaults to the last
            3 seasons if not given).
        model_dir: Directory to save into.
        resume: If True, load each position's previously saved population
            (see load_population()) and continue evolving it instead of
            starting from generation 0 -- silently falls back to a fresh
            random population for any position with nothing saved yet.
        starters_only: QB only -- train the QB model on listed starters' games
            (see position_training_rows()); other positions are unaffected.
            Defaults to False, the original behaviour.
        **train_kwargs: Passed through to train_position_model_ga() for
            every position group (e.g. generations=100, population_size=20).

    Returns:
        Dict mapping position_group -> its per-generation history (see
        train_position_model_ga()'s Returns).
    """
    from PlayerArchetypes import compute_all_archetypes, compute_all_defense_vs_archetype_profiles

    df = load_weekly_stats(years=years)
    defensive_target_stats = load_defensive_target_stats(years=years)
    player_bio = load_player_bio()
    schedule_context = load_schedule_context(years=years)
    player_age = load_player_age(years=years)
    injury_reports = load_injury_reports(years=years)
    top_cb_profile = build_top_cb_profile(defensive_target_stats, player_bio)
    context_df = build_matchup_context(df, top_cb_profile, schedule_context, injury_reports, player_bio)
    archetypes_by_position = compute_all_archetypes(df)
    defense_vs_archetype_by_position = compute_all_defense_vs_archetype_profiles(df, archetypes_by_position)

    results = {}
    for position_group in POSITION_STAT_COLUMNS:
        resume_population = None
        if resume:
            try:
                resume_population, *_ = load_population(position_group, model_dir)
            except FileNotFoundError:
                print(f"[{position_group}-GA] resume=True but no saved population found -- starting fresh.")

        print(f"\n=== Evolving {position_group} GA ensemble ===")
        top_models, population, normalizer, context_normalizer, context_fill_values, stat_columns, history = train_position_model_ga(
            position_group, position_training_rows(df, schedule_context, position_group, starters_only), context_df, top_cb_profile, player_bio, player_age, injury_reports,
            archetypes_by_position[position_group], defense_vs_archetype_by_position[position_group],
            resume_population=resume_population, **train_kwargs
        )
        window_size = train_kwargs.get("window_size", WINDOW_SIZE)

        for i, model in enumerate(top_models):
            save_model(
                model, normalizer, context_normalizer, context_fill_values, stat_columns, position_group,
                window_size, model_dir=model_dir, model_name=f"{position_group.lower()}_ga_{i}",
                starters_only=starters_only and position_group == "QB",
            )
        save_population(population, normalizer, context_normalizer, context_fill_values, stat_columns, position_group, window_size, model_dir)
        results[position_group] = history

    return results


def predict_next_game_ga(
    player_id: str,
    next_opponent: str,
    df: Optional[pd.DataFrame] = None,
    defensive_target_stats: Optional[pd.DataFrame] = None,
    player_bio: Optional[pd.DataFrame] = None,
    schedule_context: Optional[pd.DataFrame] = None,
    player_age: Optional[pd.DataFrame] = None,
    injury_reports: Optional[pd.DataFrame] = None,
    injury_status: Optional[str] = None,
    position_group: Optional[str] = None,
    model_dir: str = MODEL_DIR,
    ensemble_size: int = GA_ENSEMBLE_SIZE,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict one player's next game using this position's GA-evolved
    ensemble (see train_all_position_models_ga()) -- runs each of the
    `ensemble_size` saved individuals (reusing predict_next_game() itself,
    just varying `model_name`, so no inference logic is duplicated) and
    reports each one's prediction plus their mean/std.

    Args:
        player_id, next_opponent, df, defensive_target_stats, player_bio,
            schedule_context, player_age, injury_reports, injury_status,
            position_group, model_dir, device: Same as predict_next_game().
        ensemble_size: How many saved individuals to load (must match what
            train_all_position_models_ga()/train_position_model_ga() saved
            -- default GA_ENSEMBLE_SIZE).

    Returns:
        DataFrame indexed by stat column, columns "model_0".."model_{n-1}"
        (each individual's own prediction), "mean", "std" (spread ACROSS
        the 3 evolved individuals -- a rough agreement/disagreement signal,
        not a calibrated floor/ceiling the way predict_next_game_ensemble()'s
        pinball-loss quantile models are).

    Raises:
        FileNotFoundError: if this position's GA ensemble hasn't been
            trained yet (propagated from predict_next_game()'s load_model()) --
            see train_all_position_models_ga().
    """
    if df is None:
        df = load_weekly_stats()
    if defensive_target_stats is None:
        defensive_target_stats = load_defensive_target_stats()
    if player_bio is None:
        player_bio = load_player_bio()
    if schedule_context is None:
        schedule_context = load_schedule_context()
    if player_age is None:
        player_age = load_player_age()
    if injury_reports is None:
        injury_reports = load_injury_reports()

    player_rows = df[df["player_id"] == player_id]
    if player_rows.empty:
        raise ValueError(f"No games found for player_id={player_id!r}")
    if position_group is None:
        position_group = player_rows["position_group"].iloc[-1]

    from PlayerArchetypes import compute_style_features, classify_archetype, build_defense_vs_archetype_profile

    archetypes = classify_archetype(compute_style_features(df, position_group), position_group)
    defense_vs_archetype = build_defense_vs_archetype_profile(df, archetypes, position_group)

    common_kwargs = dict(
        df=df, defensive_target_stats=defensive_target_stats, player_bio=player_bio, schedule_context=schedule_context,
        player_age=player_age, injury_reports=injury_reports, injury_status=injury_status,
        archetypes=archetypes, defense_vs_archetype=defense_vs_archetype,
        position_group=position_group, model_dir=model_dir, device=device,
    )
    predictions = [
        predict_next_game(player_id, next_opponent, model_name=f"{position_group.lower()}_ga_{i}", **common_kwargs)
        for i in range(ensemble_size)
    ]

    stat_columns = list(predictions[0].keys())
    matrix = np.array([[p[c] for c in stat_columns] for p in predictions])

    result = {f"model_{i}": matrix[i] for i in range(ensemble_size)}
    result["mean"] = matrix.mean(axis=0)
    result["std"] = matrix.std(axis=0)
    return pd.DataFrame(result, index=stat_columns)


if __name__ == "__main__":
    train_all_position_models()

    # Live weekly projections: the validated recipe (weekly retrain, 5-seed
    # ensemble, QB on listed starters). Prefer this over predict_next_game().
    # print(project_week("QB", 2026, 6))

    # Example: after training, predict a specific player's next game against
    # a specific opponent. injury_status overrides whatever's in that
    # week's real injury report (or provides one if there isn't one yet).
    # print(predict_next_game(player_id="00-0039732", next_opponent="LV"))  # Bo Nix vs. the Raiders
    # print(predict_next_game(player_id="00-0039732", next_opponent="LV", injury_status="Questionable"))

    # Example: backtest QB projections across the first 5 weeks of the 2025
    # season, retraining fresh each week on the trailing 3 years.
    # years = [2022, 2023, 2024, 2025]
    # df = load_weekly_stats(years=years)
    # defensive_target_stats = load_defensive_target_stats(years=years)
    # player_bio = load_player_bio()
    # schedule_context = load_schedule_context(years=years)
    # player_age = load_player_age(years=years)
    # injury_reports = load_injury_reports(years=years)
    # results = walk_forward_backtest(
    #     df, defensive_target_stats, player_bio, schedule_context, player_age, injury_reports, "QB",
    #     eval_weeks=[(2025, w) for w in range(1, 6)], epochs=30,
    # )
    # print(summarize_backtest(results, POSITION_STAT_COLUMNS["QB"]))

    # Example: build residual pools for all four positions (needed once,
    # rebuild periodically as more weeks of real results come in), then get
    # a floor/median/ceiling Monte Carlo projection for one player.
    # build_all_residual_pools()
    # print(predict_next_game_monte_carlo(player_id="00-0039732", next_opponent="LV"))

    # Example: train the 2 pinball-loss quantile models per position (needed
    # once for the ensemble below; train_all_position_models() above and
    # PlayerProjectionsTransformer.train_all_position_models_hf() cover the
    # other 2 "likely" models this needs), then get a floor/likely/ceiling
    # projection combining all 4.
    # train_all_quantile_models()
    # print(predict_next_game_ensemble(player_id="00-0039732", next_opponent="LV"))
