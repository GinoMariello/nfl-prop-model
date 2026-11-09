"""Pregame features for historical snapshot studies (not verified as-of replay).

Historical completion uses valid scores when no explicit status is supplied.
Only admitted REG games on strictly earlier calendar dates enter history.
Current/same-date rows are excluded before taking the last eight completed
rows, including rows whose play metrics are missing. History spans seasons.
"""
import numpy as np
import pandas as pd

from StatScraper import save_to_sqlite, query_sqlite, drop_table_if_exists
from src.features.team_game_stats import build_team_games

GAMES_DB_PATH = "data/nfl_games.db"
ROLLING_WINDOW = 8
STAT_COLUMNS = [
    "off_plays", "off_epa_per_play", "off_success_rate", "off_pass_epa", "off_rush_epa",
    "off_yards_per_play", "off_third_down_rate", "off_redzone_td_rate", "off_turnovers", "off_sacks_allowed",
    "def_epa_per_play", "def_success_rate", "def_third_down_rate", "def_redzone_td_rate",
    "def_takeaways", "def_sacks",
    "points_for", "points_against",
]
KEEP_ASIS = ["game_id", "season", "week", "game_type", "gameday", "team", "opponent", "is_home"]
TEAM_FEATURE_COLUMNS = tuple(
    f"{side}_{name}"
    for side in ("home", "away")
    for name in [*(f"roll_{stat}" for stat in STAT_COLUMNS), "games_played_before"]
)
MARKET_FEATURE_COLUMNS = (*TEAM_FEATURE_COLUMNS, "market_home_probability")
# Protocol v2 (docs/experiment-protocol.md). The closing point spread, signed
# from the home team's perspective: positive means the home team is favoured
# (measured 2005-2024, home win rate 0.6765 when positive against 0.3430 when
# negative). Fed raw, never converted to a probability -- see select_features.
SPREAD_FEATURE_COLUMNS = (*TEAM_FEATURE_COLUMNS, "spread_line")
# Protocol v3. The fitted market-only CONTROL, not a candidate model: the
# moneyline probability alone, so `team_market - market_only` on identical rows
# isolates what the team statistics add beyond fitting the market
# (model-plan.md section 6). No new input -- a subset of MARKET_FEATURE_COLUMNS.
MARKET_ONLY_FEATURE_COLUMNS = ("market_home_probability",)
# Protocol v4. The residual model: exactly team_market's columns and rows. The
# difference is in train.py -- the market enters as a fixed logit offset there,
# not as a fitted feature -- so it needs no allowlist of its own.
# Built by qb_rest_features() from the two inputs that passed the provenance gate
# (experiment-protocol.md, "Input provenance audit"). Protocol v5 adds them to the
# residual model as team_residual_plus.
QB_REST_FEATURE_COLUMNS = ("home_qb_changed", "away_qb_changed",
                           "home_qb_prior_starts", "away_qb_prior_starts", "rest_diff")
VARIANT_COLUMNS = {"team_only": TEAM_FEATURE_COLUMNS,
                   "team_market": MARKET_FEATURE_COLUMNS,
                   "team_spread": SPREAD_FEATURE_COLUMNS,
                   "market_only": MARKET_ONLY_FEATURE_COLUMNS,
                   "team_residual": MARKET_FEATURE_COLUMNS,
                   "team_residual_plus": (*MARKET_FEATURE_COLUMNS, *QB_REST_FEATURE_COLUMNS)}


def _unique(df, keys):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f"Expected non-null unique keys: {keys}")


def _dates(values):
    # Require an unambiguous ISO calendar date; do not guess locale or timezone.
    text = values.astype("string")
    return pd.to_datetime(text.where(text.str.fullmatch(r"\d{4}-\d{2}-\d{2}")),
                          format="%Y-%m-%d", errors="coerce")


def _completed(df, scores):
    """Historical-snapshot proxy when status is absent; explicit status wins.

    This is not a live final-status detector. A supplied completed column must
    contain boolean/0/1/null values; unknown or false status is never inferred
    from scores. Without that column, finite nonnegative integer scores are
    assumed final for historical studies, matching the ingestion schema.
    """
    numeric = df[scores].apply(pd.to_numeric, errors="coerce").astype(float)
    valid = (np.isfinite(numeric) & numeric.ge(0) & numeric.mod(1).eq(0)).all(axis=1)
    if "completed" not in df:
        return valid
    status = df["completed"]
    if not status.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("completed must contain only boolean/0/1/null values")
    return valid & status.eq(True).fillna(False)


def add_rolling_features(team_game_stats: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling/as-of approach with validated, deterministic date joins.

    Build trailing postgame averages, then join ONLY to strictly earlier dates.
    Same-day games cannot see each other. History spans seasons; missing metric
    rows still occupy window slots. Unknown target dates retain missing counts.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    _unique(team_game_stats, ["game_id", "team"])
    df = team_game_stats.loc[team_game_stats.game_type.eq("REG")].copy()
    if not df.is_home.isin([True, False, 0, 1]).all():
        raise ValueError("is_home must be boolean/0/1")
    df["is_home"] = df.is_home.astype(bool)
    df["_date"] = _dates(df.gameday)
    admitted = _completed(df, ["points_for", "points_against"]) & df._date.notna()
    hist = df.loc[admitted].sort_values(["team", "_date", "game_id"]).reset_index(drop=True)
    roll_cols = [f"roll_{stat}" for stat in STAT_COLUMNS]
    dated = df.loc[df._date.notna()].sort_values(["_date", "team", "game_id"]).reset_index(drop=True)
    dated[roll_cols] = np.nan
    dated["games_played_before"] = 0
    if not hist.empty:
        rolled = hist.groupby("team")[STAT_COLUMNS].transform(
            lambda series: series.rolling(window, min_periods=1).mean())
        asof = hist[["team", "_date", "game_id"]].copy()
        asof[roll_cols] = rolled.to_numpy()
        asof["games_played_before"] = hist.groupby("team").cumcount() + 1
        # Keep the FINAL state for each team/date, including all same-day rows.
        # Otherwise sorting equal dates can select a partial history arbitrarily.
        asof = asof.drop_duplicates(["team", "_date"], keep="last")
        asof = asof.sort_values(["_date", "team"]).drop(columns="game_id")
        carried = pd.merge_asof(
            dated[["team", "_date"]], asof, on="_date", by="team",
            direction="backward", allow_exact_matches=False)
        dated[roll_cols] = carried[roll_cols].to_numpy()
        dated["games_played_before"] = carried.games_played_before.fillna(0).astype("Int64")
    undated = df.loc[df._date.isna()].copy()
    undated[roll_cols] = np.nan
    undated["games_played_before"] = pd.Series(pd.NA, index=undated.index, dtype="Int64")
    out = pd.concat([dated, undated], ignore_index=True)
    out["games_played_before"] = out.games_played_before.astype("Int64")
    return out.sort_values(["team", "_date", "game_id"]).reset_index(drop=True)[
        KEEP_ASIS + roll_cols + ["games_played_before"]]


def qb_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per REG game_id: starting-QB continuity per side, and the rest gap.

    The current game contributes only what the provenance audit cleared as known
    by kickoff: its recorded starter (fixed by the first snap) and both teams' rest
    days (schedule arithmetic). Everything else is history, admitted exactly as
    add_rolling_features admits it -- completed REG games on strictly earlier dates
    -- so a game never sees its own outcome, a later game, or a same-day game.

    qb_changed: 1.0 when this game's starter differs from the team's starter in its
      most recent earlier completed REG game, 0.0 when the same. NaN for a team's
      first admitted game or when either starter is unknown: unknown is not
      "unchanged".
    qb_prior_starts: completed REG games this QB started on earlier dates, for any
      team. Left-censored at the first season in the data (1999): a veteran then
      shows only the starts since. NaN when the starter is unknown.
    rest_diff: home_rest - away_rest. Season openers carry nflverse's fixed-7
      convention on both sides, so theirs is 0 by convention, not measurement.

    A source without the QB or rest columns gives NaN features, never a guess.
    """
    wanted = ["game_id", "game_type", "gameday", "home_team", "away_team", "home_qb_id",
              "away_qb_id", "home_rest", "away_rest", "home_score", "away_score"]
    g = games.reindex(columns=wanted + (["completed"] if "completed" in games else []))
    g = g.loc[g.game_type.eq("REG")].reset_index(drop=True)
    g["_date"] = _dates(g.gameday)
    g["_done"] = _completed(g, ["home_score", "away_score"]) & g._date.notna()

    long = pd.concat([pd.DataFrame({"game_id": g.game_id, "side": side, "team": g[f"{side}_team"],
                                    "qb": g[f"{side}_qb_id"], "_date": g._date, "_done": g._done})
                      for side in ("home", "away")], ignore_index=True)
    dated = long.loc[long._date.notna()].sort_values(["_date", "team", "game_id"])

    # The team's starter in its latest completed game, as of each date. Unknown
    # starters stay in the history so a gap reads NaN, never "unchanged".
    team_hist = long.loc[long._done].sort_values(["team", "_date", "game_id"])
    last_qb = team_hist.drop_duplicates(["team", "_date"], keep="last")[["team", "_date", "qb"]] \
        .rename(columns={"qb": "prev_qb"}).sort_values("_date")
    prev = pd.merge_asof(dated[["game_id", "side", "team", "_date", "qb"]], last_qb, on="_date",
                         by="team", direction="backward", allow_exact_matches=False)
    known = prev.qb.notna() & prev.prev_qb.notna()
    prev["qb_changed"] = (prev.qb != prev.prev_qb).astype(float).where(known)

    # How many completed games this QB had started before this date.
    starts = long.loc[long._done & long.qb.notna()].sort_values(["qb", "_date", "game_id"])
    starts["qb_prior_starts"] = starts.groupby("qb").cumcount() + 1.0
    starts = starts.drop_duplicates(["qb", "_date"], keep="last")[["qb", "_date", "qb_prior_starts"]] \
        .sort_values("_date")
    named = dated.loc[dated.qb.notna(), ["game_id", "side", "qb", "_date"]]
    named = pd.merge_asof(named, starts, on="_date", by="qb", direction="backward",
                          allow_exact_matches=False)
    named["qb_prior_starts"] = named.qb_prior_starts.fillna(0.0)

    per_side = prev[["game_id", "side", "qb_changed"]].merge(
        named[["game_id", "side", "qb_prior_starts"]], on=["game_id", "side"], how="left",
        validate="one_to_one")
    out = g[["game_id"]].copy()
    for side in ("home", "away"):
        part = per_side.loc[per_side.side.eq(side)].drop(columns="side").add_prefix(f"{side}_")
        out = out.merge(part, left_on="game_id", right_on=f"{side}_game_id", how="left",
                        validate="one_to_one").drop(columns=f"{side}_game_id")
    out["rest_diff"] = (pd.to_numeric(g.home_rest, errors="coerce")
                        - pd.to_numeric(g.away_rest, errors="coerce")).to_numpy()
    return out[["game_id", *QB_REST_FEATURE_COLUMNS]]


def normalized_home_probability(games: pd.DataFrame) -> pd.Series:
    """Proportional two-sided American-odds normalization; invalid pairs abstain.

    Finite American quotes must have magnitude >= 100. No rounding, fabricated
    opposite-side quotes, or imputation. Historical closing quotes support only
    a closing-line study, not a claim about live quote receipt times.
    """
    quotes = games.reindex(columns=["home_moneyline", "away_moneyline"]).apply(
        pd.to_numeric, errors="coerce").astype(float)
    valid = (np.isfinite(quotes) & quotes.abs().ge(100)).all(axis=1)
    magnitude = quotes.abs()
    implied = (100 / (magnitude + 100)).where(quotes > 0, magnitude / (magnitude + 100))
    return (implied.home_moneyline / implied.sum(axis=1)).where(valid)


def build_matchup_table(team_features: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Validated home/away reshape; nullable home_win is the only v1 target."""
    _unique(games, ["game_id"])
    _unique(team_features, ["game_id", "team"])
    expected = build_team_games(games)
    expected = expected.loc[expected.game_type.eq("REG"), ["game_id", "team", "opponent", "is_home"]]
    actual = team_features[expected.columns]
    comparison = expected.merge(actual, on=list(expected.columns), how="outer", indicator=True,
                                validate="one_to_one")
    if not comparison._merge.eq("both").all():
        raise ValueError("Expected exactly the scheduled home and away feature rows for every REG game")
    home = team_features.loc[team_features.is_home.eq(True)].add_prefix("home_")
    away = team_features.loc[team_features.is_home.eq(False)].add_prefix("away_")
    matchup = home.merge(away, left_on="home_game_id", right_on="away_game_id", validate="one_to_one")
    matchup = matchup.rename(columns={"home_game_id": "game_id"}).drop(columns="away_game_id")
    labels = _compute_labels(games)
    labels["market_home_probability"] = normalized_home_probability(games).reindex(labels.index)
    labels["neutral_site"] = _neutral_site(games).reindex(labels.index)
    labels["spread_line"] = pd.to_numeric(
        games.reindex(columns=["spread_line"])["spread_line"], errors="coerce").reindex(labels.index)
    return matchup.merge(labels, on="game_id", validate="one_to_one") \
        .merge(qb_rest_features(games), on="game_id", how="left", validate="one_to_one") \
        .sort_values("game_id").reset_index(drop=True)


def _neutral_site(games: pd.DataFrame) -> pd.Series:
    """True where the nominal home team has no home field.

    Carried as matchup metadata, never as a model feature. model-plan.md §4
    fixes the v1 allowlist, so adding to it is a protocol change rather than a
    builder's call, and `select_features` takes an explicit column list that
    this is not on. It exists so Stage 4 can mask or report these rows instead
    of silently scoring them as home games: the home/away row structure
    asserts a home field that 58 REG games in 2005-2024 do not have
    (international series, relocations). Their home-win rate is 0.4561 against
    0.5596 for real home games -- at n=58 that is a direction, not a measured
    effect, which is why the flag belongs in the data rather than in a comment.

    §4 excludes `is_home` as "redundant with the home/away row structure
    itself". True for the other 5125 games; false for these.

    Unknown -- not False -- when the source carries no `location` column.
    Defaulting to "home" is the silent claim this function exists to stop.
    """
    if "location" not in games:
        return pd.Series(pd.NA, index=games.index, dtype="boolean")
    location = games.location
    unexpected = sorted(set(location.dropna().unique()) - {"Home", "Neutral"})
    if unexpected:
        raise ValueError(
            f"unexpected games.location values: {unexpected[:3]} -- expected only "
            f"'Home'/'Neutral'. A new value must be classified deliberately, not read "
            f"as a home game by falling through."
        )
    return location.eq("Neutral").astype("boolean").where(location.notna())


def _compute_labels(games: pd.DataFrame) -> pd.DataFrame:
    """Only dated, completed/proxy-completed, non-tied REG games get labels."""
    labels = games.loc[games.game_type.eq("REG"), ["game_id", "home_score", "away_score"]].copy()
    valid = _completed(games, ["home_score", "away_score"]) & _dates(games.gameday).notna()
    scores = games[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
    label = (scores.home_score > scores.away_score).astype("Int64")
    labels["home_win"] = label.where(valid & scores.home_score.ne(scores.away_score)).reindex(labels.index)
    return labels


def paired_quote_mask(matchups: pd.DataFrame) -> pd.Series:
    """Rows a `team_market` model can actually use: a devigged home probability
    strictly inside (0, 1).

    The single definition of "paired". `select_features` filters on it, so
    anything that reports on the market-eligible population -- Stage 4's
    matched-coverage scoring, its per-fold counts -- must ask here rather than
    restate the predicate. Two copies that drift would put rows in the matched
    population that the model never scored, which is precisely the mismatched
    comparison model-plan.md section 6 forbids.
    """
    return matchups["market_home_probability"].between(0, 1, inclusive="neither")


def finite_spread_mask(matchups: pd.DataFrame) -> pd.Series:
    """Rows a `team_spread` model can use: a finite closing spread.

    The single definition, for the same reason `paired_quote_mask` is one. The
    spread is 100% present for every REG game from 1999 through 2024 in the
    current snapshot, so this filter removes nothing today -- it exists so that
    a future gap abstains rather than being imputed, and so Stage 4's coverage
    accounting asks one place.
    """
    return np.isfinite(pd.to_numeric(matchups["spread_line"], errors="coerce"))


def select_features(matchups: pd.DataFrame, variant: str = "team_only") -> pd.DataFrame:
    """Ordered X only; a market variant returns only its eligible rows.

    Keeps input indices for label alignment. Training callers must additionally
    filter missing labels. No learned preprocessing belongs in this builder.

    `team_spread` carries the closing spread RAW rather than converting it to a
    win probability. Converting one requires assuming a margin distribution --
    conventionally normal with a standard deviation near 13.5 -- and that
    number is itself estimated from NFL history, including the seasons a fold
    is supposed to be scored on. Handing the logistic model the raw spread
    makes the mapping a coefficient fitted per fold on training rows only.
    The moneyline devig is different and stays a probability: it is pure
    arithmetic over two quotes with no fitted parameter.
    """
    if variant not in VARIANT_COLUMNS:
        raise ValueError(f"Unknown feature variant: {variant}")
    columns = VARIANT_COLUMNS[variant]
    missing = [column for column in columns if column not in matchups]
    if missing:
        raise KeyError(f"matchup_features missing expected feature columns: {missing}")
    result = matchups.loc[:, list(columns)].copy()
    if variant in ("team_market", "market_only", "team_residual", "team_residual_plus"):
        result = result.loc[paired_quote_mask(result)]
    elif variant == "team_spread":
        result = result.loc[finite_spread_mask(result)]
    return result


def exclusion_report(games: pd.DataFrame) -> pd.DataFrame:
    """Per-season game counts; reasons overlap and must not be summed."""
    games = games.reset_index(drop=True)
    complete = _completed(games, ["home_score", "away_score"])
    regular = games.game_type.eq("REG")
    dates = _dates(games.gameday)
    status = games.get("completed", pd.Series(pd.NA, index=games.index))
    scores = games[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce")
    reasons = {
        "non_regular": ~regular,
        "unknown_completion": status.isna(),
        "not_completed": ~complete,
        "score_proxy_used": complete & ("completed" not in games),
        "missing_or_invalid_score": ~_completed(games.drop(columns="completed", errors="ignore"), ["home_score", "away_score"]),
        "ambiguous_date": dates.isna(),
        "tie": complete & scores.home_score.eq(scores.away_score),
        "invalid_market_pair": normalized_home_probability(games).isna(),
        "history_eligible": regular & complete & dates.notna(),
        "supervised_eligible": regular & complete & dates.notna() & scores.home_score.ne(scores.away_score),
    }
    return pd.DataFrame([
        {"season": season, "reason": reason, "count": int(mask.loc[group.index].sum())}
        for season, group in games.groupby("season", dropna=False, sort=True)
        for reason, mask in reasons.items()
    ], columns=["season", "reason", "count"])


def run(db_path: str = GAMES_DB_PATH) -> pd.DataFrame:
    """Rebuild derived tables only. Read schedule metadata afresh (old stats may
    predate the completion contract). Raw games/plays are never modified.
    """
    stats = query_sqlite("SELECT * FROM team_game_stats", db_path=db_path)
    games = query_sqlite("SELECT * FROM games", db_path=db_path)
    _unique(stats, ["game_id", "team"])
    skeleton = build_team_games(games)
    metrics = [s for s in STAT_COLUMNS if s not in ("points_for", "points_against")]
    # Refresh schedule metadata and restore absent aggregate rows. The schedule
    # is authoritative for teams, dates, scores, and any optional status.
    stats = skeleton.merge(stats[["game_id", "team", *metrics]],
                           on=["game_id", "team"], how="left", validate="one_to_one")
    team = add_rolling_features(stats)
    matchup = build_matchup_table(team, games)
    report = exclusion_report(games)
    # Validate all outputs before replacing any derived table. Shared helpers
    # preserve the repository's storage conventions and unique-key indexes.
    for frame, table, keys in (
        (team, "team_rolling_features", ["game_id", "team"]),
        (matchup, "matchup_features", ["game_id"]),
        (report, "feature_exclusions", ["season", "reason"]),
    ):
        drop_table_if_exists(table, db_path=db_path)
        save_to_sqlite(frame, table, key_columns=keys, db_path=db_path)
    return matchup


if __name__ == "__main__":
    run()
