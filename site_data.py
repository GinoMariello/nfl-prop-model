"""
Export one week's props board for the Prop Board page: consensus lines and odds,
our blended projection and its three sources, a model probability of the over,
expected value per side, and the paper bets placed on it.

    python3 site_data.py 2026 5      # writes site/data.json

Probability of the over: the blended projection's own 2025 misses. For each
position and stat, 2025 player-games are bucketed by projection size (quintiles);
P(over) = share of that bucket whose actual beat the projection by more than
(line - projection). Ties with the line are ignored (lines are x.5). It assumes
2025's error spread holds now, and it has never been checked against real lines --
only against actual outcomes (see calibration in the output).

EV per $1 = P(win) x decimal odds - 1, at the median price across books.

p_market is the books' own probability of the over with their margin removed
(1/decimal odds per side, normalised to sum to 1). p_over is shrunk toward 50% by
SHRINK first. The page blends p_over with it
by a trust slider: the 2026 check shows p_over overconfident at the extremes, and
it has never been tested against lines, so the market gets most of the weight.
"""
import copy
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import nfl_data_py as nfl
import pandas as pd

import PlayerProjections as pp
from log_props import DB_PATH, name_to_id, norm_name, week_only
from paper_trades import NO_UNDER, WEAK_D, median_price
from StatScraper import query_sqlite

YARDS = {"QB": ["passing_yards", "rushing_yards"], "RB": ["rushing_yards", "receiving_yards"],
         "WR": ["receiving_yards"], "TE": ["receiving_yards"]}
OUT = "site/data.json"  # beside site/index.html, the page that reads it
# The main rule: hit55f (yardage, since 2026 week 6) plus rec55 (receptions, folded in 2026-10-18).
MAIN_LEDGERS = ["projections/paper_trades_hit55f.csv", "projections/paper_trades_rec55.csv"]
OLD_LEDGER = "projections/paper_trades.csv"           # the original rule, retired after week 5


def decimal(price: float) -> float:
    return 1 + (100 / abs(price) if price < 0 else price / 100)


def load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f).assign(wk=int(re.search(r"_w(\d+)_", f).group(1))) for f in files]) \
        .drop_duplicates(["wk", "player_id"])


def blended_2025(pos: str) -> pd.DataFrame:
    """2025 projections blended the way project_week() blends live ones, with actuals."""
    own, sl, gb = (load(f"projections/2025_vegas/2025_w*_{pos}.csv"), load(f"projections/2025_sleeper/2025_w*_{pos}.csv"),
                   load(f"projections/gbm/2025_w*_{pos}.csv"))
    weeks = []
    for wk, part in own.groupby("wk"):
        s, g = sl[sl.wk == wk], gb[gb.wk == wk]
        sources = {"sleeper": s[["player_id"] + [c for c in s if c.startswith("pred_")]].rename(columns=lambda c: c.replace("pred_", "sleeper_")),
                   "gbm": g[["player_id"] + [c for c in g if c.startswith("pred_")]].rename(columns=lambda c: c.replace("pred_", "gbm_"))}
        base = part.drop(columns=[c for c in part if c.startswith(("own_", "sleeper_", "gbm_"))])
        weeks.append(pp.blend_sources(base, sources, pp.BLEND_WEIGHTS[pos]))
    return pd.concat(weeks)


# Shrink toward 50%: p = 0.5 + k * (p_raw - 0.5). Fitted out of sample (model built on
# 2025 weeks 1-9, k chosen by log loss on weeks 10-18, last-5 average as the line).
SHRINK = {("QB", "passing_yards"): 0.80, ("QB", "rushing_yards"): 1.00, ("RB", "rushing_yards"): 0.70,
          ("RB", "receiving_yards"): 0.55, ("WR", "receiving_yards"): 0.85, ("TE", "receiving_yards"): 0.80}


# Hit-rate ladder (OverModel.ladder). LADDER_W fit by log loss on 2026 weeks 2-4 over lines
# -20..+20 around Sleeper's projection (no real lines saved then); held out on 2026 week 5's
# real lines it was honest to ~5 points, slightly overconfident.
# Known limit: Sleeper stands in for the book; refit LADDER_W on logged real lines after a few weeks.
LADDER_W = 0.3
# Even so, its chances ran high on 2026 week 5's real lines (the 70% band hit ~64%): shrink
# toward 50% by LADDER_K, fit by log loss on that week over lines -20..+20 (80% range
# 0.85-1.00 resampling games). Refit as weeks of real lines pile up.
LADDER_K = 0.85
LEVELS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

# Receptions ladder: the same 2025-miss method, its own weights. Fit by log loss on 2026 weeks 2-4
# (lines = last-5 average +-2, no real lines then), held out on week 5's 116 real lines: log loss
# 0.650 vs the market's 0.660, honest by band, and 55%+ positive-EV bets went 43-27 (+10.6%).
# Unlike yards it leans on our projection. RB receptions went 8-10 there -- watch them.
# Known limit: one held-out week; refit REC_W/REC_K on logged lines after a few more.
REC_POS = ("RB", "WR", "TE")
REC_W, REC_K = 0.9, 1.0


class OverModel:
    """P(actual > line | projection) from 2025 misses, bucketed by projection size."""

    def __init__(self, history: pd.DataFrame, stat: str, shrink: float = 1.0):
        self.shrink = shrink
        h = history.dropna(subset=[f"pred_{stat}", f"actual_{stat}"])
        self.edges = np.quantile(h[f"pred_{stat}"], [0.2, 0.4, 0.6, 0.8])
        self.misses = [g[f"actual_{stat}"].to_numpy() - g[f"pred_{stat}"].to_numpy()
                       for _, g in h.groupby(np.digitize(h[f"pred_{stat}"], self.edges))]

    def p_over(self, projection: float, line: float) -> float:
        miss = self.misses[int(np.digitize(projection, self.edges))]
        return float(0.5 + self.shrink * (np.mean(miss > line - projection) - 0.5))

    def ladder(self, projection: float, line: float, w: float = LADDER_W, k: float = LADDER_K) -> dict:
        """Chance of the over at the posted line, and for each LEVELS chance the furthest
        alt line (x.5) that still clears it on each side: over[c] = the highest line the
        player beats with chance >= c, under[c] = the lowest line he stays below with chance >= c.

        Centered on LADDER_W * projection + (1 - LADDER_W) * line, unshrunk: the raw 2025
        misses are honest across a ladder of lines, but betting where we disagree with the
        book is where they run overconfident, and leaning on the book's line fixes most of it."""
        c = w * projection + (1 - w) * line
        final = c + self.misses[int(np.digitize(c, self.edges))]
        halves = np.arange(np.floor(final.min()), np.ceil(final.max()) + 1) + 0.5
        p = 0.5 + k * ((final[:, None] > halves).mean(axis=0) - 0.5)   # P(over) at each x.5 line, falling
        over = {lv: float(halves[p >= lv].max()) if (p >= lv).any() else None for lv in LEVELS}
        under = {lv: float(halves[1 - p >= lv].min()) if (1 - p >= lv).any() else None for lv in LEVELS}
        return {"p_over": round(0.5 + k * (float(np.mean(final > line)) - 0.5), 3),
                "over": {f"{lv:.2f}": (v if v is not None and v > 0 else None) for lv, v in over.items()},
                "under": {f"{lv:.2f}": v for lv, v in under.items()},
                # the same distribution as 100 quantiles, so the page can price any line (Pick'em tab);
                # raw -- the page applies "k" (the shrink above) to it
                "k": k, "dist": [round(float(v), 1) for v in np.quantile(final, (np.arange(100) + 0.5) / 100)]}

    def live(self, n: int = 100) -> dict:
        """The miss distribution as n quantiles per bucket, for the page's in-game P(over)."""
        return {"edges": [round(float(e), 2) for e in self.edges], "shrink": self.shrink,
                "q": [[round(float(v), 1) for v in np.quantile(m, (np.arange(n) + 0.5) / n)] for m in self.misses]}


def live_p_over(model: dict, projection: float, line: float, current: float, left: float) -> float:
    """In-game P(final > line), mirrored by livePOver() in site/index.html: the stat so far plus
    the projection's share for the time left, with the 2025 miss spread scaled by sqrt(time left).
    At left=1 it is the pregame p_over; at left=0 it is just current > line."""
    # Known limit: time-scaled pregame misses; ignores score, injuries and usage seen so far in the game
    if current > line:
        return 1.0   # already cleared; yards don't come off
    b = int(np.searchsorted(model["edges"], projection, side="right"))
    rest = np.maximum(0, left * projection + np.sqrt(left) * np.array(model["q"][b]))   # yards don't come off
    raw = float(np.mean(current + rest > line))
    k = 1 - (1 - model["shrink"]) * left   # shrink fades out as the game's result firms up
    return 0.5 + k * (raw - 0.5)


def calibration(models: dict) -> list:
    """Check on 2026 weeks 2-4 (never used to build the model): predicted P(actual > last-5 average) vs how often it happened."""
    rows = []
    for (pos, stat), model in models.items():
        d = load(f"projections/2026_w0[234]_{pos}.csv").dropna(subset=[f"pred_{stat}", f"actual_{stat}", f"l5_{stat}"])
        p = np.array([model.p_over(a, b) for a, b in zip(d[f"pred_{stat}"], d[f"l5_{stat}"])])
        hit = (d[f"actual_{stat}"] > d[f"l5_{stat}"]).to_numpy()
        for lo, hi in [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]:
            m = (p >= lo) & (p < hi)
            if m.sum() >= 10:
                rows.append({"pos": pos, "stat": stat, "bucket": f"{lo:.1f}-{min(hi, 1):.1f}", "n": int(m.sum()),
                             "predicted": round(float(p[m].mean()), 3), "observed": round(float(hit[m].mean()), 3)})
    return rows


def accuracy(season: int, week: int) -> list:
    """Average PPR miss by week before `week`, per position, for the blend, each of its
    sources and the last-5 average -- scored on the players every one of them projected."""
    keys = ["pred", "own", "sleeper", "gbm", "l5"]
    rows = []
    for pos in YARDS:
        for f in sorted(glob.glob(f"projections/{season}_w*_{pos}.csv")):
            wk = int(re.search(r"_w(\d+)_", f).group(1))
            if wk >= week:
                continue
            d = pd.read_csv(f)
            cols = [f"{k}_fantasy_points_ppr" for k in keys] + ["actual_fantasy_points_ppr"]
            if not set(cols) <= set(d.columns):
                continue
            d = d.dropna(subset=cols)
            if len(d) < 10:
                continue
            rows.append({"pos": pos, "week": wk, "n": len(d), "miss": {
                k: round(float((d[f"{k}_fantasy_points_ppr"] - d.actual_fantasy_points_ppr).abs().mean()), 3) for k in keys}})
    return rows


def logged_actuals(season: int, week: int, players: pd.DataFrame) -> dict:
    """Actual yardage for every player-stat with a logged line in weeks before `week` (and
    `week` itself once played), keyed "season:week:player_id:stat" -- lets the page
    settle parlays saved in any past week."""
    lines = query_sqlite("SELECT DISTINCT player_name, stat, commence_time FROM prop_lines", db_path=DB_PATH)
    lines = lines.assign(player_id=lines.player_name.map(norm_name).map(name_to_id(players))).dropna(subset=["player_id"])
    sched = nfl.import_schedules([season])
    sched = sched[(sched.game_type == "REG") & (sched.week <= week)]
    date_week = dict(zip(sched.gameday.astype(str), sched.week))
    kick = pd.to_datetime(lines.commence_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    lines = lines.assign(week=kick.map(date_week)).dropna(subset=["week"]).drop_duplicates(["player_id", "stat", "week"])
    lines = lines[lines.stat.isin(["passing_yards", "rushing_yards", "receiving_yards", "receptions"])]   # older pulls also logged TDs
    stats = query_sqlite(f"SELECT player_id, week, passing_yards, rushing_yards, receiving_yards, receptions FROM player_weekly_stats "
                         f"WHERE season = {int(season)} AND season_type = 'REG'", db_path=DB_PATH).set_index(["player_id", "week"])
    out = {}
    for r in lines.itertuples():
        key = (r.player_id, int(r.week))
        if key in stats.index:
            out[f"{season}:{int(r.week)}:{r.player_id}:{r.stat}"] = float(stats.loc[key, r.stat])
    return out


def season_logs(season: int, week: int, player_ids) -> dict:
    """player_id -> this season's regular-season games before `week`, for the page's player breakdown."""
    ids = ",".join(f"'{p}'" for p in sorted(set(player_ids)))
    if not ids:
        return {}
    w = query_sqlite("SELECT player_id, week, team, opponent_team, attempts, completions, passing_yards, carries, rushing_yards, "
                     "targets, receptions, receiving_yards, passing_tds, rushing_tds, receiving_tds, fantasy_points_ppr "
                     f"FROM player_weekly_stats WHERE season = {int(season)} AND season_type = 'REG' AND week < {int(week)} "
                     f"AND player_id IN ({ids})")
    num = lambda v: None if pd.isna(v) else round(float(v), 1)
    return {pid: [{"week": int(r.week), "team": r.team, "opp": r.opponent_team,
                   **{c: num(getattr(r, c)) for c in ("attempts", "completions", "passing_yards", "carries", "rushing_yards", "targets",
                                                      "receptions", "receiving_yards", "passing_tds", "rushing_tds", "receiving_tds",
                                                      "fantasy_points_ppr")}}
                  for r in g.sort_values("week").itertuples()]
            for pid, g in w.groupby("player_id")}


def game_actuals(season: int) -> dict:
    """Final margins for the season's played games, keyed like logged_actuals with the
    team as "<game_id>~home" / "~away" and the stat as moneyline or spread, plus the
    combined points as "<game_id>~game:total", so the page settles saved game bets the
    same way it settles props."""
    sched = nfl.import_schedules([season])
    sched = sched[(sched.game_type == "REG") & sched.home_score.notna() & sched.away_score.notna()]
    out = {}
    for r in sched.itertuples():
        for side, m in (("home", r.home_score - r.away_score), ("away", r.away_score - r.home_score)):
            for stat in ("moneyline", "spread"):
                out[f"{season}:{int(r.week)}:{r.game_id}~{side}:{stat}"] = float(m)
        out[f"{season}:{int(r.week)}:{r.game_id}~game:total"] = float(r.home_score + r.away_score)
    return out


def game_board() -> list:
    """Every game in the latest game-lines pull: consensus (median across books) spread,
    total and moneylines, best price per side, the books' range, implied team totals,
    the market's margin-free home win chance, and movement since the game's first pull."""
    g = query_sqlite("SELECT * FROM game_lines", db_path=DB_PATH)
    if g.empty:
        return []
    latest = g[g.retrieved_at == g.retrieved_at.max()]
    games = []
    for eid, ev in latest.groupby("event_id"):
        home, away = ev.home_team.iloc[0], ev.away_team.iloc[0]
        sp = ev[(ev.market == "spreads") & (ev.outcome == home)]
        sp_away = ev[(ev.market == "spreads") & (ev.outcome == away)]
        tot = ev[(ev.market == "totals") & (ev.outcome == "Over")]
        tot_u = ev[(ev.market == "totals") & (ev.outcome == "Under")]
        ml = ev[ev.market == "h2h"]
        ml_home, ml_away = ml[ml.outcome == home].price, ml[ml.outcome == away].price
        first = g[(g.event_id == eid) & (g.retrieved_at == g[g.event_id == eid].retrieved_at.min())]
        f_sp = first[(first.market == "spreads") & (first.outcome == home)].point.median()
        f_tot = first[(first.market == "totals") & (first.outcome == "Over")].point.median()
        spread, total = sp.point.median(), tot.point.median()
        p_home = None
        if len(ml_home) and len(ml_away):
            ih, ia = 1 / decimal(median_price(ml_home)), 1 / decimal(median_price(ml_away))
            p_home = round(ih / (ih + ia), 3)
        num = lambda v: None if pd.isna(v) else float(v)
        games.append({
            "id": eid, "kickoff": ev.commence_time.iloc[0], "home": home, "away": away, "books": int(ev.bookmaker.nunique()),
            "spread": num(spread), "spread_lo": num(sp.point.min()), "spread_hi": num(sp.point.max()),
            "total": num(total), "total_lo": num(tot.point.min()), "total_hi": num(tot.point.max()),
            "ml_home": num(median_price(ml_home)), "ml_away": num(median_price(ml_away)),
            "best_home": num(ml_home.max()), "best_away": num(ml_away.max()),
            # the price at the consensus spread, each side (books off the consensus number are left out)
            "spread_price_home": num(median_price(sp[sp.point == spread].price)),
            "spread_price_away": num(median_price(sp_away[sp_away.point == -spread].price)) if not pd.isna(spread) else None,
            "total_price_over": num(median_price(tot[tot.point == total].price)),
            "total_price_under": num(median_price(tot_u[tot_u.point == total].price)),
            "p_home": p_home,
            "implied_home": None if pd.isna(spread) or pd.isna(total) else round((total - spread) / 2, 1),
            "implied_away": None if pd.isna(spread) or pd.isna(total) else round((total + spread) / 2, 1),
            "spread_move": None if pd.isna(f_sp) or pd.isna(spread) else round(float(spread - f_sp), 1),
            "total_move": None if pd.isna(f_tot) or pd.isna(total) else round(float(total - f_tot), 1),
            "first_pull": first.retrieved_at.iloc[0] if len(first) else None,
        })
    attach_game_model(games)
    return sorted(games, key=lambda x: x["kickoff"])


# The game-outcome model (src/): home-win probability for each game, scored by
# src.models.predict into projections/game_model/<season>_<variant>.csv. Two variants:
# team_only (team stats alone, independent of the market) and team_residual (starts
# from the market's odds and lets team stats adjust them). Backtest 2008-2024, matched
# games: market 66.7% winners / log loss 0.608, team_residual 66.7% / 0.607, team_only
# 63.9% / 0.638 -- neither beats the market; team_only is the one that disagrees with it.
GAME_MODEL_RECORD = {"market": {"accuracy": 0.667, "log_loss": 0.608},
                     "team_residual": {"accuracy": 0.667, "log_loss": 0.607},
                     "team_only": {"accuracy": 0.639, "log_loss": 0.638}}


def attach_game_model(games: list) -> None:
    """Add model_home (team_only) and residual_home (team_residual) p(home win) to each game, matched on
    kickoff date and teams via nflverse's schedule (game_id YYYY_WW_AWAY_HOME)."""
    preds = {}
    for v in ("team_only", "team_residual"):
        for f in glob.glob(f"projections/game_model/*_{v}.csv"):
            d = pd.read_csv(f)
            preds[v] = dict(zip(d.game_id, d.p_home))
    if not preds:
        return
    seasons = sorted({int(g["kickoff"][:4]) - (1 if g["kickoff"][5:7] < "03" else 0) for g in games})
    sched = nfl.import_schedules(seasons)
    # team_desc lists some names under two codes (Los Angeles Rams: LA and LAR); keep the one the schedule uses.
    teams = nfl.import_team_desc()[["team_name", "team_abbr"]]
    teams = teams[teams.team_abbr.isin(set(sched.home_team) | set(sched.away_team))]
    abbr = dict(zip(teams.team_name, teams.team_abbr))
    key = {(r.away_team, r.home_team, str(r.gameday)): r.game_id for r in sched.itertuples()}
    for g in games:
        day = pd.Timestamp(g["kickoff"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
        gid = key.get((abbr.get(g["away"]), abbr.get(g["home"]), day))
        for v, out in (("team_only", "model_home"), ("team_residual", "residual_home")):
            p = preds.get(v, {}).get(gid)
            g[out] = None if p is None or pd.isna(p) else round(float(p), 3)
        g["game_id"] = gid


def ledger_history(ledger: pd.DataFrame) -> dict:
    """Settled paper bets by week and by stake size."""
    s = ledger[ledger.result.isin(["win", "loss", "push"])] if len(ledger) else ledger
    def agg(g):
        return {"bets": int(len(g)), "wins": int((g.result == "win").sum()), "losses": int((g.result == "loss").sum()),
                "staked": float(g.stake.sum()), "profit": round(float(g.profit.sum()), 2)}
    if not len(s):
        return {"weeks": [], "stakes": []}
    return {"weeks": [{"season": int(k[0]), "week": int(k[1]), **agg(g)} for k, g in s.groupby(["season", "week"])],
            "stakes": [{"stake": float(k), **agg(g)} for k, g in s.groupby("stake")]}


def ledger_summary(ledger: pd.DataFrame) -> dict:
    settled = ledger[ledger.result.isin(["win", "loss", "push"])] if len(ledger) else ledger
    return {"bets": int(len(ledger)), "settled": int(len(settled)),
            "wins": int((settled.result == "win").sum()) if len(settled) else 0,
            "losses": int((settled.result == "loss").sum()) if len(settled) else 0,
            "staked": float(settled.stake.sum()) if len(settled) else 0.0,
            "profit": float(settled.profit.sum()) if len(settled) else 0.0,
            "open_stake": float(ledger[ledger.result.isna()].stake.sum()) if len(ledger) else 0.0}


def hit_ledger(season: int, week: int, path: str = "projections/paper_trades_hit55.csv") -> dict:
    """A hit-rate rule's ledger (paper_trades.py, rule hit55 or hit55f): totals plus this week's bets."""
    led = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    wk = led[(led.season == season) & (led.week == week)] if len(led) else led
    cols = ["player_name", "position", "stat", "side", "line", "projection", "price", "chance", "stake", "actual", "result", "profit"]
    return {**ledger_summary(led), "week_bets": json.loads(wk[cols].to_json(orient="records")) if len(wk) else []}


def draftkings_sids(event_ids: set, to_id) -> dict:
    """(player_id, stat) -> {"over": {"sid", "line"}, "under": {...}}: DraftKings' own outcome ids for this week's
    events (prop_sids, saved by StatScraper.get_prop_lines), latest pull per outcome. The board's slip turns them
    into a betslip link; DraftKings' line can differ from the board's consensus line, so it travels with the sid."""
    try:
        s = query_sqlite("SELECT * FROM prop_sids", db_path=DB_PATH)
    except Exception:   # no table until the first pull with sids
        return {}
    s = s[s.event_id.isin(event_ids)].sort_values("retrieved_at").drop_duplicates(["event_id", "market", "player_name", "side"], keep="last")
    s = s.assign(player_id=s.player_name.map(norm_name).map(to_id)).dropna(subset=["player_id"])
    out = {}
    for r in s.itertuples():
        out.setdefault((r.player_id, r.stat), {})[r.side.lower()] = {"sid": r.sid, "line": r.line}
    return out


SIDE_FROM = {"oppf": 7, "norush": 8, "norush_avoid": 8}   # 2026 week each side test is judged from


def side_ledger(ledger: pd.DataFrame, season: int, week: int, test: str = "oppf") -> dict:
    """A side test on the main rule (nothing placed separately). Totals are the kept bets; week_bets lists
    this week's dropped ones.
      "oppf"   (paper_trades.WEAK_D): minus unders vs weak defenses, from 2026 week 7.
      "norush" minus every rushing-yards bet, from 2026 week 8 (2026-10-27: the weeks 6-7 replay went
               9.3% -> 13.2% ROI; those weeks chose it, so week 8 on is the test).
      "norush_avoid" minus rushing AND minus bets the board rates Avoid (edge_tier), from 2026 week 8 (
               2026-10-27: the only filter positive in both replay weeks, 6-7: +2.9% / +24.5%)."""
    if not len(ledger) or (test == "oppf" and "opp_allow" not in ledger):
        return None
    led = ledger[(ledger.season > 2026) | (ledger.week >= SIDE_FROM[test])]
    if test == "oppf":
        drop = (led.side == "Under") & (led.opp_allow >= WEAK_D)
    elif test == "norush":
        drop = led.stat == "rushing_yards"
    else:
        drop = pd.Series((led.stat == "rushing_yards").to_numpy() | (bet_tiers(led).to_numpy() == 0), index=led.index)
    wk = led[drop & (led.season == season) & (led.week == week)]
    cols = ["player_name", "position", "stat", "side", "line", "projection", "price", "chance", "stake", "actual", "result", "profit"]
    return {**ledger_summary(led[~drop]), "week_bets": json.loads(wk[cols].to_json(orient="records")) if len(wk) else [],
            "main": ledger_summary(led), "from_week": SIDE_FROM[test]}   # main: the whole rule over the same weeks


EDGE_Z = 0.17   # mirrors site/app/src/lib/model.ts EDGE_Z


def edge_tier(stat: str, side: str, line: float, proj: float, gbm, sleeper, sd) -> int:
    """The board's confidence tier (site/app/src/lib/model.ts edgeScore) in Python: 0 Avoid .. 3 High.
    +1 GBM on the bet's side, +1 Sleeper on it, +1 a gap above EDGE_Z of his game-to-game sd (projection on
    that side), +0.5 receptions, -1.5 rushing. test_prop_lines.demo_edge_tier_matches_board() pins the two."""
    s = 1 if side == "Over" else -1
    on = lambda v: v is not None and not pd.isna(v) and s * (v - line) > 0
    pts = float(on(gbm)) + float(on(sleeper))
    if sd and not pd.isna(sd) and on(proj) and abs(proj - line) / max(sd, 1) > EDGE_Z:
        pts += 1
    pts += 0.5 * (stat == "receptions") - 1.5 * (stat == "rushing_yards")
    return 0 if pts < 1 else 1 if pts < 2 else 2 if pts < 3 else 3


def bet_tiers(led: pd.DataFrame) -> pd.Series:
    """edge_tier for each ledger bet, from that week's own pregame projection file (GBM/Sleeper columns) and
    consistency() (games before the week). A bet whose projection row is missing rates by stat alone."""
    tiers = np.ones(len(led), dtype=int)   # by position: the main ledger is two concatenated CSVs, so labels repeat
    cache = {}
    for i, b in enumerate(led.itertuples(index=False)):
        if (b.season, b.week) not in cache:
            cache[(b.season, b.week)] = (consistency(b.season, b.week), {})
        form, proj = cache[(b.season, b.week)]
        if b.position not in proj:
            path = f"projections/{b.season}_w{b.week:02d}_{b.position}.csv"
            proj[b.position] = pd.read_csv(path).drop_duplicates("player_id").set_index("player_id") if os.path.exists(path) else None
        p = proj[b.position]
        row = p.loc[b.player_id] if p is not None and b.player_id in p.index else None
        pick = lambda c: row[c] if row is not None and c in row else None
        tiers[i] = edge_tier(b.stat, b.side, b.line, b.projection, pick(f"gbm_{b.stat}"), pick(f"sleeper_{b.stat}"),
                             (form.get((b.player_id, b.stat)) or {}).get("sd"))
    return pd.Series(tiers, index=led.index)


CONSISTENT_CV, VOLATILE_CV, FORM_GAMES = 0.55, 0.9, 8


def consistency(season: int, week: int) -> dict:
    """(player_id, stat) -> spread of his last FORM_GAMES regular-season games before this week, as
    std / mean (coefficient of variation; lower = steadier) and the games it uses. Pregame only.
    Weeks 2-6 settled bets: CV under CONSISTENT_CV won 67% (n=46), the rest ~50% (n=100)."""
    w = query_sqlite("SELECT player_id, season, week, passing_yards, rushing_yards, receiving_yards, receptions "
                     f"FROM player_weekly_stats WHERE season_type = 'REG' AND season >= {season - 1} "
                     f"AND season * 100 + week < {season * 100 + week}")
    out = {}
    for pid, g in w.sort_values(["season", "week"]).groupby("player_id"):
        g = g.tail(FORM_GAMES)
        for stat in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
            h = g[stat].dropna()
            if len(h) >= 2:
                out[(pid, stat)] = {"cv": round(float(h.std() / max(h.mean(), 1)), 2), "n": int(len(h)),
                                    "sd": round(float(h.std()), 1)}   # the board's edge size: |proj - line| / sd
    return out


def main(season: int, week: int) -> None:
    stats_for = {pos: stats + (["receptions"] if pos in REC_POS else []) for pos, stats in YARDS.items()}
    models = {(pos, stat): OverModel(blended_2025(pos), stat, SHRINK.get((pos, stat), 1.0)) for pos, stats in stats_for.items() for stat in stats}
    raw_models = {k: copy.copy(m) for k, m in models.items()}
    for m in raw_models.values():
        m.shrink = 1.0   # the ladder is calibrated by leaning on the line instead (see LADDER_W)

    lines = query_sqlite("SELECT * FROM prop_lines", db_path=DB_PATH)
    lines = lines[lines.retrieved_at == lines.retrieved_at.max()]
    sched = nfl.import_schedules([season])
    gamedays = set(sched.loc[(sched.game_type == "REG") & (sched.week == week), "gameday"].astype(str))
    lines = week_only(lines, gamedays)
    players = query_sqlite("SELECT DISTINCT player_id, player_display_name FROM player_weekly_stats "
                           "WHERE position_group IN ('QB', 'RB', 'WR', 'TE')", db_path=DB_PATH)
    lines = lines.assign(player_id=lines.player_name.map(norm_name).map(name_to_id(players))).dropna(subset=["player_id"])
    full_name = dict(zip(players.player_id, players.player_display_name))  # the Live tab matches ESPN box scores on it
    rec_line = lines[(lines.stat == "receptions") & (lines.side == "Over")].groupby("player_id").line.median()
    shop = {k: g[["bookmaker", "side", "line", "price"]].values.tolist() for k, g in lines.dropna(subset=["line", "price"]).groupby(["player_id", "stat"])}   # every book, every line, before the receptions filter
    lines = lines[(lines.stat != "receptions") | (lines.line == lines.player_id.map(rec_line))]   # books split 1.5 vs 2.5: price at the consensus line only
    odds = lines.groupby(["player_id", "stat", "side"]).price.agg(median_price).unstack("side")
    books = lines.groupby(["player_id", "stat"]).bookmaker.nunique()
    consensus = lines[lines.side == "Over"].groupby(["player_id", "stat"]).line.median()   # receptions: the CSVs lack them
    game =lines.groupby("player_id").agg(kickoff=("commence_time", "first"), home=("home_team", "first"), away=("away_team", "first"))
    dk = draftkings_sids(set(lines.event_id), name_to_id(players))

    # Main ledger since 2026 week 6: the 55% filtered rule. The original rule's ledger is shown as retired.
    main = [pd.read_csv(p) for p in MAIN_LEDGERS if os.path.exists(p)]
    ledger = pd.concat(main, ignore_index=True) if main else pd.DataFrame()
    old = pd.read_csv(OLD_LEDGER) if os.path.exists(OLD_LEDGER) else pd.DataFrame()
    bets = ledger[(ledger.season == season) & (ledger.week == week)].set_index(["player_id", "stat"]) if len(ledger) else None

    espn_rec = None
    if not (lines.stat == "receptions").any():   # weekly pulls are yardage only: DraftKings' line via ESPN, no price
        import log_closing
        e = log_closing.main(hours_ahead=24 * 7, db_path=None)
        e = week_only(e, gamedays) if len(e) else e   # the window can reach next week's games
        e = e[(e.stat == "receptions") & (e.side == "Over")].dropna(subset=["player_id"]) if len(e) else e
        espn_rec = dict(zip(e.player_id, e.line)) if len(e) else {}

    form = consistency(season, week)
    props = []
    for pos, stats in stats_for.items():
        proj = pd.read_csv(f"projections/{season}_w{week:02d}_{pos}.csv")
        for stat in stats:
            if f"pred_{stat}" not in proj:
                continue
            for _, r in proj.dropna(subset=[f"pred_{stat}"]).iterrows():
                key = (r.player_id, stat)
                if r.player_id not in game.index:
                    continue
                if stat == "receptions" and espn_rec is not None:
                    line, over, under, n_books = espn_rec.get(r.player_id), None, None, 1
                elif key in odds.index:
                    line, n_books = consensus.get(key) if stat == "receptions" else r.get(f"line_{stat}"), int(books.loc[key])
                    over, under = odds.loc[key].get("Over"), odds.loc[key].get("Under")
                else:
                    continue
                if line is None or pd.isna(line):
                    continue
                r[f"line_{stat}"] = line
                p = models[(pos, stat)].p_over(r[f"pred_{stat}"], r[f"line_{stat}"])
                p_mkt = None
                if not (pd.isna(over) or pd.isna(under)):
                    io, iu = 1 / decimal(over), 1 / decimal(under)
                    p_mkt = round(io / (io + iu), 3)
                g = game.loc[r.player_id]
                bet = bets.loc[key] if bets is not None and key in bets.index else None
                props.append({
                    "id": f"{r.player_id}:{stat}", "player": r.player_name, "name": full_name.get(r.player_id, r.player_name),
                    "pos": pos, "stat": stat,
                    "game": f"{g.away} @ {g.home}", "kickoff": g.kickoff,
                    "line": r[f"line_{stat}"], "over": None if over is None or pd.isna(over) else over,
                    "under": None if under is None or pd.isna(under) else under,
                    "books": n_books,
                    "proj": round(r[f"pred_{stat}"], 1),
                    "sources": {k: (None if pd.isna(r.get(f"{k}_{stat}")) else round(r[f"{k}_{stat}"], 1)) for k in ("own", "sleeper", "gbm")},
                    "gap": round(r[f"pred_{stat}"] - r[f"line_{stat}"], 1),
                    "p_over": round(p, 3), "p_market": p_mkt,
                    "ladder": raw_models[(pos, stat)].ladder(r[f"pred_{stat}"], r[f"line_{stat}"],
                                                             **({"w": REC_W, "k": REC_K} if stat == "receptions" else {})),
                    "ev_over": None if over is None or pd.isna(over) else round(p * decimal(over) - 1, 3),
                    "ev_under": None if under is None or pd.isna(under) else round((1 - p) * decimal(under) - 1, 3),
                    "form": form.get(key),
                    "dk": dk.get(key),
                    "shop": shop.get(key, []),   # [book, side, line, price] per book: the page picks the best for each bet
                    "actual": None if pd.isna(r.get(f"actual_{stat}")) else r[f"actual_{stat}"],
                    "bet": None if bet is None else {"side": bet.side, "stake": bet.stake, "price": bet.price,
                                                     "result": None if pd.isna(bet.result) else bet.result,
                                                     "profit": None if pd.isna(bet.profit) else bet.profit},
                })
    data = {
        "season": season, "week": week,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lines_pulled_at": lines.retrieved_at.max(),
        "props": props,
        "ladder_k": LADDER_K,
        "no_under": sorted(NO_UNDER),   # the paper rules' never-bet-the-under players; the shortlist skips them too
        "live_model": {f"{pos}:{stat}": m.live() for (pos, stat), m in models.items()},
        "calibration": calibration(models),
        "accuracy": accuracy(season, week),
        "history": ledger_history(ledger),
        "actuals": {**logged_actuals(season, week, players), **game_actuals(season)},
        "games": game_board(),
        "logs": season_logs(season, week, [p["id"].split(":")[0] for p in props]),
        "game_model_record": GAME_MODEL_RECORD,
        "game_lines_pulled_at": query_sqlite("SELECT MAX(retrieved_at) t FROM game_lines", db_path=DB_PATH).t.iloc[0],
        "ledger": ledger_summary(ledger),
        "ledger_old": ledger_summary(old),
        "ledger_hit": hit_ledger(season, week),
        "ledger_rec": hit_ledger(season, week, "projections/paper_trades_rec55.csv"),
        "ledger_oppf": side_ledger(ledger, season, week),
        "ledger_norush": side_ledger(ledger, season, week, "norush"),
        "ledger_norush_avoid": side_ledger(ledger, season, week, "norush_avoid"),
        # the main rule split by market -- rushing yards lost on real lines in the weeks 5-6 replay; watch it
        "ledger_by_stat": {s: ledger_summary(ledger[ledger.stat == s]) for s in sorted(ledger.stat.unique())} if len(ledger) else {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1, default=float)
    print(f"{len(props)} props for {season} week {week} -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(int(sys.argv[1]), int(sys.argv[2]))
