import type { BoardData, Prop, Side } from "../types";
import { dec, short, statLabel } from "./format";

/** Chance of the over: the main rule's chance (the Hit rates ladder), weighted by trust (0-1) against the market's.
 * Trust 1 is exactly what the paper rule bets on. */
export const ourChance = (p: Prop) => p.ladder?.p_over ?? p.p_over;
export const chance = (p: Prop, trust: number) => (p.p_market == null ? ourChance(p) : trust * ourChance(p) + (1 - trust) * p.p_market);
export function edges(p: Prop, trust: number) {
  const q = chance(p, trust);
  return { q, over: p.over == null ? null : q * dec(p.over) - 1, under: p.under == null ? null : (1 - q) * dec(p.under) - 1 };
}
export function best(p: Prop, trust: number): { side: Side; ev: number | null } {
  const e = edges(p, trust);
  return (e.over ?? -9) >= (e.under ?? -9) ? { side: "Over", ev: e.over } : { side: "Under", ev: e.under };
}

export interface SlipLeg { id: string; side: Side }
export interface Leg { p: Prop; side: Side; price: number; d: number; prob: number; mkt: number | null }
/** Moneyline, spread and total bets as board props, so the slip, parlays and settlement treat them like any leg.
 * Id "<game_id>~home:spread"; line is the team's handicap (0 on the moneyline) and the bet is the "Over" side.
 * The total is "<game_id>~game:total" with real Over / Under sides; like spreads it has no model of ours.
 * Our moneyline chance is the market + team-stats model (a tie with the market in backtests); spreads have no
 * model, so their chance is the market's with the margin removed. */
export function gameProps(data: BoardData): Prop[] {
  const noVig = (a?: number | null, b?: number | null) => (a == null || b == null ? null : (1 / dec(a)) / (1 / dec(a) + 1 / dec(b)));
  return (data.games || []).flatMap(g => {
    if (!g.game_id) return [];
    const spMkt = noVig(g.spread_price_home, g.spread_price_away), ours = g.residual_home ?? g.p_home;
    const base = { name: "", pos: "", game: `${g.away} @ ${g.home}`, kickoff: g.kickoff, under: null, books: g.books, proj: 0,
      sources: { own: null, sleeper: null, gbm: null }, gap: 0, ev_over: null, ev_under: null, actual: null, bet: null, form: null };
    return (["away", "home"] as const).flatMap(side => {
      const home = side === "home", flip = (p: number | null | undefined) => (p == null ? null : home ? p : 1 - p);
      const ml = home ? g.ml_home : g.ml_away, sp = home ? g.spread_price_home : g.spread_price_away, out: Prop[] = [];
      if (ml != null && ours != null)
        out.push({ ...base, id: `${g.game_id}~${side}:moneyline`, player: home ? g.home : g.away, stat: "moneyline", line: 0, over: ml, p_over: flip(ours) as number, p_market: flip(g.p_home) });
      if (g.spread != null && sp != null)
        out.push({ ...base, id: `${g.game_id}~${side}:spread`, player: home ? g.home : g.away, stat: "spread", line: home ? g.spread : -g.spread, over: sp, p_over: flip(spMkt) ?? 0.5, p_market: flip(spMkt) });
      return out;
    }).concat(g.total != null && g.total_price_over != null && g.total_price_under != null ? [{
      ...base, id: `${g.game_id}~game:total`, player: `${short(g.away)} @ ${short(g.home)}`, stat: "total" as const, line: g.total,
      over: g.total_price_over, under: g.total_price_under,
      p_over: noVig(g.total_price_over, g.total_price_under) ?? 0.5, p_market: noVig(g.total_price_over, g.total_price_under) }] : []);
  });
}
export const findProp = (data: BoardData, id: string) => data.props.find(x => x.id === id) ?? gameProps(data).find(x => x.id === id);
/** How a leg reads: "Steelers +3", "Saints to win", "J.Chase Under 7.5 receptions". */
export const legText = (l: { player: string; stat: string; side: string; line: number }) =>
  l.stat === "moneyline" ? `${l.player} to win` : l.stat === "spread" ? `${l.player} ${l.line > 0 ? "+" : ""}${l.line}` : `${l.player} ${l.side} ${l.line} ${statLabel(l.stat)}`;

/** A slip leg priced with the trust-weighted chance. */
export function legInfo(data: BoardData, s: SlipLeg, trust: number): Leg | null {
  const p = findProp(data, s.id); if (!p) return null;
  const price = s.side === "Over" ? p.over : p.under; if (price == null) return null;
  const q = chance(p, trust), m = p.p_market;
  return { p, side: s.side, price, d: dec(price), prob: s.side === "Over" ? q : 1 - q, mkt: m == null ? null : s.side === "Over" ? m : 1 - m };
}
export function parlayStats(legs: Leg[]) {
  const d = legs.reduce((a, l) => a * l.d, 1), prob = legs.reduce((a, l) => a * l.prob, 1);
  const mkt = legs.every(l => l.mkt != null) ? legs.reduce((a, l) => a * (l.mkt as number), 1) : null;
  return { d, prob, mkt, ev: prob * d - 1,
    sameGame: new Set(legs.map(l => l.p.game)).size < legs.length,
    samePlayer: new Set(legs.map(l => l.p.player)).size < legs.length };
}

export const SHORT_MIN_LINE = 10, CUSHION = 0.03, BLOWOUT = 7;
export const spreadOf = (data: BoardData, game: string) => data.games.find(x => `${x.away} @ ${x.home}` === game)?.spread ?? null;
/** Likely blowouts: a 7+ point spread. Games with no logged spread are kept. */
export const blowout = (data: BoardData, game: string) => { const sp = spreadOf(data, game); return sp != null && Math.abs(sp) >= BLOWOUT; };

export interface ShortRow { p: Prop; side: Side; ch: number; price: number | null }
/** Every posted line at T+ that isn't a blowout, beats its price's break-even and isn't a never-under;
 * pregame only, unless every game has started (then all, for review). */
export function shortlist(data: BoardData, T: number, now = Date.now(), trust = 1): { rows: ShortRow[]; played: boolean } {
  const pick = (ps: Prop[]) => ps.map(p => {
    const po = chance(p, trust), side: Side = po >= 0.5 ? "Over" : "Under";
    return { p, side, ch: Math.max(po, 1 - po), price: side === "Over" ? p.over : p.under };
  }).filter(r => r.ch >= T && !blowout(data, r.p.game)
    && (r.price == null || r.ch * dec(r.price) > 1)
    && !(r.side === "Under" && (data.no_under || []).includes(r.p.id.split(":")[0])))
    .sort((a, b) => b.ch - a.ch);
  const withLadder = data.props.filter(p => p.ladder);
  let rows = pick(withLadder.filter(p => new Date(p.kickoff).getTime() > now)), played = false;
  if (!rows.length) { rows = pick(withLadder); played = rows.length > 0; }
  return { rows, played };
}

/** mulberry32: the same seeded stream as the original page, so a week's parlay set is unchanged. */
export function seeded(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const N_PARLAYS = 10, PARLAY_STAKE = 2;
export type Parlay = { legs: Leg[] } & ReturnType<typeof parlayStats>;
/** Ten random n-leg parlays from the shortlist: each pick once, every leg from a different game.
 * Parlays skip yardage lines under 10 and QB rushing; receptions stay in. */
export function buildParlays(data: BoardData, T: number, n: number, shuffles: number, trust: number, now = Date.now()): Parlay[] {
  const rand = seeded(data.season * 100 + data.week + 7919 * shuffles);
  const legs = shortlist(data, T, now, trust).rows
    .filter(r => r.p.stat === "receptions" || (r.p.line >= SHORT_MIN_LINE && !(r.p.pos === "QB" && r.p.stat === "rushing_yards")))
    .map(r => legInfo(data, { id: r.p.id, side: r.side }, trust)).filter((l): l is Leg => l != null);
  for (let i = legs.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [legs[i], legs[j]] = [legs[j], legs[i]]; }
  const used = new Set<number>(), out: Parlay[] = [];
  for (let i = 0; i < legs.length && out.length < N_PARLAYS; i++) {
    if (used.has(i)) continue;
    const group = [i];
    for (let k = i + 1; k < legs.length && group.length < n; k++)
      if (!used.has(k) && group.every(g => legs[g].p.game !== legs[k].p.game)) group.push(k);
    if (group.length < n) continue;
    group.forEach(g => used.add(g));
    const ls = group.map(g => legs[g]);
    out.push({ legs: ls, ...parlayStats(ls) });
  }
  return out;
}

/** Pick'em chance for any line, from the prop's 100-quantile range, shrunk like the Hit rates tab. */
export function pkChance(data: BoardData, p: Prop, side: "More" | "Less", line: number | null): number | null {
  const d = p.ladder?.dist; if (!d || line == null || Number.isNaN(line)) return null;
  const n = d.filter(v => (side === "More" ? v > line : v < line)).length;
  return 0.5 + (p.ladder?.k ?? data.ladder_k ?? 1) * (n / d.length - 0.5);
}
/** P(exactly k misses), picks independent. */
export function missDist(ps: number[]) {
  let dist = [1];
  ps.forEach(q => { const next = Array(dist.length + 1).fill(0); dist.forEach((v, k) => { next[k] += v * q; next[k + 1] += v * (1 - q); }); dist = next; });
  return dist;
}
/** A saved leg's result from the board's logged actuals. */
export function legResult(data: BoardData, season: number, week: number, id: string, stat: string, line: number, over: boolean) {
  const a = data.actuals?.[`${season}:${week}:${id.split(":")[0]}:${stat}`];
  if (a == null) return "open" as const;
  if (stat === "moneyline" || stat === "spread") {   // a is the team's final margin; line its handicap
    const m = a + line; return m === 0 ? ("push" as const) : m > 0 ? ("hit" as const) : ("miss" as const);
  }
  if (a === line) return "push" as const;
  return (a > line) === over ? ("hit" as const) : ("miss" as const);
}

/** Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9). */
export function probit(p: number) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = (q: number) => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
  if (p > 1 - 0.02425) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
/** NFL final margins scatter about 13.5 points around the expected margin. */
export const MARGIN_SD = 13.5;
/** Our projected score: the team-stats model's home win chance turned into an expected margin
 * (margin = 13.5 x probit(p)), split around the market's total -- the model predicts who wins, not points. */
export function projectedScore(g: { model_home?: number | null; total: number | null }) {
  if (g.model_home == null || g.total == null) return null;
  const margin = MARGIN_SD * probit(Math.min(0.99, Math.max(0.01, g.model_home)));
  return { home: (g.total + margin) / 2, away: (g.total - margin) / 2, margin };
}

/** Edge confidence: how likely our number beats the book's line, not how steady the player is (player consistency
 * showed no edge: 53/57/52% by CV tertile). Weeks 5-7, every logged line (n=879, our projection's side): +1 each when
 * the GBM and Sleeper projections land on the same side, +1 for an edge above EDGE_Z of the player's game-to-game
 * spread, +0.5 for receptions, -1.5 for rushing. Tiers: 0 Avoid (< 1), 1 Low (< 2), 2 Medium (< 3), 3 High (3+):
 * all lines won 39/54/50/64%, week 7 main-rule bets 57/63/65/72%. (A first cut with receptions +1 and rushing -2
 * put every catch prop in High and every rush in Avoid; softened 2026-10-27.) Weights chosen
 * on those same weeks: week 8 on is the first fair test. */
export const EDGE_Z = 0.17;
export function edgeScore(p: Prop, side: Side) {
  const s = side === "Over" ? 1 : -1, on = (v: number | null | undefined) => v != null && s * (v - p.line) > 0;
  const why: string[] = [];
  let pts = 0;
  if (on(p.sources?.gbm)) { pts++; why.push("our GBM model agrees"); }
  if (on(p.sources?.sleeper)) { pts++; why.push("Sleeper's projection agrees"); }
  if (p.form?.sd && on(p.proj) && Math.abs(p.proj - p.line) / Math.max(p.form.sd, 1) > EDGE_Z) { pts++; why.push("the gap to the line is big for his usual swings"); }
  if (p.stat === "receptions") { pts += 0.5; why.push("catches have been our best market"); }
  if (p.stat === "rushing_yards") { pts -= 1.5; why.push("rushing yards have lost most bets (RB 42%, QB 38%)"); }
  return { pts, level: pts < 1 ? 0 : pts < 2 ? 1 : pts < 3 ? 2 : 3, why };
}

/** A DraftKings betslip link for the slip's legs: https://sportsbook.draftkings.com/?outcomes=<sid>+<sid>, each sid
 * URL-encoded (its # becomes %23). Multi-leg links were confirmed by hand on 2026-10-29. Legs DraftKings has no id for
 * are listed in `missing`; legs whose DraftKings line differs from ours are in `moved` (DraftKings uses its own). */
export const DK_SLIP = "https://sportsbook.draftkings.com/?outcomes=";
export function dkLink(legs: { p: Prop; side: Side }[]) {
  const found = legs.map(l => ({ ...l, o: l.side === "Over" ? l.p.dk?.over : l.p.dk?.under }));
  const ok = found.filter(l => l.o);
  return {
    url: ok.length ? DK_SLIP + ok.map(l => encodeURIComponent(l.o!.sid)).join("+") : null,
    missing: found.filter(l => !l.o).map(l => l.p),
    moved: ok.filter(l => l.o!.line !== l.p.line).map(l => ({ p: l.p, side: l.side, line: l.o!.line })),
  };
}

/** The best book for a bet: each book's line and price, priced by our chance moved along the range
 * from the posted line to that book's line. Highest expected value wins. */
export function bestBook(data: BoardData, p: Prop, side: Side, trust: number) {
  const base = side === "Over" ? chance(p, trust) : 1 - chance(p, trust);
  const at = (l: number) => pkChance(data, p, side === "Over" ? "More" : "Less", l);
  const here = at(p.line);
  let top: { book: string; line: number; price: number; ch: number; ev: number } | null = null;
  for (const [book, s, line, price] of p.shop ?? []) {
    if (s !== side) continue;
    const a = at(line), ch = a == null || here == null ? (line === p.line ? base : null) : base + a - here;
    if (ch == null) continue;
    const ev = ch * dec(price) - 1;
    if (!top || ev > top.ev) top = { book, line, price, ch, ev };
  }
  return top;
}

/** Quarter Kelly, capped at 1% of the bankroll per bet; 0 when there is no edge. */
export const KELLY_FRACTION = 0.25, BET_CAP = 0.01, GAME_CAP = 0.03;
export function kellyStake(ch: number, price: number, bankroll: number) {
  const b = dec(price) - 1, f = (ch * (b + 1) - 1) / b;
  return f <= 0 ? 0 : bankroll * Math.min(KELLY_FRACTION * f, BET_CAP);
}
/** Stakes for a list of bets, scaled down per game so one game never holds more than 3% of the bankroll
 * (bets in the same game win or lose together). Rounded down to 50 cents. */
export function sizeStakes<T extends { game: string; stake: number }>(bets: T[], bankroll: number): T[] {
  const perGame = new Map<string, number>();
  bets.forEach(b => perGame.set(b.game, (perGame.get(b.game) ?? 0) + b.stake));
  return bets.map(b => {
    const scale = Math.min(1, (GAME_CAP * bankroll) / (perGame.get(b.game) || 1));
    return { ...b, stake: Math.floor(2 * b.stake * scale) / 2 };
  });
}
