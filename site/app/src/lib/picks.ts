import type { BoardData } from "../types";
import type { SavedEntry, SavedParlay } from "./db";
import { dec } from "./format";
import { legResult } from "./model";

/** A saved parlay settled from the box scores: a miss loses it, a push drops that leg. */
export function parlayResult(data: BoardData, pl: SavedParlay) {
  const rs = pl.legs.map(l => legResult(data, pl.season, pl.week, l.id, l.stat, l.line, l.side === "Over"));
  if (rs.includes("miss")) return { status: "lost", profit: -pl.stake as number | null, rs };
  if (rs.includes("open")) return { status: "open", profit: null, rs };
  const d = pl.legs.reduce((acc, l, i) => (rs[i] === "push" ? acc : acc * dec(l.price)), 1);
  return { status: "won", profit: pl.stake * (d - 1) as number | null, rs };
}

/** A saved pick'em entry: all hit pays the full multiplier, flex pays by misses, a push needs the app. */
export function entryResult(data: BoardData, e: SavedEntry) {
  const rs = e.legs.map(l => legResult(data, e.season, e.week, l.id, l.stat, l.line, l.side === "More")), misses = rs.filter(r => r === "miss").length;
  const status = rs.includes("push") ? "check app" : rs.includes("open") ? "open" : misses === 0 ? "won" : (e.flex?.[misses] ?? 0) > 0 ? "flex paid" : "lost";
  const paid = status === "won" ? e.stake * e.mult_total : status === "flex paid" ? e.stake * e.flex[misses] : status === "lost" ? 0 : null;
  return { status, profit: paid == null ? null : paid - e.stake, rs };
}

/** The viewer's own saved parlays and pick'em entries, settled: record, profit, legs hit, still open. */
export function myRecord(data: BoardData, parlays: SavedParlay[], entries: SavedEntry[]) {
  const all = [...parlays.map(pl => parlayResult(data, pl)), ...entries.map(e => entryResult(data, e))];
  const settled = all.filter(r => r.profit != null), legs = all.flatMap(r => r.rs);
  return {
    saved: all.length, settled: settled.length, won: settled.filter(r => (r.profit as number) > 0).length,
    profit: settled.reduce((a, r) => a + (r.profit as number), 0), open: all.length - settled.length,
    hits: legs.filter(r => r === "hit").length, misses: legs.filter(r => r === "miss").length,
  };
}
