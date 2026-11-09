import type { Prop, Stat } from "../types";

export const STAT: Record<Stat, string> = { passing_yards: "pass yds", rushing_yards: "rush yds", receiving_yards: "rec yds", receptions: "receptions", moneyline: "moneyline", spread: "spread", total: "total points" };
export const statLabel = (s: string) => STAT[s as Stat] ?? s;
export const unit = (p: Prop) => (p.stat === "receptions" ? "rec" : "yds");
export const dec = (a: number) => 1 + (a < 0 ? 100 / -a : a / 100);
export const amer = (d: number) => (d >= 2 ? `+${Math.round((d - 1) * 100)}` : `${Math.round(-100 / (d - 1))}`);
export const fmtOdds = (a: number | null | undefined) => (a == null ? "-" : a > 0 ? `+${Math.round(a)}` : `${Math.round(a)}`);
export const pct = (x: number) => `${(100 * x).toFixed(0)}%`;
export const pct1 = (x: number) => `${(100 * x).toFixed(x < 0.1 ? 1 : 0)}%`;
export const money = (x: number) => `${x < 0 ? "-" : ""}$${Math.abs(x).toFixed(2)}`;
export const signedPct = (x: number, digits = 1) => `${x > 0 ? "+" : ""}${(100 * x).toFixed(digits)}%`;
export const short = (t: string) => String(t).split(" ").pop() ?? t;
export const signed = (v: number | null | undefined) => (v == null ? "-" : `${v > 0 ? "+" : ""}${v}`);
/** Break-even American price for a chance. */
export const fair = (p: number) => (p >= 0.5 ? `${Math.round((-100 * p) / (1 - p))}` : `+${Math.round((100 * (1 - p)) / p)}`);
export const implied = (a: number | null | undefined) => (a == null ? null : 1 / dec(a));
export const fmtTime = (s: string | Date, o: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) =>
  new Date(s).toLocaleString([], o);
export const BOOK: Record<string, string> = { draftkings: "DraftKings", fanduel: "FanDuel", betmgm: "BetMGM", betrivers: "BetRivers", bovada: "Bovada", betonlineag: "BetOnline", williamhill_us: "Caesars", espnbet: "ESPN Bet" };
export const bookName = (k: string) => BOOK[k] ?? k;
