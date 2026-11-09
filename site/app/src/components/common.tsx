import { useBoard } from "../context";
import type { ReactNode } from "react";
import type { Prop, Side } from "../types";
import { pct, statLabel } from "../lib/format";
import { edgeScore, legText, ourChance } from "../lib/model";

export const Tiles = ({ items }: { items: [ReactNode, ReactNode][] }) => (
  <>{items.map(([v, l], i) => <div key={i} className="panel tile"><div className="v num">{v}</div><div className="l">{l}</div></div>)}</>
);
export const Pill = ({ kind, children }: { kind?: string; children: ReactNode }) => <span className={`pill ${kind ?? ""}`}>{children}</span>;
/** Confidence levels by number: 0 Avoid, 1 Low, 2 Medium, 3 High (lib/model.ts edgeScore). */
export const CONF_LEVELS = ["Avoid", "Low", "Medium", "High"] as const;
/** The side a pick is on when the caller doesn't say: the side our chance favours. */
const ourSide = (p: Prop): Side => (ourChance(p) >= 0.5 ? "Over" : "Under");
/** Confidence that our number beats the line on this side; null for game bets (no projections to compare). */
export const confLevel = (p: Prop | null | undefined, side?: Side) => (p && p.sources ? edgeScore(p, side ?? ourSide(p)) : null);
/** Sort key for confidence: the level, then the chance within it; unrated picks last. */
export const confRank = (p: Prop, chance: number, side?: Side) => (confLevel(p, side)?.level ?? -1) + chance;
/** A pick's chance with our confidence in it (bare: the confidence alone). Looks the player up on this week's board by prop id. */
export function Conf({ p, id, week, bare, side }: { p: number | null | undefined; id: string; week?: number; bare?: boolean; side?: Side | "More" | "Less" }) {
  const { data } = useBoard();
  if (p == null) return <span className="muted">-</span>;
  const prop = week == null || week === data.week ? data.props.find(x => x.id === id) : null;
  const c = confLevel(prop, side === "More" ? "Over" : side === "Less" ? "Under" : side);
  if (c == null) return <span className="conf"><span className="num">{pct(p)}</span></span>;
  const n = c.level, why = c.why.length ? c.why.join("; ") : "neither of our other projections backs this side";
  return <span className={`conf c${n}`} title={`${n === 0 ? "Avoid" : `${CONF_LEVELS[n]} confidence`} that we beat the line: ${why}`}>{!bare && <span className="num">{pct(p)}</span>}
    <span className="dots" aria-hidden="true">{[1, 2, 3].map(k => <i key={k} className={k <= n ? "on" : ""} />)}</span>{CONF_LEVELS[n]}</span>;
}
/** A player's name that opens his breakdown; plain text for teams and players not on this week's board. */
export function PlayerLink({ id, children }: { id: string; children: ReactNode }) {
  const { data, openPlayer } = useBoard(), pid = id.split(":")[0];
  if (id.includes("~") || !data.props.some(p => p.id.split(":")[0] === pid)) return <>{children}</>;
  return <button type="button" className="plink" onClick={() => openPlayer(pid)}>{children}</button>;
}
/** A bet leg as text, with the player's name clickable: "J.Chase Under 7.5 receptions", "Steelers +3". */
export const LegLabel = ({ l }: { l: { id: string; player: string; stat: string; side: string; line: number | null } }) =>
  l.stat === "moneyline" || l.stat === "spread" || l.stat === "total" ? <>{legText({ ...l, line: l.line ?? 0 })}</>
    : <><PlayerLink id={l.id}>{l.player}</PlayerLink> {l.side} {l.line} {statLabel(l.stat)}</>;
export const Empty = ({ cols, children }: { cols: number; children: ReactNode }) => <tr><td colSpan={cols} className="empty">{children}</td></tr>;
