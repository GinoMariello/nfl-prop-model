import { useEffect, useRef } from "react";
import { useBoard } from "../context";
import { fmtOdds, statLabel } from "../lib/format";
import { chance } from "../lib/model";
import type { GameLog, Prop, Stat } from "../types";
import { Conf } from "./common";

const STATS_FOR: Record<string, Stat[]> = { QB: ["passing_yards", "rushing_yards"], RB: ["rushing_yards", "receptions", "receiving_yards"],
  WR: ["receptions", "receiving_yards"], TE: ["receptions", "receiving_yards"] };
/** The season-average strip for each position: [log field, label]. */
const STRIP: Record<string, [keyof GameLog, string][]> = {
  QB: [["passing_yards", "Pass yds"], ["attempts", "Attempts"], ["passing_tds", "Pass TD"], ["rushing_yards", "Rush yds"], ["fantasy_points_ppr", "PPR pts"]],
  RB: [["rushing_yards", "Rush yds"], ["carries", "Carries"], ["receptions", "Rec"], ["receiving_yards", "Rec yds"], ["fantasy_points_ppr", "PPR pts"]],
  WR: [["receiving_yards", "Rec yds"], ["receptions", "Rec"], ["targets", "Targets"], ["receiving_tds", "Rec TD"], ["fantasy_points_ppr", "PPR pts"]],
  TE: [["receiving_yards", "Rec yds"], ["receptions", "Rec"], ["targets", "Targets"], ["receiving_tds", "Rec TD"], ["fantasy_points_ppr", "PPR pts"]],
};
const avg = (xs: (number | null)[]) => { const v = xs.filter((x): x is number => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const f1 = (v: number | null) => (v == null ? "-" : Math.abs(v) >= 20 ? v.toFixed(0) : v.toFixed(1));
const col = (logs: GameLog[], k: keyof GameLog) => logs.map(g => g[k] as number | null);

/** One bar per game on a labelled axis, this week's line dashed across: green where our pick (side) would have won, red where not. */
function GameBars({ logs, stat, line, proj, side }: { logs: GameLog[]; stat: Stat; line: number | null; proj: number | null; side: "Over" | "Under" | null }) {
  const W = 560, H = 200, m = { t: 16, r: 58, b: 26, l: 34 };
  const vals = col(logs, stat as keyof GameLog).map(v => v ?? 0);
  const hi = Math.max(1, ...vals, line ?? 0, proj ?? 0), step0 = hi / 3, mag = 10 ** Math.floor(Math.log10(step0));
  const tick = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= step0) as number, top = tick * Math.ceil((hi * 1.08) / tick);
  const y = (v: number) => H - m.b - (Math.max(0, v) / top) * (H - m.t - m.b), step = (W - m.l - m.r) / Math.max(1, logs.length);
  const bw = Math.min(38, step * 0.56);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${statLabel(stat)} by game this season`}>
      {Array.from({ length: Math.round(top / tick) + 1 }, (_, i) => i * tick).map(t => <g key={t}>
        <line className="grid" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} /><text x={m.l - 6} y={y(t) + 4} textAnchor="end">{f1(t)}</text></g>)}
      {logs.map((g, i) => { const v = vals[i], cx = m.l + step * i + step / 2, h = Math.max(3, y(0) - y(v));
        return <g key={g.week}>
          <path d={`M${cx - bw / 2},${y(0)} v${-(h - 3)} q0,-3 3,-3 h${bw - 6} q3,0 3,3 v${h - 3} z`}
            fill={line == null || side == null ? "var(--s1)" : (side === "Over" ? v > line : v < line) ? "var(--win)" : "var(--loss)"} fillOpacity={0.78}><title>{`Week ${g.week} vs ${g.opp}: ${v}`}</title></path>
          <text x={cx} y={H - 9} textAnchor="middle">{g.opp}</text></g>; })}
      {proj != null && <line x1={m.l} x2={W - m.r} y1={y(proj)} y2={y(proj)} stroke="var(--ink)" strokeOpacity={0.45} strokeWidth={1.5} strokeDasharray="2 3" />}
      {line != null && <line x1={m.l} x2={W - m.r} y1={y(line)} y2={y(line)} stroke="var(--accent)" strokeWidth={2} strokeDasharray="7 4" />}
      {logs.map((g, i) => <text key={g.week} x={m.l + step * i + step / 2} y={y(vals[i]) - 6} textAnchor="middle" className="lbl val">{f1(vals[i])}</text>)}
      {line != null && <g><rect x={W - m.r + 4} y={y(line) - 9} width={m.r - 6} height={18} rx={3} fill="var(--accent)" />
        <text x={W - m.r + 4 + (m.r - 6) / 2} y={y(line) + 4} textAnchor="middle" className="tag">{line}</text></g>}
      {proj != null && <text x={W - m.r + 6} y={y(proj) + (line != null && Math.abs(y(proj) - y(line)) < 18 ? (y(proj) > y(line) ? 20 : -12) : 4)}>proj {proj}</text>}
    </svg>
  );
}

/** This week's pick for one market: side, chance, confidence and how often he has cleared the line. */
function PickCard({ prop, vals }: { prop: Prop; vals: (number | null)[] }) {
  const { trust } = useBoard();
  const c = chance(prop, trust), side = c >= 0.5 ? "Over" : "Under", played = vals.filter((v): v is number => v != null);
  const cleared = played.filter(v => (side === "Over" ? v > prop.line : v < prop.line)).length;
  return (
    <aside className="pl-pick">
      <div className="pl-pick-k">Our pick</div>
      <div className={`pl-pick-side ${side.toLowerCase()}`}>{side} {prop.line}</div>
      <Conf p={Math.max(c, 1 - c)} id={prop.id} />
      <dl className="kv">
        <dt>Projection</dt><dd className="num">{prop.proj}</dd>
        <dt>Odds O / U</dt><dd className="num">{fmtOdds(prop.over)} / {fmtOdds(prop.under)}</dd>
        <dt>{side} this season</dt><dd className="num">{played.length ? `${cleared} of ${played.length}` : "-"}</dd>
      </dl>
      {played.length > 0 && <div className="pl-meter" aria-hidden="true"><i style={{ width: `${(100 * cleared) / played.length}%` }} /></div>}
    </aside>
  );
}

/** The player breakdown: season averages, then each market game by game against this week's line with our pick. */
export default function PlayerPanel({ id, onClose }: { id: string; onClose: () => void }) {
  const { data, trust } = useBoard();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const d = ref.current; d?.showModal(); return () => d?.close(); }, [id]);
  const props = data.props.filter(p => p.id.split(":")[0] === id), p0 = props[0];
  const logs = data.logs?.[id] ?? [];
  if (!p0) return null;
  const stats = [...new Set([...(STATS_FOR[p0.pos] ?? []), ...props.map(p => p.stat)])];
  const team = logs.at(-1)?.team ?? "";
  const kick = new Date(p0.kickoff).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
  return (
    <dialog ref={ref} className="player-dlg" onClose={onClose} onClick={e => { if (e.target === ref.current) ref.current?.close(); }} aria-labelledby="pl-name">
      <header className="pl-head">
        <div className="pl-pos">{p0.pos}</div>
        <div className="pl-id">
          <h2 id="pl-name">{p0.name || p0.player}</h2>
          <div className="pl-meta">{team && <span>{team}</span>}<span>{logs.length} games played</span><span>Week {data.week}: {p0.game} · {kick}</span></div>
        </div>
        <button type="button" className="pl-close" aria-label="Close" onClick={() => ref.current?.close()}>×</button>
      </header>
      <div className="pl-body">
        {logs.length > 0 && <div className="pl-strip">{(STRIP[p0.pos] ?? []).map(([k, label]) => {
          const season = avg(col(logs, k)), last3 = avg(col(logs.slice(-3), k)), d = season != null && last3 != null ? last3 - season : 0;
          const trend = Math.abs(d) < 0.05 * Math.max(1, Math.abs(season ?? 0)) ? "" : d > 0 ? "up" : "down";
          return <div key={k}><div className="pl-strip-l">{label} / game</div><div className="pl-strip-v num">{f1(season)}</div>
            <div className={`pl-trend ${trend}`}>{trend === "up" ? "▲" : trend === "down" ? "▼" : "–"} last 3: {f1(last3)}</div></div>;
        })}</div>}
        {stats.map(stat => {
          const prop = props.find(p => p.stat === stat), vals = col(logs, stat as keyof GameLog);
          const side = prop ? (chance(prop, trust) >= 0.5 ? "Over" : "Under") : null;
          return <section key={stat} className="pl-market">
            <div className="pl-market-head"><h3>{statLabel(stat)}</h3>
              <span className="sub">season {f1(avg(vals))} · last 3 {f1(avg(vals.slice(-3)))}{prop ? ` · line ${prop.line}` : " · no line this week"}</span></div>
            <div className={`pl-market-body${prop ? "" : " solo"}`}>
              <div className="viz">{logs.length ? <GameBars logs={logs} stat={stat} line={prop?.line ?? null} proj={prop?.proj ?? null} side={side} />
                : <div className="empty">No games logged yet this season.</div>}</div>
              {prop && <PickCard prop={prop} vals={vals} />}
            </div>
          </section>;
        })}
        {logs.length > 0 && <section className="pl-market"><div className="pl-market-head"><h3>Game log</h3><span className="sub">regular season</span></div>
          <div className="scroll"><table className="pl-log">
            <thead><tr><th>Wk</th><th>Opp</th>{p0.pos === "QB" && <><th className="r">Cmp/Att</th><th className="r">Pass yds</th><th className="r">Pass TD</th></>}
              <th className="r">Car</th><th className="r">Rush yds</th><th className="r">Tgt</th><th className="r">Rec</th><th className="r">Rec yds</th><th className="r">TD</th><th className="r">PPR</th></tr></thead>
            <tbody>{logs.map(g => <tr key={g.week}><td className="num">{g.week}</td><td>{g.opp}</td>
              {p0.pos === "QB" && <><td className="r num">{g.completions ?? "-"}/{g.attempts ?? "-"}</td><td className="r num">{g.passing_yards ?? "-"}</td><td className="r num">{g.passing_tds ?? "-"}</td></>}
              <td className="r num">{g.carries ?? "-"}</td><td className="r num">{g.rushing_yards ?? "-"}</td><td className="r num">{g.targets ?? "-"}</td>
              <td className="r num">{g.receptions ?? "-"}</td><td className="r num">{g.receiving_yards ?? "-"}</td>
              <td className="r num">{(g.rushing_tds ?? 0) + (g.receiving_tds ?? 0)}</td><td className="r num">{g.fantasy_points_ppr ?? "-"}</td></tr>)}</tbody>
          </table></div></section>}
        <p className="note">Bars are his games this season against this week's line (gold): green where our pick would have won, red where it would have lost; the dotted line is our projection. Hover a bar for the game.</p>
      </div>
    </dialog>
  );
}
