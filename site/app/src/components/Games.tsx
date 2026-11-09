import { useBoard } from "../context";
import { fmtOdds, fmtTime, pct, short, signed, statLabel } from "../lib/format";
import type { Game, Side } from "../types";
import { gameProps, projectedScore } from "../lib/model";
import { Conf, PlayerLink } from "./common";
import Slip from "./Slip";

const Range = ({ lo, hi }: { lo?: number | null; hi?: number | null }) => (lo == null || lo === hi ? null : <span className="sub"> ({lo} to {hi})</span>);
const Moved = ({ v, unit }: { v: number | null; unit: string }) => (v == null || v === 0 ? <span className="sub">no move</span>
  : <span className={`sub ${v > 0 ? "move-up" : "move-down"}`}>{v > 0 ? "▲" : "▼"} {Math.abs(v)} {unit}</span>);
/** Where the team-stats model disagrees with the market by 5+ points of win chance. */
function Lean({ g }: { g: Game }) {
  if (g.model_home == null || g.p_home == null) return <div className="sub">No game-model pick for this game.</div>;
  const diff = g.model_home - g.p_home, team = short(diff > 0 ? g.home : g.away);
  if (Math.abs(diff) < 0.05) return <div className="sub">Team-stats model agrees with the market (within {Math.round(100 * Math.abs(diff))} pts).</div>;
  return <div><span className={`pill ${diff > 0 ? "over" : "under"}`}>Lean {team}</span> <span className="sub">team-stats model {Math.round(100 * Math.abs(diff))} pts more on {team} than the market</span></div>;
}

/** Moneyline and spread buttons for one game: each adds that team's bet to the slip, again removes it. */
function GameBets({ g }: { g: Game }) {
  const { data, slip, setSlip, trust } = useBoard();
  const bets = gameProps(data).filter(p => p.id.startsWith(`${g.game_id}~`));
  if (!bets.length) return null;
  // one button per side: a team's moneyline or spread, or the total's Over / Under; again removes it
  const toggle = (id: string, side: Side) => setSlip(s => (s.some(x => x.id === id && x.side === side) ? s.filter(x => x.id !== id)
    : [...s.filter(x => x.id !== id), { id, side }]));
  const btns = bets.flatMap(b => b.stat === "total"
    ? (["Over", "Under"] as const).map(side => ({ b, side, label: `${side} ${b.line}`, price: side === "Over" ? b.over : b.under }))
    : [{ b, side: "Over" as Side, label: `${short(b.player)}${b.stat === "spread" ? ` ${signed(b.line)}` : ""}`, price: b.over }]);
  return <div className="game-bets">{(["moneyline", "spread", "total"] as const).map(stat => {
    const row = btns.filter(x => x.b.stat === stat);
    return row.length ? <div key={stat}><span className="sub">{stat === "moneyline" ? "Moneyline" : stat === "spread" ? "Spread" : "Total"}</span>
      <span className="side-btns">{row.map(({ b, side, label, price }) => { const q0 = trust * b.p_over + (1 - trust) * (b.p_market ?? b.p_over), q = side === "Over" ? q0 : 1 - q0;
        return <button key={b.id + side} type="button" aria-pressed={slip.some(x => x.id === b.id && x.side === side)} aria-label={`${label} ${stat}`} onClick={() => toggle(b.id, side)}>
          <span>{label}</span><b>{fmtOdds(price)}</b><Conf p={q} id={b.id} /></button>; })}</span></div> : null;
  })}</div>;
}

export default function Games() {
  const { data } = useBoard();
  const games = data.games || [], r = data.game_model_record;
  const byDay: Record<string, Game[]> = {};
  games.forEach(g => { (byDay[new Date(g.kickoff).toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" })] ||= []).push(g); });
  return (
    <section className="layout"><div style={{ minWidth: 0 }}>
      <p className="note">{games.length ? `${games.length} games not yet started · consensus is the median across books · lines pulled ${fmtTime(data.game_lines_pulled_at)}; movement is since each game's first pull. `
        + (r ? `Game-model chances are home/away win chances. Backtested 2008-2024, the market picked ${pct(r.market.accuracy)} of winners, the market + stats model ${pct(r.team_residual.accuracy)} (a tie), the team-stats model ${pct(r.team_only.accuracy)} -- so a lean is where the weaker model disagrees, not a proven edge. Our projected score turns the team-stats model's win chance into a margin (NFL games land about 13.5 points either side of the expected margin) and splits it around the betting total; the model predicts who wins, not points.` : "") : ""}</p>
      {!games.length ? <div className="panel empty">No game lines logged yet. They come in with each Thursday and Saturday props pull.</div>
        : Object.entries(byDay).map(([day, gs]) => <div key={day}><div className="day">{day}</div><div className="games">{gs.map(g => {
          const props = data.props.filter(p => p.game === `${g.away} @ ${g.home}`).sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap));
          return <div key={g.id} className="panel game">
            <div className="game-head"><b>{short(g.away)} @ {short(g.home)}</b><span className="sub">{new Date(g.kickoff).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })} · {g.books} books</span></div>
            <table className="lines"><thead><tr><th></th><th className="r">{short(g.away)}</th><th className="r">{short(g.home)}</th></tr></thead><tbody>
              <tr><td>Spread</td><td className="r num">{signed(g.spread == null ? null : -g.spread)}</td><td className="r num">{signed(g.spread)}<Range lo={g.spread_lo} hi={g.spread_hi} /></td></tr>
              <tr><td>Moneyline</td><td className="r num">{fmtOdds(g.ml_away)}<div className="sub">best {fmtOdds(g.best_away)}</div></td><td className="r num">{fmtOdds(g.ml_home)}<div className="sub">best {fmtOdds(g.best_home)}</div></td></tr>
              <tr><td>Win chance</td><td className="r num">{g.p_home == null ? "-" : pct(1 - g.p_home)}</td><td className="r num">{g.p_home == null ? "-" : pct(g.p_home)}</td></tr>
              {(() => { const ps = projectedScore(g); return ps && <tr className="proj-score"><td>Projected score: team stats</td>
                <td className="r num">{ps.away.toFixed(1)}</td><td className="r num">{ps.home.toFixed(1)}</td></tr>; })()}
              <tr><td>Implied points</td><td className="r num">{g.implied_away ?? "-"}</td><td className="r num">{g.implied_home ?? "-"}</td></tr>
              <tr><td>Model: team stats</td><td className="r num">{g.model_home == null ? "-" : pct(1 - g.model_home)}</td><td className="r num">{g.model_home == null ? "-" : pct(g.model_home)}</td></tr>
              <tr><td>Model: market + stats</td><td className="r num">{g.residual_home == null ? "-" : pct(1 - g.residual_home)}</td><td className="r num">{g.residual_home == null ? "-" : pct(g.residual_home)}</td></tr>
            </tbody></table>
            <GameBets g={g} />
            <Lean g={g} />
            <div className="kv" style={{ fontSize: 13 }}><span className="muted">Total</span><span className="num">{g.total ?? "-"}<Range lo={g.total_lo} hi={g.total_hi} /></span>
              <span className="muted">Moved since first pull</span><span>spread <Moved v={g.spread_move} unit="pts" /> · total <Moved v={g.total_move} unit="pts" /></span></div>
            {props.length ? <details><summary>{props.length} player props in this game</summary>
              <table className="lines"><thead><tr><th>Player</th><th className="r">Line</th><th className="r">Proj</th><th className="r">Gap</th></tr></thead><tbody>
                {props.map(p => <tr key={p.id}><td><PlayerLink id={p.id}>{p.player}</PlayerLink> <span className="sub">{statLabel(p.stat)}</span></td><td className="r num">{p.line}</td><td className="r num">{p.proj}</td><td className="r num">{p.gap > 0 ? "+" : ""}{p.gap}</td></tr>)}
              </tbody></table></details> : <span className="sub">No player props logged for this game yet.</span>}
          </div>;
        })}</div></div>)}
    </div><Slip /></section>
  );
}
