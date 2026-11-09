import { useState } from "react";
import { useBoard } from "../context";
import { amer, fmtOdds, money, pct, pct1} from "../lib/format";
import { PARLAY_STAKE, buildParlays } from "../lib/model";
import { parlayResult } from "../lib/picks";
import { store, useStored } from "../lib/store";
import Slip from "./Slip";
import { Conf, LegLabel } from "./common";

const pillFor = (r: string) => (r === "hit" || r === "won" ? "win" : r === "miss" || r === "lost" ? "loss" : "");

export default function Parlays() {
  const b = useBoard(), { data, dbx, setSlip, minChance, trust } = b;
  const shuffleKey = `pb-shuffle-${data.season}-${data.week}`;
  const [shuffles, setShuffles] = useState(() => store.get(shuffleKey, 0));
  const [n, setN] = useStored("pb-legs", 2);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const top = buildParlays(data, minChance, n, shuffles, trust);
  const ev = top.reduce((a, c) => a + c.ev, 0) / Math.max(1, top.length);
  const res = dbx.parlays.map(pl => parlayResult(data, pl)), done = res.filter(r => r.status !== "open");
  const won = done.filter(r => r.status === "won").length;
  const staked = dbx.parlays.filter((_, i) => res[i].status !== "open").reduce((a, p) => a + p.stake, 0);
  const profit = done.reduce((a, r) => a + (r.profit ?? 0), 0);
  const del = async (id: string) => {
    if (confirmId !== id) { setConfirmId(id); return; }
    setConfirmId(null);
    try { await dbx.remove("parlays", id); } catch (e) { const err = e as { code?: string; message?: string }; dbx.setNote(`Could not delete (${err.code || err.message}).`); }
  };
  return (
    <section className="layout">
      <div style={{ display: "grid", gap: 12, minWidth: 0 }}>
        <h2>Recommended parlays</h2>
        <p className="note">Ten random 2-leg parlays from this week's shortlist (Hit rates tab), each pick used at most once and never two from the same game, at $2 each. Random on purpose: in weeks 4-5, ten random $2 parlays from these picks averaged +$5 to +$9 per $20, steadier than picking the "best" ones, which piled onto a few picks. They still lost money in about 1 week in 4-5. Reshuffle for a different set. 3-leg parlays pay about 6.5x instead of 3.5x and averaged more in weeks 4-5, but swing much harder: in week 4 over a third of 3-leg sets lost money and the worst tenth lost all $20, and in a week where picks hit ~50% they lose faster than 2-leg. Stay on 2-leg until weeks 6-8 confirm the picks.</p>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <span className="seg" role="group" aria-label="Legs per parlay">{[2, 3].map(k => <button key={k} type="button" aria-pressed={n === k} onClick={() => setN(k)}>{k}-leg</button>)}</span>
          <button type="button" className="btn ghost" onClick={() => { const v = shuffles + 1; store.set(shuffleKey, v); setShuffles(v); }}>Reshuffle</button>
          <span className="sub">{top.length ? `${top.length} parlays at $${PARLAY_STAKE} each = $${top.length * PARLAY_STAKE}; average expected value ${ev > 0 ? "+" : ""}${(100 * ev).toFixed(0)}% per parlay` : ""}</span>
        </div>
        <div className="recs">{top.length ? top.map((c, i) => (
          <div key={i} className="panel rec">
            <div><b>{c.legs.length}-leg parlay</b> <span className="num muted">{amer(c.d)}</span> · ${PARLAY_STAKE} pays {money(PARLAY_STAKE * (c.d - 1))}</div>
            <ul>{c.legs.map(l => <li key={l.p.id}><LegLabel l={{ ...l.p, side: l.side }} /> <span className="muted">({fmtOdds(l.price)})</span> <Conf p={l.prob} id={l.p.id} side={l.side} /></li>)}</ul>
            <dl className="kv"><dt>Our chance all hit</dt><dd className="num">{pct(c.prob)}</dd>
              <dt>Expected value</dt><dd className={`num ${c.ev > 0 ? "edge-pos" : ""}`}>{c.ev > 0 ? "+" : ""}{(100 * c.ev).toFixed(1)}%</dd></dl>
            <button type="button" className="btn" onClick={() => setSlip(c.legs.map(l => ({ id: l.p.id, side: l.side })))}>Load into slip</button>
          </div>))
          : <div className="panel empty">The shortlist has fewer than {n} props from different games at {pct(minChance)} this week, so there is nothing to pair. Lower the minimum chance on the Hit rates tab to see more.</div>}</div>
        <h2 style={{ marginTop: 8 }}>Saved parlays</h2>
        <p className="note">Paper only. Saved with the odds and chances at the time; each settles from the box score once its games are played. A push drops that leg; a void (player did not play) leaves it open here.</p>
        {dbx.where && <p className="note">{dbx.where}</p>}
        <div>{!dbx.parlays.length ? <div className="panel empty">No saved parlays yet. Build one in the slip and press Save parlay.</div>
          : <><div className="sub" style={{ marginBottom: 8 }}>{dbx.parlays.length} saved · {done.length ? `${won}-${done.length - won} settled, profit ${money(profit)} on ${money(staked)} staked` : "none settled yet"}</div>
            <div className="recs">{dbx.parlays.map((pl, i) => { const r = res[i]; return (
              <div key={pl.id} className="panel rec">
                <div><b>{pl.legs.length}-leg parlay</b> <span className="num muted">{amer(pl.decimal)}</span> · ${pl.stake} <span className={`pill ${pillFor(r.status)}`}>{r.status}</span>{r.profit != null && <> <span className="num">{money(r.profit)}</span></>}</div>
                <ul>{pl.legs.map((l, j) => <li key={j}><LegLabel l={l} /> <span className="muted">({fmtOdds(l.price)})</span> <Conf p={l.prob} id={l.id} week={pl.week} side={l.side} /> <span className={`pill ${pillFor(r.rs[j])}`}>{r.rs[j]}</span></li>)}</ul>
                <div className="sub">{pl.season} week {pl.week} · saved {new Date(pl.created_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })} · our chance {pct1(pl.prob)}{pl.mkt != null ? `, market ${pct1(pl.mkt)}` : ""}</div>
                <button type="button" className="btn ghost" onClick={() => del(pl.id)}>{confirmId === pl.id ? "Confirm delete" : "Delete"}</button>
              </div>); })}</div></>}</div>
      </div>
      <Slip />
    </section>
  );
}
