import { Fragment, useState } from "react";
import { useBoard } from "../context";
import { money, pct, pct1, short, statLabel } from "../lib/format";
import { missDist, pkChance } from "../lib/model";
import { entryResult } from "../lib/picks";
import { useStored } from "../lib/store";
import { Conf, LegLabel, PlayerLink } from "./common";
import type { Prop } from "../types";

interface PkLeg { id: string; side: "More" | "Less"; line: number | string | null; mult: number | string | null }
const num = (v: number | string | null) => (v === "" || v == null ? null : Number(v));
const label = (p: Prop) => `${p.player} · ${statLabel(p.stat)} · ${short(p.game.split(" @ ")[0])} @ ${short(p.game.split(" @ ")[1])}`;
const pillFor = (r: string) => (r === "hit" || r === "won" || r === "flex paid" ? "win" : r === "miss" || r === "lost" ? "loss" : "");

export default function Pickem() {
  const { data, dbx } = useBoard();
  const [pk, setPk] = useStored<PkLeg[]>("pb-pk", []);
  const [entry, setEntry] = useStored<{ stake: number | string; flex: (number | string)[] }>("pb-pk-entry", { stake: 10, flex: ["", "", ""] });
  const [note, setNote] = useState(""), [confirm, setConfirm] = useState<string | null>(null), [search, setSearch] = useState("");
  const legs = pk.map(l => { const p = data.props.find(x => x.id === l.id); if (!p) return null;
    const line = num(l.line), mult = num(l.mult); return { ...l, p, line, mult, chance: pkChance(data, p, l.side, line) }; })
    .filter((l): l is NonNullable<typeof l> => l != null);
  const ready = legs.length >= 2 && legs.every(l => l.chance != null && (l.mult ?? 0) > 0);
  const flex = entry.flex.map(v => (v === "" ? 0 : Number(v)));
  const st = ready ? (() => {
    const M = legs.reduce((a, l) => a * (l.mult as number), 1), dist = missDist(legs.map(l => l.chance as number));
    return { M, pAll: dist[0], evMax: dist[0] * M - 1, dist, evFlex: flex.some(v => v > 0) ? dist.reduce((a, v, k) => a + v * (flex[k] || 0), 0) - 1 : null,
      sameGame: new Set(legs.map(l => l.p.game)).size < legs.length };
  })() : null;
  const stake = Number(entry.stake) || 0;
  const update = (i: number, patch: Partial<PkLeg>) => setPk(list => list.map((l, j) => (j === i ? { ...l, ...patch } : l)));
  const add = (v: string) => {
    const p = data.props.find(x => label(x) === v); if (!p) return;
    setPk(list => [...list, { id: p.id, side: (p.ladder?.p_over ?? 0.5) >= 0.5 ? "More" : "Less", line: p.line, mult: "" }]); setSearch("");
  };
  const save = async () => {
    if (!st) return;
    const doc = { created_at: new Date().toISOString(), season: data.season, week: data.week, stake, mult_total: st.M, flex, p_all: st.pAll, ev_max: st.evMax, ev_flex: st.evFlex,
      legs: legs.map(l => ({ id: l.id, player: l.p.player, stat: l.p.stat, game: l.p.game, side: l.side, line: l.line, mult: l.mult, chance: l.chance })) };
    setNote("Saving…");
    try { await dbx.add("pickem", doc); setNote("Saved."); setPk([]); } catch (e) { const err = e as { code?: string; message?: string }; setNote(`Could not save (${err.code || err.message}).`); }
  };
  const remove = async (id: string) => {
    if (confirm !== id) { setConfirm(id); return; }
    setConfirm(null);
    dbx.remove("pickem", id).catch((e: { code?: string; message?: string }) => setNote(`Could not delete (${e.code || e.message}).`));
  };
  const commit = (fn: (v: string) => void) => ({
    onBlur: (e: React.FocusEvent<HTMLInputElement>) => fn(e.target.value),
    onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); },
  });
  return (
    <section className="layout">
      <div style={{ display: "grid", gap: 12, minWidth: 0 }}>
        <h2>Pick'em calculator</h2>
        <p className="note">For Sleeper Picks and other pick'em apps: add picks, choose More or Less, and type the app's line and multiplier for that pick. Our chance comes from the same model as the Hit rates tab, so it works for any line, not just the sportsbook's. A pick is worth taking when our chance beats its break-even (1 ÷ multiplier).</p>
        <label className="sub" htmlFor="pk-add">Add a pick <input id="pk-add" list="pk-props" type="search" placeholder="Search a player" style={{ minWidth: 240 }}
          value={search} onChange={e => { setSearch(e.target.value); add(e.target.value); }} /></label>
        <datalist id="pk-props">{data.props.filter(p => p.ladder?.dist).map(p => <option key={p.id} value={label(p)} />)}</datalist>
        <div className="panel scroll"><table>
          <thead><tr><th>Pick</th><th>Side</th><th className="r">App line</th><th className="r">Multiplier</th><th className="r">Our chance</th><th className="r">Break-even</th><th className="r">Edge</th><th></th></tr></thead>
          <tbody>{legs.length ? legs.map((l, i) => {
            const be = (l.mult ?? 0) > 0 ? 1 / (l.mult as number) : null, edge = l.chance != null && (l.mult ?? 0) > 0 ? l.chance * (l.mult as number) - 1 : null;
            return <tr key={`${l.id}-${i}`}>
              <td><div className="player"><PlayerLink id={l.p.id}>{l.p.player}</PlayerLink> <span className="sub">{statLabel(l.p.stat)}</span></div><div className="sub">{l.p.game} · book line {l.p.line} · proj {l.p.proj}</div></td>
              <td><span className="side-btns">{(["More", "Less"] as const).map(s => <button key={s} type="button" aria-pressed={l.side === s} onClick={() => update(i, { side: s })}>{s}</button>)}</span></td>
              <td className="r"><input type="number" step={0.5} defaultValue={l.line ?? ""} aria-label={`App line for ${l.p.player}`} {...commit(v => update(i, { line: v }))} /></td>
              <td className="r"><input type="number" step={0.01} min={1} defaultValue={l.mult ?? ""} placeholder="e.g. 1.8" aria-label={`Multiplier for ${l.p.player}`} {...commit(v => update(i, { mult: v }))} /></td>
              <td className="r"><Conf p={l.chance} id={l.id} side={l.side} /></td>
              <td className="r num">{be == null ? "-" : pct(be)}</td>
              <td className={`r num ${edge != null && edge > 0 ? "edge-pos" : "edge-neg"}`}>{edge == null ? "-" : `${edge > 0 ? "+" : ""}${(100 * edge).toFixed(0)}%`}</td>
              <td><button type="button" className="btn ghost" aria-label={`Remove ${l.p.player}`} onClick={() => setPk(list => list.filter((_, j) => j !== i))}>×</button></td></tr>;
          }) : <tr><td colSpan={8} className="empty">No picks yet. Search a player above to add one.</td></tr>}</tbody>
        </table></div>
        <h2 style={{ marginTop: 8 }}>Saved entries</h2>
        <p className="note">Paper only, saved with the lines, multipliers and chances at the time; each settles from the box score once its games are played. A pick that lands exactly on its line shows as a push. Apps usually drop it and recalculate, so check the app for that entry.</p>
        {dbx.where && <p className="note">{dbx.where}</p>}
        <div>{!dbx.entries.length ? <div className="panel empty">{dbx.entriesNote || "No saved entries yet."}</div>
          : <div className="recs">{dbx.entries.map(e => {
            const { status, profit, rs } = entryResult(data, e);
            return <div key={e.id} className="panel rec"><div><b>{e.legs.length}-pick entry</b> <span className="num muted">{e.mult_total.toFixed(2)}x</span> · ${e.stake} <span className={`pill ${pillFor(status)}`}>{status}</span>{profit != null && <> <span className="num">{money(profit)}</span></>}</div>
              <ul>{e.legs.map((l, j) => <li key={j}><LegLabel l={l} /> <span className="muted">({l.mult}x)</span> <Conf p={l.chance} id={l.id} week={e.week} side={l.side} /> <span className={`pill ${pillFor(rs[j])}`}>{rs[j]}</span></li>)}</ul>
              <div className="sub">{e.season} week {e.week} · our chance all hit {pct1(e.p_all)} · Max EV {(100 * e.ev_max).toFixed(1)}%</div>
              <button type="button" className="btn ghost" onClick={() => remove(e.id)}>{confirm === e.id ? "Confirm delete" : "Delete"}</button></div>;
          })}</div>}</div>
      </div>
      <aside className="panel slip">
        <h2>Entry</h2>
        <label className="sub">Entry fee ($) <input type="number" min={1} step={1} id="pk-stake" defaultValue={entry.stake} {...commit(v => setEntry(x => ({ ...x, stake: v })))} /></label>
        {!st ? <p className="note">{legs.length < 2 ? "Add at least 2 picks." : "Enter a line and multiplier for every pick."}</p> : <dl className="kv">
          <dt>Picks</dt><dd className="num">{legs.length}</dd>
          <dt>Max multiplier (all hit)</dt><dd className="num">{st.M.toFixed(2)}x · pays {money(stake * st.M)}</dd>
          <dt>Our chance all hit</dt><dd className="num">{pct1(st.pAll)}</dd>
          <dt>Max: expected value</dt><dd className={`num ${st.evMax > 0 ? "edge-pos" : ""}`}>{st.evMax > 0 ? "+" : ""}{(100 * st.evMax).toFixed(1)}% ({money(stake * st.evMax)})</dd>
          {st.dist.slice(1, 3).map((v, k) => <Fragment key={k}><dt>Chance of exactly {k + 1} miss{k ? "es" : ""}</dt><dd className="num">{pct1(v)}</dd></Fragment>)}
        </dl>}
        <div className="sub">Flex payouts from the app (multiplier for all correct, 1 miss, 2 misses), blank if none:</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>{[0, 1, 2].map(k =>
          <input key={k} type="number" step={0.01} min={0} defaultValue={entry.flex[k]} style={{ width: 70 }} aria-label={`Flex payout, ${k} misses`}
            {...commit(v => setEntry(x => ({ ...x, flex: x.flex.map((f, j) => (j === k ? v : f)) })))} />)}</div>
        {st && st.evFlex != null && <dl className="kv"><dt>Flex: expected value</dt><dd className={`num ${st.evFlex > 0 ? "edge-pos" : ""}`}>{st.evFlex > 0 ? "+" : ""}{(100 * st.evFlex).toFixed(1)}% ({money(stake * st.evFlex)})</dd></dl>}
        {st?.sameGame && <div className="warn">Two picks are in the same game. Their results move together, so the chance shown is off.</div>}
        <p className="note">Chances multiply as if picks were unrelated. They're already pulled toward 50% by what week 5's real lines showed, but that's one week, so treat a thin edge as no edge.</p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button type="button" className="btn" disabled={!st} onClick={save}>Save entry</button>
          <button type="button" className="btn ghost" onClick={() => { setPk([]); setNote(""); }}>Clear</button>
        </div>{note && <div className="sub">{note}</div>}
      </aside>
    </section>
  );
}
