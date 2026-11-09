import { useBoard } from "../context";
import { amer, fmtOdds, money, pct1} from "../lib/format";
import { dkLink, legInfo, legText, parlayStats, type Leg } from "../lib/model";
import { Conf, LegLabel } from "./common";

/** The parlay slip, shown beside the board and the parlays tab; saving needs the owner signed in. */
export default function Slip() {
  const { data, slip, setSlip, stake, setStake, trust, dbx } = useBoard();
  const legs = slip.map(s => legInfo(data, s, trust)).filter((l): l is Leg => l != null);
  const save = async () => {
    if (legs.length < 2) return;
    const st = parlayStats(legs);
    const doc = { created_at: new Date().toISOString(), season: data.season, week: data.week, stake, trust: Math.round(trust * 100),
      decimal: st.d, prob: st.prob, mkt: st.mkt,
      legs: legs.map(l => ({ id: l.p.id, player: l.p.player, stat: l.p.stat, side: l.side, line: l.p.line, price: l.price, game: l.p.game, prob: l.prob, mkt: l.mkt })) };
    dbx.setNote("Saving…");
    try { await dbx.add("parlays", doc); dbx.setNote("Saved. It is listed under Saved parlays."); setSlip([]); }
    catch (e) { const err = e as { code?: string; message?: string }; dbx.setNote(`Could not save (${err.code || err.message}).`); }
  };
  if (!legs.length) return <aside className="panel slip"><h2>Parlay slip</h2><p className="note">Add legs from the board with Over or Under. Two or more legs make a parlay; one leg is a straight bet.</p></aside>;
  const st = parlayStats(legs);
  const dk = dkLink(legs);
  return (
    <aside className="panel slip">
      <h2>Parlay slip</h2>
      <div className="legs">{legs.map(l => (
        <div key={l.p.id} className="leg"><div><b><LegLabel l={{ ...l.p, side: l.side }} /></b>
          <div className="sub">{l.p.game} · {fmtOdds(l.price)}</div><Conf p={l.prob} id={l.p.id} side={l.side} /></div>
          <button type="button" aria-label={`Remove ${l.p.player}`} onClick={() => setSlip(s => s.filter(x => x.id !== l.p.id))}>×</button></div>))}
      </div>
      <label className="sub">Stake ($) <input className="stake" type="number" min={1} step={1} defaultValue={stake}
        onBlur={e => setStake(Math.max(1, Number(e.target.value) || 1))}
        onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} /></label>
      <dl className="kv">
        <dt>Legs</dt><dd className="num">{legs.length}</dd>
        <dt>Odds</dt><dd className="num">{amer(st.d)} ({st.d.toFixed(2)}x)</dd>
        <dt>Pays if all hit</dt><dd className="num">{money(stake * (st.d - 1))}</dd>
        <dt>Our chance all hit</dt><dd className="num">{pct1(st.prob)}</dd>
        <dt>Market's chance all hit</dt><dd className="num">{st.mkt == null ? "-" : pct1(st.mkt)}</dd>
        <dt>Expected value</dt><dd className={`num ${st.ev > 0 ? "edge-pos" : ""}`}>{st.ev > 0 ? "+" : ""}{(100 * st.ev).toFixed(1)}% ({money(stake * st.ev)})</dd>
      </dl>
      <p className="note">Both chances multiply the legs together, which is right only if the legs are unrelated. Ours beats the market's only because each leg leans on the model, which is untested against lines.</p>
      {st.sameGame && <div className="warn">Two legs are in the same game. Their results are linked, so the chance shown is off, and books price same-game parlays differently.</div>}
      {st.samePlayer && <div className="warn">Two legs are on the same player, which most books will not allow.</div>}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button type="button" className="btn" disabled={legs.length < 2} onClick={save}>Save parlay</button>
        <button type="button" className="btn ghost" onClick={() => setSlip([])}>Clear slip</button>
        {dk.url && <a className="btn" href={dk.url} target="_blank" rel="noopener noreferrer">Open in DraftKings</a>}
      </div>
      {dk.url && (dk.missing.length > 0 || dk.moved.length > 0) && <div className="sub">
        {dk.missing.length > 0 && <div>Not on DraftKings' slip (no DraftKings id): {dk.missing.map(p => p.player).join(", ")}.</div>}
        {dk.moved.map(m => <div key={m.p.id}>DraftKings has {legText({ ...m.p, side: m.side, line: m.line })} (ours: {m.p.line}).</div>)}
      </div>}
      {dk.url && <div className="sub">Opens DraftKings with these legs on the betslip. Check the legs and price there before placing anything.</div>}
      {dbx.note && <div className="sub">{dbx.note}</div>}
    </aside>
  );
}
