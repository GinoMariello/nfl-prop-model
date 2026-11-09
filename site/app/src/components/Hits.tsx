import type { Side } from "../types";
import { useBoard } from "../context";
import { bookName, fair, fmtOdds, implied, money, pct, statLabel, unit } from "../lib/format";
import { BET_CAP, BLOWOUT, CUSHION, GAME_CAP, SHORT_MIN_LINE, bestBook, chance, kellyStake, shortlist, sizeStakes } from "../lib/model";
import { Conf, Empty, Pill, confRank } from "./common";
import { useStored } from "../lib/store";

export default function Hits({ minPct, setMinPct, onlyMeets, setOnlyMeets }: {
  minPct: number; setMinPct: (n: number) => void; onlyMeets: boolean; setOnlyMeets: (b: boolean) => void;
}) {
  const { data, pos, stat, q, trust, openPlayer } = useBoard();
  const T = minPct / 100, key = T.toFixed(2);
  const [by, setBy] = useStored("pb-hits-sort", "chance");
  const rank = (p: (typeof data.props)[number], ch: number) => (by === "conf" ? confRank(p, ch) : ch);
  const sl = shortlist(data, T, Date.now(), trust), played = sl.played;
  const [bankroll, setBankroll] = useStored("pb-bankroll", 0);
  const rows = sizeStakes([...sl.rows].sort((a, b) => rank(b.p, b.ch) - rank(a.p, a.ch)).map(r => {
    const bb = bestBook(data, r.p, r.side, trust), price = bb?.price ?? r.price;
    return { ...r, bb, game: r.p.game, stake: price == null ? 0 : kellyStake(bb?.ch ?? r.ch, price, bankroll) };
  }), bankroll);
  const ql = q.trim().toLowerCase();
  const all = data.props.filter(p => p.ladder && (!pos || p.pos === pos) && (!stat || p.stat === stat) && (!ql || p.player.toLowerCase().includes(ql))).map(p => {
    const po = chance(p, trust), side: Side = po >= 0.5 ? "Over" : "Under", ch = Math.max(po, 1 - po);
    return { p, side, ch, meets: ch >= T };
  }).filter(r => !onlyMeets || r.meets).sort((a, b) => Number(b.meets) - Number(a.meets) || rank(b.p, b.ch) - rank(a.p, a.ch));
  return (
    <section>
      <div className="controls" style={{ marginBottom: 10 }}>
        <label htmlFor="f-min">Minimum chance <select id="f-min" value={minPct} onChange={e => setMinPct(Number(e.target.value))}>{[55, 60, 65, 70, 75, 80].map(v => <option key={v} value={v}>{v}</option>)}</select>%</label>
        <label htmlFor="f-by">Sort by <select id="f-by" value={by} onChange={e => setBy(e.target.value)}><option value="chance">Our chance</option><option value="conf">Confidence</option></select></label>
        <label htmlFor="f-bank">Bankroll ($) <input id="f-bank" className="stake" type="number" min={0} step={10} defaultValue={bankroll || ""} placeholder="for stakes"
          onBlur={e => setBankroll(Math.max(0, Number(e.target.value) || 0))} onKeyDown={e => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} /></label>
        <label htmlFor="f-meets"><input id="f-meets" type="checkbox" checked={onlyMeets} onChange={e => setOnlyMeets(e.target.checked)} /> Only posted lines that meet it</label>
      </div>
      <h2>This week's shortlist</h2>
      <p className="note">Posted lines at {pct(T)} or more, no likely blowouts (spread {BLOWOUT}+), only where our chance beats the break-even of the book's price, and never the under for Gibbs, Bijan Robinson, Puka Nacua or Jaxon Smith-Njigba. These are the single bets; parlays pair them across games, leaving out yardage lines under {SHORT_MIN_LINE} and QB rushing. <b>Cushion</b> is our chance minus the break-even of the book's price; take a sportsbook bet only with {Math.round(100 * CUSHION)}+ points. For Sleeper, run these players through the Pick'em tab with its lines. Check injury news before kickoff: the model doesn't know about late inactives. <b>Best book</b> is the book whose line and price give the most value at our chance, moved along our range when its line differs from the posted one. Shopping pays: on the main rule's 248 settled bets in weeks 6-7, the best book's price on the same line beat the median price on 207 of them, lifting ROI from +9.3% to +11.0%. Enter a <b>bankroll</b> (money set aside for betting) to get stakes: a quarter of the Kelly bet at the best book, at most {pct(BET_CAP)} per bet and {pct(GAME_CAP)} per game, since one game's bets win or lose together.
        {played && <> <b>This week's games have all started, so this is for review: the next shortlist appears after Thursday's run.</b></>}</p>
      <div className="panel scroll" style={{ margin: "10px 0 18px" }}>
        <table>
          <thead><tr><th>Player</th><th>Bet</th><th className="r">Our chance</th><th className="r">Book price</th><th className="r">Break-even</th><th className="r">Cushion</th><th>Best book</th>{bankroll > 0 && <th className="r">Stake</th>}<th>Kickoff</th></tr></thead>
          <tbody>{rows.length ? rows.map(({ p, side, ch, price, bb, stake }) => {
            const be = implied(price), cush = be == null ? null : ch - be;
            return <tr key={p.id}><td><div className="player"><button type="button" className="plink" onClick={() => openPlayer(p.id.split(":")[0])}>{p.player}</button> <span className="sub">{p.pos}</span></div><div className="sub">{p.game}</div></td>
              <td><Pill kind={side.toLowerCase()}>{side}</Pill> <span className="num">{p.line}</span> <span className="sub">{statLabel(p.stat)} · proj {p.proj}</span></td>
              <td className="r"><Conf p={ch} id={p.id} side={side} /></td><td className="r num">{fmtOdds(price)}</td><td className="r num">{be == null ? "-" : pct(be)}</td>
              <td className={`r num ${cush != null && cush >= CUSHION ? "edge-pos" : "edge-neg"}`}>{cush == null ? "-" : `${cush > 0 ? "+" : ""}${(100 * cush).toFixed(0)} pts`}</td>
              <td>{bb ? <><b>{bookName(bb.book)}</b> <span className="num">{bb.line !== p.line ? `${side} ${bb.line} ` : ""}{fmtOdds(bb.price)}</span>
                <div className="sub">{bb.ev > 0 ? "+" : ""}{(100 * bb.ev).toFixed(1)}% value</div></> : <span className="sub">-</span>}</td>
              {bankroll > 0 && <td className="r num">{stake > 0 ? money(stake) : "-"}</td>}
              <td className="sub">{new Date(p.kickoff).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" })}</td></tr>;
          }) : <Empty cols={bankroll > 0 ? 9 : 8}>No posted line reaches {pct(T)} this week. Try a lower minimum.</Empty>}</tbody>
        </table>
      </div>
      <h2>Every prop</h2>
      <p className="note">For each prop: the chance the over or under hits at the book's posted line, and the furthest alt line that still hits {pct(T)} of the time. A {pct(T)} bet breaks even at <b className="num">{fair(T)}</b>, so an alt line only pays if your book's price is better than that. Chances lean 70% on the book's line and 30% on our projection, then are pulled {Math.round(100 * (1 - (data.ladder_k ?? 1)))}% toward 50%, the correction week 5's real lines called for. Receptions lean 90% on our projection with no pull (fit on weeks 2-4, held out on week 5); their lines are DraftKings' via ESPN, with no price.</p>
      <div className="panel scroll" style={{ marginTop: 10 }}>
        <table>
          <thead><tr><th>Player</th><th className="r">Posted line</th><th className="r">Projection</th><th>Chance at posted line</th><th className="r">Over, at this chance</th><th className="r">Under, at this chance</th></tr></thead>
          <tbody>{all.length ? all.map(({ p, side, ch }) => {
            const L = p.ladder as NonNullable<typeof p.ladder>, o = L.over[key], u = L.under[key];
            return <tr key={p.id}><td><div className="player"><button type="button" className="plink" onClick={() => openPlayer(p.id.split(":")[0])}>{p.player}</button> <span className="sub">{p.pos}</span></div><div className="sub">{p.game} · {statLabel(p.stat)}</div></td>
              <td className="r num">{p.line}</td><td className="r num">{p.proj}</td>
              <td><Pill kind={side.toLowerCase()}>{side}</Pill> <Conf p={ch} id={p.id} side={side} />
                <div className="sub">book's price {fmtOdds(side === "Over" ? p.over : p.under)} · break-even {fair(ch)}</div></td>
              <td className="r num">{o == null ? "-" : `Over ${o}`}<div className="sub">{o == null ? "" : o >= p.line ? "posted line qualifies" : `${(p.line - o).toFixed(0)} ${unit(p)} below posted`}</div></td>
              <td className="r num">{u == null ? "-" : `Under ${u}`}<div className="sub">{u == null ? "" : u <= p.line ? "posted line qualifies" : `${(u - p.line).toFixed(0)} ${unit(p)} above posted`}</div></td></tr>;
          }) : <Empty cols={6}>No posted line reaches {pct(T)}. Untick the box to see the alt lines that do.</Empty>}</tbody>
        </table>
      </div>
    </section>
  );
}
