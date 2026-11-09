import { useEffect, useMemo, useState } from "react";
import type { BoardData } from "./types";
import { BoardCtx, type Board } from "./context";
import { fmtTime, money, pct, statLabel } from "./lib/format";
import { store, useStored } from "./lib/store";
import { useBoardDb } from "./lib/db";
import { myRecord } from "./lib/picks";
import { findProp, type SlipLeg } from "./lib/model";
import Dashboard from "./components/Dashboard";
import BoardTab from "./components/BoardTab";
import Games from "./components/Games";
import Hits from "./components/Hits";
import Live from "./components/Live";
import Parlays from "./components/Parlays";
import Pickem from "./components/Pickem";
import PlayerPanel from "./components/Player";

const TABS = [["dash", "Dashboard"], ["board", "Lines & projections"], ["games", "Game lines"], ["hits", "Hit rates"],
  ["live", "Live"], ["parlays", "Parlays"], ["pickem", "Pick'em"]] as const;
type Tab = (typeof TABS)[number][0];
const isTab = (t: string): t is Tab => TABS.some(([k]) => k === t);

export default function App() {
  const [data, setData] = useState<BoardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTabState] = useState<Tab>(() => { const h = location.hash.slice(1), s = store.get("pb-tab", "dash"); return isTab(h) ? h : isTab(s) ? s : "dash"; });
  const setTab = (t: Tab) => { setTabState(t); store.set("pb-tab", t); };
  const [pos, setPos] = useState(""), [stat, setStat] = useState(""), [q, setQ] = useState("");
  const [trustPct, setTrustPct] = useStored("pb-trust2", 100), [onlyEdge, setOnlyEdge] = useState(false);
  const [minPct, setMinPct] = useStored("pb-min", 55), [onlyMeets, setOnlyMeets] = useState(false);
  const [slip, setSlip] = useStored<SlipLeg[]>("pb-slip", []), [stake, setStake] = useStored("pb-stake", 10);
  const dbx = useBoardDb();
  const [playerId, setPlayerId] = useState<string | null>(null);

  useEffect(() => {
    fetch("data.json").then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((d: BoardData) => { setData(d); setSlip(s => s.filter(x => findProp(d, x.id))); })
      .catch((e: Error) => setError(e.message));
  }, [setSlip]);

  const board: Board | null = useMemo(() => data && {
    data, trust: trustPct / 100, pos, stat, q, onlyEdge, minChance: minPct / 100, onlyMeets, slip, setSlip, stake, setStake, dbx, openPlayer: setPlayerId,
  }, [data, trustPct, pos, stat, q, onlyEdge, minPct, onlyMeets, slip, setSlip, stake, setStake, dbx]);

  return (
    <div className="wrap">
      <header className="bug">
        <div className="bug-wk"><span>Week</span><b>{data?.week ?? "-"}</b></div>
        <div className="bug-title">
          <h1>Prop Board{data && <small>{data.season}</small>}</h1>
          <div className="meta">
            {error ? <span>The board data did not load ({error}). Republish the page with its data.json.</span>
              : !data ? <span>Loading the board…</span>
              : <><span>{data.props.length} props</span><span>Lines pulled {fmtTime(data.lines_pulled_at)}</span><span>Built {fmtTime(data.generated_at)}</span></>}
          </div>
        </div>
        {data && (() => { const m = myRecord(data, dbx.parlays, dbx.entries); return (
          <dl className="bug-score" aria-label="Your saved picks">
            <div><dt>Your record</dt><dd>{m.settled ? `${m.won}-${m.settled - m.won}` : "-"}</dd></div>
            <div><dt>Your profit</dt><dd className={m.profit >= 0 ? "up" : "down"}>{m.settled ? money(m.profit) : "-"}</dd></div>
            <div><dt>Open</dt><dd>{m.open}</dd></div>
          </dl>); })()}
      </header>
      <p className="note">Paper trading only. Projections blend our model, Sleeper and a gradient-boosted model. "Our chance" mixes the model's probability with the market's (margin removed) by the trust setting; the model has not yet been tested against real lines.</p>

      <div className="tabs" role="tablist">
        {TABS.map(([k, label]) => <button key={k} type="button" role="tab" id={`tab-${k}`} aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>)}
      </div>

      {board && <BoardCtx.Provider value={board}>
        {["board", "parlays", "hits"].includes(tab) && (
          <div className="controls">
            <label htmlFor="f-pos">Position <select id="f-pos" value={pos} onChange={e => setPos(e.target.value)}><option value="">All</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label>
            <label htmlFor="f-stat">Stat <select id="f-stat" value={stat} onChange={e => setStat(e.target.value)}><option value="">All</option>
              {([["passing_yards", "Passing yds"], ["rushing_yards", "Rushing yds"], ["receiving_yards", "Receiving yds"], ["receptions", "Receptions"]] as const).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
            <label htmlFor="f-q">Player <input id="f-q" type="search" placeholder="Search" value={q} onChange={e => setQ(e.target.value)} /></label>
            <label htmlFor="f-trust">Trust in our model <input id="f-trust" type="range" min={0} max={100} step={5} value={trustPct} onChange={e => setTrustPct(Number(e.target.value))} /> <span className="num">{trustPct}%</span> <span className="sub" style={{ textTransform: "none", letterSpacing: 0 }}>100% = the paper rule, 0% = the market</span></label>
            <label htmlFor="f-edge"><input id="f-edge" type="checkbox" checked={onlyEdge} onChange={e => setOnlyEdge(e.target.checked)} /> Only positive edge</label>
          </div>)}
        {tab === "dash" && <Dashboard />}
        {tab === "board" && <BoardTab />}
        {tab === "games" && <Games />}
        {tab === "hits" && <Hits setMinPct={setMinPct} minPct={minPct} onlyMeets={onlyMeets} setOnlyMeets={setOnlyMeets} />}
        {tab === "live" && <Live />}
        {tab === "parlays" && <Parlays />}
        {tab === "pickem" && <Pickem />}
        {playerId && <PlayerPanel id={playerId} onClose={() => setPlayerId(null)} />}

        <details className="panel" style={{ padding: "12px 14px" }}>
          <summary>How reliable is "our chance"?</summary>
          <p className="note" style={{ marginTop: 8 }}>Checked on 2026 weeks 2-4, which the model never saw: when the model gave a player a given chance of beating his last-5-game average, how often he did. Close numbers mean the chances are honest. The high buckets run hot, which is why the market gets most of the weight by default.</p>
          <div className="scroll"><table>
            <thead><tr><th>Prop</th><th>Model chance</th><th className="r">Cases</th><th className="r">Model said</th><th className="r">Happened</th></tr></thead>
            <tbody>{board.data.calibration.map((c, i) => <tr key={i}><td>{c.pos} {statLabel(c.stat)}</td><td className="num">{c.bucket}</td><td className="r num">{c.n}</td><td className="r num">{pct(c.predicted)}</td><td className="r num">{pct(c.observed)}</td></tr>)}</tbody>
          </table></div>
        </details>
      </BoardCtx.Provider>}
      <footer className="note">Paper trading for fun, not betting advice: the model is unproven and its chances can be wrong. If you bet real money, only bet what you can afford to lose, and only where it is legal for your age (21+ in most US states). Gambling problem? Call or text 1-800-GAMBLER.</footer>
    </div>
  );
}
