import { useState } from "react";
import { useBoard } from "../context";
import { fmtOdds, pct, statLabel } from "../lib/format";
import { best, chance, edges, ourChance } from "../lib/model";
import type { Prop, Side } from "../types";
import { Conf, Empty, Pill, confRank } from "./common";
import Slip from "./Slip";

type Key = "player" | "line" | "proj" | "gap" | "chance" | "conf" | "edge";

export default function BoardTab() {
  const { data, trust, pos, stat, q, onlyEdge, slip, setSlip, openPlayer } = useBoard();
  const [sort, setSort] = useState<{ key: Key; dir: number }>({ key: "edge", dir: -1 });
  const pick = (p: Prop) => { const c = chance(p, trust); return Math.max(c, 1 - c); };
  const val = (p: Prop) => {
    switch (sort.key) {
      case "player": return p.player; case "line": return p.line; case "proj": return p.proj;
      case "gap": return Math.abs(p.gap); case "chance": return pick(p); case "conf": return confRank(p, pick(p)); default: return best(p, trust).ev ?? -9;
    }
  };
  const ql = q.trim().toLowerCase();
  const rows = data.props.filter(p => (!pos || p.pos === pos) && (!stat || p.stat === stat) && (!ql || p.player.toLowerCase().includes(ql))
    && (!onlyEdge || (best(p, trust).ev ?? -1) > 0))
    .sort((a, b) => { const x = val(a), y = val(b); return (x > y ? 1 : x < y ? -1 : 0) * sort.dir; });
  const th = (key: Key, label: string, r = false) => (
    <th className={r ? "r" : undefined} data-sort={key} onClick={() => setSort(s => ({ key, dir: s.key === key ? -s.dir : key === "player" ? 1 : -1 }))}>{label}</th>);
  const toggle = (id: string, side: Side) => setSlip(s => {
    const i = s.findIndex(x => x.id === id);
    if (i >= 0 && s[i].side === side) return s.filter((_, j) => j !== i);
    if (i >= 0) return s.map((x, j) => (j === i ? { ...x, side } : x));
    return [...s, { id, side }];
  });
  return (
    <section className="layout">
      <div className="panel scroll">
        <table>
          <thead><tr>{th("player", "Player")}{th("line", "Line", true)}{th("proj", "Projection", true)}{th("gap", "Gap", true)}
            {th("chance", "Our pick")}{th("conf", "Confidence")}{th("edge", "Best edge", true)}<th>Bet slip</th></tr></thead>
          <tbody>{rows.length ? rows.map(p => {
            const e = edges(p, trust), b = best(p, trust), inSlip = slip.find(s => s.id === p.id);
            const src = ([["own", "ours"], ["sleeper", "Sleeper"], ["gbm", "GBM"]] as const).map(([k, n]) => `${n} ${p.sources[k] ?? "-"}`).join(" · ");
            return <tr key={p.id}>
              <td><div className="player"><button type="button" className="plink" onClick={() => openPlayer(p.id.split(":")[0])}>{p.player}</button> <span className="sub">{p.pos}</span></div><div className="sub">{p.game} · {statLabel(p.stat)}</div></td>
              <td className="r num">{p.line}<div className="sub">{p.books} books</div></td>
              <td className="r num">{p.proj}<div className="sub">{src}</div></td>
              <td className="r num">{p.gap > 0 ? "+" : ""}{p.gap}</td>
              <td><Pill kind={e.q >= 0.5 ? "over" : "under"}>{e.q >= 0.5 ? "Over" : "Under"}</Pill> <span className="num">{pct(Math.max(e.q, 1 - e.q))}</span>
                <div className="sub">over: model {pct(ourChance(p))} · market {p.p_market == null ? "-" : pct(p.p_market)}</div></td>
              <td><Conf p={Math.max(e.q, 1 - e.q)} id={p.id} bare /></td>
              <td className="r"><span className={`num ${(b.ev ?? -1) > 0 ? "edge-pos" : "edge-neg"}`}>{b.ev == null ? "-" : `${b.ev > 0 ? "+" : ""}${(100 * b.ev).toFixed(1)}%`}</span><div className="sub">{b.side}</div></td>
              <td><span className="side-btns">{(["Over", "Under"] as const).map(side =>
                <button key={side} type="button" aria-label={`${side} ${p.line}`} aria-pressed={inSlip?.side === side} disabled={(side === "Over" ? p.over : p.under) == null} onClick={() => toggle(p.id, side)}><span>{side[0]} {p.line}</span><b>{fmtOdds(side === "Over" ? p.over : p.under)}</b></button>)}</span></td>
            </tr>;
          }) : <Empty cols={8}>No props match these filters.</Empty>}</tbody>
        </table>
      </div>
      <Slip />
    </section>
  );
}
