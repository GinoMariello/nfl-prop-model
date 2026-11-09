import { useBoard } from "../context";
import { money, pct, statLabel } from "../lib/format";
import { useStored } from "../lib/store";
import { LineChart, ScatterChart } from "./Charts";
import { Tiles } from "./common";
import { myRecord } from "../lib/picks";
import type { LedgerSummary } from "../types";

const SRC: [string, string, string][] = [["pred", "Blend", "--s1"], ["own", "Ours", "--s2"], ["sleeper", "Sleeper", "--s3"], ["gbm", "GBM", "--s4"], ["l5", "Last-5 avg", "--muted"]];
const POS_COLOR: Record<string, string> = { QB: "--s1", RB: "--s2", WR: "--s3", TE: "--s4" };

const SIDE_TESTS = [
  ["ledger_oppf", "Skip unders vs weak defenses", "unders where the defense allows 8%+ more than average"],
  ["ledger_norush", "Skip rushing", "every rushing-yards bet"],
  ["ledger_norush_avoid", "Skip rushing + Avoid", "rushing bets and any bet our confidence rates Avoid"],
] as const;

const recordOf = (s: LedgerSummary) => (s.settled ? `${s.wins}-${s.losses}` : "-");
const roiOf = (s: LedgerSummary) => (s.staked ? `${s.profit >= 0 ? "+" : ""}${(100 * s.profit / s.staked).toFixed(1)}%` : "-");

/** The paper rule against its side tests (site_data.side_ledger), each over the weeks since it started. Nothing is bet separately:
 * a side test is the rule's own bets minus the ones it would skip. */
function SideTests() {
  const { data } = useBoard();
  const tests = SIDE_TESTS.map(([k, name, drops]) => ({ name, drops, t: data[k] })).filter(x => x.t?.main);
  if (!tests.length) return null;
  return (
    <div className="panel chart wide">
      <div className="chart-head"><h2>Rule experiments</h2><span className="sub">the paper rule vs versions that skip some of its bets</span></div>
      <p className="note">Each test is judged only from the week it started (earlier weeks were used to pick it). Both sides of a row cover the same weeks; nothing is bet separately.</p>
      <div className="scroll"><table>
        <thead><tr><th>Test</th><th>Skips</th><th className="r">From</th><th className="r">Test record</th><th className="r">Test profit</th><th className="r">Test ROI</th>
          <th className="r">Rule record</th><th className="r">Rule profit</th><th className="r">Rule ROI</th></tr></thead>
        <tbody>{tests.map(({ name, drops, t }) => {
          const s = t!, m = s.main!;
          return <tr key={name}><td>{name}</td><td className="sub">{drops}</td><td className="r num">W{s.from_week}</td>
            <td className="r num">{recordOf(s)}</td><td className="r num">{s.settled ? money(s.profit) : "-"}</td><td className="r num">{roiOf(s)}</td>
            <td className="r num">{recordOf(m)}</td><td className="r num">{m.settled ? money(m.profit) : "-"}</td><td className="r num">{roiOf(m)}</td></tr>;
        })}</tbody>
      </table></div>
    </div>
  );
}

export default function Dashboard() {
  const { data, dbx } = useBoard();
  const [accPos, setAccPos] = useStored("pb-accpos", "WR");
  const acc = (data.accuracy || []).filter(a => a.pos === accPos).sort((a, b) => a.week - b.week);
  // the viewer's own saved parlays and pick'em entries, settled from the box scores
  const { saved, settled, won, profit, open, hits, misses } = myRecord(data, dbx.parlays, dbx.entries);
  return (
    <section className="dash">
      <div className="chart-head"><h2>How your picks are doing</h2><span className="sub">parlays and pick'em entries you saved{dbx.where ? ", in this browser" : ""}</span></div>
      {saved ? <div className="tiles"><Tiles items={[
        [settled ? `${won}-${settled - won}` : "-", settled ? `Your record · ${settled} settled` : "Your record · none settled yet"],
        [settled ? money(profit) : "-", "Your profit"],
        [hits + misses ? pct(hits / (hits + misses)) : "-", hits + misses ? `Picks that hit · ${hits} of ${hits + misses}` : "Picks that hit"],
        [String(open), `Still open · ${dbx.parlays.length} parlays, ${dbx.entries.length} entries saved`],
      ]} /></div>
        : <div className="panel empty">Nothing saved yet. Save a parlay from the bet slip or an entry on the Pick'em tab and it is tracked here, settled from the box scores.</div>}
      <div className="dash-grid">
        <div className="panel chart wide">
          <div className="chart-head"><h2>Projection miss by week</h2>
            <span className="seg" role="group" aria-label="Position">{["QB", "RB", "WR", "TE"].map(p =>
              <button key={p} type="button" aria-pressed={p === accPos} onClick={() => setAccPos(p)}>{p}</button>)}</span>
          </div>
          <p className="note">Average PPR points missed per player, lower is better. Each week is scored on the players every source projected.</p>
          {acc.length ? <LineChart labels={acc.map(a => `W${a.week} (${a.n})`)} fmt={v => v.toFixed(2)}
            series={SRC.map(([k, label, color]) => ({ label, color, dashed: k === "l5", values: acc.map(a => a.miss[k] ?? null) }))} />
            : <div className="empty">No scored {accPos} weeks yet.</div>}
        </div>
        <SideTests />
        <div className="panel chart wide">
          <div className="chart-head"><h2>Is "our chance" honest?</h2><span className="sub">2026 weeks 2-4, never used to build it</span></div>
          <p className="note">Each dot is a group of props: the chance the model gave a player of beating his last-5 average, against how often he did. Dots on the dashed line are honest; dots below it mean the model was too confident.</p>
          <ScatterChart legend={Object.entries(POS_COLOR)} pts={(data.calibration || []).map(c => ({ x: c.predicted, y: c.observed, n: c.n, color: POS_COLOR[c.pos],
            label: `${c.pos} ${statLabel(c.stat)}, model ${c.bucket.replace("-", "–")}` }))} />
        </div>
      </div>
    </section>
  );
}
