import { useRef, useState, type ReactNode, type PointerEvent } from "react";
import { money, pct } from "../lib/format";

// Inline SVG charts: one y-axis each, hover tooltips, a table view under every chart.
export const W = 640;
const M = { t: 14, r: 92, b: 26, l: 44 };

export function niceTicks(lo: number, hi: number, n = 4) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= step0) as number;
  const out: number[] = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

export interface Hit { x: number; y: number; w: number; h: number; content: ReactNode; cx?: number }
/** An SVG with a hover tooltip: hits are rectangles in viewBox units. */
export function TipSvg({ width, height, label, hits, children, onHit }: {
  width: number; height: number; label: string; hits: Hit[]; children: ReactNode; onHit?: (h: Hit | null) => void;
}) {
  const box = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null), tip = useRef<HTMLDivElement>(null);
  const [cur, setCur] = useState<{ hit: Hit; left: number; top: number } | null>(null);
  const move = (ev: PointerEvent) => {
    const s = svg.current, b = box.current; if (!s || !b) return;
    const r = s.getBoundingClientRect(), sx = ((ev.clientX - r.left) * width) / r.width, sy = ((ev.clientY - r.top) * height) / r.height;
    const h = hits.find(h => sx >= h.x && sx <= h.x + h.w && sy >= h.y && sy <= h.y + h.h) ?? null;
    onHit?.(h);
    if (!h) { setCur(null); return; }
    const br = b.getBoundingClientRect(), tw = tip.current?.offsetWidth ?? 160;
    setCur({ hit: h, left: Math.min(ev.clientX - br.left + 12, br.width - tw - 4), top: ev.clientY - br.top + 12 });
  };
  return (
    <div ref={box} style={{ position: "relative" }}>
      <svg ref={svg} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}
        onPointerMove={move} onPointerLeave={() => { setCur(null); onHit?.(null); }}>{children}</svg>
      <div ref={tip} className="tip" hidden={!cur} style={cur ? { left: cur.left, top: cur.top } : undefined}>{cur?.hit.content}</div>
    </div>
  );
}

export function TableView({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <details><summary>Show as table</summary><div className="scroll"><table>
      <thead><tr>{head.map((h, i) => <th key={i} className={i ? "r" : undefined}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className={j ? "r num" : undefined}>{c}</td>)}</tr>)}</tbody>
    </table></div></details>
  );
}

export interface Series { label: string; color: string; values: (number | null)[]; dashed?: boolean }
export function LineChart({ labels, series, fmt, H = 230 }: { labels: string[]; series: Series[]; fmt: (v: number) => string; H?: number }) {
  const [crossX, setCrossX] = useState<number | null>(null);
  const vals = series.flatMap(s => s.values).filter((v): v is number => v != null);
  // pad by the range (or the value itself when there's one point): scaling by 0.97/1.03 inverts for negatives
  const vlo = Math.min(...vals), vhi = Math.max(...vals), pad = Math.max((vhi - vlo) * 0.05, Math.abs(vhi) * 0.05, 1e-6);
  const ticks = niceTicks(vlo - pad, vhi + pad);
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x = (i: number) => M.l + (labels.length === 1 ? (W - M.l - M.r) / 2 : (i * (W - M.l - M.r)) / (labels.length - 1));
  const y = (v: number) => H - M.b - ((v - lo) / (hi - lo || 1)) * (H - M.t - M.b);
  const ends = series.map(z => {
    const pts = z.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter((p): p is number[] => p != null);
    return { z, pts, y: pts.length ? pts[pts.length - 1][1] : 0, x: pts.length ? pts[pts.length - 1][0] : 0 };
  });
  const labelled = ends.filter(e => e.pts.length).sort((a, b) => a.y - b.y);
  labelled.forEach((e, i, a) => { if (i && e.y - a[i - 1].y < 13) e.y = a[i - 1].y + 13; });   // keep end labels apart
  const step = labels.length > 1 ? (W - M.l - M.r) / (labels.length - 1) : W;
  const hits: Hit[] = labels.map((l, i) => ({ x: x(i) - step / 2, y: 0, w: step, h: H, cx: x(i),
    content: <><b>{l}</b>{series.map(z => <div key={z.label}>{z.label}: {z.values[i] == null ? "-" : fmt(z.values[i] as number)}</div>)}</> }));
  return (
    <>
      <div className="legend">{series.map(z => <span key={z.label} className={z.dashed ? "dash-key" : ""} style={{ ["--c" as string]: `var(${z.color})` }}>{z.label}</span>)}</div>
      <TipSvg width={W} height={H} label={`${series.map(z => z.label).join(", ")} by week`} hits={hits} onHit={h => setCrossX(h?.cx ?? null)}>
        {ticks.map(t => <g key={t}><line className="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} /><text x={M.l - 6} y={y(t) + 4} textAnchor="end">{fmt(t)}</text></g>)}
        {labels.map((l, i) => <text key={i} x={x(i)} y={H - 6} textAnchor="middle">{l}</text>)}
        <line x1={crossX ?? 0} x2={crossX ?? 0} y1={M.t} y2={H - M.b} stroke="var(--muted)" strokeDasharray="3 3" visibility={crossX == null ? "hidden" : "visible"} />
        {ends.map(({ z, pts }) => (
          <g key={z.label}>
            <polyline fill="none" stroke={`var(${z.color})`} strokeWidth={2} strokeLinejoin="round" strokeDasharray={z.dashed ? "5 4" : undefined} points={pts.map(p => p.join(",")).join(" ")} />
            {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={4} fill={`var(${z.color})`} stroke="var(--surface)" strokeWidth={2} />)}
          </g>))}
        {labelled.map(e => <text key={e.z.label} className="lbl" x={e.x + 8} y={e.y + 4}>{e.z.label}</text>)}
      </TipSvg>
      <TableView head={["Week", ...series.map(z => z.label)]} rows={labels.map((l, i) => [l, ...series.map(z => (z.values[i] == null ? "-" : fmt(z.values[i] as number)))])} />
    </>
  );
}

export interface BarRow { label: string; value: number; staked: number; record: string; detail: string }
export function BarChart({ rows }: { rows: BarRow[] }) {
  const vals = rows.map(r => r.value), ticks = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals));
  const lo = ticks[0], hi = ticks[ticks.length - 1], H2 = 200, m = { t: 12, r: 16, b: 26, l: 52 };
  const y = (v: number) => H2 - m.b - ((v - lo) / (hi - lo || 1)) * (H2 - m.t - m.b);
  const bw = Math.min(56, (W - m.l - m.r) / rows.length - 8), gap = (W - m.l - m.r) / rows.length;
  const hits: Hit[] = rows.map((r, i) => ({ x: m.l + gap * i, y: 0, w: gap, h: H2, content: <><b>{r.label} bets</b><div>{r.detail}</div></> }));
  return (
    <>
      <TipSvg width={W} height={H2} label="Profit by stake size" hits={hits}>
        {ticks.map(t => <g key={t}><line className="grid" x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} /><text x={m.l - 6} y={y(t) + 4} textAnchor="end">{money(t)}</text></g>)}
        {rows.map((r, i) => {
          const cx = m.l + gap * i + gap / 2, top = Math.min(y(r.value), y(0)), h = Math.max(1, Math.abs(y(r.value) - y(0)));
          return <g key={r.label}>
            <rect x={cx - bw / 2} y={top} width={bw} height={h} rx={4} fill={`var(${r.value >= 0 ? "--win" : "--loss"})`} />
            <text x={cx} y={H2 - 8} textAnchor="middle">{r.label}</text>
            <text className="lbl" x={cx} y={r.value >= 0 ? top - 5 : top + h + 13} textAnchor="middle">{money(r.value)}</text>
          </g>;
        })}
        <line className="zero" x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} />
      </TipSvg>
      <TableView head={["Stake", "Record", "Staked", "Profit"]} rows={rows.map(r => [r.label, r.record, money(r.staked), money(r.value)])} />
    </>
  );
}

export interface Dot { x: number; y: number; n: number; label: string; color: string }
export function ScatterChart({ pts, legend }: { pts: Dot[]; legend: [string, string][] }) {
  const H3 = 260, m = { t: 12, r: 16, b: 34, l: 44 }, side = Math.min(W - m.l - m.r, H3 - m.t - m.b);
  const x = (v: number) => m.l + v * side, y = (v: number) => H3 - m.b - v * side;
  const hits: Hit[] = pts.map(p => ({ x: x(p.x) - 10, y: y(p.y) - 10, w: 20, h: 20,
    content: <><b>{p.label}</b><div>model said {pct(p.x)}, happened {pct(p.y)}</div><div>{p.n} props</div></> }));
  const lx = x(1) + 24;
  return (
    <>
      <TipSvg width={W} height={H3} label="Model chance against observed rate" hits={hits}>
        {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}>
          <line className="grid" x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} /><text x={m.l - 6} y={y(t) + 4} textAnchor="end">{pct(t)}</text>
          <text x={x(t)} y={H3 - 18} textAnchor="middle">{pct(t)}</text></g>)}
        <text x={x(0.5)} y={H3 - 2} textAnchor="middle">model said</text>
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--muted)" strokeDasharray="4 4" />
        {pts.map((p, i) => <circle key={i} cx={x(p.x)} cy={y(p.y)} r={Math.max(4, Math.min(10, Math.sqrt(p.n)))} fill={`var(${p.color})`} fillOpacity={0.85} stroke="var(--surface)" strokeWidth={2} />)}
        {legend.map(([k, c], i) => <g key={k}><circle cx={lx} cy={m.t + 8 + i * 18} r={5} fill={`var(${c})`} /><text className="lbl" x={lx + 10} y={m.t + 12 + i * 18}>{k}</text></g>)}
      </TipSvg>
      <TableView head={["Group", "Model said", "Happened", "Props"]} rows={pts.map(p => [p.label, pct(p.x), pct(p.y), p.n])} />
    </>
  );
}
