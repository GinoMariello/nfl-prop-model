import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useBoard } from "../context";
import { LegLabel, Pill, PlayerLink } from "./common";
import { chance, gameProps } from "../lib/model";
import { dec, pct, short, statLabel } from "../lib/format";
import { allPlays, boxStats, espn, feedTime, gamePlays, viaHost, livePOver, normName, shortKey, timeLeft, type Espn, type PlayLine, type Status } from "../lib/live";
import { useStored } from "../lib/store";
import type { Prop } from "../types";
import { TipSvg, type Hit } from "./Charts";

interface LiveGame { status?: Status; teams?: { away: Espn; home: Espn }; stats?: ReturnType<typeof boxStats>; plays?: PlayLine[]; drives?: Espn[]; wp?: Espn[] }
type Odds = Record<string, { line?: number | null; over?: number | null; under?: number | null }>;

const small = () => matchMedia("(max-width: 600px)").matches;   // bigger text on phones needs taller rows
const FX = (yl: number) => 40 + yl * 5.6;                        // 100 yards across x 40..600; end zones outside
const ydText = (yl: number) => { const v = Math.round(Math.min(yl, 100 - yl)); return v === 50 ? "50" : String(v); };
const teamVar = (g: LiveGame, id: unknown) => (String(id) === String(g.teams?.home.team.id) ? "--s2" : "--s1");

/** Field background, yard lines, end zones labelled with the team defending each. ESPN yardLine: 0 home goal .. 100 away goal. */
function FieldBase({ g, h, top }: { g: LiveGame; h: number; top: number }) {
  const home = g.teams?.home.team.abbreviation, away = g.teams?.away.team.abbreviation, ty = top + (small() ? 18 : 12);
  return <>
    <rect x={0} y={top} width={640} height={h - top} rx={4} fill="var(--chip)" />
    <rect x={0} y={top} width={40} height={h - top} rx={4} fill="var(--s2)" fillOpacity={0.22} />
    <rect x={600} y={top} width={40} height={h - top} rx={4} fill="var(--s1)" fillOpacity={0.22} />
    <text className="lbl" x={20} y={ty} textAnchor="middle">{home}</text><text className="lbl" x={620} y={ty} textAnchor="middle">{away}</text>
    {Array.from({ length: 9 }, (_, i) => (i + 1) * 10).map(yl => <g key={yl}><line x1={FX(yl)} x2={FX(yl)} y1={top} y2={h} stroke="var(--rule)" /><text x={FX(yl)} y={top - 3} textAnchor="middle">{ydText(yl)}</text></g>)}
  </>;
}

function ActivePlay({ g, step, onStep }: { g: LiveGame; step: number | undefined; onStep: (s: number | undefined) => void }) {
  const all = gamePlays(g.drives || []); if (!all.length) return null;
  const i = Math.min(step ?? all.length - 1, all.length - 1), { p, d } = all[i], latest = i === all.length - 1;
  const top = small() ? 22 : 14, h = top + (small() ? 44 : 34), y = top + (h - top) / 2, col = teamVar(g, d.team?.id);
  const dir = String(d.team?.id) === String(g.teams?.home.team.id) ? 1 : -1;
  const x1 = FX(p.start.yardLine), x2 = FX(p.end.yardLine), gain = p.statYardage ?? 0;
  const kick = /punt|kickoff|field goal|extra point/i.test(p.type?.text || "");   // statYardage on kicks is the return/penalty
  const right = Math.max(x1, x2), lx = right > 520 ? Math.min(x1, x2) - 12 : right + 14;
  const go = (s: number) => { const n = all.length, next = s === 0 ? n - 1 : Math.max(0, Math.min(n - 1, i + s)); onStep(next >= n - 1 ? undefined : next); };
  return <div className="viz">
    <h3 style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
      <span>{latest ? "Latest play" : `Play ${i + 1} of ${all.length}`} · {d.team?.abbreviation || ""} · Q{p.period?.number ?? "-"} {p.clock?.displayValue ?? ""} · {p.start.downDistanceText || ""}</span>
      <span className="side-btns">
        <button type="button" aria-label="Previous play" disabled={i === 0} onClick={() => go(-1)}>◀</button>
        <button type="button" aria-label="Next play" disabled={latest} onClick={() => go(1)}>▶</button>
        {!latest && <button type="button" onClick={() => go(0)}>Latest</button>}
      </span>
    </h3>
    <TipSvg width={640} height={h} label={p.text || ""} hits={[{ x: 0, y: 0, w: 640, h, content: p.text || "" }]}>
      <FieldBase g={g} h={h} top={top} />
      {!kick && p.start.distance && p.start.yardsToEndzone > 0 && (() => { const fd = FX(p.start.yardLine + dir * p.start.distance);
        return <line x1={fd} x2={fd} y1={top} y2={h} stroke="var(--s4)" strokeWidth={2} />; })()}
      <line x1={x1} x2={x1} y1={top} y2={h} stroke="var(--muted)" strokeWidth={2} />
      {x1 === x2 ? <circle cx={x1} cy={y} r={5} fill="var(--surface)" stroke={`var(${col})`} strokeWidth={2} />
        : <><line x1={x1} x2={x2} y1={y} y2={y} stroke={`var(${col})`} strokeWidth={3} strokeDasharray={gain < 0 && !kick ? "5 4" : undefined} />
          <ellipse cx={x2} cy={y} rx={8} ry={5} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} /></>}
      <text className="lbl" x={lx} y={y + (small() ? 6 : 3.5)} textAnchor={right > 520 ? "end" : "start"}>{kick ? p.type.text : gain ? `${gain > 0 ? "+" : ""}${gain} yds` : p.type?.text || ""}</text>
    </TipSvg>
    <div className="sub" style={{ fontSize: 12, color: "var(--ink)" }}>{p.text || ""}</div>
  </div>;
}

function DriveStrip({ g }: { g: LiveGame }) {
  const ds = (g.drives || []).filter(d => d.start?.yardLine != null); if (!ds.length) return null;
  const row = small() ? 24 : 13, top = small() ? 22 : 14, h = top + 6 + ds.length * row, hits: Hit[] = [];
  const bars = ds.map((d, i) => {
    const y = top + row * 0.7 + i * row, a = FX(d.start.yardLine), b = FX(d.end?.yardLine ?? d.start.yardLine), col = teamVar(g, d.team?.id);
    const right = Math.max(a, b), lx = right > 560 ? Math.min(a, b) - 6 : right + 6;
    hits.push({ x: 0, y: y - row / 2, w: 640, h: row, content: <><b>{d.team?.abbreviation || ""} · {d.displayResult || "in progress"}</b><div>{d.start?.text || ""} → {d.end?.text || ""} · {d.description || ""}</div></> });
    return <g key={d.id ?? i}><rect x={Math.min(a, b)} y={y - 3} width={Math.max(3, Math.abs(b - a))} height={6} rx={3} fill={`var(${col})`} />
      <text className={d.isScore ? "lbl" : undefined} x={lx} y={y + (small() ? 6 : 3.5)} textAnchor={right > 560 ? "end" : "start"}>{d.shortDisplayResult || d.displayResult || ""}</text></g>;
  });
  return <div className="viz"><h3>Every drive</h3>
    <div className="legend"><span style={{ ["--c" as string]: "var(--s1)" }}>{g.teams?.away.team.abbreviation}</span><span style={{ ["--c" as string]: "var(--s2)" }}>{g.teams?.home.team.abbreviation}</span></div>
    <TipSvg width={640} height={h} label={`All ${ds.length} drives`} hits={hits}><FieldBase g={g} h={h} top={top} />{bars}</TipSvg></div>;
}

function WpChart({ g }: { g: LiveGame }) {
  const text: Record<string, string> = {}, when: Record<string, string> = {};
  (g.drives || []).forEach(d => (d.plays || []).forEach((p: Espn) => { text[p.id] = p.text; when[p.id] = `Q${p.period?.number ?? "-"} ${p.clock?.displayValue ?? ""}`; }));
  const pts = (g.wp || []).filter(w => w.homeWinPercentage != null); if (pts.length < 2) return null;
  const H2 = small() ? 220 : 150, m = { t: 12, b: small() ? 26 : 18, l: small() ? 112 : 64, r: 12 };
  const x = (i: number) => m.l + (i * (640 - m.l - m.r)) / (pts.length - 1), y = (v: number) => m.t + (1 - v) * (H2 - m.t - m.b);
  const home = g.teams?.home.team.abbreviation, away = g.teams?.away.team.abbreviation, step = (640 - m.l - m.r) / (pts.length - 1);
  const qs: ReactNode[] = []; let q: string | null = null;
  pts.forEach((w, i) => { const qq = (when[w.playId] || "").split(" ")[0]; if (qq && qq !== q) {
    if (q) qs.push(<line key={`l${i}`} x1={x(i)} x2={x(i)} y1={m.t} y2={H2 - m.b} stroke="var(--rule)" />);
    qs.push(<text key={`t${i}`} x={x(i) + 3} y={H2 - 6}>{qq}</text>); q = qq; } });
  const last = pts[pts.length - 1];
  const hits: Hit[] = pts.map((w, i) => ({ x: x(i) - step / 2, y: 0, w: step, h: H2, content: <><b>{when[w.playId] || ""} · {home} {pct(w.homeWinPercentage)}</b><div>{text[w.playId] || ""}</div></> }));
  return <div className="viz"><h3>{home} win chance, play by play <span className="sub">(ESPN's model) · now {pct(last.homeWinPercentage)}</span></h3>
    <TipSvg width={640} height={H2} label={`${home} win probability by play`} hits={hits}>
      {([[1, `${home} 100%`], [0.5, "50%"], [0, `${away} 100%`]] as [number, string][]).map(([v, l]) => <g key={v}>
        <line x1={m.l} x2={640 - m.r} y1={y(v)} y2={y(v)} stroke="var(--rule)" strokeDasharray={v === 0.5 ? "4 4" : undefined} /><text x={m.l - 6} y={y(v) + 3.5} textAnchor="end">{l}</text></g>)}
      {qs}
      <polyline fill="none" stroke="var(--s2)" strokeWidth={2} strokeLinejoin="round" points={pts.map((w, i) => `${x(i)},${y(w.homeWinPercentage)}`).join(" ")} />
      <circle cx={x(pts.length - 1)} cy={y(last.homeWinPercentage)} r={4} fill="var(--s2)" stroke="var(--surface)" strokeWidth={2} />
    </TipSvg></div>;
}

/** The viewer's live line + price against the model's live chance. */
function Verdict({ p, left, cur, odds }: { p: Prop; left: number; cur: number; odds: Odds }) {
  const { data } = useBoard();
  const o = odds[p.id] || {}, m = data.live_model?.[`${p.pos}:${p.stat}`];
  if (!m || o.line == null || (o.over == null && o.under == null)) return <span className="sub">enter a live line and price</span>;
  const q = livePOver(m, p.proj, o.line, cur, left);
  const io = o.over == null ? null : 1 / dec(o.over), iu = o.under == null ? null : 1 / dec(o.under);
  const book = io != null && iu != null ? io / (io + iu) : io != null ? io : 1 - (iu as number);
  const evO = o.over == null ? null : q * dec(o.over) - 1, evU = o.under == null ? null : (1 - q) * dec(o.under) - 1;
  const [side, ev] = (evO ?? -9) >= (evU ?? -9) ? ["Over", evO as number] : ["Under", evU as number];
  return <>model {pct(q)} · book {pct(book)} over · <span className={ev > 0 ? "edge-pos" : "edge-neg"}>{side} {ev > 0 ? "+" : ""}{(100 * ev).toFixed(0)}%</span></>;
}

/** Where a game bet stands from the score: won/lost/push when final, winning/losing while live. */
function gameBetState(stat: string, line: number, side: string, away: number, home: number, team: string, final: boolean) {
  const v = stat === "total" ? away + home - line : (team === "home" ? home - away : away - home) + (stat === "spread" ? line : 0);
  const m = stat === "total" && side === "Under" ? -v : v;
  return m === 0 ? (final ? "push" : "tied") : m > 0 ? (final ? "won" : "winning") : (final ? "lost" : "losing");
}

export default function Live() {
  const { data, trust, slip, dbx } = useBoard();
  // the viewer's own picks this week: the slip plus every leg of their saved parlays and pick'em entries
  const mine = new Map<string, string>();
  slip.forEach(s => mine.set(s.id, s.side));
  dbx.parlays.filter(pl => pl.week === data.week && pl.season === data.season).forEach(pl => pl.legs.forEach(l => mine.set(l.id, l.side)));
  dbx.entries.filter(e => e.week === data.week && e.season === data.season).forEach(e => e.legs.forEach(l => mine.set(l.id, l.side === "More" ? "Over" : "Under")));
  const [onlyMine, setOnlyMine] = useStored("pb-live-mine", true);
  const showMine = onlyMine && [...mine.keys()].some(id => data.props.some(p => p.id === id) || id.includes("~"));
  const [games, setGames] = useState<Record<string, LiveGame>>({});
  const [err, setErr] = useState<string | null>(null), [at, setAt] = useState<Date | null>(null);
  const [odds, setOdds] = useStored<Odds>("pb-liveodds", {});
  const [filter, setFilter] = useState<Record<string, string | null>>({}), [step, setStep] = useState<Record<string, number | undefined>>({});
  const [plays, setPlays] = useState<Record<string, boolean>>({}), [viz, setViz] = useState<Record<string, boolean>>({});   // open state the viewer chose
  const busy = useRef(false), gamesRef = useRef(games); gamesRef.current = games;

  const refresh = useCallback(async () => {
    if (busy.current) return; busy.current = true;
    try {
      const wanted = new Set(data.props.map(p => p.game)), next: Record<string, LiveGame> = { ...gamesRef.current };
      const sb = await espn("scoreboard");
      await Promise.all((sb.events || []).map(async (e: Espn) => {
        const c = e.competitions[0], side = (h: string) => c.competitors.find((x: Espn) => x.homeAway === h);
        const key = `${side("away").team.displayName} @ ${side("home").team.displayName}`; if (!wanted.has(key)) return;
        const prev = next[key] || {}, wasFinal = prev.status?.type.state === "post" && prev.stats;
        const g: LiveGame = { ...prev, status: c.status, teams: { away: side("away"), home: side("home") } };
        if (c.status.type.state !== "pre" && !wasFinal) {
          const sum = await espn(`summary?event=${e.id}`);
          g.stats = boxStats(sum); g.plays = allPlays(sum); g.wp = sum.winprobability || [];
          g.drives = [...(sum.drives?.previous || []), sum.drives?.current].filter(Boolean)
            .filter((d: Espn, i: number, a: Espn[]) => a.findIndex(x => x.id === d.id) === i);   // the current drive can also sit in previous
        }
        next[key] = g;
      }));
      setGames(next); setErr(null); setAt(new Date());
    } catch (e) { setErr((e as Error).message); }
    busy.current = false;
  }, [data]);
  useEffect(() => { refresh(); const t = setInterval(refresh, 10000); return () => clearInterval(t); }, [refresh]);   // only while this tab is open

  const byGame: Record<string, Prop[]> = {};
  data.props.forEach(p => (byGame[p.game] ||= []).push(p));
  const myGameBets = gameProps(data).flatMap(p => (mine.has(p.id) ? [{ p, side: mine.get(p.id) as string }] : []));
  if (showMine) for (const k of Object.keys(byGame)) {
    byGame[k] = byGame[k].filter(p => mine.has(p.id));
    if (!byGame[k].length && !myGameBets.some(b => b.p.game === k)) delete byGame[k];
  }
  const order = (k: string) => ({ in: 0, pre: 1, post: 2 }[games[k]?.status?.type.state ?? "pre"] ?? 1);
  const keys = Object.keys(byGame).sort((a, b) => order(a) - order(b) || byGame[a][0].kickoff.localeCompare(byGame[b][0].kickoff));
  const setOdd = (id: string, f: "line" | "over" | "under", raw: string) => {
    const v = raw === "" ? null : Number(raw);
    setOdds(o => ({ ...o, [id]: { ...(o[id] || {}), [f]: Number.isFinite(v as number) ? v : null } }));
  };
  return (
    <section>
      <p className="note">{err ? <span className="warn">Live scores didn't load ({err}). They are posted to the board every 7 minutes while games are on; try again in a few minutes.</span>
        : at ? (feedTime() ? `Scores as of ${(feedTime() as Date).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })} · posted every 7 minutes while games are on; this tab checks for new ones every 10 seconds.`
          : `ESPN data${viaHost() ? " through your Mac" : ""} as of ${at.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })} · refreshes every 10 seconds while this tab is open.`) : "Loading live games…"}</p>
      <div className="controls"><span className="seg" role="group" aria-label="Which props">
        <button type="button" aria-pressed={showMine} disabled={!mine.size} onClick={() => setOnlyMine(true)}>My picks ({mine.size})</button>
        <button type="button" aria-pressed={!showMine} onClick={() => setOnlyMine(false)}>All props</button></span>
        <span className="sub">{mine.size ? "Your picks are what's in your bet slip and your saved parlays and pick'em entries this week." : "Add picks to your bet slip or save a parlay to follow just those here."}</span></div>
      <p className="note">Live chance of the over: yards so far, plus the projection's share of the time left, with last season's projection misses scaled to that time. It doesn't see the score, injuries or how a player is being used today. Type a live line and price from your book to compare; they stay in this browser only.</p>
      {keys.length ? keys.map(key => {
        const g = games[key] || {}, st = g.status, state = st?.type.state || "pre", left = timeLeft(st);
        const ab = (h: "away" | "home") => g.teams ? g.teams[h].team.abbreviation : short(key.split(" @ ")[h === "away" ? 0 : 1]);
        const score = (h: "away" | "home") => g.teams && state !== "pre" ? <> <span className="num">{g.teams[h].score}</span></> : null;
        const when = state === "pre" ? new Date(byGame[key][0].kickoff).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" })
          : state === "in" ? <><span className="pill live">LIVE</span> {st?.type.shortDetail}</> : st?.type.shortDetail;
        const f = filter[key] ?? null, tags = f ? [f.toLowerCase(), `${f[0]}.${f.split(".").pop()?.trim()}`.toLowerCase()] : null;
        const shown = (g.plays || []).filter(x => !tags || tags.some(t => x.text.toLowerCase().includes(t))).slice(0, 40);
        const names = [...new Set(byGame[key].map(p => p.player))].sort();
        const playsOpen = plays[key] ?? state === "in", vizOpen = viz[key] ?? state === "in";
        return <div key={key} className="panel game" style={{ marginBottom: 12 }}>
          <div className="game-head"><b>{ab("away")}{score("away")} @ {ab("home")}{score("home")}</b><span className="sub">{when}</span></div>
          {(() => { const bets = myGameBets.filter(b => b.p.game === key); if (!bets.length) return null;
            const a = Number(g.teams?.away.score ?? 0), h = Number(g.teams?.home.score ?? 0);
            return <div className="chips">{bets.map(({ p, side }) => { const team = p.id.split("~")[1].split(":")[0];
              const st = state === "pre" ? "not started" : gameBetState(p.stat, p.line, side, a, h, team, state === "post");
              return <span key={p.id} className={`pill ${st === "won" || st === "winning" ? "win" : st === "lost" || st === "losing" ? "loss" : ""}`}>
                <LegLabel l={{ ...p, side }} /> · {st}</span>; })}</div>; })()}
          {state !== "pre" && g.drives?.length && g.teams ? <ActivePlay g={g} step={step[key]} onStep={s => setStep(x => ({ ...x, [key]: s }))} /> : null}
          <div className="scroll"><table className="lines">
            <thead><tr><th>Player</th><th className="r">Line</th><th className="r">Now</th><th className="r">{state === "post" ? "Final" : "Exp. final"}</th><th>{state === "post" ? "Result" : "Our pick"}</th>{state !== "post" && <><th>Your live line · over · under</th><th>Live read</th></>}</tr></thead>
            <tbody>{[...byGame[key]].sort((a, b) => a.player.localeCompare(b.player)).map(p => {
              const s = g.stats?.[normName(p.name || "")] || g.stats?.[shortKey(p.player)];
              const cur = state === "pre" ? 0 : s?.[p.stat] ?? 0, m = data.live_model?.[`${p.pos}:${p.stat}`];
              const o = odds[p.id] || {};
              const inp = (fld: "line" | "over" | "under") => <input type="number" step={fld === "line" ? 0.5 : 1} placeholder={fld} aria-label={`${fld} for ${p.player}`}
                defaultValue={o[fld] ?? ""} onInput={e => setOdd(p.id, fld, (e.target as HTMLInputElement).value)} />;
              return <tr key={p.id}>
                <td><span className="player"><PlayerLink id={p.id}>{p.player}</PlayerLink></span> <span className="sub">{statLabel(p.stat)}</span>
                  {" "}<button type="button" className="btn ghost" style={{ padding: "1px 6px", fontSize: 11 }} aria-label={`Show ${p.player}'s plays`} onClick={() => { setFilter(x => ({ ...x, [key]: p.player })); setPlays(x => ({ ...x, [key]: true })); }}>plays</button></td>
                <td className="r num">{p.line}</td>
                <td className="r num">{state === "pre" ? "-" : cur}{state !== "pre" && !s && <div className="sub">{state === "post" ? "did not play" : "not in box"}</div>}</td>
                <td className="r num">{state === "post" ? cur : (cur + left * p.proj).toFixed(0)}</td>
                {(() => {
                  // pregame: our chance (the board's); in game: the live model's over chance for the same side
                  const pre = chance(p, trust), side = mine.get(p.id) ?? (pre >= 0.5 ? "Over" : "Under");
                  if (state === "post") { const won = cur === p.line ? null : (cur > p.line) === (side === "Over");
                    return <td><Pill kind={won == null ? "" : won ? "win" : "loss"}>{side} {won == null ? "push" : won ? "hit" : "missed"}</Pill></td>; }
                  const q = state === "in" && m ? livePOver(m, p.proj, p.line, cur, left) : pre;
                  return <td><Pill kind={side.toLowerCase()}>{side}</Pill> <span className="num">{pct(side === "Over" ? q : 1 - q)}</span>{mine.has(p.id) && <span className="sub"> · yours</span>}</td>;
                })()}
                {state !== "post" && <><td><span className="live-in">{inp("line")}{inp("over")}{inp("under")}</span></td>
                <td><Verdict p={p} left={left} cur={cur} odds={odds} /></td></>}</tr>;
            })}</tbody></table></div>
          {state !== "pre" && g.drives?.length && g.teams ? <details open={vizOpen} onToggle={e => { const v = (e.target as HTMLDetailsElement).open; if (v !== vizOpen) setViz(x => ({ ...x, [key]: v })); }}>
            <summary>Field view</summary><DriveStrip g={g} /><WpChart g={g} /></details> : null}
          {state !== "pre" && <details open={playsOpen} onToggle={e => { const v = (e.target as HTMLDetailsElement).open; if (v !== playsOpen) setPlays(x => ({ ...x, [key]: v })); }}>
            <summary>Play by play{f ? ` · ${f}` : ""}</summary>
            <div className="chips"><button type="button" aria-pressed={!f} onClick={() => setFilter(x => ({ ...x, [key]: null }))}>All</button>
              {names.map(n => <button key={n} type="button" aria-pressed={f === n} onClick={() => setFilter(x => ({ ...x, [key]: n }))}>{n}</button>)}</div>
            <ul className="plays">{shown.length ? shown.map((x, i) => <li key={i} className={x.score ? "score" : ""}><span className="sub num">Q{x.q ?? "-"} {x.clock ?? ""}</span><span>{x.text}</span></li>)
              : <li><span></span><span className="sub">No plays{f ? ` mentioning ${f}` : ""} yet.</span></li>}</ul></details>}
        </div>;
      }) : <div className="panel empty">{showMine ? "None of your picks are on this week's board." : "No props on the board this week."}</div>}
    </section>
  );
}
