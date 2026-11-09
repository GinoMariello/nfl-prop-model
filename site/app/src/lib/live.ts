import type { LiveModel, Stat } from "../types";

// ESPN's public scoreboard + game summaries (CORS-open; works from the published page).
export const ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/";
const ESPN_STAT: Partial<Record<Stat, [string, string]>> = { passing_yards: ["passing", "passingYards"], rushing_yards: ["rushing", "rushingYards"],
  receiving_yards: ["receiving", "receivingYards"], receptions: ["receiving", "receptions"] };

/** Mirrors site_data.live_p_over (test_live_p_over.py checks the Python side). */
export function livePOver(m: LiveModel, proj: number, line: number, cur: number, left: number) {
  if (cur > line) return 1;
  let b = 0; while (b < m.edges.length && m.edges[b] <= proj) b++;
  const raw = m.q[b].filter(q => cur + Math.max(0, left * proj + Math.sqrt(left) * q) > line).length / m.q[b].length;
  return 0.5 + (1 - (1 - m.shrink) * left) * (raw - 0.5);
}

/* eslint-disable @typescript-eslint/no-explicit-any -- ESPN's payloads are untyped; read defensively */
export type Espn = any;
export interface Status { period: number; clock: number; type: { state: "pre" | "in" | "post"; shortDetail: string } }
/** Share of regulation left; 0 in overtime and after. */
export function timeLeft(st: Status | undefined) {
  if (!st || st.type.state === "pre") return 1;
  if (st.type.state === "post" || st.period > 4) return 0;
  return Math.max(0, ((4 - st.period) * 900 + (st.clock || 0)) / 3600);
}
export const normName = (s: string) => String(s).toLowerCase().replace(/\b(jr|sr|ii|iii|iv)\b\.?/g, "").replace(/[^a-z]/g, "");
export const shortKey = (s: string) => { const w = String(s).replace(/\b(Jr|Sr|II|III|IV)\b\.?/g, "").trim().split(/[\s.]+/); return normName(w[0][0] + w[w.length - 1]); };

/** Full-name and initial+last keys -> {stat: value}. */
export function boxStats(sum: Espn) {
  const out: Record<string, Partial<Record<Stat, number>>> = {};
  (sum.boxscore?.players || []).forEach((t: Espn) => (t.statistics || []).forEach((cat: Espn) => {
    (Object.entries(ESPN_STAT) as [Stat, [string, string]][]).forEach(([stat, [name, key]]) => {
      if (cat.name !== name) return;
      const i = cat.keys.indexOf(key); if (i < 0) return;
      cat.athletes.forEach((a: Espn) => {
        const v = Number(a.stats[i]); if (!Number.isFinite(v)) return;
        [normName(a.athlete.displayName), shortKey(a.athlete.displayName)].forEach(k => { (out[k] ||= {})[stat] = v; });
      });
    });
  }));
  return out;
}
export interface PlayLine { q?: number; clock?: string; text: string; score?: boolean }
export function allPlays(sum: Espn): PlayLine[] {
  const seen = new Set<string>(), plays: PlayLine[] = [];
  [...(sum.drives?.previous || []), sum.drives?.current].filter(Boolean).forEach((d: Espn) => (d.plays || []).forEach((p: Espn) => {
    if (!seen.has(p.id)) { seen.add(p.id); plays.push({ q: p.period?.number, clock: p.clock?.displayValue, text: p.text || "", score: p.scoringPlay }); }
  }));
  return plays.reverse();
}
export const CLOCK_ONLY = /timeout|end period|end of|two-minute|end game/i;
export function gamePlays(drives: Espn[]) {
  return drives.flatMap((d: Espn) => (d.plays || []).map((p: Espn) => ({ p, d })))
    .filter(({ p }: Espn) => p.start?.yardLine != null && p.end?.yardLine != null && !CLOCK_ONLY.test(p.type?.text || ""));
}
export async function getJson(url: string): Promise<Espn> {
  const r = await fetch(url); if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json();
}

/** The board's own copy of ESPN's data (site/live.json from live_feed.py), for when this page may not reach ESPN. */
interface Feed { at: string; scoreboard: Espn; summaries: Record<string, Espn> }
let feed: Feed | null = null, feedAt = 0, direct: boolean | null = null, host: boolean | null = null;
let mcp: Promise<Espn | null> | null = null;
/** ESPN through the espn program on the owner's Mac (espn_mcp.py), on the private board;
 * undefined when this view has no such route, which then stays off for the page load. */
async function viaMac(path: string): Promise<Espn | undefined> {
  if (host === false) return undefined;
  mcp ||= window.claude?.use ? window.claude.use("mcp").catch(() => null) : Promise.resolve(null);
  const m = await mcp;
  if (!m) { host = false; return undefined; }
  const [tool, input] = path === "scoreboard" ? ["scoreboard", {}] : ["summary", { event: path.split("=").pop() }];
  try { const r = await m.callTool("host:espn", tool, input, { cache: false }); host = true; return r.payload; }
  catch (e) {
    const code = (e as { code?: string }).code;
    if (host !== true || ["server_not_connected", "not_in_manifest", "not_granted", "capability_disabled", "capability_removed", "blocked_by_policy"].includes(code ?? "")) host = false;
    if (host === false) return undefined;
    throw new Error(`the espn program on your Mac: ${code}`);
  }
}
/** ESPN directly where the page may reach it (locally), else through the owner's Mac (private board),
 * else live.json, re-read at most every 20 seconds. */
export async function espn(path: string): Promise<Espn> {
  if (direct !== false) {
    try { const r = await getJson(ESPN + path); direct = true; return r; }
    catch (e) { if (direct) throw e; direct = false; }   // never reached ESPN in this view: use the other routes from now on
  }
  const mac = await viaMac(path);
  if (mac !== undefined) return mac;
  if (!feed || Date.now() - feedAt > 20000) { feed = await getJson(`live.json?t=${Date.now()}`); feedAt = Date.now(); }
  if (path === "scoreboard") return (feed as Feed).scoreboard;
  return (feed as Feed).summaries[path.split("=").pop() as string] ?? {};
}
/** When the scores came from live.json: the time the feed was pulled. */
export const feedTime = () => (direct === false && host === false && feed ? new Date(feed.at) : null);
/** True when the scores come through the espn program on the owner's Mac. */
export const viaHost = () => direct === false && host === true;
