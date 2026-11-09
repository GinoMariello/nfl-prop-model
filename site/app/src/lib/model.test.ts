import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { BoardData } from "../types";
import { bestBook, buildParlays, dkLink, kellyStake, sizeStakes, edgeScore, gameProps, legResult, probit, projectedScore, missDist, pkChance, seeded, shortlist, SHORT_MIN_LINE } from "./model";
import { livePOver, timeLeft } from "./live";

const data = JSON.parse(readFileSync(new URL("../../../data.json", import.meta.url), "utf8")) as BoardData;
const wr = data.live_model["WR:receiving_yards"];
const BEFORE = new Date(data.props.map(p => p.kickoff).sort()[0]).getTime() - 3600e3;   // an hour before the first kickoff

describe("livePOver (mirrors test_live_p_over.py)", () => {
  it("is decided at the final whistle", () => {
    expect(livePOver(wr, 60, 70.5, 71, 0)).toBe(1);
    expect(livePOver(wr, 60, 70.5, 70, 0)).toBe(0);
  });
  it("rises with yards so far and with time left when behind", () => {
    const ps = Array.from({ length: 16 }, (_, i) => livePOver(wr, 60, 70.5, i * 5, 0.5));
    ps.slice(1).forEach((p, i) => expect(p).toBeGreaterThanOrEqual(ps[i]));
    expect(livePOver(wr, 60, 70.5, 65, 0.1)).toBeLessThan(livePOver(wr, 60, 70.5, 65, 0.5));
    expect(livePOver(wr, 60, 70.5, 75, 0.5)).toBe(1);
  });
  it("timeLeft runs 1 before kickoff to 0 at the end", () => {
    expect(timeLeft(undefined)).toBe(1);
    expect(timeLeft({ period: 3, clock: 450, type: { state: "in", shortDetail: "" } })).toBeCloseTo((900 + 450) / 3600);
    expect(timeLeft({ period: 5, clock: 300, type: { state: "in", shortDetail: "" } })).toBe(0);
  });
});

describe("shortlist and parlays on this week's data.json", () => {
  const T = 0.55, { rows } = shortlist(data, T, BEFORE);
  it("keeps only 55%+ bets that beat their price and aren't never-unders or blowouts", () => {
    rows.forEach(r => {
      expect(r.ch).toBeGreaterThanOrEqual(T);
      if (r.price != null) expect(r.ch * (1 + (r.price < 0 ? 100 / -r.price : r.price / 100))).toBeGreaterThan(1);
      if (r.side === "Under") expect(data.no_under).not.toContain(r.p.id.split(":")[0]);
    });
  });
  it("builds parlays with each pick once, every leg in a different game, no short yardage or QB rushing", () => {
    for (const n of [2, 3]) {
      const ps = buildParlays(data, T, n, 0, 0.1, BEFORE), seen = new Set<string>();
      ps.forEach(pl => {
        expect(pl.legs).toHaveLength(n);
        expect(new Set(pl.legs.map(l => l.p.game)).size).toBe(n);
        pl.legs.forEach(l => {
          expect(seen.has(l.p.id)).toBe(false); seen.add(l.p.id);
          if (l.p.stat !== "receptions") expect(l.p.line).toBeGreaterThanOrEqual(SHORT_MIN_LINE);
          expect(l.p.pos === "QB" && l.p.stat === "rushing_yards").toBe(false);
        });
      });
      expect(buildParlays(data, T, n, 0, 0.1, BEFORE).map(p => p.legs.map(l => l.p.id))).toEqual(ps.map(p => p.legs.map(l => l.p.id)));   // seeded
    }
  });
  it("seeded() is the original mulberry32 stream", () => {
    const r = seeded(202607); expect([r(), r(), r()].map(v => v.toFixed(6))).toEqual(["0.625730", "0.761968", "0.930695"]);
  });
});

describe("pick'em math", () => {
  const p = data.props.find(x => x.ladder?.dist) as BoardData["props"][number];
  it("prices any line; More and Less add up to one off the quantiles", () => {
    const more = pkChance(data, p, "More", p.line) as number, less = pkChance(data, p, "Less", p.line) as number;
    expect(more + less).toBeLessThanOrEqual(1.0001);
    expect(pkChance(data, p, "More", p.line - 50)).toBeGreaterThan(more);
  });
  it("missDist sums to one and gives P(all hit) as the product", () => {
    const d = missDist([0.6, 0.7]);
    expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(d[0]).toBeCloseTo(0.42);
  });
});

describe("game bets (moneyline, spread)", () => {
  const g = { id: "e", game_id: "2026_07_PIT_NO", kickoff: "2026-10-25T17:00:00Z", home: "New Orleans Saints", away: "Pittsburgh Steelers", books: 8,
    spread: 3, total: 44.5, ml_home: 140, ml_away: -170, best_home: null, best_away: null, p_home: 0.4, residual_home: 0.41,
    implied_home: null, implied_away: null, spread_move: null, total_move: null, spread_price_home: -110, spread_price_away: -110, total_price_over: -110, total_price_under: -110 };
  const d = { props: [], games: [g], actuals: { "2026:7:2026_07_PIT_NO~away:spread": 3, "2026:7:2026_07_PIT_NO~home:moneyline": -3, "2026:7:2026_07_PIT_NO~game:total": 44.5 } } as unknown as BoardData;
  it("prices both teams, home spread as given and away flipped", () => {
    const ps = gameProps(d);
    expect(ps.map(p => p.id)).toEqual(["2026_07_PIT_NO~away:moneyline", "2026_07_PIT_NO~away:spread", "2026_07_PIT_NO~home:moneyline", "2026_07_PIT_NO~home:spread", "2026_07_PIT_NO~game:total"]);
    expect(ps[1].line).toBe(-3); expect(ps[3].line).toBe(3);
    expect(ps[0].p_over).toBeCloseTo(0.59); expect(ps[1].p_over).toBeCloseTo(0.5);
  });
  it("settles on the final margin: a 3-point win on -3 pushes, a loss misses", () => {
    expect(legResult(d, 2026, 7, "2026_07_PIT_NO~away:spread", "spread", -3, true)).toBe("push");
    expect(legResult(d, 2026, 7, "2026_07_PIT_NO~away:spread", "spread", -2.5, true)).toBe("hit");
    expect(legResult(d, 2026, 7, "2026_07_PIT_NO~home:moneyline", "moneyline", 0, true)).toBe("miss");
    expect(legResult(d, 2026, 7, "2026_07_PIT_NO~game:total", "total", 44.5, true)).toBe("push");
    expect(legResult(d, 2026, 7, "2026_07_PIT_NO~game:total", "total", 43.5, false)).toBe("miss");
  });
});

describe("projected score", () => {
  it("turns a win chance into a margin around the total", () => {
    expect(probit(0.5)).toBeCloseTo(0, 6); expect(probit(0.975)).toBeCloseTo(1.96, 3); expect(probit(0.01)).toBeCloseTo(-2.326, 3);
    const s = projectedScore({ model_home: 0.75, total: 44 })!;
    expect(s.margin).toBeCloseTo(9.1, 1); expect(s.home + s.away).toBeCloseTo(44); expect(s.home).toBeGreaterThan(s.away);
    expect(projectedScore({ model_home: null, total: 44 })).toBeNull();
  });
});

describe("edgeScore", () => {
  const base = { ...data.props[0], line: 50, proj: 40, stat: "receiving_yards", form: { cv: 0.5, n: 8, sd: 20 },
                 sources: { own: null, sleeper: 45, gbm: 44 } };
  it("adds a point for each projection on the bet's side and for a big edge", () => {
    expect(edgeScore(base, "Under")).toMatchObject({ pts: 3, level: 3 });   // gbm, sleeper, |40-50|/20 = 0.5 > EDGE_Z
    expect(edgeScore({ ...base, sources: { own: null, sleeper: 55, gbm: 44 } }, "Under")).toMatchObject({ pts: 2, level: 2 });
    expect(edgeScore(base, "Over")).toMatchObject({ pts: 0, level: 0 });    // the other side gets none of them
  });
  it("rates receptions up and rushing down", () => {
    expect(edgeScore({ ...base, stat: "receptions" }, "Under").level).toBe(3);
    expect(edgeScore({ ...base, stat: "receptions", sources: { own: null, sleeper: 55, gbm: 55 } }, "Under")).toMatchObject({ pts: 1.5, level: 1 });
    expect(edgeScore({ ...base, stat: "rushing_yards" }, "Under")).toMatchObject({ pts: 1.5, level: 1 });
    expect(edgeScore({ ...base, stat: "rushing_yards", sources: { own: null, sleeper: 55, gbm: 44 } }, "Under").level).toBe(0);
  });
  it("skips missing projections and spreads", () => {
    expect(edgeScore({ ...base, form: null, sources: { own: null, sleeper: null, gbm: null } }, "Under").pts).toBe(0);
  });
});

describe("dkLink", () => {
  const p = (line: number, dk: object | null) => ({ ...data.props[0], line, dk }) as (typeof data.props)[number];
  it("joins encoded DraftKings sids with + and flags missing legs and moved lines", () => {
    const a = p(5.5, { over: { sid: "0QA1#2_X", line: 5.5 }, under: { sid: "0QA1#3_X", line: 5.5 } });
    const b = p(70.5, { under: { sid: "0QA9#8_Y", line: 71.5 } });
    const c = p(9.5, null);
    const r = dkLink([{ p: a, side: "Over" }, { p: b, side: "Under" }, { p: c, side: "Under" }]);
    expect(r.url).toBe("https://sportsbook.draftkings.com/?outcomes=0QA1%232_X+0QA9%238_Y");
    expect(r.missing).toEqual([c]);
    expect(r.moved).toEqual([{ p: b, side: "Under", line: 71.5 }]);
    expect(dkLink([{ p: c, side: "Over" }]).url).toBeNull();
  });
});

describe("bestBook", () => {
  const dist = Array.from({ length: 100 }, (_, i) => 40 + i * 0.2);   // 40 to 59.8: P(under 50.5) ~ 0.53
  const p = { ...data.props[0], line: 50.5, p_market: null, ladder: { p_over: 0.47, over: {}, under: {}, k: 1, dist },
              shop: [["fanduel", "Under", 50.5, -110], ["betmgm", "Under", 52.5, -120], ["bovada", "Under", 50.5, -105], ["draftkings", "Over", 50.5, 100]] } as unknown as (typeof data.props)[number];
  it("prices each book's line along our range and keeps the best expected value", () => {
    const b = bestBook(data, p, "Under", 1)!;
    expect(b.book).toBe("betmgm");                 // two more yards (+10 pts of chance) beats a better price
    expect(b.ch).toBeCloseTo(0.63, 2);
    expect(bestBook(data, p, "Over", 1)!.book).toBe("draftkings");
    expect(bestBook(data, { ...p, shop: [] }, "Under", 1)).toBeNull();
  });
});

describe("stakes", () => {
  it("bets quarter Kelly capped at 1% per bet, nothing without an edge", () => {
    expect(kellyStake(0.6, -110, 1000)).toBeCloseTo(10);          // f = 0.16, a quarter is 4%, capped at 1%
    expect(kellyStake(0.53, -110, 1000)).toBeCloseTo(3.25, 2);    // f = 0.013, a quarter is 0.325%
    expect(kellyStake(0.5, -110, 1000)).toBe(0);
  });
  it("caps a game at 3% of the bankroll, scaling its bets alike", () => {
    const s = sizeStakes([{ game: "A", stake: 10 }, { game: "A", stake: 10 }, { game: "A", stake: 20 }, { game: "B", stake: 10 }], 1000);
    expect(s.map(x => x.stake)).toEqual([7.5, 7.5, 15, 10]);
  });
});
