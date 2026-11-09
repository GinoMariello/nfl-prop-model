// The data.json contract, as site_data.main() writes it.
export type Stat = "passing_yards" | "rushing_yards" | "receiving_yards" | "receptions" | "moneyline" | "spread" | "total";
export type Side = "Over" | "Under";
export type Result = "win" | "loss" | "push" | "void";

export interface Ladder {
  p_over: number;
  over: Record<string, number | null>;
  under: Record<string, number | null>;
  k?: number;
  dist: number[];
}
export interface PaperBet { side: Side; stake: number; price: number; result: Result | null; profit: number | null }
export interface Prop {
  id: string; player: string; name: string; pos: string; stat: Stat; game: string; kickoff: string;
  line: number; over: number | null; under: number | null; books: number; proj: number;
  sources: { own: number | null; sleeper: number | null; gbm: number | null };
  /** DraftKings' own outcome ids (site_data.draftkings_sids) for its betslip link, with its line, which can differ from ours. */
  dk?: { over?: DkOutcome; under?: DkOutcome } | null;
  /** Every book's offer (site_data "shop"): [book, side, line, American price]. */
  shop?: [string, Side, number, number][];
  gap: number; p_over: number; p_market: number | null; ladder?: Ladder;
  form?: { cv: number; n: number; sd?: number } | null;   // his last 8 games before this week: std / mean, count, std
  ev_over: number | null; ev_under: number | null; actual: number | null; bet: PaperBet | null;
}
export interface DkOutcome { sid: string; line: number }
export interface LedgerSummary { bets: number; settled: number; wins: number; losses: number; staked: number; profit: number; open_stake: number }
export interface WeekBet {
  player_name: string; position: string; stat: Stat; side: Side; line: number; projection: number;
  price: number; chance: number; stake: number; actual: number | null; result: Result | null; profit: number | null;
}
export interface HitLedger extends LedgerSummary { week_bets: WeekBet[] }
/** A side test on the paper rule (site_data.side_ledger): kept bets, this week's dropped ones, and the whole rule over the same weeks. */
export interface SideTest extends HitLedger { main?: LedgerSummary; from_week?: number }
export interface Game {
  id: string; kickoff: string; home: string; away: string; books: number;
  spread: number | null; spread_lo?: number | null; spread_hi?: number | null;
  total: number | null; total_lo?: number | null; total_hi?: number | null;
  ml_home: number | null; ml_away: number | null; best_home: number | null; best_away: number | null;
  p_home: number | null; implied_home: number | null; implied_away: number | null;
  spread_move: number | null; total_move: number | null; model_home?: number | null; residual_home?: number | null;
  game_id?: string; spread_price_home?: number | null; spread_price_away?: number | null;
  total_price_over?: number | null; total_price_under?: number | null;
}
/** One regular-season game from a player's log (season_logs in site_data.py). */
export interface GameLog {
  week: number; team: string; opp: string; attempts: number | null; completions: number | null; passing_yards: number | null;
  carries: number | null; rushing_yards: number | null; targets: number | null; receptions: number | null; receiving_yards: number | null;
  passing_tds: number | null; rushing_tds: number | null; receiving_tds: number | null; fantasy_points_ppr: number | null;
}
export interface LiveModel { edges: number[]; shrink: number; q: number[][] }
export interface Calibration { pos: string; stat: Stat; bucket: string; n: number; predicted: number; observed: number }
export interface Accuracy { pos: string; week: number; n: number; miss: Record<string, number | null> }
export interface WeekRow { season: number; week: number; bets: number; wins: number; losses: number; staked: number; profit: number }
export interface StakeRow { stake: number; bets: number; wins: number; losses: number; staked: number; profit: number }
export interface BoardData {
  season: number; week: number; generated_at: string; lines_pulled_at: string;
  props: Prop[]; ladder_k: number; no_under: string[]; live_model: Record<string, LiveModel>;
  calibration: Calibration[]; accuracy: Accuracy[]; history: { weeks: WeekRow[]; stakes: StakeRow[] };
  actuals: Record<string, number>; games: Game[]; logs?: Record<string, GameLog[]>;
  game_model_record?: { market: { accuracy: number }; team_residual: { accuracy: number }; team_only: { accuracy: number } };
  game_lines_pulled_at: string;
  ledger: LedgerSummary; ledger_old: LedgerSummary | null; ledger_hit: HitLedger | null;
  ledger_rec?: HitLedger | null; ledger_oppf?: SideTest | null; ledger_by_stat?: Record<string, LedgerSummary>;
  ledger_norush?: SideTest | null; ledger_norush_avoid?: SideTest | null;
}
