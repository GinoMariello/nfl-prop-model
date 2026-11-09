import { createContext, useContext } from "react";
import type { BoardData } from "./types";
import type { SlipLeg } from "./lib/model";
import type { useBoardDb } from "./lib/db";

export interface Board {
  data: BoardData;
  trust: number;                       // 0-1, the "Trust in our model" slider
  pos: string; stat: string; q: string; onlyEdge: boolean;
  minChance: number; onlyMeets: boolean;   // Hit rates tab
  slip: SlipLeg[]; setSlip: (f: SlipLeg[] | ((s: SlipLeg[]) => SlipLeg[])) => void;
  stake: number; setStake: (n: number) => void;
  dbx: ReturnType<typeof useBoardDb>;
  openPlayer: (playerId: string) => void;   // opens the player breakdown
}
export const BoardCtx = createContext<Board | null>(null);
export const useBoard = () => { const b = useContext(BoardCtx); if (!b) throw new Error("useBoard outside the board"); return b; };
