import { useEffect, useState } from "react";
import { store } from "./store";

// Hosted runtime: the owner's saves live in the artifact db; viewers' saves stay in their browser.
/* eslint-disable @typescript-eslint/no-explicit-any */
type Db = any;
declare global { interface Window { claude?: { use: (name: string) => Promise<any> } } }

export interface SavedParlay {
  id: string; created_at: string; season: number; week: number; stake: number; trust: number; decimal: number; prob: number; mkt: number | null;
  legs: { id: string; player: string; stat: string; side: "Over" | "Under"; line: number; price: number; game: string; prob: number; mkt: number | null }[];
}
export interface SavedEntry {
  id: string; created_at: string; season: number; week: number; stake: number; mult_total: number; flex: number[];
  p_all: number; ev_max: number; ev_flex: number | null;
  legs: { id: string; player: string; stat: string; game: string; side: "More" | "Less"; line: number; mult: number; chance: number }[];
}

type Col = "parlays" | "pickem";
const LOCAL: Record<Col, string> = { parlays: "pb-my-parlays", pickem: "pb-my-pickem" };
const newest = <T extends { created_at: string }>(xs: T[]) => xs
  .filter(x => x && typeof x.created_at === "string" && Array.isArray((x as any).legs) && (x as any).legs.length)
  .sort((a, b) => b.created_at.localeCompare(a.created_at));

/** Saved parlays and pick'em entries: the owner's in the artifact db, everyone else's in their own browser. */
export function useBoardDb() {
  const [db, setDb] = useState<Db | null>(null);
  const [note, setNote] = useState("");
  const [parlays, setParlays] = useState<SavedParlay[]>(() => newest(store.get<SavedParlay[]>(LOCAL.parlays, [])));
  const [entries, setEntries] = useState<SavedEntry[]>(() => newest(store.get<SavedEntry[]>(LOCAL.pickem, [])));
  const [entriesNote, setEntriesNote] = useState("");
  useEffect(() => {
    const c = window.claude;
    if (!c?.use) return;
    let unsubs: (() => void)[] = [];
    (async () => {
      const [shared, user] = await Promise.all([c.use("db"), c.use("user")]);
      if (!shared || !(await user?.isOwner?.())) return;   // viewers keep their own saves locally
      setDb(shared);
      const keep = (d: any) => Array.isArray(d.legs) && d.legs.length;
      unsubs = [
        shared.collection("parlays").orderBy("created_at", "desc").onSnapshot(
          (snap: any) => setParlays(snap.docs.map((d: any) => ({ id: d.id, ...d.data() })).filter(keep)),
          (err: any) => setNote(`Saved parlays could not load (${err.code || err.message}).`)),
        shared.collection("pickem").orderBy("created_at", "desc").onSnapshot(
          (snap: any) => setEntries(snap.docs.map((d: any) => ({ id: d.id, ...d.data() })).filter(keep)),
          (err: any) => setEntriesNote(`Saved entries could not load (${err.code || err.message}).`)),
      ].filter(u => typeof u === "function");
    })();
    return () => unsubs.forEach(u => u());
  }, []);
  const local = (col: Col, fn: (xs: any[]) => any[]) => {
    const next = newest(fn(store.get<any[]>(LOCAL[col], [])));
    store.set(LOCAL[col], next);
    (col === "parlays" ? setParlays : setEntries)(next);
  };
  const add = async (col: Col, doc: object) => {
    if (db) return void (await db.collection(col).add(doc));
    local(col, xs => [...xs, { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, ...doc }]);
  };
  const remove = async (col: Col, id: string) => {
    if (db) return void (await db.collection(col).doc(id).delete());
    local(col, xs => xs.filter(x => x.id !== id));
  };
  const where = db ? "" : "Saved in this browser only: other devices and other people don't see it.";
  return { db, where, note, setNote, parlays, entries, entriesNote, add, remove };
}
