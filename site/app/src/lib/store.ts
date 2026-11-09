import { useCallback, useState } from "react";

// localStorage, per viewer, never required: the same keys the original page used ("pb-*"),
// so a viewer's slip, tab and picks carry over.
export const store = {
  get<T>(k: string, d: T): T {
    try {
      const v = localStorage.getItem(k); if (v == null) return d;
      const x = JSON.parse(v);   // edited or stale data of the wrong shape falls back to the default instead of crashing the page
      return Array.isArray(d) !== Array.isArray(x) || typeof x !== typeof d ? d : (x as T);
    } catch { return d; }
  },
  set(k: string, v: unknown) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked */ } },
};

export function useStored<T>(key: string, initial: T): [T, (v: T | ((old: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => store.get(key, initial));
  const set = useCallback((v: T | ((old: T) => T)) => {
    setValue(old => { const next = typeof v === "function" ? (v as (o: T) => T)(old) : v; store.set(key, next); return next; });
  }, [key]);
  return [value, set];
}
