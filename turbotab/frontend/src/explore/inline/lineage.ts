/**
 * A Lineage (raw → adjusted → matrix) read as one track per raw column: what it becomes in the
 * model matrix and by which operation. Tracks keep the raw column's identity, so a preview that
 * renames `fat_total` to `fat_total_adj` changes a label in place instead of replacing a row.
 */
import type { Lineage, LineageNode } from "./types";

export interface Track {
  id: string;
  raw: string;
  role: string | null;
  count: number;
  /** Matrix columns this raw column becomes; empty when it leaves the model. */
  outs: string[];
  /** The raw → adjusted operation ("kept", "energy-adjusted (residual)", "log2(x + 1)", …). */
  op: string;
  /** The adjusted column's formula, when it has one. */
  formula: string | null;
  /** Other raw columns feeding this column's adjusted value (energy into each nutrient). */
  inputs: string[];
  /** Adjusted columns this raw column feeds without becoming them. */
  feeds: string[];
}

export function tracksOf(lineage: Lineage): Track[] {
  const byId = new Map(lineage.nodes.map((n) => [n.id, n]));
  const out = (id: string) => lineage.links.filter((l) => l.source === id);
  const into = (id: string) => lineage.links.filter((l) => l.target === id);
  const raws = lineage.nodes.filter((n) => n.lane === "raw");
  return raws.map((r) => {
    const toAdj = out(r.id).filter((l) => byId.get(l.target)?.lane === "adjusted");
    const own = toAdj.find((l) => {
      const a = byId.get(l.target) as LineageNode;
      const sources = into(a.id);
      return sources.length === 1 || a.label.startsWith(r.label);
    });
    const adj = own ? byId.get(own.target) : undefined;
    const outs = adj
      ? out(adj.id)
          .map((l) => byId.get(l.target))
          .filter((n): n is LineageNode => !!n && n.lane === "matrix")
          .map((n) => n.label)
      : [];
    return {
      id: r.id,
      raw: r.label,
      role: r.role,
      count: r.count,
      outs,
      op: own?.operation ?? (toAdj.length ? "feeds" : "none"),
      formula: adj?.formula ?? null,
      inputs: adj
        ? into(adj.id)
            .map((l) => l.source)
            .filter((s) => s !== r.id)
        : [],
      feeds: toAdj.filter((l) => l !== own).map((l) => l.target),
    };
  });
}

/** Tracks whose path to the matrix differs from the base lineage. */
export function touched(tracks: Track[], base: Track[]): Set<string> {
  const b = new Map(base.map((t) => [t.id, t]));
  const s = new Set<string>();
  for (const t of tracks) {
    const o = b.get(t.id);
    if (
      !o ||
      o.op !== t.op ||
      o.outs.join() !== t.outs.join() ||
      o.inputs.join() !== t.inputs.join()
    )
      s.add(t.id);
  }
  return s;
}
