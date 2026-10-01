/**
 * Where each lineage node sits, and — the part that makes morphing honest — which
 * nodes are "the same column" across options.
 *
 * A node's identity is the raw column it comes from, read from the links rather than
 * from its name: `adj:protein_adj`, `adj:protein_per_kcal` and `adj:kcal_from_protein`
 * are all `protein`, so switching methods relabels one node instead of swapping it.
 * An energy-role node fed by the energy column (`kcal_from_other`) keeps kcal's identity.
 */
import type { Lineage, LineageNode } from "../types";

export type Lane = LineageNode["lane"];

export interface PlacedNode {
  key: string;
  lane: Lane;
  identity: string;
  row: number;
  node: LineageNode;
  changed: boolean;
}

export interface PlacedLink {
  key: string;
  from: PlacedNode;
  to: PlacedNode;
  operation: string;
  /** The link carries the option's rewrite (not kept / scaled / one-hot). */
  rewrite: boolean;
}

export interface Placed {
  rows: number;
  nodes: PlacedNode[];
  links: PlacedLink[];
  /** Raw columns with no path of their own into the model, and what became of them. */
  notes: { identity: string; row: number; text: string }[];
  counts: Record<Lane, number>;
}

const PLAIN = new Set(["kept", "scaled", "one-hot", "pass-through"]);

export function placeLineage(l: Lineage): Placed {
  const byId = new Map(l.nodes.map((n) => [n.id, n]));
  const into = new Map<string, string[]>();
  const outOf = new Map<string, string[]>();
  for (const k of l.links) {
    into.set(k.target, [...(into.get(k.target) ?? []), k.source]);
    outOf.set(k.source, [...(outOf.get(k.source) ?? []), k.target]);
  }

  const identity = new Map<string, string>();
  const raw = l.nodes.filter((n) => n.lane === "raw");
  for (const n of raw) identity.set(n.id, n.column ?? n.group ?? n.id.replace(/^raw:/, ""));

  for (const n of l.nodes.filter((x) => x.lane === "adjusted")) {
    const sources = (into.get(n.id) ?? []).map((id) => byId.get(id)).filter(Boolean) as LineageNode[];
    const energySource = sources.find((x) => x.role === "energy");
    const pick =
      n.role === "energy" && energySource
        ? energySource
        : (sources.find((x) => x.role !== "energy") ?? sources[0]);
    identity.set(n.id, pick ? identity.get(pick.id)! : n.id.replace(/^adj:/, ""));
  }
  for (const n of l.nodes.filter((x) => x.lane === "matrix")) {
    const src = (into.get(n.id) ?? [])[0];
    identity.set(n.id, src ? identity.get(src) ?? src : n.id.replace(/^mx:/, ""));
  }

  // Rows: one block per raw column, as tall as its widest lane (one-hot fans out).
  const order: string[] = [];
  for (const n of raw) order.push(identity.get(n.id)!);
  for (const n of l.nodes) {
    const id = identity.get(n.id)!;
    if (!order.includes(id)) order.push(id);
  }
  const span = new Map<string, number>();
  const perLane = new Map<string, number>();
  for (const n of l.nodes) {
    const k = `${n.lane}|${identity.get(n.id)}`;
    perLane.set(k, (perLane.get(k) ?? 0) + 1);
    const id = identity.get(n.id)!;
    span.set(id, Math.max(span.get(id) ?? 1, perLane.get(k)!));
  }
  const start = new Map<string, number>();
  let row = 0;
  for (const id of order) {
    start.set(id, row);
    row += span.get(id) ?? 1;
  }

  const seen = new Map<string, number>();
  const nodes: PlacedNode[] = l.nodes.map((n) => {
    const id = identity.get(n.id)!;
    const k = `${n.lane}|${id}`;
    const sub = seen.get(k) ?? 0;
    seen.set(k, sub + 1);
    return {
      key: `${n.lane}:${id}:${sub}`,
      lane: n.lane,
      identity: id,
      row: start.get(id)! + sub,
      node: n,
      changed: n.lane === "adjusted" && !!n.formula,
    };
  });
  const placed = new Map(nodes.map((p) => [p.node.id, p]));

  const links: PlacedLink[] = l.links
    .filter((k) => placed.has(k.source) && placed.has(k.target))
    .map((k) => {
      const from = placed.get(k.source)!;
      const to = placed.get(k.target)!;
      return {
        key: `${from.key}>${to.key}`,
        from,
        to,
        operation: k.operation,
        rewrite: !PLAIN.has(k.operation),
      };
    });

  const notes: Placed["notes"] = [];
  for (const n of raw) {
    const id = identity.get(n.id)!;
    const outs = outOf.get(n.id) ?? [];
    const ownPath = nodes.some((p) => p.lane !== "raw" && p.identity === id);
    if (outs.length === 0) notes.push({ identity: id, row: start.get(id)!, text: "not a predictor" });
    else if (!ownPath) notes.push({ identity: id, row: start.get(id)!, text: `folded into ${outs.length}` });
  }

  const counts = { raw: 0, adjusted: 0, matrix: 0 } as Record<Lane, number>;
  for (const n of l.nodes) counts[n.lane] += n.count;

  return { rows: row, nodes, links, notes, counts };
}
