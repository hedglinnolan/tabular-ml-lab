/**
 * The canvas grammar's router (FOUNDATION §5): an option's footprint, measured from the engine's
 * own views, picks the canvas layout.
 *
 *   rows change                     → Flow     the participant flow, who leaves, how they differ
 *   two or more columns change      → Strip    every changed column ranked, the first focused large
 *   one column changes              → Focus    one large view with the transform player
 *   what feeds where changes        → Routing  raw columns → roles → the model's inputs
 *   the contract declares tradeoffs → Angles   two or three linked panels and an option table
 *   nothing changes                 → none     one line says so (rule 7)
 *
 * Declared tradeoffs come first (they are the teaching), then rows, then values, then routing:
 * the largest measured change gets the most room (rule 1). Pure and tested (router.test.ts).
 */
import type { ConsequenceView, HistogramData, LineageView, Preview } from "./fixture";

export type Layout = "focus" | "strip" | "flow" | "routing" | "angles" | "none" | "refused";

export interface Footprint {
  /** Flow steps that drop rows, with the share of the starting rows each drops. */
  rows: { label: string; n: number; dropped: number; share: number }[];
  /** Columns whose values change, with a change magnitude. */
  columns: { column: string; magnitude: number }[];
  /** Columns that move into or out of the model, or into a new role. */
  routing: { column: string; change: "enters" | "leaves" | "role" }[];
  /** Declared tradeoffs (the method contract's questions). */
  angles: number;
}

const unticked = (s: string) => s.replaceAll("`", "");

function histDistance(a: HistogramData, b: HistogramData): number {
  const same = a.edges.length === b.edges.length && a.edges.every((e, i) => Math.abs(e - b.edges[i]!) < 1e-9);
  if (!same) return 1;
  const ta = a.counts.reduce((x, y) => x + y, 0) || 1;
  const tb = b.counts.reduce((x, y) => x + y, 0) || 1;
  // half the L1 distance between the two shapes: 0 identical, 1 disjoint
  return a.counts.reduce((d, c, i) => d + Math.abs(c / ta - (b.counts[i] ?? 0) / tb), 0) / 2;
}

const matrixNamed = (l: LineageView["after"] | null, c: string) =>
  !!l && l.nodes.some((n) => n.lane === "matrix" && n.column === c);

/** The columns a lineage's choice re-routes: into or out of the model's inputs, or to a new role. */
export function routingOf(v: LineageView, touch: string[] = []): Footprint["routing"] {
  const out: Footprint["routing"] = [];
  const cols = new Set<string>([...v.emphasis, ...touch]);
  for (const n of v.after.nodes) if (n.lane === "matrix" && n.column && !v.before?.nodes.some((m) => m.lane === "matrix" && m.column === n.column)) cols.add(n.column);
  for (const n of v.before?.nodes ?? []) if (n.lane === "matrix" && n.column && !matrixNamed(v.after, n.column)) cols.add(n.column);
  for (const c of cols) {
    const inAfter = matrixNamed(v.after, c);
    const inBefore = matrixNamed(v.before, c);
    // A grouped "before" (a role's columns collapsed to a count) names no column: a column it
    // does not name is new to the model only if the model grew.
    if (inAfter && !inBefore) {
      if (!v.before || !v.before.collapsed || matrixSize(v.after) > matrixSize(v.before)) out.push({ column: c, change: "enters" });
      continue;
    }
    if (inBefore && !inAfter) {
      if (!v.after.collapsed) out.push({ column: c, change: "leaves" });
      continue;
    }
    // the same place in the model, a new role in the adjusted lane
    const roleAfter = v.after.nodes.find((n) => n.lane === "adjusted" && n.column === c);
    if (!roleAfter) continue;
    const roleBefore = v.before?.nodes.find((n) => n.lane === "adjusted" && n.column === c);
    if (!roleBefore || roleBefore.label !== roleAfter.label) out.push({ column: c, change: "role" });
  }
  return out;
}

const matrixSize = (l: LineageView["after"]) => l.nodes.filter((n) => n.lane === "matrix").reduce((k, n) => k + (n.count || 1), 0);

export function footprint(p: Preview): Footprint {
  const fp: Footprint = { rows: [], columns: [], routing: [], angles: p.angles?.length ?? 0 };
  for (const v of p.views as ConsequenceView[]) {
    if (v.kind === "row_flow") {
      const start = v.after[0]?.n ?? 0;
      for (const st of v.after) if (st.dropped > 0) fp.rows.push({ label: unticked(st.label), n: st.n, dropped: st.dropped, share: start ? st.dropped / start : 0 });
    } else if (v.kind === "lineage") {
      for (const r of routingOf(v)) if (!fp.routing.some((x) => x.column === r.column)) fp.routing.push(r);
    }
  }
  if (p.strip?.length) {
    for (const c of p.strip) if (c.shift > 0) fp.columns.push({ column: c.column, magnitude: c.shift });
  } else if (!fp.rows.length) {
    // Values that change, where no row leaves (a distribution under a cut is the rows' picture).
    for (const v of p.views as ConsequenceView[]) {
      if (v.kind === "distribution") {
        const d = histDistance(v.before, v.after);
        const m = d > 0 ? d : v.marks.length ? 0.01 : 0;
        if (m > 0) fp.columns.push({ column: v.column, magnitude: m });
      } else if (v.kind === "table_focus") {
        for (const c of new Set(v.changed.map(([, col]) => col))) fp.columns.push({ column: c, magnitude: 1 });
      } else if (v.kind === "relationship") {
        const dr = v.r_before !== null && v.r_after !== null ? Math.abs(v.r_after - v.r_before) : 0;
        if (dr >= 0.05 || v.y_label_before !== v.y_label_after) fp.columns.push({ column: v.y_label_before, magnitude: Math.min(1, dr) });
      }
    }
  }
  return fp;
}

export function route(p: Preview, opts: { disabled?: boolean } = {}): Layout {
  if (opts.disabled || p.refusal) return "refused";
  const fp = footprint(p);
  if (fp.angles >= 2) return "angles";
  if (fp.rows.length) return "flow";
  if (fp.columns.length >= 2) return "strip";
  if (fp.columns.length === 1) return "focus";
  if (fp.routing.length) return "routing";
  return "none";
}

/** The views a layout draws, primary first: at most three visible, the rest behind "More angles". */
export function viewsFor(layout: Layout, p: Preview): { shown: ConsequenceView[]; more: ConsequenceView[] } {
  const views = p.views as ConsequenceView[];
  const first: Record<Layout, ConsequenceView["kind"][]> = {
    flow: ["row_flow", "distribution", "table_focus", "lineage", "relationship"],
    focus: ["distribution", "relationship", "table_focus", "lineage", "row_flow"],
    strip: ["relationship", "lineage", "distribution", "table_focus", "row_flow"],
    routing: ["lineage", "row_flow", "distribution", "relationship", "table_focus"],
    angles: [],
    none: [],
    refused: [],
  };
  if (layout === "angles" || layout === "none" || layout === "refused") return { shown: [], more: [] };
  const rank = first[layout];
  const sorted = [...views].sort((a, b) => rank.indexOf(a.kind) - rank.indexOf(b.kind));
  // The Strip draws its own list and focused column; the engine's views follow it.
  const cap = layout === "strip" ? 1 : 3;
  return { shown: sorted.slice(0, cap), more: sorted.slice(cap) };
}
