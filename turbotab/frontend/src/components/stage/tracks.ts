/**
 * A consequence view as a sequence of real, labeled states — the player's material.
 *
 * Every view kind has the same shape here: state 0 is the data as recorded, the last state is the
 * data with the option applied, and the states between are the method's storyboard (§12.1). The
 * player moves one position across all of a preview's views at once; a view with a shorter
 * storyboard maps that position onto its own states, so every view starts and lands together.
 *
 * The morph rule (§11): a view's values glide between two states only when both are in the same
 * unit. Across a change of unit (g → g per kcal, g → kcal, counts → log2) the states crossfade,
 * because a point drawn half-way between 80 g and 0.04 g/kcal is not a value of anything.
 */
import type {
  ConsequenceView,
  DistributionView,
  HistogramData,
  Lineage,
  LineageView,
  RelationshipView,
  RowFlowView,
  RowStep,
  TableFocusView,
} from "../../api/m1-stage-types";
import { fmtInt, fmtR } from "./format";

// ── states ───────────────────────────────────────────────────────────────────

export interface RelationshipState {
  label: string;
  yLabel: string;
  points: [number, number][];
  r: number | null;
  /** The storyboard's own fitted line (a frame), drawn emphasized; else null. */
  fit: { slope: number; intercept: number } | null;
}

export interface DistributionState {
  label: string;
  /** What the values are (a column, or a step's own quantity): decides a change of unit. */
  unit: string;
  hist: HistogramData;
}

export interface LineageState {
  label: string;
  lineage: Lineage;
}

export interface TableState {
  label: string;
  columns: string[];
  values: Map<number, Record<string, unknown>>;
  /** The rows shown leave the analysis whole (complete cases). */
  gone: boolean;
}

export interface RowFlowState {
  label: string;
  steps: RowStep[];
}

export type StateOf<V extends ConsequenceView> = V extends RelationshipView
  ? RelationshipState
  : V extends DistributionView
    ? DistributionState
    : V extends LineageView
      ? LineageState
      : V extends TableFocusView
        ? TableState
        : RowFlowState;

export interface Track<V extends ConsequenceView = ConsequenceView> {
  view: V;
  states: StateOf<V>[];
  /** The view's evidence is the recorded data only: before and after are the same. */
  still: boolean;
}

export const NOW_LABEL = "Your data now";
export const WITH_LABEL = "With this choice";

function relationshipStates(v: RelationshipView): RelationshipState[] {
  const before: RelationshipState = {
    label: NOW_LABEL,
    yLabel: v.y_label_before,
    points: v.points_before,
    r: v.r_before,
    fit: null,
  };
  // A frame names its values only when they change (a residual); otherwise they are the last
  // state's (the fit is drawn over the data as it is).
  let yLabel = before.yLabel;
  const frames: RelationshipState[] = v.story.map((f) => ({
    label: f.label,
    yLabel: (yLabel = f.y_label ?? yLabel),
    points: f.points,
    r: f.r,
    fit: f.fit_line,
  }));
  const after: RelationshipState = {
    label: WITH_LABEL,
    yLabel: v.y_label_after,
    points: v.points_after.length ? v.points_after : v.points_before,
    r: v.r_after,
    fit: null,
  };
  return [before, ...frames, after];
}

function distributionStates(v: DistributionView): DistributionState[] {
  let unit = v.before_label;
  return [
    { label: v.before_label, unit, hist: v.before },
    ...v.story.map((f) => ({ label: f.label, unit: (unit = f.x_label ?? unit), hist: f.hist })),
    { label: v.after_label, unit: v.after_label, hist: v.after },
  ];
}

function lineageStates(v: LineageView): LineageState[] {
  return [
    { label: NOW_LABEL, lineage: v.before ?? v.after },
    ...v.story.map((f) => ({ label: f.label, lineage: f.lineage })),
    { label: WITH_LABEL, lineage: v.after },
  ];
}

function tableStates(v: TableFocusView): TableState[] {
  const side = (which: "before" | "after") =>
    new Map(v.rows.map((r) => [r.row_id, r[which] as Record<string, unknown>]));
  // Rows that leave whole (complete cases) have no "after": they are shown leaving, not blank.
  const leave = v.columns_after.length === 0 && v.rows.every((r) => Object.keys(r.after).length === 0);
  return [
    {
      label: NOW_LABEL,
      columns: v.columns_before.length ? v.columns_before : v.columns_after,
      values: side("before"),
      gone: false,
    },
    ...v.story.map((f) => ({
      label: f.label,
      columns: f.columns,
      values: new Map(f.rows.map((r) => [r.row_id, r.values])),
      gone: false,
    })),
    {
      label: WITH_LABEL,
      columns: v.columns_after.length ? v.columns_after : v.columns_before,
      values: leave ? side("before") : side("after"),
      gone: leave,
    },
  ];
}

function rowFlowStates(v: RowFlowView): RowFlowState[] {
  return [
    { label: NOW_LABEL, steps: v.before },
    ...v.story.map((f) => ({ label: f.label, steps: f.steps })),
    { label: WITH_LABEL, steps: v.after },
  ];
}

const samePairs = (a: [number, number][], b: [number, number][]) =>
  a === b || (a.length === b.length && a.every((p, i) => p[0] === b[i]![0] && p[1] === b[i]![1]));

const sameHist = (a: HistogramData, b: HistogramData) =>
  a === b ||
  (a.n_missing === b.n_missing &&
    a.counts.length === b.counts.length &&
    a.counts.every((c, i) => c === b.counts[i]) &&
    a.edges.every((e, i) => e === b.edges[i]));

function isStill(v: ConsequenceView): boolean {
  if (v.story.length) return false;
  switch (v.kind) {
    case "relationship":
      return v.y_label_before === v.y_label_after && samePairs(v.points_before, v.points_after);
    case "distribution":
      return sameHist(v.before, v.after);
    case "lineage":
      return v.before === null || JSON.stringify(v.before) === JSON.stringify(v.after);
    case "row_flow":
      return JSON.stringify(v.before) === JSON.stringify(v.after);
    case "table_focus":
      return v.changed.length === 0 && v.columns_before.join() === v.columns_after.join();
  }
}

export function trackOf<V extends ConsequenceView>(view: V): Track<V> {
  let states: unknown[];
  switch (view.kind) {
    case "relationship":
      states = relationshipStates(view as RelationshipView);
      break;
    case "distribution":
      states = distributionStates(view as DistributionView);
      break;
    case "lineage":
      states = lineageStates(view as LineageView);
      break;
    case "table_focus":
      states = tableStates(view as TableFocusView);
      break;
    default:
      states = rowFlowStates(view as RowFlowView);
  }
  return { view, states: states as StateOf<V>[], still: isStill(view) };
}

// ── the storyboard the player runs ───────────────────────────────────────────

export interface Storyboard {
  /** The index of "with this choice". */
  last: number;
  /** One label per real state, 0 … last. */
  labels: string[];
}

/**
 * The preview's storyboard: the longest of its views' (the primary's on a tie). Views with fewer
 * states follow it proportionally (`localPos`).
 */
export function storyboardOf(tracks: Track[]): Storyboard {
  let best: Track | null = null;
  for (const t of tracks) if (!best || t.states.length > best.states.length) best = t;
  if (!best) return { last: 1, labels: [NOW_LABEL, WITH_LABEL] };
  const labels = best.states.map((s, i, all) =>
    i === 0 ? NOW_LABEL : i === all.length - 1 ? WITH_LABEL : s.label,
  );
  return { last: Math.max(1, labels.length - 1), labels };
}

/** A global player position on a view whose own storyboard is shorter. */
export function localPos(pos: number, globalLast: number, localLast: number): number {
  if (globalLast === localLast) return pos;
  return (pos * localLast) / Math.max(1, globalLast);
}

/** The whole state a view shows for a (whole-step) heading of the global storyboard. */
export function localStep(step: number, globalLast: number, localLast: number, forward = true): number {
  const p = localPos(step, globalLast, localLast);
  return forward ? Math.ceil(p - 1e-9) : Math.floor(p + 1e-9);
}

// ── units ────────────────────────────────────────────────────────────────────

/** Suffixes and prefixes that state a column's unit changed. */
const UNIT_MARKERS: [RegExp, string][] = [
  [/(^|_)per_?kcal$|per kcal|\/\s*kcal|per 1,?000 kcal/i, "per-kcal"],
  [/^kcal_from_|kcal from /i, "kcal"],
  [/(^|_)(log2?|ln|log10)(_|\(|$)|^log/i, "log"],
  [/(^|_)(z|zscore|std)$|z-score/i, "z"],
  [/(_pct|_percent|%)$|percent of/i, "percent"],
];

export function unitMarker(label: string): string {
  for (const [re, unit] of UNIT_MARKERS) if (re.test(label)) return unit;
  return "";
}

export interface Extent {
  label: string;
  lo: number;
  hi: number;
}

/**
 * Whether two states are in the same unit, so their values may glide into each other. A unit
 * marker in the name (per kcal, kcal from, log, z, percent) that differs is a change of unit;
 * so is a range more than eight times wider or narrower, which a rescaling would produce.
 */
export function sameUnit(a: Extent, b: Extent): boolean {
  if (unitMarker(a.label) !== unitMarker(b.label)) return false;
  const sa = Math.abs(a.hi - a.lo);
  const sb = Math.abs(b.hi - b.lo);
  if (sa === 0 || sb === 0) return sa === sb;
  const ratio = sa > sb ? sa / sb : sb / sa;
  return ratio <= 8;
}

export function extentOfPoints(label: string, points: [number, number][]): Extent {
  let lo = Infinity;
  let hi = -Infinity;
  for (const [, y] of points) {
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  if (!Number.isFinite(lo)) return { label, lo: 0, hi: 0 };
  return { label, lo, hi };
}

export function extentOfHist(label: string, h: HistogramData): Extent {
  if (!h.edges.length) return { label, lo: 0, hi: 0 };
  return { label, lo: h.edges[0]!, hi: h.edges[h.edges.length - 1]! };
}

/**
 * Consecutive states in the same unit share a run (and one axis, so a glide is a real change of
 * value); a change of unit starts a new run, crossed by a crossfade.
 */
export function unitRuns(extents: Extent[]): number[] {
  const runs: number[] = [];
  let run = 0;
  extents.forEach((e, i) => {
    if (i > 0 && !sameUnit(extents[i - 1]!, e)) run++;
    runs.push(run);
  });
  return runs;
}

export type Transition = "morph" | "crossfade";

/** How a view passes from one state to the next (or from one option's state to another's). */
export function transitionBetween(a: Extent, b: Extent, sameShape = true): Transition {
  return sameShape && sameUnit(a, b) ? "morph" : "crossfade";
}

// ── the clipped tail (§ fix: a long empty tail) ──────────────────────────────

export interface Clip {
  /** Bins drawn: 0 … bins − 1. */
  bins: number;
  hi: number;
  /** Rows beyond the drawn range. */
  over: number;
  /** The largest edge, for the overflow note. */
  max: number;
}

/**
 * Draw a histogram up to the bin holding its 99.5th percentile (or its last mark), so one
 * 15,000-kcal day does not squeeze the other 21,848 into a third of the axis. The rows past the
 * cut are counted, never hidden silently.
 */
export function clipTail(h: HistogramData, keep: number[] = [], q = 0.995): Clip {
  const n = h.counts.length;
  const total = h.counts.reduce((a, c) => a + c, 0);
  const max = h.edges[n] ?? 0;
  if (n === 0 || total === 0) return { bins: n, hi: max, over: 0, max };
  let cum = 0;
  let cut = n;
  for (let i = 0; i < n; i++) {
    cum += h.counts[i]!;
    if (cum >= q * total) {
      cut = i + 1;
      break;
    }
  }
  for (const v of keep) {
    while (cut < n && h.edges[cut]! < v) cut++;
  }
  // Clip only a tail worth clipping: at least a fifth of the axis.
  if (cut >= n || (max - h.edges[cut]!) / (max - h.edges[0]!) < 0.2) {
    return { bins: n, hi: max, over: 0, max };
  }
  let over = 0;
  for (let i = cut; i < n; i++) over += h.counts[i]!;
  return { bins: cut, hi: h.edges[cut]!, over, max };
}

// ── the pinned readout (§11) ─────────────────────────────────────────────────

export interface ReadoutItem {
  key: string;
  name: string;
  before: string;
  after: string;
}

function matrixCount(l: Lineage): number {
  return l.nodes.filter((n) => n.lane === "matrix").reduce((s, n) => s + n.count, 0);
}

/** The headline numbers of a preview, as recorded → with this choice. Real values only. */
export function readoutOf(views: ConsequenceView[]): ReadoutItem[] {
  const out: ReadoutItem[] = [];
  for (const v of views) {
    if (v.kind === "relationship" && v.r_before !== null && v.r_after !== null) {
      out.push({ key: "r", name: "r", before: fmtR(v.r_before), after: fmtR(v.r_after) });
    } else if (v.kind === "row_flow") {
      const a = v.before[v.before.length - 1];
      const b = v.after.filter((s) => s.key !== "holdout").at(-1);
      if (a && b) out.push({ key: "n", name: "n", before: fmtInt(a.n), after: fmtInt(b.n) });
    } else if (v.kind === "lineage" && v.before) {
      const a = matrixCount(v.before);
      const b = matrixCount(v.after);
      out.push({ key: "cols", name: "columns", before: fmtInt(a), after: fmtInt(b) });
    }
  }
  const seen = new Set<string>();
  return out.filter((i) => (seen.has(i.key) ? false : (seen.add(i.key), true))).slice(0, 3);
}

// ── titles ───────────────────────────────────────────────────────────────────

const STOP = new Set(["the", "and", "with", "from", "into", "that", "this", "each", "rows", "than"]);

/**
 * A title says each thing once: "kcal outside sex-specific kcal ranges" reads "kcal outside
 * sex-specific ranges". Only content words (four letters or more, or data names) are deduplicated.
 */
export function noRepeats(title: string): string {
  const seen = new Set<string>();
  const words = title.split(/(\s+)/);
  const out: string[] = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i]!;
    if (/^\s+$/.test(w)) {
      out.push(w);
      continue;
    }
    const core = w.replace(/[`.,;:()]/g, "").toLowerCase();
    const content = core.length >= 4 || /[_\d]/.test(core) || core === "kcal";
    if (content && !STOP.has(core) && seen.has(core)) {
      if (out.length && /^\s+$/.test(out[out.length - 1]!)) out.pop();
      continue;
    }
    if (content) seen.add(core);
    out.push(w);
  }
  return out.join("");
}
