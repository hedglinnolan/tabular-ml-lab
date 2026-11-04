/**
 * First look's guide and gallery as data, pure (FOUNDATION §3 stage 3, "a guide plus a gallery";
 * §5 rule 6, the outcome's own views open at their gates and each is recorded; §10 rule 3;
 * understanding/research/FIRST_LOOK_BRIEF.md §1–§6). Nothing here decides what may be shown: each
 * gate's state is read from what the server served for it, and each record posted is the one the
 * server handed back.
 *
 *   columnLooks   every column but the outcome, in the table's order, by how it is drawn
 *   outcomeDoor   the outcome alone, from GET /columns: shut (the server's line), the record to
 *                 post, or open on the rows it reads
 *   rowsDoor      the outcome beside the other columns, from a GET /table window: the same three
 *   exploreLooks  the explore stage's findings: the outcome's relationships (each opened on its
 *                 own and drawn only once recorded, though Explore serves their points before
 *                 then), its distribution (its door is GET /columns'), and the outcome-free looks,
 *                 each with the lever options "Decide now" can record
 *   beats         the card's few beats, one per gallery section
 *   gateMark, tableMark   what each read depends on, so a changed answer reads it again
 *   barsOf, curveOf       the drawings' scale math
 */
import type { ColumnSummary, Decision, Histogram, ProjectState, ProjectView, QuestLog, TableWindow } from "../../api/schema";
import { fmtInt } from "../../util/format";

// ── the explore stage's artifact, as served ────────────────────────────────────

/**
 * GET /stages/explore's artifact (turbotab/core/stages/explore.py `ExploreArtifact`). The contract
 * has the type, but src/api's stage map does not list the stage yet, so the fields read here are
 * spelled here; model.test.ts checks them against the contract's.
 */
export interface ExploreArtifact {
  purpose: string | null;
  /** "training" under Predict, "analyzed" otherwise */
  rows: string;
  n_rows: number;
  n_holdout: number;
  findings: ExploreFinding[];
  /** continuous predictors beyond the relationships computed */
  more: number;
  proposed_subgroups: string[];
  /** the outcome views recorded as looked at, by key (`relationship:age`) */
  viewed: string[];
  sentence: string;
}

export interface ExploreFinding {
  id: string;
  kind: string;
  summary: string;
  columns: string[];
  /** it shows the outcome, so opening it is recorded */
  outcome_view: boolean;
  viewed: boolean;
  /** relationship · distribution · table_focus */
  view: string;
  points: BinPoint[];
  groups: GroupQuality[];
  detail: string | null;
  lever: Lever | null;
  /** the `view_outcome` to post on opening it; null where nothing is served to open */
  record: { [key: string]: unknown } | null;
  label: string | null;
}

export interface BinPoint {
  x: number;
  y: number | null;
  n: number;
}

export interface GroupQuality {
  group: string;
  n: number;
  missing_share: number;
}

export interface Lever {
  question: string;
  options: LeverOption[];
}

export interface LeverOption {
  key: string;
  label: string;
  customary: string;
  sound: string;
  rung: string;
  in_fold: boolean;
  decision: { [key: string]: unknown } | null;
}

// ── sizes ─────────────────────────────────────────────────────────────────────

/** The most columns the gallery draws, in the table's order (a request each). */
export const GALLERY_MAX = 40;
/** A category's levels drawn, most frequent first (the summary's own order). */
export const LEVELS_SHOWN = 6;
/** The rows the window of the outcome beside the other columns shows. */
export const ROWS_SHOWN = 8;
/** The columns beside the outcome in that window, in the table's order. */
export const COLUMNS_BESIDE = 11;

// ── every column but the outcome ─────────────────────────────────────────────

export type ColumnDraw =
  | { kind: "histogram" }
  | { kind: "levels"; levels: { label: string; count: number }[]; more: number }
  | { kind: "line"; line: string };

export interface ColumnLook {
  name: string;
  /** the column's kind, in plain words */
  kind: string;
  /** rows with a value */
  n: number;
  /** rows blank */
  blank: number;
  draw: ColumnDraw;
}

const KIND: Record<ColumnSummary["dtype"], string> = {
  numeric: "numbers",
  integer: "whole numbers",
  categorical: "categories",
  boolean: "true or false",
  datetime: "dates",
  text: "text",
};

/** Each column but the outcome, in the order served: a number by its histogram (GET /histogram,
 *  which draws numeric and integer columns only), a category by its most frequent levels, an
 *  identifier, a date or text by one line. */
export function columnLooks(columns: readonly ColumnSummary[], state: Pick<ProjectState, "target" | "roles"> | null): ColumnLook[] {
  const target = state?.target ?? null;
  const roles = state?.roles ?? {};
  return columns
    .filter((c) => c.name !== target)
    .map((c) => {
      let draw: ColumnDraw;
      if (roles[c.name] === "identifier") draw = { kind: "line", line: `An identifier: ${fmtInt(c.n_unique)} distinct values` };
      else if (c.dtype === "numeric" || c.dtype === "integer") draw = { kind: "histogram" };
      else if ((c.dtype === "categorical" || c.dtype === "boolean") && c.top?.length) {
        const top = c.top.slice(0, LEVELS_SHOWN);
        draw = { kind: "levels", levels: top.map((t) => ({ label: String(t.value), count: t.count })), more: Math.max(0, c.n_unique - top.length) };
      } else draw = { kind: "line", line: `${fmtInt(c.n_unique)} distinct values` };
      return { name: c.name, kind: KIND[c.dtype], n: c.n, blank: c.n_missing, draw };
    });
}

// ── the outcome's doors ──────────────────────────────────────────────────────

export type Door =
  | { kind: "shut"; line: string }
  | { kind: "record"; line: string; record: Decision }
  | { kind: "open"; rows: "training" | "analyzed" | null };

/** The outcome alone, as GET /columns serves its summary: withheld with the server's line before
 *  its gate (core/outcome_gate.py `alone_gate`), with the `view_outcome` to post once open and not
 *  recorded for the rows it reads, else served on those rows, which it names. A summary that says
 *  neither (no gate served) is not taken as open. */
export function outcomeDoor(summary: ColumnSummary): Door {
  if (summary.withheld) {
    return summary.record
      ? { kind: "record", line: summary.withheld, record: summary.record as unknown as Decision }
      : { kind: "shut", line: summary.withheld };
  }
  if (!summary.rows) return { kind: "shut", line: "The server has not said that the outcome is open." };
  return { kind: "open", rows: summary.rows };
}

export type RowsDoor =
  | { kind: "shut"; line: string }
  | { kind: "record"; line: string; record: Decision }
  | { kind: "open"; note: string | null };

/** The outcome beside the other columns, as a GET /table window serves it: left out with its
 *  line before its gate (`beside_gate`), with the record to post once open and not recorded, else
 *  in the window, with any line on rows left blank or its label (`EXPLORATORY`). */
export function rowsDoor(window: TableWindow, target: string): RowsDoor {
  if (window.columns.includes(target)) return { kind: "open", note: window.withheld[target] ?? window.labels[target] ?? null };
  const line = window.withheld[target] ?? "";
  return window.record ? { kind: "record", line, record: window.record as unknown as Decision } : { kind: "shut", line };
}

// ── Explore's looks ──────────────────────────────────────────────────────────

export type LookState =
  | { kind: "free" }
  | { kind: "waits"; line: string | null }
  | { kind: "record"; record: Decision }
  | { kind: "drawn" };

export interface Look {
  finding: ExploreFinding;
  state: LookState;
  /** "Decide now": the lever's options that record a decision */
  options: LeverOption[];
}

export interface Looks {
  relationships: Look[];
  /** The outcome's distribution: its door is GET /columns' (its record names the rows it reads),
   *  so only its lever is read from here. */
  distribution: ExploreFinding | null;
  free: Look[];
  more: number;
}

/** Whether an outcome view is recorded as looked at for the outcome chosen now: the folded log's
 *  entry for it names this outcome, or none (explore.py `_viewed`). */
export function recordedLook(state: Pick<ProjectState, "target" | "outcome_views">, key: string): boolean {
  const v = state.outcome_views?.[key];
  return !!v && (v.target == null || v.target === state.target);
}

export const decideNow = (f: ExploreFinding): LeverOption[] => (f.lever?.options ?? []).filter((o) => o.decision !== null);

/**
 * The explore stage's findings as the gallery shows them. An outcome view waits where the server
 * serves nothing to open (its points withheld, its record null: before the lock under Estimate,
 * before a fresh draw under Predict), says why in the server's words, and offers nothing to post;
 * it is drawn only once recorded as looked at, by the log (the folded state) or by the artifact.
 */
export function exploreLooks(artifact: ExploreArtifact, state: Pick<ProjectState, "target" | "outcome_views">): Looks {
  const relationships: Look[] = [];
  const free: Look[] = [];
  let distribution: ExploreFinding | null = null;
  for (const f of artifact.findings) {
    if (f.kind === "outcome_distribution") {
      distribution = f;
      continue;
    }
    if (!f.outcome_view) {
      free.push({ finding: f, state: { kind: "free" }, options: decideNow(f) });
      continue;
    }
    const looked = f.viewed || f.columns.every((c) => recordedLook(state, `${f.view}:${c}`));
    const look: LookState =
      !f.points.length || !f.record
        ? { kind: "waits", line: f.detail }
        : looked
          ? { kind: "drawn" }
          : { kind: "record", record: f.record as unknown as Decision };
    relationships.push({ finding: f, state: look, options: decideNow(f) });
  }
  return { relationships, distribution, free, more: artifact.more };
}

// ── the card's beats ─────────────────────────────────────────────────────────

export type ExploreBeat =
  | { kind: "fresh"; looks: number; decidable: number; rows: string; n_rows: number; n_holdout: number }
  | { kind: "waits"; stage: string }
  | { kind: "running" }
  | { kind: "error"; error: string | null }
  | { kind: "none" };

export interface Beat {
  key: "columns" | "noticed" | "explore" | "outcome";
  title: string;
  line: string;
}

export interface BeatsInput {
  /** columns but the outcome */
  columns: number;
  /** of them, drawn now */
  drawn: number;
  /** noticings, each decided in another stage (the card lists them) */
  noticed: number;
  target: string | null;
  door: Door | null;
  explore: ExploreBeat;
}

const plural = (n: number, one: string, many = `${one}s`) => `${fmtInt(n)} ${n === 1 ? one : many}`;
export const rowsWords = (rows: "training" | "analyzed" | null) =>
  rows === "training" ? "training rows" : rows === "analyzed" ? "rows analyzed" : "rows it reads";

/** The gallery section a beat brings into view. */
export const sectionId = (key: string) => `first-look-${key}`;

/** Each beat's name, which its gallery section carries too. */
export const TITLE = { columns: "Each column, as it is", noticed: "What was noticed", explore: "What Explore found" } as const;
export const outcomeTitle = (target: string) => `The outcome, ${target}`;

/** What Explore read and found, or why it has not yet. */
export function exploreLine(e: ExploreBeat): string {
  if (e.kind === "fresh") {
    let said = `${plural(e.looks, "look")} on the ${fmtInt(e.n_rows)} ${e.rows} rows`;
    if (e.n_holdout) said += `; the ${fmtInt(e.n_holdout)} held-out rows are never read`;
    said += ".";
    if (e.decidable) said += ` ${fmtInt(e.decidable)} can be decided now.`;
    return said;
  }
  if (e.kind === "waits") return `Explore has not run yet: it reads the rows ${e.stage} keeps.`;
  if (e.kind === "running") return "Explore is reading the rows.";
  if (e.kind === "error") return e.error ? `Explore did not finish: ${e.error}` : "Explore did not finish.";
  return "";
}

/** The card's beats, in the gallery's order: each column, what was noticed (listed on the card),
 *  what Explore found, and the outcome last, behind its gate (FIRST_LOOK_BRIEF §1.2, §6.2). */
export function beats(input: BeatsInput): Beat[] {
  const out: Beat[] = [];
  const { columns, drawn } = input;
  let line = `${plural(columns, "column")}, each drawn as it is on every row of the table.`;
  if (drawn < columns) line += ` The first ${fmtInt(drawn)} are drawn here.`;
  out.push({ key: "columns", title: TITLE.columns, line });
  if (input.noticed > 0) {
    out.push({
      key: "noticed",
      title: TITLE.noticed,
      line: `${plural(input.noticed, "noticing")}, each decided in the stage whose answer it changes, listed below.`,
    });
  }
  if (input.explore.kind !== "none") out.push({ key: "explore", title: TITLE.explore, line: exploreLine(input.explore) });
  if (input.target && input.door) {
    const d = input.door;
    out.push({
      key: "outcome",
      title: outcomeTitle(input.target),
      line: d.kind === "open" ? `Open on the ${rowsWords(d.rows)}; each view of it is recorded as looked at.` : d.line,
    });
  }
  return out;
}


/** The noticings First look lists, each decided in the stage whose answer it changes. */
export const noticedCount = (log: QuestLog): number => log.stages.reduce((n, s) => n + s.lines.filter((l) => l.source === "finding").length, 0);

// ── what each read depends on ────────────────────────────────────────────────

/**
 * What the outcome's gates read (core/outcome_gate.py; the service's `_outcome_state`,
 * `_outcome_rows` and Fit's press): the decision log, each stage's state, and whether Fit was
 * pressed. The outcome's reads are keyed by it, so a gate that opens, or a look recorded, is read
 * again rather than shown as it stood; a progress tick changes nothing.
 */
export function gateMark(view: Pick<ProjectView, "decisions" | "stages">, log: Pick<QuestLog, "fit">): string {
  const seq = view.decisions.reduce((m, d) => Math.max(m, d.seq), 0);
  const stages = Object.keys(view.stages)
    .sort()
    .map((n) => `${n}:${view.stages[n]?.status ?? ""}:${view.stages[n]?.key ?? ""}`)
    .join(",");
  return `${view.decisions.length}:${seq}|${log.fit?.pressed ? "pressed" : ""}|${stages}`;
}

/** The table the data routes read (the service's `store`: the working table, else the oriented
 *  one, else the raw file), by those stages' states and keys: each column's histogram is read
 *  again when it changes, as events.ts reads the columns again. */
export function tableMark(view: Pick<ProjectView, "stages">): string {
  return ["ingest", "oriented", "working"].map((n) => `${view.stages[n]?.status ?? ""}:${view.stages[n]?.key ?? ""}`).join("|");
}

// ── the drawings' scale math ─────────────────────────────────────────────────

export interface Box {
  width: number;
  height: number;
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Bar {
  x: number;
  w: number;
  /** the bar's top, and its height above the baseline */
  y: number;
  h: number;
  from: number;
  to: number;
  count: number;
}

/** A histogram's bars: each bin over its edges, its height in proportion to its count, the
 *  tallest filling the plot. */
export function barsOf(h: Pick<Histogram, "edges" | "counts">, box: Box): Bar[] {
  const { edges, counts } = h;
  if (edges.length < 2 || counts.length !== edges.length - 1) return [];
  const lo = edges[0]!;
  const hi = edges[edges.length - 1]!;
  const x0 = box.left;
  const x1 = box.width - box.right;
  const base = box.height - box.bottom;
  const room = base - box.top;
  const top = Math.max(...counts);
  const at = (v: number) => (hi > lo ? x0 + ((v - lo) / (hi - lo)) * (x1 - x0) : x0);
  return counts.map((count, i) => {
    const from = edges[i]!;
    const to = edges[i + 1]!;
    const h = top > 0 ? (count / top) * room : 0;
    const x = at(from);
    return { x, w: Math.max(0, at(to) - x - 1), y: base - h, h, from, to, count };
  });
}

export interface CurvePoint {
  cx: number;
  cy: number;
  x: number;
  y: number;
  n: number;
}

/** A relationship's points (each tenth of the predictor: its median value, the outcome's mean or
 *  the event's share there, its rows) in the plot, higher outcomes higher; a tenth with no value
 *  is left out. */
export function curveOf(points: readonly BinPoint[], box: Box): CurvePoint[] {
  const kept = points.filter((p): p is BinPoint & { y: number } => p.y !== null && Number.isFinite(p.y));
  if (!kept.length) return [];
  const xs = kept.map((p) => p.x);
  const ys = kept.map((p) => p.y);
  const [xlo, xhi] = [Math.min(...xs), Math.max(...xs)];
  const [ylo, yhi] = [Math.min(...ys), Math.max(...ys)];
  const x0 = box.left;
  const x1 = box.width - box.right;
  const y0 = box.top;
  const y1 = box.height - box.bottom;
  const sx = (v: number) => (xhi > xlo ? x0 + ((v - xlo) / (xhi - xlo)) * (x1 - x0) : (x0 + x1) / 2);
  const sy = (v: number) => (yhi > ylo ? y1 - ((v - ylo) / (yhi - ylo)) * (y1 - y0) : (y0 + y1) / 2);
  return kept.map((p) => ({ cx: sx(p.x), cy: sy(p.y), x: p.x, y: p.y, n: p.n }));
}
