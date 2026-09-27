/**
 * Fixture -> view states. Pure functions, so the prototype's pictures are a function of the
 * engine's numbers and nothing else.
 */
import {
  ENERGY,
  FOCUS_NUTRIENT,
  N_TRAIN,
  fmtInt,
  fmtNum,
  formulaRhs,
  residualCenter,
  sourceOf,
  view,
  type DistributionView,
  type EnergyOption,
  type Lineage,
  type TableFocusView,
} from "../data";
import { ENERGY_NOW_NOTE, ENERGY_OPTIONS } from "../copy";
import type { StripState } from "../Stage";
import type { HistState } from "../views/Histogram";
import type { LineageState } from "../views/Lineage";
import type { HeadNote, TableState } from "../views/MorphTable";
import type { ScatterState } from "../views/Scatter";

/** The option keys that carry a preview (partition on all 7 is refused; its exit previews). */
export const ENERGY_PREVIEWED = ["residual", "density", "density_multivariate", "standard", "none", "partition3"];

const tableOf = (o: EnergyOption) => view(o.extra_views, "table_focus") as TableFocusView;

// ── relationship ─────────────────────────────────────────────────────────────

export function energyScatter(): { xs: number[]; xLabel: string; states: Record<string, ScatterState> } {
  const base = view(ENERGY.residual!.preview!.views, "relationship")!;
  const states: Record<string, ScatterState> = {
    now: {
      ys: base.points_before.map((p) => p[1]),
      yLabel: base.y_label_before,
      r: base.r_before,
      note: ENERGY_NOW_NOTE,
      after: false,
    },
  };
  for (const key of ENERGY_PREVIEWED) {
    const rel = view(ENERGY[key]!.preview!.views, "relationship")!;
    states[key] = {
      ys: rel.points_after.map((p) => p[1]),
      yLabel: rel.y_label_after,
      r: rel.r_after,
      note: ENERGY_OPTIONS[key]!.sidenote,
      after: true,
    };
  }
  return { xs: base.points_before.map((p) => p[0]), xLabel: base.x_label, states };
}

// ── the working table ────────────────────────────────────────────────────────

export const ENERGY_SLOTS = tableOf(ENERGY.residual!).columns_before;
const ROWS = 5;

/**
 * Which slot (raw column) an adjusted column comes from: the engine's own lineage says (its first
 * input), else the name does.
 */
export function slotFor(after: string, slots: string[], o?: EnergyOption): string | null {
  const first = o?.adjuster_lineage?.find((a) => a.output === after)?.inputs[0];
  if (first && slots.includes(first)) return first;
  return sourceOf(after, slots);
}

export function energyTable(): { slots: string[]; rowIds: number[]; states: Record<string, TableState> } {
  const ref = tableOf(ENERGY.residual!);
  const rows = ref.rows.slice(0, ROWS);
  const rowIds = rows.map((r) => r.row_id);
  const nowCols: TableState["cols"] = {};
  for (const slot of ENERGY_SLOTS)
    nowCols[slot] = { name: slot, values: rows.map((r) => fmtNum(r.before[slot] as number)), status: "same" };
  const states: Record<string, TableState> = { now: { cols: nowCols, notes: [] } };
  const nutrients = ENERGY_SLOTS.filter((s) => s !== "kcal");
  const first = nutrients[0]!;
  const last = nutrients[nutrients.length - 1]!;
  const center = residualCenter();

  for (const key of ENERGY_PREVIEWED) {
    const o = ENERGY[key]!;
    const tf = tableOf(o);
    const byRow = new Map(tf.rows.map((r) => [r.row_id, r]));
    const cols: TableState["cols"] = {};
    for (const slot of ENERGY_SLOTS) cols[slot] = { ...nowCols[slot]! };
    for (const slot of tf.columns_before) {
      const out = tf.columns_after.find((c) => slotFor(c, tf.columns_before, o) === slot);
      if (!out) {
        cols[slot] = { ...nowCols[slot]!, status: "dropped" };
        continue;
      }
      const values = rowIds.map((id) => fmtNum(byRow.get(id)?.after[out] as number));
      const changed = values.some((v, i) => v !== nowCols[slot]!.values[i]) || out !== slot;
      cols[slot] = { name: out, values, status: changed ? "changed" : "same" };
    }
    const notes: HeadNote[] = [];
    const dropped = (o.dropped_columns ?? []).includes("kcal");
    switch (key) {
      case "residual":
        notes.push({
          id: "n",
          text: center ? `nutrient − b × (kcal − ${fmtInt(center)})` : "nutrient − b × kcal",
          from: first,
          to: last,
          tone: "formula",
        });
        break;
      case "density":
      case "density_multivariate":
        notes.push({ id: "n", text: "nutrient ÷ kcal", from: first, to: last, tone: "formula" });
        break;
      case "standard":
      case "none":
        notes.push({ id: "same", text: "unchanged", from: first, to: last });
        break;
      case "partition3": {
        for (const a of o.adjuster_lineage ?? []) {
          const slot = slotFor(a.output, ENERGY_SLOTS, o);
          if (!slot || a.output === slot) continue;
          notes.push({ id: `p-${slot}`, text: formulaRhs(a.formula).replace(/\(.*\)/, "Σ"), from: slot, to: slot, tone: "formula" });
        }
        break;
      }
    }
    if (dropped) notes.push({ id: "k", text: "leaves the model", from: "kcal", to: "kcal" });
    else if (key === "density_multivariate") notes.push({ id: "k", text: "own term", from: "kcal", to: "kcal" });
    else if (key === "standard") notes.push({ id: "k", text: "beside them", from: "kcal", to: "kcal" });
    states[key] = { cols, notes };
  }
  return { slots: ENERGY_SLOTS, rowIds, states };
}

// ── pipeline strip ───────────────────────────────────────────────────────────

export function energyStrip(): Record<string, StripState> {
  const nowCols = ENERGY.none!.matrix_columns!.length;
  const mk = (cols: number, touched: boolean): StripState => ({
    rows: { value: fmtInt(N_TRAIN), sub: "training rows" },
    columns: { value: String(cols), sub: "enter the model" },
    results: { value: "—", sub: "not fitted yet" },
    touched: touched ? ["columns"] : [],
  });
  const out: Record<string, StripState> = { now: mk(nowCols, false) };
  for (const key of ENERGY_PREVIEWED) {
    const o = ENERGY[key]!;
    const changed = (o.extra_views[0] as TableFocusView).n_affected_columns > 0;
    out[key] = mk(o.matrix_columns!.length, changed);
  }
  return out;
}

// ── distribution & lineage (the other two views of the preview) ─────────────

export function energyDistribution(): Record<string, HistState> {
  const d0 = view(ENERGY.residual!.preview!.views, "distribution") as DistributionView;
  const table = tableOf(ENERGY.residual!);
  const rows = table.rows.slice(0, ROWS);
  const out: Record<string, HistState> = {
    now: {
      edges: d0.before.edges,
      counts: d0.before.counts,
      domain: `before:${d0.before_label}`,
      label: d0.before_label,
      marks: [],
      after: false,
      rug: rows.map((r) => ({ row: r.row_id, value: r.before[FOCUS_NUTRIENT] as number })),
    },
  };
  for (const key of ENERGY_PREVIEWED) {
    const o = ENERGY[key]!;
    const d = view(o.preview!.views, "distribution") as DistributionView;
    const tf = tableOf(o);
    const col = o.focus_after_column ?? FOCUS_NUTRIENT;
    const same = d.after_label.includes("unchanged");
    out[key] = {
      edges: d.after.edges,
      counts: d.after.counts,
      domain: same ? `before:${d0.before_label}` : `${key}:${d.after_label}`,
      label: same ? d0.before_label : d.after_label,
      marks: [],
      after: true,
      note: d.caption,
      rug: rows.map((r) => {
        const hit = tf.rows.find((x) => x.row_id === r.row_id);
        const v = (hit?.after[col] ?? hit?.before[FOCUS_NUTRIENT] ?? r.before[FOCUS_NUTRIENT]) as number;
        return { row: r.row_id, value: v };
      }),
    };
  }
  return out;
}

export function energyLineage(): { states: Record<string, LineageState>; sources: string[] } {
  const lv = view(ENERGY.residual!.preview!.views, "lineage")!;
  const states: Record<string, LineageState> = {
    now: { lineage: lv.before as Lineage, emphasis: [], after: false },
  };
  for (const key of ENERGY_PREVIEWED) {
    const v = view(ENERGY[key]!.preview!.views, "lineage")!;
    states[key] = { lineage: v.after, emphasis: v.emphasis, after: true };
  }
  const sources = (lv.before as Lineage).nodes.filter((n) => n.lane === "raw").map((n) => n.label);
  return { states, sources };
}

export const ENERGY_BASIS = ENERGY.residual!.preview!.basis;
