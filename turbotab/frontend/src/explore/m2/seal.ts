/**
 * The seal's storyboard geometry — pure. Every row of the table is one cell; a cell keeps its
 * identity from the file to its side of the seal (DESIGN_LANGUAGE §05.2). The states are the
 * draw's own steps (turbotab.engine.draw_holdout): group by the unit when the grain names one,
 * order by time when the split is chronological, draw, partition, seal.
 *
 * A unit the draw split across the seal (grouping abandoned) keeps a hole where its held-out row
 * was: the leak is drawn, not described.
 */
import type { SealVariant } from "./types";

export type Phase = "flat" | "stacked" | "ordered" | "drawn" | "split" | "sealed";

export interface SealGeom {
  x: Float32Array;
  y: Float32Array;
  /** 0..1: marked as held out (drawn) */
  mark: Float32Array;
  /** 0..1: placed in its lane (training teal, held out sealed) */
  placed: Float32Array;
  boundary: number; // 0..1 opacity of the chronological boundary
  boxes: number; // 0..1 opacity of the two lanes' frames
  sealed: number; // 0..1 the seal closed
  straddle: number; // 0..1 the lines from a split unit's hole to its held-out row
}

export interface SealPlan {
  phases: Phase[];
  labels: string[];
  cell: number;
  gap: number;
  states: SealGeom[];
  height: number;
  /** The partitioned lanes' own height (the frames end here, not at the tallest state). */
  splitHeight: number;
  width: number;
  /** The two lanes once partitioned: [x0, x1] each. */
  train: [number, number];
  held: [number, number];
  top: number;
  /** Chronological: x of the boundary in the ordered phase, and the time axis. */
  boundaryX: number | null;
  axis: { x: number; label: string }[];
  /** Pairs of cells (row indices) a straddle line joins, drawn from the hole to the held-out row. */
  straddles: [number, number][];
  /** Where a split unit's missing rows sat in the training lane (the holes), by held-out row. */
  holes: Map<number, [number, number]>;
}

const TOP = 30;

function sizeFor(n: number): [number, number] {
  if (n <= 40) return [18, 4];
  if (n <= 320) return [10, 2];
  return [7, 2];
}

export function phasesOf(v: SealVariant): { phases: Phase[]; labels: string[] } {
  const g = v.group_column ? `\`${v.group_column}\`` : "";
  const pct = `${Math.round(v.fraction * 100)}%`;
  switch (v.key) {
    case "grouped":
      return {
        phases: ["flat", "drawn", "split", "sealed"],
        labels: [
          "",
          `Draw ${pct} of ${v.unit_noun} at random`,
          `${v.n_hold_rows} ${v.unit_noun} to the held-out side`,
          "",
        ],
      };
    case "chronological":
      return {
        phases: ["flat", "stacked", "ordered", "split", "sealed"],
        labels: [
          "",
          `Group ${v.row_noun} by ${g}`,
          `Order ${v.unit_noun} by their last ${v.time_column ?? "date"}`,
          `The ${v.n_hold_units} seen last are held out`,
          "",
        ],
      };
    case "abandoned":
      return {
        phases: ["flat", "stacked", "drawn", "split", "sealed"],
        labels: [
          "",
          `Group ${v.row_noun} by ${g}`,
          `${v.n_units} ${v.unit_noun}: too few, so draw ${v.n_hold_rows} ${v.row_noun}`,
          `${v.straddle} ${v.unit_noun} land on both sides`,
          "",
        ],
      };
    default:
      return {
        phases: ["flat", "drawn", "split", "sealed"],
        labels: ["", `Draw ${pct} of ${v.row_noun} at random`, "Rows to the held-out side", ""],
      };
  }
}

export function plan(v: SealVariant, width: number): SealPlan {
  const n = v.n_rows;
  const [c, g] = sizeFor(n);
  const step = c + g;
  const W = Math.max(240, width);
  const { phases, labels } = phasesOf(v);
  const known = v.group_column !== null;
  const unitOf = v.row_unit;
  const nUnits = known ? (v.n_units ?? n) : n;
  const members: number[][] = Array.from({ length: nUnits }, () => []);
  unitOf.forEach((u, i) => members[u]!.push(i));
  const P = Math.max(...members.map((m) => m.length));
  const stackH = P * step;
  const hold = v.hold;

  const trainX: [number, number] = [0, Math.round(W * 0.58)];
  const heldX: [number, number] = [Math.round(W * 0.64), W];

  // ── per-phase positions ──
  const flat = () => {
    const L = Math.max(1, Math.floor(W / step));
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = (i % L) * step;
      y[i] = TOP + Math.floor(i / L) * step;
    }
    return { x, y, h: TOP + Math.ceil(n / L) * step };
  };
  const stacks = (order: number[], x0: number, x1: number, skip?: (row: number) => boolean) => {
    const L = Math.max(1, Math.floor((x1 - x0) / step));
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const lineH = P > 1 ? stackH + 6 : step;
    order.forEach((u, j) => {
      const col = j % L;
      const line = Math.floor(j / L);
      members[u]!.forEach((row, k) => {
        if (skip?.(row)) return;
        x[row] = x0 + col * step;
        y[row] = TOP + line * lineH + k * step;
      });
    });
    return { x, y, h: TOP + Math.ceil(order.length / L) * lineH, L, lineH };
  };
  const cells = (rows: number[], x0: number, x1: number) => {
    const L = Math.max(1, Math.floor((x1 - x0) / step));
    const x = new Map<number, [number, number]>();
    rows.forEach((row, j) => x.set(row, [x0 + (j % L) * step, TOP + Math.floor(j / L) * step]));
    return { pos: x, h: TOP + Math.ceil(rows.length / L) * step };
  };

  const allUnits = Array.from({ length: nUnits }, (_, u) => u);
  const f = flat();
  const st = known ? stacks(allUnits, 0, W) : f;

  // Chronological: units by last-seen time, piled where they share a column of the axis.
  let ordered: { x: Float32Array; y: Float32Array; h: number } | null = null;
  let boundaryX: number | null = null;
  const axis: { x: number; label: string }[] = [];
  if (v.unit_time && known) {
    const tMax = Math.max(...v.unit_time);
    const cols = Math.max(1, Math.floor(W / step) - 1);
    const colOf = (t: number) => Math.min(cols, Math.round((t / Math.max(1, tMax)) * cols));
    const pile = new Map<number, number>();
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const byTime = [...allUnits].sort((a, b) => v.unit_time![a]! - v.unit_time![b]! || a - b);
    let maxLevel = 0;
    for (const u of byTime) {
      const col = colOf(v.unit_time[u]!);
      const level = pile.get(col) ?? 0;
      pile.set(col, level + 1);
      maxLevel = Math.max(maxLevel, level + 1);
      members[u]!.forEach((row, k) => {
        x[row] = col * step;
        y[row] = k * step - level * (stackH + 2);
      });
    }
    const base = TOP + maxLevel * (stackH + 2);
    for (let i = 0; i < n; i++) y[i] = y[i]! + base - stackH;
    ordered = { x, y, h: base + 4 };
    // The boundary: between the last training unit and the first held-out one.
    const heldUnits = byTime.filter((u) => members[u]!.some((r) => hold[r] === 1));
    const firstHeld = heldUnits[0];
    if (firstHeld !== undefined) boundaryX = colOf(v.unit_time[firstHeld]!) * step - g / 2 - 1;
    if (v.time_start && v.time_end) {
      axis.push({ x: 0, label: v.time_start });
      axis.push({ x: cols * step, label: v.time_end });
      if (v.boundary && boundaryX !== null) axis.push({ x: boundaryX, label: v.boundary });
    }
  }

  // Partition: training on the left, held out on the right.
  const trainUnits = allUnits.filter((u) => members[u]!.some((r) => hold[r] === 0));
  const heldRows = Array.from({ length: n }, (_, i) => i).filter((i) => hold[i] === 1);
  const sx = new Float32Array(n);
  const sy = new Float32Array(n);
  let splitH: number;
  const holes = new Map<number, [number, number]>();
  const straddles: [number, number][] = [];
  if (known && v.basis === "grouped") {
    const heldUnits = allUnits.filter((u) => members[u]!.every((r) => hold[r] === 1));
    const tr = stacks(trainUnits, trainX[0], trainX[1]);
    const he = stacks(heldUnits, heldX[0], heldX[1]);
    for (let i = 0; i < n; i++) {
      const src = hold[i] === 1 ? he : tr;
      sx[i] = src.x[i]!;
      sy[i] = src.y[i]!;
    }
    splitH = Math.max(tr.h, he.h);
  } else if (known) {
    // Grouping abandoned: units stay stacked on the training side with holes where a row was drawn.
    const tr = stacks(trainUnits, trainX[0], trainX[1]);
    const holdTrainSlot = stacks(trainUnits, trainX[0], trainX[1]);
    const he = cells(heldRows, heldX[0], heldX[1]);
    for (let i = 0; i < n; i++) {
      if (hold[i] === 1) {
        const p = he.pos.get(i)!;
        sx[i] = p[0];
        sy[i] = p[1];
        holes.set(i, [holdTrainSlot.x[i]!, holdTrainSlot.y[i]!]);
        const mate = members[unitOf[i]!]!.find((r) => hold[r] === 0);
        if (mate !== undefined) straddles.push([i, mate]);
      } else {
        sx[i] = tr.x[i]!;
        sy[i] = tr.y[i]!;
      }
    }
    splitH = Math.max(tr.h, he.h);
  } else {
    const tr = cells(
      Array.from({ length: n }, (_, i) => i).filter((i) => hold[i] === 0),
      trainX[0],
      trainX[1],
    );
    const he = cells(heldRows, heldX[0], heldX[1]);
    for (let i = 0; i < n; i++) {
      const p = (hold[i] === 1 ? he.pos : tr.pos).get(i)!;
      sx[i] = p[0];
      sy[i] = p[1];
    }
    splitH = Math.max(tr.h, he.h);
  }

  const zeros = () => new Float32Array(n);
  const ones = () => new Float32Array(n).fill(1);
  const marks = () => Float32Array.from(hold);
  const states: SealGeom[] = phases.map((ph) => {
    const base = { boundary: 0, boxes: 0, sealed: 0, straddle: 0 };
    switch (ph) {
      case "flat":
        return { x: f.x, y: f.y, mark: zeros(), placed: zeros(), ...base };
      case "stacked":
        return { x: st.x, y: st.y, mark: zeros(), placed: zeros(), ...base };
      case "ordered":
        return { x: ordered!.x, y: ordered!.y, mark: marks(), placed: zeros(), ...base, boundary: 1 };
      case "drawn": {
        const src = phases.includes("stacked") ? st : f;
        return { x: src.x, y: src.y, mark: marks(), placed: zeros(), ...base, straddle: straddles.length ? 1 : 0 };
      }
      case "split":
        return { x: sx, y: sy, mark: marks(), placed: ones(), ...base, boxes: 1, straddle: straddles.length ? 1 : 0 };
      case "sealed":
        return { x: sx, y: sy, mark: marks(), placed: ones(), ...base, boxes: 1, sealed: 1, straddle: straddles.length ? 1 : 0 };
    }
  });

  const height = Math.max(f.h, st.h, ordered?.h ?? 0, splitH) + 8;
  return {
    phases,
    labels,
    cell: c,
    gap: g,
    states,
    height,
    splitHeight: splitH,
    width: W,
    train: trainX,
    held: heldX,
    top: TOP,
    boundaryX,
    axis,
    straddles,
    holes,
  };
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export function lerpGeom(a: SealGeom, b: SealGeom, t: number): SealGeom {
  const n = a.x.length;
  const out: SealGeom = {
    x: new Float32Array(n),
    y: new Float32Array(n),
    mark: new Float32Array(n),
    placed: new Float32Array(n),
    boundary: mix(a.boundary, b.boundary, t),
    boxes: mix(a.boxes, b.boxes, t),
    sealed: mix(a.sealed, b.sealed, t),
    straddle: mix(a.straddle, b.straddle, t),
  };
  for (let i = 0; i < n; i++) {
    out.x[i] = mix(a.x[i]!, b.x[i]!, t);
    out.y[i] = mix(a.y[i]!, b.y[i]!, t);
    out.mark[i] = mix(a.mark[i]!, b.mark[i]!, t);
    out.placed[i] = mix(a.placed[i]!, b.placed[i]!, t);
  }
  return out;
}
