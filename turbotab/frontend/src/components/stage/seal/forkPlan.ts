/**
 * The seal's storyboard geometry (lifted from /lab/m2's seal.ts) — pure. Every drawn row is one
 * cell; a cell keeps its identity from the file to its side of the seal (DESIGN_LANGUAGE §05.2).
 * The states are the draw's own steps (turbotab/core/seal.py): group by the unit when the grain
 * names one, order by time when the split is chronological, draw, partition, seal.
 *
 * A unit the draw split across the seal (grouping abandoned) keeps a hole where its held-out row
 * was, joined to that row: the leak is drawn, not described.
 *
 * The server sends the cells (`SealCells`: whole units in file order, at most 600 rows) with the
 * counts over every row, so the picture reads the same at 300 rows or 300,000.
 */
import type { SealCells } from "../../../api/m2-stage-types";

export type Phase = "flat" | "stacked" | "ordered" | "drawn" | "split" | "sealed";
export type ForkKey = "grouped" | "chronological" | "abandoned" | "undetermined" | "one_row_per_unit";

export interface SealGeom {
  x: Float32Array;
  y: Float32Array;
  /** 0..1: marked as held out (drawn). */
  mark: Float32Array;
  /** 0..1: placed in its lane. */
  placed: Float32Array;
  boundary: number;
  boxes: number;
  sealed: number;
  straddle: number;
}

export interface ForkPlan {
  key: ForkKey;
  phases: Phase[];
  labels: string[];
  cell: number;
  gap: number;
  states: SealGeom[];
  height: number;
  /** The partitioned lanes' own height (the frames end here). */
  splitHeight: number;
  width: number;
  train: [number, number];
  held: [number, number];
  top: number;
  boundaryX: number | null;
  axis: { x: number; label: string }[];
  /** Pairs of cells (held-out row, a training row of the same unit) a straddle line joins. */
  straddles: [number, number][];
  /** Where a split unit's held-out row sat among its unit's rows on the training side. */
  holes: Map<number, [number, number]>;
}

const TOP = 30;

export function forkKey(c: SealCells): ForkKey {
  if (c.chronological && c.unit_time) return "chronological";
  return c.state;
}

/** Whether the cells come in known units (the picture groups them). */
function unitsKnown(c: SealCells): boolean {
  return c.unit !== null && (c.state === "grouped" || c.state === "abandoned");
}

function sizeFor(n: number): [number, number] {
  if (n <= 40) return [18, 4];
  if (n <= 320) return [10, 2];
  return [7, 2];
}

const plain = (t: string) => t.replace(/`/g, "");

/** The draw's own steps and their labels (≤ 8 words each); the first is the data as it is. */
export function phasesOf(c: SealCells): { phases: Phase[]; labels: string[] } {
  const col = c.column ?? "the unit";
  const time = c.time_column ?? "time";
  switch (forkKey(c)) {
    case "grouped":
      return {
        phases: ["flat", "drawn", "split", "sealed"],
        labels: ["", `Draw whole ${col} units at random`, `${c.n_holdout_units ?? 0} units to the held-out side`, `Sealed: grouped by ${col}`],
      };
    case "chronological":
      return unitsKnown(c)
        ? {
            phases: ["flat", "stacked", "ordered", "split", "sealed"],
            labels: [
              "",
              `Group rows by ${col}`,
              `Order units by their last ${time}`,
              `The ${c.n_holdout_units ?? 0} units seen last held out`,
              `Sealed: by time, grouped by ${col}`,
            ],
          }
        : {
            phases: ["flat", "ordered", "split", "sealed"],
            labels: ["", `Order rows by ${time}`, `The latest ${c.n_holdout} rows held out`, "Sealed: by time, by row"],
          };
    case "abandoned":
      return {
        phases: ["flat", "stacked", "drawn", "split", "sealed"],
        labels: [
          "",
          `Group rows by ${col}`,
          `Draw rows, not whole ${col} units`,
          `${c.straddle ?? 0} units land on both sides`,
          "Sealed by row: grouping abandoned",
        ],
      };
    case "one_row_per_unit":
      return {
        phases: ["flat", "drawn", "split", "sealed"],
        labels: ["", "Draw rows at random", "Rows to the held-out side", "Sealed: one row per unit"],
      };
    default:
      return {
        phases: ["flat", "drawn", "split", "sealed"],
        labels: ["", "Draw rows at random", "Rows to the held-out side", "Sealed by row: basis undetermined"],
      };
  }
}

/** The basis as the seal names it: "grouped by `participant_id`" and the rest. */
export function basisText(c: SealCells): string {
  if (forkKey(c) === "chronological") return c.unit && c.column ? `chronological, grouped by \`${c.column}\`` : "chronological, by row";
  return c.label;
}

export function forkPlan(c: SealCells, width: number): ForkPlan {
  const n = c.hold.length;
  const [cell, gap] = sizeFor(n);
  const step = cell + gap;
  const W = Math.max(240, width);
  const key = forkKey(c);
  const { phases, labels } = phasesOf(c);
  const known = unitsKnown(c);
  const unitOf = c.unit ?? Array.from({ length: n }, (_, i) => i);
  const nUnits = n ? Math.max(...unitOf) + 1 : 0;
  const members: number[][] = Array.from({ length: nUnits }, () => []);
  unitOf.forEach((u, i) => members[u]!.push(i));
  const P = Math.max(1, ...members.map((m) => m.length));
  const stackH = P * step;
  const hold = c.hold;

  const trainX: [number, number] = [0, Math.round(W * 0.58)];
  const heldX: [number, number] = [Math.round(W * 0.64), W];

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
  const stacks = (order: number[], x0: number, x1: number) => {
    const L = Math.max(1, Math.floor((x1 - x0) / step));
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const lineH = P > 1 ? stackH + 6 : step;
    order.forEach((u, j) => {
      const col = j % L;
      const line = Math.floor(j / L);
      members[u]!.forEach((row, k) => {
        x[row] = x0 + col * step;
        y[row] = TOP + line * lineH + k * step;
      });
    });
    return { x, y, h: TOP + Math.ceil(order.length / L) * lineH };
  };
  const cells = (rows: number[], x0: number, x1: number) => {
    const L = Math.max(1, Math.floor((x1 - x0) / step));
    const pos = new Map<number, [number, number]>();
    rows.forEach((row, j) => pos.set(row, [x0 + (j % L) * step, TOP + Math.floor(j / L) * step]));
    return { pos, h: TOP + Math.ceil(rows.length / L) * step };
  };

  const allUnits = Array.from({ length: nUnits }, (_, u) => u);
  const f = flat();
  const st = known ? stacks(allUnits, 0, W) : f;

  // Chronological: units by their last time, piled where they share a column of the axis.
  let ordered: { x: Float32Array; y: Float32Array; h: number } | null = null;
  let boundaryX: number | null = null;
  const axis: { x: number; label: string }[] = [];
  const times = c.unit_time;
  if (times && times.length === nUnits) {
    const cols = Math.max(1, Math.floor(W / step) - 1);
    const colOf = (t: number) => Math.min(cols, Math.max(0, Math.round(t * cols)));
    const pile = new Map<number, number>();
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const byTime = [...allUnits].sort((a, b) => times[a]! - times[b]! || a - b);
    let maxLevel = 0;
    const unitH = known ? stackH : step;
    for (const u of byTime) {
      const col = colOf(times[u]!);
      const level = pile.get(col) ?? 0;
      pile.set(col, level + 1);
      maxLevel = Math.max(maxLevel, level + 1);
      members[u]!.forEach((row, k) => {
        x[row] = col * step;
        y[row] = k * step - level * (unitH + 2);
      });
    }
    const base = TOP + maxLevel * (unitH + 2);
    for (let i = 0; i < n; i++) y[i] = y[i]! + base - unitH;
    ordered = { x, y, h: base + 4 };
    const heldUnits = byTime.filter((u) => members[u]!.some((r) => hold[r] === 1));
    const firstHeld = heldUnits[0];
    if (firstHeld !== undefined) boundaryX = colOf(times[firstHeld]!) * step - gap / 2 - 1;
    if (c.time_start && c.time_end) {
      axis.push({ x: 0, label: c.time_start });
      axis.push({ x: cols * step, label: c.time_end });
      if (c.boundary && boundaryX !== null) axis.push({ x: boundaryX, label: c.boundary });
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
  if (known && c.state === "grouped") {
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
    // Grouping abandoned: units stay stacked on the training side, with a hole where a row was drawn.
    const tr = stacks(trainUnits, trainX[0], trainX[1]);
    const he = cells(heldRows, heldX[0], heldX[1]);
    const inTrain = new Set(trainUnits);
    for (let i = 0; i < n; i++) {
      if (hold[i] === 1) {
        const p = he.pos.get(i)!;
        sx[i] = p[0];
        sy[i] = p[1];
        if (inTrain.has(unitOf[i]!)) {
          holes.set(i, [tr.x[i]!, tr.y[i]!]);
          const mate = members[unitOf[i]!]!.find((r) => hold[r] === 0);
          if (mate !== undefined) straddles.push([i, mate]);
        }
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
  const split = straddles.length ? 1 : 0;
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
        return { x: src.x, y: src.y, mark: marks(), placed: zeros(), ...base, straddle: split };
      }
      case "split":
        return { x: sx, y: sy, mark: marks(), placed: ones(), ...base, boxes: 1, straddle: split };
      case "sealed":
        return { x: sx, y: sy, mark: marks(), placed: ones(), ...base, boxes: 1, sealed: 1, straddle: split };
    }
  });

  return {
    key,
    phases,
    labels: labels.map(plain),
    cell,
    gap,
    states,
    height: Math.max(f.h, st.h, ordered?.h ?? 0, splitH) + 8,
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
  const n = Math.min(a.x.length, b.x.length);
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
