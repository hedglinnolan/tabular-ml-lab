/** Column statistics the mock server computes from its in-memory tables. */
import { bin, deviation, mean, quantileSorted } from "d3-array";
import type {
  ColumnInfo,
  ColumnSummary,
  Dtype,
  Histogram,
  Scalar,
  ValueCount,
} from "../api/schema";
import type { MockColumn, MockDataset } from "./datasets";

const isMissing = (v: Scalar) => v === null || v === "";

function numbers(col: MockColumn): number[] {
  const out: number[] = [];
  for (const v of col.values) if (typeof v === "number" && Number.isFinite(v)) out.push(v);
  return out;
}

function counts(col: MockColumn): Map<Scalar, number> {
  const m = new Map<Scalar, number>();
  for (const v of col.values) if (!isMissing(v)) m.set(v, (m.get(v) ?? 0) + 1);
  return m;
}

export function topValues(col: MockColumn, k = 5): ValueCount[] {
  return [...counts(col).entries()]
    .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
    .slice(0, k)
    .map(([value, count]) => ({ value, count }));
}

export const isNumericDtype = (d: Dtype) => d === "numeric" || d === "integer";

export function nUnique(col: MockColumn): number {
  return counts(col).size;
}

export function nMissing(col: MockColumn): number {
  return col.values.reduce<number>((s, v) => s + (isMissing(v) ? 1 : 0), 0);
}

export function columnInfo(col: MockColumn): ColumnInfo {
  return {
    name: col.name,
    dtype: col.dtype,
    physical_type: col.physical_type,
    n_missing: nMissing(col),
    n_unique: nUnique(col),
    sample: col.values.filter((v) => !isMissing(v)).slice(0, 5),
  };
}

export function columnSummary(col: MockColumn): ColumnSummary {
  const missing = nMissing(col);
  const base = {
    name: col.name,
    dtype: col.dtype,
    n: col.values.length - missing,
    n_missing: missing,
    n_unique: nUnique(col),
  };
  if (isNumericDtype(col.dtype)) {
    const xs = numbers(col).sort((a, b) => a - b);
    const q = (p: number) => (xs.length ? (quantileSorted(xs, p) ?? null) : null);
    return {
      ...base,
      mean: mean(xs) ?? null,
      std: deviation(xs) ?? null,
      min: xs[0] ?? null,
      q25: q(0.25),
      median: q(0.5),
      q75: q(0.75),
      max: xs[xs.length - 1] ?? null,
      top: null,
    };
  }
  return {
    ...base,
    mean: null,
    std: null,
    min: null,
    q25: null,
    median: null,
    q75: null,
    max: null,
    top: topValues(col),
  };
}

export function histogram(col: MockColumn, bins = 20): Histogram {
  const xs = numbers(col);
  const binned = bin().thresholds(bins)(xs);
  const edges: number[] = [];
  const cts: number[] = [];
  binned.forEach((b, i) => {
    if (i === 0) edges.push(b.x0 ?? 0);
    edges.push(b.x1 ?? 0);
    cts.push(b.length);
  });
  return { column: col.name, edges, counts: cts, n_missing: nMissing(col) };
}

export function findColumn(ds: MockDataset, name: string): MockColumn | undefined {
  return ds.columns.find((c) => c.name === name);
}
