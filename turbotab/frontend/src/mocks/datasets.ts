/**
 * The mock server's tables, generated deterministically in the browser.
 *
 * dietary: 600 rows x 17 columns shaped like turbotab/sample_data/dietary_recalls.csv
 *          (300 people x 2 recalls, seeded under- and over-reports, compositional
 *          macronutrient shares, hba1c constant within a person).
 * genomics: 60 rows x 2,000 columns (sample id, condition, 1,998 gene counts) so the
 *          wide-table paths (picker, grouping, 2-D virtualized preview) are exercised.
 */
import type { Dtype, Scalar } from "../api/schema";

export interface MockColumn {
  name: string;
  dtype: Dtype;
  physical_type: string;
  values: Scalar[];
}

export interface MockDataset {
  name: string;
  columns: MockColumn[];
  nRows: number;
  sourceBytes: number;
}

/** mulberry32: a small seeded PRNG so every mock run shows the same numbers. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(r: () => number): number {
  const u = Math.max(r(), 1e-12);
  const v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const round = (x: number, dp: number) => {
  const f = 10 ** dp;
  return Math.round(x * f) / f;
};

function estimateBytes(columns: MockColumn[], nRows: number): number {
  let bytes = columns.reduce((s, c) => s + c.name.length + 1, 0);
  for (let i = 0; i < Math.min(nRows, 50); i++) {
    for (const c of columns) bytes += String(c.values[i] ?? "").length + 1;
  }
  return Math.round((bytes / Math.min(nRows, 50)) * nRows);
}

function dataset(name: string, columns: MockColumn[]): MockDataset {
  const nRows = columns[0]?.values.length ?? 0;
  return { name, columns, nRows, sourceBytes: estimateBytes(columns, nRows) };
}

export function dietaryRecalls(): MockDataset {
  const r = rng(4242);
  const cols: Record<string, Scalar[]> = {
    participant_id: [],
    recall_number: [],
    recall_date: [],
    age: [],
    sex: [],
    bmi: [],
    energy_kcal: [],
    protein_pct_kcal: [],
    fat_pct_kcal: [],
    carbohydrate_pct_kcal: [],
    alcohol_pct_kcal: [],
    protein_g: [],
    fat_g: [],
    carbohydrate_g: [],
    fiber_g: [],
    sodium_mg: [],
    hba1c: [],
  };
  const under = new Set<number>();
  const over = new Set<number>();
  while (under.size < 12) under.add(Math.floor(r() * 600));
  while (over.size < 8) {
    const i = Math.floor(r() * 600);
    if (!under.has(i)) over.add(i);
  }
  const push = (k: string, v: Scalar) => cols[k]!.push(v);
  const start = Date.UTC(2024, 4, 1);
  for (let p = 1; p <= 300; p++) {
    const id = `P${String(p).padStart(3, "0")}`;
    const age = 20 + Math.floor(r() * 60);
    const sex = r() < 0.52 ? "F" : "M";
    const bmi = r() < 0.012 ? null : round(Math.min(48, Math.max(17, 27 + 5 * normal(r))), 1);
    const hba1c = round(Math.min(9.8, Math.max(4.3, 5.6 + 0.55 * normal(r))), 1);
    const first = start + Math.floor(r() * 60) * 86400000;
    const gap = 3 + Math.floor(r() * 12);
    for (let recall = 1; recall <= 2; recall++) {
      const row = (p - 1) * 2 + (recall - 1);
      const day = recall === 1 ? first : first + gap * 86400000;
      // Natural days stay inside 520-4,950 kcal; the seeded reports below are the extremes.
      let energy = Math.round(Math.min(4950, Math.max(520, 2100 * Math.exp(0.42 * normal(r)))));
      if (under.has(row)) energy = 240 + Math.round(r() * 190);
      if (over.has(row)) energy = 6100 + Math.round(r() * 1800);
      const protein = round(Math.min(35, Math.max(8, 17 + 3 * normal(r))), 2);
      const alcohol = r() < 0.6 ? 0 : round(0.5 + r() * 7.5, 2);
      const fat = round(Math.min(55, Math.max(15, 34 + 6 * normal(r))), 2);
      const carb = round(100 - protein - fat - alcohol, 2);
      push("participant_id", id);
      push("recall_number", recall);
      push("recall_date", new Date(day).toISOString().slice(0, 10));
      push("age", age);
      push("sex", sex);
      push("bmi", bmi);
      push("energy_kcal", energy);
      push("protein_pct_kcal", protein);
      push("fat_pct_kcal", fat);
      push("carbohydrate_pct_kcal", carb);
      push("alcohol_pct_kcal", alcohol);
      push("protein_g", round((energy * protein) / 400, 1));
      push("fat_g", round((energy * fat) / 900, 1));
      push("carbohydrate_g", round((energy * carb) / 400, 1));
      push("fiber_g", r() < 0.008 ? null : round(Math.max(2, 18 + 6 * normal(r)), 1));
      push("sodium_mg", Math.round(Math.max(600, 3300 + 900 * normal(r))));
      push("hba1c", hba1c);
    }
  }
  const types: Record<string, [Dtype, string]> = {
    participant_id: ["categorical", "VARCHAR"],
    recall_number: ["integer", "BIGINT"],
    recall_date: ["datetime", "DATE"],
    age: ["integer", "BIGINT"],
    sex: ["categorical", "VARCHAR"],
  };
  return dataset(
    "dietary_recalls.csv",
    Object.entries(cols).map(([name, values]) => {
      const [dtype, physical_type] = types[name] ?? ["numeric", "DOUBLE"];
      return { name, dtype, physical_type, values };
    }),
  );
}

export function genomicsWide(nGenes = 1998, nSamples = 60): MockDataset {
  const r = rng(1998);
  const columns: MockColumn[] = [
    {
      name: "sample_id",
      dtype: "categorical",
      physical_type: "VARCHAR",
      values: Array.from({ length: nSamples }, (_, i) => `S${String(i + 1).padStart(2, "0")}`),
    },
    {
      name: "condition",
      dtype: "categorical",
      physical_type: "VARCHAR",
      values: Array.from({ length: nSamples }, (_, i) => (i % 2 === 0 ? "case" : "control")),
    },
  ];
  for (let g = 0; g < nGenes; g++) {
    const mu = Math.exp(4 + 1.5 * normal(r));
    const values: Scalar[] = [];
    for (let s = 0; s < nSamples; s++) {
      const effect = s % 2 === 0 && g % 37 === 0 ? 1.8 : 1;
      const m = mu * effect;
      values.push(Math.max(0, Math.round(m + Math.sqrt(m + 0.1 * m * m) * normal(r))));
    }
    columns.push({
      name: `ENSG${String(100000 + g * 17).padStart(11, "0")}`,
      dtype: "integer",
      physical_type: "BIGINT",
      values,
    });
  }
  return dataset("genomics_counts_wide.csv", columns);
}
