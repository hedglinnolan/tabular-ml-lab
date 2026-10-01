/**
 * A synthetic table shaped like the local NHANES export the M1 journey runs on
 * (_tt_tmp_nhanes.csv: 21,849 rows × 29 columns, not in git). Same column names, types and
 * the counts that drive the journey: 194 recalls under 500 kcal and 307 over 5,000; the
 * medication questions blank for most people (asked of 2,996 people both times, 2,943 of
 * them inside the 500–5,000 screen); the imputed_* flags true on the real numbers of rows.
 * Every value is drawn from a seeded generator, so each mock run shows the same table. It is
 * not NHANES data: the marginals are calibrated to the export's summary statistics only.
 */
import type { Scalar } from "../api/schema";
import { rng, type MockColumn, type MockDataset } from "./datasets";

const N = 21_849;
const CYCLES: [number, number][] = [
  [2001, 2501],
  [2003, 2295],
  [2005, 2299],
  [2007, 2624],
  [2009, 2807],
  [2011, 2363],
  [2013, 2463],
  [2015, 2227],
  [2017, 2270],
];

function normal(r: () => number): number {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

const round = (x: number, dp: number) => {
  const f = 10 ** dp;
  return Math.round(x * f) / f;
};
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** `k` distinct indices from `pool`, weighted by `weight` (seeded). */
function pick(r: () => number, pool: number[], k: number, weight?: (i: number) => number) {
  const keyed = pool.map((i) => ({ i, key: Math.pow(r(), 1 / Math.max(1e-6, weight?.(i) ?? 1)) }));
  keyed.sort((a, b) => b.key - a.key);
  return new Set(keyed.slice(0, k).map((x) => x.i));
}

export const NHANES_NAME = "nhanes_diet_glucose.csv";

export function nhanesLike(): MockDataset {
  const r = rng(2001);
  const col: Record<string, Scalar[]> = {};
  const names = [
    "SEQN",
    "cycle_begin_year",
    "age",
    "gender",
    "bp_sys",
    "bp_di",
    "weight",
    "height",
    "bmi",
    "waist",
    "kcal",
    "protein",
    "sugar",
    "carb",
    "fat_total",
    "fat_sat",
    "fat_mon",
    "fat_poly",
    "hdl",
    "glucose",
    "triglycerides",
    "meds_hbp",
    "meds_chol",
    "imputed_weight",
    "imputed_height",
    "imputed_bmi",
    "imputed_waist",
    "imputed_bp_sys",
    "imputed_bp_di",
  ];
  for (const n of names) col[n] = new Array<Scalar>(N);

  const all = Array.from({ length: N }, (_, i) => i);
  const under = pick(r, all, 194);
  const over = pick(
    r,
    all.filter((i) => !under.has(i)),
    307,
  );
  const ages: number[] = [];
  const glucoseBase: number[] = [];
  let seqn = 9966;
  let cycle = 0;
  let inCycle = 0;
  for (let i = 0; i < N; i++) {
    if (inCycle >= CYCLES[cycle]![1]) {
      cycle += 1;
      inCycle = 0;
    }
    inCycle += 1;
    col.SEQN![i] = seqn;
    seqn += 1 + Math.floor(r() * 6.5);
    col.cycle_begin_year![i] = CYCLES[cycle]![0];
    const age = Math.round(clamp(18 + 67 * Math.pow(r(), 0.92), 18, 85));
    ages.push(age);
    col.age![i] = age;
    const female = r() < 0.512;
    col.gender![i] = female ? "female" : "male";
    const height = round(female ? 161 + 7 * normal(r) : 175 + 7.5 * normal(r), 1);
    const bmi = round(clamp(Math.exp(Math.log(28) + 0.22 * normal(r)), 14, 80), 2);
    col.height![i] = height;
    col.bmi![i] = bmi;
    col.weight![i] = round((bmi * height * height) / 10_000, 1);
    col.waist![i] = round(clamp(98 + 1.9 * (bmi - 28.8) + 7 * normal(r), 56, 200), 1);
    col.bp_sys![i] = round(clamp(104 + 0.42 * age + 14 * normal(r), 70, 240), 1);
    col.bp_di![i] = round(clamp(69 + 11 * normal(r), 20, 134), 1);
    col.hdl![i] = Math.round(clamp(54 + 15 * normal(r) - (female ? 0 : 7), 12, 200));
    col.triglycerides![i] = Math.round(clamp(Math.exp(Math.log(105) + 0.52 * normal(r)), 15, 2500));

    let kcal = Math.exp(Math.log(1950) + 0.42 * normal(r));
    kcal = clamp(kcal, 520, 4950);
    if (under.has(i)) kcal = 40 + r() * 440;
    if (over.has(i)) kcal = 5040 + Math.pow(r(), 2.2) * 9000;
    kcal = Math.round(kcal);
    col.kcal![i] = kcal;
    const pShare = clamp(0.155 + 0.045 * normal(r), 0.06, 0.34);
    const fShare = clamp(0.33 + 0.085 * normal(r), 0.1, 0.58);
    const cShare = clamp(1 - pShare - fShare - Math.max(0, 0.02 * normal(r)), 0.15, 0.75);
    const protein = (kcal * pShare) / 4;
    const fat = (kcal * fShare) / 9;
    const carb = (kcal * cShare) / 4;
    col.protein![i] = round(protein, 2);
    col.fat_total![i] = round(fat, 2);
    col.carb![i] = round(carb, 2);
    col.sugar![i] = round(carb * clamp(0.45 + 0.12 * normal(r), 0.08, 0.92), 2);
    const sat = fat * clamp(0.33 + 0.04 * normal(r), 0.2, 0.45);
    const mon = fat * clamp(0.36 + 0.03 * normal(r), 0.25, 0.45);
    col.fat_sat![i] = round(sat, 2);
    col.fat_mon![i] = round(mon, 2);
    col.fat_poly![i] = round(
      Math.min(fat - sat - mon, fat * clamp(0.22 + 0.04 * normal(r), 0.1, 0.3)),
      2,
    );

    // Fasting glucose: mostly 85–110 mg/dL, rising with age and BMI, with a diabetic tail.
    const diabetic = r() < 0.04 + 0.0018 * age + 0.004 * Math.max(0, bmi - 30);
    const g =
      82 +
      0.22 * age +
      0.55 * (bmi - 28) -
      0.0012 * (carb - 250) +
      9 * normal(r) +
      (diabetic ? Math.exp(Math.log(55) + 0.65 * normal(r)) : 0);
    glucoseBase.push(g);
    col.glucose![i] = round(clamp(g, 40, 686), 1);
  }

  // The medication questions: asked of older people with higher glucose far more often.
  const kept = all.filter((i) => !under.has(i) && !over.has(i));
  const outside = all.filter((i) => under.has(i) || over.has(i));
  const weight = (i: number) => Math.pow(ages[i]! / 50, 2.4) * Math.pow(glucoseBase[i]! / 100, 2);
  const both = new Set([...pick(r, kept, 2943, weight), ...pick(r, outside, 53, weight)]);
  const rest = all.filter((i) => !both.has(i));
  const hbpOnly = pick(r, rest, 6297 - 2996, weight);
  const cholOnly = pick(
    r,
    rest.filter((i) => !hbpOnly.has(i)),
    4645 - 2996,
    weight,
  );
  for (let i = 0; i < N; i++) {
    const hbp = both.has(i) || hbpOnly.has(i);
    const chol = both.has(i) || cholOnly.has(i);
    col.meds_hbp![i] = hbp ? r() < 0.88 : null;
    col.meds_chol![i] = chol ? r() < 0.78 : null;
  }

  const flags: [string, number][] = [
    ["imputed_weight", 235],
    ["imputed_height", 248],
    ["imputed_bmi", 306],
    ["imputed_waist", 658],
  ];
  for (const [name, k] of flags) {
    const on = pick(r, all, k);
    for (let i = 0; i < N; i++) col[name]![i] = on.has(i);
  }
  const bp = pick(r, all, 1876);
  for (let i = 0; i < N; i++) {
    col.imputed_bp_sys![i] = bp.has(i);
    col.imputed_bp_di![i] = bp.has(i);
  }

  const types: Record<string, [MockColumn["dtype"], string]> = {
    SEQN: ["integer", "BIGINT"],
    cycle_begin_year: ["integer", "BIGINT"],
    age: ["integer", "BIGINT"],
    gender: ["categorical", "VARCHAR"],
    meds_hbp: ["boolean", "BOOLEAN"],
    meds_chol: ["boolean", "BOOLEAN"],
  };
  const columns: MockColumn[] = names.map((name) => {
    const [dtype, physical_type] =
      types[name] ?? (name.startsWith("imputed_") ? ["boolean", "BOOLEAN"] : ["numeric", "DOUBLE"]);
    return { name, dtype, physical_type, values: col[name]! };
  });
  return { name: NHANES_NAME, columns, nRows: N, sourceBytes: 3_412_906 };
}
