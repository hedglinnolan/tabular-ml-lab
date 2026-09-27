/**
 * The app's words for the prototype — every sentence a user might read, in one place, so the
 * word budgets (BLUEPRINT §11.4, M1_CONTRACT §5) can be checked by a test. Numbers inside the
 * sentences are the fixture's, quoted; backticks mark data values (mono chips).
 *
 * Budgets: question ≤ 14 · why ≤ 22 · option label ≤ 4 · option consequence ≤ 16 ·
 * finding summary ≤ 20 · lever ≤ 5 · sidenote ≤ 8 · in-place why ≤ 60.
 */

export const BUDGET = {
  question: 14,
  why: 22,
  label: 4,
  consequence: 16,
  summary: 20,
  lever: 5,
  sidenote: 8,
  whyInPlace: 60,
} as const;

// ── S1 · energy adjustment ───────────────────────────────────────────────────

export const ENERGY_Q = {
  kicker: "Energy adjustment",
  question: "How should the 7 nutrients account for total energy, `kcal`?",
  why: "`fat_total` correlates 0.86 with `kcal`: people who eat more eat more of everything.",
};

export interface OptionCopy {
  label: string;
  /** Used in "With …" at the scrub's right end. */
  short: string;
  consequence: string;
  /** Beside the scatter's line end: what the picture shows, in the app's voice. */
  sidenote: string;
  /** The recorded sentence (methods-section voice). */
  sentence: string;
}

export const ENERGY_OPTIONS: Record<string, OptionCopy> = {
  residual: {
    label: "Willett residual",
    short: "the Willett residual",
    consequence: "Each nutrient loses the part `kcal` predicts; `kcal` leaves the model.",
    sidenote: "flat: the energy share is removed",
    sentence:
      "Energy was adjusted by the residual method: each nutrient was replaced by its residual on `kcal`, fit on training rows.",
  },
  density: {
    label: "Density alone",
    short: "density alone",
    consequence: "Each nutrient is divided by `kcal`; `kcal` leaves the model.",
    sidenote: "divided by kcal: nearly flat",
    sentence:
      "Energy was adjusted by nutrient density: each nutrient was divided by `kcal`, and `kcal` left the model.",
  },
  density_multivariate: {
    label: "Density + energy term",
    short: "density + energy term",
    consequence: "Each nutrient is divided by `kcal`; `kcal` stays as its own term.",
    sidenote: "divided by kcal: nearly flat",
    sentence:
      "Energy was adjusted by the multivariate density model: each nutrient was divided by `kcal`, and `kcal` stayed as a term.",
  },
  standard: {
    label: "Standard model",
    short: "the standard model",
    consequence: "Nutrients stay as recorded; `kcal` enters the model beside them.",
    sidenote: "unchanged: kcal sits beside it",
    sentence:
      "Energy was adjusted by the standard model: nutrients entered as recorded, with `kcal` beside them.",
  },
  none: {
    label: "No adjustment",
    short: "no adjustment",
    consequence: "Nutrients stay as recorded; the columns match the standard model.",
    sidenote: "unchanged",
    sentence: "Nutrients were not adjusted for energy; they entered the model as recorded.",
  },
  partition: {
    label: "Energy partition",
    short: "energy partition",
    consequence: "`sugar` has no Atwater factor, so `kcal` cannot be split across all 7.",
    sidenote: "",
    sentence: "",
  },
  partition3: {
    label: "Partition, 3 macronutrients",
    short: "partition on 3 macronutrients",
    consequence: "`protein`, `carb` and `fat_total` become their kcal; the rest of `kcal` stays.",
    sidenote: "same shape, now in kcal",
    sentence:
      "Energy was partitioned: `protein`, `carb` and `fat_total` entered as their kcal, beside kcal from everything else.",
  },
};

export const ENERGY_NOW_NOTE = "climbs with kcal";

/** Order is judgment (§11.9): the usual first; each step down changes one thing. */
export const ENERGY_ORDER = [
  "residual",
  "density",
  "density_multivariate",
  "standard",
  "none",
  "partition",
] as const;

// ── S2 · exclusions ──────────────────────────────────────────────────────────

export const EXCL_Q = {
  kicker: "Exclusions",
  question: "Which reported `kcal` intakes should be excluded as implausible?",
  why: "`kcal` runs from 0 to 15,594 a day; recalls at the extremes are usually misreports.",
};

export const EXCL_OPTIONS: Record<string, OptionCopy> = {
  keep_all: {
    label: "Keep every row",
    short: "every row kept",
    consequence: "Nothing is excluded; all 21,849 rows stay.",
    sidenote: "no step added",
    sentence: "No rows were excluded for implausible intake.",
  },
  kcal_500_5000: {
    label: "500–5,000 kcal",
    short: "500–5,000 kcal",
    consequence: "Rows outside 500–5,000 kcal leave: 501 of 21,849.",
    sidenote: "one range for everyone",
    sentence: "`501` rows outside `500`–`5,000` kcal were excluded as implausible intakes.",
  },
  sex_specific: {
    label: "Sex-specific (Willett)",
    short: "sex-specific cut-offs",
    consequence: "Women 500–3,500, men 800–4,200 kcal: 1,419 rows leave.",
    sidenote: "cut-offs by `gender`",
    sentence:
      "`1,419` rows outside `500`–`3,500` kcal (women) or `800`–`4,200` kcal (men) were excluded as implausible intakes.",
  },
};

// ── S3 · findings ────────────────────────────────────────────────────────────

export interface FindingCopy {
  summary: string;
  /** Outcome-labeled; null when M1 has no control for it (the summary says so). */
  lever: string | null;
  routes_to: "energy_adjustment" | "exclusions" | "roles" | null;
}

export const FINDING_COPY: Record<string, FindingCopy> = {
  "pack::dietary::energy_adjustment": {
    summary: "All 7 nutrients rise with `kcal` (r 0.62 to 0.86), so total energy confounds each one.",
    lever: "Adjust for energy",
    routes_to: "energy_adjustment",
  },
  "pack::dietary::implausible_intake": {
    summary: "501 records report `kcal` below 500 or above 5,000; nothing is excluded unless you choose.",
    lever: "Choose exclusions",
    routes_to: "exclusions",
  },
  binary_text__gender: {
    summary: "`gender` is text with two values; the model codes `male` as 1. No control flips it yet.",
    lever: null,
    routes_to: null,
  },
  binary_text__imputed_bmi: {
    summary: "`imputed_bmi` is already true/false (306 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__imputed_bp_di: {
    summary: "`imputed_bp_di` is already true/false (1,876 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__imputed_bp_sys: {
    summary: "`imputed_bp_sys` is already true/false (1,876 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__imputed_height: {
    summary: "`imputed_height` is already true/false (248 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__imputed_waist: {
    summary: "`imputed_waist` is already true/false (658 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__imputed_weight: {
    summary: "`imputed_weight` is already true/false (235 true); reading it as binary changes no value.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__meds_hbp: {
    summary: "`meds_hbp` is true/false with 15,552 blank; it is excluded from this model.",
    lever: "Review roles",
    routes_to: "roles",
  },
  boolean_as_text__meds_hbp: {
    summary: "`meds_hbp` stores its 6,297 answers as text, so a model would refuse it as is.",
    lever: "Review roles",
    routes_to: "roles",
  },
  binary_text__meds_chol: {
    summary: "`meds_chol` is true/false with 17,204 blank; it is excluded from this model.",
    lever: "Review roles",
    routes_to: "roles",
  },
  boolean_as_text__meds_chol: {
    summary: "`meds_chol` stores its 4,645 answers as text, so a model would refuse it as is.",
    lever: "Review roles",
    routes_to: "roles",
  },
};

/** Same-kind findings share one paged card (§11.7). */
export const FINDING_GROUPS: { key: string; title: string; ids: string[] }[] = [
  {
    key: "flags",
    title: "Flags already true/false",
    ids: [
      "binary_text__imputed_bmi",
      "binary_text__imputed_bp_di",
      "binary_text__imputed_bp_sys",
      "binary_text__imputed_height",
      "binary_text__imputed_waist",
      "binary_text__imputed_weight",
    ],
  },
  {
    key: "text",
    title: "True/false stored as text",
    ids: [
      "binary_text__meds_hbp",
      "boolean_as_text__meds_hbp",
      "binary_text__meds_chol",
      "boolean_as_text__meds_chol",
    ],
  },
];

/** Pushed, ranked: what changes the most numbers downstream first. */
export const FINDINGS_PUSHED = [
  "pack::dietary::energy_adjustment",
  "pack::dietary::implausible_intake",
  "binary_text__gender",
];

// ── S4 · the wide transform ──────────────────────────────────────────────────

export const WIDE_Q = {
  kicker: "Transform counts",
  question: "Log-transform the 495 count columns?",
  why: "Counts are right-skewed: `gene_0430` has skewness 3.2; after log2(x + 1), 0.1.",
};

export const WIDE_OPTIONS: Record<string, OptionCopy> = {
  log: {
    label: "log2(x + 1)",
    short: "log2(x + 1)",
    consequence: "Every count column is compressed; large counts shrink most.",
    sidenote: "skewness 3.2 → 0.1",
    sentence: "The `495` count columns were transformed by `log2(x + 1)`.",
  },
  keep: {
    label: "Keep raw counts",
    short: "raw counts",
    consequence: "Nothing changes; the 495 columns enter as counts.",
    sidenote: "unchanged",
    sentence: "The `495` count columns entered the model as raw counts.",
  },
};

export function words(text: string): number {
  return text.replace(/`/g, "").split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
}
