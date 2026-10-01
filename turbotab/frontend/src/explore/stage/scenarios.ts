/**
 * The four prototype scenarios, built from the real-data fixture.
 *
 * The Record's words (question, why, option lines, sentences) are written here in the
 * app's voice; every number in them is read from the fixture, never typed. Word budgets
 * (BLUEPRINT §11.4) are enforced by scenarios.test.ts.
 */
import { A, B, viewOf, type EnergyOption } from "./fixture";
import { fmtInt, fmtR } from "./format";
import type {
  ConsequenceView,
  DistributionView,
  Lineage,
  LineageView,
  RowStep,
  TableFocusView,
} from "./types";

export type ScenarioId = "energy" | "exclusions" | "findings" | "transform";

export interface StageView {
  view: ConsequenceView;
  /** A short mono fact for the thumbnail header (computed, never typed). */
  chip?: string;
  /** Show the recorded state only: a finding's evidence, not a choice's consequence. */
  evidence?: boolean;
  /** Row flow only: who each exclusion removes, by level (below / above the cut). */
  byLevel?: Record<string, { below: number; above: number }>;
  /** A live-pipeline section shown beside evidence: where the open question sits. */
  openLabel?: string;
  openAfter?: string;
}

export interface Preview {
  /** The stage bar's tag: "Preview" for an option, "Evidence" for a finding. */
  pill?: string;
  /** What the stage bar says about recording (default: nothing is recorded). */
  aside?: string;
  /** Names the option in the stage bar. */
  label: string;
  basis: string;
  /** Said once above the views, e.g. why a refused option shows a narrower preview. */
  note?: string;
  views: StageView[];
}

export interface OptionModel {
  key: string;
  label: string;
  /** ≤ 16 words; backticks mark data. */
  line: string;
  tag?: string;
  badge?: string;
  refused?: { exitLabel: string; exitKey: string };
  preview: Preview;
  /** The decision sentence once recorded. */
  sentence: string;
  /** Layer 2, opened in place: what the option estimates or rests on (≤ 60 words). */
  why?: string;
}

export interface LiveRows {
  steps: RowStep[];
  split?: { train: number; holdout: number };
  /** The open question's place in the flow: after this step key. */
  openAfter?: string;
  byLevel?: Record<string, { below: number; above: number }>;
}

export interface LiveModel {
  rows: LiveRows;
  lineage: Lineage;
  /** The open question acts on the adjusted lane. */
  lineageOpen?: string;
  waits: string[];
}

export interface Dataset {
  name: string;
  rows: number;
  cols: number;
}

export interface QuestionModel {
  id: ScenarioId;
  dataset: Dataset;
  kicker: string;
  question: string;
  why: string;
  options: OptionModel[];
  /** Hidden recorded options (e.g. the partition exit) that can still be recorded. */
  extra?: OptionModel[];
  live: LiveModel;
  liveAfter: (key: string) => LiveModel;
  /** Distribution drawn as two stacked panels (the variable changes, not just its rows). */
  stacked: boolean;
  /** Terms that define themselves (dotted underline → one sentence). */
  terms?: Record<string, string>;
}

const NHANES: Dataset = { name: "_tt_tmp_nhanes", rows: A.dataset.n_rows, cols: A.dataset.n_cols };
const GENOMICS: Dataset = {
  name: "genomics_expression",
  rows: B.dataset.n_rows,
  cols: B.dataset.n_cols,
};

// ── shared facts ─────────────────────────────────────────────────────────────

const EA = A.energy_adjustment;
const ENERGY = EA.energy_column;
const NUTRIENTS = EA.nutrients;
const TOP = EA.correlation_with_energy[0]!;
const BOTTOM = EA.correlation_with_energy[EA.correlation_with_energy.length - 1]!;

function matrixCount(l: Lineage): number {
  return l.nodes.filter((n) => n.lane === "matrix").reduce((s, n) => s + n.count, 0);
}

function lineageChip(v: LineageView): string {
  const before = v.before ? matrixCount(v.before) : null;
  const after = matrixCount(v.after);
  const changed = v.after.nodes.filter((n) => n.lane === "adjusted" && n.formula).length;
  const cols = before !== null && before !== after ? `${before} → ${after} columns` : `${after} columns`;
  return changed ? `${cols} · ${changed} rewritten` : `${cols} · none rewritten`;
}

// ── S1 · energy adjustment ───────────────────────────────────────────────────

const byMethod = Object.fromEntries(EA.options.map((o) => [o.method, o])) as Record<
  string,
  EnergyOption
>;
const SUBSET = EA.partition_on_macronutrient_totals;
const n7 = NUTRIENTS.length;
/** The nutrient the refusal names, read from the real Refusal message. */
const NO_ATWATER = /(\w+) carries no energy/.exec(byMethod.partition!.refusal!.error.message)?.[1];

const ENERGY_LINES: Record<string, string> = {
  residual: `All ${n7} nutrients become residuals on \`${ENERGY}\`; \`${ENERGY}\` leaves the model.`,
  density: `All ${n7} nutrients are divided by \`${ENERGY}\`; \`${ENERGY}\` leaves the model.`,
  density_multivariate: `All ${n7} nutrients are divided by \`${ENERGY}\`; \`${ENERGY}\` stays in as its own term.`,
  standard: `Nutrients enter unchanged, with \`${ENERGY}\` beside them in the model.`,
  partition: `Not available: \`${NO_ATWATER}\` has no Atwater factor, so \`${ENERGY}\` cannot be split by it.`,
  none: `Nutrients enter as absolute intakes, confounded by total energy.`,
};

const ENERGY_SENTENCES: Record<string, string> = {
  residual: `Energy was adjusted by the residual method: each nutrient is replaced by its residual on \`${ENERGY}\`, fit on training rows.`,
  density: `Energy was adjusted by nutrient density: each nutrient is divided by \`${ENERGY}\`, and \`${ENERGY}\` leaves the model.`,
  density_multivariate: `Energy was adjusted by the multivariate density model: each nutrient is divided by \`${ENERGY}\`, which stays in as a term.`,
  standard: `Energy was adjusted by the standard model: nutrients enter unchanged beside \`${ENERGY}\`.`,
  partition: `Energy was partitioned into \`${ENERGY}\` from \`protein\`, \`carb\` and \`fat_total\`, and \`${ENERGY}\` from everything else.`,
  none: `No energy adjustment was applied: nutrients enter as absolute intakes.`,
};

/** Usual first (the pack names residual, with density beside it); the nothing-answer last. */
const ENERGY_ORDER = ["residual", "density", "density_multivariate", "standard", "partition", "none"];

function energyPreview(o: EnergyOption, label: string, note?: string): Preview {
  const p = o.preview!;
  const rel = viewOf(p.views, "relationship");
  const lin = viewOf(p.views, "lineage");
  const dist = viewOf(p.views, "distribution");
  return {
    label,
    basis: p.basis,
    note,
    views: [
      { view: rel, chip: `r ${fmtR(rel.r_before)} → ${fmtR(rel.r_after)}` },
      { view: lin, chip: lineageChip(lin) },
      { view: dist, chip: dist.after_label },
    ],
  };
}

function energyOption(method: string): OptionModel {
  const o = byMethod[method]!;
  if (method === "partition") {
    const subset = SUBSET.nutrients.map((n) => `\`${n}\``);
    return {
      key: "partition",
      label: o.label,
      line: ENERGY_LINES.partition!,
      badge: o.method_card.standing ?? undefined,
      refused: { exitLabel: `Partition ${SUBSET.nutrients.join(", ")}`, exitKey: "partition_subset" },
      why: o.applicable.reason,
      preview: energyPreview(
        SUBSET,
        o.label,
        `Not available with \`${NO_ATWATER}\`. Shown instead on ${subset.slice(0, -1).join(", ")} and ${subset[subset.length - 1]}, which carry energy.`,
      ),
      sentence: ENERGY_SENTENCES.partition!,
    };
  }
  return {
    key: method,
    label: o.label,
    line: ENERGY_LINES[method]!,
    tag: method === "residual" ? "usual" : undefined,
    badge: o.method_card.standing ?? undefined,
    preview: energyPreview(o, o.label),
    sentence: ENERGY_SENTENCES[method]!,
    why: o.estimand,
  };
}

const cohortFlow = viewOf(
  A.exclusions.options.find((o) => o.key === "kcal_500_5000")!.preview.views,
  "row_flow",
);
const unadjusted = viewOf(byMethod.none!.preview!.views, "lineage").before!;

const energyLive: LiveModel = {
  rows: {
    steps: cohortFlow.after,
    split: { train: A.setup.n_train, holdout: A.setup.n_holdout },
    byLevel: A.exclusions.options.find((o) => o.key === "kcal_500_5000")!.counts.by_level,
  },
  lineage: unadjusted,
  lineageOpen: "energy adjustment: this question",
  waits: ["energy adjustment", "models"],
};

const ENERGY_Q: QuestionModel = {
  id: "energy",
  dataset: NHANES,
  kicker: "Energy adjustment",
  question: "How should nutrients be adjusted for total energy?",
  why: `People who eat more eat more of everything: \`${TOP.column}\` tracks \`${ENERGY}\` at r ${fmtR(TOP.r)} on your training rows.`,
  options: ENERGY_ORDER.map(energyOption),
  extra: [
    {
      key: "partition_subset",
      label: SUBSET.label,
      line: "",
      preview: energyPreview(SUBSET, SUBSET.label),
      sentence: ENERGY_SENTENCES.partition!,
    },
  ],
  live: energyLive,
  liveAfter: (key) => {
    const o = key === "partition_subset" ? SUBSET : byMethod[key]!;
    return {
      ...energyLive,
      lineage: viewOf(o.preview!.views, "lineage").after,
      lineageOpen: undefined,
      waits: ["models"],
    };
  },
  stacked: true,
  terms: {
    "Atwater factor":
      "The energy one gram of a nutrient yields: 4 kcal for protein and carbohydrate, 9 for fat.",
  },
};

// ── S2 · exclusions ──────────────────────────────────────────────────────────

const EX = Object.fromEntries(A.exclusions.options.map((o) => [o.key, o]));
const EX_ORDER = ["sex_specific", "kcal_500_5000", "keep_all"];

function rangeOf(o: (typeof A.exclusions.options)[number]) {
  const rule = (o.decision.rules as { low: number | null; high: number | null; by: unknown }[])[0];
  return rule;
}

const sexRule = rangeOf(EX.sex_specific!) as unknown as {
  by: { ranges: Record<string, [number, number]> };
};
const neutral = rangeOf(EX.kcal_500_5000!)!;
const r = (a: [number, number]) => `${fmtInt(a[0])}–${fmtInt(a[1])}`;
const women = sexRule.by.ranges.female!;
const men = sexRule.by.ranges.male!;
const neutralRange = r([neutral.low!, neutral.high!]);

const EX_LINES: Record<string, string> = {
  sex_specific: `Drops ${fmtInt(EX.sex_specific!.counts.excluded)} rows: women outside ${r(women)}, men outside ${r(men)} kcal a day.`,
  kcal_500_5000: `Drops ${fmtInt(EX.kcal_500_5000!.counts.excluded)} rows outside ${neutralRange} kcal a day, the same range for everyone.`,
  keep_all: `No one is excluded; all ${fmtInt(EX.keep_all!.counts.kept)} rows go forward.`,
};

const EX_SENTENCES: Record<string, string> = {
  sex_specific: `\`${fmtInt(EX.sex_specific!.counts.excluded)}\` rows outside sex-specific ranges (women \`${r(women)}\`, men \`${r(men)}\` kcal) were excluded as implausible intakes.`,
  kcal_500_5000: `\`${fmtInt(EX.kcal_500_5000!.counts.excluded)}\` rows outside \`${neutralRange}\` kcal were excluded as implausible intakes.`,
  keep_all: "No rows were excluded for implausible intake.",
};

const EX_LABELS: Record<string, string> = {
  sex_specific: "Sex-specific (Willett)",
  kcal_500_5000: `${neutralRange} kcal`,
  keep_all: "Keep every row",
};

/** The longest day, from the implausible-intake finding (histogram edges are rounded). */
const implausible = A.findings.findings.find((f) => f.id === "pack::dietary::implausible_intake")!;
const kcalMax = Number(
  (/Observed range [\d,.]+ to ([\d,.]+)/.exec(implausible.detail)?.[1] ?? "0").replace(/,/g, ""),
);

function exclusionOption(key: string): OptionModel {
  const o = EX[key]!;
  const flow = viewOf(o.preview.views, "row_flow");
  const dist = viewOf(o.preview.views, "distribution");
  return {
    key,
    label: EX_LABELS[key]!,
    line: EX_LINES[key]!,
    tag: key === "sex_specific" ? "usual" : undefined,
    badge: o.evidence?.status,
    why: o.evidence ? `${o.evidence.quote.replace(/\s*\[[A-Z]+\]/g, "")}.` : undefined,
    preview: {
      label: EX_LABELS[key]!,
      basis: o.preview.basis,
      views: [
        {
          view: flow,
          chip: `${fmtInt(flow.before[flow.before.length - 1]!.n)} → ${fmtInt(o.counts.final)} rows`,
          byLevel: o.counts.by_level,
        },
        { view: dist, chip: dist.marks.length ? `${dist.marks.length} cuts` : "no cut" },
      ],
    },
    sentence: EX_SENTENCES[key]!,
  };
}

const exLive: LiveModel = {
  rows: { steps: viewOf(EX.keep_all!.preview.views, "row_flow").before, openAfter: "outcome_measured" },
  lineage: unadjusted,
  waits: ["exclusions", "energy adjustment", "models"],
};

const EXCLUSIONS_Q: QuestionModel = {
  id: "exclusions",
  dataset: NHANES,
  kicker: "Eligibility",
  question: "Which rows should be excluded as implausible energy intakes?",
  why: `\`${ENERGY}\` runs from 0 to ${fmtInt(kcalMax)} a day; values far outside a plausible day usually mean misreporting.`,
  options: EX_ORDER.map(exclusionOption),
  live: exLive,
  liveAfter: (key) => {
    const o = EX[key]!;
    return {
      ...exLive,
      rows: {
        steps: viewOf(o.preview.views, "row_flow").after,
        byLevel: o.counts.by_level,
      },
      waits: ["energy adjustment", "models"],
    };
  },
  stacked: false,
};

// ── S4 · the wide transform ──────────────────────────────────────────────────

const BV = B.preview.views;
const tf = viewOf(BV, "table_focus");
const gd = viewOf(BV, "distribution");
const gl = viewOf(BV, "lineage");
const skew = /Skewness of (\S+) goes from ([\d.]+) to ([\d.]+); median (\S+) becomes ([\d.]+)/.exec(
  gd.caption,
);
const [, skewCol, skewBefore, , medianBefore] = skew ?? [];
const nCounts = B.count_columns.n;
const shown = tf.columns_after.length;

const rawTable: TableFocusView = {
  ...tf,
  title: `The ${shown} columns a log would change most`,
  caption: `The ${nCounts} count columns enter as recorded; shown: the ${shown} a log would change most.`,
  columns_after: tf.columns_before,
  rows: tf.rows.map((row) => ({ ...row, after: row.before })),
  changed: [],
  n_affected_columns: 0,
};
const rawDist: DistributionView = {
  ...gd,
  after: gd.before,
  after_label: gd.before_label,
  caption: `${skewCol} keeps its skewness of ${skewBefore} and its median of ${medianBefore}.`,
};
const rawLineage: LineageView = {
  ...gl,
  after: gl.before!,
  caption: `The ${nCounts} count columns enter as recorded, then scaled; nothing is rewritten.`,
};

const TRANSFORM_Q: QuestionModel = {
  id: "transform",
  dataset: GENOMICS,
  kicker: "Transform",
  question: `Should the ${nCounts} count columns be log-transformed?`,
  why: `Counts are right-skewed: \`${skewCol}\` has skewness ${skewBefore}, so a few large values set its scale.`,
  options: [
    {
      key: "log",
      label: "Log2(x + 1)",
      line: `All ${nCounts} count columns become ${B.transform.formula}; covariates pass through unchanged.`,
      preview: {
        label: `${B.transform.formula} on every count column`,
        basis: B.preview.basis,
        views: [
          { view: tf, chip: `${shown} of ${nCounts} shown` },
          { view: gd, chip: gd.after_label },
          { view: gl, chip: lineageChip(gl) },
        ],
      },
      sentence: `All \`${nCounts}\` count columns were transformed to \`${B.transform.formula}\`.`,
    },
    {
      key: "raw",
      label: "Keep raw counts",
      line: `Counts enter as recorded, then scaled; \`${skewCol}\` keeps its long right tail.`,
      preview: {
        label: "Raw counts",
        basis: B.preview.basis,
        views: [
          { view: rawTable, chip: `${shown} of ${nCounts} shown` },
          { view: rawDist, chip: rawDist.after_label },
          { view: rawLineage, chip: lineageChip(rawLineage) },
        ],
      },
      sentence: `The \`${nCounts}\` count columns were kept as raw counts.`,
    },
  ],
  live: {
    rows: { steps: [{ key: "loaded", label: "Rows loaded", n: B.dataset.n_rows, dropped: 0, reason: null, decision_id: null }] },
    lineage: gl.before!,
    lineageOpen: "transform: this question",
    waits: ["transform", "models"],
  },
  liveAfter: (key) => ({
    rows: { steps: [{ key: "loaded", label: "Rows loaded", n: B.dataset.n_rows, dropped: 0, reason: null, decision_id: null }] },
    lineage: key === "log" ? gl.after : gl.before!,
    waits: ["models"],
  }),
  stacked: true,
};

export const QUESTIONS: Record<Exclude<ScenarioId, "findings">, QuestionModel> = {
  energy: ENERGY_Q,
  exclusions: EXCLUSIONS_Q,
  transform: TRANSFORM_Q,
};

export const DATASETS: Record<ScenarioId, Dataset> = {
  energy: NHANES,
  exclusions: NHANES,
  findings: NHANES,
  transform: GENOMICS,
};

export const FACTS = { ENERGY, TOP, BOTTOM, kcalMax, exLive, energyLive };
