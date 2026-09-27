/**
 * The four scenarios, read from the real-data fixture (docs/turbotab-next/m1/explore/fixtures.json,
 * copied byte for byte beside this file because Vite serves nothing outside the frontend; a test
 * holds the two copies identical). Every number the prototype shows comes from here: the only
 * text authored in this file is the app's voice around those numbers — short labels, one-line
 * consequences, finding claims — and each of those builds its figures from the fixture.
 */
import raw from "./fixtures.json";
import { fmtInt } from "./format";
import type {
  DistributionView,
  EnergyOptionRaw,
  Evidence,
  ExclusionOptionRaw,
  FindingRaw,
  Fixture,
  LineageView,
  RelationshipView,
  RowFlowView,
  TableFocusView,
} from "./types";

export const FIXTURE = raw as unknown as Fixture;
const A = FIXTURE.scenario_a;
const B = FIXTURE.scenario_b;

// ── shared ────────────────────────────────────────────────────────────────────

function viewOf<K extends string, V extends { kind: string }>(
  views: V[] | undefined,
  kind: K,
): Extract<V, { kind: K }> | null {
  return (views?.find((v) => v.kind === kind) as Extract<V, { kind: K }>) ?? null;
}

function num(s: string): number {
  return Number(s.replace(/,/g, ""));
}

/** "`a`, `b` and `c`" */
function listCode(items: string[]): string {
  const c = items.map((i) => `\`${i}\``);
  return c.length <= 1 ? c.join("") : `${c.slice(0, -1).join(", ")} and ${c[c.length - 1]}`;
}

function must<T>(v: T | null | undefined, what: string): T {
  if (v === null || v === undefined) throw new Error(`fixture: ${what} is missing`);
  return v;
}

export interface Stats {
  mean: number;
  sd: number;
}

function stats(values: number[]): Stats {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return { mean, sd };
}

/** One state of a scatter: the y values of the same people, in their own column's units. */
export interface ScatterState {
  key: string;
  yLabel: string;
  ys: number[];
  stats: Stats;
  r: number | null;
}

export interface OptionBase {
  key: string;
  /** ≤ 4 words: what the user presses. */
  label: string;
  /** ≤ 16 words, the app's voice; backticks mark data. */
  consequence: string;
  usual: boolean;
  evidence: Evidence | null;
  /** When the option cannot be recorded here: the reason (≤ 16 words) — the card stays. */
  refused: string | null;
  /** The record button's label: says what pressing it does. */
  choose: string;
  /** The decision sentence it would record. */
  sentence: string;
}

// ── S1 · energy adjustment ────────────────────────────────────────────────────

const setup = A.setup;
const EA = A.energy_adjustment;
const focusR = must(EA.correlation_with_energy[0], "correlation_with_energy");
const nTrain = setup.n_train;

export interface EnergyOption extends OptionBase {
  method: string;
  estimandKind: string;
  standing: string | null;
  why: string;
  rel: RelationshipView | null;
  scatter: ScatterState | null;
  lineage: LineageView | null;
  dist: DistributionView | null;
  table: TableFocusView | null;
  matrixColumns: string[];
  kcalInModel: boolean;
  /** The refused method on a subset it accepts (partition on the macronutrient totals). */
  variant: EnergyOption | null;
  variantLever: string | null;
}

const ENERGY_VOICE: Record<
  string,
  { label: string; consequence: string; choose: string; sentence: string }
> = {
  residual: {
    label: "Residual",
    consequence: `Each nutrient keeps only what \`${EA.energy_column}\` does not explain; \`${EA.energy_column}\` leaves the model.`,
    choose: "Use the residual method",
    sentence: `Energy was adjusted by the residual method: each nutrient is replaced by its residual on \`${EA.energy_column}\`, fit on \`${fmtInt(nTrain)}\` training rows.`,
  },
  density_multivariate: {
    label: "Density + energy",
    consequence: `Each nutrient becomes grams per \`${EA.energy_column}\`; \`${EA.energy_column}\` stays in the model as its own term.`,
    choose: "Use density with energy",
    sentence: `Energy was adjusted by the multivariate density method: each nutrient is divided by \`${EA.energy_column}\`, and \`${EA.energy_column}\` stays in the model.`,
  },
  standard: {
    label: "Standard",
    consequence: `Nutrients enter unchanged and \`${EA.energy_column}\` enters beside them, so energy is held fixed.`,
    choose: "Use the standard model",
    sentence: `Energy was adjusted by the standard model: \`${EA.energy_column}\` enters the model beside the unchanged nutrients.`,
  },
  density: {
    label: "Density alone",
    consequence: `Each nutrient becomes grams per \`${EA.energy_column}\`, and \`${EA.energy_column}\` leaves the model.`,
    choose: "Use density alone",
    sentence: `Energy was adjusted by nutrient density: each nutrient is divided by \`${EA.energy_column}\`, and \`${EA.energy_column}\` leaves the model.`,
  },
  none: {
    label: "No adjustment",
    consequence: "Nutrients enter as absolute intakes; nothing is adjusted for total energy.",
    choose: "Leave energy unadjusted",
    sentence: "Nutrients enter the model as absolute intakes; energy was not adjusted.",
  },
  partition: {
    label: "Energy partition",
    consequence: `\`${EA.energy_column}\` splits into energy from each nutrient and energy from everything else.`,
    choose: "Use the partition model",
    sentence: `Energy was partitioned: \`${EA.energy_column}\` splits into energy from each nutrient and from everything else.`,
  },
};

/** Judgment is order, never absence (§11.9): the field default first, the refused one last. */
const ENERGY_ORDER = [
  "residual",
  "density_multivariate",
  "standard",
  "density",
  "none",
  "partition",
];
const USUAL_ENERGY = "residual";

function refusalReason(o: EnergyOptionRaw): string | null {
  if (o.applicable.ok) return null;
  // The fixture's reason names the nutrient without an energy factor; say only that.
  const m = /(\w+) carries no energy/.exec(o.applicable.reason);
  const culprit = m?.[1];
  return culprit
    ? `\`${culprit}\` has no energy factor, so \`${EA.energy_column}\` cannot be split across these nutrients.`
    : "Not applicable to these nutrients.";
}

function whyOf(o: EnergyOptionRaw): string {
  const caveat = o.method_card.caveats[0];
  const text = caveat ? `${o.estimand} ${caveat}` : o.estimand;
  return text.split(/\s+/).length <= 60 ? text : o.estimand;
}

function energyOption(o: EnergyOptionRaw, isVariant = false): EnergyOption {
  const voice = must(ENERGY_VOICE[o.method], `voice for ${o.method}`);
  const views = o.preview?.views;
  const rel = viewOf(views, "relationship");
  const table = viewOf(o.extra_views, "table_focus");
  const scatter: ScatterState | null = rel
    ? {
        key: isVariant ? `${o.method}-subset` : o.method,
        yLabel: rel.y_label_after,
        ys: rel.points_after.map((p) => p[1]),
        stats: stats(rel.points_after.map((p) => p[1])),
        r: rel.r_after,
      }
    : null;
  const subset = isVariant ? (o.nutrients ?? []) : [];
  return {
    key: isVariant ? `${o.method}-subset` : o.method,
    method: o.method,
    label: voice.label,
    consequence: isVariant
      ? `\`${EA.energy_column}\` splits into energy from ${listCode(subset)}, plus energy from everything else.`
      : voice.consequence,
    usual: o.method === USUAL_ENERGY && !isVariant,
    evidence: o.method_card.standing
      ? { status: o.method_card.standing, source: o.method_card.source }
      : null,
    refused: isVariant ? null : refusalReason(o),
    choose: isVariant ? `Use partition on these ${subset.length}` : voice.choose,
    sentence: isVariant
      ? `Energy was partitioned for ${listCode(subset)}: \`${EA.energy_column}\` splits into energy from each of them and from everything else.`
      : voice.sentence,
    estimandKind: o.method_card.kind,
    standing: o.method_card.standing,
    why: whyOf(o),
    rel,
    scatter,
    lineage: viewOf(views, "lineage"),
    dist: viewOf(views, "distribution"),
    table,
    matrixColumns: o.matrix_columns ?? [],
    kcalInModel: (o.matrix_columns ?? []).includes(EA.energy_column),
    variant: null,
    variantLever: null,
  };
}

function buildEnergy(): EnergyOption[] {
  const byMethod = new Map(EA.options.map((o) => [o.method, o]));
  return ENERGY_ORDER.map((m) => {
    const o = must(byMethod.get(m), `energy option ${m}`);
    const opt = energyOption(o);
    if (opt.refused) {
      const sub = EA.partition_on_macronutrient_totals;
      opt.variant = energyOption(sub, true);
      opt.variantLever = `Preview on ${listCode(sub.nutrients ?? [])}`;
    }
    return opt;
  });
}

const first = must(
  EA.options.find((o) => o.preview),
  "an energy preview",
);
const firstRel = must(viewOf(first.preview?.views, "relationship"), "relationship view");
const firstLineage = must(viewOf(first.preview?.views, "lineage"), "lineage view");
const firstTable = must(viewOf(first.extra_views, "table_focus"), "table focus view");

export const ENERGY = {
  question: "How should nutrient intakes be adjusted for total energy?",
  why: `\`${focusR.column}\` rises with \`${EA.energy_column}\` (r ${focusR.r.toFixed(2)}): people who eat more eat more of everything.`,
  energyColumn: EA.energy_column,
  focus: focusR.column,
  xs: firstRel.points_before.map((p) => p[0]),
  xLabel: firstRel.x_label,
  /** The data now: the nutrient as recorded, before any option. */
  base: {
    key: "base",
    yLabel: firstRel.y_label_before,
    ys: firstRel.points_before.map((p) => p[1]),
    stats: stats(firstRel.points_before.map((p) => p[1])),
    r: firstRel.r_before,
  } satisfies ScatterState,
  baseLineage: firstLineage.before,
  baseTable: firstTable,
  basis: first.preview?.basis ?? "",
  nShown: firstRel.points_before.length,
  nTrain,
  options: buildEnergy(),
};

// ── S2 · exclusions ───────────────────────────────────────────────────────────

export interface ExclusionOption extends OptionBase {
  flow: RowFlowView;
  dist: DistributionView;
  excluded: number;
  kept: number;
  byLevel: { level: string; below: number; above: number; low: number; high: number }[];
}

interface RangeRule {
  kind: string;
  column: string;
  low: number | null;
  high: number | null;
  by: { column: string; ranges: Record<string, [number, number]> } | null;
}

function rulesOf(o: ExclusionOptionRaw): RangeRule[] {
  return ((o.decision as { rules?: RangeRule[] }).rules ?? []) as RangeRule[];
}

const implausible = must(
  A.findings.findings.find((f) => f.id.endsWith("implausible_intake")),
  "implausible intake finding",
);
const observedRange = /Observed range ([\d,]+) to ([\d,]+)/.exec(implausible.detail);

function exclusionOption(o: ExclusionOptionRaw): ExclusionOption {
  const rules = rulesOf(o);
  const rule = rules[0];
  const flow = must(viewOf(o.preview.views, "row_flow"), "row flow");
  const dist = must(viewOf(o.preview.views, "distribution"), "distribution");
  const col = dist.column;
  const k = (v: number) => fmtInt(v);
  let label = "Keep every row";
  let consequence = `No intake is excluded; all \`${k(o.counts.final)}\` recalls stay, however implausible.`;
  let choose = "Keep every row";
  let sentence = `No rows were excluded for implausible intake; all \`${k(o.counts.final)}\` were kept.`;
  const byLevel: ExclusionOption["byLevel"] = [];
  if (rule?.by) {
    const levels = Object.entries(rule.by.ranges);
    const women = levels.find(([l]) => l === "female");
    const men = levels.find(([l]) => l === "male");
    label = "Sex-specific cut-offs";
    consequence =
      women && men
        ? `Women outside \`${k(women[1][0])}\`–\`${k(women[1][1])}\` and men outside \`${k(men[1][0])}\`–\`${k(men[1][1])}\` kcal a day are excluded.`
        : "Each level has its own plausible range; rows outside it are excluded.";
    choose = "Exclude by sex-specific range";
    sentence = `\`${k(o.counts.excluded)}\` rows outside sex-specific \`${col}\` ranges were excluded as implausible intakes.`;
    for (const [level, [low, high]] of levels) {
      const c = o.counts.by_level[level];
      if (c) byLevel.push({ level, below: c.below, above: c.above, low, high });
    }
  } else if (rule) {
    const low = rule.low ?? 0;
    const high = rule.high ?? 0;
    label = `${k(low)}–${k(high)} kcal`;
    consequence = `Recalls under \`${k(low)}\` or over \`${k(high)}\` kcal a day are excluded, for everyone.`;
    choose = `Exclude outside ${k(low)}–${k(high)}`;
    sentence = `\`${k(o.counts.excluded)}\` rows outside \`${k(low)}\`–\`${k(high)}\` kcal were excluded as implausible intakes.`;
    const c = o.counts.by_level.all;
    if (c) byLevel.push({ level: "all", below: c.below, above: c.above, low, high });
  }
  return {
    key: o.key,
    label,
    consequence,
    usual: false, // the pack offers both rules; neither is pre-selected (§0)
    evidence: o.evidence,
    refused: null,
    choose,
    sentence,
    flow,
    dist,
    excluded: o.counts.excluded,
    kept: o.counts.final,
    byLevel,
  };
}

const EX = A.exclusions;
const loaded = must(
  setup.cohort_steps.find((s) => s.key === "loaded"),
  "loaded step",
).n;

export const EXCLUSIONS = {
  question: "Which energy intakes are too implausible to keep?",
  why: `Recalls run from ${observedRange?.[1] ?? "0"} to \`${observedRange?.[2] ?? "?"}\` kcal a day; the extremes are misreporting, not diet.`,
  basis: must(EX.options[0], "exclusion option").preview.basis,
  loaded,
  options: EX.options.map(exclusionOption),
};

// ── S3 · findings ─────────────────────────────────────────────────────────────

export type Route = "energy" | "exclusions" | "roles";

export type Evidence1 =
  | { kind: "scatter" }
  | { kind: "tails"; low: number; high: number }
  | { kind: "levels"; levels: { label: string; n: number }[] }
  | { kind: "share"; parts: { label: string; n: number; tone: "on" | "off" | "blank" }[] };

export interface FindingCard {
  id: string;
  /** ≤ 20 words: the claim, in the app's voice, from the file's own numbers. */
  claim: string;
  lever: string;
  route: Route;
  evidence: Evidence | null;
  columns: string[];
  picture: Evidence1;
  sources: string[];
}

export interface FindingGroup {
  key: string;
  /** ≤ 8 words: what the paged card is about. */
  title: string;
  type: string;
  pages: FindingCard[];
  size: number;
}

const FINDINGS = A.findings.findings;
const nRows = A.dataset.n_rows;

function twoValues(f: FindingRaw): { label: string; n: number }[] {
  const out: { label: string; n: number }[] = [];
  const re = /'([^']+)' \(([\d,]+) rows\)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(f.detail))) out.push({ label: m[1]!, n: num(m[2]!) });
  return out;
}

function blanks(f: FindingRaw): number {
  const m = /([\d,]+) blank/.exec(f.detail);
  return m ? num(m[1]!) : 0;
}

function buildFindings(): { pushed: FindingCard[]; groups: FindingGroup[]; total: number } {
  const energy = must(
    FINDINGS.find((f) => f.id.endsWith("energy_adjustment")),
    "energy finding",
  );
  const lowHigh = /below (\d+) on ([\d,]+) record\(s\) and above (\d+) on ([\d,]+)/.exec(
    implausible.detail,
  );
  const nImplausible = /^([\d,]+) records/.exec(implausible.title)?.[1] ?? "";
  const gender = must(
    FINDINGS.find((f) => f.id === "binary_text__gender"),
    "gender finding",
  );
  const gLevels = twoValues(gender).sort((a, b) => a.label.localeCompare(b.label));

  const pushed: FindingCard[] = [
    {
      id: energy.id,
      claim: `\`${focusR.column}\` rises with \`kcal\` (r ${focusR.r.toFixed(2)}), so every nutrient effect is confounded by total intake.`,
      lever: "Adjust for energy",
      route: "energy",
      evidence: energy.evidence,
      columns: ["kcal", ...EA.nutrients],
      picture: { kind: "scatter" },
      sources: [energy.id],
    },
    {
      id: implausible.id,
      claim: lowHigh
        ? `\`${nImplausible}\` recalls report under \`${lowHigh[1]}\` or over \`${fmtInt(num(lowHigh[3]!))}\` kcal a day: \`${lowHigh[2]}\` low, \`${lowHigh[4]}\` high.`
        : implausible.title,
      lever: "Choose an exclusion rule",
      route: "exclusions",
      evidence: implausible.evidence,
      columns: ["kcal"],
      picture: {
        kind: "tails",
        low: lowHigh ? num(lowHigh[1]!) : 500,
        high: lowHigh ? num(lowHigh[3]!) : 5000,
      },
      sources: [implausible.id],
    },
    {
      id: gender.id,
      claim: "`gender` is text; which level becomes 1 sets the sign of its coefficient.",
      lever: "Choose the level coded 1",
      route: "roles",
      evidence: null,
      columns: ["gender"],
      picture: { kind: "levels", levels: gLevels },
      sources: [gender.id],
    },
  ];

  const flags = FINDINGS.filter((f) => /^binary_text__imputed_/.test(f.id));
  const flagPages: FindingCard[] = flags.map((f) => {
    const col = must(f.affected_columns[0], "flag column");
    const vals = twoValues(f);
    const on = vals.find((v) => v.label === "True")?.n ?? 0;
    return {
      id: f.id,
      claim: `\`${col}\` is true on \`${fmtInt(on)}\` of \`${fmtInt(nRows)}\` rows; it reads as a 0/1 flag.`,
      lever: "Link flags to their columns",
      route: "roles",
      evidence: null,
      columns: [col],
      picture: {
        kind: "share",
        parts: [
          { label: "true", n: on, tone: "on" },
          { label: "false", n: nRows - on, tone: "off" },
        ],
      },
      sources: [f.id],
    };
  });

  const medCols = [
    ...new Set(FINDINGS.filter((f) => /__meds_/.test(f.id)).flatMap((f) => f.affected_columns)),
  ].sort((a, b) => (a === "meds_hbp" ? -1 : b === "meds_hbp" ? 1 : a.localeCompare(b)));
  const medPages: FindingCard[] = medCols.map((col) => {
    const about = FINDINGS.filter((f) => f.affected_columns.includes(col) && /__meds_/.test(f.id));
    const typed = must(
      about.find((f) => f.id.startsWith("binary_text__")),
      `binary finding for ${col}`,
    );
    const vals = twoValues(typed);
    const on = vals.find((v) => v.label === "True")?.n ?? 0;
    const off = vals.find((v) => v.label === "False")?.n ?? 0;
    const blank = blanks(typed);
    return {
      id: `meds__${col}`,
      claim: `\`${col}\` holds True/False as text, \`${fmtInt(blank)}\` blank; the blanks are asked about later.`,
      lever: "Read as yes/no",
      route: "roles",
      evidence: null,
      columns: [col],
      picture: {
        kind: "share",
        parts: [
          { label: "true", n: on, tone: "on" },
          { label: "false", n: off, tone: "off" },
          { label: "blank", n: blank, tone: "blank" },
        ],
      },
      sources: about.map((f) => f.id),
    };
  });

  const groups: FindingGroup[] = [
    {
      key: "flags",
      title: `${flagPages.length} imputation flags are already true/false`,
      type: "imputation flags",
      pages: flagPages,
      size: flags.length,
    },
    {
      key: "meds",
      title: `${medPages.length} medication columns hold True/False as text`,
      type: "medication notices",
      pages: medPages,
      size: medPages.reduce((a, p) => a + p.sources.length, 0),
    },
  ];
  const covered = new Set([
    ...pushed.flatMap((p) => p.sources),
    ...groups.flatMap((g) => g.pages.flatMap((p) => p.sources)),
  ]);
  const uncovered = FINDINGS.filter((f) => !covered.has(f.id));
  if (uncovered.length)
    throw new Error(`fixture: findings not presented: ${uncovered.map((f) => f.id)}`);
  return { pushed, groups, total: FINDINGS.length };
}

export const FINDING_SET = {
  ...buildFindings(),
  nRows,
  nCols: A.dataset.n_cols,
  outcome: setup.outcome,
  roles: setup.state.roles,
  columns: A.dataset.columns,
  kcalHist: must(viewOf(EX.options[0]?.preview.views, "distribution"), "kcal histogram").before,
};

// ── S4 · the wide transform ───────────────────────────────────────────────────

const bViews = B.preview.views;
const bTable = must(viewOf(bViews, "table_focus"), "genomics table focus");
const bDist = must(viewOf(bViews, "distribution"), "genomics distribution");
const bLineage = must(viewOf(bViews, "lineage"), "genomics lineage");
const skew = /from ([\d.]+) to ([\d.]+); median ([\d.,]+) becomes ([\d.,]+)/.exec(bDist.caption);

export interface WideOption extends OptionBase {
  transformed: boolean;
  skew: number | null;
  median: string;
}

const nCounts = B.count_columns.n;
const skewBefore = skew ? Number(skew[1]) : null;
const skewAfter = skew ? Number(skew[2]) : null;

export const WIDE = {
  question: `Transform the \`${nCounts}\` gene count columns before modeling?`,
  why: `Counts are right-skewed: \`${bDist.column}\` has median \`${skew?.[3] ?? "?"}\` but reaches \`${fmtInt(bDist.before.edges[bDist.before.edges.length - 1] ?? 0)}\`.`,
  dataset: B.dataset,
  nCounts,
  firstCol: B.count_columns.first,
  lastCol: B.count_columns.last,
  target: B.target,
  formula: B.transform.formula,
  kindNote: B.transform.note,
  changeMetric: B.change_metric,
  basis: B.preview.basis,
  table: bTable,
  dist: bDist,
  lineage: bLineage,
  mostChanged: B.most_changed,
  options: [
    {
      key: "keep",
      label: "Keep raw counts",
      consequence: `Counts enter as recorded; \`${bDist.column}\` keeps a skewness of \`${skew?.[1] ?? "?"}\`.`,
      usual: false,
      evidence: null,
      refused: null,
      choose: "Keep raw counts",
      sentence: `The \`${nCounts}\` count columns enter the model as raw counts.`,
      transformed: false,
      skew: skewBefore,
      median: skew?.[3] ?? "",
    },
    {
      key: "log2",
      label: B.transform.formula,
      consequence: `All \`${nCounts}\` count columns become ${B.transform.formula}; \`${bDist.column}\`'s skewness falls to \`${skew?.[2] ?? "?"}\`.`,
      usual: false,
      evidence: null,
      refused: null,
      choose: `Apply ${B.transform.formula}`,
      sentence: `The \`${nCounts}\` count columns were transformed to ${B.transform.formula}.`,
      transformed: true,
      skew: skewAfter,
      median: skew?.[4] ?? "",
    },
  ] satisfies WideOption[],
};
