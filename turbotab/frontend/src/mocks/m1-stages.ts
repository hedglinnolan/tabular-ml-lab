/**
 * The M1 stage artifacts the mock server computes (M1_CONTRACT §3), from its own tables so
 * what the Record and the banner show is at least true of the mock data. Recognizers,
 * proposals and the participant flow follow the Python stages (turbotab/core/stages) closely
 * enough for the client to meet the same shapes and the same journeys; the fit and the
 * substitution curves are deterministic stand-ins with plausible numbers, not statistics.
 */
import type {
  CohortArtifact,
  DesignArtifact,
  EnergyMethod,
  ExclusionRule,
  FitArtifact,
  Lineage,
  LineageNode,
  MissingColumn,
  ProposalsArtifact,
  Role,
  RoleProposal,
  RolesArtifact,
  RowStep,
  ShelfArtifact,
  SplitArtifact,
  SubstitutionArtifact,
} from "../api/m1-types";
import type { DecisionRecord, Lens, ProjectState, Scalar, Task } from "../api/schema";
import type { MockColumn, MockDataset } from "./datasets";
import { findColumn, isNumericDtype, nMissing, nUnique } from "./stats";

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

/** Whether a column's blanks can be a level of their own: categorical, or two values at most
 *  (turbotab/core/models/pipeline.py `takes_level`). */
export function takesLevel(c: MockColumn): boolean {
  return ["boolean", "categorical", "text"].includes(c.dtype) || nUnique(c) <= 2;
}
const missing = (v: Scalar | undefined) => v === null || v === undefined || v === "";

// ── recognizing columns ──────────────────────────────────────────────────────

const NUTRIENT =
  /^(protein|carb|carbohydrate|carbs|sugar|sugars|fat|fat_total|total_fat|fat_sat|fat_mon|fat_poly|sfa|mufa|pufa|fiber|fibre|sodium|alcohol|cholesterol)(_g|_mg)?$/i;
const SHARE = /_(pct|percent)_kcal$|_per_kcal$|_density$/i;
const ENERGY = /kcal|energy|calor/i;
const IDENT = /(^|_)(id|seqn)$|^seqn$|^(participant|subject|person|patient|sample)(_?id)?$/i;
const DESIGN = /^(wt|sdmv)|(^|_)(psu|strata|stratum|survey_weight)$/i;
const FLAG = /^(imputed|flag|is_imputed)_(.+)$|^(.+)_(imputed|flag|imp)$/i;
const TIME = /date|year|cycle|visit|time|recall_number|wave/i;
const DEMOGRAPHIC = /^(age|sex|gender|race|ethnicity|education|income|smoking|smoker)$/i;
const CLINICAL =
  /^(bmi|bp_sys|bp_di|sbp|dbp|weight|height|waist|hdl|ldl|triglycerides|crp|hba1c|glucose)$/i;
const MEDS = /^(meds?_|medication)/i;

/** Atwater factors (kcal per gram) for the nutrients that carry energy. */
const ATWATER: { pattern: RegExp; factor: number; family: string }[] = [
  { pattern: /^protein/i, factor: 4, family: "protein" },
  { pattern: /^(carb|carbohydrate|carbs)/i, factor: 4, family: "carbohydrate" },
  {
    pattern: /^(fat|fat_total|total_fat|fat_sat|fat_mon|fat_poly|sfa|mufa|pufa)/i,
    factor: 9,
    family: "fat",
  },
  { pattern: /^alcohol/i, factor: 7, family: "alcohol" },
];

/** A nutrient amount that carries energy (not a share of energy; sugar is counted in carbohydrate). */
export function energyBearing(column: string): boolean {
  if (SHARE.test(column) || ENERGY.test(column)) return false;
  if (/^sugars?(_g)?$/i.test(column)) return true; // energy-bearing, but no Atwater factor of its own
  return ATWATER.some((a) => a.pattern.test(column));
}

function atwater(column: string): number | null {
  if (/^sugars?(_g)?$/i.test(column)) return null;
  return ATWATER.find((a) => a.pattern.test(column))?.factor ?? null;
}

const PARENTS: [RegExp, RegExp][] = [
  [/^sugars?(_g)?$/i, /^(carb|carbohydrate|carbs)(_g)?$/i],
  [/^(fat_sat|fat_mon|fat_poly|sfa|mufa|pufa)(_g)?$/i, /^(fat_total|total_fat|fat)(_g)?$/i],
];

function numbersOf(col: MockColumn): (number | null)[] {
  return col.values.map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
}

/** §12.5: the parent a component nutrient is part of, when the data agree (≤ parent on ≥ 99% of rows). */
function nestedIn(ds: MockDataset, column: string): string | null {
  for (const [child, parent] of PARENTS) {
    if (!child.test(column)) continue;
    const p = ds.columns.find((c) => parent.test(c.name));
    const c = findColumn(ds, column);
    if (!p || !c) continue;
    const a = numbersOf(c);
    const b = numbersOf(p);
    let n = 0;
    let inside = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] === null || b[i] === null) continue;
      n += 1;
      if (a[i]! <= b[i]! + 1e-9) inside += 1;
    }
    if (n > 0 && inside / n >= 0.99) return p.name;
  }
  return null;
}

function propose(ds: MockDataset, col: MockColumn, lens: Lens[]): RoleProposal {
  const name = col.name;
  const base = { column: name, linked_to: null, unit: null, nested_in: null } as const;
  const unique = nUnique(col);
  const flag = FLAG.exec(name);
  if (flag) {
    const target = flag[2] ?? flag[3] ?? null;
    const linked = target && findColumn(ds, target) ? target : null;
    return {
      ...base,
      proposed: "flag",
      confidence: "high",
      reason: linked
        ? `Marks which \`${linked}\` values were filled in.`
        : "A flag on another column's values.",
      linked_to: linked,
    };
  }
  const textual = col.dtype === "categorical" || col.dtype === "text";
  if (IDENT.test(name) || (textual && unique >= 0.95 * ds.nRows)) {
    return {
      ...base,
      proposed: "identifier",
      confidence: "high",
      reason: "Id-like name, a different value on nearly every row.",
    };
  }
  if (DESIGN.test(name)) {
    return {
      ...base,
      proposed: "design",
      confidence: "high",
      reason: "A survey weight, stratum or cluster by its name.",
    };
  }
  if (TIME.test(name)) {
    return {
      ...base,
      proposed: "time",
      confidence: "high",
      reason: "Names when a row was measured: a date, cycle or visit.",
    };
  }
  if (isNumericDtype(col.dtype) && ENERGY.test(name) && !SHARE.test(name)) {
    return {
      ...base,
      proposed: "energy",
      confidence: "high",
      reason: "Total energy intake by its name.",
      unit: /kj/i.test(name) ? "kJ" : "kcal",
    };
  }
  if (isNumericDtype(col.dtype) && (NUTRIENT.test(name) || SHARE.test(name))) {
    return {
      ...base,
      proposed: "exposure",
      confidence: SHARE.test(name) ? "medium" : "high",
      reason: SHARE.test(name)
        ? "A nutrient's share of energy, not an amount."
        : "A nutrient amount, by its name.",
      unit: SHARE.test(name) ? "% kcal" : /_mg$/i.test(name) ? "mg" : "g",
      nested_in: nestedIn(ds, name),
    };
  }
  if (DEMOGRAPHIC.test(name)) {
    return {
      ...base,
      proposed: "covariate",
      confidence: "high",
      reason: "A demographic, the usual adjustment set.",
    };
  }
  if (CLINICAL.test(name)) {
    return {
      ...base,
      proposed: "covariate",
      confidence: "medium",
      reason: "A clinical measure, read as a covariate.",
    };
  }
  if (MEDS.test(name)) {
    return {
      ...base,
      proposed: "covariate",
      confidence: "medium",
      reason: "Medication use, read as a covariate.",
    };
  }
  const omics = lens.includes("genomics") || lens.includes("metabolomics");
  if (omics && isNumericDtype(col.dtype)) {
    return {
      ...base,
      proposed: "exposure",
      confidence: "medium",
      reason: "A measured feature of the assay.",
    };
  }
  if (unique <= 1) {
    return {
      ...base,
      proposed: "excluded",
      confidence: "high",
      reason: "One value on every row: it cannot inform a model.",
    };
  }
  if (col.dtype === "text") {
    return {
      ...base,
      proposed: "excluded",
      confidence: "medium",
      reason: "Free text; a model cannot read it as is.",
    };
  }
  return {
    ...base,
    proposed: "excluded",
    confidence: "low",
    reason: "Not recognized; left out until you include it.",
  };
}

export function rolesArtifact(ds: MockDataset, state: ProjectState): RolesArtifact {
  const lens = state.lens ?? [];
  const columns = ds.columns
    .filter((c) => c.name !== state.target)
    .map((c) => propose(ds, c, lens));
  let repeats: RolesArtifact["repeats"] = null;
  const id = columns.find((c) => c.proposed === "identifier");
  if (id) {
    const col = findColumn(ds, id.column)!;
    const counts = new Map<Scalar, number>();
    for (const v of col.values) counts.set(v, (counts.get(v) ?? 0) + 1);
    const max = Math.max(...counts.values());
    if (max > 1) repeats = { column: id.column, n_units: counts.size, max_rows_per_unit: max };
  }
  return { columns, repeats };
}

/** The roles in force: the record's, else what the roles stage proposes. */
export function rolesInForce(ds: MockDataset, state: ProjectState): Record<string, Role> {
  if (state.roles) return state.roles as Record<string, Role>;
  return Object.fromEntries(rolesArtifact(ds, state).columns.map((c) => [c.column, c.proposed]));
}

const PREDICTOR: Role[] = ["exposure", "energy", "covariate"];

// ── proposals ────────────────────────────────────────────────────────────────

const EXCLUSION_EVIDENCE = {
  status: "CONVENTION",
  source: "research/NUTRITION_PACK.md#02 · Implausible intake exclusions",
};
const ENERGY_EVIDENCE = {
  status: "CONVENTION",
  source: "research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature",
};

function levelKey(v: Scalar): string {
  return v === null ? "" : String(v).trim();
}

/** True where `rule` removes the row (missing values are never removed by a range). */
export function ruleExcludes(ds: MockDataset, rule: ExclusionRule, i: number): boolean {
  const col = findColumn(ds, rule.column);
  const v = col?.values[i];
  if (typeof v !== "number") return false;
  let low = rule.low;
  let high = rule.high;
  if (rule.by) {
    const lv = levelKey(findColumn(ds, rule.by.column)?.values[i] ?? null);
    const range = rule.by.ranges[lv];
    if (range) [low, high] = range;
  }
  return (low !== null && v < low) || (high !== null && v > high);
}

function sexColumn(ds: MockDataset): { column: string; female: string[]; male: string[] } | null {
  const col = ds.columns.find((c) => /^(sex|gender)$/i.test(c.name));
  if (!col) return null;
  const levels = [...new Set(col.values.filter((v) => !missing(v)).map(levelKey))];
  const female = levels.filter((l) => /^(f|female|woman|women)$/i.test(l));
  const male = levels.filter((l) => /^(m|male|man|men)$/i.test(l));
  return female.length && male.length ? { column: col.name, female, male } : null;
}

function measured(ds: MockDataset, target: string | null): boolean[] {
  const col = target ? findColumn(ds, target) : undefined;
  return Array.from({ length: ds.nRows }, (_, i) => (col ? !missing(col.values[i]) : true));
}

function pearson(a: (number | null)[], b: (number | null)[]): number | null {
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === null || b[i] === null) continue;
    n += 1;
    sa += a[i]!;
    sb += b[i]!;
  }
  if (n < 3) return null;
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === null || b[i] === null) continue;
    const da = a[i]! - ma;
    const db = b[i]! - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  return va && vb ? cov / Math.sqrt(va * vb) : null;
}

const METHODS: EnergyMethod[] = [
  "none",
  "standard",
  "residual",
  "density_multivariate",
  "density",
  "partition",
];

export function proposalsArtifact(ds: MockDataset, state: ProjectState): ProposalsArtifact {
  const roles = rolesInForce(ds, state);
  const missingCols = missingReading(ds, roles, state.target);
  if (!(state.lens ?? []).includes("dietary")) {
    return {
      exclusions: [],
      energy: null,
      missing: missingCols,
      coach: {},
      n_base: ds.nRows,
      basis: "Nothing is proposed: the dietary lens is not chosen.",
    };
  }
  const energy =
    Object.keys(roles).find((c) => roles[c] === "energy") ??
    ds.columns.find((c) => isNumericDtype(c.dtype) && ENERGY.test(c.name) && !SHARE.test(c.name))
      ?.name ??
    null;
  const nutrients = Object.keys(roles).filter(
    (c) => roles[c] === "exposure" && c !== state.target && energyBearing(c),
  );
  const base = measured(ds, state.target);
  const exclusions: ProposalsArtifact["exclusions"] = [];
  if (energy) {
    const sex = sexColumn(ds);
    const screens: [string, string, ExclusionRule][] = [];
    if (sex) {
      const ranges: Record<string, [number | null, number | null]> = {};
      for (const f of sex.female) ranges[f] = [500, 3500];
      for (const m of sex.male) ranges[m] = [800, 4200];
      screens.push([
        "willett_by_sex",
        "Willett, by sex: women 500–3,500 and men 800–4,200 kcal a day",
        {
          kind: "range",
          column: energy,
          low: null,
          high: null,
          by: { column: sex.column, ranges },
          reason: "implausible intakes (Willett's sex-specific cut-offs)",
        },
      ]);
    }
    for (const [lo, hi] of [
      [500, 5000],
      [500, 3500],
    ] as const) {
      screens.push([
        `sex_neutral_${lo}_${hi}`,
        `Sex-neutral: ${fmt(lo)}–${fmt(hi)} kcal a day`,
        {
          kind: "range",
          column: energy,
          low: lo,
          high: hi,
          by: null,
          reason: `implausible intakes (sex-neutral ${fmt(lo)}–${fmt(hi)} kcal a day)`,
        },
      ]);
    }
    for (const [key, label, rule] of screens) {
      let affected = 0;
      for (let i = 0; i < ds.nRows; i++) if (base[i] && ruleExcludes(ds, rule, i)) affected += 1;
      exclusions.push({ key, label, rule, affected, evidence: { ...EXCLUSION_EVIDENCE } });
    }
  }
  const applicability: Record<string, { ok: boolean; reason: string }> = {};
  const list = nutrients.map((n) => `\`${n}\``);
  const listed =
    list.length <= 1
      ? (list[0] ?? "")
      : `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`;
  for (const m of METHODS) {
    if (m === "none") {
      applicability[m] = {
        ok: true,
        reason: "Nutrients enter the model as absolute intakes; nothing is adjusted.",
      };
    } else if (!energy || nutrients.length === 0) {
      applicability[m] = {
        ok: false,
        reason: !energy
          ? "No column has the energy role, so there is no total energy to adjust against."
          : "No exposure is a nutrient that carries energy, so there is nothing to adjust.",
      };
    } else if (m === "partition") {
      const missingFactor = nutrients.find((n) => atwater(n) === null);
      applicability[m] = missingFactor
        ? {
            ok: false,
            reason: `Energy partition splits total energy into kcal from each chosen nutrient and kcal from everything else, so every chosen nutrient must carry energy in a known unit. \`${missingFactor}\` carries no energy of its own: no Atwater factor is known for it.`,
          }
        : {
            ok: true,
            reason: `\`${energy}\` is split into kcal from ${listed} and kcal from everything else.`,
          };
    } else {
      const reasons: Record<string, string> = {
        standard: `\`${energy}\` stays in the model beside ${listed}.`,
        residual: `Each of ${listed} is regressed on \`${energy}\` within the fitting rows.`,
        density_multivariate: `Each of ${listed} is divided by \`${energy}\`, which stays in the model.`,
        density: `Each of ${listed} is divided by \`${energy}\`, which leaves the model.`,
      };
      applicability[m] = { ok: true, reason: reasons[m]! };
    }
  }
  const r_with_energy: Record<string, number> = {};
  if (energy) {
    const e = numbersOf(findColumn(ds, energy)!);
    for (const n of nutrients) {
      const r = pearson(e, numbersOf(findColumn(ds, n)!));
      if (r !== null) r_with_energy[n] = Math.round(r * 1000) / 1000;
    }
  }
  const notes: string[] = [];
  const parts = nutrients.filter((n) => nestedIn(ds, n));
  const byParent = new Map<string, string[]>();
  for (const p of parts) {
    const parent = nestedIn(ds, p)!;
    byParent.set(parent, [...(byParent.get(parent) ?? []), p]);
  }
  for (const [parent, kids] of byParent) {
    const k = kids.map((x) => `\`${x}\``);
    const named = k.length === 1 ? k[0] : `${k.slice(0, -1).join(", ")} and ${k[k.length - 1]}`;
    notes.push(
      `${named} ${kids.length === 1 ? "is a part" : "are parts"} of \`${parent}\`: choosing them together counts its energy twice in a partition or a substitution.`,
    );
  }
  const sex = sexColumn(ds);
  const usual: EnergyMethod | null = applicability.residual?.ok
    ? "residual"
    : applicability.standard?.ok
      ? "standard"
      : null;
  return {
    exclusions,
    energy: {
      energy_column: energy,
      nutrients,
      strata_candidates: sex ? [sex.column] : [],
      applicability,
      usual,
      usual_evidence: usual ? { ...ENERGY_EVIDENCE } : null,
      r_with_energy,
      notes,
      not_adjusted: [],
    },
    missing: missingCols,
    coach: cardLines(ds, energy, base),
    n_base: base.filter(Boolean).length,
    basis: `Counted on the ${fmt(base.filter(Boolean).length)} rows with \`${state.target ?? "the outcome"}\` measured.`,
  };
}

/** The decision cards' coach lines, at most one per card (turbotab/core/coach.py `card_lines`):
 *  the exclusions card names the rows outside the plausible-intake range. */
function cardLines(
  ds: MockDataset,
  energy: string | null,
  base: boolean[],
): ProposalsArtifact["coach"] {
  const col = energy ? findColumn(ds, energy) : undefined;
  if (!col) return {};
  let below = 0;
  let above = 0;
  col.values.forEach((v, i) => {
    if (!base[i] || typeof v !== "number") return;
    if (v < 500) below += 1;
    else if (v > 5000) above += 1;
  });
  const rows = (n: number) => `\`${fmt(n)}\` ${n === 1 ? "row" : "rows"}`;
  const text = below
    ? `${rows(below)} below \`500\` kcal: likely under-reporting`
    : above
      ? `${rows(above)} above \`5,000\` kcal: likely over-reporting`
      : null;
  return text ? { exclusions: { text, anchor: { kind: "column", ref: col.name } } } : {};
}

/** §12.4: each predictor's blanks, and whether they likely mean "not asked". */
function missingReading(
  ds: MockDataset,
  roles: Record<string, Role>,
  target: string | null,
): ProposalsArtifact["missing"] {
  const columns: MissingColumn[] = [];
  for (const [name, role] of Object.entries(roles)) {
    if (!PREDICTOR.includes(role) || name === target) continue;
    const col = findColumn(ds, name);
    if (!col) continue;
    const n = nMissing(col);
    if (n === 0) continue;
    const share = n / ds.nRows;
    const yesNo = col.dtype === "boolean" || nUnique(col) === 2;
    const likely = share >= 0.5 && (yesNo || MEDS.test(name));
    columns.push({
      column: name,
      n_missing: n,
      share,
      likely_not_asked: likely,
      reason: likely
        ? `\`${name}\` is a yes/no answer blank on ${Math.round(share * 100)}% of rows; blanks there usually mean the question was not asked.`
        : `\`${name}\` is blank on ${fmt(n)} rows.`,
    });
  }
  columns.sort((a, b) => b.share - a.share);
  const notAsked = columns.filter((c) => c.likely_not_asked).map((c) => c.column);
  if (!notAsked.length) return { columns, leave_out: null };
  const cols = notAsked.map((c) => findColumn(ds, c)).filter((c): c is MockColumn => !!c);
  let n = 0;
  for (let i = 0; i < ds.nRows; i++) if (cols.some((c) => missing(c.values[i]))) n += 1;
  return { columns, leave_out: { columns: notAsked, n_rows: n, share: n / ds.nRows } };
}

// ── the participant flow and the split ───────────────────────────────────────

export interface CohortRows {
  artifact: CohortArtifact;
  rows: Uint8Array;
  measured: Uint8Array;
}

export function cohort(
  ds: MockDataset,
  state: ProjectState,
  records: DecisionRecord[],
  dropColumns: string[],
): CohortRows {
  const target = state.target!;
  const tcol = findColumn(ds, target);
  const rows = new Uint8Array(ds.nRows).fill(1);
  const meas = new Uint8Array(ds.nRows);
  const steps: RowStep[] = [
    {
      key: "loaded",
      label: "Rows loaded",
      n: ds.nRows,
      dropped: 0,
      reason: null,
      decision_id: null,
    },
  ];
  let n = ds.nRows;
  let dropped = 0;
  for (let i = 0; i < ds.nRows; i++) {
    if (tcol && missing(tcol.values[i])) {
      rows[i] = 0;
      dropped += 1;
    } else meas[i] = 1;
  }
  n -= dropped;
  const targetRec = [...records].reverse().find((r) => r.decision.kind === "set_target");
  steps.push({
    key: "outcome_measured",
    label: `${target} measured`,
    n,
    dropped,
    reason: `${target} is missing`,
    decision_id: targetRec?.id ?? null,
  });
  const exRec = [...records].reverse().find((r) => r.decision.kind === "set_exclusions");
  for (const rule of state.exclusions ?? []) {
    let d = 0;
    for (let i = 0; i < ds.nRows; i++) {
      if (rows[i] && ruleExcludes(ds, rule, i)) {
        rows[i] = 0;
        d += 1;
      }
    }
    n -= d;
    const range = rule.by
      ? `${rule.column} within ranges by ${rule.by.column}`
      : `${rule.column} within ${rule.low === null ? "−∞" : fmt(rule.low)}–${rule.high === null ? "∞" : fmt(rule.high)}`;
    steps.push({
      key: `exclude_${rule.column}`,
      label: range,
      n,
      dropped: d,
      reason: rule.reason,
      decision_id: exRec?.id ?? null,
    });
  }
  const roles = (state.roles ?? {}) as Record<string, Role>;
  const predictors = Object.keys(roles).filter(
    (c) => PREDICTOR.includes(roles[c]!) && c !== target && !dropColumns.includes(c),
  );
  if (state.missing?.strategy === "complete_case") {
    // M2 (constitution §07): a column whose blanks become their own `Missing` level drops no row.
    const levels = state.missing.categorical === "missing_category";
    const cols = predictors
      .map((p) => findColumn(ds, p))
      .filter((c): c is MockColumn => !!c && !(levels && takesLevel(c)));
    let d = 0;
    for (let i = 0; i < ds.nRows; i++) {
      if (!rows[i]) continue;
      if (cols.some((c) => missing(c.values[i]))) {
        rows[i] = 0;
        d += 1;
      }
    }
    n -= d;
    const missRec = [...records].reverse().find((r) => r.decision.kind === "set_missing");
    steps.push({
      key: "complete_cases",
      label: "Complete cases",
      n,
      dropped: d,
      reason: "a predictor is missing",
      decision_id: missRec?.id ?? null,
    });
  }
  return { artifact: { steps, n_final: n, predictors }, rows, measured: meas };
}

/** A row's place in [0, 1), fixed by the seed: the same split can be drawn again. */
function unit(seed: number, key: string): number {
  let h = 0x811c9dc5 ^ seed;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  return (h >>> 0) / 4294967296;
}

export function split(
  ds: MockDataset,
  state: ProjectState,
  c: CohortRows,
  roles: RolesArtifact | null,
  task: Task | null,
): SplitArtifact {
  const spec = state.split!;
  const repeats = roles?.repeats ?? null;
  const idCol = repeats ? findColumn(ds, repeats.column) : undefined;
  let nTrain = 0;
  let nHold = 0;
  let nMeasured = 0;
  for (let i = 0; i < ds.nRows; i++) {
    if (!c.measured[i]) continue;
    nMeasured += 1;
    const key = idCol ? String(idCol.values[i]) : String(i);
    const sealed = spec.holdout > 0 && unit(spec.seed, key) < spec.holdout;
    if (!c.rows[i]) continue;
    if (sealed) nHold += 1;
    else nTrain += 1;
  }
  const target = state.target ?? "the outcome";
  return {
    n_train: nTrain,
    n_holdout: nHold,
    holdout: spec.holdout,
    seed: spec.seed,
    folds: spec.folds,
    grouped_by: repeats?.column ?? null,
    n_groups: repeats?.n_units ?? null,
    stratified: task === "binary" || task === "multiclass",
    note:
      spec.holdout > 0
        ? `Held-out rows are drawn from the ${fmt(nMeasured)} rows with \`${target}\` measured, so no later answer moves a row across the seal.`
        : "No rows are held out; every score comes from cross-validation.",
    basis: sealBasis(state, repeats),
    chronology: null,
    exploratory: sealBasis(state, repeats).exploratory,
  };
}

/** The seal's basis as the server decides it (turbotab/core/seal.py), from the grain and repeats. */
function sealBasis(
  state: ProjectState,
  repeats: RolesArtifact["repeats"] | null,
): SplitArtifact["basis"] {
  const grain = state.grain?.grain ?? null;
  if (repeats && grain !== "one_row_per_unit")
    return {
      state: "grouped",
      column: repeats.column,
      label: `grouped by \`${repeats.column}\``,
      sentence: `Held out by \`${repeats.column}\`: each of the \`${fmt(repeats.n_units)}\` units sits wholly on one side, so no unit is both trained on and scored.`,
      exploratory: false,
      source: grain === "repeated" ? "grain" : "roles",
      n_units: repeats.n_units,
    };
  if (repeats)
    return {
      state: "abandoned",
      column: repeats.column,
      label: "repetition found but grouping abandoned",
      sentence: `Each row was said to be a different unit, but \`${repeats.column}\` repeats; the held-out rows were drawn by row as answered. Treat held-out scores as exploratory.`,
      exploratory: true,
      source: "grain",
      n_units: repeats.n_units,
    };
  if (grain === "one_row_per_unit")
    return {
      state: "one_row_per_unit",
      column: null,
      label: "one row per unit",
      sentence:
        "Held out by row: each row was said to be a different unit, and no identifier repeats.",
      exploratory: false,
      source: "grain",
      n_units: null,
    };
  return {
    state: "undetermined",
    column: null,
    label: "undetermined",
    sentence:
      "Held out by row, because whether a unit can appear in more than one row was not answered. This is not a verified clean split: treat held-out scores as exploratory.",
    exploratory: true,
    source: null,
    n_units: null,
  };
}

// ── the shelf, the design, the fit ───────────────────────────────────────────

const FAMILY = {
  linear: {
    label: {
      regression: "Linear regression",
      binary: "Logistic regression",
      multiclass: "Multinomial logistic regression",
    },
    bias: "Straight-line, additive effects; one reportable coefficient per predictor.",
  },
  elastic_net: {
    label: {
      regression: "Elastic net",
      binary: "Elastic net (logistic)",
      multiclass: "Elastic net (multinomial)",
    },
    bias: "Linear, with correlated predictors shrunk together toward zero.",
  },
  boosted_trees: {
    label: {
      regression: "Gradient-boosted trees",
      binary: "Gradient-boosted trees",
      multiclass: "Gradient-boosted trees",
    },
    bias: "Steps, curves and interactions found by many shallow trees; no coefficients.",
  },
} as const;
type FamilyKey = keyof typeof FAMILY;

export function familyLabel(key: string, task: Task | null): string {
  const f = FAMILY[key as FamilyKey];
  return f ? f.label[task ?? "regression"] : key;
}

/** Seconds per million cells of the model matrix, five folds and a final fit (mock rates). */
const COST_PER_MILLION_CELLS: Record<FamilyKey, number> = {
  linear: 0.8,
  elastic_net: 12,
  boosted_trees: 25,
};

/** The server's `cost.duration` + `cost.say`: a fit's length, naming the width or length only
 * when the fit is long enough for that to matter (NOTEWORTHY_SECONDS). */
function cost(seconds: number, n: number, p: number): { estimate_seconds: number; estimate: string } {
  const about = (k: number, unit: string) => `about ${k} ${unit}${k === 1 ? "" : "s"}`;
  const text =
    seconds < 1
      ? "under a second"
      : seconds < 60
        ? about(seconds < 10 ? Math.round(seconds) : 5 * Math.round(seconds / 5), "second")
        : seconds < 3600
          ? about(Math.max(1, Math.round(seconds / 60)), "minute")
          : about(Math.max(1, Math.round(seconds / 3600)), "hour");
  const named =
    seconds < 10
      ? text
      : p > 200
        ? `${text} at \`${fmt(p)}\` columns`
        : n >= 100_000
          ? `${text} on \`${fmt(n)}\` rows`
          : text;
  return { estimate_seconds: seconds, estimate: named };
}

export function shelf(state: ProjectState, c: CohortArtifact, task: Task | null): ShelfArtifact {
  const p = Math.max(1, c.predictors.length);
  const n = c.n_final;
  const inference = state.purpose === "inference";
  const order: FamilyKey[] = inference
    ? ["linear", "elastic_net", "boosted_trees"]
    : ["elastic_net", "linear", "boosted_trees"];
  return {
    families: order.map((key, i) => {
      const concerns: string[] = [];
      if (n / p < 20 && key !== "elastic_net")
        concerns.push(`Only ${fmt(n / p)} rows per predictor; estimates will be noisy.`);
      if (key === "boosted_trees" && n < 5000)
        concerns.push(`${fmt(n)} rows is little for trees; they may fit noise.`);
      if (key === "boosted_trees" && inference)
        concerns.push("No coefficients to report; inference reads coefficients.");
      return {
        key,
        label: familyLabel(key, task),
        rank: i + 1,
        fit: concerns.length === 0 ? "good" : concerns.length === 1 ? "fair" : "poor",
        concerns,
        inductive_bias: FAMILY[key].bias,
        // §12.6: the measured cost before a fit (the mock's rough per-cell rates), worded as the
        // server's cost.say words it.
        ...cost(Math.max(0.3, ((n * p) / 1e6) * COST_PER_MILLION_CELLS[key]), n, p),
      };
    }),
    basis: `Ranked for ${state.purpose ?? "prediction"} on ${fmt(n)} rows and ${fmt(p)} predictors.`,
  };
}

const ESTIMAND: Record<EnergyMethod, string | null> = {
  none: null,
  standard: "More of the nutrient in place of other calories, at the same total energy.",
  residual:
    "More of the nutrient in place of other calories, at the same total energy, in its own units.",
  density_multivariate: "A higher share of energy from the nutrient, at the same total energy.",
  density:
    "A higher share of energy from the nutrient; the effect of total energy is not held fixed.",
  partition: "Adding calories from the nutrient, with calories from everything else held fixed.",
};

const OPERATION: Record<EnergyMethod, string> = {
  none: "kept",
  standard: "kept beside energy",
  residual: "energy-adjusted (residual)",
  density_multivariate: "per kcal (density, energy kept)",
  density: "per kcal (density)",
  partition: "kcal from the nutrient (partition)",
};

export function design(
  ds: MockDataset,
  state: ProjectState,
  c: CohortArtifact,
  nTrain: number,
): DesignArtifact {
  const roles = (state.roles ?? {}) as Record<string, Role>;
  const ea = state.energy_adjustment;
  const method: EnergyMethod = ea?.method ?? "none";
  const energy = ea?.energy_column ?? null;
  const nutrients = new Set(method === "none" ? [] : (ea?.nutrients ?? []));
  const nodes: LineageNode[] = [];
  const links: Lineage["links"] = [];
  const node = (
    id: string,
    column: string | null,
    lane: LineageNode["lane"],
    role: Role | null,
    label: string,
    formula: string | null = null,
    count = 1,
  ): void => {
    nodes.push({ id, column, lane, role, label, formula, group: null, count });
  };
  const collapse = c.predictors.length > 24;
  const groups = new Map<Role, string[]>();
  for (const p of c.predictors) groups.set(roles[p]!, [...(groups.get(roles[p]!) ?? []), p]);
  const encoded = (p: string): string => {
    const col = findColumn(ds, p);
    if (col && (col.dtype === "categorical" || col.dtype === "text")) {
      const levels = [...new Set(col.values.filter((v) => !missing(v)).map(String))].sort();
      return `${p}_${levels[levels.length - 1] ?? "1"}`;
    }
    return p;
  };
  if (collapse) {
    for (const [role, cols] of groups) {
      const id = `raw:${role}`;
      nodes.push({
        id,
        column: null,
        lane: "raw",
        role,
        label: `${cols.length} ${role} columns`,
        formula: null,
        group: role,
        count: cols.length,
      });
      const adjusted = role === "exposure" && method !== "none";
      if (adjusted) {
        nodes.push({
          id: `adj:${role}`,
          column: null,
          lane: "adjusted",
          role,
          label: `${cols.length} adjusted`,
          formula: null,
          group: role,
          count: cols.length,
        });
        links.push({ source: id, target: `adj:${role}`, operation: OPERATION[method] });
      }
      nodes.push({
        id: `mx:${role}`,
        column: null,
        lane: "matrix",
        role,
        label: `${cols.length} inputs`,
        formula: null,
        group: role,
        count: cols.length,
      });
      links.push({
        source: adjusted ? `adj:${role}` : id,
        target: `mx:${role}`,
        operation: "kept",
      });
    }
  } else {
    for (const p of c.predictors) {
      const role = roles[p]!;
      node(`raw:${p}`, p, "raw", role, p);
      let from = `raw:${p}`;
      if (nutrients.has(p)) {
        const name =
          method === "residual"
            ? `${p}_adj`
            : method === "partition"
              ? `kcal_from_${p}`
              : `${p}_per_kcal`;
        const formula =
          method === "residual"
            ? `${p} − ĝ(${energy})`
            : method === "partition"
              ? `${p} × ${atwater(p) ?? "?"}`
              : method === "standard"
                ? null
                : `${p} / ${energy}`;
        if (method !== "standard") {
          node(`adj:${p}`, name, "adjusted", role, name, formula);
          links.push({ source: from, target: `adj:${p}`, operation: OPERATION[method] });
          from = `adj:${p}`;
        }
      }
      // Residual and density take energy out of the model: its lane ends at raw.
      if (p === energy && (method === "residual" || method === "density")) continue;
      if (p === energy && method === "partition") {
        node(`adj:${p}`, "kcal_other", "adjusted", role, "kcal_other", `${p} − Σ kcal_from`);
        links.push({
          source: from,
          target: `adj:${p}`,
          operation: "kcal from everything else (partition)",
        });
        from = `adj:${p}`;
      }
      const label = nodes.find((x) => x.id === from)!.label;
      const out = encoded(label);
      node(`mx:${p}`, out, "matrix", role, out);
      links.push({
        source: from,
        target: `mx:${p}`,
        operation: out !== label ? "one-hot" : "kept",
      });
    }
  }
  const matrixCols = nodes.filter((x) => x.lane === "matrix").reduce((s, x) => s + x.count, 0);
  const families = state.models ?? [];
  const steps = (key: string) => {
    const out: { key: string; label: string; detail: string }[] = [];
    if (state.missing?.strategy === "impute")
      out.push({
        key: "impute",
        label: "Impute",
        detail: "Median of the training rows, learned inside each fold.",
      });
    if (method !== "none")
      out.push({ key: "energy", label: "Energy adjustment", detail: OPERATION[method] });
    out.push({
      key: "onehot",
      label: "One-hot",
      detail: "Text columns become indicators; the first level is the reference.",
    });
    if (key !== "boosted_trees")
      out.push({
        key: "scale",
        label: "Scale",
        detail: "Centered and scaled on the training rows.",
      });
    out.push({
      key: "model",
      label: familyLabel(key, state.task),
      detail: FAMILY[key as FamilyKey]?.bias ?? "",
    });
    return out;
  };
  const bearing = c.predictors.filter((p) => roles[p] === "exposure" && energyBearing(p));
  const pairs: DesignArtifact["substitution_pairs"] = [];
  for (const d of bearing) {
    for (const r of bearing) {
      if (d === r) continue;
      if (nestedIn(ds, d) === r || nestedIn(ds, r) === d) continue;
      pairs.push({ donor: d, recipient: r });
    }
  }
  return {
    lineage: { nodes, links, collapsed: collapse },
    matrix: { n_rows: nTrain, n_cols: matrixCols },
    models: families.map((key) => ({
      family: key,
      label: familyLabel(key, state.task),
      steps: steps(key),
    })),
    estimand: ESTIMAND[method],
    substitution_pairs: pairs,
    warnings: [],
    nested: [],
    left_out: state.missing?.drop_columns ?? [],
  };
}

/** A stable pseudo-random number in [-1, 1) for a label. */
function jitter(label: string): number {
  return unit(7, label) * 2 - 1;
}

const METHOD_SHIFT: Record<EnergyMethod, number> = {
  none: 0.002,
  standard: 0.001,
  residual: 0,
  density_multivariate: -0.001,
  density: -0.002,
  partition: -0.003,
};

export function fit(
  state: ProjectState,
  d: DesignArtifact,
  s: SplitArtifact,
  task: Task | null,
): FitArtifact {
  const families = state.models ?? [];
  const method = state.energy_adjustment?.method ?? "none";
  const imputed = state.missing?.strategy === "impute";
  const classification = task === "binary" || task === "multiclass";
  const primary = classification ? (task === "binary" ? "auc" : "accuracy") : "r2";
  const labels: Record<string, string> = classification
    ? task === "binary"
      ? { auc: "AUC", brier: "Brier score", log_loss: "Log loss" }
      : { accuracy: "Accuracy", macro_f1: "Macro-F1", log_loss: "Log loss" }
    : { r2: "R²", rmse: "RMSE", mae: "MAE" };
  const BASE: Record<string, number> = imputed
    ? { linear: 0.058, elastic_net: 0.06, boosted_trees: 0.041 }
    : { linear: 0.076, elastic_net: 0.077, boosted_trees: -0.039 };
  const models = families.map((family) => {
    const tag = `${family}:${method}:${state.missing?.strategy}:${s.n_train}`;
    const mean = (BASE[family] ?? 0.05) + METHOD_SHIFT[method] + 0.002 * jitter(tag);
    const sd = family === "boosted_trees" ? 0.05 : 0.035;
    const folds = Array.from(
      { length: s.folds },
      (_, k) => mean + sd * jitter(`${tag}:${k}`) * 1.2,
    );
    const cv: FitArtifact["models"][number]["cv"] = classification
      ? {
          [primary]: { mean: 0.68 + mean, sd: 0.02, folds: folds.map((f) => 0.68 + f) },
          log_loss: { mean: 0.61 - mean, sd: 0.01, folds: folds.map((f) => 0.61 - f) },
        }
      : {
          r2: { mean, sd, folds },
          rmse: {
            mean: 46.4 * Math.sqrt(1 - mean),
            sd: 1.6,
            folds: folds.map((f) => 46.4 * Math.sqrt(1 - f)),
          },
          mae: {
            mean: 30.1 * Math.sqrt(1 - mean),
            sd: 1.1,
            folds: folds.map((f) => 30.1 * Math.sqrt(1 - f)),
          },
        };
    const holdout: Record<string, number | null> | null = s.n_holdout
      ? classification
        ? { [primary]: 0.66 + mean, log_loss: 0.62 - mean }
        : { r2: mean - 0.06, rmse: 52.9 * Math.sqrt(1 - mean + 0.06), mae: 33.4 }
      : null;
    const below = !classification && mean < 0;
    const concerns = below
      ? [`Predicts worse than the outcome's average: CV R² −${Math.abs(mean).toFixed(2)}.`]
      : [];
    const coefficients =
      family === "boosted_trees"
        ? null
        : d.lineage.nodes
            .filter((x) => x.lane === "matrix" && x.column)
            .map((x) => {
              const estimate = 0.4 * jitter(`${family}:${x.column}:${method}`);
              const half =
                state.purpose === "inference"
                  ? 0.15 + 0.1 * Math.abs(jitter(`ci:${x.column}`))
                  : null;
              return {
                feature: x.column!,
                estimate,
                ci_low: half === null ? null : estimate - half,
                ci_high: half === null ? null : estimate + half,
                p:
                  state.purpose === "inference"
                    ? Math.min(1, Math.abs(jitter(`p:${x.column}`)))
                    : null,
                se: null,
                df: null,
              };
            });
    return {
      family,
      label: familyLabel(family, task),
      cv,
      holdout,
      coefficients,
      fit_seconds: family === "boosted_trees" ? 1.73 : family === "elastic_net" ? 0.77 : 0.03,
      concerns,
      baseline: {
        metric: primary,
        value: classification ? 0.5 : -0.002,
        label: classification ? "the class prior" : "the outcome's average",
      },
      versus_baseline: null,
      inference: null,
    };
  });
  // The fit computes the held-out scores; as on the server (turbotab/core/seal.py), the route
  // withholds them until the seal is opened (the stage handler in m1-record.ts). Opening changes
  // no stage's key, so a fit computed while sealed must still carry them.
  return {
    task: task ?? "regression",
    primary_metric: primary,
    metric_labels: labels,
    n_train: s.n_train,
    n_holdout: s.n_holdout,
    models,
    holdout_sealed: false,
    changed_after_seal: false,
    post_seal_decisions: [],
  };
}

export function substitution(
  state: ProjectState,
  f: FitArtifact,
  d: DesignArtifact,
): SubstitutionArtifact {
  const spec = state.substitution!;
  const ks = [0, 100, 200, 300, 400, 500, 600].map((k) => (k * spec.step_kcal) / 100);
  const slope: Record<string, number> = { linear: -1.46, elastic_net: -0.63, boosted_trees: 0.85 };
  const banded = spec.n_boot > 0;
  return {
    carried: [],
    band: banded
      ? {
          n_boot: spec.n_boot,
          n_rows: Math.min(2000, f.n_train),
          grouped_by: null,
          seconds: 39.5,
          failed: 0,
        }
      : null,
    band_estimate: banded ? null : { n_boot: 50, seconds: 40 },
    donor: spec.donor,
    recipient: spec.recipient,
    step_kcal: spec.step_kcal,
    ks,
    total_kind: "fixed",
    estimand: d.estimand,
    note: `Calories move from \`${spec.donor}\` to \`${spec.recipient}\` with total energy held fixed.`,
    basis: `Averaged over ${fmt(f.n_train)} training rows.`,
    models: f.models.map((m) => {
      const b =
        (slope[m.family] ?? 0.2) + 0.1 * jitter(`${m.family}:${spec.donor}:${spec.recipient}`);
      const delta = ks.map((k) => (b * k) / 100);
      const half = (k: number) => (0.35 + 0.004 * k) * (m.family === "boosted_trees" ? 2 : 1);
      return {
        family: m.family,
        label: m.label,
        delta,
        ci_low: banded ? delta.map((v, i) => v - half(ks[i]!)) : null,
        ci_high: banded ? delta.map((v, i) => v + half(ks[i]!)) : null,
        on_support_fraction: ks.map((k) => Math.max(0.44, 1 - k / 1070)),
        stopped_at: ks[ks.length - 1]!,
        effect_label: `${b >= 0 ? "+" : "−"}${Math.abs(b).toFixed(2)} per ${fmt(spec.step_kcal)} kcal at k = ${fmt(spec.step_kcal)}`,
      };
    }),
  };
}
