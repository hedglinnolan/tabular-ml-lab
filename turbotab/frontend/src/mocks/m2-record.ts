/**
 * The M2 mock for the Record (M2_CONTRACT §10, dev:mock only), layered on the M1 mock
 * (m1-record.ts): the stages the opening sequence reads (`oriented`, `structure`, `seal_plan`),
 * mirrored from turbotab/core/sequence.py, structure_previews.py and seal.py closely enough that
 * the Record meets the same shapes it meets on the real server; the Router's M2 rules (the grain
 * stated when an identifier is unique, the seal opened as the last step, findings held for a
 * question); the M2 sentences and refusals; findings with their repairs, served with what
 * answered them; and the held-out scores withheld until the seal is opened.
 *
 * Its numbers are the mock tables' own. Every M2 stage here is computed when read, so it is
 * always fresh; a changed answer that changes what one reads emits that stage's new key.
 */
import type { PreviewResult } from "../api/m1-stage-types";
import type { InterviewStep, QuestionKey } from "../api/m1-types";
import type { OrientedArtifact, RepairOption, SealPlan, StructureArtifact } from "../api/m2-types";
import type {
  Decision,
  DecisionRecord,
  Finding,
  FindingsArtifact,
  ProjectState,
  Refusal,
  Scalar,
  StageStatus,
} from "../api/schema";
import { rng, type MockColumn, type MockDataset } from "./datasets";
import type { MockProject } from "./db";
import { liveWriters } from "./m1-router";
import { columnInfo, findColumn, histogram, isNumericDtype, nMissing, nUnique } from "./stats";

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");
const tick = (s: string | number) => `\`${s}\``;
const IDENT =
  /(^|_)(id|seqn)$|^seqn$|^(participant|subject|person|patient|sample|respondent)(_?id)?$/i;
const DAY = 86_400_000;

function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

function listing(items: string[], limit = 3): string {
  const ticked = items.slice(0, limit).map(tick);
  const more = items.length - ticked.length;
  if (more > 0) ticked.push(`${more} more`);
  if (ticked.length <= 1) return ticked.join("");
  return `${ticked.slice(0, -1).join(", ")} and ${ticked[ticked.length - 1]}`;
}

// ── a feature-major table, to exercise the orientation question ──────────────

export const FEATURE_MAJOR_NAME = "metabolomics_features_in_rows.csv";

/** 120 metabolite features × 24 samples, exported features-in-rows (OPENING_SEQUENCE §04). */
export function metabolomicsFeatureMajor(): MockDataset {
  const r = rng(1847);
  const nFeatures = 120;
  const samples = Array.from({ length: 24 }, (_, i) => `S${String(i + 1).padStart(2, "0")}`);
  const ids = Array.from({ length: nFeatures }, (_, i) => `mz_${String(i + 1).padStart(4, "0")}`);
  const columns: MockColumn[] = [
    { name: "feature_id", dtype: "text", physical_type: "VARCHAR", values: ids },
  ];
  const abundance = ids.map(() => Math.exp(4 + 3.2 * (r() * 2 - 1) * 1.5));
  for (const s of samples) {
    columns.push({
      name: s,
      dtype: "numeric",
      physical_type: "DOUBLE",
      values: abundance.map((a) =>
        r() < 0.04 ? null : Math.round(a * (0.85 + 0.3 * r()) * 100) / 100,
      ),
    });
  }
  return { name: FEATURE_MAJOR_NAME, columns, nRows: nFeatures, sourceBytes: 31_200 };
}

/** The same table turned around: one row per sample, a column per feature, plus a made-up group. */
function turned(ds: MockDataset): MockDataset {
  const label = ds.columns[0]!;
  const samples = ds.columns.slice(1);
  const columns: MockColumn[] = [
    {
      name: "sample_id",
      dtype: "text",
      physical_type: "VARCHAR",
      values: samples.map((c) => c.name),
    },
    {
      name: "responder",
      dtype: "categorical",
      physical_type: "VARCHAR",
      values: samples.map((_, i) => (i % 3 === 0 ? "yes" : "no")),
    },
  ];
  label.values.forEach((feature, f) => {
    columns.push({
      name: String(feature),
      dtype: "numeric",
      physical_type: "DOUBLE",
      values: samples.map((c) => c.values[f] ?? null),
    });
  });
  return { name: ds.name, columns, nRows: samples.length, sourceBytes: ds.sourceBytes };
}

const isFeatureMajor = (ds: MockDataset) =>
  ds.name === FEATURE_MAJOR_NAME && ds.columns[0]?.name === "feature_id";

// ── oriented: the shape reading (turbotab/orientation.py) ─────────────────────

function orientedArtifact(p: MockProject, state: ProjectState): OrientedArtifact {
  const ds = p.source;
  const featureMajor = isFeatureMajor(ds);
  const numeric = ds.columns.filter((c) => isNumericDtype(c.dtype));
  const reading: OrientedArtifact["reading"] = featureMajor
    ? {
        reading: "feature_major",
        ratio: 23.0,
        s_rows: 1.15,
        s_cols: 0.05,
        n_rows: ds.nRows,
        n_numeric: numeric.length,
        sentence: `Across ${fmt(ds.nRows)} rows and ${fmt(numeric.length)} numeric columns, the rows differ from each other by orders of magnitude and the columns barely differ at all. In an assay table that is what features in rows looks like.`,
        confidence: "medium",
      }
    : {
        reading: "sample_major",
        ratio: 0.4,
        s_rows: 0.2,
        s_cols: 0.5,
        n_rows: ds.nRows,
        n_numeric: numeric.length,
        sentence: "The table's shape reads as one row per sample.",
        confidence: "medium",
      };
  return {
    n_rows: ds.nRows,
    n_cols: ds.columns.length,
    columns: ds.columns.map(columnInfo),
    source_bytes: ds.sourceBytes,
    parquet_bytes: Math.round(ds.sourceBytes * 0.6),
    ingest_seconds: 0.05,
    fingerprint: p.fingerprint,
    warnings: [],
    transposed: state.orientation === "feature_major",
    reading,
    turn: featureMajor
      ? {
          label_column: "feature_id",
          n_features: ds.nRows,
          n_samples: ds.columns.length - 1,
          refusal: null,
          code: null,
        }
      : { label_column: null, n_features: 0, n_samples: ds.nRows, refusal: null, code: null },
  };
}

// ── structure: grain, units, repeats, aggregation (turbotab/grain.py, repeats.py) ──

interface Units {
  column: string;
  of: Map<string, number[]>;
}

function unitsOf(ds: MockDataset, column: string): Units | null {
  const col = findColumn(ds, column);
  if (!col) return null;
  const of = new Map<string, number[]>();
  col.values.forEach((v, i) => {
    if (v === null || v === "") return;
    const k = String(v);
    of.set(k, [...(of.get(k) ?? []), i]);
  });
  return { column, of };
}

function grainEvidence(ds: MockDataset, target: string | null) {
  const out: StructureArtifact["grain"]["evidence"] = [];
  for (const col of ds.columns) {
    if (col.name === target || ds.nRows > 30_000 || ds.columns.length > 200) continue;
    if (!["categorical", "text", "integer"].includes(col.dtype)) continue;
    const distinct = nUnique(col);
    if (distinct <= 1 || distinct >= ds.nRows) continue;
    const counts = new Map<string, number>();
    for (const v of col.values)
      if (v !== null) counts.set(String(v), (counts.get(String(v)) ?? 0) + 1);
    const sizes = [...counts.values()];
    const modal = [
      ...sizes.reduce((m, s) => m.set(s, (m.get(s) ?? 0) + 1), new Map<number, number>()),
    ].sort((a, b) => b[1] - a[1])[0]![0];
    out.push({
      column: col.name,
      n_distinct: distinct,
      n_rows: ds.nRows,
      rows_per: Math.round((ds.nRows / distinct) * 100) / 100,
      modal_rows_per: modal,
      regular_share:
        Math.round((sizes.filter((x) => x === modal).length / sizes.length) * 1000) / 1000,
    });
  }
  const score = (e: (typeof out)[number]) => (IDENT.test(e.column) ? 10 : 0) + e.regular_share;
  return out.sort((a, b) => score(b) - score(a)).slice(0, 3);
}

/** A recognized identifier unique on every row, if there is one (§10: the grain is then stated). */
export function uniqueIdentifier(ds: MockDataset, target: string | null): string | null {
  const col = ds.columns.find(
    (c) => c.name !== target && IDENT.test(c.name) && nMissing(c) === 0 && nUnique(c) === ds.nRows,
  );
  return col?.name ?? null;
}

function spacingOf(ds: MockDataset, units: Units) {
  const date = ds.columns.find((c) => c.dtype === "datetime");
  if (!date) return null;
  const gaps: number[] = [];
  for (const rows of units.of.values()) {
    const days = rows
      .map((i) => Date.parse(String(date.values[i])))
      .filter((t) => Number.isFinite(t))
      .sort((a, b) => a - b);
    for (let k = 1; k < days.length; k++) gaps.push(Math.round((days[k]! - days[k - 1]!) / DAY));
  }
  if (!gaps.length) return null;
  const sorted = [...gaps].sort((a, b) => a - b);
  const mean = gaps.reduce((s, g) => s + g, 0) / gaps.length;
  const sd = Math.sqrt(gaps.reduce((s, g) => s + (g - mean) ** 2, 0) / gaps.length);
  return {
    column: date.name,
    n_people: units.of.size,
    n_gaps: gaps.length,
    min_days: sorted[0]!,
    max_days: sorted[sorted.length - 1]!,
    median_days: sorted[Math.floor(sorted.length / 2)]!,
    cv: mean ? sd / mean : 0,
    all_identical: sorted[0] === sorted[sorted.length - 1] && sorted[0] === 0,
  };
}

function replicateIndex(ds: MockDataset, units: Units): string | null {
  const idx = ds.columns.find(
    (c) => c.dtype === "integer" && /number|visit|wave|index/i.test(c.name),
  );
  if (!idx) return null;
  for (const rows of units.of.values()) {
    const vals = rows.map((i) => idx.values[i]);
    if (!vals.every((v, k) => v === k + 1)) return null;
  }
  return idx.name;
}

const DIETARY_REASON =
  "A single 24-hour recall is a noisy estimate of usual intake, and that noise attenuates diet–outcome associations toward the null. Using their mean rather than a single day reduces the within-person measurement error.";

function structureArtifact(p: MockProject, state: ProjectState): StructureArtifact {
  const ds = p.source;
  const evidence = grainEvidence(ds, state.target);
  const suggested = evidence.map((e) => e.column);
  const top = evidence[0];
  const grain: StructureArtifact["grain"] = {
    suggested,
    evidence,
    if_one_row:
      top && IDENT.test(top.column) && top.rows_per >= 1.5 && top.regular_share >= 0.8
        ? {
            columns: [top.column],
            message: `\`${top.column}\` has ${fmt(top.n_distinct)} distinct values across ${fmt(top.n_rows)} rows, about ${Math.round(top.rows_per)} each. That is the shape of repeated measures, and you answered one row per person. One of those two readings is wrong, and which one changes how the held-out rows are chosen.`,
          }
        : null,
  };
  const id = state.grain?.grain === "repeated" ? state.grain.id_column : null;
  const units = id ? unitsOf(ds, id) : null;
  const out: StructureArtifact = {
    grain,
    units: null,
    repeats: null,
    outcome: null,
    aggregation: null,
    time_columns: ds.columns.filter((c) => c.dtype === "datetime").map((c) => c.name),
    time_column: ds.columns.find((c) => c.dtype === "datetime")?.name ?? null,
  };
  if (!units) return out;
  const sizes = [...units.of.values()].map((r) => r.length);
  out.units = {
    column: units.column,
    n_units: units.of.size,
    max_rows_per_unit: Math.max(...sizes),
    min_rows_per_unit: Math.min(...sizes),
    n_missing: nMissing(findColumn(ds, units.column)!),
  };
  const spacing = spacingOf(ds, units);
  const index = replicateIndex(ds, units);
  if (index) out.time_columns = [...out.time_columns, index];
  let reading: "repeats" | "time_points" | null = null;
  const evidenceText: string[] = [];
  if (spacing) {
    if (spacing.median_days >= 28 && spacing.cv < 0.2) {
      reading = "time_points";
      evidenceText.push(
        `a person's records in ${tick(spacing.column)} are ${spacing.median_days} days apart at the median, ranging ${spacing.min_days} to ${spacing.max_days} — regular enough to be a schedule`,
      );
    } else {
      reading = "repeats";
      evidenceText.push(
        `the gaps between one person's records in ${tick(spacing.column)} run ${spacing.min_days} to ${spacing.max_days} days (median ${spacing.median_days}), too close together and too uneven to be a visit schedule`,
      );
    }
  }
  if (index) evidenceText.push(`${tick(index)} numbers each person's records in order`);
  const kindNow = state.repeat_kind?.repeat_kind ?? reading;
  out.repeats = {
    reading,
    stated: reading !== null,
    confidence: reading === "time_points" ? "high" : reading ? "medium" : null,
    evidence: evidenceText,
    sentence: reading
      ? `Not asked: these look like ${reading === "repeats" ? "repeated measurements of the same quantity rather than different time points" : "different time points rather than repeated measurements of the same quantity"} — ${evidenceText.join("; ")}.`
      : "Whether these are repeats or time points cannot be read from the data.",
    spacing,
    replicate_index: index,
    n_units_read: units.of.size,
  };
  const t = state.target ? findColumn(ds, state.target) : undefined;
  if (t) {
    let varying = 0;
    for (const rows of units.of.values()) {
      const vals = new Set(rows.map((i) => String(t.values[i])));
      if (vals.size > 1) varying += 1;
    }
    out.outcome = {
      column: t.name,
      varies: varying > 0,
      n_units_varying: varying,
      numeric: isNumericDtype(t.dtype),
    };
  }
  out.aggregation =
    kindNow === "time_points"
      ? {
          kind: "time_points",
          recommended: null,
          reason:
            "No default. These are different time points, and averaging them destroys the signal — the change over time is the signal. Which summary replaces it comes from your research question, not from the data.",
          marker: "offered",
          from_pack: null,
          options: ["mean", "first", "last", "change"],
        }
      : {
          kind: "repeats",
          recommended: "mean",
          reason: (state.lens ?? []).includes("dietary")
            ? DIETARY_REASON
            : "Replicate measurements of one quantity: their mean reduces measurement error rather than losing information.",
          marker: "derived",
          from_pack: (state.lens ?? []).includes("dietary") ? "dietary" : null,
          options: ["mean", "first", "last", "change"],
        };
  return out;
}

// ── seal_plan: what a held-out set of each size can measure (turbotab/core/seal.py) ──

function sealPlan(
  p: MockProject,
  state: ProjectState,
  task: string | null,
  nAnalyzed: number,
): SealPlan {
  const grain = effectiveGrain(p, state);
  const units =
    grain?.grain === "repeated" && grain.id_column ? unitsOf(p.source, grain.id_column) : null;
  const combined = units && state.unit === "unit";
  const basis: SealPlan["basis"] = units
    ? {
        state: "grouped",
        column: units.column,
        label: `grouped by \`${units.column}\``,
        sentence: combined
          ? `Each row is one \`${units.column}\` after combining their rows, so no unit is on both sides of the seal.`
          : `Held out by \`${units.column}\`: each of the \`${fmt(units.of.size)}\` units sits wholly on one side, so no unit is both trained on and scored.`,
        exploratory: false,
        source: combined ? "aggregation" : "grain",
        n_units: units.of.size,
      }
    : grain?.grain === "one_row_per_unit"
      ? {
          state: "one_row_per_unit",
          column: grain.id_column,
          label: "one row per unit",
          sentence: "Held out by row: each row is a different unit, and no identifier repeats.",
          exploratory: false,
          source: "grain",
          n_units: null,
        }
      : {
          state: "undetermined",
          column: null,
          label: "undetermined",
          sentence:
            "Held out by row, because whether a unit can appear in more than one row is not known. This is not a verified clean split: treat held-out scores as exploratory.",
          exploratory: true,
          source: null,
          n_units: null,
        };
  const n = combined ? units!.of.size : nAnalyzed;
  const binary = task === "binary";
  const floor: SealPlan["floor"] = binary
    ? {
        unit: "events",
        n: 100,
        text: "A held-out score needs at least 100 events and 100 non-events to be estimated with useful precision.",
        source: "Collins, Ogundimu & Altman, Stat Med 2016;35:214–226",
        convention: false,
      }
    : {
        unit: "rows",
        n: 100,
        text: "A convention: at least 100 held-out rows, by analogy with the 100-event rule for binary outcomes.",
        source: null,
        convention: true,
      };
  const measure = (h: number) => {
    const held = Math.round(n * h);
    if (binary) {
      const rare = Math.round(held * 0.27);
      const w = 1.96 * Math.sqrt((0.75 * 0.25) / Math.max(1, rare)) * 0.75;
      return {
        held,
        text: `About \`${fmt(held)}\` held-out rows, \`${fmt(rare)}\` in the rarer class: AUC known to about ±${w.toFixed(2)}.`,
        below: rare < 100,
      };
    }
    if (held < 40)
      return {
        held,
        text: `About \`${fmt(held)}\` held-out rows: too few to measure an R².`,
        below: true,
      };
    return {
      held,
      text: `About \`${fmt(held)}\` held-out rows: R² known to about ±${(1.96 * Math.sqrt(2 / held)).toFixed(2)}.`,
      below: held < 100,
    };
  };
  const options: SealPlan["options"] = [
    {
      holdout: 0,
      label: "Cross-validation only",
      n_holdout: 0,
      measures: "Every row trains and is scored by cross-validation; no untouched final score.",
      below_floor: false,
    },
    ...[0.1, 0.2, 0.3].map((h) => {
      const m = measure(h);
      return {
        holdout: h,
        label: `Hold out ${Math.round(h * 100)}%`,
        n_holdout: m.held,
        measures: m.text,
        below_floor: m.below,
      };
    }),
  ];
  const usual = measure(0.2);
  const cvFirst = usual.below;
  return {
    task: task ?? "regression",
    n_measured: nAnalyzed,
    n_analyzed: n,
    basis,
    chronology: state.temporal?.temporal
      ? {
          drawn: true,
          time_column: state.temporal.time_column,
          boundary: null,
          n_units: units ? Math.round(units.of.size * 0.2) : null,
          n_undated: 0,
          sentence: `The latest units by their last ${tick(state.temporal.time_column ?? "date")} are held out; every training unit's last visit comes earlier.`,
        }
      : null,
    exploratory: basis.exploratory,
    floor,
    options,
    cv_first: cvFirst,
    reason: cvFirst
      ? `Of \`${fmt(n)}\` rows analyzed, a 20% holdout leaves about \`${fmt(usual.held)}\` held-out rows, below the floor of ${floor.n} ${floor.unit}; cross-validation reuses every row, so it comes first.`
      : `Of \`${fmt(n)}\` rows analyzed, a 20% holdout leaves about \`${fmt(usual.held)}\` held-out rows, above the floor of ${floor.n} ${floor.unit}.`,
    precision_note:
      "Widths are approximate 95% intervals of a score on that many held-out rows: R² by (1 − R²)·√(2/n) at R² = 0, AUC by Hanley & McNeil (1982) at an AUC of 0.75.",
    refusal: null,
  };
}

// ── the grain stated when an identifier is unique on every row (§10, §12.5) ──

/** The grain in force: the recorded answer, else the stated reading of a unique identifier. */
export function effectiveGrain(p: MockProject, state: ProjectState): ProjectState["grain"] {
  if (state.grain) return state.grain;
  const id = uniqueIdentifier(p.source, state.target);
  return id ? { grain: "one_row_per_unit", id_column: id, acknowledged: false } : null;
}

// ── findings: repairs, dispositions, what answered them (turbotab/core/repairs.py) ──

const repair = (
  f: Finding,
  key: string,
  label: string,
  consequence: string,
  effect: RepairOption["effect"],
  sentence: string,
  params: Record<string, unknown>,
): RepairOption => ({
  key,
  label,
  consequence,
  row_local: true,
  effect,
  sentence,
  decision: { kind: "apply_repair", finding_id: f.id, option: key, params },
});

const BANDS: Record<string, [number, number, string]> = {
  kcal: [100, 30000, "kcal"],
  energy_kcal: [100, 30000, "kcal"],
  glucose: [10, 2000, "mg/dL"],
  bp_sys: [40, 300, "mmHg"],
  bp_di: [15, 220, "mmHg"],
};

function impossibleFinding(ds: MockDataset): Finding | null {
  const hits: { column: string; n: number; band: [number, number, string] }[] = [];
  for (const [name, band] of Object.entries(BANDS)) {
    const col = findColumn(ds, name);
    if (!col) continue;
    const n = col.values.filter(
      (v) => typeof v === "number" && (v < band[0] || v > band[1]),
    ).length;
    if (n) hits.push({ column: name, n, band });
  }
  if (!hits.length) return null;
  const first = hits[0]!;
  const total = hits.reduce((s, h) => s + h.n, 0);
  const cols = hits.map((h) => h.column);
  const f: Finding = {
    id: "pack::clinical::impossible_vs_extreme",
    severity: "warning",
    title: "Physiologically impossible values",
    detail: "Values no living person could have: entry errors, not extremes.",
    why_it_matters: "A split drawn over corrupted values is a worse split.",
    affected_columns: cols,
    source: "pack",
    lens: "clinical",
    evidence: { status: "CONVENTION", source: "research/CLINICAL_SURVEY_PACK.md §A1.2" },
    summary: `${tick(first.column)} has ${tick(fmt(first.n))} impossible values outside ${tick(fmt(first.band[0]))}–${tick(fmt(first.band[1]))}${hits.length > 1 ? ` (${hits.length - 1} more columns too)` : ""}.`,
    routes_to: "exclusions",
    lever_label: "Exclude rows by range",
    group: null,
    repairs: [],
    disposition: null,
    answered_by: null,
  };
  const params = {
    bands: Object.fromEntries(hits.map((h) => [h.column, [h.band[0], h.band[1]]])),
    units: Object.fromEntries(hits.map((h) => [h.column, h.band[2]])),
  };
  const named = listing(cols, 2);
  f.repairs = [
    repair(
      f,
      "set_missing",
      "Set to missing",
      `${tick(fmt(total))} impossible values in ${named} become blank; abnormal but real ones stay.`,
      "values",
      `${tick(fmt(total))} physiologically impossible values in ${named} were set to missing.`,
      params,
    ),
    repair(
      f,
      "exclude_rows",
      "Exclude those rows",
      `${tick(fmt(total))} rows holding an impossible value leave the analysis before the seal.`,
      "rows",
      `${tick(fmt(total))} rows with a physiologically impossible value in ${named} were excluded.`,
      params,
    ),
    repair(
      f,
      "unusable",
      "Mark columns unusable",
      `${named} leave the predictors; their values stay in the table, unused.`,
      "columns",
      `${named} were set aside as unusable for holding physiologically impossible values.`,
      params,
    ),
  ];
  return f;
}

function withRepairs(ds: MockDataset, f: Finding): Finding {
  if (!f.id.startsWith("binary_text__")) return f;
  const column = f.affected_columns[0]!;
  const col = findColumn(ds, column);
  if (!col) return f;
  const counts = new Map<string, number>();
  for (const v of col.values)
    if (v !== null && v !== "") counts.set(String(v), (counts.get(String(v)) ?? 0) + 1);
  const [a, b] = [...counts.keys()].sort();
  if (a === undefined || b === undefined) return f;
  const level = (one: string, zero: string) =>
    repair(
      f,
      "level",
      `${tick(one)} counts as 1`,
      `${tick(column)} becomes ${tick(1)} for ${tick(one)} (${tick(fmt(counts.get(one)!))} rows) and ${tick(0)} for ${tick(zero)} (${tick(fmt(counts.get(zero)!))}).`,
      "values",
      `${tick(column)} was recoded with ${tick(one)} as ${tick(1)} and ${tick(zero)} as ${tick(0)}.`,
      { column, one, zero },
    );
  return {
    ...f,
    summary: `${tick(column)} is two-level text (${tick(a)}, ${tick(b)}); which level counts as 1 is not asked yet.`,
    group: null,
    repairs: [level(a, b), level(b, a)],
  };
}

/** Finding id -> its live disposition record (later writes win; reverted records do not count). */
function dispositionRecords(records: DecisionRecord[]): Map<string, DecisionRecord> {
  const cancelled = new Set<string>();
  for (const r of [...records].sort((x, y) => y.seq - x.seq)) {
    if (cancelled.has(r.id)) continue;
    if (r.decision.kind === "revert") cancelled.add(r.decision.decision_id);
  }
  const out = new Map<string, DecisionRecord>();
  for (const r of [...records].sort((x, y) => x.seq - y.seq)) {
    const d = r.decision;
    if (cancelled.has(r.id)) continue;
    if (d.kind === "apply_repair" || d.kind === "defer_finding" || d.kind === "dismiss_finding")
      out.set(d.finding_id, r);
  }
  return out;
}

const PREDICTOR_ROLES = new Set(["exposure", "covariate", "energy"]);

/** Whether the answer to the question a finding routes to settles it (repairs.py `matching_answer`):
 *  roles give its column the role it asks for; an exclusion rule names one of its columns; any
 *  other question is settled by being answered. */
function matchingAnswer(f: Finding, state: ProjectState): boolean {
  const route = f.routes_to;
  if (!route || (state as unknown as Record<string, unknown>)[route] == null) return false;
  const cols = f.affected_columns.filter((c) => c !== state.target);
  const family = f.id.split("__")[0]!;
  if (route === "roles") {
    const roles = (state.roles ?? {}) as Record<string, string>;
    if (/identifier|repeats/.test(family)) return !!cols[0] && roles[cols[0]] === "identifier";
    if (family === "voice::flag") return !!cols[0] && roles[cols[0]] === "flag";
    if (/unnamed|constant/.test(family))
      return cols.every((c) => c in roles && !PREDICTOR_ROLES.has(roles[c]!));
    return cols.every((c) => c in roles);
  }
  if (route === "exclusions") {
    const named = new Set((state.exclusions ?? []).map((r) => r.column));
    return !cols.length || cols.some((c) => named.has(c));
  }
  return true;
}

/** The disposition a finding decision writes into the `findings` slot. */
export function dispositionOf(d: Decision): NonNullable<ProjectState["findings"]>[string] | null {
  switch (d.kind) {
    case "apply_repair":
      return {
        action: "applied",
        option: d.option,
        params: d.params ?? {},
        to: null,
        reason: null,
      };
    case "defer_finding":
      return { action: "deferred", option: null, params: {}, to: d.to, reason: null };
    case "dismiss_finding":
      return { action: "dismissed", option: null, params: {}, to: null, reason: d.reason ?? null };
    default:
      return null;
  }
}

// ── sentences, refusals ──────────────────────────────────────────────────────

const ATTEST = "My answer is right; the data is like this";

export interface M2Mock {
  /** Stage statuses for the M2 stages (always fresh), merged into the view. */
  statuses(p: MockProject): Record<string, StageStatus>;
  /** Emit a stage event for each M2 stage whose key changed. */
  reconcile(pid: string, p: MockProject, emit: (type: "stage", data: StageStatus) => void): void;
  artifact(
    p: MockProject,
    stage: string,
    extra: { task: string | null; nAnalyzed: number },
  ): unknown;
  route(
    p: MockProject,
    steps: InterviewStep[],
    stages: Record<string, StageStatus>,
  ): InterviewStep[];
  routingState(p: MockProject, state: ProjectState): ProjectState;
  validate(p: MockProject, d: Decision, stages: Record<string, StageStatus>): Refusal | null;
  /** `heldOut`: the drawn split's held-out rows, the count the seal's opening states. */
  sentence(
    p: MockProject,
    d: Decision,
    state: ProjectState,
    heldOut?: number | null,
  ): string | null;
  findings(p: MockProject, artifact: FindingsArtifact): FindingsArtifact;
  preview(p: MockProject, d: Decision): PreviewResult | null;
  /** Before a decision: a turned table replaces the one the mock reads. */
  beforeDecision(p: MockProject, d: Decision): void;
}

const M2_STAGES = ["oriented", "structure", "seal_plan"] as const;
type M2Stage = (typeof M2_STAGES)[number];

export function m2Mock(foldOf: (records: DecisionRecord[]) => ProjectState): M2Mock {
  const lastKey = new Map<string, string>();

  const keyOf = (p: MockProject, stage: M2Stage): string => {
    const s = foldOf(p.records);
    const reads: Record<M2Stage, unknown[]> = {
      oriented: [s.orientation, s.lens],
      structure: [s.grain, s.target, s.lens, s.repeat_kind, s.unit],
      seal_plan: [s.grain, s.unit, s.temporal, s.roles, s.target, s.task, s.exclusions, s.missing],
    };
    return hash(JSON.stringify([stage, p.fingerprint, reads[stage]]));
  };
  const status = (p: MockProject, stage: M2Stage): StageStatus => ({
    stage,
    status: "fresh",
    key: keyOf(p, stage),
    fresh: true,
    missing: [],
    error: null,
    job_id: null,
    progress: null,
    updated_at: new Date().toISOString(),
    cancelled: false,
  });

  return {
    statuses(p) {
      const s = foldOf(p.records);
      const out: Record<string, StageStatus> = {};
      for (const st of M2_STAGES) {
        if (st === "seal_plan" && !s.target) continue;
        out[st] = status(p, st);
      }
      return out;
    },

    reconcile(pid, p, emit) {
      for (const [name, st] of Object.entries(this.statuses(p))) {
        const k = `${pid}:${name}`;
        if (lastKey.get(k) === st.key) continue;
        lastKey.set(k, st.key!);
        emit("stage", st);
      }
    },

    artifact(p, stage, extra) {
      const state = foldOf(p.records);
      switch (stage) {
        case "oriented":
          return orientedArtifact(p, state);
        case "structure":
          return structureArtifact(p, state);
        case "seal_plan":
          return sealPlan(p, state, extra.task, extra.nAnalyzed);
        default:
          return undefined;
      }
    },

    routingState(p, state) {
      // What is stated, not asked, counts as answered for what follows it: the grain when an
      // identifier is unique on every row, the repeats when the structure reading states them.
      // The grain is read once the outcome is chosen (an identifier is never the outcome), and
      // so once the table is the right way round: never stated ahead of the questions before it.
      const grain = state.grain ?? (state.target ? effectiveGrain(p, state) : null);
      let repeatKind = state.repeat_kind;
      if (!repeatKind && grain?.grain === "repeated") {
        const reading = structureArtifact(p, { ...state, grain }).repeats;
        if (reading?.stated && reading.reading)
          repeatKind = {
            repeat_kind: reading.reading,
            time_column:
              reading.reading === "time_points" ? (reading.spacing?.column ?? null) : null,
          };
      }
      return { ...state, grain, repeat_kind: repeatKind };
    },

    route(p, steps, stages) {
      const state = foldOf(p.records);
      const stated = !state.grain ? uniqueIdentifier(p.source, state.target) : null;
      const repeats =
        !state.repeat_kind && state.grain?.grain === "repeated"
          ? structureArtifact(p, state).repeats
          : null;
      const held = new Map<string, string[]>();
      for (const [fid, d] of Object.entries(state.findings ?? {})) {
        if (d.action === "deferred" && d.to) held.set(d.to, [...(held.get(d.to) ?? []), fid]);
      }
      const out = steps.map((st) => {
        let next = st;
        const unanswered = st.status === "answered" && st.decision_id === null;
        if (st.key === "grain" && stated && unanswered) {
          next = {
            ...st,
            status: "skipped",
            reason: `Not asked: every ${tick(stated)} appears once, so each person is one row.`,
          };
        }
        if (st.key === "repeat_kind" && repeats?.stated && unanswered) {
          next = { ...st, status: "skipped", reason: repeats.sentence };
        }
        return { ...next, deferred_findings: held.get(st.key) ?? [] };
      });
      // §12.1: the Router's last step is opening the seal, once a fit is fresh.
      const opened = p.records.find((r) => r.decision.kind === "open_seal");
      const firstUnanswered = out.find((st) => st.status === "open" || st.status === "waiting");
      const seal: InterviewStep = (() => {
        const key = "open_seal" as QuestionKey;
        if (opened)
          return {
            key,
            status: "answered",
            decision_id: opened.id,
            reason: null,
            waiting_on: [],
            deferred_findings: [],
          };
        if (state.split && state.split.holdout === 0)
          return {
            key,
            status: "not_applicable",
            decision_id: null,
            reason: "No rows were sealed: every score comes from cross-validation.",
            waiting_on: [],
            deferred_findings: [],
          };
        if (firstUnanswered)
          return {
            key,
            status: "waiting",
            decision_id: null,
            reason: null,
            waiting_on: [firstUnanswered.key, "fit"],
            deferred_findings: [],
          };
        if (stages.fit?.status !== "fresh")
          return {
            key,
            status: "waiting",
            decision_id: null,
            reason: null,
            waiting_on: ["fit"],
            deferred_findings: [],
          };
        return {
          key,
          status: "open",
          decision_id: null,
          reason: null,
          waiting_on: [],
          deferred_findings: [],
        };
      })();
      return [...out, seal];
    },

    validate(p, d, stages) {
      const state = foldOf(p.records);
      const refuse = (
        code: string,
        message: string,
        exits: Refusal["error"]["exits"] = [],
      ): Refusal => ({
        error: { code, message, exits },
      });
      switch (d.kind) {
        case "set_grain": {
          if (d.acknowledged) return null;
          if (d.grain === "repeated" && !d.id_column)
            return refuse(
              "no_id_column",
              "Name the column that says which unit a row belongs to; the held-out rows keep each unit's rows together by it.",
            );
          if (d.grain === "one_row_per_unit") {
            const found = structureArtifact(p, state).grain.if_one_row;
            if (found)
              return refuse("data_repeats", found.message, [
                {
                  label: `Rows repeat per \`${found.columns[0]}\``,
                  decision: {
                    kind: "set_grain",
                    grain: "repeated",
                    id_column: found.columns[0]!,
                    acknowledged: false,
                  },
                },
                {
                  label: ATTEST,
                  decision: {
                    kind: "set_grain",
                    grain: "one_row_per_unit",
                    id_column: null,
                    acknowledged: true,
                  },
                },
              ]);
          }
          return null;
        }
        case "set_aggregation": {
          const outcome = structureArtifact(p, state).outcome;
          if (outcome?.varies && !d.outcome)
            return refuse(
              "which_outcome",
              `\`${outcome.column}\` changes within ${fmt(outcome.n_units_varying)} units, so combining their rows needs to know which outcome to keep.`,
              (["first", "last"] as const).map((o) => ({
                label: o === "first" ? "Keep the first outcome" : "Keep the last outcome",
                decision: { kind: "set_aggregation" as const, method: d.method, outcome: o },
              })),
            );
          return null;
        }
        case "open_seal":
          if (state.seal_opened)
            return refuse(
              "seal_already_open",
              "The held-out rows were opened once already; their scores stand in the record, and a seal opens only once.",
            );
          if (!state.split || state.split.holdout === 0)
            return refuse(
              "nothing_sealed",
              "Every row trains under cross-validation alone, so there are no held-out rows to open.",
            );
          if (stages.fit?.status !== "fresh")
            return refuse(
              "fit_not_fresh",
              "The models are not fitted for the current answers yet; the held-out rows open only on a fresh fit.",
            );
          return null;
        case "set_target":
          if (!findColumn(p.source, d.column))
            return refuse("unknown_column", `No column named '${d.column}' in this table.`);
          return null;
        case "defer_finding": {
          const keys = [
            "lens",
            "orientation",
            "target",
            "event",
            "task",
            "purpose",
            "grain",
            "repeat_kind",
            "unit",
            "aggregation",
            "temporal",
            "roles",
            "exclusions",
            "missing",
            "split",
            "energy_adjustment",
            "models",
            "substitution",
          ];
          if (!keys.includes(d.to))
            return refuse("unknown_question", `There is no question '${d.to}' to hold it for.`, [
              {
                label: "Dismiss it",
                decision: { kind: "dismiss_finding", finding_id: d.finding_id, reason: null },
              },
            ]);
          return null;
        }
        default:
          return null;
      }
    },

    sentence(p, d, state, heldOut = null) {
      const ds = p.source;
      switch (d.kind) {
        case "set_orientation":
          return d.orientation === "sample_major"
            ? "The table was confirmed as one row per sample, as supplied, and was not transposed."
            : `The table was supplied with features in rows and samples in columns, and was transposed to one row per sample before any diagnosis was run; its ${fmt(
                // Already turned by the time the sentence is written (beforeDecision).
                isFeatureMajor(ds) ? ds.nRows : ds.columns.length - 2,
              )} rows became measurement columns.`;
        case "set_event": {
          const col = findColumn(ds, d.column);
          const others = col
            ? [...new Set(col.values.filter((v) => v !== null).map(String))].filter(
                (v) => v !== d.level,
              )
            : [];
          return `${tick(d.level)} of ${tick(d.column)} was taken as the event and coded 1${others.length && others.length <= 3 ? `; ${others.map(tick).join(" and ")} ${others.length === 1 ? "was" : "were"} coded 0` : ""}.`;
        }
        case "set_grain": {
          const g = d.grain as string;
          if (g === "unknown")
            return "Whether one participant appears in more than one row was answered as not known; the held-out rows are drawn by row and their scores are labeled exploratory.";
          if (g === "one_row_per_unit")
            return "Each row was declared a different participant: no one appears in more than one row.";
          const units = d.id_column ? unitsOf(ds, d.id_column) : null;
          const most = units ? Math.max(...[...units.of.values()].map((r) => r.length)) : 0;
          return `Participants were declared to appear in more than one row, identified by ${tick(d.id_column ?? "?")}${units ? `: ${fmt(ds.nRows)} rows from ${fmt(units.of.size)} of them, at most ${fmt(most)} each` : ""}.`;
        }
        case "set_repeat_kind":
          return d.repeat_kind === "repeats"
            ? "Each participant's rows were taken as repeated measurements of the same quantity, not different time points."
            : `Each participant's rows were taken as different time points${d.time_column ? `, ordered by ${tick(d.time_column)}` : ""}.`;
        case "set_unit": {
          const who = state.grain?.id_column ? tick(state.grain.id_column) : "participant";
          return d.unit === "unit"
            ? `The analysis was set at one row per ${who}: each one's rows are combined into one.`
            : "The analysis was kept at one row per record; each participant's rows stay together on one side of the split.";
        }
        case "set_aggregation": {
          const how = {
            mean: "by their mean",
            first: "by keeping the first row",
            last: "by keeping the last row",
            change: "as the change from the first to the last",
          }[d.method];
          const id = state.grain?.id_column;
          const units = id ? unitsOf(ds, id) : null;
          const n = units ? `: ${fmt(ds.nRows)} rows became ${fmt(units.of.size)}` : "";
          const outcome =
            d.outcome && state.target
              ? `; the outcome ${tick(state.target)} was taken as their ${d.outcome === "mean" ? "mean" : `${d.outcome} value`}`
              : "";
          return `Each participant's rows were combined into one ${how}${n}${outcome}.`;
        }
        case "set_temporal":
          return d.temporal
            ? `The model was declared to predict a later outcome from earlier measurements: the held-out rows are the latest${d.time_column ? ` by ${tick(d.time_column)}` : ""}.`
            : "The model was declared not to predict forward in time: rows are held out at random.";
        case "open_seal": {
          const n = heldOut ?? (state.split ? Math.round(ds.nRows * state.split.holdout) : null);
          return `The ${n ? `${fmt(n)} ` : ""}held-out rows were opened once and scored; those scores are fixed in the record, and any later change is marked as made after the seal was opened.`;
        }
        case "apply_repair": {
          const f = currentFindings.get(p.summary.id)?.find((x) => x.id === d.finding_id);
          const option = f?.repairs.find(
            (o) =>
              o.key === d.option &&
              JSON.stringify(o.decision.params) === JSON.stringify(d.params ?? {}),
          );
          return option?.sentence ?? `The repair “${d.option}” was applied.`;
        }
        case "defer_finding": {
          const f = currentFindings.get(p.summary.id)?.find((x) => x.id === d.finding_id);
          const name = f?.affected_columns.length
            ? `The finding on ${listing(f.affected_columns, 3)}`
            : "A finding";
          const where: Partial<Record<string, string>> = {
            exclusions: "the eligibility question",
            missing: "the missing-values question",
            roles: "the column roles",
            energy_adjustment: "the energy-adjustment question",
          };
          return `${name} was set aside for ${where[d.to] ?? `the ${d.to.replace(/_/g, " ")} question`}, where it will be raised again.`;
        }
        case "dismiss_finding": {
          const f = currentFindings.get(p.summary.id)?.find((x) => x.id === d.finding_id);
          const name = f?.affected_columns.length
            ? `The finding on ${listing(f.affected_columns, 3)}`
            : "A finding";
          return d.reason
            ? `${name} was dismissed: ${d.reason.replace(/\.$/, "")}.`
            : `${name} was dismissed, with no reason given; it stays in the record.`;
        }
        default:
          return null;
      }
    },

    findings(p, artifact) {
      const state = foldOf(p.records);
      const ds = p.source;
      let list = artifact.findings.map((f) => withRepairs(ds, f));
      if ((state.lens ?? []).includes("clinical")) {
        const imp = impossibleFinding(ds);
        if (imp && !list.some((f) => f.id === imp.id)) list = [imp, ...list];
      }
      currentFindings.set(p.summary.id, list);
      const writers = dispositionRecords(p.records);
      const slotWriters = liveWriters(p.records, state);
      const annotated = list.map((f) => {
        const rec = writers.get(f.id);
        const disposition = rec ? dispositionOf(rec.decision) : null;
        // Its own disposition first; else the answer to its routed question, when that answer
        // settles it (turbotab/core/repairs.py `annotate`, `matching_answer`).
        const routed =
          !rec && f.routes_to && matchingAnswer(f, state)
            ? (slotWriters.get(f.routes_to) ?? null)
            : null;
        return { ...f, disposition, answered_by: rec?.id ?? routed };
      });
      return { ...artifact, findings: annotated };
    },

    preview(p, d) {
      if (d.kind !== "apply_repair") return null;
      const f = currentFindings.get(p.summary.id)?.find((x) => x.id === d.finding_id);
      const option = f?.repairs.find(
        (o) =>
          o.key === d.option &&
          JSON.stringify(o.decision.params) === JSON.stringify(d.params ?? {}),
      );
      if (!f || !option) return null;
      return repairPreview(p.source, f, option);
    },

    beforeDecision(p, d) {
      if (
        d.kind === "set_orientation" &&
        d.orientation === "feature_major" &&
        isFeatureMajor(p.source)
      ) {
        p.source = turned(p.source);
        p.fingerprint = hash(
          `${p.source.name}:turned:${p.source.nRows}:${p.source.columns.length}`,
        );
      }
    },
  };
}

/** The findings each project was last served, so a repair's sentence is its offered one. */
const currentFindings = new Map<string, Finding[]>();

// ── a repair's preview: the changed cells and the column's distribution ─────

function repairPreview(ds: MockDataset, f: Finding, option: RepairOption): PreviewResult {
  const params = option.decision.params as Record<string, unknown>;
  const views: PreviewResult["views"] = [];
  if (f.id.startsWith("binary_text__")) {
    const column = String(params.column);
    const col = findColumn(ds, column)!;
    const rows = Array.from({ length: Math.min(8, ds.nRows) }, (_, i) => i);
    views.push({
      kind: "table_focus",
      title: `\`${column}\` as it would be coded`,
      caption: `\`${String(params.one)}\` becomes \`1\` and \`${String(params.zero)}\` becomes \`0\`.`,
      emphasis: [column],
      coach: [],
      columns_before: [column],
      columns_after: [column],
      rows: rows.map((i) => {
        const v = col.values[i] ?? null;
        return {
          row_id: i,
          before: { [column]: v },
          after: { [column]: v === null ? null : String(v) === String(params.one) ? 1 : 0 },
        };
      }),
      changed: rows.map((i) => [i, column] as [number, string]),
      n_affected_columns: 1,
      story: [],
    });
  } else {
    const bands = (params.bands ?? {}) as Record<string, [number, number]>;
    const [column, band] = Object.entries(bands)[0] ?? [];
    const col = column ? findColumn(ds, column) : undefined;
    if (column && col && band) {
      const bad: number[] = [];
      col.values.forEach((v, i) => {
        if (typeof v === "number" && (v < band[0] || v > band[1]) && bad.length < 8) bad.push(i);
      });
      const after = (v: Scalar) =>
        option.key === "set_missing" ? null : option.key === "exclude_rows" ? v : v;
      views.push({
        kind: "table_focus",
        title:
          option.key === "exclude_rows"
            ? `The rows that would leave`
            : option.key === "unusable"
              ? `\`${column}\`, set aside`
              : `The cells that would be blanked`,
        caption: option.consequence,
        emphasis: [column],
        coach: [],
        columns_before: [column],
        columns_after: option.key === "unusable" ? [] : [column],
        rows: bad.map((i) => ({
          row_id: i,
          before: { [column]: col.values[i] ?? null },
          after: option.key === "exclude_rows" ? {} : { [column]: after(col.values[i] ?? null) },
        })),
        changed: bad.map((i) => [i, column] as [number, string]),
        n_affected_columns: Object.keys(bands).length,
        story: [],
      });
      const h = histogram(col, 30);
      views.push({
        kind: "distribution",
        title: `\`${column}\` on every loaded row`,
        caption: `The impossibility band runs ${fmt(band[0])} to ${fmt(band[1])}.`,
        emphasis: [column],
        coach: [],
        column,
        before: h,
        after: h,
        before_label: "as loaded",
        after_label: "with this repair",
        marks: [{ value: band[0], label: `floor ${fmt(band[0])}`, group: null }],
        story: [],
      });
    }
  }
  return {
    kind: "apply_repair",
    views,
    basis: `Read across all ${fmt(ds.nRows)} rows, as the finding was.`,
    note: null,
    caution: null,
  };
}
