/**
 * Stand-ins for the Python packs: lens hints, task detection and findings.
 * They compute from the mock table so what the UI shows is at least true of it.
 */
import type {
  Confidence,
  Finding,
  Lens,
  LensHint,
  Scalar,
  TargetInfoArtifact,
  Task,
} from "../api/schema";
import type { MockDataset } from "./datasets";
import { findColumn, histogram, isNumericDtype, nMissing, nUnique, topValues } from "./stats";

const fmt = (n: number) => n.toLocaleString("en-US");

function numericCols(ds: MockDataset) {
  return ds.columns.filter((c) => isNumericDtype(c.dtype));
}

function isCountMatrix(ds: MockDataset): boolean {
  const nums = numericCols(ds);
  if (nums.length < 30) return false;
  return nums.every((c) =>
    c.values.every((v) => v === null || (Number.isInteger(v) && (v as number) >= 0)),
  );
}

export function lensHints(ds: MockDataset): LensHint[] {
  const hints: LensHint[] = [];
  const nums = numericCols(ds);
  if (isCountMatrix(ds)) {
    hints.push({
      lens: "genomics",
      because:
        "every one of these columns holds non-negative whole numbers, which is what a count matrix looks like",
    });
  } else if (nums.length >= 30) {
    hints.push({
      lens: "metabolomics",
      because: `there are ${fmt(nums.length)} measurement columns across ${fmt(ds.nRows)} rows`,
    });
  }
  if (ds.columns.some((c) => /kcal/i.test(c.name))) {
    hints.push({ lens: "dietary", because: "there is a total-energy column" });
  }
  const clinical = ds.columns.filter((c) =>
    /^(bmi|hba1c|ldl|hdl|sbp|dbp|glucose|crp)$/i.test(c.name),
  );
  if (clinical.length >= 2) {
    hints.push({
      lens: "clinical",
      because: `${clinical.map((c) => `\`${c.name}\``).join(" and ")} match clinical reference measurements`,
    });
  }
  return hints;
}

export function detectTask(
  ds: MockDataset,
  target: string,
): {
  task: Task;
  confidence: Confidence;
  reason: string;
} {
  const col = findColumn(ds, target);
  if (!col) return { task: "regression", confidence: "low", reason: "The column was not found." };
  const unique = nUnique(col);
  // The server's voice (turbotab/core/stages/target.py task_reason): one sentence, data first.
  if (unique === 2) {
    const [a, b] = topValues(col, 2).map((t) => `\`${String(t.value)}\``);
    return {
      task: "binary",
      confidence: "high",
      reason: `Two values, ${a} and ${b} — read as a binary outcome.`,
    };
  }
  if (isNumericDtype(col.dtype)) {
    if (col.dtype === "integer" && unique <= 10) {
      return {
        task: "multiclass",
        confidence: "medium",
        reason: `Whole numbers with ${fmt(unique)} distinct values; class codes, counts and ordinal scores all look like this — read as a multiclass outcome.`,
      };
    }
    return {
      task: "regression",
      confidence: "high",
      reason: `Continuous, with ${fmt(unique)} distinct values — read as a regression outcome.`,
    };
  }
  if (unique <= 20) {
    return {
      task: "multiclass",
      confidence: "high",
      reason: `Text with ${fmt(unique)} distinct values — read as a multiclass outcome.`,
    };
  }
  return {
    task: "multiclass",
    confidence: "low",
    reason: `Text with ${fmt(unique)} distinct values, which reads more like an identifier than an outcome.`,
  };
}

export function targetInfo(
  ds: MockDataset,
  target: string,
  override: Task | null,
): TargetInfoArtifact {
  const col = findColumn(ds, target)!;
  const detected = detectTask(ds, target);
  const numeric = isNumericDtype(col.dtype) && nUnique(col) > 2;
  return {
    column: target,
    task: override ?? detected.task,
    detected_task: detected.task,
    confidence: detected.confidence,
    reason: detected.reason,
    histogram: numeric ? histogram(col, 24) : null,
    classes: numeric ? null : topValues(col, 12),
    unit: null,
    unit_source: null,
  };
}

/** Which rows share a participant, when an id column repeats. */
function repeats(ds: MockDataset): { column: string; people: number; perPerson: number } | null {
  const id = ds.columns.find((c) => /(participant|subject|person|patient)_?id$/i.test(c.name));
  if (!id) return null;
  const people = nUnique(id);
  if (people === ds.nRows) return null;
  return { column: id.name, people, perPerson: Math.round(ds.nRows / people) };
}

function constantWithin(ds: MockDataset, idCol: string, target: string): boolean {
  const id = findColumn(ds, idCol);
  const t = findColumn(ds, target);
  if (!id || !t) return false;
  const seen = new Map<Scalar, Scalar>();
  for (let i = 0; i < ds.nRows; i++) {
    const k = id.values[i] ?? null;
    const v = t.values[i] ?? null;
    if (seen.has(k) && seen.get(k) !== v) return false;
    seen.set(k, v);
  }
  return true;
}

const SEVERITY_ORDER = { critical: 0, warning: 1, info: 2 } as const;

type Draft = Omit<
  Finding,
  "summary" | "routes_to" | "lever_label" | "group" | "repairs" | "disposition" | "answered_by"
>;

/** Where the mock's findings route, mirroring turbotab/core/stages/finding_words.py. */
const LEVERS: Record<string, [NonNullable<Finding["routes_to"]>, string]> = {
  implausible_energy: ["exclusions", "Choose an exclusion rule"],
  energy_adjustment: ["energy_adjustment", "Adjust for energy"],
  compositional_macros: ["roles", "Leave one part out"],
  repeats: ["roles", "Mark as identifier"],
  p_much_greater_than_n: ["models", "Choose penalized models"],
};

/** The M1 fields the server adds: a one-line summary, its lever (or saying there is none), a pager key. */
function voiced(drafts: Draft[]): Finding[] {
  const kind = (id: string) => id.split("__")[0]!;
  const sizes = new Map<string, number>();
  for (const d of drafts) sizes.set(kind(d.id), (sizes.get(kind(d.id)) ?? 0) + 1);
  return drafts.map((d) => {
    const lever = LEVERS[kind(d.id)];
    const claim = d.title.endsWith(".") ? d.title : `${d.title}.`;
    return {
      ...d,
      summary: lever ? claim : `${claim} No control for this yet.`,
      routes_to: lever ? lever[0] : null,
      lever_label: lever ? lever[1] : null,
      group: (sizes.get(kind(d.id)) ?? 0) > 1 ? kind(d.id) : null,
      repairs: [],
      disposition: null,
      answered_by: null,
    };
  });
}

export function findings(
  ds: MockDataset,
  lenses: Lens[],
  target: string | null,
): { findings: Finding[]; basis: string } {
  const out: Draft[] = [];
  const checked: string[] = ["structural diagnosis"];

  // Structural: two-valued text columns, repeating ids, missing cells.
  for (const col of ds.columns) {
    if (col.dtype !== "categorical" || nUnique(col) !== 2) continue;
    const vals = topValues(col, 2).map((t) => String(t.value));
    out.push({
      id: `binary_text__${col.name}`,
      severity: "warning",
      title: `\`${col.name}\` holds two text values, \`${vals[0]}\` and \`${vals[1]}\``,
      detail:
        "A model needs a number here. The column will be encoded as one indicator, and the record will say which value became 1.",
      why_it_matters: "Which value is coded 1 decides the sign of every coefficient on it.",
      affected_columns: [col.name],
      source: "structural",
      lens: null,
      evidence: null,
    });
  }
  const rep = repeats(ds);
  if (rep) {
    out.push({
      id: `repeats__${rep.column}`,
      severity: "warning",
      title: `\`${rep.column}\` repeats: ${fmt(rep.people)} people, ${rep.perPerson} rows each`,
      detail:
        "Rows from one person are not independent. A split that puts one recall in training and the other in testing measures memory, not prediction.",
      why_it_matters: "Held-out performance is inflated unless the split keeps each person whole.",
      affected_columns: [rep.column],
      source: "structural",
      lens: null,
      evidence: { status: "settled", source: "research/NUTRITION_PACK.md §03" },
    });
  }
  for (const col of ds.columns.slice(0, 200)) {
    const m = nMissing(col);
    if (m === 0) continue;
    out.push({
      id: `missing__${col.name}`,
      severity: "info",
      title: `\`${col.name}\` is missing in ${fmt(m)} rows (${((100 * m) / ds.nRows).toFixed(1)}%)`,
      detail: "Too few to change the analysis on their own; how they are filled is asked later.",
      why_it_matters: null,
      affected_columns: [col.name],
      source: "profile",
      lens: null,
      evidence: null,
    });
  }

  if (lenses.includes("dietary")) {
    checked.push("dietary pack");
    const energy = ds.columns.find((c) => /energy_kcal|kcal/i.test(c.name));
    if (energy) {
      const vals = energy.values.filter((v): v is number => typeof v === "number");
      const outside = vals.filter((v) => v < 800 || v > 4500).length;
      const low = vals.filter((v) => v < 500).length;
      const high = vals.filter((v) => v > 5000).length;
      if (outside > 0) {
        out.push({
          id: "implausible_energy",
          severity: "warning",
          title: `${fmt(outside)} recalls report energy outside 800–4,500 kcal`,
          detail: `${fmt(low)} fall below 500 kcal and ${fmt(high)} above 5,000. These are possible days and poor estimates of usual intake.`,
          why_it_matters:
            "Excluding them changes N, so it is an eligibility criterion you state, never a silent filter.",
          affected_columns: [energy.name],
          source: "pack",
          lens: "dietary",
          evidence: { status: "convention", source: "research/NUTRITION_PACK.md §02" },
        });
      }
      const shares = ds.columns.filter((c) => /_pct_kcal$/.test(c.name));
      if (shares.length >= 3) {
        out.push({
          id: "compositional_macros",
          severity: "info",
          title: `${shares.length === 4 ? "Four" : fmt(shares.length)} macronutrient shares sum to 100% in every row`,
          detail:
            "Parts of a whole are negatively correlated by construction, so their collinearity is structural rather than a finding about diet.",
          why_it_matters:
            "Substitution models, not the usual all-in regression, answer questions about them.",
          affected_columns: shares.map((c) => c.name),
          source: "pack",
          lens: "dietary",
          evidence: { status: "settled", source: "research/NUTRITION_PACK.md §05" },
        });
      }
      if (target && target !== energy.name) {
        out.push({
          id: "energy_adjustment",
          severity: "warning",
          title: `Nutrient associations with \`${target}\` are confounded by total energy`,
          detail:
            "People who eat more eat more of everything. Without adjustment, every nutrient partly measures total intake.",
          why_it_matters:
            "The adjustment is required; its form (residual or density) is asked in M1.",
          affected_columns: [energy.name, target],
          source: "pack",
          lens: "dietary",
          evidence: { status: "settled", source: "research/NUTRITION_PACK.md §04" },
        });
      }
      if (target === energy.name) {
        out.push({
          id: "energy_is_outcome",
          severity: "critical",
          title: `\`${target}\` is both the outcome and the energy-adjustment reference`,
          detail:
            "Adjusting the nutrients for the outcome itself removes the association being studied.",
          why_it_matters: "Energy adjustment will be withheld for this target.",
          affected_columns: [target],
          source: "pack",
          lens: "dietary",
          evidence: { status: "settled", source: "research/NUTRITION_PACK.md §04" },
        });
      }
    }
    if (rep && target && constantWithin(ds, rep.column, target)) {
      out.push({
        id: "outcome_constant_within_person",
        severity: "info",
        title: `\`${target}\` does not vary within a person`,
        detail:
          "It was measured once and copied onto every recall, so combining a person's recalls raises no question about which outcome to keep.",
        why_it_matters: null,
        affected_columns: [target, rep.column],
        source: "pack",
        lens: "dietary",
        evidence: null,
      });
    }
  }
  if (lenses.includes("clinical")) {
    checked.push("clinical pack");
    const hba1c = findColumn(ds, "hba1c");
    if (hba1c) {
      out.push({
        id: "units__hba1c",
        severity: "info",
        title: "`hba1c` reads as percent (NGSP), not mmol/mol",
        detail: "Every value lies between 4 and 10, the range of the percent scale.",
        why_it_matters: "The methods section must state the unit; the two scales differ tenfold.",
        affected_columns: ["hba1c"],
        source: "pack",
        lens: "clinical",
        evidence: { status: "convention", source: "research/CLINICAL_SURVEY_PACK.md §A1" },
      });
    }
  }
  if (lenses.includes("genomics")) {
    checked.push("genomics pack");
    const nums = numericCols(ds);
    if (isCountMatrix(ds)) {
      out.push({
        id: "raw_counts",
        severity: "warning",
        title: `${fmt(nums.length)} columns hold raw counts`,
        detail: "Counts are not comparable across samples until library size is accounted for.",
        why_it_matters:
          "Normalization happens inside each training fold, never on the whole table.",
        affected_columns: nums.slice(0, 40).map((c) => c.name),
        source: "pack",
        lens: "genomics",
        evidence: { status: "settled", source: "research/GENOMICS_PACK.md §02" },
      });
    }
    if (nums.length > ds.nRows) {
      out.push({
        id: "p_much_greater_than_n",
        severity: "warning",
        title: `${fmt(nums.length)} features across ${fmt(ds.nRows)} samples`,
        detail:
          "With far more columns than rows, any flexible model can fit the training rows perfectly.",
        why_it_matters:
          "Performance is only credible on rows the model never saw, with selection inside the folds.",
        affected_columns: [],
        source: "pack",
        lens: "genomics",
        evidence: { status: "settled", source: "research/GENOMICS_PACK.md §08" },
      });
    }
  }
  if (lenses.includes("metabolomics")) checked.push("metabolomics pack");
  if (lenses.includes("survey")) checked.push("survey pack");

  out.sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
  return {
    findings: voiced(out),
    basis: `Checked on all ${fmt(ds.nRows)} rows: ${checked.join(", ")}.`,
  };
}
