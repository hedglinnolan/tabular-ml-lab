/**
 * S3 — the 13 real NHANES findings, within the doctrine (BLUEPRINT §11.7):
 * ≤ 3 pushed, same-kind findings share one paged card, each is a one-line claim plus
 * its lever, the rest are counted and typed. Focusing a card puts its evidence on the
 * stage — the claim is shown on the user's data instead of argued in a paragraph.
 *
 * Claims are written here in the app's voice. Every number is read from the fixture:
 * the structural findings' counts are parsed from their `detail` sentences.
 */
import { A, viewOf, type RawFinding } from "./fixture";
import { fmtInt, fmtR } from "./format";
import { FACTS, type ScenarioId, type StageView } from "./scenarios";
import type { DistributionView, LineageView, RelationshipView, RowFlowView } from "./types";

/** Beside a finding's evidence: the live pipeline section its lever acts on. */
const liveLineage = (title: string): LineageView => ({
  kind: "lineage",
  title,
  caption: "",
  emphasis: [],
  before: null,
  after: FACTS.energyLive.lineage,
});
const liveRows: RowFlowView = {
  kind: "row_flow",
  title: "Rows, where an exclusion rule acts",
  caption: "",
  emphasis: [],
  before: FACTS.exLive.rows.steps,
  after: FACTS.exLive.rows.steps,
};

export interface FindingPage {
  key: string;
  /** The raw finding ids this page speaks for. */
  sources: string[];
  /** ≤ 20 words; backticks mark data. */
  claim: string;
  stage: { label: string; basis: string; views: StageView[] };
}


export interface FindingCard {
  id: string;
  origin: string;
  badge?: string;
  severity: string;
  pages: FindingPage[];
  /** The control that acts on the claim, or why there is none in this version. */
  lever: { label: string; to: ScenarioId } | { label: string; receipt: string } | null;
}

const F = A.findings.findings;
const byId = Object.fromEntries(F.map((f) => [f.id, f])) as Record<string, RawFinding>;
const ROLES = A.setup.state.roles;
const PREDICTORS = new Set(A.setup.predictors);
const N = A.dataset.n_rows;

/** "'imputed_bmi' holds two values — 'True' (306 rows) and 'False' (21,543 rows)…" */
function levelsOf(f: RawFinding): { level: string; n: number }[] {
  return [...f.detail.matchAll(/'([^']+)' \(([\d,]+) rows\)/g)].map((m) => ({
    level: m[1]!,
    n: Number(m[2]!.replace(/,/g, "")),
  }));
}
const falseFirst = (a: { level: string }, b: { level: string }) =>
  Number(a.level !== "False") - Number(b.level !== "False");

const blankOf = (f: RawFinding) =>
  Number((/with ([\d,]+) blank/.exec(f.detail)?.[1] ?? "0").replace(/,/g, ""));

function levelView(
  column: string,
  levels: { level: string; n: number }[],
  blank: number,
  caption: string,
  names?: string[],
): DistributionView {
  const hist = {
    edges: levels.map((_, i) => i).concat(levels.length),
    counts: levels.map((l) => l.n),
    n_missing: blank,
  };
  return {
    kind: "distribution",
    title: `${column}, as recorded`,
    caption,
    emphasis: [column],
    column,
    before: hist,
    after: hist,
    before_label: column,
    after_label: column,
    marks: [],
    levels: names ?? levels.map((l) => l.level),
  };
}

// ── 1 · energy adjustment (pack) ─────────────────────────────────────────────

const residual = A.energy_adjustment.options.find((o) => o.method === "residual")!;
const rel = viewOf(residual.preview!.views, "relationship");
const asRecorded: RelationshipView = {
  ...rel,
  title: `${rel.y_label_before} against ${rel.x_label}, as recorded`,
  caption: `\`${rel.y_label_before}\` rises with \`${rel.x_label}\` at r ${fmtR(rel.r_before)} on ${fmtInt(A.setup.n_train)} training rows.`,
  points_after: rel.points_before,
  y_label_after: rel.y_label_before,
  r_after: rel.r_before,
};

const energyCard: FindingCard = {
  id: "energy",
  origin: "dietary pack",
  badge: byId["pack::dietary::energy_adjustment"]!.evidence?.status,
  severity: byId["pack::dietary::energy_adjustment"]!.severity,
  pages: [
    {
      key: "energy",
      sources: ["pack::dietary::energy_adjustment"],
      claim: `Every nutrient tracks total energy: \`${FACTS.TOP.column}\` correlates ${fmtR(FACTS.TOP.r)} with \`${FACTS.ENERGY}\`, \`${FACTS.BOTTOM.column}\` least at ${fmtR(FACTS.BOTTOM.r)}.`,
      stage: {
        label: "Evidence",
        basis: residual.preview!.basis,
        views: [
          { view: asRecorded, evidence: true },
          {
            view: liveLineage("Columns, where the adjustment acts"),
            openLabel: "energy adjustment acts here",
            chip: `${A.setup.predictors.length} predictors`,
          },
        ],
      },
    },
  ],
  lever: { label: "Adjust for energy", to: "energy" },
};

// ── 2 · implausible intake (pack) ────────────────────────────────────────────

const neutral = A.exclusions.options.find((o) => o.key === "kcal_500_5000")!;
const kcalDist = viewOf(neutral.preview.views, "distribution");
const all = neutral.counts.by_level.all!;
const [lo, hi] = kcalDist.marks.map((m) => m.value);
const implausibleView: DistributionView = {
  ...kcalDist,
  title: `${kcalDist.column}, every loaded row`,
  caption: `${fmtInt(all.below)} rows fall below ${fmtInt(lo!)} and ${fmtInt(all.above)} above ${fmtInt(hi!)} kcal; the longest day reported is ${fmtInt(FACTS.kcalMax)}.`,
  after: kcalDist.before,
  after_label: kcalDist.before_label,
};

const implausibleCard: FindingCard = {
  id: "implausible",
  origin: "dietary pack",
  badge: byId["pack::dietary::implausible_intake"]!.evidence?.status,
  severity: byId["pack::dietary::implausible_intake"]!.severity,
  pages: [
    {
      key: "implausible",
      sources: ["pack::dietary::implausible_intake"],
      claim: `${fmtInt(neutral.counts.excluded)} rows report an implausible day: ${fmtInt(all.below)} below ${fmtInt(lo!)} kcal and ${fmtInt(all.above)} above ${fmtInt(hi!)}.`,
      stage: {
        label: "Evidence",
        basis: neutral.preview.basis,
        views: [
          { view: implausibleView, evidence: true },
          {
            view: liveRows,
            openAfter: "outcome_measured",
            openLabel: "an exclusion rule acts here",
            chip: `${fmtInt(N)} rows`,
          },
        ],
      },
    },
  ],
  lever: { label: "Choose an exclusion rule", to: "exclusions" },
};

// ── 3 · imputation flags (structural, paged) ─────────────────────────────────

const flagIds = F.filter((f) => f.id.startsWith("binary_text__imputed_")).map((f) => f.id);

function flagPage(id: string): FindingPage {
  const f = byId[id]!;
  const col = f.affected_columns[0]!;
  const levels = levelsOf(f).sort(falseFirst);
  const yes = levels.find((l) => l.level === "True")!.n;
  const kept = ROLES[col] === "flag" && !PREDICTORS.has(col);
  return {
    key: col,
    sources: [id],
    claim: `\`${col}\` is a true/false flag, true on ${fmtInt(yes)} rows${kept ? "; it stays out of the model" : ""}.`,
    stage: {
      label: "Evidence",
      basis: `All ${fmtInt(N)} loaded rows`,
      views: [
        {
          view: levelView(col, levels, 0, `\`${col}\` is true on ${fmtInt(yes)} of ${fmtInt(N)} rows.`),
          evidence: true,
        },
      ],
    },
  };
}

const flagsCard: FindingCard = {
  id: "flags",
  origin: "structure",
  severity: "warning",
  pages: flagIds.map(flagPage),
  lever: {
    label: "Review in Roles",
    receipt: `Roles recorded all ${flagIds.length} as flags; none enters the model.`,
  },
};

// ── the rest: counted and typed, one press away ──────────────────────────────

const gender = byId["binary_text__gender"]!;
const gLevels = levelsOf(gender).sort((a, b) => a.level.localeCompare(b.level));
const genderCard: FindingCard = {
  id: "gender",
  origin: "structure",
  severity: gender.severity,
  pages: [
    {
      key: "gender",
      sources: [gender.id],
      claim: "`gender` is text: `male` becomes 1, `female` 0. This version cannot change which level is 1.",
      stage: {
        label: "Evidence",
        basis: `All ${fmtInt(N)} loaded rows`,
        views: [
          {
            view: levelView(
              "gender",
              gLevels,
              0,
              `${fmtInt(gLevels[0]!.n)} rows read \`${gLevels[0]!.level}\` and ${fmtInt(gLevels[1]!.n)} read \`${gLevels[1]!.level}\`.`,
              gLevels.map((l) => `${l.level} → ${l.level === "male" ? 1 : 0}`),
            ),
            evidence: true,
          },
          { view: { ...liveLineage("Columns: gender enters as gender_male"), emphasis: ["gender"] } },
        ],
      },
    },
  ],
  lever: null,
};

const medsCols = ["meds_hbp", "meds_chol"];
const medsCard: FindingCard = {
  id: "meds",
  origin: "structure",
  severity: "warning",
  pages: medsCols.map((col) => {
    const f = byId[`binary_text__${col}`]!;
    const levels = levelsOf(f).sort(falseFirst);
    const blank = blankOf(f);
    const excluded = ROLES[col] === "excluded";
    return {
      key: col,
      sources: [f.id, `boolean_as_text__${col}`],
      claim: `\`${col}\` is True/False stored as text and blank on ${fmtInt(blank)} rows${excluded ? "; its role keeps it out of the model" : ""}.`,
      stage: {
        label: "Evidence",
        basis: `All ${fmtInt(N)} loaded rows`,
        views: [
          {
            view: levelView(
              col,
              levels,
              blank,
              `\`${col}\`: ${fmtInt(levels[1]!.n)} true, ${fmtInt(levels[0]!.n)} false, ${fmtInt(blank)} blank.`,
            ),
            evidence: true,
          },
        ],
      },
    };
  }),
  lever: null,
};

export const PUSHED: FindingCard[] = [energyCard, implausibleCard, flagsCard];
export const REST: FindingCard[] = [genderCard, medsCard];
export const ALL_FINDING_IDS = F.map((f) => f.id);

const restCount = REST.flatMap((c) => c.pages.flatMap((p) => p.sources)).length;
export const REST_LINE = `${restCount} more: \`gender\` read from text, and ${medsCols.map((c) => `\`${c}\``).join(" and ")} stored as text.`;
export const FINDINGS_BASIS = A.findings.basis;
