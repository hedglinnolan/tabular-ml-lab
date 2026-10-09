/**
 * The /lab/views fixtures for the table, forest and page views: the engine's own artifacts, cut
 * from the captured journeys by extract_fixtures.py, and the view inputs built from them through
 * the same adapters a screen will use. The page's drafted sentences are the one thing the engine
 * does not serve yet (the exhibit model, SIZING C7a): the Results sentence here is composed from
 * the engine's numbers and the Discussion is the engine's own concerns.
 */
import type { EffectsArtifact } from "../../../api/m3-types";
import type { CohortArtifact } from "../../../api/m1-types";
import type { ProfileArtifact } from "../../../api/schema";
import { fmtInt, fmtNum } from "../../stage/format";
import {
  LOCK_GATE,
  forestFromEffects,
  numberInPlacementOrder,
  outcomeOf,
  sentence,
  table1FromArtifact,
  table1FromProfile,
  table2FromEffects,
} from "../adapters";
import type { ExhibitEntry, ExhibitModel } from "../contracts";
import type { Placement } from "../types";
import raw from "./fixtures.json";

interface Fixtures {
  nhanes: { effects: EffectsArtifact; profile: ProfileArtifact; cohort: CohortArtifact; concerns: string[] };
  timeVarying: { effects: EffectsArtifact };
}

export const FX = raw as unknown as Fixtures;

/** The gate's line on the lab's "before Fit" samples. */
export const GATE = LOCK_GATE;

const nhanes = FX.nhanes;
export const TABLE1_COLUMNS = ["age", "gender", "bmi", "waist", "kcal", "sugar", "protein", "carb", "fat_total"];

/** The codebook's names for Table 1's columns, with their units. The engine serves a unit for kcal
 *  alone (the roles artifact), so these stand in for the codebook until it serves each column's
 *  paper label (an engine contract item). NHANES: age in years, measured BMI and waist, intakes
 *  from the first day's 24-hour recall. */
export const TABLE1_LABELS: Record<string, string> = {
  age: "Age, years",
  gender: "Gender",
  bmi: "Body mass index, kg/m²",
  waist: "Waist circumference, cm",
  kcal: "Energy intake, kcal/day",
  sugar: "Total sugars, g/day",
  protein: "Protein, g/day",
  carb: "Carbohydrate, g/day",
  fat_total: "Total fat, g/day",
};

export const table1Artifact = table1FromProfile(nhanes.profile, {
  columns: TABLE1_COLUMNS,
  unit: "participants",
  outcome: outcomeOf(nhanes.effects),
  analyzed: nhanes.cohort.n_final,
  labels: TABLE1_LABELS,
});
/** Table 1 has no estimate: its gate is open. */
export const table1 = table1FromArtifact(table1Artifact, null);
export const table2 = table2FromEffects(nhanes.effects);
export const forest = forestFromEffects(nhanes.effects);
export const table2Ratio = table2FromEffects(FX.timeVarying.effects);
export const forestRatio = forestFromEffects(FX.timeVarying.effects);

/**
 * Several exposures, for the comparison palette and Table 2's group rows. No captured journey
 * declares more than one exposure, so this one is cut from NHANES's own fit: sugar's estimates
 * from the effects, and protein's and carbohydrate's from the same fitted models' appendix terms
 * (the unadjusted model holds sugar alone). Every number is the engine's; only the choice to call
 * protein and carbohydrate exposures is the lab's.
 */
export const multiEffects: EffectsArtifact = (() => {
  const exposures = ["sugar", "protein", "carb"];
  const fam = nhanes.effects.families[0]!;
  const terms = new Map((fam.appendix ?? []).map((a) => [a.key, a.terms]));
  return {
    ...nhanes.effects,
    exposures,
    families: [
      {
        ...fam,
        sequence: fam.sequence.map((fit) => ({
          ...fit,
          effects: exposures.flatMap((x) => {
            const own = fit.effects?.find((e) => e.feature === x);
            if (own) return [own];
            const t = terms.get(fit.key)?.find((e) => e.feature === x);
            if (!t) return [];
            const { why: _why, ...effect } = t;
            return [effect];
          }),
        })),
      },
    ],
  };
})();
export const table2Multi = table2FromEffects(multiEffects);
export const forestMulti = forestFromEffects(multiEffects);

/** The paper's exhibits as the exhibit model (C7a) would place them, numbered in placement
 *  order; `placements` moves any of them first, as the author's choice would. */
export function exhibitModel(placements: Partial<Record<string, Placement>> = {}): ExhibitModel {
  const fam = nhanes.effects.families[0]!;
  const m2 = fam.sequence.find((s) => s.key === "model_2")!;
  const e = m2.effects!.find((x) => x.feature === nhanes.effects.exposure)!;
  const dir = e.estimate! < 0 ? "lower" : "higher";
  const abs = (v: number | null) => fmtNum(v === null ? null : Math.abs(v));
  const [lo, hi] = [e.ci_low!, e.ci_high!].map(Math.abs).sort((a, b) => a - b);
  const forestCaption = `${forest.measure}, with 95% intervals, by model.`;
  const entries: Omit<ExhibitEntry, "number">[] = [
    {
      key: "table1",
      kind: "table",
      caption: `${table1.title}.`,
      placement: placements.table1 ?? "results",
      placement_allowed: ["results", "supplement", "left_out"],
      fixed_reason: null,
      wordings: [],
      view: { kind: "table", source: "profile" },
    },
    {
      key: "table2",
      kind: "table",
      caption: table2.title,
      placement: placements.table2 ?? "results",
      placement_allowed: ["results"],
      fixed_reason: "the locked primary always stays in Results.",
      wordings: [],
      view: { kind: "table", source: "effects" },
    },
    {
      key: "figure1",
      kind: "figure",
      caption: forestCaption,
      placement: placements.figure1 ?? "supplement",
      placement_allowed: ["results", "discussion", "supplement", "left_out"],
      fixed_reason: null,
      wordings: [],
      view: { kind: "forest", source: "effects" },
    },
  ];
  const numbers = numberInPlacementOrder(entries);
  const results =
    `Among ${fmtInt(m2.n_rows)} participants, each additional unit of ${nhanes.effects.exposure} was associated with a ` +
    `${abs(e.estimate)} ${dir} mean ${outcomeOf(nhanes.effects) ?? "outcome"} ` +
    `(95% CI ${fmtNum(lo!)} to ${fmtNum(hi!)}) in the primary model, adjusted for ${m2.adjusted_for.length} characteristics (${numbers.get("table2")}).`;
  const exhibits: ExhibitEntry[] = entries.map((x) => ({
    ...x,
    number: numbers.get(x.key) ?? null,
    wordings: x.key === "table2" ? [{ strength: "association", text: results, recommended: true }] : x.wordings,
  }));
  return { exhibits, text: { results: [results], discussion: nhanes.concerns.map(sentence) } };
}
