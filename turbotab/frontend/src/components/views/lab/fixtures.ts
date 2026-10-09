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
import { forestFromEffects, outcomeOf, table1FromArtifact, table1FromProfile, table2FromEffects } from "../adapters";
import type { ExhibitModel } from "../contracts";
import raw from "./fixtures.json";

interface Fixtures {
  nhanes: { effects: EffectsArtifact; profile: ProfileArtifact; cohort: CohortArtifact; concerns: string[] };
  timeVarying: { effects: EffectsArtifact };
}

export const FX = raw as unknown as Fixtures;

const nhanes = FX.nhanes;
export const TABLE1_COLUMNS = ["age", "gender", "bmi", "waist", "kcal", "sugar", "protein", "carb", "fat_total"];

export const table1 = table1FromArtifact(
  table1FromProfile(nhanes.profile, {
    columns: TABLE1_COLUMNS,
    unit: "participants",
    rows: nhanes.effects.rows,
    n: nhanes.cohort.n_final,
  }),
);
export const table2 = table2FromEffects(nhanes.effects);
export const forest = forestFromEffects(nhanes.effects);
export const table2Ratio = table2FromEffects(FX.timeVarying.effects);
export const forestRatio = forestFromEffects(FX.timeVarying.effects);

const plain = (s: string) => s.replaceAll("`", "");

/** The paper's exhibits as the exhibit model (C7a) would place them by default. */
export function exhibitModel(): ExhibitModel {
  const fam = nhanes.effects.families[0]!;
  const m2 = fam.sequence.find((s) => s.key === "model_2")!;
  const e = m2.effects!.find((x) => x.feature === nhanes.effects.exposure)!;
  const dir = e.estimate! < 0 ? "lower" : "higher";
  const abs = (v: number | null) => fmtNum(v === null ? null : Math.abs(v));
  const [lo, hi] = [e.ci_low!, e.ci_high!].map(Math.abs).sort((a, b) => a - b);
  const results =
    `Among ${fmtInt(m2.n_rows)} participants, each additional unit of ${nhanes.effects.exposure} was associated with a ` +
    `${abs(e.estimate)} ${dir} mean ${outcomeOf(nhanes.effects) ?? "outcome"} ` +
    `(95% CI ${fmtNum(lo!)} to ${fmtNum(hi!)}) in the primary model, adjusted for ${m2.adjusted_for.length} characteristics (Table 2).`;
  const forestCaption = `${forest.measure}, with 95% intervals, by model.`;
  return {
    exhibits: [
      {
        key: "table1",
        number: "Table 1",
        kind: "table",
        caption: `${table1.title}.`,
        placement: "results",
        placement_allowed: ["results", "supplement", "left_out"],
        fixed_reason: null,
        wordings: [],
        view: { kind: "table", source: "profile" },
      },
      {
        key: "table2",
        number: "Table 2",
        kind: "table",
        caption: table2.title,
        placement: "results",
        placement_allowed: ["results"],
        fixed_reason: "the locked primary always stays in Results.",
        wordings: [{ strength: "association", text: results, recommended: true }],
        view: { kind: "table", source: "effects" },
      },
      {
        key: "figure1",
        number: "Figure 1",
        kind: "figure",
        caption: forestCaption,
        placement: "supplement",
        placement_allowed: ["results", "discussion", "supplement", "left_out"],
        fixed_reason: null,
        wordings: [],
        view: { kind: "forest", source: "effects" },
      },
    ],
    text: { results: [results], discussion: nhanes.concerns.map(plain) },
  };
}
