/**
 * The engine's role words as the mock serves them (dev:mock and tests only): read from
 * src/quest/cards/roles/fixture.json, which turbotab/core/tests/test_quest_unconfirmed_roles.py
 * writes from core/quest.py and core/readings.py. The product reads these words from the server
 * (the role lines, and the predictor groups served beside the roles stage's proposals); only the
 * mock and the tests read the fixture (W1 part 2's U6).
 */
import type { PredictorGroups } from "../api/m1-types";
import ROLES from "../quest/cards/roles/fixture.json";

/** What the predictor roles' names differ by under Predict in a project with these lenses
 *  (core/readings.py `predict_kind`, whose lens lists the fixture holds). */
export type PredictKind = keyof typeof ROLES.under.prediction.kinds;

export function predictKind(lenses: readonly string[] | null | undefined): PredictKind {
  const { dietary, assay } = ROLES.under.prediction.lenses;
  const on = new Set(lenses ?? []);
  const d = dietary.some((k) => on.has(k));
  const a = assay.some((k) => on.has(k));
  return d && a ? "both" : d ? "dietary" : a ? "assay" : "neither";
}

/** The engine's names of the nutrients that carry energy, as the mock reads its own tables' names
 *  (core/stages/rows.py `energy_bearing` reads a column's name; the mock's tables name these). */
const ENERGY_BEARING = /^(protein|carb|carbohydrate|sugar|fat|fat_total|fat_sat|fat_mon|fat_poly|alcohol|fiber)(_g)?$/i;

/** The predictor groups the server serves beside the roles stage's proposals
 *  (core/readings.py `served_groups`), for a mock project's lenses and columns: named by the lenses
 *  alone, each note empty (the ruling of 2026-10-18), and the same under `fill` and `no_fill`. The
 *  task is the server's argument, which no group reads. */
export function mockPredictorGroups(
  lenses: readonly string[] | null | undefined,
  _task: string | null | undefined,
  columns: readonly string[],
): PredictorGroups {
  const groups = ROLES.under.prediction.kinds[predictKind(lenses)].groups as PredictorGroups["no_fill"];
  return { fill: groups, no_fill: groups, energy_bearing: columns.filter((c) => ENERGY_BEARING.test(c)) };
}
