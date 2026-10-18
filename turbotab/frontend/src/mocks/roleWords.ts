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

/** Whether the screened elastic net is on the Predict shelf of a project with this task
 *  (core/readings.py `screen_on_shelf`). No task read yet: no. */
export function screenOnShelf(task: string | null | undefined): boolean {
  return !!task && (ROLES.under.prediction.screen_tasks as readonly string[]).includes(task);
}

/** The engine's names of the nutrients that carry energy, as the mock reads its own tables' names
 *  (core/stages/rows.py `energy_bearing` reads a column's name; the mock's tables name these). */
const ENERGY_BEARING = /^(protein|carb|carbohydrate|sugar|fat|fat_total|fat_sat|fat_mon|fat_poly|alcohol|fiber)(_g)?$/i;

/** The predictor groups the server serves beside the roles stage's proposals
 *  (core/readings.py `served_groups`), for a mock project's lenses, task and columns. */
export function mockPredictorGroups(
  lenses: readonly string[] | null | undefined,
  task: string | null | undefined,
  columns: readonly string[],
): PredictorGroups {
  const shelf = ROLES.under.prediction.kinds[predictKind(lenses)].groups[screenOnShelf(task) ? "screen" : "no screen"];
  return {
    fill: shelf.fill as PredictorGroups["fill"],
    no_fill: shelf.no_fill as PredictorGroups["no_fill"],
    energy_bearing: columns.filter((c) => ENERGY_BEARING.test(c)),
  };
}
