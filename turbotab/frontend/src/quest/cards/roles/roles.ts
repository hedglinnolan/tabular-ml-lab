/**
 * A column's own role confirmation (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14 rule 2:
 * a proposal below high confidence "needs its own confirmation; it never rides along in a bulk
 * 'confirm all'"). The engine lists each column a roles answer recorded below high as proposed as a
 * counted line of Your data of its own (core/quest.py `_role_lines`): a reading line with one role
 * item, whose ways forward are each one `confirm_role` of that column. Pure; RoleCard.tsx draws it.
 * fixture.json is the engine's own (turbotab/core/tests/test_quest_unconfirmed_roles.py writes it).
 */
import { PREDICTOR_ROLES, type Role } from "../../../api/m1-types";
import type { QuestLine, QuestLog, ReadItem, ReadOption, Refusal } from "../../../api/schema";
import FIXTURE from "./fixture.json";

/**
 * What the predictor roles' names differ by under Predict in a project with these lenses
 * (core/readings.py `predict_kind`, whose lens lists the fixture holds): the energy-aware fill under
 * the dietary lens, normalization and batch correction under an assay lens, both, or neither.
 */
export type PredictKind = keyof typeof FIXTURE.under.prediction.kinds;

export function predictKind(lenses: readonly string[] | null | undefined): PredictKind {
  const { dietary, assay } = FIXTURE.under.prediction.lenses;
  const on = new Set(lenses ?? []);
  const d = dietary.some((k) => on.has(k));
  const a = assay.some((k) => on.has(k));
  return d && a ? "both" : d ? "dietary" : a ? "assay" : "neither";
}

/** One predictor role under Predict as the roles card names it (core/readings.py
 *  `predictor_groups`): `one` in a role line's words, `head` the group's name for one column and
 *  for many, and `note` the steps that treat it differently, as a sentence. */
export interface PredictorGroup {
  one: string;
  head: readonly [string, string];
  note: string;
}

/** The three predictor roles' groups under Predict in a project with these lenses: the engine's
 *  words (the roles ruling of 2026-10-17), from the fixture its test writes. */
export function predictorGroups(lenses: readonly string[] | null | undefined): Record<"exposure" | "energy" | "covariate", PredictorGroup> {
  const g = FIXTURE.under.prediction.kinds[predictKind(lenses)].groups;
  const group = (x: { one: string; head: string[]; note: string }): PredictorGroup => ({ one: x.one, head: [x.head[0]!, x.head[1]!], note: x.note });
  return { exposure: group(g.exposure), energy: group(g.energy), covariate: group(g.covariate) };
}

/** Whether a role enters the models as a predictor (core/readings.py PREDICTOR_ROLES). */
export function isPredictorRole(role: string): boolean {
  return (PREDICTOR_ROLES as readonly string[]).includes(role as Role);
}

/** The crosswalk card a column's role line carries (core/quest.py ROLE_LINE, `confirm_role`'s). */
export const ROLE_LINE_ID = "decision:confirm_role";

export function isRoleLine(line: Pick<QuestLine, "source" | "id">): boolean {
  return line.source === "reading" && line.id === ROLE_LINE_ID;
}

/** The line's role reading: the column, its role and the evidence for it. */
export function roleItem(line: Pick<QuestLine, "items">): ReadItem | null {
  return line.items.find((i) => i.kind === "role") ?? null;
}

interface RoleDecision {
  kind: "confirm_role";
  column: string;
  role: string;
}

function asRole(o: ReadOption): RoleDecision | null {
  const d = o.decision as Partial<RoleDecision> | null | undefined;
  return d && d.kind === "confirm_role" && typeof d.column === "string" && typeof d.role === "string" ? (d as RoleDecision) : null;
}

/**
 * The ways forward the card offers: each one `confirm_role` of the item's own column, so a press
 * confirms one column, never several, and never another (a block confirmation, or any other
 * decision, is not offered here). The confirmation of the role as recorded comes first.
 */
export function ownOptions(item: ReadItem): { option: ReadOption; role: string; confirms: boolean }[] {
  return item.options
    .map((option) => ({ option, d: asRole(option) }))
    .filter((x): x is { option: ReadOption; d: RoleDecision } => x.d !== null && x.d.column === item.column)
    .map(({ option, d }) => ({ option, role: d.role, confirms: d.role === item.value }))
    .sort((a, b) => Number(b.confirms) - Number(a.confirms));
}

/**
 * A refusal as the card answers a press with it: its ways forward cut to those that record nothing
 * (a way to look elsewhere) or confirm this column alone, so no press here confirms several columns,
 * whatever a refusal offers (a replayed capture offers the block it recorded).
 */
export function ownRefusal(refusal: Refusal, column: string): Refusal {
  const own = (decision: unknown) => {
    const d = decision as Partial<RoleDecision> | null | undefined;
    return !d || (d.kind === "confirm_role" && d.column === column);
  };
  return { ...refusal, error: { ...refusal.error, exits: refusal.error.exits.filter((e) => own(e.decision)) } };
}

/** Your data's role lines still waiting for their own confirmation. */
export function waitingRoles(log: QuestLog): QuestLine[] {
  return log.stages.flatMap((s) => s.lines).filter((l) => isRoleLine(l) && l.status === "open");
}

/** The waiting role lines whose column was proposed in a predictor role (W1-T): each would enter
 *  the models once confirmed, so it is not a predictor yet. A flag, a time, a design column, an
 *  identifier and a column proposed as left out never enter them as proposed, so none is counted. */
export function waitingPredictors(log: QuestLog): QuestLine[] {
  return waitingRoles(log).filter((l) => isPredictorRole(roleItem(l)?.value ?? ""));
}

/** "3 columns wait for their own confirmation": how many wait, without a period. */
export function waitingClause(n: number): string {
  return n === 1 ? "1 column waits for its own confirmation" : `${n.toLocaleString("en-US")} columns wait for their own confirmation`;
}

/** How many columns proposed as predictors wait (the walkthrough's ruling words; count them with
 *  `waitingPredictors`); null when none does. The count is the served lines'. */
export function waitingSentence(n: number): string | null {
  if (n <= 0) return null;
  return `${waitingClause(n)} and ${n === 1 ? "is not a predictor" : "are not predictors"} yet.`;
}
