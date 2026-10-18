/**
 * A column's own role confirmation (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14 rule 2:
 * a proposal below high confidence "needs its own confirmation; it never rides along in a bulk
 * 'confirm all'"). The engine lists each column a roles answer recorded below high as proposed in a
 * counted line of Your data (core/quest.py `_role_lines`): a reading line with one role item, whose
 * ways forward are each one `confirm_role` of that column; or, where the roles stage proposed
 * several columns with one proposal (one role, one confidence, one reason word for word: ruling R-g,
 * one judgment of the recognizer), one group line: its first item (`role_group`) reads the group
 * whole and carries its one press, a `confirm_readings` of exactly its columns, and each column
 * follows as a role item whose presses move it alone. Pure; RoleCard.tsx draws it. The words are
 * the engine's; fixture.json (written by turbotab/core/tests/test_quest_unconfirmed_roles.py) is a
 * test's and the mock's input, never the product's.
 */
import { PREDICTOR_ROLES, type PredictorGroups, type Role } from "../../../api/m1-types";
import type { QuestLine, QuestLog, ReadItem, ReadOption, Refusal } from "../../../api/schema";

/** One predictor role under Predict as the roles card names it (core/readings.py
 *  `predictor_groups`): `one` in a role line's words, `head` the group's name for one column and
 *  for many, and `note` the steps that treat it differently, as a sentence (empty where the project
 *  runs none). */
export interface PredictorGroup {
  one: string;
  head: readonly [string, string];
  note: string;
}

/** Whether the energy-aware fill runs with these roles (core/readings.py `fill_runs`): a column
 *  holds the energy role and a Predictor column carries energy (the columns the server reads as
 *  carrying energy, `energy_bearing`). */
export function fillRuns(roles: Readonly<Record<string, string>> | null | undefined, energyBearing: readonly string[]): boolean {
  const all = Object.entries(roles ?? {});
  const bearing = new Set(energyBearing);
  return all.some(([, r]) => r === "energy") && all.some(([c, r]) => r === "exposure" && bearing.has(c));
}

/** The three predictor roles' groups under Predict, as the server serves them for the project's
 *  lenses and task beside the roles stage's proposals (`predictor_groups`; W1 part 2's U6), each
 *  note naming only the steps the project can run with these roles (the energy-aware fill only
 *  where it would run). Null where the server served none. */
export function servedGroups(
  served: PredictorGroups | null | undefined,
  roles: Readonly<Record<string, string>> | null | undefined,
): Record<"exposure" | "energy" | "covariate", PredictorGroup> | null {
  if (!served) return null;
  const g = fillRuns(roles, served.energy_bearing) ? served.fill : served.no_fill;
  const group = (role: string): PredictorGroup => {
    const x = g[role];
    return { one: x?.one ?? role, head: [x?.head[0] ?? role, x?.head[1] ?? role], note: x?.note ?? "" };
  };
  return { exposure: group("exposure"), energy: group("energy"), covariate: group("covariate") };
}

/** Whether a role enters the models as a predictor (core/readings.py PREDICTOR_ROLES). */
export function isPredictorRole(role: string): boolean {
  return (PREDICTOR_ROLES as readonly string[]).includes(role as Role);
}

/** The crosswalk card a column's role line carries (core/quest.py ROLE_LINE, `confirm_role`'s). */
export const ROLE_LINE_ID = "decision:confirm_role";
/** A group line's first item: the group read whole, with its one press (core/quest.py
 *  ROLE_GROUP_ITEM). */
export const ROLE_GROUP_ITEM = "role_group";

export function isRoleLine(line: Pick<QuestLine, "source" | "id">): boolean {
  return line.source === "reading" && line.id === ROLE_LINE_ID;
}

/** The line's role reading: the column, its role and the evidence for it (a group line's first
 *  column's own). */
export function roleItem(line: Pick<QuestLine, "items">): ReadItem | null {
  return line.items.find((i) => i.kind === "role") ?? null;
}

/** A group line's first item (ruling R-g): how many columns, their role, the confidence and the
 *  shared reason, with the group's one press. Null on a column's own line. */
export function groupItem(line: Pick<QuestLine, "items">): ReadItem | null {
  return line.items[0]?.kind === ROLE_GROUP_ITEM ? line.items[0] : null;
}

/** A group line's columns, each its own role item, in the engine's order. */
export function memberItems(line: Pick<QuestLine, "items">): ReadItem[] {
  return line.items.filter((i) => i.kind === "role");
}

interface GroupDecision {
  kind: "confirm_readings";
  items: { reading: string; column: string; value: string }[];
}

/** Whether a decision confirms exactly these columns, each as `role`, in one block. */
function confirmsWhole(decision: unknown, columns: readonly string[], role: string): boolean {
  const d = decision as Partial<GroupDecision> | null | undefined;
  if (!d || d.kind !== "confirm_readings" || !Array.isArray(d.items)) return false;
  const listed = d.items.map((i) => i.column);
  return (
    d.items.every((i) => i.reading === "role" && i.value === role) &&
    listed.length === columns.length &&
    new Set(listed).size === listed.length &&
    columns.every((c) => listed.includes(c))
  );
}

/**
 * A group line's one press: the engine's `confirm_readings` of exactly the group's columns that wait,
 * each with its proposed role (the engine accepts no other block of roles below high), never a
 * press that lists a column of another line. Null once the group is answered.
 */
export function groupPress(line: Pick<QuestLine, "items" | "status">): ReadOption | null {
  const head = groupItem(line);
  if (!head || line.status !== "open") return null;
  const columns = memberItems(line).map((i) => i.column);
  return head.options.find((o) => confirmsWhole(o.decision, columns, head.value)) ?? null;
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

/**
 * A refusal as a group card answers a press with it: its ways forward cut to those that record
 * nothing, confirm this group whole (its columns that wait, each as proposed: ruling R-g), or move one
 * of its columns alone, so no press here confirms a column of another line.
 */
export function groupRefusal(refusal: Refusal, line: Pick<QuestLine, "items">): Refusal {
  const head = groupItem(line);
  const columns = memberItems(line).map((i) => i.column);
  const own = (decision: unknown) => {
    const d = decision as Partial<RoleDecision> | null | undefined;
    if (!d) return true;
    if (d.kind === "confirm_role") return columns.includes(String(d.column));
    return !!head && confirmsWhole(decision, columns, head.value);
  };
  return { ...refusal, error: { ...refusal.error, exits: refusal.error.exits.filter((e) => own(e.decision)) } };
}

/** Your data's role lines still waiting for their confirmation. */
export function waitingRoles(log: QuestLog): QuestLine[] {
  return log.stages.flatMap((s) => s.lines).filter((l) => isRoleLine(l) && l.status === "open");
}

/** The waiting role lines whose columns were proposed in a predictor role (W1-T): each would enter
 *  the models once confirmed, so it is not a predictor yet. A flag, a time, a design column, an
 *  identifier and a column proposed as left out never enter them as proposed, so none is counted. */
export function waitingPredictors(log: QuestLog): QuestLine[] {
  return waitingRoles(log).filter((l) => isPredictorRole((groupItem(l) ?? roleItem(l))?.value ?? ""));
}

/** How many columns those lines hold (`waitingPredictors`): a column's own line one, a group line
 *  each of its columns (ruling R-g), counted from the served lines' items. */
export function waitingPredictorColumns(log: QuestLog): number {
  return waitingPredictors(log).reduce((n, l) => n + (groupItem(l) ? memberItems(l).length : 1), 0);
}

/** "3 columns wait for their own confirmation": how many wait, without a period. */
export function waitingClause(n: number): string {
  return n === 1 ? "1 column waits for its own confirmation" : `${n.toLocaleString("en-US")} columns wait for their own confirmation`;
}

/** How many columns proposed as predictors wait (the walkthrough's ruling words; count them with
 *  `waitingPredictorColumns`); null when none does. The count is the served lines'. */
export function waitingSentence(n: number): string | null {
  if (n <= 0) return null;
  return `${waitingClause(n)} and ${n === 1 ? "is not a predictor" : "are not predictors"} yet.`;
}
