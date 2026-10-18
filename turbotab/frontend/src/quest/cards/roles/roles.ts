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
 *  `predictor_groups`): `one` in a role line's words, and `head` the group's name for one column and
 *  for many. The card names no step that treats a group apart (the ruling of 2026-10-18), so a
 *  served group's `note`, which the server keeps empty, is not read. */
export interface PredictorGroup {
  one: string;
  head: readonly [string, string];
}

/** The three predictor roles' groups under Predict, as the server serves them for the project's
 *  lenses beside the roles stage's proposals (`predictor_groups`; W1 part 2's U6). The served shape
 *  holds each group twice (`fill`, `no_fill`), the same since the groups read the lenses alone
 *  (core/readings.py `served_groups`), so either is the card's. Null where the server served none. */
export function servedGroups(
  served: PredictorGroups | null | undefined,
): Record<"exposure" | "energy" | "covariate", PredictorGroup> | null {
  if (!served) return null;
  const group = (role: string): PredictorGroup => {
    const x = served.no_fill[role];
    return { one: x?.one ?? role, head: [x?.head[0] ?? role, x?.head[1] ?? role] };
  };
  return { exposure: group("exposure"), energy: group("energy"), covariate: group("covariate") };
}

/** The Predict roles card's one sentence about its predictor groups (core/readings.py
 *  `predictors_sentence`, the rulings at W1-V's cap and at W1-X's), true in every project: `n`
 *  columns hold a predictor role, whichever group holds them. It makes no claim about steps, nor
 *  about what the models get, since a missing-values answer or a repair can leave a predictor-role
 *  column out of them. `count` is `n` as the card prints it. */
export function predictorsSentence(n: number, count: string): string {
  return n === 1
    ? "One column holds a predictor role."
    : `${count} columns hold a predictor role, whichever group holds them.`;
}

/** The roles stage's clauses in Estimate's words, each with its words under Predict (core/readings.py
 *  `_STUDY_CLAUSES`, which the engine's `role_evidence` applies wherever it quotes a reason; the
 *  roles fixture holds the engine's, and roles.test.tsx pins these to them): a nutrient is no longer
 *  "what you study", a characteristic no longer "adjusted for rather than studied", and a column
 *  read "as a covariate" is read as a predictor. */
export const STUDY_CLAUSES: readonly (readonly [string, string])[] = [
  [": what you study;", ";"],
  [", usually adjusted for rather than studied.", "."],
  [", adjusted for rather than studied.", "."],
  [": what you study under the omics lens.", " under the omics lens."],
  [": what you study under the dietary lens.", ", under the dietary lens."],
  [": what you study, not", ", not"],
  ["read as what you study", "read as a predictor"],
  ["read as a covariate", "read as a predictor"],
];

/** The roles stage's `reason` for a column's proposal, said under the project's goal: under
 *  prediction in Predict's words (`STUDY_CLAUSES`, in order, as core/readings.py `role_evidence`
 *  replaces them), else as the stage said it. The served artifact holds the stage's own words. */
export function roleEvidence(reason: string, purpose: string | null | undefined): string {
  if (purpose !== "prediction") return reason;
  return STUDY_CLAUSES.reduce((said, [estimate, predict]) => said.replaceAll(estimate, predict), reason);
}

/** Whether a role is a predictor role (core/readings.py PREDICTOR_ROLES); a column that holds one
 *  is in the models unless an answer left it out (core/decisions.py `left_out`). */
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

/** The columns a group's first item confirms in its one press (its `confirm_readings`, each as the
 *  role proposed), in the engine's order; empty once no press waits. */
function pressedColumns(head: ReadItem): string[] {
  for (const o of head.options) {
    const d = o.decision as Partial<GroupDecision> | null | undefined;
    if (d && d.kind === "confirm_readings" && Array.isArray(d.items) && d.items.every((i) => i.reading === "role" && i.value === head.value)) {
      return d.items.map((i) => i.column);
    }
  }
  return [];
}

/** One reading row of a line, or a group's (ruling R-g): its first item and the columns it confirms. */
export interface ReadingBlock {
  head: ReadItem;
  members: ReadItem[];
}

/**
 * A reading line's items as rows (Readings.tsx): a group's first item (`role_group`) with the role
 * items its one press lists, which follow it (core/quest.py `_ask_items`: the fit's refusal asks a
 * judgment whole), as one block; every other item a block of its own.
 */
export function readingBlocks(items: readonly ReadItem[]): ReadingBlock[] {
  const out: ReadingBlock[] = [];
  let i = 0;
  while (i < items.length) {
    const head = items[i]!;
    i += 1;
    const listed = head.kind === ROLE_GROUP_ITEM ? new Set(pressedColumns(head)) : new Set<string>();
    const members: ReadItem[] = [];
    while (i < items.length && items[i]!.kind === "role" && listed.has(items[i]!.column)) {
      members.push(items[i]!);
      i += 1;
    }
    out.push({ head, members });
  }
  return out;
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
