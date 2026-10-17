/**
 * A column's own role confirmation (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14 rule 2:
 * a proposal below high confidence "needs its own confirmation; it never rides along in a bulk
 * 'confirm all'"). The engine lists each column a roles answer recorded below high as proposed as a
 * counted line of Your data of its own (core/quest.py `_role_lines`): a reading line with one role
 * item, whose ways forward are each one `confirm_role` of that column. Pure; RoleCard.tsx draws it.
 * fixture.json is the engine's own (turbotab/core/tests/test_quest_unconfirmed_roles.py writes it).
 */
import type { QuestLine, QuestLog, ReadItem, ReadOption, Refusal } from "../../../api/schema";

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

/** How many columns wait (the walkthrough's ruling words); null when none does. The count is the
 *  served lines'. */
export function waitingSentence(n: number): string | null {
  if (n <= 0) return null;
  if (n === 1) return "1 column waits for its own confirmation and is not a predictor yet.";
  return `${n.toLocaleString("en-US")} columns wait for their own confirmation and are not predictors yet.`;
}
