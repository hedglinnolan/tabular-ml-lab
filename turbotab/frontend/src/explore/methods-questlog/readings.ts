/**
 * The Data section's readings as slots (BLUEPRINT §14.2 folded into §11.4): each unsettled reading
 * the open question needs, with the engine's guess, its evidence and its alternatives; a
 * homogeneous family is one slot that lists its members.
 *
 * Built from the ask the engine attaches to the open question at each moment of the scenario: the
 * screens ask `kcal`'s unit and days, the exposure and its effect asks the roles proposed below
 * high confidence, the fit asks whether whole-number columns hold codes or amounts.
 */
import { INF, type AskGroup, type Exit, type RoleProposal, type StepAsk } from "./data";

export interface ReadingSlot {
  id: string;
  kind: string;
  columns: string[];
  guessWords: string;
  /** The engine's evidence for the guess (the ask's own, and the role proposal's reason). */
  evidence: string[];
  confidence: RoleProposal["confidence"] | null;
  consumer: string;
  /** The answers the engine offers for this reading, the guess first. */
  options: Exit[];
}

const proposal = (c: string) => INF.roles.columns.find((p) => p.column === c);

function exitsFor(group: AskGroup, exits: Exit[]): Exit[] {
  const out = exits.filter((e) => {
    const d = e.decision as { kind?: string; reading?: string; column?: string } | null;
    if (!d) return false;
    if (d.kind === "set_column_unit") return group.kind === "unit" && group.columns.includes(String(d.column));
    return d.kind === "confirm_reading" && d.reading === group.kind && group.columns.includes(String(d.column));
  });
  // The guess first, then its alternatives; a family keeps one line per answer (its first member's).
  const seen = new Set<string>();
  return out.filter((e) => {
    const d = e.decision as { value?: string; days?: number };
    const value = String(d.value ?? d.days);
    if (seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function slotOf(group: AskGroup, consumer: string, exits: Exit[], i: number): ReadingSlot {
  const first = group.columns[0]!;
  const p = proposal(first);
  const evidence = [group.evidence];
  // The ask's evidence for a role is how it was recorded; the proposal says why it was guessed.
  if (group.kind === "role" && p) evidence.unshift(p.reason);
  return {
    id: `${group.kind}:${group.columns.join(",")}:${i}`,
    kind: group.kind,
    columns: group.columns,
    guessWords: group.guess_words,
    evidence,
    confidence: group.kind === "role" && p ? p.confidence : null,
    consumer,
    options: exitsFor(group, exits),
  };
}

/** The open question's readings, in the card's order. */
export function readingSlots(ask: StepAsk | null): ReadingSlot[] {
  return ask ? ask.groups.map((g, i) => slotOf(g, ask.consumer, ask.exits, i)) : [];
}

/** The ask's block confirm: it lists exactly the readings it settles, each with its value. */
export function blockOf(ask: StepAsk | null) {
  const exit = ask?.exits[0];
  const d = exit?.decision as unknown as { kind?: string; items?: { reading: string; column: string; value: string }[] } | null;
  if (!exit || !d || d.kind !== "confirm_readings" || !d.items) return null;
  return { label: exit.label, items: d.items };
}

export const READING_WORDS: Record<string, string> = {
  role: "role",
  code_or_count: "code or amount",
  unit: "unit and days",
};

export const VALUE_WORDS: Record<string, string> = {
  covariate: "a covariate",
  flag: "a flag on another column",
  amount: "an amount",
  code: "codes for categories",
};
