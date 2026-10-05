/**
 * The Data section's readings as slots (BLUEPRINT §14.2 folded into §11.4): each unsettled reading
 * a number-changing consumer needs, with the engine's guess, its evidence and its alternatives,
 * grouped by consequence; a homogeneous family is one slot that lists its members.
 *
 * Built from the asks the engine attached to three questions of the same drive: the eligibility
 * question ("the screens": `kcal`'s days), the exposure question ("the exposure and its effect":
 * the roles proposed below high confidence) and the model question ("the fit": code or amount).
 */
import { INF, type AskGroup, type Exit, type RoleProposal } from "./data";

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

function stepAsk(moment: "m2" | "m3" | "m5", key: string) {
  return INF.moments[moment].view.interview.find((s) => s.key === key)?.ask ?? null;
}

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
    const value = String((e.decision as { value?: string; days?: number }).value ?? (e.decision as { days?: number }).days);
    if (seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function slotOf(group: AskGroup, consumer: string, exits: Exit[], i: number): ReadingSlot {
  const first = group.columns[0]!;
  const p = proposal(first);
  const evidence = [group.evidence];
  if (group.kind === "role" && p) {
    // The ask's evidence for a role is how it was recorded; the proposal says why it was guessed.
    evidence.unshift(p.reason);
  }
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

/** Every reading slot the Data section holds after the roles are recorded, by consequence:
 *  which rows a screen keeps, then the roles, then codes. */
export function readingSlots(): ReadingSlot[] {
  const out: ReadingSlot[] = [];
  const screens = stepAsk("m2", "exclusions");
  const exposure = stepAsk("m3", "estimand");
  const fit = stepAsk("m5", "models");
  screens?.groups.forEach((g, i) => out.push(slotOf(g, screens.consumer, screens.exits, i)));
  exposure?.groups.forEach((g, i) => out.push(slotOf(g, exposure.consumer, fit?.exits ?? exposure.exits, i)));
  fit?.groups
    .filter((g) => g.kind !== "role")
    .forEach((g, i) => out.push(slotOf(g, fit.consumer, fit.exits, i)));
  return out;
}

/** The block confirm the engine offers once three readings were confirmed one at a time (the
 *  second ask of the drive): it lists exactly the readings it settles, each with its value. */
export function unlockedBlock() {
  const exit = INF.asks.models_1.exits[0]!;
  const items = (exit.decision as unknown as { items: { reading: string; column: string; value: string }[] }).items;
  return { label: exit.label, items };
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
