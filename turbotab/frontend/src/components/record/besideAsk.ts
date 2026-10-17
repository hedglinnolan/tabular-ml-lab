/**
 * A refusal said beside the ledger's ask card that asks its readings (finding 12 of the 2026-10-17
 * walkthrough: the readings shown once). The Record's answer to a press on an asked question
 * (AskFor) and the shell's stage, whose card is the one place a choice is recorded (Stage
 * `recordHere={false}`), both say it this way. And a refusal's ways forward never confirm a
 * column's role together with other readings (`oneRoleAtATime`), wherever it is said.
 */
import type { Refusal } from "../../api/schema";

type Exit = Refusal["error"]["exits"][number];

const confirms = (d: Record<string, unknown> | null) =>
  d?.kind === "confirm_reading" || d?.kind === "confirm_readings";
/** Where a refusal's own words stop and the ledger's question about the readings begins
 *  (core/readings.py `ask_text`; core/quest.py `asks` splits a refusal there too). */
const ASK_TEXT = /\s*Tell me about (?:this column|these columns): /;

/**
 * The refusal as said beside the ask card: that card lists each reading the refusal waits on, with
 * its evidence and its confirmation, so the answer keeps the consumer's own words, which say what
 * it waits for, up to the ledger's question, and the ways forward that are not those
 * confirmations. A refusal that offers no confirmation is said whole.
 */
export function besideAsk(refusal: Refusal): Refusal {
  const exits = refusal.error.exits.filter((x) => !confirms(x.decision));
  if (exits.length === refusal.error.exits.length) return refusal;
  const message = refusal.error.message;
  const cut = message.search(ASK_TEXT);
  return { error: { ...refusal.error, message: cut > 0 ? message.slice(0, cut) : message, exits } };
}

/**
 * Whether an exit confirms a column's role together with other readings in one press. A role the
 * ledger still asks was proposed below high confidence, which needs its own confirmation and never
 * rides along in a block (BLUEPRINT §14 rule 2; ruling R-b of 2026-10-17): no press anywhere
 * confirms it with others ("Confirm each of the 7 as shown"). A confirmation of one role alone is
 * its own, and a block of other readings (codes or amounts) is not one of these.
 */
export function confirmsRoleInBlock(exit: Exit): boolean {
  const d = exit.decision as { kind?: string; items?: { reading?: string }[] } | null;
  if (d?.kind !== "confirm_readings" || !d.items || d.items.length < 2) return false;
  return d.items.some((i) => i.reading === "role");
}

/** The refusal with no exit that confirms a role in a block (`confirmsRoleInBlock`). */
export function oneRoleAtATime(refusal: Refusal): Refusal {
  const exits = refusal.error.exits.filter((x) => !confirmsRoleInBlock(x));
  return exits.length === refusal.error.exits.length ? refusal : { error: { ...refusal.error, exits } };
}
