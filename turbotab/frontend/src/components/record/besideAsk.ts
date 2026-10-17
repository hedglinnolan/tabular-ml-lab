/**
 * A refusal said beside the ledger's ask card that asks its readings (finding 12 of the 2026-10-17
 * walkthrough: the readings shown once). The Record's answer to a press on an asked question
 * (AskFor) and the shell's stage, whose card is the one place a choice is recorded (Stage
 * `recordHere={false}`), both say it this way.
 */
import type { Refusal } from "../../api/schema";

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
