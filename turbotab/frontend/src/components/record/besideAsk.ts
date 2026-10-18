/**
 * A refusal said beside the ledger's ask card that asks its readings (finding 12 of the 2026-10-17
 * walkthrough: the readings shown once). The Record's answer to a press on an asked question
 * (AskFor) and the shell's stage, whose card is the one place a choice is recorded (Stage
 * `recordHere={false}`), both say it this way. And a refusal's ways forward never confirm a
 * column's role together with other readings (`oneRoleAtATime`), wherever it is said, but in one
 * group's press (ruling R-g).
 */
import type { Decision, ProjectView, Refusal } from "../../api/schema";

type Exit = Refusal["error"]["exits"][number];

/**
 * The decision kinds each question that can carry the ledger's ask card records: core/ask.py
 * `CONSUMERS` (with exposure_form's "form"), each kind's question as core/sequence.py `question_of`
 * reads it. The shell's contract fixture holds the engine's own (`asked_by`), which besideAsk.test
 * pins this to.
 */
export const ASKED_BY: Readonly<Record<string, readonly string[]>> = {
  aggregation: ["set_aggregation"],
  survey: ["set_survey"],
  exclusions: ["set_exclusions"],
  estimand: ["set_estimand"],
  adjustment: ["set_adjustment"],
  energy_adjustment: ["set_energy_adjustment"],
  form: ["set_exposure_form", "set_forms"],
  models: ["select_models"],
};

/**
 * Whether the card that records `decision` asks, beside it, the readings a refusal of it waits on:
 * its question is the open step with the ledger's ask card, where AskFor draws that card and says
 * such a refusal `besideAsk` (W1-W round 1, P4). Any other card (a declaration's, or a question
 * that is not the open step with an ask) asks no readings.
 */
export function askedBeside(view: Pick<ProjectView, "interview">, decision: Decision): boolean {
  const question = Object.keys(ASKED_BY).find((q) => ASKED_BY[q]!.includes(decision.kind));
  const step = question ? view.interview.find((s) => s.key === question) : undefined;
  return step?.status === "open" && !!step.ask;
}

const confirms = (d: Record<string, unknown> | null) =>
  d?.kind === "confirm_reading" || d?.kind === "confirm_readings";
/** Where a refusal's own words stop and the ledger's question about the readings begins
 *  (core/readings.py `ask_text`; core/quest.py `asks` splits a refusal there too). */
const ASK_TEXT = /\s*Tell me about (?:this column|these columns): /;

/** The way through that leads to the readings the ask card lists, in place of their confirmations
 *  (the re-walk's N4): AskFor brings that card into view and focuses it. */
export const READINGS_ABOVE: Exit = { label: "Answer the readings above", decision: null };

/**
 * The refusal as said beside the ask card: that card lists each reading the refusal waits on, with
 * its evidence and its confirmation, so the answer keeps the consumer's own words, which say what
 * it waits for, up to the ledger's question, and the ways forward that are not those
 * confirmations, led by one to the readings the card lists (`READINGS_ABOVE`). A refusal that
 * offers no confirmation is said whole.
 */
export function besideAsk(refusal: Refusal): Refusal {
  const exits = refusal.error.exits.filter((x) => !confirms(x.decision));
  if (exits.length === refusal.error.exits.length) return refusal;
  const message = refusal.error.message;
  const cut = message.search(ASK_TEXT);
  return {
    error: { ...refusal.error, message: cut > 0 ? message.slice(0, cut) : message, exits: [READINGS_ABOVE, ...exits] },
  };
}

/**
 * Whether an exit confirms a column's role together with other readings in one press. A role the
 * ledger still asks was proposed below high confidence, which never rides along in a block
 * (BLUEPRINT §14 rule 2): no press anywhere confirms it with other readings ("Confirm each of the 7
 * as shown"). A confirmation of one role alone is its own, a block of other readings (codes or
 * amounts) is not one of these, and neither is one group's press (ruling R-g of 2026-10-17): a block
 * of roles alone, each as one proposed role, which the engine offers only for the columns its roles
 * stage proposed alike and accepts for no other.
 */
export function confirmsRoleInBlock(exit: Exit): boolean {
  const d = exit.decision as { kind?: string; items?: { reading?: string; value?: string }[] } | null;
  if (d?.kind !== "confirm_readings" || !d.items || d.items.length < 2) return false;
  if (!d.items.some((i) => i.reading === "role")) return false;
  const oneGroup = d.items.every((i) => i.reading === "role" && i.value === d.items![0]!.value);
  return !oneGroup;
}

/** The refusal with no exit that confirms a role in a block (`confirmsRoleInBlock`). */
export function oneRoleAtATime(refusal: Refusal): Refusal {
  const exits = refusal.error.exits.filter((x) => !confirmsRoleInBlock(x));
  return exits.length === refusal.error.exits.length ? refusal : { error: { ...refusal.error, exits } };
}
