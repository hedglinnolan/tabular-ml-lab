/**
 * The quest log's objective list, derived from the kit's manuscript (walk.ts): the methods
 * section's guideline sections in order (STROBE 6, 7, 8, 12, 13–17), each with the slots a person
 * answers (its objectives) and how many are recorded. The sentences the engine states without a
 * question are not objectives. Nothing here is new copy: an objective's name is the manuscript's
 * own head, and where a head repeats inside a section, the columns its question names.
 */
import {
  FX,
  STEPS,
  STEP_BY_ID,
  frontier,
  isResult,
  plain,
  type Section,
  type SectionId,
  type WalkState,
} from "../calm-kit";

export type ObjectiveStatus = "done" | "open" | "waiting";

export interface Objective {
  /** The manuscript slot. */
  id: string;
  name: string;
  status: ObjectiveStatus;
  /** The step a click opens (null while it waits for an earlier answer). */
  step: string | null;
  /** The card is on this objective now. */
  current: boolean;
}

export interface QuestSection {
  id: SectionId;
  title: string;
  /** The guideline item ("STROBE 7"). */
  item: string;
  objectives: Objective[];
  done: number;
  /** The card is on one of its objectives (for Results: on Table 2 or what mattered). */
  current: boolean;
}

/** The columns a question is about, in its own words: "age and gender", "bp_di". */
function subjectOf(question: string): string | null {
  const relate = /^How (?:do|does) (.+?) relate\b/.exec(question);
  if (relate) return plain(relate[1]);
  const first = /`([^`]+)`/.exec(question);
  return first ? first[1]! : null;
}

export function questSections(sections: Section[], open: string): QuestSection[] {
  const openSlot = STEP_BY_ID[open]?.slot ?? null;
  const openSection = isResult(open) ? "results" : (STEP_BY_ID[open]?.section ?? null);
  return sections.map((sec) => {
    const asked = sec.entries.filter((e) => e.kind !== "stated" && STEP_BY_ID[e.step ?? ""]);
    const repeats = (head: string) => asked.filter((e) => e.head === head).length > 1;
    const objectives: Objective[] = asked.map((e) => {
      const subject = repeats(e.head) ? subjectOf(STEP_BY_ID[e.step!]!.question) : null;
      const status: ObjectiveStatus =
        e.kind === "recorded" ? "done" : e.kind === "blank" ? "open" : "waiting";
      return {
        id: e.id,
        name: subject ? `${e.head}: ${subject}` : e.head,
        status,
        step: status === "waiting" ? null : e.step,
        current: e.id === openSlot,
      };
    });
    return {
      id: sec.id,
      title: sec.title,
      item: sec.item,
      objectives,
      done: objectives.filter((o) => o.status === "done").length,
      current: sec.id === openSection,
    };
  });
}

/** Where "Next objective" goes: the open slot, when the card is somewhere else (null: it is on it,
 *  or the plan is locked and nothing is open). */
export function nextObjective(s: WalkState): string | null {
  if (s.locked) return null;
  const f = frontier(s);
  return f === s.open ? null : f;
}

/** The card's kicker: the guideline section and the objective ("Variables · Adjustment set"), with
 *  the step when one objective takes several questions. Null on a result moment (the kit's label). */
export function kickerOf(open: string): string | null {
  const step = STEP_BY_ID[open];
  if (!step) return null;
  const section = FX.sections.find((x) => x.id === step.section)!;
  const members = STEPS.filter((x) => x.slot === step.slot);
  const part =
    members.length > 1 ? ` · step ${members.indexOf(step) + 1} of ${members.length}` : "";
  return `${section.title} · ${step.head}${part}`;
}
