/**
 * The quest log's objective list: the methods section's sections in order (the kit's manuscript
 * sections, walk.ts), each with the questions a person answers there (its objectives) and how many
 * are answered. The sentences the engine states without a question are not objectives. Nothing here
 * is new copy, and nothing speaks the manuscript's register: an objective is named by its question
 * in the card's own plain words (as the map names its nodes), and the card keeps the kit's stage
 * label, so the four structures' cards read alike (FOUNDATION §2 and §6).
 */
import {
  FX,
  STEPS,
  STEP_BY_ID,
  frontier,
  isResult,
  plain,
  reachable,
  type SectionId,
  type WalkState,
} from "../calm-kit";

export type ObjectiveStatus = "done" | "open" | "waiting";

export interface Objective {
  /** The step. */
  id: string;
  /** Its question, without the backticks that mark data values. */
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
  objectives: Objective[];
  done: number;
  /** The card is on one of its objectives (for Results: on Table 2 or what mattered). */
  current: boolean;
}

export function questSections(s: WalkState): QuestSection[] {
  const openSection = isResult(s.open) ? "results" : (STEP_BY_ID[s.open]?.section ?? null);
  return FX.sections.map((sec) => {
    const objectives: Objective[] = STEPS.filter((x) => x.section === sec.id).map((x) => {
      const status: ObjectiveStatus =
        x.id in s.answers ? "done" : reachable(s, x.id) ? "open" : "waiting";
      return {
        id: x.id,
        name: plain(x.question),
        status,
        step: status === "waiting" ? null : x.id,
        current: x.id === s.open,
      };
    });
    return {
      id: sec.id,
      title: sec.title,
      objectives,
      done: objectives.filter((o) => o.status === "done").length,
      current: sec.id === openSection,
    };
  });
}

/** Where "Next objective" goes: the open question, when the card is somewhere else (null: it is on
 *  it, or the plan is locked and nothing is open). */
export function nextObjective(s: WalkState): string | null {
  if (s.locked) return null;
  const f = frontier(s);
  return f === s.open ? null : f;
}
