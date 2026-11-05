/**
 * The card a way forward opens (G12; FOUNDATION §9 item 1: each item's exit opens the stage that
 * owns it). A way forward that names no decision names its card where it can: `target`, the quest
 * stage and the quest line whose card it is, by the keys GET /quest serves (core/exit_target.py);
 * a null key opens the stage on its open line. One that opens nothing (keeping the answers, waiting
 * for work under way) is served with no target. The triage of the open noticings is no line of the
 * quest log, so a target that opens it is the shell's own (`triageTarget`).
 *
 * The quest shell opens cards (QuestScreen's `OpenCard` provider, its `openStage`); where nothing
 * provides it (Classic, the labs) a way forward is its caller's press, as it was.
 */
import { createContext, useContext } from "react";

export interface ExitTarget {
  stage: string;
  key: string | null;
}

/** The target a served way forward names, or null: none, or none in the served shape. Read from any
 *  way forward, typed (`Exit`) or as a stage's status serves it (`Refused.exits`, plain objects). */
export function targetOf(exit: unknown): ExitTarget | null {
  const target = (exit as { target?: unknown } | null)?.target;
  if (typeof target !== "object" || target === null) return null;
  const { stage, key } = target as { stage?: unknown; key?: unknown };
  if (typeof stage !== "string" || (key != null && typeof key !== "string")) return null;
  return { stage, key: key ?? null };
}

/** The key a target that opens the triage of the open noticings names it by (`triageTarget`). */
const TRIAGE = "triage";

/** The triage of the open noticings (GET /triage; FOUNDATION §7) in `stage`, its gate's, where the
 *  shell draws it (under Estimate at Models' end, StageCard; under Predict in Results' column), on
 *  the noticing `noticing` names (its id, a triage item's), else as a whole. The quest log serves
 *  the triage as no line, so its key is the shell's: `triage`, and the noticing after a colon. */
export function triageTarget(stage: string, noticing: string | null = null): ExitTarget {
  return { stage, key: noticing === null ? TRIAGE : `${TRIAGE}:${noticing}` };
}

/** The triage a request's line names (`triageTarget`), with the noticing it brings into view (null:
 *  the triage as a whole); null where the line names no triage. */
export function triageOf(line: string | null): { noticing: string | null } | null {
  if (line === TRIAGE) return { noticing: null };
  return line?.startsWith(`${TRIAGE}:`) ? { noticing: line.slice(TRIAGE.length + 1) } : null;
}

/** How a target is named on its press (`data-target`): the stage, and its line after a colon. */
export function targetName(target: ExitTarget): string {
  return target.key ? `${target.stage}:${target.key}` : target.stage;
}

/** Opens a target's card: the shell's, or null where nothing opens cards. */
export const OpenCard = createContext<((target: ExitTarget) => void) | null>(null);

/** The shell's card opener for a way forward: `open(exit)` where the way forward names no decision
 *  and a card, and the shell opens cards; else null. */
export function useOpenTarget(): ((exit: unknown) => (() => void) | null) {
  const open = useContext(OpenCard);
  return (exit) => {
    if (!open || (exit as { decision?: unknown } | null)?.decision) return null;
    const target = targetOf(exit);
    return target ? () => open(target) : null;
  };
}
