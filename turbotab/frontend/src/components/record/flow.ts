/**
 * Where each Router step stands in the Record's flow (pure; the Record and its tests share it).
 *
 * Answers always stay in place: nothing earlier is deleted. The first unanswered question, when it
 * waits on a stage, stands in place as a pending row. Every later unanswered, inapplicable or
 * stated step is listed under "Then": a reading is said when its turn comes, never ahead of the
 * question before it (a grain stated from a unique identifier waits for the purpose). A step the
 * user reopened is always in the flow, where its question is asked again.
 */
import type { InterviewStep, QuestionKey } from "../../api/m1-types";

export interface Flow {
  /** The first open or waiting step's index, or -1 when every step is settled. */
  firstAt: number;
  /** The first unanswered step, when it waits on a stage: a pending row in place. */
  pendingStep: InterviewStep | null;
  /** The steps rendered in the flow, in the Router's order (the pending one excluded). */
  inline: InterviewStep[];
  /** The steps listed under "Then". */
  next: InterviewStep[];
  /** Runs of adjacent inapplicable steps in the flow: the first holds the run, the rest null. */
  naRuns: Map<QuestionKey, InterviewStep[] | null>;
}

export function layoutFlow(
  interview: readonly InterviewStep[],
  reopened: Partial<Record<QuestionKey, boolean>> = {},
): Flow {
  const firstAt = interview.findIndex((st) => st.status === "open" || st.status === "waiting");
  const firstUnanswered = firstAt === -1 ? undefined : interview[firstAt];
  const pendingStep =
    firstUnanswered?.status === "waiting" &&
    firstUnanswered.waiting_on.every((w) => !interview.some((x) => x.key === w))
      ? firstUnanswered
      : null;
  const later = (st: InterviewStep, i: number) =>
    st !== pendingStep &&
    !reopened[st.key] &&
    (st.status === "waiting" ||
      ((st.status === "not_applicable" || st.status === "skipped") &&
        firstAt !== -1 &&
        i > firstAt));
  const inline = interview.filter((st, i) => !later(st, i) && st !== pendingStep);
  const next = interview.filter(later);
  const naRuns = new Map<QuestionKey, InterviewStep[] | null>();
  let run: InterviewStep[] | null = null;
  for (const st of interview) {
    const shown = st === pendingStep || inline.includes(st);
    if (!shown) continue;
    const na = st.status === "not_applicable" && !reopened[st.key];
    if (na && run) {
      run.push(st);
      naRuns.set(st.key, null);
    } else if (na) {
      run = [st];
      naRuns.set(st.key, run);
    } else run = null;
  }
  return { firstAt, pendingStep, inline, next, naRuns };
}
