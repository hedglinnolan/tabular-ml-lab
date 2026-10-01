/**
 * The interview Router, mirrored for the mock server from turbotab/core/interview.py
 * (M1_CONTRACT §1). The real client never runs this: it renders what the server says.
 */
import type { InterviewStep, QuestionKey, Role } from "../api/m1-types";
import { QUESTION_KEYS } from "../api/m1-types";
import type { DecisionRecord, ProjectState, StageStatus, TargetInfoArtifact } from "../api/schema";

const NEEDS: Record<QuestionKey, string[]> = {
  lens: ["ingest"],
  target: ["ingest"],
  task: ["target_info"],
  purpose: [],
  roles: ["roles"],
  exclusions: ["proposals"],
  missing: [],
  split: [],
  energy_adjustment: ["proposals"],
  models: ["shelf"],
  substitution: ["fit"],
};
const MUST_BE_FRESH: Partial<Record<QuestionKey, string>> = { substitution: "fit" };

const SLOT_OF: Record<string, QuestionKey> = {
  set_lens: "lens",
  set_target: "target",
  set_task: "task",
  set_purpose: "purpose",
  set_roles: "roles",
  set_exclusions: "exclusions",
  set_missing: "missing",
  set_split: "split",
  set_energy_adjustment: "energy_adjustment",
  select_models: "models",
  set_substitution: "substitution",
};

/** Stages whose work for the current answers is under way or about to start. */
export function pendingStages(
  stages: Record<string, StageStatus>,
  deps: Record<string, string[]>,
): Set<string> {
  const memo = new Map<string, boolean>();
  const pending = (name: string): boolean => {
    if (memo.has(name)) return memo.get(name)!;
    memo.set(name, false);
    const st = stages[name];
    // The mock announces "stale" for a moment before it queues the recompute (so a veil can be
    // seen); the real engine queues at once. A stale stage that was not stopped is about to run.
    const result =
      st?.status === "queued" ||
      st?.status === "running" ||
      (st?.status === "stale" && !st.cancelled) ||
      (st?.status === "idle" && !st.cancelled && (deps[name] ?? []).some(pending));
    memo.set(name, result);
    return result;
  };
  return new Set(Object.keys(stages).filter(pending));
}

function liveWriters(records: DecisionRecord[], state: ProjectState): Map<QuestionKey, string> {
  const reverted = new Set(
    records.flatMap((r) => (r.decision.kind === "revert" ? [r.decision.decision_id] : [])),
  );
  const out = new Map<QuestionKey, string>();
  for (const r of [...records].sort((a, b) => a.seq - b.seq)) {
    const d = r.decision;
    if (reverted.has(r.id) || d.kind === "revert") continue;
    const slot = SLOT_OF[d.kind];
    if (!slot) continue;
    if (d.kind === "set_task" && (d.column !== state.target || d.task !== state.task)) continue;
    out.set(slot, r.id);
  }
  return out;
}

export function route(
  state: ProjectState,
  stages: Record<string, StageStatus>,
  targetInfo: TargetInfoArtifact | null,
  records: DecisionRecord[],
  deps: Record<string, string[]>,
  bearing: (column: string) => boolean,
): InterviewStep[] {
  const pending = pendingStages(stages, deps);
  const writers = liveWriters(records, state);
  const roles = state.roles as Record<string, Role> | null;
  const steps: InterviewStep[] = [];
  let first: QuestionKey | null = null;
  for (const key of QUESTION_KEYS) {
    const value = state[key];
    const decision_id = writers.get(key) ?? null;
    let na: string | null = null;
    if (key === "energy_adjustment") {
      if (state.lens && !state.lens.includes("dietary"))
        na = "The dietary lens is off, so energy adjustment does not apply.";
      else if (roles && !Object.values(roles).includes("energy"))
        na = "No column has the energy role, so there is no total energy to adjust against.";
      else if (roles && !Object.entries(roles).some(([c, r]) => r === "exposure" && bearing(c)))
        na = "No exposure is a nutrient that carries energy, so there is nothing to adjust.";
    } else if (key === "substitution" && roles) {
      const n = Object.entries(roles).filter(([c, r]) => r === "exposure" && bearing(c)).length;
      if (n < 2)
        na =
          n === 1
            ? "Only 1 exposure carries energy; a substitution swaps kcal between two."
            : "No exposure carries energy; a substitution swaps kcal between two.";
    }
    if (na) {
      steps.push({
        key,
        status: "not_applicable",
        reason: na,
        decision_id: value !== null ? decision_id : null,
        waiting_on: [],
      });
      continue;
    }
    if (value !== null && value !== undefined) {
      steps.push({ key, status: "answered", decision_id, reason: null, waiting_on: [] });
      continue;
    }
    if (
      key === "task" &&
      state.target !== null &&
      targetInfo &&
      targetInfo.column === state.target &&
      targetInfo.confidence === "high"
    ) {
      steps.push({
        key,
        status: "skipped",
        reason: targetInfo.reason,
        decision_id: null,
        waiting_on: [],
      });
      continue;
    }
    const own = NEEDS[key].filter((s) => pending.has(s));
    const fresh = MUST_BE_FRESH[key];
    if (fresh && stages[fresh]?.status !== "fresh" && !own.includes(fresh)) own.push(fresh);
    if (first === null) {
      first = key;
      steps.push({
        key,
        status: own.length ? "waiting" : "open",
        decision_id: null,
        reason: null,
        waiting_on: own,
      });
    } else {
      steps.push({
        key,
        status: "waiting",
        decision_id: null,
        reason: null,
        waiting_on: [first, ...own],
      });
    }
  }
  return steps;
}
