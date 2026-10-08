/**
 * The interview Router, mirrored for the mock server from turbotab/core/interview.py
 * (M1_CONTRACT §1). The real client never runs this: it renders what the server says.
 */
import type { InterviewStep, QuestionKey, Role } from "../api/m1-types";
import { QUESTION_KEYS } from "../api/m1-types";
import type { DecisionRecord, ProjectState, StageStatus, TargetInfoArtifact } from "../api/schema";

const NEEDS: Record<QuestionKey, string[]> = {
  lens: ["ingest"],
  orientation: ["oriented"],
  target: ["ingest"],
  event: ["target_info"],
  task: ["target_info"],
  follow_up: ["target_info"],
  purpose: [],
  grain: ["structure"],
  repeat_kind: ["structure"],
  unit: [],
  aggregation: ["structure"],
  temporal: ["structure"],
  roles: ["roles"],
  clusters: ["roles"],
  survey: ["proposals"],
  estimand: ["proposals"],
  adjustment: ["proposals"],
  time_varying: ["time_varying"],
  exclusions: ["proposals"],
  missing: [],
  split: [],
  energy_adjustment: ["proposals"],
  form: ["forms"],
  modification: [],
  causal: ["causal_design"],
  models: ["shelf"],
  substitution: ["fit"],
  open_seal: ["fit"],
};
const MUST_BE_FRESH: Partial<Record<QuestionKey, string>> = { substitution: "fit", open_seal: "fit" };

const SLOT_OF: Record<string, QuestionKey> = {
  set_lens: "lens",
  set_target: "target",
  set_task: "task",
  set_purpose: "purpose",
  set_roles: "roles",
  set_survey: "survey",
  set_exclusions: "exclusions",
  set_missing: "missing",
  set_split: "split",
  set_energy_adjustment: "energy_adjustment",
  select_models: "models",
  set_substitution: "substitution",
  set_orientation: "orientation",
  set_event: "event",
  set_grain: "grain",
  set_repeat_kind: "repeat_kind",
  set_unit: "unit",
  set_aggregation: "aggregation",
  set_temporal: "temporal",
  open_seal: "open_seal",
};

/**
 * The opening sequence's gates (M2_CONTRACT §1), mirrored. Orientation fires only on the mock's
 * features-in-rows table (`featureMajor`); the repeats chain follows the grain answer.
 */
function sequenceGate(
  key: QuestionKey,
  state: ProjectState,
  targetInfo: TargetInfoArtifact | null,
  featureMajor = false,
): string | null {
  const repeated = state.grain === null ? null : state.grain.grain === "repeated";
  switch (key) {
    case "orientation":
      return state.lens === null
        ? null
        : state.lens.some((l) => l === "metabolomics" || l === "genomics")
          ? featureMajor
            ? null
            : "The table's shape reads as one row per sample."
          : "No assay lens is on, and other tables are not exported turned around.";
    case "event": {
      const task =
        state.task ?? (targetInfo && targetInfo.column === state.target ? targetInfo.task : null);
      return task && task !== "binary"
        ? `The outcome is read as ${task}, so there is no event level to choose.`
        : null;
    }
    case "repeat_kind":
      return repeated === false
        ? "Each unit appears once, so there are no repeats to tell apart."
        : null;
    case "unit":
      return repeated === false ? "Each unit appears once, so each row already is one." : null;
    case "aggregation":
      if (repeated === false) return "Each unit appears once, so there is nothing to combine.";
      return repeated && state.unit === "row"
        ? "Records stay as they are, so nothing is combined."
        : null;
    case "temporal":
      if (repeated === false) return "Each unit appears once, so no row comes later than another.";
      if (repeated && state.repeat_kind?.repeat_kind === "repeats")
        return "The rows are repeats of one measurement, not different time points.";
      return repeated && state.unit === "unit"
        ? "Each unit's rows are combined, so no time points stay as rows."
        : null;
    case "survey":
      // turbotab/core/survey.py: asked under inference when a column reads as a survey weight;
      // the mock's tables carry none.
      if (state.purpose === "prediction")
        return "Under prediction the scores describe the rows they were computed on; they are not weighted to a population.";
      if (state.purpose === null || state.roles === null) return null;
      return Object.keys(state.roles).some((c) => /^WT(DRD1|DR2D|MEC2YR|INT2YR)$/i.test(c))
        ? null
        : "No column reads as a survey weight, so there is no surveyed population to weight to.";
    case "form":
      // turbotab/core/methods/exposure_form.py: stated under prediction; the mock never asks it.
      return "Each predictor enters as each family takes it; a spline can be declared for any.";
    case "modification":
      // turbotab/core/methods/interaction.py: stated until a modifier is declared.
      return "No effect modifier or second exposure is declared.";
    case "causal":
      // turbotab/core/causal.py: the causal lane is never offered under prediction.
      return state.purpose === "prediction"
        ? "Under prediction no coefficient is read as an effect, so no causal estimate is offered."
        : null;
    case "follow_up": {
      // turbotab/core/estimand.py follow_up_gate: only an outcome followed over time.
      const task =
        state.task ?? (targetInfo && targetInfo.column === state.target ? targetInfo.task : null);
      return task && task !== "binary" && task !== "time_to_event"
        ? `The outcome is read as ${task.replace(/_/g, " ")}, so no event is followed over time.`
        : null;
    }
    case "estimand":
    case "adjustment":
      // estimand.py _purpose_gate
      return state.purpose === "prediction"
        ? "Under prediction no coefficient is read as an effect, so no exposure, effect or adjustment set is declared."
        : null;
    case "time_varying":
      // time_varying.py lane_gate
      if (state.purpose === "prediction")
        return "Under prediction no coefficient is read as an effect, so no exposure is followed through time.";
      return repeated === false || (repeated && state.unit === "unit")
        ? "The rows are not a unit's time points, so no exposure changes over time."
        : null;
    default:
      return null;
  }
}

/** The questions the Router states rather than asks (each one's "Not asked:" reason), mirrored:
 *  the follow-up under a lens whose yes/no outcome is a status at sampling, the grouping when no
 *  column reads as one, and the causal lane, one step away under inference. */
function skipGate(
  key: QuestionKey,
  state: ProjectState,
  targetInfo: TargetInfoArtifact | null,
): string | null {
  switch (key) {
    case "follow_up": {
      const followed = !state.lens || state.lens.some((l) => l === "clinical" || l === "dietary");
      const task =
        state.task ?? (targetInfo && targetInfo.column === state.target ? targetInfo.task : null);
      if (followed || task === "time_to_event" || !targetInfo || targetInfo.column !== state.target)
        return null;
      return (targetInfo.follow_up ?? []).length
        ? null
        : "no numeric column reads as a follow-up time, and under this lens a yes/no outcome is a status at sampling, so it is read as counted over one period for everyone.";
    }
    case "clusters":
      if (state.roles === null) return null;
      return Object.values(state.roles).includes("cluster")
        ? null
        : "no column reads as a site, centre, household or batch that groups the participants.";
    case "causal":
      return state.purpose === "inference"
        ? "the primary model estimates the declared effect; the causal estimators are one step away."
        : null;
    default:
      return null;
  }
}

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

export function liveWriters(
  records: DecisionRecord[],
  state: ProjectState,
): Map<QuestionKey, string> {
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
  opts: { featureMajor?: boolean } = {},
): InterviewStep[] {
  const pending = pendingStages(stages, deps);
  const writers = liveWriters(records, state);
  const roles = state.roles as Record<string, Role> | null;
  const steps: Omit<InterviewStep, "deferred_findings">[] = [];
  let first: QuestionKey | null = null;
  for (const key of QUESTION_KEYS) {
    // open_seal writes the seal_opened slot (turbotab/core/interview.py SLOT_OF); the follow-up
    // is answered by a time to event's follow-up or a yes/no outcome's "same for everyone".
    // FORM: the form and modifier questions read their own slots (turbotab/core/interview.py)
    const value =
      key === "open_seal"
        ? state.seal_opened
        : key === "follow_up"
          ? (state.follow_up ?? state.censoring)
          : key === "form"
            ? state.exposure_forms
            : key === "modification"
              ? state.modifications
              : state[key];
    const decision_id = writers.get(key) ?? null;
    if (key === "orientation" && value !== null) {
      steps.push({ key, status: "answered", decision_id, reason: null, waiting_on: [], followup: null, ask: null });
      continue;
    }
    let na: string | null = sequenceGate(key, state, targetInfo, opts.featureMajor);
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
        followup: null,
        ask: null,
      });
      continue;
    }
    if (value !== null && value !== undefined) {
      steps.push({ key, status: "answered", decision_id, reason: null, waiting_on: [], followup: null, ask: null });
      continue;
    }
    const stated = skipGate(key, state, targetInfo);
    if (stated) {
      steps.push({
        key,
        status: "skipped",
        reason: stated,
        decision_id: null,
        waiting_on: [],
        followup: null,
        ask: null,
      });
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
        followup: null,
        ask: null,
      });
      continue;
    }
    const own = NEEDS[key].filter((s) => pending.has(s));
    const fresh = MUST_BE_FRESH[key];
    const held = fresh ? stages[fresh] : undefined;
    // A fit that failed or was stopped holds nothing back: the step opens and says why.
    if (
      fresh &&
      held?.status !== "fresh" &&
      held?.status !== "error" &&
      !held?.cancelled &&
      !own.includes(fresh)
    )
      own.push(fresh);
    if (first === null) {
      first = key;
      steps.push({
        key,
        status: own.length ? "waiting" : "open",
        decision_id: null,
        reason: null,
        waiting_on: own,
        followup: null,
        ask: null,
      });
    } else {
      steps.push({
        key,
        status: "waiting",
        decision_id: null,
        reason: null,
        waiting_on: [first, ...own],
        followup: null,
        ask: null,
      });
    }
  }
  return steps.map((s) => ({ ...s, deferred_findings: [] }));
}
