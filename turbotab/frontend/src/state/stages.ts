/** A stage's latest result for a project, fetched once the stage has started. */
import { useStageResult } from "../api/queries";
import type { AnyStageName } from "../api/m1-types";
import type { ProjectView } from "../api/schema";

export function useStage<S extends AnyStageName>(pid: string, view: ProjectView, name: S) {
  const status = view.stages[name];
  // Idle: never computed. Blocked: waits on an answer; an older result, if any, stays cached.
  const enabled = status !== undefined && status.status !== "idle" && status.status !== "blocked";
  return useStageResult(pid, name, enabled).data;
}
