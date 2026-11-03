/**
 * The way forward from a stage that will not restart by itself: one that failed
 * ("Try again") or whose work the user cancelled ("Recompute"). Renders nothing
 * for any other status, so it can sit beside every veil tag and pending line, and
 * nothing for a stage the engine refused in words (StageStatus.refusal): the refusal
 * is for the current answers, so running them again refuses again; its ways forward
 * are StageRefusal's, where the stage's failure is said.
 */
import { useRunStage } from "../api/queries";
import type { AnyStageName } from "../api/m1-types";
import type { StageStatus } from "../api/schema";
import { refusalOf } from "./StageRefusal";
import styles from "./StageRetry.module.css";

/** Whether a stage will not restart by itself: it failed (a refusal among its failures), or its
 *  work was stopped. */
export function needsRetry(status: StageStatus | undefined): boolean {
  return status !== undefined && (status.status === "error" || status.cancelled);
}

export function StageRetry({ pid, status }: { pid: string; status: StageStatus | undefined }) {
  const run = useRunStage(pid);
  if (!status || !needsRetry(status) || refusalOf(status)) return null;
  const failed = status.status === "error";
  return (
    <button
      type="button"
      className={styles.retry}
      disabled={run.isPending}
      onClick={() => run.mutate(status.stage as AnyStageName)}
      data-testid={`retry-${status.stage}`}
      title={
        failed
          ? "Runs this computation again for the current answers."
          : "You stopped this computation. Runs it again for the current answers."
      }
    >
      {failed ? "Try again" : "Recompute"}
    </button>
  );
}
