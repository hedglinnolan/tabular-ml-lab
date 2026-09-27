/**
 * The way forward from a stage that will not restart by itself: one that failed
 * ("Try again") or whose work the user cancelled ("Recompute"). Renders nothing
 * for any other status, so it can sit beside every veil tag and pending line.
 */
import { useRunStage } from "../api/queries";
import type { StageName, StageStatus } from "../api/schema";
import styles from "./StageRetry.module.css";

export function needsRetry(status: StageStatus | undefined): boolean {
  return status !== undefined && (status.status === "error" || status.cancelled);
}

export function StageRetry({ pid, status }: { pid: string; status: StageStatus | undefined }) {
  const run = useRunStage(pid);
  if (!status || !needsRetry(status)) return null;
  const failed = status.status === "error";
  return (
    <button
      type="button"
      className={styles.retry}
      disabled={run.isPending}
      onClick={() => run.mutate(status.stage as StageName)}
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
