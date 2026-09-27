/**
 * Propagate / stale veil. Content downstream of a changed decision is not deleted:
 * it is desaturated, veiled, tagged and made `inert` — visibly recoverable. Sections
 * veil in document order (`order` x a short stagger) so the user watches the edit's
 * blast radius draw itself, and un-veil as soon as their fresh result arrives.
 */
import type { CSSProperties, ReactNode } from "react";
import { DUR, useMotionPrefs } from "./prefs";
import styles from "./StaleVeil.module.css";

export type VeilState = "fresh" | "stale" | "recomputing";

interface Props {
  state: VeilState;
  /** Position in the downstream sweep (0 = nearest the change). */
  order?: number;
  children: ReactNode;
  className?: string;
  label?: string;
  testId?: string;
}

const TAG: Record<Exclude<VeilState, "fresh">, string> = {
  stale: "stale — an earlier answer changed",
  recomputing: "recomputing",
};

export function StaleVeil({ state, order = 0, children, className, label, testId }: Props) {
  const { reduced } = useMotionPrefs();
  const veiled = state !== "fresh";
  const delay = veiled && !reduced ? order * DUR.propagateStepMs : 0;
  const style = { "--veil-delay": `${delay}ms` } as CSSProperties;
  return (
    <div
      className={[styles.root, veiled ? styles.veiled : "", className ?? ""].join(" ")}
      data-veil={state}
      data-testid={testId}
      aria-label={label}
      role={label ? "group" : undefined}
      style={style}
    >
      <div className={styles.tagSlot} aria-live="polite">
        {veiled ? <span className={styles.tag}>{TAG[state]}</span> : null}
      </div>
      <div className={styles.body} inert={veiled}>
        {children}
      </div>
    </div>
  );
}

/** A stage's display state from its status and whether an (older) result is on screen. */
export function veilFor(
  status: { status: string; fresh: boolean; key: string | null } | undefined,
  result: { fresh: boolean; key: string | null; artifact: unknown } | undefined,
): VeilState {
  if (!result?.artifact) return "fresh"; // nothing on screen to veil
  if (!status) return "fresh";
  // A result is current only for the key the stage has now: one fetched fresh for an
  // earlier answer is not fresh for this one.
  if (status.status === "fresh" && result.fresh && result.key === status.key) return "fresh";
  if (status.status === "queued" || status.status === "running") return "recomputing";
  if (status.status === "fresh") return "recomputing"; // fresh result is on its way
  return "stale";
}
