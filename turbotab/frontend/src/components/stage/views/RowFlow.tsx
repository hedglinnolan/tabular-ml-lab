/**
 * row_flow — participant flow, n at each step (CONSORT / STROBE): the rows half of modeling
 * decision provenance.
 *
 * The same component is the live pipeline's rows and an exclusion's preview, so previewing a rule
 * does not replace the picture: the step arrives inside it, the bars below shorten. Removed rows
 * are hatched — a pattern, not a hue, because "these rows leave" is not one of the palette's
 * claims. Counts never roll through in-between values: each is a real step's count.
 */
import { AnimatePresence, motion } from "motion/react";
import type { RowFlowView, RowStep } from "../../../api/m1-stage-types";
import type { CoachNote } from "../../../api/m2-stage-types";
import { useTransitions } from "../../../motion/prefs";
import c from "../coach/coach.module.css";
import { MAX_NOTES, notesFor } from "../coach/place";
import { fmtInt } from "../format";
import { Rich } from "../text";
import { localPos, noRepeats, type Track } from "../tracks";
import { usePlayerStore, usePlayerUi } from "../usePlayer";
import { useSize } from "./geometry";
import s from "./views.module.css";

interface Props {
  steps: RowStep[];
  compact?: boolean;
  /** Draw changed steps in the "with this choice" hue. */
  preview?: boolean;
  /** Step keys the choice acts on. */
  emphasis?: string[];
  /** The open question's place in the flow: after this step key. */
  openAfter?: string | null;
  openLabel?: string;
  /** Say why each step drops rows (the full row flow). */
  reasons?: boolean;
  /** The coach's notes (M2_CONTRACT §6): one on a step sits under that step, in amber. */
  coach?: CoachNote[];
}

/**
 * A step that folds rows into a partner row (one row per unit): the same count as rows that leave,
 * but a different claim, so it is drawn as a fold and said as "folded in", never "−300".
 */
export function folds(step: RowStep): boolean {
  return step.key === "combined" && step.dropped > 0;
}

const HOLD = new Set(["holdout"]);

export function RowFlow({ steps, compact = false, preview, emphasis, openAfter, openLabel, reasons, coach }: Props) {
  const t = useTransitions();
  const n0 = Math.max(1, steps[0]?.n ?? 1);
  const picked = new Set(emphasis ?? []);
  // The caller decides which notes a flow this size carries (`notesFor`); at most two.
  const notes = (coach ?? []).slice(0, MAX_NOTES);
  const onStep = (key: string) => notes.filter((n) => n.anchor.kind === "step" && n.anchor.ref === key);
  const loose = notes.filter((n) => n.anchor.kind !== "step" || !steps.some((st) => st.key === n.anchor.ref));
  const rows: ({ kind: "step"; step: RowStep } | { kind: "open" })[] = [];
  for (const step of steps) {
    rows.push({ kind: "step", step });
    if (openAfter && step.key === openAfter) rows.push({ kind: "open" });
  }
  return (
    <div className={compact ? s.flowCompact : s.flow} data-view="row_flow">
      {loose.map((n) => (
        <p key={n.text} className={c.inline} data-testid="coach-note" data-purpose="coach_note" data-anchor={n.anchor.kind}>
          <Rich text={n.text} />
        </p>
      ))}
      <AnimatePresence initial={false}>
        {rows.map((r) =>
          r.kind === "open" ? (
            <motion.div
              key="open"
              className={s.flowOpen}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={t.arrive}
            >
              <span className={s.flowOpenText}>{openLabel ?? "the open question acts here"}</span>
            </motion.div>
          ) : (
            <motion.div
              key={r.step.key}
              className={s.flowRow}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={t.arrive}
              data-step={r.step.key}
            >
              <div className={s.flowLine}>
                <span className={s.flowLabel}>
                  <Rich text={noRepeats(r.step.label)} />
                </span>
                <span className={s.flowTrack}>
                  <motion.span
                    className={
                      HOLD.has(r.step.key)
                        ? s.flowHold
                        : preview && picked.has(r.step.key)
                          ? s.flowKeptNow
                          : s.flowKept
                    }
                    initial={false}
                    animate={{ width: `${(100 * r.step.n) / n0}%` }}
                    transition={t.arrive}
                  />
                  <motion.span
                    className={folds(r.step) ? s.flowFolded : s.flowGone}
                    initial={false}
                    animate={{ width: `${(100 * r.step.dropped) / n0}%` }}
                    transition={t.arrive}
                  />
                </span>
                <span className={s.flowCount}>{fmtInt(r.step.n)}</span>
                <span className={s.flowDrop}>
                  {r.step.dropped ? (folds(r.step) ? `${fmtInt(r.step.dropped)} folded in` : `−${fmtInt(r.step.dropped)}`) : ""}
                </span>
              </div>
              {reasons && r.step.dropped && r.step.reason && !compact ? (
                <div className={s.flowDetail}>
                  <Rich text={r.step.reason} />
                </div>
              ) : null}
              {onStep(r.step.key).map((n) => (
                <p key={n.text} className={c.inline} data-testid="coach-note" data-purpose="coach_note" data-anchor="step">
                  <Rich text={n.text} />
                </p>
              ))}
            </motion.div>
          ),
        )}
      </AnimatePresence>
    </div>
  );
}

/** The row flow at whichever real state the player shows. */
export function RowFlowTrack({
  track,
  globalLast,
  compact,
}: {
  track: Track<RowFlowView>;
  globalLast: number;
  compact?: boolean;
}) {
  const ui = usePlayerUi(usePlayerStore());
  const [ref, { w }] = useSize<HTMLDivElement>();
  const localLast = track.states.length - 1;
  const shown = Math.min(localLast, Math.round(localPos(ui.nearest, globalLast, localLast)));
  return (
    <div data-state={shown} ref={ref}>
      <RowFlow
        steps={track.states[shown]!.steps}
        compact={compact}
        preview={shown > 0}
        emphasis={track.view.emphasis}
        reasons={!compact}
        coach={notesFor(track.view.coach, !!compact, w)}
      />
    </div>
  );
}
