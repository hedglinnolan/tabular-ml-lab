/**
 * row_flow — participant flow, n at each step (CONSORT / STROBE).
 *
 * The same component is the live pipeline's Rows section and an exclusion's preview,
 * so previewing a rule does not replace the Rows picture: the exclusion step arrives
 * inside it, the bars below shorten, and the counts tween. Removed rows are hatched —
 * a pattern, not a hue, because "these rows leave" is not one of the palette's claims.
 */
import { AnimatePresence, motion } from "motion/react";
import { fmtInt } from "../format";
import type { RowStep } from "../types";
import s from "./views.module.css";
import { Tween, useStageTransitions } from "../motion";

interface Props {
  steps: RowStep[];
  compact?: boolean;
  split?: { train: number; holdout: number };
  openAfter?: string;
  openLabel?: string;
  byLevel?: Record<string, { below: number; above: number }>;
  /** Draw kept bars in the "after" hue (a preview) rather than the neutral one. */
  preview?: boolean;
}

const LEVEL: Record<string, string> = { female: "women", male: "men" };

function levelLine(byLevel: Props["byLevel"]): string | null {
  if (!byLevel) return null;
  const parts = Object.entries(byLevel)
    .filter(([, v]) => v.below + v.above > 0)
    .map(([k, v]) =>
      k === "all"
        ? `${fmtInt(v.below)} below · ${fmtInt(v.above)} above`
        : `${LEVEL[k] ?? k} ${fmtInt(v.below)} below · ${fmtInt(v.above)} above`,
    );
  return parts.length ? parts.join("   ") : null;
}

export function RowFlow({ steps, compact = false, split, openAfter, openLabel, byLevel, preview }: Props) {
  const t = useStageTransitions();
  const n0 = steps[0]?.n ?? 1;
  const detail = levelLine(byLevel);
  const rows: ({ kind: "step"; step: RowStep } | { kind: "open" })[] = [];
  for (const step of steps) {
    rows.push({ kind: "step", step });
    if (openAfter && step.key === openAfter) rows.push({ kind: "open" });
  }
  return (
    <div className={compact ? s.flowCompact : s.flow} data-view="row_flow">
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
              <span className={s.flowOpenText}>{openLabel ?? "this question"}</span>
            </motion.div>
          ) : (
            <motion.div
              key={r.step.key}
              className={s.flowRow}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={t.arrive}
            >
              <div className={s.flowLine}>
                <span className={s.flowLabel}>{r.step.label}</span>
                <span className={s.flowTrack}>
                  <motion.span
                    className={preview && r.step.dropped ? s.flowKeptNow : preview ? s.flowKeptAfter : s.flowKept}
                    initial={false}
                    animate={{ width: `${(100 * r.step.n) / n0}%` }}
                    transition={t.arrive}
                  />
                  <motion.span
                    className={s.flowGone}
                    initial={false}
                    animate={{ width: `${(100 * r.step.dropped) / n0}%` }}
                    transition={t.arrive}
                  />
                </span>
                <span className={s.flowCount}>
                  <Tween value={r.step.n} format={fmtInt} />
                </span>
                <span className={s.flowDrop}>
                  {r.step.dropped ? (
                    <>
                      −<Tween value={r.step.dropped} format={fmtInt} />
                    </>
                  ) : null}
                </span>
              </div>
              {r.step.dropped && detail && !compact ? <div className={s.flowDetail}>{detail}</div> : null}
            </motion.div>
          ),
        )}
      </AnimatePresence>
      {split ? (
        <div className={s.flowRow}>
          <div className={s.flowLine}>
            <span className={s.flowLabel}>Split</span>
            <span className={s.flowTrack}>
              <span className={s.flowTrain} style={{ width: `${(100 * split.train) / n0}%` }} />
              <span className={s.flowHold} style={{ width: `${(100 * split.holdout) / n0}%` }} />
            </span>
            <span className={s.flowCount}>{fmtInt(split.train)}</span>
            <span className={s.flowSplitNote}>+ {fmtInt(split.holdout)} held out</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}
