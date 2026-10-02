/**
 * The row flow under a reshape. One new distinction the M1 flow did not need: rows that are
 * **combined** into a partner row are not rows that **leave**. The mean folds 300 recalls into
 * their partners (drawn as a fold, bracketed back onto the row they joined); first and last drop
 * 300 recalls (the hatch the flow already uses for rows that leave). Same count, different claim.
 */
import { motion } from "motion/react";
import { usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { useTransitions } from "../../../motion/prefs";
import { Rich } from "../../../components/stage/text";
import { fmtInt } from "../data";
import type { FlowStep } from "../types";
import s from "./views.module.css";

export function ReshapeFlow({ steps, last, emphasis }: { steps: FlowStep[]; last: number; emphasis?: string }) {
  const ui = usePlayerUi(usePlayerStore());
  const t = useTransitions();
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));
  const n0 = Math.max(1, steps[0]?.n ?? 1);
  const shown = at >= 2 ? steps : steps.slice(0, 1);
  return (
    <div className={s.flow} data-view="row_flow" data-state={at}>
      {shown.map((st) => (
        <motion.div
          key={st.key}
          className={s.flowRow}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={t.arrive}
          data-step={st.key}
        >
          <div className={s.flowLine}>
            <span className={s.flowLabel}>
              <Rich text={st.label} />
            </span>
            <span className={s.flowTrack}>
              <motion.span
                className={st.key === emphasis ? s.flowKeptNow : s.flowKept}
                initial={false}
                animate={{ width: `${(100 * st.n) / n0}%` }}
                transition={t.arrive}
              />
              {st.combined ? (
                <motion.span
                  className={s.flowFolded}
                  initial={false}
                  animate={{ width: `${(100 * st.combined) / n0}%` }}
                  transition={t.arrive}
                />
              ) : null}
              {st.dropped ? (
                <motion.span
                  className={s.flowGone}
                  initial={false}
                  animate={{ width: `${(100 * st.dropped) / n0}%` }}
                  transition={t.arrive}
                />
              ) : null}
            </span>
            <span className={s.flowCount}>{fmtInt(st.n)}</span>
            <span className={s.flowDrop}>
              {st.combined ? `${fmtInt(st.combined)} folded in` : st.dropped ? `−${fmtInt(st.dropped)}` : ""}
            </span>
          </div>
          {st.reason ? <div className={s.flowDetail}>{st.reason}</div> : null}
        </motion.div>
      ))}
    </div>
  );
}
