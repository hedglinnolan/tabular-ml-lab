/**
 * The Results with the seal (M2_CONTRACT §3): the fit computed held-out scores and the server
 * withholds them until the seal is opened once. Sealed, each family's held-out column says so,
 * with the glyph; the open action is a CONSEQUENCE (DESIGN_LANGUAGE §09: full-width interruption,
 * declarative then first-person, resolve or attest) at the end of the Results. Opened, the hollow
 * dots arrive and the card settles into a recorded line. A later upstream change still refits, and
 * the Results say plainly that these numbers came after the seal was opened — with the scores as
 * first opened kept beside them, never overwritten.
 */
import { scaleLinear } from "d3-scale";
import { motion } from "motion/react";
import { useSize } from "../../../components/stage/views/geometry";
import { Rich } from "../../../components/stage/text";
import { useTransitions } from "../../../motion/prefs";
import { fmtInt, fmtMetric, fmtNum } from "../data";
import type { Fixture, ModelFit } from "../types";
import { SealGlyph } from "./SealGlyph";
import s from "./views.module.css";

export type SealPhase = "sealed" | "opened" | "post";

const ROW = 44;

function concern(m: ModelFit, baseline: number, label: string): string | null {
  const d = m.cv.mean - baseline;
  if (Math.abs(d) < 0.0005) return `Ties the outcome's average: CV ${label} ${fmtMetric(m.cv.mean)}`;
  if (d < 0) return `Predicts worse than the outcome's average: CV ${label} ${fmtMetric(m.cv.mean)}`;
  return null;
}

export function SealedResults({
  fx,
  phase,
  onOpen,
  openedAs,
  changedAs,
}: {
  fx: Fixture["results"];
  phase: SealPhase;
  onOpen: () => void;
  openedAs: number;
  changedAs: number;
}) {
  const t = useTransitions();
  const [ref, { w }] = useSize<HTMLDivElement>();
  const first = fx.fits.residual;
  const now = phase === "post" ? fx.fits.density : first;
  const all = [first, fx.fits.density].flatMap((f) => [
    f.baseline,
    ...f.models.flatMap((m) => [m.cv.mean - m.cv.sd, m.cv.mean + m.cv.sd, m.holdout]),
  ]);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.08;
  const x = scaleLinear().domain([lo - pad, hi + pad]).range([8, Math.max(60, w - 8)]);
  const ticks = x.ticks(5);
  const open = phase !== "sealed";
  const metric = fx.label;

  return (
    <div className={s.results} data-view="results" data-phase={phase}>
      {phase === "post" ? (
        <div className={s.postBand} role="status" data-testid="post-seal">
          <span className={s.postKicker}>Changed after the seal was opened</span>
          <span className={s.postText}>
            The energy method changed (#{changedAs}) after the held-out scores were seen once (#{openedAs}). These
            numbers are post-seal; the scores as first opened stay in the record.
          </span>
        </div>
      ) : null}
      <section className={s.section}>
        <div className={s.sectionHead}>
          <h3 className={s.kicker}>Models compared</h3>
          <span className={s.sectionAside}>
            {open ? (
              <>
                held out · opened once, #{openedAs}
              </>
            ) : (
              <>
                <SealGlyph state="grouped" recorded size={14} className={s.inlineGlyph} /> held-out rows: sealed
              </>
            )}
          </span>
        </div>
        <div className={s.cmpHead}>
          <span />
          <span className={s.cmpAxisLabel}>{metric}, cross-validated (higher is better)</span>
          <span className={s.cmpNumHead}>CV mean ± SD · held out</span>
        </div>
        {now.models.map((m, i) => {
          const before = first.models[i]!;
          const c = concern(m, now.baseline, metric);
          return (
            <div key={m.family} className={s.cmpRow} data-family={m.family}>
              <div className={s.cmpName}>
                <span className={s.family}>{m.label}</span>
                {c ? <span className={s.concern}>{c}</span> : null}
              </div>
              <div className={s.cmpPlot} ref={i === 0 ? ref : undefined}>
                {w > 0 ? (
                  <svg width={w} height={ROW} className={s.svg} aria-hidden="true">
                    <line x1={x(now.baseline)} x2={x(now.baseline)} y1={0} y2={ROW} className={s.baseLine} />
                    {phase === "post" ? (
                      <circle cx={x(before.holdout)} cy={ROW / 2} r={7} className={s.hollowFirst} />
                    ) : null}
                    {open ? (
                      <motion.circle
                        initial={{ opacity: 0, r: 2 }}
                        animate={{ opacity: 1, r: 7 }}
                        transition={t.arrive}
                        cx={x(m.holdout)}
                        cy={ROW / 2}
                        className={phase === "post" ? s.hollowPost : s.hollow}
                      />
                    ) : null}
                    <line x1={x(m.cv.mean - m.cv.sd)} x2={x(m.cv.mean + m.cv.sd)} y1={ROW / 2} y2={ROW / 2} className={s.interval} />
                    <circle cx={x(m.cv.mean)} cy={ROW / 2} r={4.5} className={s.dot} />
                  </svg>
                ) : null}
              </div>
              <div className={s.cmpNums}>
                <span className="num">
                  {fmtMetric(m.cv.mean)} ± {fmtMetric(m.cv.sd)}
                </span>
                {open ? (
                  <span className={phase === "post" ? s.cmpHoldPost : s.cmpHold}>
                    held out {fmtMetric(m.holdout)}
                    {phase === "post" ? <span className={s.firstOpened}> · first {fmtMetric(before.holdout)}</span> : null}
                  </span>
                ) : (
                  <span className={s.cmpSealed}>
                    <SealGlyph state="grouped" recorded size={12} className={s.inlineGlyph} /> sealed
                  </span>
                )}
              </div>
            </div>
          );
        })}
        <div className={s.cmpAxis}>
          <span />
          <div>
            {w > 0 ? (
              <svg width={w} height={30} className={s.svg} aria-hidden="true">
                <line x1={x.range()[0]} x2={x.range()[1]} y1={1} y2={1} className={s.axisLine} />
                {ticks.map((tk) => (
                  <text key={tk} x={x(tk)} y={14} textAnchor="middle" className={s.tick}>
                    {fmtNum(tk, 2)}
                  </text>
                ))}
                <text x={x(now.baseline)} y={27} textAnchor="middle" className={s.baseLabel}>
                  baseline {fmtMetric(now.baseline)}
                </text>
              </svg>
            ) : null}
          </div>
          <span />
        </div>
        <p className={s.legendLine}>
          <span className={s.keyDot} /> CV mean ± SD
          {open ? (
            <>
              <span className={s.keyHollow} /> held out{phase === "post" ? " now" : ""}
              {phase === "post" ? (
                <>
                  <span className={s.keyHollowFirst} /> held out as first opened
                </>
              ) : null}
            </>
          ) : null}
          <span className={s.keyBase} /> baseline: the outcome&apos;s average, scored the same way
        </p>
        <p className={s.basis}>
          <Rich
            text={`Mean ± SD over ${fx.folds}-fold cross-validation on ${fmtInt(fx.n_train)} training participants; ${fmtInt(
              fx.n_holdout,
            )} held-out participants ${open ? "scored once" : "sealed"}, grouped by \`${fx.group_column}\`.`}
          />
        </p>
      </section>

      {phase === "sealed" ? (
        <section className={s.consequence} aria-labelledby="open-seal-title" data-testid="open-seal-card">
          <div className={s.conRule} aria-hidden="true" />
          <div className={s.conHead}>
            <SealGlyph state="grouped" recorded size={26} />
            <span className={s.conSignal}>Opened once</span>
          </div>
          <h3 id="open-seal-title" className={s.conTitle}>
            Opening the seal scores the three models on {fmtInt(fx.n_holdout)} participants no choice has seen.
          </h3>
          <p className={s.conBody}>
            It happens once. The held-out {metric} is then fixed in the record; any later change still refits, and is
            marked post-seal in the Results and the manuscript.
          </p>
          <div className={s.conExits}>
            <button type="button" className={s.conAttest} onClick={onOpen} data-testid="open-seal">
              I&apos;m done choosing: open the seal
            </button>
            <span className={s.conOr}>or keep choosing; nothing is opened until you press it.</span>
          </div>
        </section>
      ) : (
        <motion.div
          layout
          className={phase === "post" ? `${s.sealRecorded} ${s.sealRecordedPost}` : s.sealRecorded}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={t.settle}
          data-testid="seal-opened"
        >
          <span className={s.sealRecordedText}>
            The seal was opened once (#{openedAs}): held-out {metric} for {fmtInt(fx.n_holdout)} participants is fixed in
            the record.
          </span>
        </motion.div>
      )}
    </div>
  );
}
