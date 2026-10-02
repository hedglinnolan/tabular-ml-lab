/**
 * Model comparison (§13): each family's cross-validated mean ± SD as a dot and interval on one
 * shared axis, its held-out score as a hollow dot, the outcome's baseline as a reference line.
 * Concerns sit beside the family's name; the order is the shelf's. Hover or focus a row for the
 * split and folds behind its numbers.
 */
import { scaleLinear } from "d3-scale";
import type { FitArtifact } from "../../../api/m1-stage-types";
import { fmtNum } from "../format";
import type { GlyphState, SealPhase } from "../seal/phase";
import { SealGlyph } from "../seal/SealGlyph";
import seal from "../seal/seal.module.css";
import { Rich } from "../text";
import { useSize } from "../views/geometry";
import type { Comparison as ComparisonData } from "./model";
import s from "./results.module.css";

interface Props {
  data: ComparisonData;
  fit: FitArtifact;
  basis: string;
  /** Where the seal stands: held-out scores show only once it is opened. */
  phase?: SealPhase;
  /** The seal's glyph for its basis (never a clean lock for an exploratory one). */
  glyph?: GlyphState;
}

const ROW = 46;

export function Comparison({ data, fit, basis, phase = "sealed", glyph = "closed" }: Props) {
  const post = phase === "post_seal";
  const [ref, { w }] = useSize<HTMLDivElement>();
  const x = scaleLinear().domain(data.domain).range([8, Math.max(40, w - 8)]);
  const ticks = x.ticks(w < 300 ? 3 : 5);
  const base = data.baseline;
  return (
    <div className={s.comparison} data-testid="model-comparison">
      <div className={s.cmpHead}>
        <span />
        <span className={s.cmpAxisLabel}>
          {data.label}, cross-validated{data.higherIsBetter ? " (higher is better)" : " (lower is better)"}
        </span>
        <span className={s.cmpNumHead}>CV mean ± SD · held out</span>
      </div>
      {data.rows.map((r) => {
        const m = fit.models.find((x) => x.family === r.family);
        const folds = m?.cv[data.metric]?.folds ?? [];
        return (
          <div key={r.family} className={s.cmpRow} tabIndex={0} data-family={r.family}>
            <div className={s.cmpName}>
              <span className={s.family}>{r.label}</span>
              {r.concerns.map((c, i) => (
                <span key={i} className={s.concern}>
                  <Rich text={c} />
                </span>
              ))}
            </div>
            <div className={s.cmpPlot} ref={r === data.rows[0] ? ref : undefined}>
              {w > 0 ? (
                <svg width={w} height={ROW} className={s.svg} aria-hidden="true">
                  {base ? <line x1={x(base.value)} x2={x(base.value)} y1={0} y2={ROW} className={s.baseLine} /> : null}
                  {r.holdout !== null ? (
                    <circle cx={x(r.holdout)} cy={ROW / 2} r={7} className={post ? s.hollowPost : s.hollow} />
                  ) : null}
                  {r.mean !== null ? (
                    <>
                      <line
                        x1={x(r.mean - (r.sd ?? 0))}
                        x2={x(r.mean + (r.sd ?? 0))}
                        y1={ROW / 2}
                        y2={ROW / 2}
                        className={s.interval}
                      />
                      <circle cx={x(r.mean)} cy={ROW / 2} r={4.5} className={s.dot} />
                    </>
                  ) : null}
                </svg>
              ) : null}
            </div>
            <div className={s.cmpNums}>
              <span className="num">
                {fmtNum(r.mean)} ± {fmtNum(r.sd, 2)}
              </span>
              {r.holdout !== null ? (
                <span className={post ? s.cmpHoldPost : s.cmpHold}>held out {fmtNum(r.holdout)}</span>
              ) : phase === "sealed" ? (
                <span className={s.cmpSealed} data-testid="held-out-cell-sealed">
                  <SealGlyph state={glyph} recorded size={12} className={seal.inlineGlyph} /> sealed
                </span>
              ) : (
                <span className={s.cmpHold}>no held-out rows</span>
              )}
            </div>
            <div className={s.tip} role="tooltip">
              <strong>{r.label}</strong>: CV {data.label} {fmtNum(r.mean)} ± {fmtNum(r.sd, 2)} over folds{" "}
              <span className="num">{folds.map((f) => fmtNum(f, 2)).join(", ")}</span>
              {r.holdout !== null ? `; held out ${fmtNum(r.holdout)} on ${fit.n_holdout.toLocaleString("en-US")} rows.` : "."}{" "}
              <Rich text={basis} />
            </div>
          </div>
        );
      })}
      <div className={s.cmpAxis}>
        <span />
        <div className={s.cmpAxisPlot}>
          {w > 0 ? (
            <svg width={w} height={34} className={s.svg} aria-hidden="true">
              <line x1={x.range()[0]} x2={x.range()[1]} y1={1} y2={1} className={s.axisLine} />
              {ticks.map((t) => (
                <text key={t} x={x(t)} y={14} textAnchor="middle" className={s.tick}>
                  {fmtNum(t, 2)}
                </text>
              ))}
              {base ? (
                <text x={x(base.value)} y={29} textAnchor="middle" className={s.baseLabel}>
                  baseline {fmtNum(base.value, 2)}
                </text>
              ) : null}
            </svg>
          ) : null}
        </div>
        <span />
      </div>
      <p className={s.legendLine}>
        <span className={s.keyDot} /> CV mean ± SD
        {phase === "opened" || post ? (
          <>
            <span className={post ? s.keyHollowPost : s.keyHollow} /> held out{post ? ", after the opening" : ""}
          </>
        ) : null}
        {base ? (
          <>
            <span className={s.keyBase} /> baseline: {base.label}, scored the same way
          </>
        ) : null}
      </p>
      <p className={s.basis}>
        <Rich text={basis} />
      </p>
    </div>
  );
}
