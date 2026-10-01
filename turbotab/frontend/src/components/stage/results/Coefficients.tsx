/**
 * Coefficients for the linear families (§13): a forest plot of the exposures, with confidence
 * intervals under inference. Under prediction the estimates are drawn without intervals and a
 * note says they are not interpreted.
 */
import { useState } from "react";
import { scaleLinear } from "d3-scale";
import type { Coefficient, FittedModel } from "../../../api/m1-stage-types";
import { fmtNum } from "../format";
import { useSize } from "../views/geometry";
import { forestDomain } from "./model";
import s from "./results.module.css";

interface Props {
  models: { model: FittedModel; coefficients: Coefficient[] }[];
  purpose: string | null;
  target: string;
}

const ROW = 24;

function fmtP(p: number | null): string {
  if (p === null || !Number.isFinite(p)) return "";
  if (p < 0.001) return "p < 0.001";
  return `p ${p.toFixed(3)}`;
}

export function Coefficients({ models, purpose, target }: Props) {
  const [pick, setPick] = useState(0);
  const [ref, { w }] = useSize<HTMLDivElement>();
  if (!models.length) return null;
  const current = models[Math.min(pick, models.length - 1)]!;
  const coefs = current.coefficients;
  const inference = purpose === "inference";
  const domain = forestDomain(coefs);
  const x = scaleLinear().domain(domain).nice().range([10, Math.max(60, w - 10)]);
  const ticks = x.ticks(w < 300 ? 3 : 6);
  const h = coefs.length * ROW + 8;
  return (
    <div className={s.forest} data-testid="coefficients">
      {models.length > 1 ? (
        <div className={s.switcher} role="radiogroup" aria-label="Family">
          {models.map((m, i) => (
            <button
              key={m.model.family}
              type="button"
              role="radio"
              aria-checked={i === pick}
              className={s.switch}
              onClick={() => setPick(i)}
            >
              {m.model.label}
            </button>
          ))}
        </div>
      ) : null}
      <div className={s.forestGrid}>
        <div className={s.forestNames}>
          {coefs.map((c) => (
            <span key={c.feature} className={s.feature} style={{ height: ROW }}>
              {c.feature}
            </span>
          ))}
        </div>
        <div ref={ref} className={s.forestPlot}>
          {w > 0 ? (
            <svg width={w} height={h + 26} className={s.svg} role="img" aria-label={`${current.model.label} coefficients`}>
              {ticks.map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={h} className={t === 0 ? s.zeroLine : s.gridLine} />
              ))}
              {coefs.map((c, i) => {
                const y = i * ROW + ROW / 2;
                const hasCi = inference && c.ci_low !== null && c.ci_high !== null;
                return (
                  <g key={c.feature}>
                    {hasCi ? <line x1={x(c.ci_low!)} x2={x(c.ci_high!)} y1={y} y2={y} className={s.interval} /> : null}
                    {c.estimate !== null ? (
                      <rect x={x(c.estimate) - 4} y={y - 4} width={8} height={8} className={s.square} />
                    ) : null}
                  </g>
                );
              })}
              {ticks.map((t) => (
                <text key={`t${t}`} x={x(t)} y={h + 14} textAnchor="middle" className={s.tick}>
                  {fmtNum(t, 2)}
                </text>
              ))}
            </svg>
          ) : null}
        </div>
        <div className={s.forestNums}>
          {coefs.map((c) => (
            <span key={c.feature} className={s.coefNum} style={{ height: ROW }}>
              {fmtNum(c.estimate)}
              {inference && c.ci_low !== null && c.ci_high !== null ? (
                <span className={s.ci}>
                  [{fmtNum(c.ci_low)}, {fmtNum(c.ci_high)}] {fmtP(c.p)}
                </span>
              ) : null}
            </span>
          ))}
        </div>
      </div>
      <p className={s.basis}>
        {inference
          ? `Change in predicted ${target} per unit of each input, holding the others; 95% confidence intervals.`
          : `Under prediction the coefficients describe the fitted model; they are not interpreted as effects, so no intervals are drawn.`}
      </p>
    </div>
  );
}
