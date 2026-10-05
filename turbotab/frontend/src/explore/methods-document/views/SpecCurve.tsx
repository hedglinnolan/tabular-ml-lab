/**
 * "Which of my decisions mattered?" (BLUEPRINT §11.4): the exposure's estimate across the
 * alternatives declared before the estimates were shown — the model sequence, the declared
 * secondary model and the declared sensitivity analysis — each row one decision changed from the
 * primary. A specification curve read as sensitivity, never as a way to choose: the primary is the
 * reported estimate whatever the others say.
 *
 * Every row is a server estimate (the effects, secondary and sensitivity stages); the chart is
 * one series with the primary emphasized, so no categorical palette is needed (dataviz: highlight
 * one, gray the rest), and each row prints its numbers, so the figure is also its own table.
 */
import { useState } from "react";
import { Rich } from "../../../components/stage/text";
import s from "../doc.module.css";

export interface SpecRow {
  key: string;
  label: string;
  /** Which declared decision this row changes, in words. */
  changes: string;
  estimate: number;
  low: number;
  high: number;
  n: number;
  primary: boolean;
  source: string;
}

const fmt = (x: number) => (x < 0 ? `−${Math.abs(x).toFixed(3)}` : x.toFixed(3));

export function SpecCurve({ rows, unit, exposure }: { rows: SpecRow[]; unit: string; exposure: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const sorted = [...rows].sort((a, b) => a.estimate - b.estimate);
  const lo = Math.min(0, ...rows.map((r) => r.low));
  const hi = Math.max(0, ...rows.map((r) => r.high));
  const pad = (hi - lo) * 0.08;
  const min = lo - pad;
  const max = hi + pad;
  const W = 340;
  const x = (v: number) => ((v - min) / (max - min)) * W;
  const primary = rows.find((r) => r.primary);
  const span = rows.length ? Math.max(...rows.map((r) => r.estimate)) - Math.min(...rows.map((r) => r.estimate)) : 0;
  const ticks = niceTicks(min, max);
  const below = rows.every((r) => r.high < 0);
  const above = rows.every((r) => r.low > 0);
  return (
    <figure className={s.spec} data-purpose="spec_curve">
      <div className={s.specHead}>
        <span className={s.specKicker}>Sensitivity, not a choice</span>
        <p className={s.specLead}>
          The estimate of <Rich text={`\`${exposure}\``} /> across the {rows.length} specifications declared before any
          estimate was shown spans <span className="num">{fmt(span)}</span>;{" "}
          {below
            ? "every interval lies below zero."
            : above
              ? "every interval lies above zero."
              : "some intervals include zero."}
        </p>
      </div>
      <div className={s.specGrid} role="table" aria-label="The estimate in each declared specification">
        <div className={s.specAxisRow} role="row">
          <span role="columnheader" className={s.specColHead}>
            Specification
          </span>
          <svg className={s.specAxis} viewBox={`0 0 ${W} 22`} preserveAspectRatio="none" aria-hidden="true">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={16} y2={22} className={s.specTick} />
                <text x={x(t)} y={11} textAnchor="middle" className={s.specTickText}>
                  {fmt(t).replace(/0+$/, "").replace(/\.$/, "")}
                </text>
              </g>
            ))}
          </svg>
          <span role="columnheader" className={s.specColHead}>
            Estimate (95% CI)
          </span>
        </div>
        {sorted.map((r) => (
          <div
            key={r.key}
            role="row"
            className={s.specRow}
            data-primary={r.primary || undefined}
            data-hover={hover === r.key || undefined}
            onPointerEnter={() => setHover(r.key)}
            onPointerLeave={() => setHover(null)}
          >
            <span role="cell" className={s.specLabel}>
              <span className={s.specName}>
                <Rich text={r.label} />
              </span>
              <span className={s.specChanges}>{r.changes}</span>
            </span>
            <svg
              role="cell"
              className={s.specPlot}
              viewBox={`0 0 ${W} 30`}
              preserveAspectRatio="none"
              aria-label={`${r.label}: ${fmt(r.estimate)} (${fmt(r.low)} to ${fmt(r.high)})`}
            >
              {ticks.map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={30} className={t === 0 ? s.specZero : s.specGridLine} />
              ))}
              {primary ? (
                <line x1={x(primary.estimate)} x2={x(primary.estimate)} y1={0} y2={30} className={s.specRef} />
              ) : null}
              <line x1={x(r.low)} x2={x(r.high)} y1={15} y2={15} className={s.specWhisker} />
              <circle cx={x(r.estimate)} cy={15} r={r.primary ? 5.5 : 4.5} className={s.specDot} />
            </svg>
            <span role="cell" className={s.specNum}>
              {fmt(r.estimate)} <span className={s.specCi}>({fmt(r.low)}, {fmt(r.high)})</span>
              <span className={s.specN}>n {r.n.toLocaleString("en-US")}</span>
            </span>
            {hover === r.key ? (
              <span className={s.specTip} role="tooltip">
                {r.source}
              </span>
            ) : null}
          </div>
        ))}
      </div>
      <figcaption className={s.specCaption}>
        <Rich text={unit} /> The dashed line is the primary estimate; zero is the solid line. Each row is the same
        model with one declared decision changed.
      </figcaption>
    </figure>
  );
}

function niceTicks(min: number, max: number): number[] {
  const range = max - min;
  const step = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1].find((s) => range / s <= 6) ?? 0.1;
  const out: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-12; t += step) out.push(Math.round(t / step) * step);
  return out;
}
