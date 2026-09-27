/**
 * relationship — a nutrient against energy. Every dot is one training row, and the rows are the
 * same in every option (the fixture samples them once), so moving between options MOVES the dots:
 * the reader watches the same people fall into line or stay tilted. Each option's y values sit in
 * their own units, mapped on the same band (mean − 3.5 sd … mean + 5 sd), so tilt is comparable
 * across options while the axis keeps the column's real units.
 */
import { useCallback, useMemo, useRef } from "react";
import { scaleLinear } from "d3-scale";
import type { ScatterState } from "../fixture";
import { fmtTick } from "../format";
import { useMorph } from "../useMorph";
import s from "./views.module.css";

const X_DOMAIN: [number, number] = [400, 5100];

/** Pixels per standard deviation are the same for every option, so tilt compares; the band is
 *  centered on the middle 98% of that option's values, so every cloud fills the plot. */
const HALF_BAND_SD = 3.1;

function quantile(sorted: number[], q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i]!;
}

const domains = new WeakMap<ScatterState, [number, number]>();

export function yDomain(st: ScatterState): [number, number] {
  const hit = domains.get(st);
  if (hit) return hit;
  const sorted = [...st.ys].sort((a, b) => a - b);
  const mid = (quantile(sorted, 0.01) + quantile(sorted, 0.99)) / 2;
  const d: [number, number] = [mid - HALF_BAND_SD * st.stats.sd, mid + HALF_BAND_SD * st.stats.sd];
  domains.set(st, d);
  return d;
}

function stats1(xs: number[]) {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return { mean, sd };
}

interface Props {
  xs: number[];
  state: ScatterState;
  ghost?: ScatterState | null;
  width: number;
  height: number;
  variant: "stage" | "spark";
  sample?: number[];
  tone?: "c1" | "c2";
  xLabel?: string;
  title: string;
}

const Z_LINE: [number, number] = [-2.2, 2.6];

export function Scatter({
  xs,
  state,
  ghost,
  width,
  height,
  variant,
  sample,
  tone = "c1",
  xLabel,
  title,
}: Props) {
  const stage = variant === "stage";
  const m = stage ? { l: 40, r: 8, t: 8, b: 22 } : { l: 3, r: 3, t: 3, b: 3 };
  const idx = useMemo(() => sample ?? xs.map((_, i) => i), [sample, xs]);
  const x = useMemo(
    () =>
      scaleLinear()
        .domain(X_DOMAIN)
        .range([m.l, width - m.r]),
    [m.l, m.r, width],
  );
  const xStats = useMemo(() => stats1(xs), [xs]);
  const top = m.t;
  const bottom = height - m.b;

  const yOf = useCallback(
    (st: ScatterState) => scaleLinear().domain(yDomain(st)).range([bottom, top]).clamp(true),
    [bottom, top],
  );
  const y = useMemo(() => yOf(state), [yOf, state]);

  // Target pixel positions: one per drawn dot, then the two ends of the trend line.
  const target = useMemo(() => {
    const out = new Float64Array(idx.length + 2);
    idx.forEach((i, k) => (out[k] = y(state.ys[i]!)));
    const r = state.r ?? 0;
    Z_LINE.forEach((z, k) => (out[idx.length + k] = y(state.stats.mean + r * z * state.stats.sd)));
    return out;
  }, [idx, y, state]);

  const dots = useRef<(SVGCircleElement | null)[]>([]);
  const line = useRef<SVGLineElement | null>(null);
  const apply = useCallback(
    (v: Float64Array) => {
      const n = idx.length;
      for (let k = 0; k < n; k++) dots.current[k]?.setAttribute("cy", v[k]!.toFixed(1));
      line.current?.setAttribute("y1", v[n]!.toFixed(1));
      line.current?.setAttribute("y2", v[n + 1]!.toFixed(1));
    },
    [idx],
  );
  useMorph(target, apply);

  const lineX = Z_LINE.map((z) => x(xStats.mean + z * xStats.sd)) as [number, number];

  const ghostPts = useMemo(() => {
    if (!ghost) return null;
    const gy = yOf(ghost);
    const r = ghost.r ?? 0;
    return {
      dots: idx.map((i) => [x(xs[i]!), gy(ghost.ys[i]!)] as const),
      line: Z_LINE.map((z) => gy(ghost.stats.mean + r * z * ghost.stats.sd)) as [number, number],
    };
  }, [ghost, yOf, idx, x, xs]);

  const yTicks = stage ? y.ticks(4) : [];
  const xTicks = stage ? [1000, 2000, 3000, 4000] : [];
  const rDot = stage ? 2.1 : 1.05;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={stage ? s.scatterStage : s.scatterSpark}
      data-tone={tone}
      role="img"
      aria-label={title}
    >
      {stage ? (
        <g className={s.axis} aria-hidden="true">
          {yTicks.map((t) => (
            <g key={`y${t}`} transform={`translate(0 ${y(t)})`}>
              <line x1={m.l} x2={width - m.r} className={s.grid} />
              <text x={m.l - 7} dy="0.32em" textAnchor="end">
                {fmtTick(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={x(t)} y={height - 6} textAnchor="middle">
              {fmtTick(t)}
            </text>
          ))}
          {xLabel ? (
            <text x={width - m.r} y={height - 6} textAnchor="end" className={s.axisName}>
              {xLabel} →
            </text>
          ) : null}
        </g>
      ) : null}
      {ghostPts ? (
        <g className={s.ghost} aria-hidden="true">
          {ghostPts.dots.map(([gx, gy], k) => (
            <circle key={k} cx={gx.toFixed(1)} cy={gy.toFixed(1)} r={rDot} />
          ))}
          <line
            x1={lineX[0]}
            x2={lineX[1]}
            y1={ghostPts.line[0]}
            y2={ghostPts.line[1]}
            className={s.ghostLine}
          />
        </g>
      ) : null}
      <g className={s.dots} aria-hidden="true" data-dots={variant}>
        {idx.map((i, k) => (
          <circle
            key={i}
            ref={(el) => {
              dots.current[k] = el;
            }}
            cx={x(xs[i]!).toFixed(1)}
            r={rDot}
          />
        ))}
      </g>
      <line ref={line} x1={lineX[0]} x2={lineX[1]} className={s.trend} aria-hidden="true" />
    </svg>
  );
}
