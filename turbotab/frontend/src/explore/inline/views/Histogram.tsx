/**
 * distribution — a histogram, optionally split into the rows a choice keeps and the rows it
 * removes (`kept` must share `hist`'s bin edges). Bars are the same bins in every option, so
 * their heights morph when the previewed option changes; a different binning remounts instead,
 * because bars with different edges are not the same objects (§05.2).
 */
import { useMemo } from "react";
import { motion } from "motion/react";
import { scaleLinear } from "d3-scale";
import { useTransitions } from "../../../motion/prefs";
import { fmtInt, fmtTick } from "../format";
import type { HistogramData, Mark } from "../types";
import s from "./views.module.css";

interface Props {
  hist: HistogramData;
  kept?: HistogramData | null;
  marks?: Mark[];
  width: number;
  height: number;
  variant: "stage" | "spark";
  /** Draw bins that start below this value; the rest are counted, not drawn. */
  xMax?: number;
  /** Shared across small multiples so their bars compare. */
  yMax?: number;
  tone?: "c1" | "c2" | "faint";
  title: string;
  /** Group label shown beside a mark's value ("women", "men"). */
  groupLabel?: (group: string) => string;
  ticks?: number[];
  showOverflow?: boolean;
}

export function overflowOf(hist: HistogramData, xMax: number) {
  let n = 0;
  hist.edges.slice(0, -1).forEach((e, i) => {
    if (e >= xMax) n += hist.counts[i] ?? 0;
  });
  return n;
}

export function Histogram({
  hist,
  kept,
  marks = [],
  width,
  height,
  variant,
  xMax,
  yMax,
  tone = "c1",
  title,
  groupLabel = (g) => g,
  ticks,
  showOverflow = true,
}: Props) {
  const t = useTransitions();
  const stage = variant === "stage";
  const groups = [...new Set(marks.map((mk) => mk.group ?? ""))];
  const markRows = stage ? groups.length : 0;
  const m = stage ? { l: 6, r: 8, t: 8 + markRows * 15, b: 22 } : { l: 1, r: 1, t: 2, b: 2 };

  const bins = useMemo(() => {
    const out: { x0: number; x1: number; all: number; kept: number }[] = [];
    for (let i = 0; i < hist.counts.length; i++) {
      const x0 = hist.edges[i]!;
      const x1 = hist.edges[i + 1]!;
      if (xMax !== undefined && x0 >= xMax) break;
      const all = hist.counts[i] ?? 0;
      out.push({ x0, x1, all, kept: kept ? (kept.counts[i] ?? 0) : all });
    }
    return out;
  }, [hist, kept, xMax]);

  const lo = hist.edges[0] ?? 0;
  const hi = bins.length ? bins[bins.length - 1]!.x1 : (hist.edges[hist.edges.length - 1] ?? 1);
  const x = scaleLinear()
    .domain([lo, hi])
    .range([m.l, width - m.r]);
  const top = yMax ?? Math.max(1, ...bins.map((b) => b.all));
  const y = scaleLinear()
    .domain([0, top])
    .range([height - m.b, m.t]);
  const base = height - m.b;
  const overflow = xMax !== undefined ? overflowOf(hist, xMax) : 0;
  const keptOverflow = xMax !== undefined && kept ? overflowOf(kept, xMax) : overflow;
  const gap = stage ? 1.5 : 0.8;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={stage ? s.histStage : s.histSpark}
      data-tone={tone}
      role="img"
      aria-label={title}
    >
      <line x1={m.l} x2={width - m.r} y1={base + 0.5} y2={base + 0.5} className={s.baseline} />
      {bins.map((b, i) => {
        const bx = x(b.x0) + gap / 2;
        const bw = Math.max(0.5, x(b.x1) - x(b.x0) - gap);
        const hKept = base - y(b.kept);
        const hGone = base - y(b.all) - hKept;
        return (
          <g key={i}>
            <motion.rect
              className={s.barGone}
              x={bx}
              width={bw}
              initial={false}
              animate={{ y: base - hKept - hGone, height: Math.max(0, hGone) }}
              transition={t.arrive}
            />
            <motion.rect
              className={s.bar}
              x={bx}
              width={bw}
              initial={false}
              animate={{ y: base - hKept, height: Math.max(0, hKept) }}
              transition={t.arrive}
            />
          </g>
        );
      })}
      {marks.map((mk, i) => {
        if (mk.value < lo || mk.value > hi) return null;
        const row = groups.indexOf(mk.group ?? "");
        const mx = x(mk.value);
        // A line starts at its own label's row, so it never crosses another group's label; the
        // first cut of a group carries the group's name.
        const firstOfGroup = marks.findIndex((o) => (o.group ?? "") === (mk.group ?? "")) === i;
        const ty = 6 + row * 15 + 5;
        return (
          <g key={`${mk.group}-${mk.value}-${i}`} className={s.mark} data-group={mk.group ?? "all"}>
            <line x1={mx} x2={mx} y1={stage ? ty - 8 : 0} y2={base} />
            {stage ? (
              <text x={mx + 4} y={ty} textAnchor="start">
                {mk.group && firstOfGroup ? (
                  <tspan className={s.markGroup}>{groupLabel(mk.group)} </tspan>
                ) : null}
                {fmtInt(mk.value)}
              </text>
            ) : null}
          </g>
        );
      })}
      {stage ? (
        <g className={s.axis} aria-hidden="true">
          {(ticks ?? x.ticks(5)).map((tv) => (
            <text key={tv} x={x(tv)} y={height - 5} textAnchor="middle">
              {fmtTick(tv)}
            </text>
          ))}
        </g>
      ) : null}
      {stage && showOverflow && overflow > 0 ? (
        <text x={width - m.r} y={base - 9} textAnchor="end" className={s.overflow}>
          <title>
            {`${fmtInt(overflow)} rows lie beyond the drawn range, up to ${fmtInt(hist.edges[hist.edges.length - 1] ?? 0)}` +
              (kept ? `; ${fmtInt(overflow - keptOverflow)} of them excluded` : "")}
          </title>
          +{fmtInt(overflow)} more → {fmtTick(hist.edges[hist.edges.length - 1] ?? 0)}
        </text>
      ) : null}
    </svg>
  );
}
