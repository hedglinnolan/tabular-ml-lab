/** A small step-outline histogram. d3 does the math; React renders the SVG. */
import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { area, curveStepAfter } from "d3-shape";
import type { Histogram } from "../../api/schema";
import { fmtInt, fmtStat } from "../../util/format";
import styles from "./MiniHistogram.module.css";

const W = 120;
const H = 32;

export function MiniHistogram({ histogram, label }: { histogram: Histogram; label: string }) {
  const { edges, counts } = histogram;
  const d = useMemo(() => {
    if (counts.length === 0 || edges.length < 2) return null;
    const x = scaleLinear()
      .domain([edges[0]!, edges[edges.length - 1]!])
      .range([0, W]);
    const y = scaleLinear()
      .domain([0, Math.max(1, ...counts)])
      .range([H, 1]);
    const points: [number, number][] = counts.map((c, i) => [edges[i]!, c]);
    points.push([edges[edges.length - 1]!, counts[counts.length - 1]!]);
    return area<[number, number]>()
      .x((p) => x(p[0]))
      .y0(H)
      .y1((p) => y(p[1]))
      .curve(curveStepAfter)(points);
  }, [edges, counts]);
  if (!d) return null;
  const total = counts.reduce((a, b) => a + b, 0);
  return (
    <figure className={styles.fig}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        role="img"
        aria-label={`${label}: ${fmtInt(total)} values from ${fmtStat(edges[0]!)} to ${fmtStat(edges[edges.length - 1]!)}`}
      >
        <path d={d} className={styles.area} />
      </svg>
      <figcaption className={styles.axis} aria-hidden="true">
        <span>{fmtStat(edges[0]!)}</span>
        <span>{fmtStat(edges[edges.length - 1]!)}</span>
      </figcaption>
    </figure>
  );
}
