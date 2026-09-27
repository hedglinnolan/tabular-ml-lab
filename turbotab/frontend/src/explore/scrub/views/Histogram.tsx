/**
 * distribution — a histogram with its cut marks and, optionally, a rug of the table's rows.
 *
 * Identity rule, decided by keys: states that share `domain` (same bins, e.g. all rows vs the rows
 * an exclusion keeps) morph bar by bar — the bar really is the same kcal range losing rows. States
 * on different axes (counts vs log counts) crossfade their bars: bin 7 of one is not bin 7 of the
 * other. The rug ticks are real rows (the ones in the table), so they glide from their old value
 * to their new one on either kind of change.
 */
import { useCallback, useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { fmtNum } from "../data";
import { SceneBuilder, cached, unionKeys, type Scene } from "../engine/morph";
import { attrs, css, place, useMorph, useRegistry } from "../engine/scrub";
import { useWidth } from "../engine/useSize";
import s from "./views.module.css";

export interface CutMark {
  value: number;
  label: string;
  group: string | null;
  /** Rows this cut removes on its outer side, and which side that is. */
  cut: number;
  side: "below" | "above";
}

export interface HistState {
  edges: number[];
  counts: number[];
  /** Outline of the counts before the choice (only in an "after" state). */
  ghost?: number[];
  /** States with the same domain morph bars; different domains crossfade. */
  domain: string;
  label: string;
  marks: CutMark[];
  rug?: { row: number; value: number }[];
  after: boolean;
  note?: string;
}

interface Props {
  states: Record<string, HistState>;
  height?: number;
  /** Bins at or beyond this value are drawn narrow, past an axis break (a long, thin tail). */
  breakAt?: number;
  lanes?: (string | null)[];
  span?: readonly [number, number];
  hotRow?: number | null;
  label?: string;
}

const M = { l: 14, r: 14, b: 40 };
const LANE_H = 19;

export function ScrubHistogram({
  states,
  height = 220,
  breakAt,
  lanes = [null],
  span,
  hotRow = null,
  label,
}: Props) {
  const [wrap, width] = useWidth<HTMLDivElement>(600);
  const top = lanes.length * LANE_H + 8;
  const iw = width - M.l - M.r;
  const base = height - M.b;
  const ih = base - top;
  const { map, reg } = useRegistry<HTMLElement | SVGElement>();

  // One y scale per domain, shared by every state on it, so a kept bar is visibly shorter.
  const yMax = useMemo(() => {
    const out: Record<string, number> = {};
    for (const st of Object.values(states)) {
      const m = Math.max(...st.counts, ...(st.ghost ?? []));
      out[st.domain] = Math.max(out[st.domain] ?? 0, m);
    }
    return out;
  }, [states]);

  const xOf = useCallback(
    (edges: number[]) => {
      const lo = edges[0]!;
      const hi = edges[edges.length - 1]!;
      if (breakAt === undefined || breakAt >= hi) return scaleLinear().domain([lo, hi]).range([M.l, M.l + iw]);
      const mainW = iw * 0.84;
      const gap = 12;
      const main = scaleLinear().domain([lo, breakAt]).range([M.l, M.l + mainW]);
      const tail = scaleLinear().domain([breakAt, hi]).range([M.l + mainW + gap, M.l + iw]);
      const f = (v: number) => (v <= breakAt ? main(v) : tail(v));
      f.ticks = () => main.ticks(6).filter((t) => t < breakAt * 0.97);
      f.breakX = M.l + mainW + gap / 2;
      return f as unknown as ReturnType<typeof scaleLinear<number, number>> & { breakX?: number };
    },
    [breakAt, iw],
  );

  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? states.now!;
      const x = xOf(st.edges);
      const y = scaleLinear().domain([0, yMax[st.domain] ?? 1]).range([0, ih]);
      const sb = new SceneBuilder();
      st.counts.forEach((c, i) => {
        const x0 = x(st.edges[i]!);
        const x1 = x(st.edges[i + 1]!);
        sb.set(`bar:${st.domain}:${i}`, { x: x0 + 0.5, w: Math.max(1, x1 - x0 - 1), h: y(c), c: st.after ? 1 : 0, o: 1 });
        const g = st.ghost?.[i];
        if (g !== undefined && g > c)
          sb.set(`ghost:${st.domain}:${i}`, { x: x0 + 0.5, w: Math.max(1, x1 - x0 - 1), h: y(g), o: 1 });
      });
      // Ticks too close to the end label would collide with it.
      for (const v of x.ticks(6))
        if (M.l + iw - x(v) > 34) sb.set(`xt:${st.domain}:${v}`, { x: x(v), o: 1, swap: 1 });
      sb.set(`xe:${st.domain}:${st.edges[st.edges.length - 1]}`, { x: M.l + iw, o: 1, swap: 1 });
      sb.set(`xl:${st.label}`, { o: 1, swap: 1 });
      const bx = (x as { breakX?: number }).breakX;
      if (bx !== undefined) sb.set(`brk:${st.domain}`, { x: bx, o: 1 });
      for (const m of st.marks) {
        const lane = Math.max(0, lanes.indexOf(m.group));
        sb.set(`mk:${m.group ?? "all"}:${m.value}:${m.side}:${m.cut}`, { x: x(m.value), y: lane * LANE_H, o: 1 });
      }
      for (const g of new Set(st.marks.map((m) => m.group))) if (g) sb.set(`lane:${g}`, { y: lanes.indexOf(g) * LANE_H, o: 1 });
      for (const r of st.rug ?? []) sb.set(`rug:${r.row}`, { x: x(r.value), o: 1 });
      if (st.note) sb.set(`note:${st.note}`, { o: 1, swap: 1 });
      return sb.scene;
    });
  }, [states, xOf, yMax, ih, iw, lanes]);

  const keys = useMemo(() => unionKeys(Object.keys(states).map((k) => sceneOf(k))), [states, sceneOf]);

  const apply = useCallback(
    (sc: Scene) => {
      for (const [k, el] of map.current) {
        const isLine = k.endsWith("#line");
        const it = sc.items.get(isLine ? k.slice(0, -5) : k);
        const kind = k.slice(0, k.indexOf(":"));
        if (isLine) place(el, it, { y: false });
        else if (kind === "bar" || kind === "ghost") {
          const o = it ? Math.min(1, it.o ?? 1) : 0;
          css(el, { opacity: String(o) });
          if (it) {
            attrs(el, { x: String(it.x) });
            attrs(el, { width: String(it.w) });
            attrs(el, { y: String(base - (it.h ?? 0)) });
            attrs(el, { height: String(Math.max(0, it.h ?? 0)) });
            if (kind === "bar") css(el, { fill: `color-mix(in oklab, var(--c2) ${Math.round((it.c ?? 0) * 100)}%, var(--c4))` });
          }
        } else if (kind === "mk" || kind === "lane") place(el, it, { x: kind === "mk" });
        else if (kind === "xt" || kind === "xe" || kind === "rug" || kind === "brk") place(el, it, { y: false });
        else place(el, it, { x: false, y: false });
      }
    },
    [map, base],
  );

  useMorph({ sceneOf, apply, span });

  const markKeys = keys.filter((k) => k.startsWith("mk:"));
  return (
    <div ref={wrap} className={s.hist} style={{ height }} role="img" aria-label={label}>
      <svg width={width} height={height} className={s.histSvg} aria-hidden="true">
        <defs>
          <pattern id="tt-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="4" className={s.hatchLine} />
          </pattern>
        </defs>
        {keys
          .filter((k) => k.startsWith("ghost:"))
          .map((k) => (
            <rect key={k} ref={reg(k)} className={s.ghost} style={{ opacity: 0 }} />
          ))}
        {keys
          .filter((k) => k.startsWith("bar:"))
          .map((k) => (
            <rect key={k} ref={reg(k)} className={s.bar} style={{ opacity: 0 }} />
          ))}
        <line className={s.axis} x1={M.l} x2={M.l + iw} y1={base + 0.5} y2={base + 0.5} />
        {keys
          .filter((k) => k.startsWith("brk:"))
          .map((k) => (
            <g key={k} ref={reg(k)} style={{ opacity: 0 }}>
              <rect x={-5} y={base - 4} width={10} height={9} className={s.breakMask} />
              <path d={`M -5 ${base + 4} l 4 -8 M 1 ${base + 4} l 4 -8`} className={s.breakGlyph} />
            </g>
          ))}
        {keys
          .filter((k) => k.startsWith("xt:") || k.startsWith("xe:"))
          .map((k) => (
            <g key={k} ref={reg(k)} style={{ opacity: 0 }}>
              <line className={s.tick} y1={base} y2={base + 4} />
              <text className={s.tickText} y={base + 16} textAnchor={k.startsWith("xe:") ? "end" : "middle"}>
                {fmtNum(Number(k.split(":").pop()), 3)}
              </text>
            </g>
          ))}
        {keys
          .filter((k) => k.startsWith("rug:"))
          .map((k) => {
            const row = Number(k.slice(4));
            return (
              <line
                key={k}
                ref={reg(k)}
                className={s.rug}
                data-hot={hotRow === row || undefined}
                y1={base - 9}
                y2={base - 1}
                style={{ opacity: 0 }}
              />
            );
          })}
        {markKeys.map((k) => {
          const [, group] = k.split(":");
          return (
            <line
              key={k}
              ref={reg(`${k}#line`)}
              className={s.markLine}
              data-group={group}
              x1={0}
              x2={0}
              y1={top - 4}
              y2={base}
              style={{ opacity: 0 }}
            />
          );
        })}
      </svg>
      {keys
        .filter((k) => k.startsWith("lane:"))
        .map((k) => (
          <span key={k} ref={reg(k)} className={s.laneLabel} style={{ opacity: 0 }}>
            {k.slice(5)}
          </span>
        ))}
      {markKeys.map((k) => {
        const [, group, value, side, cut] = k.split(":");
        return (
          <span
            key={k}
            ref={reg(k)}
            className={s.markLabel}
            data-side={side}
            data-group={group}
            style={{ opacity: 0 }}
          >
            <span className={s.markInner}>
              {side === "below" ? <span className={s.cut}>−{fmtNum(Number(cut))} ←</span> : null}
              <span className={s.markValue}>{fmtNum(Number(value))}</span>
              {side === "above" ? <span className={s.cut}>→ −{fmtNum(Number(cut))}</span> : null}
            </span>
          </span>
        );
      })}
      <div className={s.histLabels}>
        {keys
          .filter((k) => k.startsWith("xl:"))
          .map((k) => (
            <span key={k} ref={reg(k)} className={s.histLabel} style={{ opacity: 0 }}>
              {k.slice(3)}
            </span>
          ))}
      </div>
      {keys
        .filter((k) => k.startsWith("note:"))
        .map((k) => (
          <span key={k} ref={reg(k)} className={s.histNote} style={{ opacity: 0 }}>
            {k.slice(5)}
          </span>
        ))}
    </div>
  );
}
