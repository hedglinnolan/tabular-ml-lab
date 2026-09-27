/**
 * relationship — a nutrient against energy, as recorded and after the option.
 *
 * The 800 sampled rows are the same rows in both panels, so they are the one thing
 * allowed to move: when an option is previewed, each point glides from where it sits to
 * where that option would put it. The left panel never moves; it is the reference the
 * eye compares against. Axes that change meaning cross-fade instead of sliding.
 */
import { useCallback, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { extent } from "d3-array";
import { fmtR, fmtTick } from "../format";
import type { RelationshipView } from "../types";
import { useMorph, useSize } from "./hooks";
import s from "./views.module.css";
import { Tween, useStageTransitions } from "../motion";

interface Props {
  view: RelationshipView;
  compact?: boolean;
  /** Only the recorded state (a finding's evidence). */
  evidence?: boolean;
}

const PAD = { l: 44, r: 12, t: 42, b: 30 };

function ols(points: [number, number][]): [number, number] {
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (const [x, y] of points) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const n = points.length;
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  return [slope, (sy - slope * sx) / n];
}

function dotPath(v: Float64Array, count: number, r: number): string {
  const d: string[] = [];
  for (let i = 0; i < count; i++) {
    const x = v[i * 2]!;
    const y = v[i * 2 + 1]!;
    d.push(`M${(x - r).toFixed(1)},${y.toFixed(1)}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`);
  }
  return d.join("");
}

function Axis({
  x,
  y,
  box,
  compact,
}: {
  x: ScaleLinear<number, number>;
  y: ScaleLinear<number, number>;
  box: { x0: number; x1: number; y0: number; y1: number };
  compact: boolean;
}) {
  const width = box.x1 - box.x0;
  const xt = x.ticks(width < 160 ? 1 : compact || width < 260 ? 2 : 4);
  const yt = y.ticks(compact ? 3 : 4);
  return (
    <g className={s.axis}>
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.x0} x2={box.x1} y1={y(t)} y2={y(t)} className={s.grid} />
          <text x={box.x0 - 6} y={y(t)} dy="0.32em" textAnchor="end">
            {fmtTick(t)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={`x${t}`} x={x(t)} y={box.y1 + 15} textAnchor="middle">
          {fmtTick(t)}
        </text>
      ))}
      <line x1={box.x0} x2={box.x1} y1={box.y1} y2={box.y1} className={s.baseline} />
    </g>
  );
}

export function Relationship({ view, compact = false, evidence = false }: Props) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const t = useStageTransitions();
  const pathRef = useRef<SVGPathElement>(null);
  const lineRef = useRef<SVGLineElement>(null);
  const n = view.points_after.length;
  const r = compact ? 1.5 : 2.1;

  const geo = useMemo(() => {
    if (w < 60 || h < 60) return null;
    const gap = compact ? 18 : 30;
    const beforeW = evidence ? 0 : Math.round((w - gap) * 0.36);
    const b = { x0: PAD.l, x1: beforeW - PAD.r, y0: PAD.t, y1: h - PAD.b };
    const a = {
      x0: evidence ? PAD.l : beforeW + gap + PAD.l,
      x1: w - PAD.r,
      y0: PAD.t,
      y1: h - PAD.b,
    };
    const xMax = extent(view.points_before, (p) => p[0])[1] ?? 1;
    const xb = scaleLinear().domain([0, xMax]).nice().range([b.x0, b.x1]);
    const xa = scaleLinear().domain([0, xMax]).nice().range([a.x0, a.x1]);
    const yb = scaleLinear()
      .domain([0, extent(view.points_before, (p) => p[1])[1] ?? 1])
      .nice()
      .range([b.y1, b.y0]);
    const [lo, hi] = extent(view.points_after, (p) => p[1]) as [number, number];
    const ya = scaleLinear()
      .domain([Math.min(0, lo), hi])
      .nice()
      .range([a.y1, a.y0]);
    return { b, a, xb, xa, yb, ya };
  }, [w, h, compact, evidence, view.points_before, view.points_after]);

  const beforePath = useMemo(() => {
    if (!geo || evidence) return "";
    const v = new Float64Array(n * 2);
    view.points_before.forEach(([x, y], i) => {
      v[i * 2] = geo.xb(x);
      v[i * 2 + 1] = geo.yb(y);
    });
    return dotPath(v, n, r);
  }, [geo, evidence, n, r, view.points_before]);

  const [slopeB, icB] = useMemo(() => ols(view.points_before), [view.points_before]);
  const [slopeA, icA] = useMemo(() => ols(view.points_after), [view.points_after]);

  // The morph runs in unit coordinates (0..1 of the after panel), which depend on the
  // data's nice domains only — never on the panel's pixel size.
  const unit = useMemo(() => {
    const ux = scaleLinear()
      .domain([0, extent(view.points_before, (p) => p[0])[1] ?? 1])
      .nice();
    const [lo, hi] = extent(view.points_after, (p) => p[1]) as [number, number];
    const uya = scaleLinear().domain([Math.min(0, lo), hi]).nice();
    const uyb = scaleLinear()
      .domain([0, extent(view.points_before, (p) => p[1])[1] ?? 1])
      .nice();
    const pack = (pts: [number, number][], y: (v: number) => number, slope: number, ic: number) => {
      const v = new Float64Array(n * 2 + 4);
      pts.forEach(([px, py], i) => {
        v[i * 2] = ux(px);
        v[i * 2 + 1] = y(py);
      });
      const [d0, d1] = ux.domain() as [number, number];
      v.set([ux(d0), y(ic + slope * d0), ux(d1), y(ic + slope * d1)], n * 2);
      return v;
    };
    return {
      target: pack(view.points_after, uya, slopeA, icA),
      start: pack(view.points_before, uyb, slopeB, icB),
    };
  }, [view.points_before, view.points_after, n, slopeA, icA, slopeB, icB]);

  const draw = useCallback(
    (u: Float64Array) => {
      if (!geo) return;
      const { x0, x1, y0, y1 } = geo.a;
      const v = new Float64Array(u.length);
      for (let i = 0; i < u.length; i += 2) {
        v[i] = x0 + u[i]! * (x1 - x0);
        v[i + 1] = y1 - u[i + 1]! * (y1 - y0);
      }
      pathRef.current?.setAttribute("d", dotPath(v, n, r));
      const l = lineRef.current;
      if (l) {
        l.setAttribute("x1", v[n * 2]!.toFixed(1));
        l.setAttribute("y1", v[n * 2 + 1]!.toFixed(1));
        l.setAttribute("x2", v[n * 2 + 2]!.toFixed(1));
        l.setAttribute("y2", v[n * 2 + 3]!.toFixed(1));
      }
    },
    [geo, n, r],
  );

  useMorph(geo ? unit.target : null, draw, evidence ? null : unit.start);

  const afterKey = `${view.y_label_after}|${geo?.ya.domain().join(",")}`;
  const lineB = geo
    ? (() => {
        const [d0, d1] = geo.xb.domain() as [number, number];
        return { x1: geo.xb(d0), y1: geo.yb(icB + slopeB * d0), x2: geo.xb(d1), y2: geo.yb(icB + slopeB * d1) };
      })()
    : null;

  return (
    <div ref={ref} className={s.fill} data-view="relationship">
      {geo ? (
        <svg width={w} height={h} className={s.svg} role="img" aria-label={view.title}>
          <defs>
            <clipPath id={`clip-rel-a-${compact}`}>
              <rect x={geo.a.x0} y={geo.a.y0 - 4} width={geo.a.x1 - geo.a.x0} height={geo.a.y1 - geo.a.y0 + 4} />
            </clipPath>
            <clipPath id={`clip-rel-b-${compact}`}>
              <rect x={geo.b.x0} y={geo.b.y0 - 4} width={geo.b.x1 - geo.b.x0} height={geo.b.y1 - geo.b.y0 + 4} />
            </clipPath>
          </defs>

          {!evidence ? (
            <g>
              <text x={geo.b.x0} y={14} className={s.panelKicker}>
                AS RECORDED
              </text>
              <text x={geo.b.x0} y={27} className={s.panelLabel}>
                {view.y_label_before}
              </text>
              <Axis x={geo.xb} y={geo.yb} box={geo.b} compact={compact} />
              <g clipPath={`url(#clip-rel-b-${compact})`}>
                <path d={beforePath} className={s.dotsBefore} />
                {lineB ? <line {...lineB} className={s.fit} /> : null}
              </g>
              {!compact ? (
                <text x={geo.b.x1} y={27} textAnchor="end" className={s.rSmall}>
                  r {fmtR(view.r_before)}
                </text>
              ) : null}
              <text x={(geo.b.x0 + geo.b.x1) / 2} y={h - 2} textAnchor="middle" className={s.axisTitle}>
                {view.x_label}
              </text>
            </g>
          ) : null}

          <g>
            {!evidence ? (
              <text x={geo.a.x0} y={14} className={s.panelKickerNow}>
                AFTER
              </text>
            ) : null}
            <AnimatePresence initial={false}>
              <motion.g
                key={afterKey}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0 } }}
                transition={t.arrive}
              >
                <text x={geo.a.x0} y={evidence ? 14 : 27} className={s.panelLabel}>
                  {view.y_label_after}
                </text>
                <Axis x={geo.xa} y={geo.ya} box={geo.a} compact={compact} />
              </motion.g>
            </AnimatePresence>
            <g clipPath={`url(#clip-rel-a-${compact})`}>
              <path ref={pathRef} className={evidence ? s.dotsBefore : s.dotsAfter} />
              <line ref={lineRef} className={s.fit} />
            </g>
            <text x={(geo.a.x0 + geo.a.x1) / 2} y={h - 2} textAnchor="middle" className={s.axisTitle}>
              {view.x_label}
            </text>
          </g>
        </svg>
      ) : null}
      {geo ? (
        <div
          className={compact ? s.rBadgeCompact : s.rBadge}
          style={{ left: geo.a.x1, top: compact ? 0 : -2 }}
          aria-label={`correlation with ${view.x_label}: ${fmtR(view.r_after)}`}
        >
          <span className={s.rLetter}>r</span>
          <Tween value={view.r_after ?? 0} format={fmtR} className={s.rValue} />
        </div>
      ) : null}
    </div>
  );
}
