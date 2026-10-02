/**
 * relationship — a nutrient against energy, at whichever real state the player shows.
 *
 * The sampled rows are the same rows in every state, so they are what moves: in the residual
 * storyboard the fitted line draws through the cloud, the points drop to their residuals, and the
 * cloud re-centers on the average. Within one unit the points glide on one shared axis, so a
 * movement is a real change of value; across a change of unit (g → g per kcal) the clouds
 * crossfade and the axis changes with them. Text (the axis label, r) always names a real state.
 */
import { useCallback, useId, useLayoutEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import type { RelationshipView } from "../../../api/m1-stage-types";
import { useTransitions } from "../../../motion/prefs";
import { CoachLayer } from "../coach/CoachLayer";
import { bandHeight, notesFor, type Span } from "../coach/place";
import { fmtR, fmtTick } from "../format";
import { extentOfPoints, localPos, unitRuns, type Track } from "../tracks";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../usePlayer";
import { dotPath, lerp, lerpArray, ols, useSize } from "./geometry";
import { useBlend, type Snapshot } from "./useBlend";
import s from "./views.module.css";

interface Props {
  track: Track<RelationshipView>;
  globalLast: number;
  compact?: boolean;
}

const PAD = { l: 52, r: 16, t: 34, b: 36 };
const PAD_C = { l: 40, r: 10, t: 26, b: 28 };

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
  const xt = x.ticks(width < 200 ? 2 : compact || width < 360 ? 3 : 6);
  const yt = y.ticks(compact ? 3 : 5);
  return (
    <g className={s.axis}>
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.x0} x2={box.x1} y1={y(t)} y2={y(t)} className={t === 0 ? s.zero : s.grid} />
          <text x={box.x0 - 7} y={y(t)} dy="0.32em" textAnchor="end">
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

export function Relationship({ track, globalLast, compact = false }: Props) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const t = useTransitions();
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const clip = `rel-${useId().replace(/:/g, "")}`;
  const { states } = track;
  const localLast = states.length - 1;
  const r = compact ? 1.6 : 2.2;

  const extents = useMemo(() => states.map((st) => extentOfPoints(st.yLabel, st.points)), [states]);
  const runs = useMemo(() => unitRuns(extents), [extents]);

  // The coach's band sits above the picture (M2_CONTRACT §6); a narrow thumbnail draws no notes.
  const notes = useMemo(() => notesFor(track.view.coach, compact, w), [track.view.coach, compact, w]);
  const band = bandHeight(notes.length);
  const geo = useMemo(() => {
    if (w < 80 || h < 80 + band) return null;
    const pad = compact ? PAD_C : PAD;
    const box = { x0: pad.l, x1: w - pad.r, y0: pad.t + band, y1: h - pad.b };
    let xLo = 0;
    let xHi = 0;
    const lo = new Map<number, number>();
    const hi = new Map<number, number>();
    states.forEach((st, i) => {
      const run = runs[i]!;
      for (const [px, py] of st.points) {
        if (px < xLo) xLo = px;
        if (px > xHi) xHi = px;
        if (py < (lo.get(run) ?? 0)) lo.set(run, py);
        if (py > (hi.get(run) ?? -Infinity)) hi.set(run, py);
      }
    });
    const x = scaleLinear().domain([xLo, xHi || 1]).nice().range([box.x0, box.x1]);
    const ys = new Map<number, ScaleLinear<number, number>>();
    for (const run of new Set(runs)) {
      const top = hi.get(run) ?? 1;
      ys.set(
        run,
        scaleLinear()
          .domain([Math.min(0, lo.get(run) ?? 0), top > 0 ? top : 1])
          .nice()
          .range([box.y1, box.y0]),
      );
    }
    return { box, x, ys };
  }, [w, h, compact, states, runs, band]);

  /** Each state in pixels: its points, then the two ends of its line. */
  const pix = useMemo(() => {
    if (!geo) return null;
    const [d0, d1] = geo.x.domain() as [number, number];
    return states.map((st, i) => {
      const y = geo.ys.get(runs[i]!)!;
      const v = new Float64Array(st.points.length * 2 + 4);
      st.points.forEach(([px, py], k) => {
        v[k * 2] = geo.x(px);
        v[k * 2 + 1] = y(py);
      });
      const [slope, ic] = st.fit ? [st.fit.slope, st.fit.intercept] : ols(st.points);
      v.set([geo.x(d0), y(ic + slope * d0), geo.x(d1), y(ic + slope * d1)], st.points.length * 2);
      return v;
    });
  }, [geo, states, runs]);

  const pathA = useRef<SVGPathElement>(null);
  const pathB = useRef<SVGPathElement>(null);
  const ghost = useRef<SVGPathElement>(null);
  const lineA = useRef<SVGLineElement>(null);
  const lineB = useRef<SVGLineElement>(null);
  const drawn = useRef<Snapshot | null>(null);
  const posRef = useRef(0);
  const redrawRef = useRef<() => void>(() => {});
  const blend = useBlend(() => redrawRef.current());

  // A new option: blend from what is on screen (before the first draw with the new data).
  const lastTrack = useRef(track);
  useLayoutEffect(() => {
    if (lastTrack.current === track) return;
    lastTrack.current = track;
    const lp = Math.round(Math.min(localLast, localPos(posRef.current, globalLast, localLast)));
    const sameShape = drawn.current?.values.length === states[lp]!.points.length * 2 + 4;
    blend.start(drawn.current, extents[lp]!, sameShape);
  }, [track, localLast, globalLast, states, extents, blend]);
  useLayoutEffect(() => blend.cancel(), [w, h, blend]);

  const draw = useCallback(
    (p: number) => {
      posRef.current = p;
      if (!pix) return;
      const lp = Math.max(0, Math.min(localLast, localPos(p, globalLast, localLast)));
      let i = Math.floor(lp);
      let f = lp - i;
      if (i >= localLast) {
        i = localLast;
        f = 0;
      }
      const n = states[i]!.points.length;
      let a: Float64Array = pix[i]!;
      let aOp = 1;
      let b: Float64Array | null = null;
      let bOp = 0;
      let nb = n;
      if (f > 0) {
        const next = pix[i + 1]!;
        if (runs[i] === runs[i + 1] && next.length === a.length) a = lerpArray(a, next, f);
        else {
          b = next;
          nb = states[i + 1]!.points.length;
          aOp = 1 - f;
          bOp = f;
        }
      }
      const bl = blend.current();
      let ghostOp = 0;
      if (bl) {
        if (bl.mode === "morph" && bl.from.values.length === a.length && !b) {
          a = lerpArray(bl.from.values, a, bl.m);
        } else {
          ghostOp = 1 - bl.m;
          aOp *= bl.m;
          bOp *= bl.m;
        }
      }
      const g = ghost.current;
      if (g) {
        if (bl && ghostOp > 0) {
          const fn = (bl.from.values.length - 4) / 2;
          g.setAttribute("d", dotPath(bl.from.values, fn, r));
          g.style.opacity = String(ghostOp);
        } else g.style.opacity = "0";
      }
      const setLayer = (
        path: SVGPathElement | null,
        line: SVGLineElement | null,
        v: Float64Array | null,
        count: number,
        op: number,
      ) => {
        if (path) {
          path.setAttribute("d", v && op > 0 ? dotPath(v, count, r) : "");
          path.style.opacity = String(op);
        }
        if (line) {
          if (v && op > 0) {
            line.setAttribute("x1", v[count * 2]!.toFixed(1));
            line.setAttribute("y1", v[count * 2 + 1]!.toFixed(1));
            line.setAttribute("x2", v[count * 2 + 2]!.toFixed(1));
            line.setAttribute("y2", v[count * 2 + 3]!.toFixed(1));
          }
          line.style.opacity = String(op);
        }
      };
      setLayer(pathA.current, lineA.current, a, n, aOp);
      setLayer(pathB.current, lineB.current, b, nb, bOp);
      // The storyboard's fitted line draws itself as the player moves into its step.
      const emph = (k: number) => (states[k]?.fit ? 1 : 0);
      const e = f > 0 ? lerp(emph(i), emph(i + 1), f) : emph(i);
      for (const line of [lineA.current, lineB.current]) {
        if (!line) continue;
        line.style.strokeDashoffset = e > 0 && e < 1 ? String(1 - e) : "0";
        line.dataset.emph = e >= 1 ? "1" : e > 0 ? "drawing" : "0";
      }
      drawn.current = { values: (b && f >= 0.5 ? b : a).slice(), extent: extents[f >= 0.5 ? i + 1 : i]! };
    },
    [pix, localLast, globalLast, states, runs, blend, extents, r],
  );
  useLayoutEffect(() => {
    redrawRef.current = () => draw(posRef.current);
  }, [draw]);
  usePlayerFrame(store, draw);

  // Text names the real state the drawing is nearest to; the drawing itself is motion.
  const shown = Math.min(localLast, Math.round(localPos(ui.nearest, globalLast, localLast)));
  const state = states[shown]!;
  const run = runs[shown]!;
  const y = geo?.ys.get(run);
  const view = track.view;
  // A note about a column points at its axis; a range brackets the x axis; points ring their middle.
  const spans: (Span | null)[] = notes.map((n) => {
    if (!geo || !y) return null;
    const { box } = geo;
    const ref = n.anchor.ref;
    if (n.anchor.kind === "range" && Array.isArray(ref) && ref.length >= 2) {
      const [lo, hi] = geo.x.domain() as [number, number];
      const x0 = geo.x(Math.max(lo, Math.min(hi, Number(ref[0]))));
      const x1 = geo.x(Math.max(lo, Math.min(hi, Number(ref[1]))));
      return { x0, x1, y: box.y1 + 1, mark: "bracket" };
    }
    if (n.anchor.kind === "column") {
      if (ref === view.x_label) return null;
      return { x0: box.x0, x1: box.x0, y: box.y0 + 4, mark: "tick" };
    }
    if (n.anchor.kind === "points" && Array.isArray(ref) && ref.length) {
      const pts = ref.map((i) => state.points[Number(i)]).filter((p): p is [number, number] => !!p);
      if (!pts.length) return null;
      const cx = pts.reduce((a, p) => a + geo.x(p[0]), 0) / pts.length;
      const cy = pts.reduce((a, p) => a + y(p[1]), 0) / pts.length;
      return { x0: cx, x1: cx, y: cy, mark: "ring" };
    }
    return null;
  });

  return (
    <div ref={ref} className={s.fill} data-view="relationship" data-state={shown}>
      {geo && y ? (
        <svg
          width={w}
          height={h}
          className={s.svg}
          role="img"
          aria-label={`${view.y_label_before} against ${view.x_label}: ${state.label}, r ${fmtR(state.r)}`}
        >
          <defs>
            <clipPath id={clip}>
              <rect
                x={geo.box.x0 - 3}
                y={geo.box.y0 - 6}
                width={geo.box.x1 - geo.box.x0 + 6}
                height={geo.box.y1 - geo.box.y0 + 6}
              />
            </clipPath>
          </defs>
          <AnimatePresence initial={false}>
            <motion.g
              key={`${run}|${y.domain().join(",")}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={t.arrive}
            >
              <Axis x={geo.x} y={y} box={geo.box} compact={compact} />
            </motion.g>
          </AnimatePresence>
          <AnimatePresence initial={false} mode="popLayout">
            <motion.text
              key={state.yLabel}
              x={geo.box.x0}
              y={(compact ? 12 : 16) + band}
              className={s.panelLabel}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0.12 } }}
              transition={t.arrive}
            >
              {state.yLabel}
            </motion.text>
          </AnimatePresence>
          <g clipPath={`url(#${clip})`}>
            <path ref={ghost} className={s.dotsWith} />
            <path ref={pathA} className={shown === 0 ? s.dotsNow : s.dotsWith} />
            <path ref={pathB} className={s.dotsWith} />
            <line ref={lineA} className={s.fitLine} pathLength={1} />
            <line ref={lineB} className={s.fitLine} pathLength={1} />
          </g>
          <text
            x={(geo.box.x0 + geo.box.x1) / 2}
            y={h - 4}
            textAnchor="middle"
            className={s.axisTitle}
          >
            {view.x_label}
          </text>
        </svg>
      ) : null}
      {notes.length && geo ? <CoachLayer notes={notes} spans={spans} width={w} /> : null}
      {geo ? (
        <div
          className={compact ? s.rBadgeCompact : s.rBadge}
          style={{ left: geo.box.x1, top: band ? band - 4 : undefined }}
          data-testid="r-badge"
        >
          <span className={s.rLetter}>r</span>
          <span className={shown === 0 ? s.rValueNow : s.rValue}>{fmtR(state.r)}</span>
        </div>
      ) : null}
    </div>
  );
}
