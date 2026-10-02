/**
 * distribution — one column's values at whichever real state the player shows.
 *
 * Bars keep their identity (bin i) across states. Within one unit they glide: rows leaving under
 * an exclusion rule shrink the bars and the leaving part is hatched, a residual moves each bin to
 * where its values went. Across a change of unit (g → g per kcal, g → kcal) the histograms
 * crossfade: a bar half-way between two units is not a value of anything. A long empty tail is
 * clipped at the 99.5th percentile and the rows past it are counted in a note.
 */
import { useCallback, useId, useLayoutEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import type { DistributionView, Mark } from "../../../api/m1-stage-types";
import { useTransitions } from "../../../motion/prefs";
import { CoachLayer } from "../coach/CoachLayer";
import { bandHeight, notesFor, type Span } from "../coach/place";
import { clean, fmtInt, fmtTick } from "../format";
import {
  clipTail,
  extentOfHist,
  localPos,
  unitRuns,
  type Clip,
  type DistributionState,
  type Track,
} from "../tracks";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../usePlayer";
import { barPath, lerpArray, useSize } from "./geometry";
import { useBlend, type Snapshot } from "./useBlend";
import s from "./views.module.css";

interface Props {
  track: Track<DistributionView>;
  globalLast: number;
  compact?: boolean;
}

const GROUP_NAME: Record<string, string> = { female: "women", male: "men" };

function markClass(m: Mark) {
  if (m.group === "female") return s.markA;
  if (m.group === "male") return s.markB;
  return s.markAll;
}

function XTicks({
  x,
  box,
  count,
}: {
  x: ScaleLinear<number, number>;
  box: { x0: number; x1: number; y1: number };
  count: number;
}) {
  const [lo, hi] = x.domain() as [number, number];
  const ticks = x.ticks(count).filter((v) => v >= lo && v <= hi);
  return (
    <g className={s.axis}>
      <line x1={box.x0} x2={box.x1} y1={box.y1} y2={box.y1} className={s.baseline} />
      {ticks.map((v) => (
        <text key={v} x={x(v)} y={box.y1 + 14} textAnchor="middle">
          {fmtTick(v)}
        </text>
      ))}
    </g>
  );
}

interface RunGeo {
  x: ScaleLinear<number, number>;
  y: ScaleLinear<number, number>;
}

export function Distribution({ track, globalLast, compact = false }: Props) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const t = useTransitions();
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const uid = useId().replace(/:/g, "");
  const view = track.view;
  const { states } = track;
  const localLast = states.length - 1;
  const marks = view.marks;

  const extents = useMemo(() => states.map((st) => extentOfHist(st.unit, st.hist)), [states]);
  const runs = useMemo(() => unitRuns(extents), [extents]);
  const clips = useMemo<Clip[]>(
    () =>
      states.map((st, i) =>
        clipTail(st.hist, runs[i] === runs[0] ? marks.map((m) => m.value) : []),
      ),
    [states, runs, marks],
  );
  /** Bins that only lose rows, state to state, are drawn over the first state, hatched. */
  const overlay = useMemo(() => {
    const e0 = states[0]!.hist.edges;
    return states.map(
      (st, i) =>
        runs[i] === runs[0] &&
        st.hist.edges.length === e0.length &&
        st.hist.edges.every((e, k) => Math.abs(e - e0[k]!) < 1e-9),
    );
  }, [states, runs]);

  // The coach's band sits above the picture (M2_CONTRACT §6); a narrow thumbnail draws no notes.
  const notes = useMemo(() => notesFor(view.coach, compact, w), [view.coach, compact, w]);
  // A thumbnail wide enough to carry notes is laid out like a primary card (room for its labels).
  const small = compact && !notes.length;
  const band = bandHeight(notes.length);
  const geo = useMemo(() => {
    if (w < 80 || h < 60) return null;
    const pad = small
      ? { l: 10, r: 10, t: 34, b: 22 }
      : { l: 14, r: 14, t: band + (marks.some((m) => m.group) ? 58 : 44), b: 24 };
    const box = { x0: pad.l, x1: w - pad.r, y0: pad.t, y1: h - pad.b };
    const byRun = new Map<number, RunGeo>();
    for (const run of new Set(runs)) {
      let lo = Infinity;
      let hi = -Infinity;
      let top = 1;
      states.forEach((st, i) => {
        if (runs[i] !== run) return;
        lo = Math.min(lo, clean(st.hist.edges[0] ?? 0));
        hi = Math.max(hi, clips[i]!.hi);
        for (let k = 0; k < clips[i]!.bins; k++) top = Math.max(top, st.hist.counts[k] ?? 0);
      });
      byRun.set(run, {
        x: scaleLinear().domain([lo, hi]).range([box.x0, box.x1]),
        y: scaleLinear().domain([0, top]).range([box.y1, box.y0]),
      });
    }
    return { box, byRun };
  }, [w, h, small, states, runs, clips, marks, band]);

  /** Each state in pixels: four numbers per bar, then the hatch over the first state. */
  const pix = useMemo(() => {
    if (!geo) return null;
    const base0 = states[0]!.hist.counts;
    return states.map((st, i) => {
      const { x, y } = geo.byRun.get(runs[i]!)!;
      const nb = st.hist.counts.length;
      const v = new Float64Array(nb * 8);
      for (let k = 0; k < nb; k++) {
        const c = st.hist.counts[k] ?? 0;
        const x0 = x(st.hist.edges[k]!) + 0.5;
        const x1 = x(st.hist.edges[k + 1]!) - 0.5;
        const top = c > 0 ? Math.min(y(c), geo.box.y1 - 1.5) : geo.box.y1;
        v.set([x0, x1, top, geo.box.y1], k * 4);
        const gone = overlay[i] && i > 0 ? Math.max(0, (base0[k] ?? 0) - c) : 0;
        const goneTop = gone > 0 ? Math.min(y(c + gone), top - 1) : top;
        v.set([x0, x1, goneTop, top], nb * 4 + k * 4);
      }
      return v;
    });
  }, [geo, states, runs, overlay]);

  const barsA = useRef<SVGPathElement>(null);
  const barsB = useRef<SVGPathElement>(null);
  const hatchA = useRef<SVGPathElement>(null);
  const ghost = useRef<SVGPathElement>(null);
  const drawn = useRef<Snapshot | null>(null);
  const posRef = useRef(0);
  const redrawRef = useRef<() => void>(() => {});
  const blend = useBlend(() => redrawRef.current());

  const lastTrack = useRef(track);
  useLayoutEffect(() => {
    if (lastTrack.current === track) return;
    lastTrack.current = track;
    const lp = Math.round(Math.min(localLast, localPos(posRef.current, globalLast, localLast)));
    const sameShape = drawn.current?.values.length === states[lp]!.hist.counts.length * 8;
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
      let a: Float64Array = pix[i]!;
      let aOp = 1;
      let b: Float64Array | null = null;
      let bOp = 0;
      if (f > 0) {
        const next = pix[i + 1]!;
        if (runs[i] === runs[i + 1] && next.length === a.length) a = lerpArray(a, next, f);
        else {
          b = next;
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
      const half = (v: Float64Array) => v.length / 8;
      if (ghost.current) {
        ghost.current.setAttribute("d", bl && ghostOp > 0 ? barPath(bl.from.values, half(bl.from.values)) : "");
        ghost.current.style.opacity = String(ghostOp);
      }
      if (barsA.current) {
        barsA.current.setAttribute("d", aOp > 0 ? barPath(a, half(a)) : "");
        barsA.current.style.opacity = String(aOp);
      }
      if (hatchA.current) {
        hatchA.current.setAttribute("d", aOp > 0 ? barPath(a.subarray(a.length / 2), half(a)) : "");
        hatchA.current.style.opacity = String(aOp);
      }
      if (barsB.current) {
        barsB.current.setAttribute("d", b && bOp > 0 ? barPath(b, half(b)) : "");
        barsB.current.style.opacity = String(bOp);
      }
      drawn.current = { values: (b && f >= 0.5 ? b : a).slice(), extent: extents[f >= 0.5 ? i + 1 : i]! };
    },
    [pix, localLast, globalLast, runs, blend, extents],
  );
  useLayoutEffect(() => {
    redrawRef.current = () => draw(posRef.current);
  }, [draw]);
  usePlayerFrame(store, draw);

  // Text names the real state the drawing is nearest to; the drawing itself is motion.
  const shown = Math.min(localLast, Math.round(localPos(ui.nearest, globalLast, localLast)));
  const state: DistributionState = states[shown]!;

  const run = runs[shown]!;
  const rg = geo?.byRun.get(run);
  const clip = clips[shown]!;
  const showMarks = run === runs[0];
  // A note about a stretch of the axis brackets it under the bars; one about the column points at
  // the picture as a whole. Ranges are on the first state's axis (the cut is drawn there).
  const axis = geo?.byRun.get(runs[0]!);
  const spans: (Span | null)[] = notes.map((n) => {
    if (!geo || !axis) return null;
    const { box } = geo;
    if (n.anchor.kind === "range" && Array.isArray(n.anchor.ref) && n.anchor.ref.length >= 2) {
      const [lo, hi] = n.anchor.ref as number[];
      const x0 = Math.max(box.x0, Math.min(box.x1, axis.x(Math.max(lo!, axis.x.domain()[0]!))));
      const x1 = Math.max(box.x0, Math.min(box.x1, axis.x(Math.min(hi!, axis.x.domain()[1]!))));
      return { x0, x1, y: box.y1 + 1, mark: "bracket" };
    }
    return null;
  });

  return (
    <div className={s.distWrap} data-view="distribution" data-state={shown}>
      <div ref={ref} className={s.fill}>
        {geo && rg ? (
          <svg width={w} height={h} className={s.svg} role="img" aria-label={`${view.title}: ${state.label}`}>
            <defs>
              <pattern
                id={`hatch-${uid}`}
                width="5"
                height="5"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <line x1="0" y1="0" x2="0" y2="5" className={s.hatchLine} />
              </pattern>
              <clipPath id={`clip-${uid}`}>
                <rect x={geo.box.x0} y={0} width={geo.box.x1 - geo.box.x0} height={h} />
              </clipPath>
            </defs>
            <AnimatePresence initial={false} mode="popLayout">
              <motion.text
                key={state.label}
                x={geo.box.x0}
                y={(small ? 12 : 15) + band}
                className={shown === 0 ? s.stackLabelNow : s.stackLabelWith}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0.12 } }}
                transition={t.arrive}
              >
                {state.label}
              </motion.text>
            </AnimatePresence>
            <g clipPath={`url(#clip-${uid})`}>
              <path ref={ghost} className={s.barWith} />
              <path ref={barsA} className={shown === 0 ? s.barNow : s.barWith} />
              <path ref={hatchA} fill={`url(#hatch-${uid})`} className={s.removed} />
              <path ref={barsB} className={s.barWith} />
            </g>
            <AnimatePresence initial={false}>
              <motion.g
                key={`${run}|${rg.x.domain().join(",")}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={t.arrive}
              >
                <XTicks x={rg.x} box={geo.box} count={small ? 4 : 7} />
              </motion.g>
            </AnimatePresence>
            {showMarks
              ? marks.map((m) => {
                  const row = m.group === "male" ? 1 : 0;
                  const mx = rg.x(m.value);
                  if (mx < geo.box.x0 - 1 || mx > geo.box.x1 + 1) return null;
                  const name = m.group ? `${GROUP_NAME[m.group] ?? m.group} ` : "";
                  const label = small ? `${name}${fmtInt(m.value)}` : `${name}${m.label}`;
                  const ly = (small ? 22 : 34) + band + row * (small ? 10 : 13);
                  return (
                    <g key={`${m.group}|${m.value}`} className={markClass(m)}>
                      <line x1={mx} x2={mx} y1={ly + 3} y2={geo.box.y1} className={s.markLine} />
                      <text x={mx + 3} y={ly} className={s.markText}>
                        {label}
                      </text>
                    </g>
                  );
                })
              : null}
          </svg>
        ) : null}
        {notes.length && geo ? <CoachLayer notes={notes} spans={spans} width={w} /> : null}
      </div>
      {clip.over > 0 && !small ? (
        <p className={s.overflow}>
          Not drawn: {fmtInt(clip.over)} {clip.over === 1 ? "row" : "rows"} from {fmtTick(clip.hi)} to{" "}
          {fmtTick(clip.max)}, past the 99.5th percentile.
        </p>
      ) : null}
    </div>
  );
}
