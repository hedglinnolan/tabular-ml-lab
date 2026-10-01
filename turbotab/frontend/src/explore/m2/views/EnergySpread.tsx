/**
 * Energy per row, before and after the reshape — the measurement-error argument made visible:
 * the mean of two recalls is a narrower distribution than single days (and the under-reported
 * days fold into plausible means), while keeping one day keeps all of its day-to-day error.
 *
 * Bars morph between real states on one axis (the unit is the same); a change score is another
 * unit, so it crossfades instead. The coach's notes (M2_CONTRACT §6: ≤ 2, ≤ 12 words, amber) point
 * at the picture — a range on the axis, the spread — and never pre-select anything.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { scaleLinear } from "d3-scale";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { barPath, useSize } from "../../../components/stage/views/geometry";
import { fmtInt } from "../data";
import type { CoachNote, EnergyView } from "../types";
import { useMorph } from "../useMorph";
import { mix } from "./canvas";
import s from "./views.module.css";

const PLOT_H = 96;
const BAND = 48; // the coach's band above the plot: two notes, one line each
const AXIS = 20;
const NOTE_H = 22;
const WHISKER_Y = BAND + 3;

interface Shape {
  bars: Float64Array; // [x0, x1, top, base] per bar
  mean: number;
  sd: number;
  alphaNow: number;
  alphaWith: number;
}

function lerpShape(a: Shape, b: Shape, t: number): Shape {
  const bars = new Float64Array(a.bars.length);
  for (let i = 0; i < bars.length; i++) bars[i] = mix(a.bars[i]!, b.bars[i] ?? a.bars[i]!, t);
  return {
    bars,
    mean: mix(a.mean, b.mean, t),
    sd: mix(a.sd, b.sd, t),
    alphaNow: mix(a.alphaNow, b.alphaNow, t),
    alphaWith: mix(a.alphaWith, b.alphaWith, t),
  };
}

export function EnergySpread({
  view,
  notes,
  last,
  methodKey,
}: {
  view: EnergyView;
  notes: CoachNote[];
  last: number;
  methodKey: string;
}) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const [ref, { w }] = useSize<HTMLDivElement>();
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));
  const after = at >= 2;
  const width = Math.max(120, w);

  const xBefore = useMemo(
    () => scaleLinear().domain([view.before_edges[0]!, view.before_edges[view.before_edges.length - 1]!]).range([4, width - 4]),
    [view.before_edges, width],
  );
  const xAfter = useMemo(
    () => scaleLinear().domain([view.edges[0]!, view.edges[view.edges.length - 1]!]).range([4, width - 4]),
    [view.edges, width],
  );
  const share = (c: number, n: number) => (100 * c) / Math.max(1, n);
  const yMax = useMemo(() => {
    const m = Math.max(
      ...view.before.counts.map((c) => share(c, view.before.n)),
      ...view.after.counts.map((c) => share(c, view.after.n)),
    );
    return Math.max(1, m * 1.08);
  }, [view]);
  const y = useMemo(() => scaleLinear().domain([0, yMax]).range([BAND + PLOT_H, BAND]), [yMax]);

  const shapes = useMemo(() => {
    const bars = (counts: number[], n: number, edges: number[], x: (v: number) => number) => {
      const out = new Float64Array(counts.length * 4);
      counts.forEach((c, i) => {
        out[i * 4] = x(edges[i]!) + 0.6;
        out[i * 4 + 1] = x(edges[i + 1]!) - 0.6;
        out[i * 4 + 2] = y(share(c, n));
        out[i * 4 + 3] = BAND + PLOT_H;
      });
      return out;
    };
    const b: Shape = {
      bars: bars(view.before.counts, view.before.n, view.before_edges, xBefore),
      mean: xBefore(view.before.mean),
      sd: xBefore(view.before.mean + view.before.sd) - xBefore(view.before.mean),
      alphaNow: 1,
      alphaWith: 0,
    };
    const a: Shape = {
      bars: view.unit_changes ? b.bars : bars(view.after.counts, view.after.n, view.edges, xAfter),
      mean: view.unit_changes ? b.mean : xAfter(view.after.mean),
      sd: view.unit_changes ? b.sd : xAfter(view.after.mean + view.after.sd) - xAfter(view.after.mean),
      alphaNow: 0,
      alphaWith: 1,
    };
    return [b, b, a, a];
  }, [view, xBefore, xAfter, y]);

  const nowPath = useRef<SVGPathElement>(null);
  const withPath = useRef<SVGPathElement>(null);
  const changePath = useRef<SVGPathElement>(null);
  const whisker = useRef<SVGGElement>(null);
  const coachG = useRef<SVGGElement>(null);
  const changeBars = useMemo(() => {
    if (!view.unit_changes) return "";
    const out = new Float64Array(view.after.counts.length * 4);
    view.after.counts.forEach((c, i) => {
      out[i * 4] = xAfter(view.edges[i]!) + 0.6;
      out[i * 4 + 1] = xAfter(view.edges[i + 1]!) - 0.6;
      out[i * 4 + 2] = y(share(c, view.after.n));
      out[i * 4 + 3] = BAND + PLOT_H;
    });
    return barPath(out, view.after.counts.length);
  }, [view, xAfter, y]);

  const paint = useCallback(
    (sh: Shape) => {
      const n = sh.bars.length / 4;
      if (nowPath.current) {
        nowPath.current.setAttribute("d", barPath(sh.bars, n));
        // The bars are "now" gray until the combination, then "with this choice" teal.
        nowPath.current.style.opacity = view.unit_changes ? String(sh.alphaNow) : String(0.15 + 0.85 * sh.alphaNow);
      }
      if (withPath.current) {
        withPath.current.setAttribute("d", view.unit_changes ? "" : barPath(sh.bars, n));
        withPath.current.style.opacity = String(sh.alphaWith);
      }
      if (changePath.current) changePath.current.style.opacity = String(sh.alphaWith);
      if (whisker.current) {
        whisker.current.setAttribute("transform", `translate(${sh.mean.toFixed(1)},0)`);
        const line = whisker.current.querySelector("line[data-span]");
        line?.setAttribute("x1", (-sh.sd).toFixed(1));
        line?.setAttribute("x2", sh.sd.toFixed(1));
        const ends = whisker.current.querySelectorAll("line[data-end]");
        ends[0]?.setAttribute("x1", (-sh.sd).toFixed(1));
        ends[0]?.setAttribute("x2", (-sh.sd).toFixed(1));
        ends[1]?.setAttribute("x1", sh.sd.toFixed(1));
        ends[1]?.setAttribute("x2", sh.sd.toFixed(1));
        whisker.current.style.opacity = view.unit_changes ? String(sh.alphaNow) : "1";
      }
      if (coachG.current) coachG.current.style.opacity = String(sh.alphaWith);
    },
    [view],
  );
  const target = useCallback(
    (pos: number) => {
      const p = (pos * 3) / Math.max(1, last);
      const i = Math.max(0, Math.min(3, Math.floor(p)));
      return i >= 3 ? shapes[3]! : lerpShape(shapes[i]!, shapes[i + 1]!, p - i);
    },
    [shapes, last],
  );
  const redrawRef = useRef<() => void>(() => {});
  const blend = useMorph<Shape>(
    lerpShape,
    useCallback(() => redrawRef.current(), []),
  );
  const draw = useCallback(
    (pos: number) => paint(blend(target(pos), methodKey)),
    [paint, blend, target, methodKey],
  );
  useEffect(() => {
    redrawRef.current = () => draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  // Where each note's anchor sits, on the after axis; each note gets a line of the band and a
  // straight leader down to what it is about.
  const anchors = notes.map((nt, i) => {
    const ref = nt.anchor.ref as number[];
    const top = i * NOTE_H + NOTE_H - 3;
    if (nt.anchor.kind === "range") {
      if (ref[0]! > 0) {
        // The spread: the after state's ±1 SD, which the whisker draws.
        return { kind: "spread" as const, x0: xAfter(ref[0]!), x1: xAfter(ref[1]!), lx: xAfter(ref[1]!), top, bottom: WHISKER_Y - 3 };
      }
      const x0 = xAfter(Math.max(ref[0]!, view.edges[0]!));
      const x1 = xAfter(ref[1]!);
      return { kind: "range" as const, x0, x1, lx: (x0 + x1) / 2, top, bottom: BAND + PLOT_H + 4 };
    }
    const x = xAfter(ref[0]!);
    return { kind: "point" as const, x0: x, x1: x, lx: x, top, bottom: BAND + 6 };
  });
  const ticks = (view.unit_changes && after ? xAfter : xBefore).ticks(width < 420 ? 4 : 6);
  const xT = view.unit_changes && after ? xAfter : xBefore;

  return (
    <div className={s.spread} ref={ref} data-view="distribution" data-state={at}>
      {w > 0 ? (
        <svg width={width} height={BAND + PLOT_H + AXIS} className={s.svg} role="img"
          aria-label={`${view.before_label} against ${view.after_label}`}>
          <line x1={4} x2={width - 4} y1={BAND + PLOT_H + 0.5} y2={BAND + PLOT_H + 0.5} className={s.axisLine} />
          <path ref={nowPath} className={s.barNow} />
          <path ref={withPath} className={s.barWith} />
          {view.unit_changes ? <path ref={changePath} d={changeBars} className={s.barWith} style={{ opacity: 0 }} /> : null}
          <g ref={whisker} className={s.whisker}>
            <line data-span y1={WHISKER_Y} y2={WHISKER_Y} />
            <line data-end y1={WHISKER_Y - 3} y2={WHISKER_Y + 3} />
            <line data-end y1={WHISKER_Y - 3} y2={WHISKER_Y + 3} />
          </g>
          {ticks.map((t) => (
            <text key={t} x={xT(t)} y={BAND + PLOT_H + 14} textAnchor="middle" className={s.tick}>
              {fmtInt(t).replace("-", "−")}
            </text>
          ))}
          <g ref={coachG} className={s.coachMarks} style={{ opacity: 0 }}>
            {anchors.map((a, i) => (
              <g key={i}>
                {a.kind === "range" ? (
                  <path d={`M${a.x0 + 0.5},${a.bottom - 3}V${a.bottom + 1}H${a.x1 - 0.5}V${a.bottom - 3}`} className={s.coachBracket} />
                ) : a.kind === "point" ? (
                  <line x1={a.x0} x2={a.x0} y1={BAND + 4} y2={BAND + PLOT_H} className={s.coachBracket} />
                ) : (
                  <circle cx={a.lx} cy={WHISKER_Y} r={2.4} className={s.coachDot} />
                )}
                <line x1={a.lx} x2={a.lx} y1={a.top} y2={a.kind === "range" ? a.bottom + 1 : a.bottom} className={s.coachLeader} />
              </g>
            ))}
          </g>
        </svg>
      ) : null}
      {notes.map((nt, i) => {
        const a = anchors[i]!;
        // A note sits on its own line of the band, starting just left of its leader and pulled
        // back from the right edge so it stays inside the card (its width estimated from its text).
        return (
          <p
            key={nt.text}
            className={s.coachNote}
            data-shown={after || undefined}
            data-testid="coach-note"
            style={{
              top: i * NOTE_H,
              left: Math.max(0, Math.min(a.lx - 14, width - nt.text.length * 6.4 - 18)),
            }}
          >
            {nt.text}
          </p>
        );
      })}
      <div className={s.legendGrid}>
        <span className={s.keyNow} />
        <span>
          {view.before_label} · n {fmtInt(view.before.n)} · SD {fmtInt(view.before.sd)}
        </span>
        <span className={s.legendAxis}>% of rows · kcal</span>
        <span className={s.keyWith} />
        <span>
          {view.after_label} · n {fmtInt(view.after.n)}
          {view.unit_changes ? "" : ` · SD ${fmtInt(view.after.sd)}`}
        </span>
      </div>
    </div>
  );
}
