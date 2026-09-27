/**
 * relationship — a before/after scatter whose points are the same training rows in every state.
 * x never changes (kcal), so each dot moves only vertically: the row survives, its value changes.
 * The least-squares line of the dots drawn and the sidenote ride together, so the note stays
 * beside exactly the thing that changed. The r in the note is the engine's (all training rows),
 * never a number computed mid-scrub.
 */
import { useCallback, useMemo, useRef } from "react";
import { extent } from "d3-array";
import { scaleLinear } from "d3-scale";
import { Prose } from "../../../components/Prose";
import { fmtNum, fmtR } from "../data";
import { SceneBuilder, cached, unionKeys, type Scene } from "../engine/morph";
import { attrs, place, useMorph, useRegistry } from "../engine/scrub";
import { mixRgb, useTokenColors, useWidth } from "../engine/useSize";
import s from "./views.module.css";

export interface ScatterState {
  ys: number[];
  yLabel: string;
  r: number | null;
  note: string;
  /** false only for "your data now" — the dots wear the before color. */
  after: boolean;
}

interface Props {
  xs: number[];
  xLabel: string;
  states: Record<string, ScatterState>;
  height?: number;
  noteWidth?: number;
  span?: readonly [number, number];
}

const M = { l: 50, r: 16, t: 16, b: 30 };

function ols(xs: Float32Array, ys: Float32Array) {
  let mx = 0;
  let my = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    mx += xs[i]!;
    my += ys[i]!;
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) ** 2;
  }
  const b = sxx ? sxy / sxx : 0;
  return { a: my - b * mx, b };
}

export function ScrubScatter({ xs, xLabel, states, height = 300, noteWidth = 190, span }: Props) {
  const [wrap, width] = useWidth<HTMLDivElement>(720);
  const plotW = Math.max(320, width - noteWidth);
  const iw = plotW - M.l - M.r;
  const ih = height - M.t - M.b;
  const colors = useTokenColors(["--c4", "--c2"]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const fit = useRef<SVGLineElement>(null);
  const { map, reg } = useRegistry<HTMLElement | SVGElement>();

  const x = useMemo(() => {
    const [lo = 0, hi = 1] = extent(xs);
    return scaleLinear().domain([lo, hi]).range([M.l, M.l + iw]);
  }, [xs, iw]);
  const xpx = useMemo(() => Float32Array.from(xs, (v) => x(v)), [xs, x]);
  const [x0, x1] = x.range() as [number, number];

  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? states.now!;
      const [lo = 0, hi = 1] = extent(st.ys);
      // The exact extent, so a pure change of units (grams -> kcal) leaves every dot in place.
      const y = scaleLinear().domain([lo, hi]).range([M.t + ih, M.t]);
      const ypx = Float32Array.from(st.ys, (v) => y(v));
      const { a, b } = ols(xpx, ypx);
      const dk = `${st.yLabel}:${lo}:${hi}`;
      const sb = new SceneBuilder()
        .array("y", ypx)
        .set("pts", { c: st.after ? 1 : 0 })
        .set("fit", { y1: a + b * x0, y2: a + b * x1, o: 1 })
        .set(`yl:${st.yLabel}`, { o: 1, swap: 1 })
        .set(`note:${fmtR(st.r)}|${st.note}`, { y: a + b * x1, o: 1, swap: 1 });
      for (const v of y.ticks(5)) sb.set(`yt:${dk}:${v}`, { y: y(v), o: 1, swap: 1 });
      return sb.scene;
    });
  }, [states, xpx, x0, x1, ih]);

  const keys = useMemo(
    () => unionKeys(Object.keys(states).map((k) => sceneOf(k))),
    [states, sceneOf],
  );

  const apply = useCallback(
    (sc: Scene) => {
      const cv = canvas.current;
      const ys = sc.arrays.get("y");
      const before = colors["--c4"];
      const after = colors["--c2"];
      if (cv && ys && before && after) {
        const dpr = dpr_();
        const ctx = cv.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, plotW, height);
          ctx.fillStyle = mixRgb(before, after, sc.items.get("pts")?.c ?? 0, 0.62);
          ctx.beginPath();
          for (let i = 0; i < ys.length; i++) {
            ctx.moveTo(xpx[i]! + 2.3, ys[i]!);
            ctx.arc(xpx[i]!, ys[i]!, 2.3, 0, Math.PI * 2);
          }
          ctx.fill();
        }
      }
      const f = sc.items.get("fit");
      if (fit.current && f) {
        attrs(fit.current, { y1: String(f.y1) });
        attrs(fit.current, { y2: String(f.y2) });
      }
      for (const [k, el] of map.current) {
        const it = sc.items.get(k);
        if (k.startsWith("yt:")) place(el, it, { x: false });
        else if (k.startsWith("note:")) place(el, it ? { ...it, y: (it.y ?? 0) - 11 } : it, { x: false });
        else place(el, it, { x: false, y: false });
      }
    },
    [colors, map, plotW, height, xpx],
  );

  useMorph({ sceneOf, apply, span });

  const dpr = dpr_();
  return (
    <div ref={wrap} className={s.scatter} style={{ height }}>
      <canvas
        ref={canvas}
        width={Math.round(plotW * dpr)}
        height={Math.round(height * dpr)}
        style={{ width: plotW, height }}
        className={s.canvas}
        aria-hidden="true"
      />
      <svg className={s.overlay} width={plotW} height={height} aria-hidden="true">
        <line className={s.axis} x1={M.l} x2={M.l + iw} y1={M.t + ih + 0.5} y2={M.t + ih + 0.5} />
        {x.ticks(5).map((v) => (
          <g key={v} transform={`translate(${x(v)}, ${M.t + ih})`}>
            <line className={s.tick} y2={4} />
            <text className={s.tickText} y={17} textAnchor="middle">
              {fmtNum(v)}
            </text>
          </g>
        ))}
        <text className={s.axisName} x={M.l + iw} y={M.t + ih + 28} textAnchor="end">
          {xLabel}
        </text>
        {keys
          .filter((k) => k.startsWith("yt:"))
          .map((k) => (
            <g key={k} ref={reg(k)} style={{ opacity: 0 }}>
              <line className={s.gridLine} x1={M.l} x2={M.l + iw} />
              <text className={s.tickText} x={M.l - 7} dy="0.32em" textAnchor="end">
                {fmtNum(Number(k.split(":").pop()), 3)}
              </text>
            </g>
          ))}
        <line ref={fit} className={s.fit} x1={x0} x2={x1} />
      </svg>
      <div className={s.yLabels}>
        {keys
          .filter((k) => k.startsWith("yl:"))
          .map((k) => (
            <span key={k} ref={reg(k)} className={s.yLabel} style={{ opacity: 0 }}>
              {k.slice(3)}
            </span>
          ))}
      </div>
      <div className={s.notes} style={{ left: plotW - M.r + 4, width: noteWidth + M.r - 4 }}>
        {keys
          .filter((k) => k.startsWith("note:"))
          .map((k) => {
            const [r, text] = k.slice(5).split("|");
            return (
              <div key={k} ref={reg(k)} className={s.note} style={{ opacity: 0 }}>
                <span className={s.noteR}>r {r}</span>
                {text ? (
                  <span className={s.noteText}>
                    <Prose text={text} />
                  </span>
                ) : null}
              </div>
            );
          })}
      </div>
    </div>
  );
}

function dpr_(): number {
  return typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);
}
