/**
 * The whole table as a strip, one hairline per row (all 600), beside the window the table
 * magnifies. It carries the reshape at full scale: rows gather (a stacked export's second half
 * zips into the first), pairs meet, and the strip settles at half its height — 600 → 300 is a
 * length you can see, and the ticks show which rows the table beside it is.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { useTheme } from "../../../theme";
import { fmtInt } from "../data";
import { kindOf, windowRows } from "../reshape";
import type { MethodFixture, ReshapeFixture } from "../types";
import { useMorph } from "../useMorph";
import { mix, mixRGB, prepare, rgba, token } from "./canvas";
import s from "./views.module.css";

const W = 14;
const BRACKET = 10;

interface StripState {
  y: Float32Array;
  alpha: Float32Array;
  /** 0 = the row as loaded, 1 = a row of the table with this choice. */
  hue: Float32Array;
  height: number;
}

function lerpStrip(a: StripState, b: StripState, t: number): StripState {
  const n = a.y.length;
  const out: StripState = {
    y: new Float32Array(n),
    alpha: new Float32Array(n),
    hue: new Float32Array(n),
    height: mix(a.height, b.height, t),
  };
  for (let i = 0; i < n; i++) {
    out.y[i] = mix(a.y[i]!, b.y[i]!, t);
    out.alpha[i] = mix(a.alpha[i]!, b.alpha[i]!, t);
    out.hue[i] = mix(a.hue[i]!, b.hue[i]!, t);
  }
  return out;
}

export function RowStrip({
  fx,
  method,
  last,
  height,
}: {
  fx: ReshapeFixture;
  method: MethodFixture;
  last: number;
  height: number;
}) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const { theme } = useTheme();
  const canvas = useRef<HTMLCanvasElement>(null);
  const n = fx.strip.unit.length;
  const P = fx.per_unit;
  const h = height / n;
  const keep = kindOf(method) === "keep";
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));

  const states = useMemo(() => {
    // Which row each unit keeps (first / last) is computed for every row by the capture.
    const kept = method.strip_kept ?? null;
    const make = (st: number): StripState => {
      const y = new Float32Array(n);
      const alpha = new Float32Array(n);
      const hue = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u = fx.strip.unit[i]!;
        const k = fx.strip.k[i]!;
        const isKept = keep ? kept?.[i] === 1 : true;
        if (st === 0) {
          y[i] = i * h;
          alpha[i] = 1;
        } else if (st === 1) {
          y[i] = (u * P + k) * h;
          alpha[i] = 1;
        } else if (st === 2) {
          y[i] = keep ? (u * P + k) * h : (u * P + (P - 1) / 2) * h;
          alpha[i] = isKept ? 1 : 0.28;
          hue[i] = isKept ? 1 : 0;
        } else {
          y[i] = u * h;
          alpha[i] = keep ? (isKept ? 1 : 0) : k === 0 ? 1 : 0;
          hue[i] = 1;
        }
      }
      return { y, alpha, hue, height: st === 3 ? (n / P) * h : n * h };
    };
    return [0, 1, 2, 3].map(make);
  }, [fx, method, n, P, h, keep]);

  const windowIdx = useMemo(() => windowRows(fx).map((r) => r.row), [fx]);

  const paint = useCallback(
    (st: StripState) => {
      const el = canvas.current;
      if (!el) return;
      const ctx = prepare(el, W + BRACKET + 2, height + 2);
      if (!ctx || theme === undefined) return; // theme: repaint when the tokens change
      const now = token(el, "--c4");
      const withC = token(el, "--c1");
      const ink = token(el, "--ink");
      const line = token(el, "--line");
      ctx.fillStyle = rgba(line, 0.6);
      ctx.fillRect(0, 0, W, st.height);
      for (let i = 0; i < n; i++) {
        const a = st.alpha[i]!;
        if (a <= 0.01) continue;
        const k = fx.strip.k[i]!;
        const hue = st.hue[i]!;
        ctx.fillStyle = rgba(mixRGB(now, withC, hue), a * (k === 0 ? 0.95 : 0.42 + 0.5 * hue));
        ctx.fillRect(0, st.y[i]!, W, Math.max(h, 0.6));
      }
      // The window's rows: one tick each, on the right.
      ctx.strokeStyle = rgba(ink, 0.8);
      ctx.lineWidth = 1;
      for (const r of windowIdx) {
        const a = st.alpha[r]!;
        if (a <= 0.05) continue;
        const y = Math.round(st.y[r]! + h / 2) + 0.5;
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.moveTo(W + 2, y);
        ctx.lineTo(W + BRACKET, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    },
    [fx, n, h, height, windowIdx, theme],
  );

  const target = useCallback(
    (pos: number) => {
      const p = (pos * 3) / Math.max(1, last);
      const i = Math.max(0, Math.min(3, Math.floor(p)));
      return i >= 3 ? states[3]! : lerpStrip(states[i]!, states[i + 1]!, p - i);
    },
    [states, last],
  );
  const redrawRef = useRef<() => void>(() => {});
  const blend = useMorph<StripState>(
    lerpStrip,
    useCallback(() => redrawRef.current(), []),
  );
  const draw = useCallback(
    (pos: number) => paint(blend(target(pos), method.key)),
    [paint, blend, target, method.key],
  );
  useEffect(() => {
    redrawRef.current = () => draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  const nNow = at === 3 ? method.n_after : fx.dataset.rows;
  return (
    <div className={s.strip} data-view="row_strip" aria-label={`All ${fmtInt(nNow)} rows`}>
      <span className={s.stripCount}>{fmtInt(nNow)}</span>
      <canvas ref={canvas} style={{ width: W + BRACKET + 2, height: height + 2 }} />
    </div>
  );
}
