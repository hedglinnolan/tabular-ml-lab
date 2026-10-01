/** Small drawing helpers shared by the views (pure; pixel arrays are flat Float64Arrays). */
import { useLayoutEffect, useRef, useState, type RefObject } from "react";

export interface Size {
  w: number;
  h: number;
}

/** The content box of an element, kept current with a ResizeObserver. */
export function useSize<T extends HTMLElement>(): [RefObject<T | null>, Size] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // contentRect is the layout size, unaffected by an ancestor's mid-flight transform.
    const ro = new ResizeObserver((entries) => {
      const r = entries[entries.length - 1]!.contentRect;
      const w = Math.round(r.width);
      const h = Math.round(r.height);
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function lerpArray(a: Float64Array, b: Float64Array, t: number, out?: Float64Array): Float64Array {
  const o = out ?? new Float64Array(b.length);
  for (let i = 0; i < b.length; i++) o[i] = a[i]! + (b[i]! - a[i]!) * t;
  return o;
}

/** One SVG path of n dots (two arcs each): 800 points cost one DOM node. */
export function dotPath(v: Float64Array, count: number, r: number): string {
  const d: string[] = [];
  for (let i = 0; i < count; i++) {
    const x = v[i * 2]!;
    const y = v[i * 2 + 1]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    d.push(`M${(x - r).toFixed(1)},${y.toFixed(1)}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`);
  }
  return d.join("");
}

/** Bars as one path: each bar is [x0, x1, yTop, yBase] in the flat array. */
export function barPath(v: Float64Array, count: number): string {
  const d: string[] = [];
  for (let i = 0; i < count; i++) {
    const x0 = v[i * 4]!;
    const x1 = v[i * 4 + 1]!;
    const top = v[i * 4 + 2]!;
    const base = v[i * 4 + 3]!;
    if (x1 - x0 <= 0.2 || base - top <= 0.05) continue;
    d.push(`M${x0.toFixed(1)},${base.toFixed(1)}V${top.toFixed(1)}H${x1.toFixed(1)}V${base.toFixed(1)}Z`);
  }
  return d.join("");
}

/** Least squares through points: [slope, intercept]. */
export function ols(points: [number, number][]): [number, number] {
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
  const den = n * sxx - sx * sx;
  if (!n || den === 0) return [0, n ? sy / n : 0];
  const slope = (n * sxy - sx * sy) / den;
  return [slope, (sy - slope * sx) / n];
}
