import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

/** The width of an element, following resizes. */
export function useWidth<E extends HTMLElement>(fallback: number) {
  const ref = useRef<E>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.round(el.getBoundingClientRect().width));
    const ro = new ResizeObserver(([e]) => {
      if (e) setW(Math.round(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** px per `ch` of the mono face at a size, re-measured once the vendored font has loaded. */
export function useCh(sizePx: number): number {
  const measure = () => {
    const c = document.createElement("canvas").getContext("2d");
    if (!c) return sizePx * 0.6;
    c.font = `${sizePx}px "JetBrains Mono", ui-monospace, monospace`;
    return c.measureText("0").width || sizePx * 0.6;
  };
  const [ch, setCh] = useState(measure);
  useEffect(() => {
    let live = true;
    void document.fonts?.ready.then(() => {
      if (live) setCh(measure());
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizePx]);
  return ch;
}

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

function parseColor(v: string): Rgb {
  const s = v.trim();
  if (s.startsWith("#")) {
    const h = s.length === 4 ? [...s.slice(1)].map((c) => c + c).join("") : s.slice(1, 7);
    return {
      r: parseInt(h.slice(0, 2), 16),
      g: parseInt(h.slice(2, 4), 16),
      b: parseInt(h.slice(4, 6), 16),
    };
  }
  const m = /rgba?\(([^)]+)\)/.exec(s);
  if (m) {
    const [r = 0, g = 0, b = 0] = m[1]!.split(",").map((x) => parseFloat(x));
    return { r, g, b };
  }
  return { r: 128, g: 128, b: 128 };
}

/** Re-read tokens when the theme attribute or the system scheme changes. */
function subscribeTheme(cb: () => void): () => void {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  mq?.addEventListener("change", cb);
  return () => {
    mo.disconnect();
    mq?.removeEventListener("change", cb);
  };
}

/** Token colors for canvas drawing (a canvas cannot read CSS variables), following the theme. */
export function useTokenColors(names: string[]): Record<string, Rgb> {
  const key = names.join(",");
  const snapshot = useSyncExternalStore(
    subscribeTheme,
    () => {
      const cs = getComputedStyle(document.documentElement);
      return key
        .split(",")
        .map((n) => cs.getPropertyValue(n).trim())
        .join("|");
    },
    () => "",
  );
  return useMemo(() => {
    const values = snapshot.split("|");
    return Object.fromEntries(key.split(",").map((n, i) => [n, parseColor(values[i] ?? "")]));
  }, [snapshot, key]);
}

export function mixRgb(a: Rgb, b: Rgb, p: number, alpha = 1): string {
  const m = (x: number, y: number) => Math.round(x + (y - x) * p);
  return `rgba(${m(a.r, b.r)}, ${m(a.g, b.g)}, ${m(a.b, b.b)}, ${alpha})`;
}
