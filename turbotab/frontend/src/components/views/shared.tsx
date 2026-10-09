/**
 * What every exhibit view shares: the one line it says instead of an empty frame, the tooltip
 * (pointer and keyboard focus alike), the table alternative behind one disclosure, and the
 * container's width so text keeps its set size at any width.
 */
import { useLayoutEffect, useRef, useState, type FocusEvent, type PointerEvent, type ReactNode } from "react";
import v from "./views.module.css";

/** One line saying why there is nothing to draw (FOUNDATION §5 rule 7): never an empty frame. */
export function OneLine({ text, view }: { text: string; view: string }) {
  return (
    <div className={v.view} data-exhibit-view={view} data-state="line">
      <p className={v.line} role="note">
        {text}
      </p>
    </div>
  );
}

export interface TipState {
  value: string;
  label: string;
  x: number;
  y: number;
}

/** The tooltip leads with the value and follows with its label (dataviz interaction rules). */
export function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const node = tip ? (
    <div className={v.tip} style={{ left: tip.x, top: tip.y }} role="status" data-testid="view-tip">
      <b>{tip.value}</b> · {tip.label}
    </div>
  ) : null;
  const on = (value: string, label: string) => ({
    onPointerMove: (e: PointerEvent) => setTip({ value, label, x: e.clientX, y: e.clientY }),
    onPointerLeave: () => setTip(null),
    onFocus: (e: FocusEvent<Element>) => {
      const r = e.currentTarget.getBoundingClientRect();
      setTip({ value, label, x: r.left + r.width / 2, y: r.top });
    },
    onBlur: () => setTip(null),
  });
  return { node, on, shown: tip };
}

/** Every chart's table alternative, one quiet disclosure under it. */
export function TableAlternative({ children, label = "Show as a table" }: { children: ReactNode; label?: string }) {
  return (
    <details className={v.alt} data-testid="table-alternative">
      <summary>{label}</summary>
      {children}
    </details>
  );
}

/** The container's width in CSS px (a fallback until it is measured, and in tests). */
export function useWidth<T extends HTMLElement>(fallback: number): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setW(Math.max(160, Math.round(width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** The comparison palette, in its fixed order (FOUNDATION §4): sage, plum, ochre, steel, clay. */
export const CATEGORICAL = ["var(--cat-1)", "var(--cat-2)", "var(--cat-3)", "var(--cat-4)", "var(--cat-5)"] as const;
