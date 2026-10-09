/**
 * What every exhibit view shares (FOUNDATION §5 rule 9, designed once): a frame with one title and
 * one line saying what the view is, a legend when it draws two or more entities, the drawing or its
 * table (one quiet toggle), and, when there is nothing to draw, one line saying why instead of an
 * empty frame (§5 rule 7).
 */
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { fmtInt } from "../../stage/format";
import v from "./views.module.css";

/** The categorical slots in their fixed order: sage, plum, ochre, steel, clay (FOUNDATION §4). */
export type Slot = 1 | 2 | 3 | 4 | 5;
export const SLOTS: readonly Slot[] = [1, 2, 3, 4, 5];
export const slotColor = (slot: Slot | null): string => (slot ? `var(--cat-${slot})` : "var(--data-context)");

/**
 * The data gray one step stronger than --data-context, for a mark that must hold 3:1 on the canvas
 * in both themes (a recorded removal drawn as "your data now").
 */
export const DATA_CONTEXT_STRONG = "color-mix(in oklab, var(--data-fit) 30%, var(--data-context))";

/** A count of rows in words: "1 row", "666 rows". */
export const rowsWord = (n: number): string => `${fmtInt(n)} ${n === 1 ? "row" : "rows"}`;

/** At most this many series are named by direct labels on the drawing; more take a legend (never both). */
export const DIRECT_LABELS_UP_TO = 4;

export interface LegendItem {
  key: string;
  label: string;
  /** a categorical slot, or null for gray (your data now, or "other") */
  slot: Slot | null;
  /** a role color instead of a slot (indigo for what the choice touches), for a key, not an entity */
  color?: string;
  shape?: "square" | "dot";
}

export function Legend({
  items,
  focus = null,
  onFocus,
  keyOnly = false,
}: {
  items: LegendItem[];
  /** a key to one role color (the trim's removal), shown even with one item */
  keyOnly?: boolean;
  focus?: string | null;
  /** When given, pointing at (or focusing) an item isolates its entity in the drawing. */
  onFocus?: (key: string | null) => void;
}) {
  if (items.length < 2 && !keyOnly) return null; // one series: the title or a direct label names it
  return (
    <ul className={v.legend} aria-label="Legend">
      {items.map((it) => {
        const body = (
          <>
            <i className={v.swatch} data-shape={it.shape ?? "square"} style={{ background: it.color ?? slotColor(it.slot) }} aria-hidden="true" />
            {it.label}
          </>
        );
        return (
          <li key={it.key}>
            {onFocus ? (
              <button
                type="button"
                data-dim={focus !== null && focus !== it.key ? "true" : undefined}
                onPointerEnter={() => onFocus(it.key)}
                onPointerLeave={() => onFocus(null)}
                onFocus={() => onFocus(it.key)}
                onBlur={() => onFocus(null)}
              >
                {body}
              </button>
            ) : (
              body
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** A hover tooltip: marks call `tip.on(text)`; their hit targets are larger than the marks. */
export function useTip() {
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  return {
    node: tip ? (
      <div className={v.tip} style={{ left: tip.x, top: tip.y }} role="status">
        {tip.text}
      </div>
    ) : null,
    show: (text: string, x: number, y: number) => setTip({ text, x, y }),
    hide: () => setTip(null),
    on: (text: string) => ({
      onPointerMove: (e: React.PointerEvent) => setTip({ text, x: e.clientX, y: e.clientY }),
      onPointerLeave: () => setTip(null),
    }),
  };
}

/** A mark the keyboard can reach: its place in the drawing's own units, and what it says. */
export interface KeyMark<T> {
  id: T;
  x: number;
  y: number;
  text: string;
}

/**
 * The keyboard's path through a drawing's marks: the drawing is one tab stop, and the arrow keys
 * (Home, End) step from mark to mark in the view's reading order, each saying what hover says;
 * Escape or leaving clears it. `viewWidth` is the drawing's viewBox width, to place the tooltip.
 */
export function useKeyMarks<T>(
  marks: KeyMark<T>[],
  tip: ReturnType<typeof useTip>,
  viewWidth: number,
  onActive?: (id: T | null) => void,
) {
  const [at, setAt] = useState<number | null>(null);
  const go = (el: Element, k: number | null) => {
    setAt(k);
    const m = k === null ? null : marks[k];
    onActive?.(m ? m.id : null);
    if (!m) {
      tip.hide();
      return;
    }
    const box = el.getBoundingClientRect();
    const s = (box.width || viewWidth) / viewWidth;
    tip.show(m.text, box.left + m.x * s, box.top + m.y * s);
  };
  return {
    active: at === null ? null : (marks[at]?.id ?? null),
    props: {
      tabIndex: 0,
      onKeyDown: (e: React.KeyboardEvent<SVGSVGElement>) => {
        if (!marks.length) return;
        const last = marks.length - 1;
        const step: Record<string, number | null> = {
          ArrowRight: at === null ? 0 : Math.min(last, at + 1),
          ArrowDown: at === null ? 0 : Math.min(last, at + 1),
          ArrowLeft: at === null ? 0 : Math.max(0, at - 1),
          ArrowUp: at === null ? 0 : Math.max(0, at - 1),
          Home: 0,
          End: last,
          Escape: null,
        };
        if (!(e.key in step)) return;
        e.preventDefault();
        go(e.currentTarget, step[e.key]!);
      },
      onBlur: (e: React.FocusEvent<SVGSVGElement>) => go(e.currentTarget, null),
    },
  };
}

/** The container's width in CSS pixels, so text stays at its set size at any width. */
export function useWidth(fallback = 560): [React.RefObject<HTMLElement | null>, number] {
  const ref = useRef<HTMLElement | null>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setW(Math.max(280, Math.round(width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export interface FrameProps {
  title?: string;
  /** one line saying what the view is (the canvas register: plain words) */
  caption?: ReactNode;
  legend?: ReactNode;
  /** When set, the view has nothing to draw: this one line replaces the drawing and the table. */
  empty?: string | null;
  /** the table alternative, rendered only when chosen */
  table: () => ReactNode;
  children: ReactNode;
  frameRef?: React.RefObject<HTMLElement | null>;
  /** the exhibit view kind (FOUNDATION §5 rule 9), on the figure as data-exhibit-view */
  kind: string;
}

export function ViewFrame({ title, caption, legend, empty, table, children, frameRef, kind }: FrameProps) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className={v.frame} ref={frameRef as React.RefObject<HTMLElement>} data-exhibit-view={kind}>
      {title || !empty ? (
        <div className={v.head}>
          {title ? <h3>{title}</h3> : null}
          {empty ? null : (
            <button type="button" className={v.toggle} aria-pressed={asTable} onClick={() => setAsTable((t) => !t)}>
              {asTable ? "Show the chart" : "Show as a table"}
            </button>
          )}
        </div>
      ) : null}
      {empty ? (
        <p className={v.empty} role="note">
          {empty}
        </p>
      ) : (
        <>
          {caption ? <p className={v.caption}>{caption}</p> : null}
          {asTable ? (
            <div className={v.tableWrap}>{table()}</div>
          ) : (
            <>
              {legend}
              {children}
            </>
          )}
        </>
      )}
    </figure>
  );
}

/**
 * Text cut to fit `px` at about `charPx` per character (Source Sans 3 at 11–12 px), so a label
 * stays inside the drawing's bounds; the full text rides in a <title> where it is drawn.
 */
export function fitText(s: string, px: number, charPx = 6.2): string {
  const max = Math.max(4, Math.floor(px / charPx));
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

/** A percent for axes and tooltips: whole percents from 10%, one decimal below. */
export function fmtPct(share: number): string {
  const p = share * 100;
  if (p === 0) return "0%";
  if (Math.abs(p) >= 10) return `${Math.round(p)}%`;
  return `${+p.toFixed(1)}%`;
}

/**
 * A bar whose data end is rounded (4 px) and whose baseline end is square, growing up (dir −1) or
 * down (dir +1) from `base`.
 */
export function barPath(x: number, base: number, w: number, h: number, dir: 1 | -1): string {
  if (w <= 0 || h <= 0) return "";
  const r = Math.min(4, w / 2, h);
  const end = base + dir * h;
  const inner = end - dir * r;
  return [
    `M${x},${base}`,
    `L${x},${inner}`,
    `Q${x},${end} ${x + r},${end}`,
    `L${x + w - r},${end}`,
    `Q${x + w},${end} ${x + w},${inner}`,
    `L${x + w},${base}`,
    "Z",
  ].join(" ");
}
