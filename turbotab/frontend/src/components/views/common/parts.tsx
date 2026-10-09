/**
 * The parts every exhibit view is built from, designed once (FOUNDATION §5 rule 9, §6): the
 * palette by role, the container's width, the one line a view says instead of an empty frame, the
 * table alternative, the legend, the tooltips, the keyboard's path through the marks, the axes and
 * the crosshair. A view composes these and never restyles them, so every exhibit reads the same.
 * The scale math lives beside this file in ./scale.ts; the stylesheet in ./views.module.css.
 */
import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type FocusEvent,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import type { ScaleLinear } from "d3-scale";
import { fmtInt } from "../../stage/format";
import { anchorAt, inner, type Box } from "./scale";
import v from "./views.module.css";

export { fmtNum, fmtTick } from "../../stage/format";

// ── color by role (FOUNDATION §4) ───────────────────────────────────────────

/** The comparison palette's slots, in their fixed order: sage, plum, ochre, steel, clay. Color
 *  follows the entity, never its rank. */
export type Slot = 1 | 2 | 3 | 4 | 5;
export const SLOTS: readonly Slot[] = [1, 2, 3, 4, 5];
/** A slot's color; null is gray (your data now, or "other"). */
export const slotColor = (slot: Slot | null): string => (slot ? `var(--cat-${slot})` : "var(--data-context)");
/** The palette as colors, in its fixed order. */
export const CATEGORICAL: readonly string[] = SLOTS.map(slotColor);

/**
 * Gray is the data now. A mark drawn as "your data now" sits a third of the way from
 * `--data-context` toward the canvas ink: the token alone is about 2:1 on the canvas in light and
 * 2.4:1 in dark, under the 3:1 a meaningful graphical mark needs. The mix reaches 3.5:1 (light) and
 * 5.2:1 (dark) and stays at least 2.6:1 from the fit's ink, so the primary still stands out.
 * Derived from the tokens, so both themes follow them.
 */
export const NOW = "color-mix(in srgb, var(--data-context) 65%, var(--canvas-ink))";
/** Indigo is exactly what the pointed choice touches. */
export const CHOICE = "var(--data-affected)";
/** Ink is the fit. */
export const FIT = "var(--data-fit)";

// ── sizes ───────────────────────────────────────────────────────────────────

/** A marker's radius. With the canvas ring painted under the fill (paint-order), the colored disc
 *  is 10 px across, above the 8 px minimum. */
export const MARK_R = 5;
/** The width of a direct label's character: 12 px Source Sans 3 is under 6.4 px a character. */
export const CHAR_W = 6.4;
/** At most this many series are named by direct labels on the drawing; more take a legend. */
export const DIRECT_LABELS_UP_TO = 4;

// ── words ───────────────────────────────────────────────────────────────────

/** A count of rows in words: "1 row", "666 rows". */
export const rowsWord = (n: number): string => `${fmtInt(n)} ${n === 1 ? "row" : "rows"}`;

/** A percent for axes and tooltips: whole percents from 10%, one decimal below. */
export function fmtPct(share: number): string {
  const p = share * 100;
  if (p === 0) return "0%";
  if (Math.abs(p) >= 10) return `${Math.round(p)}%`;
  return `${+p.toFixed(1)}%`;
}

/** Text cut to fit `px` at about `charPx` per character, so a label stays inside the drawing's
 *  bounds; the full text rides in a <title> where it is drawn. */
export function fitText(s: string, px: number, charPx = 6.2): string {
  const max = Math.max(4, Math.floor(px / charPx));
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

// ── the container's width ───────────────────────────────────────────────────

/**
 * The container's width in CSS px (the fallback until it is measured, and in tests), so text keeps
 * its set size at any width. The ref is a callback, so the observer attaches whenever the element
 * mounts: a view that first says its one line (a closed gate, no rows) and draws later is still
 * measured.
 */
export function useWidth<T extends HTMLElement = HTMLElement>(fallback = 600, min = 280): [(el: T | null) => void, number] {
  const [w, setW] = useState(fallback);
  const observer = useRef<ResizeObserver | null>(null);
  const ref = useCallback(
    (el: T | null) => {
      observer.current?.disconnect();
      observer.current = null;
      if (!el || typeof ResizeObserver === "undefined") return;
      const ro = new ResizeObserver((entries) => {
        const width = entries[0]?.contentRect.width;
        if (width) setW(Math.max(min, Math.round(width)));
      });
      ro.observe(el);
      observer.current = ro;
    },
    [min],
  );
  return [ref, w];
}

// ── the one line, and the table alternative ─────────────────────────────────

/** One line saying why there is nothing to draw (FOUNDATION §5 rule 7): never an empty frame. */
export function OneLine({ text, view, title }: { text: ReactNode; view: string; title?: ReactNode }) {
  return (
    <figure className={v.frame} data-exhibit-view={view} data-empty="true">
      {title ? (
        <div className={v.head}>
          <h3>{title}</h3>
        </div>
      ) : null}
      <p className={v.empty} role="note">
        {text}
      </p>
    </figure>
  );
}

/**
 * Every view's table alternative: the same numbers as the picture, one quiet disclosure under it.
 * A function child renders only once the disclosure is opened (a long table costs nothing closed).
 */
export function TableAlternative({ children, label = "Show as a table" }: { children: ReactNode | (() => ReactNode); label?: string }) {
  const lazy = typeof children === "function";
  const [open, setOpen] = useState(false);
  return (
    <details className={v.alt} data-testid="table-alternative" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{label}</summary>
      <div className={v.tableWrap}>{lazy ? (open ? (children as () => ReactNode)() : null) : children}</div>
    </details>
  );
}

export interface TableSpec {
  caption: string;
  head: string[];
  rows: { key: string; cells: ReactNode[]; primary?: boolean }[];
}

/** A table alternative from a spec: the first cell of each row is its header. */
export function Numbers({ table }: { table: TableSpec }) {
  return (
    <TableAlternative>
      <table className={v.table}>
        <caption>{table.caption}</caption>
        <thead>
          <tr>
            {table.head.map((h) => (
              <th key={h} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r) => (
            <tr key={r.key} data-primary={r.primary || undefined}>
              {r.cells.map((c, i) =>
                i === 0 ? (
                  <th key={i} scope="row">
                    {c}
                  </th>
                ) : (
                  <td key={i}>{c}</td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </TableAlternative>
  );
}

// ── the frame ───────────────────────────────────────────────────────────────

export interface FrameProps {
  title?: ReactNode;
  /** one line saying what the view is (the canvas register: plain words) */
  caption?: ReactNode;
  legend?: ReactNode;
  /** When set, the view has nothing to draw: this one line replaces the drawing and the table. */
  empty?: string | null;
  /** the table alternative's content, rendered only when it is opened */
  table: () => ReactNode;
  children: ReactNode;
  frameRef?: (el: HTMLElement | null) => void;
  /** the exhibit view kind (FOUNDATION §5 rule 9), on the figure as data-exhibit-view */
  kind: string;
}

/** A view's frame: one title, one line saying what it is, the legend, the drawing and its table. */
export function ViewFrame({ title, caption, legend, empty, table, children, frameRef, kind }: FrameProps) {
  if (empty)
    return (
      <div ref={frameRef}>
        <OneLine text={empty} view={kind} title={title} />
      </div>
    );
  return (
    <figure className={v.frame} ref={frameRef} data-exhibit-view={kind}>
      {title ? (
        <div className={v.head}>
          <h3>{title}</h3>
        </div>
      ) : null}
      {caption ? <p className={v.caption}>{caption}</p> : null}
      {legend}
      {children}
      <TableAlternative>{table}</TableAlternative>
    </figure>
  );
}

// ── the legend ──────────────────────────────────────────────────────────────

export interface LegendItem {
  key?: string;
  label: string;
  /** a categorical slot, or null for gray (your data now, or "other") */
  slot?: Slot | null;
  /** a role color instead of a slot (indigo for what the choice touches, ink for the fit) */
  color?: string;
  /** the swatch drawn as the mark is: a line, a thin line, a dot, a band, or a square */
  mark?: "line" | "thin" | "dot" | "band" | "square";
  /** a band's wash, as drawn */
  opacity?: number;
}

/**
 * The legend: present for two or more series, so identity never rests on color alone (one series
 * is named by its title or a direct label). `keyOnly` keys one role color even with one item.
 * With `onFocus`, pointing at (or focusing) an item isolates its entity in the drawing.
 */
export function Legend({
  items,
  focus = null,
  onFocus,
  keyOnly = false,
  testId,
}: {
  items: LegendItem[];
  keyOnly?: boolean;
  focus?: string | null;
  onFocus?: (key: string | null) => void;
  testId?: string;
}) {
  if (items.length < 2 && !keyOnly) return null;
  return (
    <ul className={v.legend} aria-label="Legend" data-testid={testId}>
      {items.map((it) => {
        const key = it.key ?? it.label;
        const body = (
          <>
            <i
              className={v.swatch}
              data-mark={it.mark ?? "square"}
              style={{ background: it.color ?? slotColor(it.slot ?? null), opacity: it.opacity ?? 1 }}
              aria-hidden="true"
            />
            {it.label}
          </>
        );
        return (
          <li key={key}>
            {onFocus ? (
              <button
                type="button"
                data-dim={focus !== null && focus !== key ? "true" : undefined}
                onPointerEnter={() => onFocus(key)}
                onPointerLeave={() => onFocus(null)}
                onFocus={() => onFocus(key)}
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

// ── the tooltips ────────────────────────────────────────────────────────────

interface TipState {
  value: ReactNode;
  label?: ReactNode;
  x: number;
  y: number;
}

/**
 * The hover tooltip for marks, at the pointer (or above a focused mark): it leads with the value
 * and follows with its label. Marks spread `tip.on(value, label)`; their hit targets are larger
 * than the marks. `show` and `hide` drive it from a hit test or the keyboard.
 */
export function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const node = tip ? (
    <div className={v.tip} style={{ left: tip.x, top: tip.y }} role="status" data-testid="view-tip">
      {tip.label === undefined ? (
        tip.value
      ) : (
        <>
          <b>{tip.value}</b> · {tip.label}
        </>
      )}
    </div>
  ) : null;
  const show = (value: ReactNode, x: number, y: number, label?: ReactNode) => setTip({ value, label, x, y });
  const hide = () => setTip(null);
  const on = (value: ReactNode, label?: ReactNode) => ({
    onPointerMove: (e: PointerEvent) => setTip({ value, label, x: e.clientX, y: e.clientY }),
    onPointerLeave: hide,
    onFocus: (e: FocusEvent<Element>) => {
      const r = e.currentTarget.getBoundingClientRect();
      setTip({ value, label, x: r.left + r.width / 2, y: r.top });
    },
    onBlur: hide,
  });
  return { node, on, show, hide, shown: tip };
}

/** Where an in-plot tooltip's center sits so all of it stays inside [0, width]: centred on the
 *  point, pushed in near either edge by its own measured width. */
export function tipLeft(x: number, width: number, tipWidth: number, margin = 4): number {
  const half = tipWidth / 2;
  if (tipWidth + 2 * margin >= width) return width / 2;
  return Math.min(Math.max(x, half + margin), width - half - margin);
}

/** The crosshair's tooltip, inside the plot, at a point in the view's own pixels. Its width is
 *  measured, so a wide tooltip near an edge is pushed inside rather than clipped. */
export function Tip({ at, width, children }: { at: { x: number; y: number } | null; width: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [tw, setTw] = useState(180);
  // Re-measured whenever the content or the point changes; settles in one pass (same width, no set).
  useLayoutEffect(() => {
    const w = ref.current?.offsetWidth;
    if (w && Math.abs(w - tw) > 0.5) setTw(w);
  }, [at, children, tw]);
  if (!at) return null;
  const left = tipLeft(at.x, width, tw);
  return (
    // Near the top edge the tooltip opens below the point, so it stays inside the view.
    <div
      ref={ref}
      className={v.tip}
      data-anchor="plot"
      style={{ left, top: at.y, ...(at.y < 96 ? { transform: "translate(-50%, 14px)" } : null) }}
      role="status"
    >
      {children}
    </div>
  );
}

export interface TipLine {
  label: string;
  value: string;
  color?: string;
}

export function TipLines({ head, lines }: { head?: string; lines: TipLine[] }) {
  return (
    <>
      {head ? <div style={{ fontWeight: 600 }}>{head}</div> : null}
      {lines.map((l) => (
        <div key={l.label}>
          {l.color ? <i style={{ background: l.color }} /> : null}
          <span>
            {l.label} {l.value}
          </span>
        </div>
      ))}
    </>
  );
}

// ── the keyboard's path ─────────────────────────────────────────────────────

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
export function useKeyMarks<T>(marks: KeyMark<T>[], tip: ReturnType<typeof useTip>, viewWidth: number, onActive?: (id: T | null) => void) {
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
      onKeyDown: (e: KeyboardEvent<SVGSVGElement>) => {
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
      onBlur: (e: FocusEvent<SVGSVGElement>) => go(e.currentTarget, null),
    },
  };
}

// ── axes and the crosshair ──────────────────────────────────────────────────

/** The width the y tick labels take (11 px tabular figures ≈ 6.2 px each). */
export function leftLabelWidth(ticks: number[], fmt: (v: number) => string): number {
  return Math.max(0, ...ticks.map((t) => fmt(t).length)) * 6.2;
}

/** The left margin the y tick labels need. */
export const leftFor = (ticks: number[], fmt: (v: number) => string, min = 34) => Math.max(min, Math.ceil(leftLabelWidth(ticks, fmt)) + 12);

/** Horizontal gridlines and the y tick labels, left of the plot. */
export function YAxis({ y, ticks, box, fmt, title }: { y: ScaleLinear<number, number>; ticks: number[]; box: Box; fmt: (v: number) => string; title?: string }) {
  const { x0, x1 } = inner(box);
  return (
    <g data-axis="y">
      {ticks.map((t) => (
        <g key={t} data-tick={t}>
          <line className={v.grid} x1={x0} x2={x1} y1={y(t)} y2={y(t)} />
          <text className={v.axis} x={x0 - 6} y={y(t) + 4} textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      {title ? (
        <text className={v.axisTitle} x={Math.max(0, x0 - 6 - leftLabelWidth(ticks, fmt))} y={12}>
          {title}
        </text>
      ) : null}
    </g>
  );
}

/** The x tick labels under the plot, anchored so the first and last stay inside the view. */
export function XAxis({ x, ticks, box, fmt, title }: { x: ScaleLinear<number, number>; ticks: number[]; box: Box; fmt: (v: number) => string; title?: string }) {
  const { x0, x1, y1 } = inner(box);
  return (
    <g data-axis="x">
      <line className={v.grid} x1={x0} x2={x1} y1={y1} y2={y1} />
      {ticks.map((t) => (
        <text key={t} data-tick={t} className={v.axis} x={x(t)} y={y1 + 15} textAnchor={anchorAt(x(t), box.width)}>
          {fmt(t)}
        </text>
      ))}
      {title ? (
        <text className={v.axisTitle} x={x1} y={y1 + 31} textAnchor="end">
          {title}
        </text>
      ) : null}
    </g>
  );
}

/**
 * The crosshair's surface: a transparent plot-sized target that snaps the pointer to the nearest
 * x, and the arrow keys step along it. The hit target is the whole plot, larger than any mark.
 */
export function Crosshair({
  box,
  count,
  indexAt,
  index,
  setIndex,
  label,
}: {
  box: Box;
  count: number;
  /** the index nearest to x, in the view's pixels */
  indexAt: (x: number) => number;
  index: number | null;
  setIndex: (i: number | null) => void;
  label: string;
}) {
  const { x0, x1, y0, y1 } = inner(box);
  const [focused, setFocused] = useState(false);
  const move = (e: PointerEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement;
    const r = svg?.getBoundingClientRect();
    const scale = r && r.width ? box.width / r.width : 1;
    const x = (e.clientX - (r?.left ?? 0)) * scale;
    setIndex(indexAt(x));
  };
  const key = (e: KeyboardEvent<SVGRectElement>) => {
    if (!count) return;
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (e.key === "Home") setIndex(0);
    else if (e.key === "End") setIndex(count - 1);
    else if (step) setIndex(Math.min(count - 1, Math.max(0, (index ?? (step > 0 ? -1 : count)) + step)));
    else if (e.key === "Escape") setIndex(null);
    else return;
    e.preventDefault();
  };
  return (
    <>
      <rect
        className={v.overlay}
        x={x0}
        y={y0}
        width={Math.max(0, x1 - x0)}
        height={Math.max(0, y1 - y0)}
        tabIndex={0}
        role="slider"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={Math.max(0, count - 1)}
        aria-valuenow={index ?? 0}
        onPointerMove={move}
        onPointerLeave={() => setIndex(null)}
        onKeyDown={key}
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false);
          setIndex(null);
        }}
      />
      {focused ? <rect className={v.focusRing} x={x0 - 2} y={y0 - 2} width={x1 - x0 + 4} height={y1 - y0 + 4} rx={4} /> : null}
    </>
  );
}

// ── shapes ──────────────────────────────────────────────────────────────────

/** A bar whose data end is rounded (4 px) and whose baseline end is square, growing up (dir −1) or
 *  down (dir +1) from `base`. */
export function barPath(x: number, base: number, w: number, h: number, dir: 1 | -1): string {
  if (w <= 0 || h <= 0) return "";
  const r = Math.min(4, w / 2, h);
  const end = base + dir * h;
  const inner = end - dir * r;
  return [`M${x},${base}`, `L${x},${inner}`, `Q${x},${end} ${x + r},${end}`, `L${x + w - r},${end}`, `Q${x + w},${end} ${x + w},${inner}`, `L${x + w},${base}`, "Z"].join(" ");
}

export { v as sharedStyles };
