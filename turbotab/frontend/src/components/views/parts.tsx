/**
 * The parts every exhibit view is built from: the figure with its table alternative, the empty
 * line, the axes, the legend, the crosshair and the tooltip. A view composes these; it never
 * restyles them, so every curve on the tapestry reads the same (FOUNDATION §5 rule 9, §6).
 */
import { useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import type { ScaleLinear } from "d3-scale";
import { anchorAt, inner, type Box } from "./scale";
import s from "./curves.module.css";

export { fmtNum, fmtTick } from "../stage/format";

/** The comparison palette, in its fixed order: sage, plum, ochre, steel, clay (FOUNDATION §4). */
export type Slot = 1 | 2 | 3 | 4 | 5;
export const slotColor = (slot: Slot) => `var(--cat-${slot})`;

/**
 * Gray is the data now; indigo is exactly what the pointed choice touches; ink is the fit.
 *
 * The data-now gray is drawn a third of the way from `--data-context` toward the canvas ink: the
 * token alone is about 2:1 on the canvas in light and 2.4:1 in dark, under the 3:1 a meaningful
 * graphical mark needs. The mix reaches 3.5:1 (light) and 5.2:1 (dark) and stays at least 2.6:1 from
 * the fit's ink, so the primary (ink, and larger) still stands out. Derived from the tokens, so both themes
 * follow them.
 */
export const NOW = "color-mix(in srgb, var(--data-context) 65%, var(--canvas-ink))";
export const CHOICE = "var(--data-affected)";
export const FIT = "var(--data-fit)";

/** A marker's radius. With the canvas ring painted under the fill (`.ring`, paint-order), the
 *  colored disc is 10 px across, above the 8 px minimum. */
export const MARK_R = 5;

/** The width of a direct label's text: 12 px Source Sans 3 is under 6.4 px a character. */
export const CHAR_W = 6.4;

/** The container's width in CSS pixels, so text keeps its set size at any width. */
export function useWidth(fallback = 600): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
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

/** One line saying why there is nothing to draw: never an empty frame (FOUNDATION §5 rule 7). */
export function Empty({ kind, why, title }: { kind: string; why: string; title?: ReactNode }) {
  return (
    <figure className={s.fig} data-view={kind} data-empty="true">
      {title ? <h3>{title}</h3> : null}
      <p className={s.empty} role="status">
        {why}
      </p>
    </figure>
  );
}

export interface TableSpec {
  caption: string;
  head: string[];
  rows: { key: string; cells: ReactNode[]; primary?: boolean }[];
}

/** The table alternative: the same numbers as the picture, one disclosure away. */
export function Numbers({ table }: { table: TableSpec }) {
  return (
    <details className={s.numbers} data-testid="view-table">
      <summary>Show the numbers</summary>
      <div className={s.tableWrap}>
        <table>
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
                    <th key={i} scope="row" style={{ fontWeight: r.primary ? 600 : 400 }}>
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
      </div>
    </details>
  );
}

export interface KeyItem {
  label: string;
  color: string;
  mark: "line" | "thin" | "dot" | "band";
  /** a band's wash, as drawn */
  opacity?: number;
}

/** The legend: present for two or more series, so identity never rests on color alone. */
export function Legend({ items }: { items: KeyItem[] }) {
  if (items.length < 2) return null;
  return (
    <ul className={s.key} aria-label="Legend">
      {items.map((it) => (
        <li key={it.label}>
          <i data-mark={it.mark} style={{ background: it.color, opacity: it.opacity ?? 1 }} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/** Horizontal gridlines and the y tick labels, left of the plot. */
export function YAxis({ y, ticks, box, fmt, title }: { y: ScaleLinear<number, number>; ticks: number[]; box: Box; fmt: (v: number) => string; title?: string }) {
  const { x0, x1 } = inner(box);
  return (
    <g data-axis="y">
      {ticks.map((t) => (
        <g key={t} data-tick={t}>
          <line className={s.grid} x1={x0} x2={x1} y1={y(t)} y2={y(t)} />
          <text className={s.axis} x={x0 - 6} y={y(t) + 4} textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      {title ? (
        <text className={s.title} x={Math.max(0, x0 - 6 - leftLabelWidth(ticks, fmt))} y={12}>
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
      <line className={s.grid} x1={x0} x2={x1} y1={y1} y2={y1} />
      {ticks.map((t) => (
        <text key={t} data-tick={t} className={s.axis} x={x(t)} y={y1 + 15} textAnchor={anchorAt(x(t), box.width)}>
          {fmt(t)}
        </text>
      ))}
      {title ? (
        <text className={s.title} x={x1} y={y1 + 31} textAnchor="end">
          {title}
        </text>
      ) : null}
    </g>
  );
}

/** The width the y tick labels take (11 px tabular figures ≈ 6.2 px each). */
export function leftLabelWidth(ticks: number[], fmt: (v: number) => string): number {
  return Math.max(0, ...ticks.map((t) => fmt(t).length)) * 6.2;
}

/** The left margin the y tick labels need. */
export const leftFor = (ticks: number[], fmt: (v: number) => string, min = 34) => Math.max(min, Math.ceil(leftLabelWidth(ticks, fmt)) + 12);

/** Where the tooltip's center sits so all of it stays inside [0, width]: centred on the point,
 *  pushed in near either edge by its own measured width. */
export function tipLeft(x: number, width: number, tipWidth: number, margin = 4): number {
  const half = tipWidth / 2;
  if (tipWidth + 2 * margin >= width) return width / 2;
  return Math.min(Math.max(x, half + margin), width - half - margin);
}

/** The tooltip, inside the plot, at a point in the view's own pixels. Its width is measured, so a
 *  wide tooltip near an edge is pushed inside rather than clipped. */
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
    <div ref={ref} className={s.tip} style={{ left, top: at.y, ...(at.y < 96 ? { transform: "translate(-50%, 14px)" } : null) }} role="status">
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
        className={s.overlay}
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
      {focused ? <rect className={s.focusRing} x={x0 - 2} y={y0 - 2} width={x1 - x0 + 4} height={y1 - y0 + 4} rx={4} /> : null}
    </>
  );
}

export { s as viewStyles };
