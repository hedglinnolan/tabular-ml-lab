/**
 * Wide data shows what the choice touched (§11.3): a heat grid of the ≤ 12 columns whose shape
 * changed most, over ≤ 8 rows, and a barcode of every affected column so the reader sees where
 * the shown twelve sit among all of them. The cells are the same cells before and after, so a
 * transform recolors them in place.
 */
import { useMemo } from "react";
import { motion } from "motion/react";
import { useTransitions } from "../../../motion/prefs";
import { fmtCell } from "../format";
import type { TableFocusView } from "../types";
import s from "./views.module.css";

function norm(values: number[]): number[] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return values.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0));
}

interface GridProps {
  view: TableFocusView;
  transformed: boolean;
  width: number;
  cell?: { w: number; h: number };
  title: string;
}

export function HeatGrid({ view, transformed, width, cell = { w: 30, h: 19 }, title }: GridProps) {
  const t = useTransitions();
  const cols = transformed ? view.columns_after : view.columns_before;
  const emph = new Set(view.emphasis);
  const headH = 30;
  const colW = Math.min(cell.w, Math.floor(width / cols.length));
  const height = headH + view.rows.length * cell.h;

  // Each column normalized over the shown rows, before and after, so color is "where this value
  // sits in its column" — the thing a log transform changes.
  const shade = useMemo(() => {
    const side = transformed ? "after" : "before";
    return cols.map((c) => norm(view.rows.map((r) => Number(r[side][c] ?? 0))));
  }, [cols, view.rows, transformed]);

  return (
    <svg
      width={colW * cols.length}
      height={height}
      className={s.grid2}
      role="img"
      aria-label={title}
    >
      {cols.map((c, j) => (
        <text
          key={c}
          x={j * colW + colW / 2}
          y={headH - 8}
          textAnchor="middle"
          className={emph.has(c) ? s.colHeadEmph : s.colHead}
        >
          {c.replace(/^gene_0*/, "")}
        </text>
      ))}
      {view.rows.map((r, i) =>
        cols.map((c, j) => {
          const v = shade[j]![i]!;
          const raw = r.before[c];
          const after = r.after[c];
          return (
            <g key={`${r.row_id}-${c}`}>
              <motion.rect
                className={emph.has(c) ? s.cellEmph : s.cell}
                x={j * colW}
                y={headH + i * cell.h}
                width={colW}
                height={cell.h}
                initial={false}
                animate={{ fillOpacity: 0.06 + 0.84 * v }}
                transition={t.arrive}
              >
                <title>{`${c}, row ${r.row_id}: ${fmtCell(raw)} → ${fmtCell(after)}`}</title>
              </motion.rect>
              <rect
                className={s.cellFrame}
                x={j * colW}
                y={headH + i * cell.h}
                width={colW}
                height={cell.h}
              />
              {emph.has(c) ? (
                <text
                  x={j * colW + colW / 2}
                  y={headH + i * cell.h + cell.h / 2}
                  dy="0.34em"
                  textAnchor="middle"
                  className={s.cellText}
                  style={{ fill: v > 0.55 ? "var(--surface)" : "var(--ink)" }}
                >
                  {fmtCell(transformed ? after : raw)}
                </text>
              ) : null}
            </g>
          );
        }),
      )}
      {cols.map((c, j) =>
        emph.has(c) ? (
          <rect
            key={`box-${c}`}
            className={s.emphBox}
            x={j * colW + 0.5}
            y={headH - 0.5}
            width={colW - 1}
            height={view.rows.length * cell.h + 1}
          />
        ) : null,
      )}
    </svg>
  );
}

interface BarcodeProps {
  n: number;
  shown: string[];
  on: boolean;
  width: number;
  height?: number;
  title: string;
}

/** One tick per affected column; the shown ones stand taller. */
export function Barcode({ n, shown, on, width, height = 16, title }: BarcodeProps) {
  const pos = new Set(
    shown.map((c) => Number(/(\d+)$/.exec(c)?.[1] ?? -1) - 1).filter((i) => i >= 0),
  );
  const w = width / n;
  return (
    <svg width={width} height={height} className={s.barcode} role="img" aria-label={title}>
      {Array.from({ length: n }, (_, i) => {
        const isShown = pos.has(i);
        return (
          <rect
            key={i}
            x={i * w}
            y={isShown ? 0 : 4}
            width={Math.max(0.6, w - 0.35)}
            height={isShown ? height : height - 8}
            data-on={on}
            data-shown={isShown}
          />
        );
      })}
    </svg>
  );
}
