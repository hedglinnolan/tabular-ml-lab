/**
 * A small window onto the working data. Rows and columns are both virtualized,
 * and only the visible block is fetched (GET /table in pages of rows x chunks of
 * columns), so a 2,000-column or 10-million-row table costs the same to show.
 */
import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useTablePages, type TablePage } from "../../api/queries";
import type { Dtype, Scalar } from "../../api/schema";
import { cx, fmtInt, fmtValue } from "../../util/format";
import styles from "./TablePreview.module.css";

const ROW_H = 25;
const HEAD_H = 28;
const COL_W = 112;
const IDX_W = 54;
const PAGE = 50;
const CHUNK = 24;

interface Props {
  pid: string;
  columns: string[];
  /** Numbers align right, everything else left. */
  dtypes: Dtype[];
  nRows: number;
  target: string | null;
}

export function TablePreview({ pid, columns, dtypes, nRows, target }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line react-hooks/incompatible-library -- TanStack Virtual is the chosen virtualizer
  const rowV = useVirtualizer({
    count: nRows,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_H,
    overscan: 6,
    scrollPaddingStart: HEAD_H,
  });
  const colV = useVirtualizer({
    horizontal: true,
    count: columns.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => COL_W,
    overscan: 2,
  });
  const vRows = rowV.getVirtualItems();
  const vCols = colV.getVirtualItems();
  const r0 = vRows[0]?.index ?? 0;
  const r1 = vRows[vRows.length - 1]?.index ?? Math.min(nRows - 1, 12);
  const c0 = vCols[0]?.index ?? 0;
  const c1 = vCols[vCols.length - 1]?.index ?? Math.min(columns.length - 1, 4);

  const pages = useMemo(() => {
    const out: (TablePage & { chunk: number })[] = [];
    if (nRows === 0 || columns.length === 0) return out;
    for (let p = Math.floor(r0 / PAGE); p <= Math.floor(r1 / PAGE); p++) {
      for (let ch = Math.floor(c0 / CHUNK); ch <= Math.floor(c1 / CHUNK); ch++) {
        out.push({
          offset: p * PAGE,
          limit: PAGE,
          chunk: ch,
          columns: columns.slice(ch * CHUNK, (ch + 1) * CHUNK),
        });
      }
    }
    return out;
  }, [r0, r1, c0, c1, columns, nRows]);

  const results = useTablePages(pid, pages);

  const cell = (r: number, c: number): Scalar | undefined => {
    const pi = pages.findIndex(
      (p) => r >= p.offset && r < p.offset + PAGE && p.chunk === Math.floor(c / CHUNK),
    );
    if (pi < 0) return undefined;
    const data = results[pi]?.data;
    if (!data) return undefined;
    const row = data.rows[r - data.offset];
    const ci = data.columns.indexOf(columns[c]!);
    return row && ci >= 0 ? row[ci] : undefined;
  };

  const width = colV.getTotalSize() + IDX_W;
  return (
    <div className={styles.wrap}>
      <div className={styles.caption}>
        <span>Working data</span>
        <span className="num">
          {fmtInt(nRows)} × {fmtInt(columns.length)}
        </span>
      </div>
      <div
        ref={scrollRef}
        className={styles.scroll}
        role="grid"
        aria-label="Working data preview"
        aria-rowcount={nRows + 1}
        aria-colcount={columns.length + 1}
        tabIndex={0}
      >
        <div style={{ width, height: rowV.getTotalSize() + HEAD_H, position: "relative" }}>
          <div
            className={styles.head}
            role="row"
            aria-rowindex={1}
            style={{ width, height: HEAD_H }}
          >
            <div className={cx(styles.corner, styles.idx)} role="columnheader" aria-colindex={1}>
              #
            </div>
            {vCols.map((c) => {
              const name = columns[c.index]!;
              return (
                <div
                  key={c.key}
                  role="columnheader"
                  aria-colindex={c.index + 2}
                  className={cx(styles.hcell, name === target && styles.target)}
                  style={{ transform: `translateX(${c.start + IDX_W}px)`, width: COL_W }}
                  title={name}
                >
                  {name}
                </div>
              );
            })}
          </div>
          {vRows.map((r) => (
            <div
              key={r.key}
              role="row"
              aria-rowindex={r.index + 2}
              className={styles.row}
              style={{ transform: `translateY(${r.start + HEAD_H}px)`, width, height: ROW_H }}
            >
              <div className={styles.idx} role="rowheader">
                {fmtInt(r.index + 1)}
              </div>
              {vCols.map((c) => {
                const v = cell(r.index, c.index);
                const missing = v === null;
                return (
                  <div
                    key={c.key}
                    role="gridcell"
                    aria-colindex={c.index + 2}
                    className={cx(
                      styles.cell,
                      columns[c.index] === target && styles.targetCell,
                      (dtypes[c.index] === "numeric" || dtypes[c.index] === "integer") &&
                        styles.numeric,
                      missing && styles.missing,
                    )}
                    style={{ transform: `translateX(${c.start + IDX_W}px)`, width: COL_W }}
                  >
                    {v === undefined ? "" : missing ? "·" : fmtValue(v)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
