/**
 * Matrix (FOUNDATION §5 rule 9): a correlation or missingness heatmap on one gray ramp, your data
 * as it is (§4). A correlation's cells darken with the size of r either way, its sign printed in the
 * cell and said on hover; a share of blanks darkens from none to all. The key names every step's
 * bounds. Labels keep the declared order (named in the caption), values are printed only where they
 * reach `label_at` and the cell has room, and every cell answers on hover or by the arrow keys, its
 * row and column lit. Its table is the long form: one line per pair for a correlation.
 */
import { useState } from "react";
import { fmtInt } from "../../stage/format";
import { ViewFrame, fmtPct, rowsWord, useKeyMarks, useTip, useWidth } from "../common/frame";
import v from "../common/views.module.css";
import { MATRIX, NOT_COMPUTED, RAMP, STEP_LABELS, fillOf, fmtCell, inkOf, layoutMatrix } from "./layout";
import type { MatrixInput } from "./types";

/** The scale's key: every step with the range it covers, and "not computed" when a drawn cell is. */
function ScaleKey({ kind, missing }: { kind: MatrixInput["kind"]; missing: boolean }) {
  const lead = kind === "correlation" ? "Size of r, either way:" : "Blank:";
  return (
    <div className={v.scale} aria-label={`Scale. ${lead} ${STEP_LABELS[kind].join(", ")}${missing ? ", not computed" : ""}`}>
      <span>{lead}</span>
      <ol>
        {STEP_LABELS[kind].map((label, k) => (
          <li key={label}>
            <i style={{ background: RAMP[k] }} aria-hidden="true" />
            {label}
          </li>
        ))}
        {missing ? (
          <li>
            <i style={{ border: `1px solid ${NOT_COMPUTED}` }} aria-hidden="true" />
            not computed
          </li>
        ) : null}
      </ol>
    </div>
  );
}

export function MatrixView({ input, title }: { input: MatrixInput; title?: string }) {
  const [ref, W] = useWidth();
  const [lit, setLit] = useState<{ r: number; c: number } | null>(null);
  const tip = useTip();
  const res = layoutMatrix(input, W);
  const l = "layout" in res ? res.layout : null;
  const kind = input.kind;

  const say = (r: number, c: number, val: number | null, n: number | null) => {
    const pair = kind === "correlation" ? `${input.rows[r]} and ${input.cols[c]}` : `${input.cols[c]} · ${input.rows[r]}`;
    if (val === null) return `${pair} · not computed`;
    const value = kind === "correlation" ? `r ${fmtCell(kind, val)}` : `${fmtPct(val)} blank`;
    return `${pair} · ${value}${n !== null ? ` · ${rowsWord(n)}` : ""}`;
  };
  const keys = useKeyMarks(
    (l?.cells ?? []).map((c) => ({ id: c, x: c.x + (l?.cell ?? 0) / 2, y: c.y, text: say(c.r, c.c, c.v, c.n) })),
    tip,
    l?.width ?? W,
    (c) => setLit(c ? { r: c.r, c: c.c } : null),
  );
  if (!l) {
    return <ViewFrame title={title} empty={"empty" in res ? res.empty : null} table={() => null} frameRef={ref} kind="matrix">{null}</ViewFrame>;
  }
  const what =
    kind === "correlation"
      ? `${input.method ?? "Pearson"} correlation of each pair, on the rows where both are recorded`
      : "The share of blanks";
  const caption = `${what}${input.selection ? `, ${input.selection}` : ""}. Order: ${input.order}.${l.capped ? ` The first ${fmtInt(l.capped.shown)} of ${fmtInt(l.capped.of)} columns.` : ""}`;
  const missing = l.cells.some((c) => c.v === null);

  // The long form: one line per drawn pair (a correlation), or one line per group (blanks).
  const table = () =>
    kind === "correlation" ? (
      <table className={v.table}>
        <thead>
          <tr>
            <th scope="col">Pair</th>
            <th scope="col">r</th>
            <th scope="col">Rows</th>
          </tr>
        </thead>
        <tbody>
          {l.cells.map((c) => (
            <tr key={`${c.r},${c.c}`}>
              <td>
                {input.rows[c.r]} and {input.cols[c.c]}
              </td>
              <td>{c.v === null ? "not computed" : fmtCell(kind, c.v)}</td>
              <td>{c.n === null ? "" : fmtInt(c.n)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    ) : (
      <table className={v.table}>
        <thead>
          <tr>
            <th scope="col">Group</th>
            {input.cols.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {input.rows.map((r, i) => (
            <tr key={r}>
              <th scope="row" style={{ textAlign: "left", fontWeight: 400 }}>
                {r}
              </th>
              {input.cols.map((c, j) => {
                const val = input.values[i]![j];
                return <td key={c}>{val === null || val === undefined ? "not computed" : fmtPct(val)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    );

  return (
    <ViewFrame title={title} caption={caption} legend={<ScaleKey kind={kind} missing={missing} />} table={table} frameRef={ref} kind="matrix">
      <div className={v.tableWrap} style={{ maxHeight: "none" }}>
        <svg
          className={v.svg}
          viewBox={`0 0 ${l.width} ${l.height}`}
          style={l.width > W ? { width: l.width } : undefined}
          role="img"
          aria-label={`${kind === "correlation" ? "Correlations among" : "Blanks in"} ${fmtInt(input.cols.length)} columns; the arrow keys read each cell`}
          {...keys.props}
        >
          {l.rows.map((r) => (
            <text
              key={r.i}
              className={v.axis}
              x={l.left - 6}
              y={r.y + 4}
              textAnchor="end"
              style={lit?.r === r.i ? { fill: "var(--canvas-ink)", fontWeight: 600 } : undefined}
            >
              <title>{r.full}</title>
              {r.label}
            </text>
          ))}
          {l.cols.map((c) => (
            <text
              key={c.j}
              className={v.axis}
              transform={`translate(${c.x + 3},${l.top - 6}) rotate(-45)`}
              style={lit?.c === c.j ? { fill: "var(--canvas-ink)", fontWeight: 600 } : undefined}
            >
              <title>{c.full}</title>
              {c.label}
            </text>
          ))}
          {l.cells.map((cell) => {
            const s = l.cell - MATRIX.gap;
            const hover = tip.on(say(cell.r, cell.c, cell.v, cell.n));
            return (
              <g
                key={`${cell.r},${cell.c}`}
                data-cell={`${cell.r},${cell.c}`}
                onPointerEnter={() => setLit({ r: cell.r, c: cell.c })}
                onPointerMove={hover.onPointerMove}
                onPointerLeave={() => {
                  setLit(null);
                  hover.onPointerLeave();
                }}
              >
                <rect
                  x={cell.x + MATRIX.gap / 2}
                  y={cell.y + MATRIX.gap / 2}
                  width={s}
                  height={s}
                  rx={2}
                  style={{ fill: cell.v === null ? "none" : fillOf(kind, cell.v), stroke: cell.v === null ? NOT_COMPUTED : "none" }}
                />
                {keys.active === cell ? (
                  <rect x={cell.x + 0.5} y={cell.y + 0.5} width={l.cell - 1} height={l.cell - 1} rx={3} fill="none" style={{ stroke: "var(--canvas-ink)" }} strokeWidth={1.5} />
                ) : null}
                {/* the hit target: the whole cell, gap included */}
                <rect x={cell.x} y={cell.y} width={l.cell} height={l.cell} fill="transparent" />
                {cell.printed && cell.v !== null ? (
                  <text className={v.cellText} x={cell.x + l.cell / 2} y={cell.y + l.cell / 2 + 4} textAnchor="middle" style={{ fill: inkOf(kind, cell.v) }}>
                    {fmtCell(kind, cell.v)}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>
      {tip.node}
    </ViewFrame>
  );
}
