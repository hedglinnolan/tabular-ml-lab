/**
 * Matrix (FOUNDATION §5 rule 9): a correlation or missingness heatmap. A correlation's scale
 * diverges through a neutral midpoint at 0, in equal steps per arm; a share of blanks runs from
 * the same neutral at none to gray at all. Labels keep the declared order (named in the caption),
 * values are printed only where they reach `label_at` and the cell has room, and every cell
 * answers on hover, its row and column lit.
 */
import { useState } from "react";
import { fmtInt, fmtR } from "../../stage/format";
import { ViewFrame, fmtPct, useTip, useWidth } from "../common/frame";
import v from "../common/views.module.css";
import { MATRIX, fillOf, fmtCell, inkOf, layoutMatrix } from "./layout";
import type { MatrixInput } from "./types";

/** The scale's key: every step, with its two ends and the midpoint named. */
function ScaleKey({ kind }: { kind: MatrixInput["kind"] }) {
  const values = kind === "correlation" ? [-0.9, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 0.9] : [0, 0.2, 0.4, 0.7, 0.9];
  const ends = kind === "correlation" ? ["−1 move opposite", "move together 1"] : ["none blank", "all blank"];
  return (
    <div className={v.scale} aria-label={kind === "correlation" ? "Scale: −1 to 1, neutral at 0" : "Scale: none blank to all blank"}>
      <span>{ends[0]}</span>
      <ol>
        {values.map((x) => (
          <li key={x} style={{ background: fillOf(kind, x) }} />
        ))}
      </ol>
      <span>{ends[1]}</span>
    </div>
  );
}

export function MatrixView({ input, title }: { input: MatrixInput; title?: string }) {
  const [ref, W] = useWidth();
  const [lit, setLit] = useState<{ r: number; c: number } | null>(null);
  const tip = useTip();
  const res = layoutMatrix(input, W);
  if ("empty" in res) {
    return <ViewFrame title={title} empty={res.empty} table={() => null} frameRef={ref} kind="matrix">{null}</ViewFrame>;
  }
  const l = res.layout;
  const kind = input.kind;
  const what =
    kind === "correlation"
      ? `${input.method ?? "Pearson"} correlation of each pair, on the rows where both are recorded`
      : "The share of blanks in each column";
  const caption = `${what}. Order: ${input.order}.${l.capped ? ` The first ${fmtInt(l.capped.shown)} of ${fmtInt(l.capped.of)} columns.` : ""}`;

  const say = (r: number, c: number, val: number | null, n: number | null) => {
    const pair = kind === "correlation" ? `${input.rows[r]} and ${input.cols[c]}` : `${input.cols[c]} · ${input.rows[r]}`;
    if (val === null) return `${pair} · not computed`;
    const value = kind === "correlation" ? `r ${fmtR(val)}` : `${fmtPct(val)} blank`;
    return `${pair} · ${value}${n !== null ? ` · ${fmtInt(n)} rows` : ""}`;
  };

  const table = () => (
    <table className={v.table}>
      <thead>
        <tr>
          <th scope="col">{kind === "correlation" ? "" : "Group"}</th>
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
              const hidden = input.symmetric && j >= i;
              return <td key={c}>{hidden || val === null || val === undefined ? "" : kind === "correlation" ? fmtR(val) : fmtPct(val)}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <ViewFrame title={title} caption={caption} legend={<ScaleKey kind={kind} />} table={table} frameRef={ref} kind="matrix">
      <div className={v.tableWrap} style={{ maxHeight: "none" }}>
        <svg
          className={v.svg}
          viewBox={`0 0 ${l.width} ${l.height}`}
          style={l.width > W ? { width: l.width } : undefined}
          role="img"
          aria-label={`${kind === "correlation" ? "Correlations among" : "Blanks in"} ${fmtInt(input.cols.length)} columns`}
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
                  style={{ fill: cell.v === null ? "none" : fillOf(kind, cell.v), stroke: cell.v === null ? "var(--canvas-line)" : "none" }}
                />
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
