/**
 * table_focus — the working table narrowed to what the choice touches: ≤ 12 columns,
 * ≤ 8 rows, and a count of the rest. Cells keep their identity (row id × column), so a
 * value tweens from what it was to what the option makes it; changed cells are tinted.
 * This is how the view stays readable at 495 or 20,000 columns: it never draws them.
 */
import { useMemo } from "react";
import { cellFormatter, fmtInt } from "../format";
import type { TableFocusView } from "../types";
import s from "./views.module.css";
import { Tween } from "../motion";

interface Props {
  view: TableFocusView;
  compact?: boolean;
  /** Column → how much its shape changed (ranks the columns shown). */
  rank?: Record<string, number>;
  nRows: number;
  /** The columns the shown ones are drawn from, e.g. 495 count columns. */
  universe: { n: number; noun: string };
}

/** The longest prefix every shown column shares, up to its last underscore ("gene_"). */
function sharedPrefix(cols: string[]): string {
  if (cols.length < 3) return "";
  let p = cols[0]!;
  for (const c of cols) while (!c.startsWith(p)) p = p.slice(0, -1);
  const cut = p.lastIndexOf("_");
  return cut >= 2 ? p.slice(0, cut + 1) : "";
}

/** One format per column, so `5` and `5.39` never sit side by side as if different kinds. */
function columnFormat(values: unknown[]): (v: number) => string {
  const nums = values.filter((v): v is number => typeof v === "number");
  const decimal = nums.find((v) => !Number.isInteger(v));
  return cellFormatter(decimal ?? nums[0] ?? 0);
}

function Cell({
  value,
  changed,
  format,
}: {
  value: unknown;
  changed: boolean;
  format: (v: number) => string;
}) {
  return (
    <td className={changed ? `${s.cell} ${s.cellChanged}` : s.cell}>
      {typeof value === "number" ? (
        <Tween value={value} format={format} />
      ) : (
        <span className="num">{String(value ?? "")}</span>
      )}
    </td>
  );
}

export function TableFocus({ view, compact = false, rank, nRows, universe }: Props) {
  const cols = view.columns_after.length ? view.columns_after : view.columns_before;
  const changed = new Set(view.changed.map(([r, c]) => `${r}|${c}`));
  const top = rank ? Math.max(...cols.map((c) => rank[c] ?? 0)) : 0;
  const rows = compact ? view.rows.slice(0, 5) : view.rows;
  const shownCols = compact ? cols.slice(0, 5) : cols;
  const rest = view.n_affected_columns ? view.n_affected_columns - shownCols.length : 0;
  const prefix = sharedPrefix(shownCols);
  const formats = useMemo(
    () =>
      Object.fromEntries(
        shownCols.map((c) => [c, columnFormat(rows.map((r) => r.after[c] ?? r.before[c]))]),
      ),
    [shownCols, rows],
  );
  return (
    <div className={s.tableWrap} data-view="table_focus">
      <div className={s.tableScroll}>
        <table className={compact ? s.tableCompact : s.table}>
          <thead>
            <tr>
              <th className={s.rowHead} scope="col">
                {prefix ? <span className={s.prefix}>{prefix}…</span> : "row"}
              </th>
              {shownCols.map((c) => (
                <th key={c} scope="col" className={s.colHead} title={c}>
                  <span className={s.colName}>{prefix ? c.slice(prefix.length) : c}</span>
                  {rank && !compact ? (
                    <span className={s.rankTrack} title="shape change (Wasserstein-1, z-scored)">
                      <span className={s.rankBar} style={{ width: `${(100 * (rank[c] ?? 0)) / (top || 1)}%` }} />
                    </span>
                  ) : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.row_id}>
                <th scope="row" className={s.rowId}>
                  {row.row_id}
                </th>
                {shownCols.map((c) => (
                  <Cell
                    key={c}
                    value={row.after[c] ?? row.before[c]}
                    changed={changed.has(`${row.row_id}|${c}`)}
                    format={formats[c]!}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={s.tableFoot}>
        <span>
          {shownCols.length} of {fmtInt(universe.n)} {universe.noun} · {rows.length} of {fmtInt(nRows)} rows
        </span>
        {rest > 0 ? (
          <span className={s.tableRest}>{fmtInt(rest)} more change the same way</span>
        ) : (
          <span className={s.tableRest}>no value changes</span>
        )}
      </div>
    </div>
  );
}
