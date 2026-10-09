/**
 * The table view kind (FOUNDATION §5 rule 9, designed once here): an exhibit table as the paper
 * prints it, Table 1 or Table 2.
 *
 * Design decisions, recorded:
 * - Three rules only (above the header, under it, under the last row), as journals set tables; no
 *   zebra stripes, no boxes, no color. The data is the only loud thing.
 * - The caption is the table's own <caption>: its number in the heavier weight, then the title in
 *   the paper's register.
 * - Numbers are tabular; count and summary columns align right on their digits; an estimate with
 *   its interval reads left to right, "−0.0199 (−0.0327 to −0.00718)", "to" between the limits so
 *   a negative limit never reads as a range.
 * - The locked primary row is set in the heavier weight, never colored.
 * - Footnotes are quiet: a superscript mark on the row, the note under the table in the muted ink;
 *   notes that apply to the whole table carry no mark.
 * - A table is its own table alternative: real <th scope> cells, readable by a screen reader.
 * - Rows share keys with the forest beside them: pointing at a row lights it in both.
 * - Under a closed gate (rule 6) the table says one line instead of any estimate.
 */
import type { Purpose } from "../stage/purposes";
import { fmtCell, isNumeric } from "./format";
import { OneLine } from "./shared";
import type { Gate, TableData, TableRow } from "./types";
import v from "./views.module.css";

export const TABLE_PURPOSE: Purpose = {
  question: "matters",
  answer: "the numbers the paper reports, each with its interval and the rows behind it",
};

export interface ExhibitTableProps {
  data: TableData;
  gate?: Gate;
  /** The row pointed at here or in a linked view. */
  lit?: string | null;
  onLit?: (key: string | null) => void;
}

/** A column is numeric when every filled cell in it is a count, a summary or a p-value. */
export function numericColumns(data: TableData): Set<string> {
  const out = new Set<string>();
  for (const c of data.columns) {
    const cells = data.rows.flatMap((r) => (r.kind === "row" && r.cells[c.key] ? [r.cells[c.key]] : []));
    if (cells.length && cells.every(isNumeric)) out.add(c.key);
  }
  return out;
}

function Marks({ marks }: { marks?: string[] }) {
  if (!marks?.length) return null;
  return <span className={v.mark}>{marks.join(",")}</span>;
}

export function ExhibitTable({ data, gate, lit, onLit }: ExhibitTableProps) {
  if (gate) return <OneLine view="table" text={gate} />;
  if (!data.rows.some((r) => r.kind === "row")) return <OneLine view="table" text={data.empty ?? `${data.number} has no rows yet.`} />;
  const numeric = numericColumns(data);
  const pointer = (r: TableRow) =>
    onLit && r.kind === "row" ? { onPointerEnter: () => onLit(r.key), onPointerLeave: () => onLit(null) } : {};
  return (
    <div className={v.view} data-exhibit-view="table">
      <div className={v.wrap}>
        <table className={v.table}>
          <caption className={v.caption}>
            <b>{data.number}.</b> {data.title}
          </caption>
          <thead>
            <tr>
              <th scope="col">{data.stub}</th>
              {data.columns.map((c) => (
                <th scope="col" key={c.key} className={numeric.has(c.key) ? v.num : undefined}>
                  {c.label}
                  {c.sub ? <small>{c.sub}</small> : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) =>
              r.kind === "group" ? (
                <tr key={r.key} data-group="true">
                  <th scope="rowgroup" colSpan={data.columns.length + 1}>
                    {r.label}
                    <Marks marks={r.marks} />
                  </th>
                </tr>
              ) : (
                <tr key={r.key} data-row={r.key} data-primary={r.primary ? "true" : undefined} data-lit={lit === r.key ? "true" : undefined} {...pointer(r)}>
                  <th scope="row" data-indent={r.indent ? "true" : undefined}>
                    {r.label}
                    <Marks marks={r.marks} />
                    {r.sub ? <small>{r.sub}</small> : null}
                  </th>
                  {data.columns.map((c) => {
                    const cell = r.cells[c.key];
                    const cls = numeric.has(c.key) ? v.num : cell?.kind === "estimate" ? v.est : undefined;
                    return (
                      <td key={c.key} className={cls}>
                        {fmtCell(cell)}
                      </td>
                    );
                  })}
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      {data.footnotes.length ? (
        <ul className={v.notes} data-testid="footnotes">
          {data.footnotes.map((f, i) => (
            <li key={i}>
              {f.mark ? <sup>{f.mark}</sup> : null}
              {f.text}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
