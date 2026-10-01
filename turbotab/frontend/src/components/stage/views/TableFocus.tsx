/**
 * table_focus — the working table narrowed to what the choice touches: ≤ 12 columns, ≤ 8 rows,
 * and a count of the rest, so the view reads the same at 20 or 20,000 columns.
 *
 * Columns morph by identity, as in the `scrub` prototype: a column keeps its place when the choice
 * renames it (`fat_total` → `fat_total_adj`, the header relabels in place), a column the choice
 * removes folds away where it stood, and the rest close up — never a neighbor taking over its slot
 * or its tint. Cells show a real state's values only, tinted where the choice changes them, keyed
 * by (row, column); rows that leave whole are struck through.
 */
import { useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { TableFocusView } from "../../../api/m1-stage-types";
import { useTransitions } from "../../../motion/prefs";
import { cellFormatter, fmtInt } from "../format";
import { localPos, type TableState, type Track } from "../tracks";
import { usePlayerStore, usePlayerUi } from "../usePlayer";
import s from "./views.module.css";

interface Props {
  view: TableFocusView;
  state: TableState;
  /** Tint the cells the choice changes (any state but "your data now"). */
  showChanges: boolean;
  compact?: boolean;
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

function show(value: unknown, format: (v: number) => string) {
  if (value === null || value === undefined || value === "") return <span className={s.blank}>blank</span>;
  if (typeof value === "number") return <span className="num">{format(value)}</span>;
  if (typeof value === "boolean") return <span className="num">{value ? "True" : "False"}</span>;
  return <span className="num">{String(value)}</span>;
}

/**
 * Each column's identity: its name in "your data now". A column the choice renames keeps the
 * identity of the one it replaces (same position when nothing leaves, else the name it extends).
 */
export function identities(before: string[], after: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const c of before) out.set(c, c);
  const same = before.length === after.length;
  after.forEach((c, i) => {
    if (out.has(c)) return;
    const was = same ? before[i] : before.find((b) => c.startsWith(`${b}_`));
    out.set(c, was !== undefined && !after.includes(was) ? was : c);
  });
  return out;
}

export function TableFocus({ view, state, showChanges, compact = false }: Props) {
  const t = useTransitions();
  const before = view.columns_before.length ? view.columns_before : view.columns_after;
  const idOf = useMemo(() => identities(before, view.columns_after), [before, view.columns_after]);
  // Every column either state shows, by identity, in the order "your data now" has them.
  const slots = useMemo(() => {
    const order: string[] = [];
    for (const c of [...before, ...view.columns_after]) {
      const id = idOf.get(c) ?? c;
      if (!order.includes(id)) order.push(id);
    }
    return compact ? order.slice(0, 5) : order;
  }, [before, view.columns_after, idOf, compact]);
  const nameIn = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of state.columns) m.set(idOf.get(c) ?? c, c);
    return m;
  }, [state.columns, idOf]);
  const cols = slots.map((id) => nameIn.get(id)).filter((c): c is string => c !== undefined);
  const rowIds = (compact ? view.rows.slice(0, 5) : view.rows).map((r) => r.row_id);
  const changed = useMemo(() => new Set(view.changed.map(([r, c]) => `${r}|${c}`)), [view.changed]);
  const prefix = sharedPrefix(cols);
  const formats = useMemo(
    () =>
      Object.fromEntries(
        cols.map((c) => [c, columnFormat(rowIds.map((id) => state.values.get(id)?.[c]))]),
      ),
    [cols, rowIds, state],
  );
  const shownCols = cols.length;
  const rest = Math.max(0, view.n_affected_columns - shownCols);

  return (
    <div className={s.tableWrap} data-view="table_focus">
      <div className={s.tableScroll}>
        <table className={compact ? s.tableCompact : s.table}>
          <thead>
            <tr>
              <th className={s.rowHead} scope="col">
                {prefix ? <span className={s.prefix}>{prefix}…</span> : "row"}
              </th>
              {slots.map((id) => {
                const c = nameIn.get(id);
                return (
                  <th
                    key={id}
                    scope="col"
                    className={c === undefined ? `${s.colHead} ${s.colGone}` : s.colHead}
                    title={c ?? id}
                    aria-hidden={c === undefined || undefined}
                  >
                    <span className={s.colSlot}>
                      <AnimatePresence initial={false} mode="popLayout">
                        <motion.span
                          key={c ?? id}
                          className={showChanges && c !== undefined && c !== id ? s.colNameNew : s.colName}
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -4 }}
                          transition={t.arrive}
                        >
                          {prefix && (c ?? id).startsWith(prefix) ? (c ?? id).slice(prefix.length) : (c ?? id)}
                        </motion.span>
                      </AnimatePresence>
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rowIds.map((id) => (
              <tr key={id} className={state.gone ? s.rowGone : undefined}>
                <th scope="row" className={s.rowId}>
                  {id}
                  {state.gone ? <span className={s.leaves}>leaves</span> : null}
                </th>
                {slots.map((col) => {
                  const c = nameIn.get(col);
                  if (c === undefined) {
                    // Folded away where it stood: no value, and never a tint.
                    return (
                      <td key={col} className={`${s.cell} ${s.cellGone}`} aria-hidden="true">
                        <span className={s.colSlot} />
                      </td>
                    );
                  }
                  const hit = showChanges && (changed.has(`${id}|${c}`) || changed.has(`${id}|${col}`));
                  return (
                    <td key={col} className={hit ? `${s.cell} ${s.cellChanged}` : s.cell}>
                      <span className={s.colSlot}>{show(state.values.get(id)?.[c], formats[c]!)}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={s.tableFoot}>
        <span>
          {shownCols} {shownCols === 1 ? "column" : "columns"} · {rowIds.length} sample rows
        </span>
        {rest > 0 ? (
          <span className={s.tableRest}>{fmtInt(rest)} more change the same way</span>
        ) : view.changed.length === 0 ? (
          <span className={s.tableRest}>no value changes</span>
        ) : null}
      </div>
    </div>
  );
}

/** The table at whichever real state the player shows. */
export function TableTrack({
  track,
  globalLast,
  compact,
}: {
  track: Track<TableFocusView>;
  globalLast: number;
  compact?: boolean;
}) {
  const ui = usePlayerUi(usePlayerStore());
  const localLast = track.states.length - 1;
  const shown = Math.min(localLast, Math.round(localPos(ui.nearest, globalLast, localLast)));
  return (
    <div className={s.fill} data-state={shown}>
      <TableFocus view={track.view} state={track.states[shown]!} showChanges={shown > 0} compact={compact} />
    </div>
  );
}
