/**
 * The working table under a reshape (DESIGN_LANGUAGE §05.2, item 4): a window of the table, five
 * units' rows, played through the method's storyboard. Rows are elements that persist from the
 * file to the settled table; their positions glide between real states; their values are only ever
 * a real state's values, and a value that changes rolls (the old one leaves upward, the new one
 * arrives) instead of counting through numbers nobody computed.
 *
 * Wide tables read the same: the window shows the affected columns only (≤ 8 beside the
 * identifier), and the footer counts the rest (BLUEPRINT §11.3).
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useTransitions } from "../../../motion/prefs";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { columnFormatter, fmtInt, sharedPrefix } from "../data";
import {
  changedAt,
  kindOf,
  layout,
  lerpLayout,
  RH,
  sourceRows,
  valuesAt,
  windowRows,
  type Layout,
} from "../reshape";
import type { MethodFixture, ReshapeFixture } from "../types";
import { useMorph } from "../useMorph";
import s from "./views.module.css";

const CH = 6.7; // mono 11px advance, px
const ROW_COL = 74;

function widthFor(name: string, values: string[]): number {
  const longest = Math.max(name.length, ...values.map((v) => v.length));
  return Math.round(longest * CH + 13);
}

function Roll({ text, className }: { text: string; className?: string }) {
  const t = useTransitions();
  return (
    <span className={s.roll}>
      <AnimatePresence initial={false} mode="popLayout">
        <motion.span
          key={text}
          className={className}
          initial={{ y: "70%", opacity: 0 }}
          animate={{ y: "0%", opacity: 1 }}
          exit={{ y: "-70%", opacity: 0 }}
          transition={t.row}
        >
          {text}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

export function ReshapeTable({ fx, method, last }: { fx: ReshapeFixture; method: MethodFixture; last: number }) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const rows = useMemo(() => windowRows(fx), [fx]);
  const cols = useMemo(() => [...fx.columns.record, ...fx.columns.shown], [fx]);
  // A wide table's names share a prefix ("ft_"); at ≤ 8 shown columns there is room to print them whole.
  const prefix = fx.columns.shown.some((c) => c.length > 14) ? sharedPrefix(fx.columns.shown) : "";
  const tracked = fx.window.tracked;
  const kind = kindOf(method);
  // The state the text shows: the nearest real state, mapped onto this view's four.
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));

  // Formats and widths per column, from every value any method can show.
  const formats = useMemo(() => {
    const out: Record<string, (v: unknown) => string> = {};
    for (const c of cols) {
      const own = rows.map((r) => r.values[c] ?? null);
      out[`0|${c}`] = columnFormatter(own) as (v: unknown) => string;
      const after = Object.values(method.units).map((u) => u.values[c] ?? null);
      out[`1|${c}`] = columnFormatter([...own, ...after].filter((v) => v !== null)) as (v: unknown) => string;
    }
    return out;
  }, [cols, rows, method]);
  const baseWidths = useMemo(() => {
    const w: Record<string, number> = {};
    for (const c of cols) {
      const shown = [...rows.map((r) => formats[`1|${c}`]!(r.values[c] ?? null))];
      w[c] = widthFor(prefix && c.startsWith(prefix) ? c.slice(prefix.length) : c, shown);
    }
    return w;
  }, [cols, rows, formats, prefix]);
  const idWidth = widthFor(fx.id_column, fx.window.units);

  const states = useMemo(
    () => [0, 1, 2, 3].map((st) => layout(fx, method, st, baseWidths)),
    [fx, method, baseWidths],
  );

  const root = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const rowEls = useRef<(HTMLDivElement | null)[]>([]);
  const frameEls = useRef<(HTMLDivElement | null)[]>([]);
  const gapEls = useRef<(HTMLDivElement | null)[]>([]);

  const paint = useCallback(
    (l: Layout) => {
      const el = root.current;
      if (!el) return;
      cols.forEach((c, i) => el.style.setProperty(`--wc${i}`, `${l.widths[c]!.toFixed(1)}px`));
      // The table's own height follows the rows (the settled table is half as tall); the body
      // keeps the tallest state's height so nothing below reflows during a flip (DRIVE_RUBRIC §5.7).
      if (body.current) body.current.style.setProperty("--rows-h", `${l.height.toFixed(1)}px`);
      l.rows.forEach((g, i) => {
        const r = rowEls.current[i];
        if (!r) return;
        r.style.transform = `translateY(${g.y.toFixed(2)}px)`;
        r.style.opacity = g.alpha.toFixed(3);
        r.style.setProperty("--struck", g.struck.toFixed(3));
      });
      l.frames.forEach((f, i) => {
        const r = frameEls.current[i];
        if (!r) return;
        r.style.transform = `translateY(${f.y.toFixed(2)}px)`;
        r.style.height = `${f.h.toFixed(2)}px`;
        r.style.opacity = f.alpha.toFixed(3);
      });
      l.gaps.forEach((g, i) => {
        const r = gapEls.current[i];
        if (!r) return;
        r.style.transform = `translateY(${g.y.toFixed(2)}px)`;
        r.style.opacity = g.alpha.toFixed(3);
      });
    },
    [cols],
  );

  const target = useCallback(
    (pos: number): Layout => {
      const p = (pos * 3) / Math.max(1, last);
      const i = Math.max(0, Math.min(3, Math.floor(p)));
      const f = p - i;
      return i >= 3 ? states[3]! : lerpLayout(states[i]!, states[i + 1]!, f);
    },
    [states, last],
  );
  const drawRef = useRef<(pos: number) => void>(() => {});
  const blend = useMorph<Layout>(
    lerpLayout,
    useCallback(() => drawRef.current(eased(store.get().pos)), [store]),
  );
  const draw = useCallback(
    (pos: number) => paint(blend(target(pos), method.key)),
    [paint, blend, target, method.key],
  );
  useEffect(() => {
    drawRef.current = draw;
  }, [draw]);
  usePlayerFrame(store, draw);

  const nRows = at === 3 ? method.n_after : fx.dataset.rows;
  const shownRows = at === 3 ? fx.window.units.length : rows.length;
  const folded = at >= 2 ? method.folds : [];
  const nMore = fx.columns.more_count ?? fx.columns.more.length;

  return (
    <div className={s.rt} ref={root} data-view="reshape_table" data-state={at}>
      <div className={s.rtHead} role="row">
        <span className={s.rtRowHead} style={{ width: ROW_COL }}>
          {at >= 2 && kind === "combine" ? "from rows" : "row"}
        </span>
        <span className={s.rtIdHead} style={{ width: idWidth }}>
          {fx.id_column}
        </span>
        {cols.map((c, i) => (
          <span
            key={c}
            className={folded.includes(c) ? `${s.rtColHead} ${s.rtFolding}` : s.rtColHead}
            style={{ width: `var(--wc${i})` }}
            title={c}
          >
            <span className={s.rtSlot}>{prefix && c.startsWith(prefix) ? c.slice(prefix.length) : c}</span>
          </span>
        ))}
      </div>
      <div className={s.rtBody} ref={body} style={{ height: Math.max(...states.map((x) => x.height)) }}>
        <div className={s.rtExtent} aria-hidden="true" />
        {fx.window.units.map((u, i) => (
          <div
            key={u}
            ref={(el) => {
              frameEls.current[i] = el;
            }}
            className={u === tracked ? `${s.rtFrame} ${s.rtFrameTracked}` : s.rtFrame}
            aria-hidden="true"
          />
        ))}
        {states[0]!.gaps.map((g, i) => (
          <div
            key={`gap-${i}`}
            ref={(el) => {
              gapEls.current[i] = el;
            }}
            className={s.rtGap}
            aria-hidden="true"
          >
            ⋮ {fmtInt(g.skipped)} rows
          </div>
        ))}
        {rows.map((r, i) => {
          const values = valuesAt(method, r, at);
          const f = at >= 2 ? "1" : "0";
          return (
            <div
              key={r.row}
              ref={(el) => {
                rowEls.current[i] = el;
              }}
              className={s.rtRow}
              data-tracked={r.unit === tracked || undefined}
              data-row={r.row}
              data-unit={r.unit}
              data-k={r.k}
              style={{ height: RH }}
            >
              <span className={s.rtRowId} style={{ width: ROW_COL }}>
                <Roll text={sourceRows(fx, method, r, at)} />
              </span>
              <span className={s.rtId} style={{ width: idWidth }}>
                {r.unit}
              </span>
              {cols.map((c, j) => {
                const v = values[c];
                const text = v === undefined ? "" : formats[`${f}|${c}`]!(v ?? null);
                const hit = changedAt(method, r, at, c);
                return (
                  <span
                    key={c}
                    className={hit ? `${s.rtCell} ${s.rtChanged}` : s.rtCell}
                    style={{ width: `var(--wc${j})` }}
                  >
                    <Roll text={text} className={text === "blank" ? s.blank : undefined} />
                  </span>
                );
              })}
              <span className={s.rtStrike} aria-hidden="true" />
            </div>
          );
        })}
      </div>
      <div className={s.rtFoot}>
        <span>
          {shownRows} of {fmtInt(nRows)} rows ·{" "}
          {at < 2
            ? `${cols.length} of ${fmtInt(fx.columns.n_changed + fx.columns.record.length)} columns this choice touches`
            : kind === "combine"
              ? `${cols.length - folded.length} of ${fmtInt(fx.columns.n_changed)} changed columns`
              : "no value changes; whole rows leave"}
        </span>
        <span className={s.rtRest}>
          {nMore > 0 && kind === "combine" ? `${fmtInt(nMore)} more ${method.key === "mean" ? "average" : "change"} the same way` : ""}
        </span>
      </div>
    </div>
  );
}
