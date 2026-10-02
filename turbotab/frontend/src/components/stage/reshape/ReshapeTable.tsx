/**
 * The working table under a reshape (lifted from /lab/m2; DESIGN_LANGUAGE §05.2, item 4): the
 * units the preview shows, played through the method's storyboard. Rows are elements that persist
 * from the file to the settled table; their positions glide between real states; their values are
 * only ever a real state's values, and a value that changes rolls (the old one leaves upward, the
 * new one arrives) instead of counting through numbers nobody computed. The "from rows" column is
 * the row map made visible: a combined row says which rows of the file it stands for.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { TableFocusView } from "../../../api/m1-stage-types";
import { useTransitions } from "../../../motion/prefs";
import { CoachLayer } from "../coach/CoachLayer";
import { bandHeight, notesFor, type Span } from "../coach/place";
import { fmtInt } from "../format";
import { eased } from "../player";
import { localPos, type Track } from "../tracks";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../usePlayer";
import { useSize } from "../views/geometry";
import { useMorph } from "../views/useMorph";
import {
  changedAt,
  layout,
  lerpLayout,
  RH,
  sourceRows,
  valuesAt,
  type Layout,
  type ReshapeModel,
} from "./reshape";
import s from "./reshape.module.css";

const CH = 6.7; // mono 11.5 px advance, px
const ROW_COL = 74;

function widthFor(name: string, values: string[]): number {
  const longest = Math.max(name.length, ...values.map((v) => v.length));
  return Math.round(longest * CH + 14);
}

/** How many decimals a value needs (capped), so a column prints one way within a state. */
function decimalsOf(v: number, cap = 2): number {
  for (let d = 0; d < cap; d++) if (Math.abs(v * 10 ** d - Math.round(v * 10 ** d)) < 1e-6) return d;
  return cap;
}

/** One format per column: `5` and `5.39` never sit side by side as if different kinds. */
function columnFormatter(values: unknown[], cap = 2): (v: unknown) => string {
  const nums = values.filter((v): v is number => typeof v === "number");
  const d = nums.reduce((m, v) => Math.max(m, decimalsOf(v, cap)), 0);
  return (v) => {
    if (v === null || v === undefined || v === "") return "blank";
    if (typeof v === "number")
      return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
    // A date read as a timestamp at midnight is printed as the date it is.
    return String(v).replace(/^(\d{4}-\d{2}-\d{2})T00:00:00(\.0+)?$/, "$1");
  };
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

export function ReshapeTable({
  track,
  model,
  globalLast,
  compact = false,
}: {
  track: Track<TableFocusView>;
  model: ReshapeModel;
  globalLast: number;
  compact?: boolean;
}) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const localLast = track.states.length - 1;
  // The four real states map onto the track's states (now · records · combined · with this choice).
  const toFour = useCallback(
    (pos: number) => (Math.max(0, Math.min(localLast, localPos(pos, globalLast, localLast))) * 3) / Math.max(1, localLast),
    [localLast, globalLast],
  );
  const at = Math.min(3, Math.round(toFour(ui.nearest)));
  const rows = model.rows;
  const cols = model.columns;
  const kind = model.kind;

  const formats = useMemo(() => {
    const out: Record<string, (v: unknown) => string> = {};
    for (const c of [model.idColumn, ...cols]) {
      const own = rows.map((r) => r.values[c] ?? null);
      const after = Object.values(model.settled).map((u) => u.values[c] ?? null);
      out[`0|${c}`] = columnFormatter(own);
      out[`1|${c}`] = columnFormatter([...own, ...after].filter((v) => v !== null));
    }
    return out;
  }, [model, cols, rows]);
  const widths = useMemo(() => {
    const w: Record<string, number> = {};
    for (const c of cols) {
      const shown = [
        ...rows.map((r) => formats[`1|${c}`]!(r.values[c] ?? null)),
        ...Object.values(model.settled).map((u) => formats[`1|${c}`]!(u.values[c] ?? null)),
      ];
      w[c] = widthFor(c, shown);
    }
    return w;
  }, [cols, rows, formats, model]);
  const idWidth = widthFor(model.idColumn, model.units);
  const sourceWidth = Math.max(
    ROW_COL,
    ...Object.values(model.settled).map((u) => Math.round(u.sources.join("+").length * CH + 16)),
  );

  const states = useMemo(() => [0, 1, 2, 3].map((st) => layout(model, st)), [model]);

  const body = useRef<HTMLDivElement>(null);
  const rowEls = useRef<(HTMLDivElement | null)[]>([]);
  const frameEls = useRef<(HTMLDivElement | null)[]>([]);
  const gapEls = useRef<(HTMLDivElement | null)[]>([]);

  const paint = useCallback((l: Layout) => {
    // The table's own extent follows the rows (the settled table is shorter); the body keeps the
    // tallest state's height so nothing below reflows during a flip (DRIVE_RUBRIC §5.7).
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
  }, []);

  const target = useCallback(
    (pos: number): Layout => {
      const p = toFour(pos);
      const i = Math.max(0, Math.min(3, Math.floor(p)));
      return i >= 3 ? states[3]! : lerpLayout(states[i]!, states[i + 1]!, p - i);
    },
    [states, toFour],
  );
  // Another method of the same question morphs from what is drawn (no storyboard replay).
  const drawRef = useRef<(pos: number) => void>(() => {});
  const blend = useMorph<Layout>(
    lerpLayout,
    useCallback(() => drawRef.current(eased(store.get().pos)), [store]),
  );
  const morphKey = `${kind}|${Object.values(model.settled)
    .map((u) => u.row)
    .join(",")}|${JSON.stringify(Object.values(model.settled)[0]?.values ?? {})}`;
  const draw = useCallback((pos: number) => paint(blend(target(pos), morphKey)), [paint, blend, target, morphKey]);
  useEffect(() => {
    drawRef.current = draw;
  }, [draw]);
  usePlayerFrame(store, draw);

  // The coach (M2_CONTRACT §6): a note about a column points at its header.
  const [root, { w }] = useSize<HTMLDivElement>();
  const notes = notesFor(track.view.coach, compact, w);
  const band = bandHeight(notes.length);
  const spans: (Span | null)[] = notes.map((n) => {
    if (n.anchor.kind !== "column" || typeof n.anchor.ref !== "string") return null;
    const ref = n.anchor.ref;
    if (ref === model.idColumn) return { x0: sourceWidth + 6, x1: sourceWidth + idWidth - 6, y: band + 4, mark: "none" };
    const i = cols.indexOf(ref);
    if (i < 0) return null;
    const x0 = sourceWidth + idWidth + cols.slice(0, i).reduce((a, c) => a + widths[c]!, 0);
    return { x0: x0 + 6, x1: x0 + widths[ref]! - 6, y: band + 4, mark: "none" };
  });

  const settledRows = at === 3 ? model.units.length : rows.length;
  const f = at >= 2 ? "1" : "0";

  return (
    <div className={s.rt} ref={root} data-view="reshape_table" data-state={at} data-kind={kind} style={band ? { paddingTop: band } : undefined}>
      <div className={s.scroll}>
        <div className={s.rtHead} role="row">
          <span className={s.rtRowHead} style={{ width: sourceWidth }}>
            {at >= 2 && kind === "combine" ? "from rows" : "row"}
          </span>
          <span className={s.rtIdHead} style={{ width: idWidth }}>
            {model.idColumn}
          </span>
          {cols.map((c) => (
            <span key={c} className={s.rtColHead} style={{ width: widths[c] }} title={c}>
              {c}
            </span>
          ))}
        </div>
        <div className={s.rtBody} ref={body} style={{ height: Math.max(...states.map((x) => x.height)) }}>
          <div className={s.rtExtent} aria-hidden="true" />
          {model.units.map((u, i) => (
            <div
              key={u}
              ref={(el) => {
                frameEls.current[i] = el;
              }}
              className={s.rtFrame}
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
              ⋮ {fmtInt(g.skipped)} {g.skipped === 1 ? "row" : "rows"}
            </div>
          ))}
          {rows.map((r, i) => {
            const values = valuesAt(model, r, at);
            return (
              <div
                key={r.row}
                ref={(el) => {
                  rowEls.current[i] = el;
                }}
                className={s.rtRow}
                data-row={r.row}
                data-unit={r.unit}
                data-k={r.k}
                style={{ height: RH }}
              >
                <span className={s.rtRowId} style={{ width: sourceWidth }}>
                  <Roll text={sourceRows(model, r, at)} />
                </span>
                <span className={s.rtId} style={{ width: idWidth }}>
                  {r.unit}
                </span>
                {cols.map((c) => {
                  const v = values[c];
                  const text = v === undefined ? "" : formats[`${f}|${c}`]!(v ?? null);
                  return (
                    <span
                      key={c}
                      className={changedAt(model, r, at, c) ? `${s.rtCell} ${s.rtChanged}` : s.rtCell}
                      style={{ width: widths[c] }}
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
      </div>
      <div className={s.rtFoot}>
        <span>
          {settledRows} {settledRows === 1 ? "row" : "rows"} of {model.units.length} units ·{" "}
          {at < 2
            ? `${cols.length} of the columns the choice touches`
            : kind === "combine"
              ? `${cols.length} columns combined`
              : "no value changes; whole records leave"}
        </span>
        <span className={s.rtRest}>
          {kind === "combine" && model.nAffected > cols.length
            ? `${fmtInt(model.nAffected - cols.length)} more combine the same way`
            : ""}
        </span>
      </div>
      {notes.length ? <CoachLayer notes={notes} spans={spans} width={w} /> : null}
    </div>
  );
}
