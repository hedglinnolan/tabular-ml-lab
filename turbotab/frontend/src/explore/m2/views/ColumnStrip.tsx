/**
 * Every column of the table as one strip, by what the choice does to it: changes, folds away (a
 * record-level column a combined row no longer has), passes through (constant within a unit), or
 * names the unit. The ticks mark the columns the table shows. It reads the same at 17 columns and at
 * 20,002 — which is the point: the window shows the affected columns, the strip accounts for the rest.
 */
import { usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { fmtInt } from "../data";
import { kindOf } from "../reshape";
import type { MethodFixture, ReshapeFixture } from "../types";
import s from "./views.module.css";

type Seg = { kind: "id" | "record" | "constant" | "measure"; count: number; shown: number[] };

function segments(fx: ReshapeFixture): { segs: Seg[]; total: number } {
  const all = fx.columns.all;
  const kinds = fx.columns.kinds;
  const shown = new Set([...fx.columns.record, ...fx.columns.shown]);
  if (all) {
    const segs: Seg[] = [];
    all.forEach((c) => {
      const kind = kinds[c] ?? "measure";
      const lastSeg = segs[segs.length - 1];
      const mark = shown.has(c);
      if (lastSeg && lastSeg.kind === kind) {
        if (mark) lastSeg.shown.push(lastSeg.count);
        lastSeg.count += 1;
      } else segs.push({ kind, count: 1, shown: mark ? [0] : [] });
    });
    return { segs, total: all.length };
  }
  // A wide table lists its kinds, not every name: id, record, then the measures as one run.
  const n = fx.columns.n_all ?? 0;
  const measures = n - 1 - fx.columns.record.length;
  const ticks = fx.columns.shown.map((c) => Number(c.replace(/\D/g, "")) - 1).filter((i) => i >= 0);
  return {
    segs: [
      { kind: "id", count: 1, shown: [] },
      { kind: "record", count: fx.columns.record.length, shown: [0] },
      { kind: "measure", count: measures, shown: ticks },
    ],
    total: n,
  };
}

export function ColumnStrip({ fx, method, last }: { fx: ReshapeFixture; method: MethodFixture; last: number }) {
  const ui = usePlayerUi(usePlayerStore());
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));
  const { segs, total } = segments(fx);
  const combine = kindOf(method) === "combine";
  const after = at >= 2;
  const nChange = fx.columns.n_changed;
  const nFold = method.folds.length;
  const nPass = fx.columns.constant.length;
  const summary = !after
    ? `${fmtInt(total)} columns`
    : combine
      ? `${fmtInt(total)} → ${fmtInt(total - nFold)} columns: ${fmtInt(nChange)} change${nFold ? ` · ${nFold} fold away` : ""}${nPass ? ` · ${nPass} pass through` : ""}`
      : `${fmtInt(total)} columns: no value changes; rows leave whole`;
  return (
    <div className={s.cs} data-view="column_strip" data-state={at}>
      <div className={s.csBar} aria-hidden="true">
        {segs.map((g, i) => {
          const folding = after && combine && g.kind === "record";
          const changing = after && combine && g.kind === "measure";
          return (
            <span
              key={i}
              className={s.csSeg}
              data-kind={g.kind}
              data-changing={changing || undefined}
              data-folding={folding || undefined}
              style={{ flexGrow: folding ? 0.0001 : g.count, minWidth: folding ? 0 : g.kind === "measure" ? 6 : 5 }}
            >
              {g.shown.map((t) => (
                <i key={t} className={s.csTick} style={{ left: `${((t + 0.5) / g.count) * 100}%` }} />
              ))}
            </span>
          );
        })}
      </div>
      <span className={s.csLabel}>{summary}</span>
    </div>
  );
}
