/**
 * The orientation flip: a feature-major table turning around (OPENING_SEQUENCE §03, question 1.5).
 *
 * The storyboard is `orientation.transpose`'s own steps: the feature names are read as names
 * (the label column), the table turns (every cell (feature, sample) moves to (sample, feature) —
 * a true correspondence, so the motion asserts nothing false), and each new column is read as
 * numbers (the turned cells arrive as text; per-column coercion makes them numbers again). One
 * sample column is followed throughout: it becomes one row.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { fmtInt } from "../data";
import type { OrientationFixture } from "../types";
import { mix } from "./canvas";
import s from "./views.module.css";

const CW = 78;
const CH = 26;
const LW = 84;
const HH = 28;

/** A turned cell before coercion: the text the transpose hands over. */
function rawText(v: number | null): string {
  return v === null ? "" : String(v);
}

function perColumn(values: (number | null)[]): (v: number | null) => string {
  const nums = values.filter((v): v is number => v !== null);
  const big = nums.some((v) => Math.abs(v) >= 1000);
  const frac = nums.some((v) => !Number.isInteger(v));
  return (v) => {
    if (v === null) return "blank";
    if (big) return fmtInt(v);
    return frac ? v.toFixed(1) : String(v);
  };
}

export function TurnTable({ fx, last }: { fx: OrientationFixture; last: number }) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const n = fx.features.length;
  const m = fx.samples.length;
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));
  const track = fx.samples.indexOf(fx.tracked);

  // Before the turn a column (one sample) mixes run order, age and intensities, so each cell keeps
  // its own print; after it, a column is one feature, printed one way.
  const byFeature = useMemo(() => fx.cells.map((row) => perColumn(row)), [fx.cells]);
  const sampleFmt = useCallback((v: number | null) => (v === null ? "blank" : v >= 1000 ? fmtInt(v) : String(v)), []);

  const cellEls = useRef<(HTMLDivElement | null)[]>([]);
  const featEls = useRef<(HTMLDivElement | null)[]>([]);
  const sampEls = useRef<(HTMLDivElement | null)[]>([]);
  const outline = useRef<HTMLDivElement>(null);

  const draw = useCallback(
    (pos: number) => {
      const p = (pos * 3) / Math.max(1, last);
      // The turn is the step from 1 to 2; before and after it the grid stands still.
      const t = Math.max(0, Math.min(1, p - 1));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          const el = cellEls.current[i * m + j];
          if (!el) continue;
          const x = mix(LW + j * CW, LW + i * CW, t);
          const y = mix(HH + i * CH, HH + j * CH, t);
          el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        }
        const f = featEls.current[i];
        if (f) {
          const x = mix(0, LW + i * CW, t);
          const y = mix(HH + i * CH, 0, t);
          f.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
          f.style.width = `${mix(LW, CW, t).toFixed(1)}px`;
          f.style.height = `${mix(CH, HH, t).toFixed(1)}px`;
        }
      }
      for (let j = 0; j < m; j++) {
        const sEl = sampEls.current[j];
        if (!sEl) continue;
        const x = mix(LW + j * CW, 0, t);
        const y = mix(0, HH + j * CH, t);
        sEl.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        sEl.style.width = `${mix(CW, LW, t).toFixed(1)}px`;
        sEl.style.height = `${mix(HH, CH, t).toFixed(1)}px`;
      }
      const o = outline.current;
      if (o && track >= 0) {
        const x = mix(LW + track * CW, 0, t);
        const y = mix(0, HH + track * CH, t);
        const w = mix(CW, LW + n * CW, t);
        const h = mix(HH + n * CH, CH, t);
        o.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        o.style.width = `${w.toFixed(1)}px`;
        o.style.height = `${h.toFixed(1)}px`;
      }
    },
    [n, m, last, track],
  );
  const drawRef = useRef(draw);
  useEffect(() => {
    drawRef.current = draw;
    draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  const turned = at >= 2;
  const coerced = at >= 3;
  const named = at >= 1;
  const rows = turned ? fx.after.rows : fx.before.rows;
  const cols = turned ? fx.after.cols : fx.before.cols;

  return (
    <div className={s.tt} data-view="turn_table" data-state={at}>
      <div className={s.ttGrid} style={{ width: LW + Math.max(n, m) * CW, height: HH + Math.max(n, m) * CH }}>
        <div className={s.ttCorner} style={{ width: LW, height: HH }}>
          {turned ? fx.sample_column : fx.label_column}
        </div>
        <div ref={outline} className={s.ttOutline} aria-hidden="true" />
        {fx.features.map((f, i) => (
          <div
            key={f}
            ref={(el) => {
              featEls.current[i] = el;
            }}
            className={s.ttFeature}
            data-named={named || undefined}
            data-turned={turned || undefined}
          >
            {f}
          </div>
        ))}
        {fx.samples.map((smp, j) => (
          <div
            key={smp}
            ref={(el) => {
              sampEls.current[j] = el;
            }}
            className={s.ttSample}
            data-tracked={j === track || undefined}
            data-turned={turned || undefined}
          >
            {smp}
          </div>
        ))}
        {fx.cells.map((row, i) =>
          row.map((v, j) => (
            <div
              key={`${i}-${j}`}
              ref={(el) => {
                cellEls.current[i * m + j] = el;
              }}
              className={s.ttCell}
              data-raw={(turned && !coerced) || undefined}
              data-tracked={j === track || undefined}
            >
              {turned && !coerced ? rawText(v) : coerced ? byFeature[i]!(v) : sampleFmt(v)}
            </div>
          )),
        )}
      </div>
      <div className={s.rtFoot}>
        <span>
          {n} of {fmtInt(rows)} rows · {m} of {fmtInt(cols - 1)} {turned ? "feature" : "sample"} columns
        </span>
        <span className={s.rtRest}>
          {turned ? (coerced ? "each column read as numbers" : "turned cells arrive as text") : `rows are named by ${fx.label_column}`}
        </span>
      </div>
    </div>
  );
}
