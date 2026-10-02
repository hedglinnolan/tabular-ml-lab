/**
 * The orientation turn (lifted from /lab/m2; OPENING_SEQUENCE §03, question 1.5): a feature-major
 * table turning around. The preview of `set_orientation` (turbotab/core/structure_previews.py) is a
 * table_focus with two real frames — the table's corner as supplied, then turned — and the turn is
 * a true correspondence: every cell (feature, sample) moves to (sample, feature), so the motion
 * asserts nothing false. One sample column is followed throughout: it becomes one row.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { TableFocusView } from "../../../api/m1-stage-types";
import { CoachLayer } from "../coach/CoachLayer";
import { bandHeight, notesFor, type Span } from "../coach/place";
import { fmtInt } from "../format";
import { eased } from "../player";
import { localPos, type Track } from "../tracks";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../usePlayer";
import { mix } from "../views/canvas";
import { useSize } from "../views/geometry";
import s from "./reshape.module.css";

export interface TurnModel {
  labelColumn: string;
  sampleColumn: string;
  features: string[];
  samples: string[];
  /** cells[feature][sample], as supplied. */
  cells: unknown[][];
}

const same = (a: unknown, b: unknown) => a === b || (a == null && b == null) || String(a) === String(b);

/**
 * A turn, read from a table_focus: the last frame must be the first one transposed — its columns
 * the first frame's feature names, its rows the first frame's sample columns, every cell the same
 * value. Anything else is not a turn (null), and the generic table draws it.
 */
export function turnOf(view: TableFocusView): TurnModel | null {
  if (view.story.length < 2) return null;
  const a = view.story[0]!;
  const b = view.story[view.story.length - 1]!;
  if (a.columns.length < 2 || !a.rows.length) return null;
  const label = a.columns[0]!;
  const samples = a.columns.slice(1);
  const features = a.rows.map((r) => String(r.values[label]));
  const sampleColumn = b.columns[0]!;
  if (b.columns.length !== features.length + 1 || b.columns.slice(1).some((c, i) => c !== features[i])) return null;
  if (b.rows.length !== samples.length || b.rows.some((r, j) => String(r.values[sampleColumn]) !== samples[j])) return null;
  const cells = a.rows.map((r) => samples.map((smp) => r.values[smp]));
  for (let i = 0; i < features.length; i++)
    for (let j = 0; j < samples.length; j++) if (!same(b.rows[j]!.values[features[i]!], cells[i]![j])) return null;
  return { labelColumn: label, sampleColumn, features, samples, cells };
}

function print(v: unknown): string {
  if (v === null || v === undefined || v === "") return "blank";
  if (typeof v === "number") return Math.abs(v) >= 1000 ? fmtInt(v) : String(+v.toPrecision(4)).replace("-", "−");
  return String(v);
}

/** Each ring's share of the turn: rings overlap, so the whole turn still spans the step. */
const RING = 0.5;

/**
 * A cell's own progress through the turn, for a cell `d` places from the diagonal (of at most
 * `dMax`): the nearest ring first, each eased. A diagonal cell does not move.
 */
export function ringT(t: number, d: number, dMax: number): number {
  if (d <= 0 || dMax <= 1) return t * t * (3 - 2 * t);
  const start = ((d - 1) / (dMax - 1)) * (1 - RING);
  const u = Math.max(0, Math.min(1, (t - start) / RING));
  return u * u * (3 - 2 * u);
}

const CH = 6.6;
const ROW_H = 26;
const HEAD_H = 28;

export function TurnTable({
  track,
  model,
  globalLast,
}: {
  track: Track<TableFocusView>;
  model: TurnModel;
  globalLast: number;
}) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const localLast = track.states.length - 1;
  const toFour = useCallback(
    (pos: number) => (Math.max(0, Math.min(localLast, localPos(pos, globalLast, localLast))) * 3) / Math.max(1, localLast),
    [localLast, globalLast],
  );
  const at = Math.min(3, Math.round(toFour(ui.nearest)));
  const n = model.features.length;
  const m = model.samples.length;
  const texts = useMemo(() => model.cells.map((row) => row.map(print)), [model]);
  const CW = useMemo(
    () => Math.max(64, Math.round(Math.max(...model.features.map((f) => f.length), ...texts.flat().map((t) => t.length)) * CH + 18)),
    [model, texts],
  );
  const LW = useMemo(
    () => Math.max(84, Math.round(Math.max(model.labelColumn.length, model.sampleColumn.length, ...model.samples.map((x) => x.length), ...model.features.map((f) => f.length)) * CH + 14)),
    [model],
  );

  const cellEls = useRef<(HTMLDivElement | null)[]>([]);
  const featEls = useRef<(HTMLDivElement | null)[]>([]);
  const sampEls = useRef<(HTMLDivElement | null)[]>([]);
  const outline = useRef<HTMLDivElement>(null);

  const draw = useCallback(
    (pos: number) => {
      // The turn is the step from 1 to 2; before and after it the grid stands still.
      const t = Math.max(0, Math.min(1, toFour(pos) - 1));
      // The labels are the grid's ring −1: feature i's name and sample i's swap as a pair.
      const rings = Math.max(n, m);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          const el = cellEls.current[i * m + j];
          if (!el) continue;
          // Ring by ring out from the diagonal: a cell and its mirror swap while the others wait,
          // so the cells never all meet on the diagonal at once mid-turn.
          const tc = ringT(t, Math.abs(i - j), rings);
          const x = mix(LW + j * CW, LW + i * CW, tc);
          const y = mix(HEAD_H + i * ROW_H, HEAD_H + j * ROW_H, tc);
          el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        }
        const f = featEls.current[i];
        if (f) {
          const tf = ringT(t, i + 1, rings);
          f.style.transform = `translate(${mix(0, LW + i * CW, tf).toFixed(1)}px, ${mix(HEAD_H + i * ROW_H, 0, tf).toFixed(1)}px)`;
          f.style.width = `${mix(LW, CW, tf).toFixed(1)}px`;
          f.style.height = `${mix(ROW_H, HEAD_H, tf).toFixed(1)}px`;
        }
      }
      for (let j = 0; j < m; j++) {
        const el = sampEls.current[j];
        if (!el) continue;
        const ts = ringT(t, j + 1, rings);
        el.style.transform = `translate(${mix(LW + j * CW, 0, ts).toFixed(1)}px, ${mix(0, HEAD_H + j * ROW_H, ts).toFixed(1)}px)`;
        el.style.width = `${mix(CW, LW, ts).toFixed(1)}px`;
        el.style.height = `${mix(HEAD_H, ROW_H, ts).toFixed(1)}px`;
      }
      const o = outline.current;
      if (o) {
        o.style.transform = `translate(${mix(LW, 0, t).toFixed(1)}px, ${mix(0, HEAD_H, t).toFixed(1)}px)`;
        o.style.width = `${mix(CW, LW + n * CW, t).toFixed(1)}px`;
        o.style.height = `${mix(HEAD_H + n * ROW_H, ROW_H, t).toFixed(1)}px`;
      }
    },
    [n, m, toFour, CW, LW],
  );
  useEffect(() => {
    draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  // The coach (M2_CONTRACT §6): a note about the label column points at the corner that names it;
  // any other note stands on the band (the cells move under it, so it is pinned to none of them).
  const [root, { w }] = useSize<HTMLDivElement>();
  const notes = notesFor(track.view.coach, false, w);
  const band = bandHeight(notes.length);
  const spans: (Span | null)[] = notes.map((note) =>
    note.anchor.kind === "column" && (note.anchor.ref === model.labelColumn || note.anchor.ref === model.sampleColumn)
      ? { x0: 6, x1: LW - 6, y: band + 4, mark: "none" }
      : null,
  );

  const turned = at >= 2;
  const named = at >= 1;
  return (
    <div className={s.tt} ref={root} data-view="turn_table" data-state={at} style={band ? { paddingTop: band } : undefined}>
      <div className={s.ttScroll}>
        <div className={s.ttGrid} style={{ width: LW + Math.max(n, m) * CW, height: HEAD_H + Math.max(n, m) * ROW_H }}>
          <div className={s.ttCorner} style={{ width: LW, height: HEAD_H }}>
            {turned ? model.sampleColumn : model.labelColumn}
          </div>
          <div ref={outline} className={s.ttOutline} aria-hidden="true" />
          {model.features.map((f, i) => (
            <div
              key={`f-${f}-${i}`}
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
          {model.samples.map((smp, j) => (
            <div
              key={`s-${smp}-${j}`}
              ref={(el) => {
                sampEls.current[j] = el;
              }}
              className={s.ttSample}
              data-tracked={j === 0 || undefined}
              data-turned={turned || undefined}
            >
              {smp}
            </div>
          ))}
          {texts.map((row, i) =>
            row.map((text, j) => (
              <div
                key={`${i}-${j}`}
                ref={(el) => {
                  cellEls.current[i * m + j] = el;
                }}
                className={s.ttCell}
                style={{ width: CW }}
              >
                {text === "blank" ? <span className={s.blank}>blank</span> : text}
              </div>
            )),
          )}
        </div>
      </div>
      <div className={s.rtFoot}>
        <span>
          {n} features × {m} samples, of the table&apos;s corner
        </span>
        <span className={s.rtRest}>{turned ? "each sample column became one row" : `rows are named by ${model.labelColumn}`}</span>
      </div>
      {notes.length ? <CoachLayer notes={notes} spans={spans} width={w} /> : null}
    </div>
  );
}
