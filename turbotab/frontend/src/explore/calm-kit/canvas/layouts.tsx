/**
 * The five layouts of the canvas grammar (FOUNDATION §5). Each draws the engine's views for one
 * option; one flip and one storyboard drive every view at once; views are linked (pointing at a
 * column or a flow step lights it in every view).
 */
import { useState, type ReactNode } from "react";
import type { Angle, ConsequenceView, Option, Step, StripColumn } from "../fixture";
import { route } from "../router";
import { fmtNum, fmtR, plain } from "../text";
import k from "../kit.module.css";
import { Cells, FlowBars, Hist, Lineage, Scatter, type Linked } from "./views";

export interface LayoutProps {
  option: Option;
  after: boolean;
  frame: number | null;
  linked: Linked;
}

/** One engine view, drawn by its kind; `bare` drops its title (an Angles panel's question heads it). */
export function View({
  view,
  after,
  frame,
  linked,
  touch,
  keep,
  bare = false,
}: {
  view: ConsequenceView;
  after: boolean;
  frame: number | null;
  linked: Linked;
  touch?: string[];
  /** Columns named in a lineage even where the others are grouped (the question's own). */
  keep?: string[];
  bare?: boolean;
}) {
  switch (view.kind) {
    case "distribution": {
      const f = after && frame !== null ? view.story?.[frame] : null;
      const cut = view.before_label === "Every measured value" || view.after_label.startsWith("Kept");
      return (
        <Hist
          title={bare ? undefined : plain(view.title)}
          before={view.before}
          after={f ? f.hist : view.after}
          mode={cut ? "cut" : "transform"}
          after_on={after}
          marks={view.marks}
          unit={plain(view.column)}
        />
      );
    }
    case "relationship":
      return (
        <div className={k.panel}>
          {bare ? null : <h3>{plain(view.title)}</h3>}
          <Scatter view={view} after_on={after} frame={frame} />
        </div>
      );
    case "row_flow":
      return <FlowBars view={view} after_on={after} linked={linked} bare={bare} />;
    case "lineage":
      return (
        <div className={k.panel}>
          {bare ? null : <h3>{plain(view.title)}</h3>}
          <Lineage view={view} after_on={after} frame={frame} touch={touch} keep={keep} linked={linked} />
        </div>
      );
    case "table_focus":
      return <Cells view={view} after_on={after} bare={bare} />;
  }
}

export function Views({ views, keep, ...p }: { views: ConsequenceView[]; keep?: string[] } & Omit<LayoutProps, "option">) {
  return (
    <div className={k.panels}>
      {views.map((v, i) => (
        <View key={`${v.kind}-${i}`} view={v} after={p.after} frame={p.frame} linked={p.linked} keep={keep} />
      ))}
    </div>
  );
}

// ── Strip ────────────────────────────────────────────────────────────────────

const STRIP_TOP = 12;

/** A column's measure: now only, or before → after with the choice. */
function measure(c: StripColumn, after: boolean): string {
  if (!after) return `mean ${fmtNum(c.mean_before)} · SD ${fmtNum(c.sd_before)}`;
  // the per-calorie columns change scale: say so by the mean; otherwise by the spread
  const rescaled = Math.abs(c.mean_after) < Math.abs(c.mean_before) / 50;
  if (rescaled) return `mean ${fmtNum(c.mean_before)} → ${fmtNum(c.mean_after)}`;
  return `SD ${fmtNum(c.sd_before)} → ${fmtNum(c.sd_after)} · r ${fmtR(c.r_before)} → ${fmtR(c.r_after)}`;
}

export interface StripProps {
  cols: StripColumn[];
  /** With this choice; false also for a storyboard frame that still draws the data as recorded. */
  after: boolean;
  linked: Linked;
  /** The engine's views for the focused column (its scatter against total calories). */
  rest: ReactNode;
  /** The heading when the choice is not shown (at rest: the columns as recorded). */
  title?: string;
  /** The focused column: every view on the canvas follows it (FOUNDATION §5 rule 4). Without it
   *  the strip keeps its own focus, starting on the first column. */
  focus?: string | null;
  onFocus?: (column: string | null) => void;
  /** The focused column's own values, drawn large (they follow the storyboard); without it, its
   *  histogram from the strip's numbers. */
  focusView?: ReactNode;
  /** @deprecated a column to focus; pass `focus`. */
  focusColumn?: string | null;
}

export function Strip({ cols, after, linked, rest, title, focus, onFocus, focusView, focusColumn }: StripProps) {
  const [own, setOwn] = useState<string | null>(null);
  const chosen = focusColumn ?? (focus !== undefined ? focus : own);
  const setChosen = onFocus ?? setOwn;
  const at = chosen ? cols.findIndex((c) => c.column === chosen) : -1;
  const f = cols[at >= 0 ? at : 0]!;
  const max = Math.max(...cols.map((c) => c.shift), 1e-9);
  const shown = cols.slice(0, STRIP_TOP);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const i = Math.max(0, Math.min(shown.length - 1, shown.indexOf(f) + (e.key === "ArrowDown" ? 1 : -1)));
      setChosen(shown[i]!.column);
      (e.currentTarget.querySelectorAll("button")[i] as HTMLButtonElement | undefined)?.focus();
    }
  };
  return (
    <div className={k.panels}>
      <div className={k.panel}>
        <h3>{after || !title ? "Every column this choice changes, most first" : title}</h3>
        <ul className={k.strip} onKeyDown={onKey} aria-label={after ? "Changed columns" : "Columns"} data-now={!after || undefined}>
          {shown.map((c) => (
            <li key={c.column}>
              <button
                type="button"
                className={k.stripRow}
                title={c.desc}
                data-focus={c === f}
                data-lit={linked.lit === c.column || linked.lit === c.output || undefined}
                aria-pressed={c === f}
                // Pointing moves the focus, and every view follows it. Only a pointer that really
                // moves does: when the views above change height, the browser re-hovers whatever now
                // sits under a still pointer, which must not move the focus again.
                onPointerMove={(e) => {
                  if (e.pointerType === "mouse" && (e.movementX || e.movementY) && c !== f) setChosen(c.column);
                }}
                onPointerEnter={() => linked.setLit(c.column)}
                onPointerLeave={() => linked.setLit(null)}
                onFocus={() => setChosen(c.column)}
                onClick={() => setChosen(c.column)}
              >
                <span className={k.stripName}>{after ? c.output : c.column}</span>
                <span className={k.stripBar} aria-hidden="true">
                  <i style={{ width: `${after ? (100 * c.shift) / max : 0}%` }} />
                </span>
                <span className={k.stripMeasure}>{measure(c, after)}</span>
              </button>
            </li>
          ))}
        </ul>
        {cols.length > STRIP_TOP ? <p className={k.stripMore}>and {cols.length - STRIP_TOP} more</p> : null}
      </div>
      {focusView ?? (
        <Hist
          title={`${after ? f.output : f.column}, ${after ? "with this choice" : "as recorded"}`}
          before={f.hist_before}
          after={f.hist_after}
          mode="transform"
          after_on={after}
          unit={f.column}
        />
      )}
      {rest}
    </div>
  );
}

// ── Angles ───────────────────────────────────────────────────────────────────

/** What can follow a choice, as marks: filled runs, hollow cannot (its reason on hover). */
function Marks({ list, after }: { list: NonNullable<Angle["list"]>; after: boolean }) {
  return (
    <ul className={k.marks} data-after={after || undefined}>
      {list.map((x) => (
        <li key={x.name} data-ok={x.ok} title={x.why ? plain(x.why) : undefined}>
          <i aria-hidden="true" />
          <span>{plain(x.name)}</span>
          <span className={k.srOnly}>{x.ok ? " (can follow)" : " (cannot follow)"}</span>
        </li>
      ))}
    </ul>
  );
}

export interface AnglesProps {
  step: Step;
  /** The panels: the pointed option's, or at rest the question's own (data now only). */
  angles: Angle[];
  views: ConsequenceView[];
  /** The option the table marks (the one pointed at or chosen). */
  activeId: string | null;
  after: boolean;
  frame: number | null;
  linked: Linked;
}

/** Each panel one question and one picture; one small table, an option per row and at most two
 *  short answers beside its name (FOUNDATION §5). */
export function Angles({ step, angles, views, activeId, after, frame, linked }: AnglesProps) {
  const rows = step.options.filter((o) => !o.disabled && o.preview.angles?.length);
  const heads = rows[0]?.preview.angles?.slice(0, 2).map((a) => a.head) ?? [];
  return (
    <div className={k.panels}>
      <div className={k.angles}>
        {angles.map((a) => {
          const v = a.view !== undefined ? views[a.view] : undefined;
          return (
            <section key={a.question} className={k.angle} data-wide={v?.kind === "lineage" || undefined}>
              <h3>{a.question}</h3>
              {v ? (
                v.kind === "lineage" ? (
                  <Lineage view={v} after_on={after} frame={frame} touch={a.touch} linked={linked} />
                ) : (
                  <View view={v} after={after} frame={frame} linked={linked} touch={a.touch} bare />
                )
              ) : a.list ? (
                <Marks list={a.list} after={after} />
              ) : null}
            </section>
          );
        })}
      </div>
      <div className={k.tableWrap}>
        <table className={k.otable} aria-label="Each option, side by side">
          <thead>
            <tr>
              <th scope="col">Option</th>
              {heads.map((h) => (
                <th key={h} scope="col">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((o) => (
              <tr key={o.id} data-active={o.id === activeId}>
                <th scope="row">{plain(o.name)}</th>
                {(o.preview.angles ?? []).slice(0, 2).map((a) => (
                  <td key={a.head}>{plain(a.cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export { route };
