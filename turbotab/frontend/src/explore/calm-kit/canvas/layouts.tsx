/**
 * The five layouts of the canvas grammar (FOUNDATION §5). Each draws the engine's views for one
 * option; one flip and one storyboard drive every view at once; views are linked (pointing at a
 * column or a flow step lights it in every view).
 */
import { useState, type ReactNode } from "react";
import type { ConsequenceView, Option, Step, StripColumn } from "../fixture";
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

/** One engine view, drawn by its kind. */
export function View({ view, after, frame, linked, touch }: { view: ConsequenceView; after: boolean; frame: number | null; linked: Linked; touch?: string[] }) {
  switch (view.kind) {
    case "distribution": {
      const f = after && frame !== null ? view.story?.[frame] : null;
      const cut = view.before_label === "Every measured value" || view.after_label.startsWith("Kept");
      return (
        <Hist
          title={plain(view.title)}
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
          <h3>{plain(view.title)}</h3>
          <Scatter view={view} after_on={after} frame={frame} />
        </div>
      );
    case "row_flow":
      return <FlowBars view={view} after_on={after} linked={linked} />;
    case "lineage":
      return (
        <div className={k.panel}>
          <h3>{plain(view.title)}</h3>
          <Lineage view={view} after_on={after} frame={frame} touch={touch} linked={linked} />
        </div>
      );
    case "table_focus":
      return <Cells view={view} after_on={after} />;
  }
}

export function Views({ views, ...p }: { views: ConsequenceView[] } & Omit<LayoutProps, "option">) {
  return (
    <div className={k.panels}>
      {views.map((v, i) => (
        <View key={`${v.kind}-${i}`} view={v} after={p.after} frame={p.frame} linked={p.linked} />
      ))}
    </div>
  );
}

// ── Strip ────────────────────────────────────────────────────────────────────

const STRIP_TOP = 12;

function measure(c: StripColumn): string {
  // the per-calorie columns change scale: say so by the mean; otherwise by the spread
  const rescaled = Math.abs(c.mean_after) < Math.abs(c.mean_before) / 50;
  if (rescaled) return `mean ${fmtNum(c.mean_before)} → ${fmtNum(c.mean_after)}`;
  return `SD ${fmtNum(c.sd_before)} → ${fmtNum(c.sd_after)} · r ${fmtR(c.r_before)} → ${fmtR(c.r_after)}`;
}

export function Strip({ option, after, linked, rest }: LayoutProps & { rest: ReactNode }) {
  const cols = option.preview.strip ?? [];
  const [focus, setFocus] = useState(0);
  const lit = linked.lit ? cols.findIndex((c) => c.column === linked.lit || c.output === linked.lit) : -1;
  const f = cols[lit >= 0 ? lit : Math.min(focus, cols.length - 1)]!;
  const max = Math.max(...cols.map((c) => c.shift), 1e-9);
  const shown = cols.slice(0, STRIP_TOP);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setFocus((i) => Math.max(0, Math.min(shown.length - 1, i + (e.key === "ArrowDown" ? 1 : -1))));
    }
  };
  return (
    <div className={k.panels}>
      <div className={k.panel}>
        <h3>Every column this choice changes, most first</h3>
        <ul className={k.strip} onKeyDown={onKey} aria-label="Changed columns">
          {shown.map((c, i) => (
            <li key={c.column}>
              <button
                type="button"
                className={k.stripRow}
                data-focus={c === f}
                onPointerEnter={() => {
                  setFocus(i);
                  linked.setLit(c.column);
                }}
                onPointerLeave={() => linked.setLit(null)}
                onFocus={() => setFocus(i)}
                onClick={() => setFocus(i)}
              >
                <span className={k.stripName}>{after ? c.output : c.column}</span>
                <span className={k.stripBar} aria-hidden="true">
                  <i style={{ width: `${after ? (100 * c.shift) / max : 0}%` }} />
                </span>
                <span className={k.stripMeasure}>{measure(c)}</span>
              </button>
            </li>
          ))}
        </ul>
        {cols.length > STRIP_TOP ? <p className={k.stripMore}>and {cols.length - STRIP_TOP} more</p> : null}
      </div>
      <Hist
        title={`${after ? f.output : f.column}, ${after ? "with this choice" : "as recorded"}`}
        before={f.hist_before}
        after={f.hist_after}
        mode="transform"
        after_on={after}
        unit={f.column}
      />
      {rest}
    </div>
  );
}

// ── Angles ───────────────────────────────────────────────────────────────────

export function Angles({ option, step, after, frame, linked }: LayoutProps & { step: Step }) {
  const angles = option.preview.angles ?? [];
  const rows = step.options.filter((o) => !o.disabled && o.preview.angles?.length);
  return (
    <div className={k.panels}>
      <div className={k.angles}>
        {angles.map((a) => (
          <section key={a.question} className={k.angle} data-wide={(a.view !== undefined && option.preview.views[a.view]?.kind === "lineage") || undefined}>
            <h3>{a.question}</h3>
            {a.view !== undefined && option.preview.views[a.view] ? (
              option.preview.views[a.view]!.kind === "lineage" ? (
                <Lineage view={option.preview.views[a.view] as never} after_on={after} frame={frame} touch={a.touch} linked={linked} />
              ) : (
                <View view={option.preview.views[a.view]!} after={after} frame={frame} linked={linked} touch={a.touch} />
              )
            ) : null}
            {a.text ? <p className={k.angleText}>{plain(a.text)}</p> : null}
            {a.list ? (
              <ul className={k.angleList}>
                {a.list.map((x) => (
                  <li key={x.name} data-ok={x.ok}>
                    {x.ok ? "Runs: " : "Refused: "}
                    {plain(x.name)}
                    {x.why ? <small>{plain(x.why)}</small> : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </section>
        ))}
      </div>
      <div className={k.tableWrap}>
        <table className={k.otable} aria-label="Each option against each question">
          <thead>
            <tr>
              <th scope="col">Option</th>
              {angles.map((a) => (
                <th key={a.question} scope="col">
                  {a.question}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((o) => (
              <tr key={o.id} data-active={o.id === option.id}>
                <th scope="row">{plain(o.name)}</th>
                {(o.preview.angles ?? []).map((a) => (
                  <td key={a.question}>{plain(a.cell)}</td>
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
