/**
 * The canvas: the dynamic window where the user's data speaks (FOUNDATION §3, §5). One flip ("Your
 * data now" ⇄ "With this choice") and one storyboard drive every view; the router picks the layout
 * from the option's footprint; at most three views show, the rest behind "More angles". After the
 * lock it holds Table 2 and "Which of my decisions mattered?".
 */
import { useState, type ReactNode } from "react";
import type { ConsequenceView, Option, Step } from "../fixture";
import { route, viewsFor, type Layout } from "../router";
import { fmtInt, fmtR, plain } from "../text";
import type { Flip, WalkApi } from "../useWalk";
import { storyLength } from "../useWalk";
import k from "../kit.module.css";
import { Angles, Strip, View, Views } from "./layouts";
import { Mattered, Table2 } from "./Results";
import type { Linked } from "./views";

export interface Readout {
  label: string;
  now: string;
  after: string | null;
}

/** The headline numbers pinned beside the views (BLUEPRINT §11.1): rows, r, the model's inputs. */
export function readoutOf(o: Option | null): Readout[] {
  if (!o) return [];
  const out: Readout[] = [];
  const views = o.preview.views as ConsequenceView[];
  const flow = views.find((v) => v.kind === "row_flow");
  if (flow && flow.kind === "row_flow") {
    const a = flow.before[flow.before.length - 1]?.n ?? 0;
    const b = flow.after[flow.after.length - 1]?.n ?? a;
    out.push({ label: "Rows", now: fmtInt(a), after: b !== a ? fmtInt(b) : null });
  }
  const rel = views.find((v) => v.kind === "relationship");
  if (rel && rel.kind === "relationship" && rel.r_before !== null && rel.r_after !== null) {
    out.push({ label: `r with ${plain(rel.x_label)}`, now: fmtR(rel.r_before), after: fmtR(rel.r_after) !== fmtR(rel.r_before) ? fmtR(rel.r_after) : null });
  }
  const lin = views.find((v) => v.kind === "lineage");
  if (lin && lin.kind === "lineage" && lin.before && out.length < 2) {
    const size = (l: typeof lin.after) => l.nodes.filter((n) => n.lane === "matrix").reduce((c, n) => c + (n.count || 1), 0);
    const a = size(lin.before);
    const b = size(lin.after);
    if (a !== b) out.push({ label: "Model inputs", now: fmtInt(a), after: fmtInt(b) });
  }
  return out.slice(0, 2);
}

function captionOf(o: Option | null, layout: Layout, after: boolean, primary: ConsequenceView | undefined): ReactNode {
  if (!o) return "Point at an option to see what it does to your data.";
  if (layout === "refused") {
    // one calm line: the engine's refusal when it is short, else the option's own line, which says it
    const full = plain(o.refusal ?? o.preview.refusal?.message);
    return full.split(/\s+/).length <= 30 ? full : plain(o.what);
  }
  if (!after) {
    const flow = o.preview.views.find((v) => v.kind === "row_flow");
    if (flow && flow.kind === "row_flow") return <>Your data now: <b>{fmtInt(flow.before[flow.before.length - 1]?.n ?? 0)}</b> rows.</>;
    return "Your data now, before this choice.";
  }
  if (layout === "none") return plain(o.preview.note ?? primary?.caption ?? null);
  return plain(primary?.caption ?? o.preview.note);
}

function keyOf(layout: Layout, primary: ConsequenceView | undefined): ReactNode {
  if (layout === "none" || layout === "refused") return null;
  const cut = primary?.kind === "row_flow" || (primary?.kind === "distribution" && primary.before_label === "Every measured value");
  return (
    <div className={k.key}>
      <span>
        <i style={{ background: "var(--data-context)" }} />
        {cut ? "Stays" : "Your data now"}
      </span>
      <span>
        <i style={{ background: "var(--data-affected)" }} />
        {cut ? "Leaves with this choice" : "What this choice touches"}
      </span>
    </div>
  );
}

export interface CanvasFrameProps {
  step: Step | null;
  option: Option | null;
  flip: Flip;
  setFlip: (f: Flip) => void;
  frame: number | null;
  setFrame: (i: number | null) => void;
  title?: string;
  testid?: string;
}

/** The canvas for one option (no walk needed: the kit demo draws every layout with it). */
export function CanvasFrame({ step, option, flip, setFlip, frame, setFrame, title = "Your data", testid = "canvas" }: CanvasFrameProps) {
  const [lit, setLit] = useState<string | null>(null);
  const linked: Linked = { lit, setLit };
  const layout: Layout = option ? route(option.preview, { disabled: option.disabled }) : "none";
  const after = flip === "after" && !!option && layout !== "refused";
  const { shown, more } = option ? viewsFor(layout, option.preview) : { shown: [], more: [] };
  const primary = shown[0] ?? (option?.preview.views[0] as ConsequenceView | undefined);
  const n = storyLength(option);
  const story = primary && "story" in primary ? (primary.story ?? []) : [];
  const coach = after ? (shown.flatMap((v) => v.coach)[0]?.text ?? (layout === "none" ? primary?.coach[0]?.text : undefined)) : undefined;
  const readout = layout === "angles" ? [] : after || !option ? readoutOf(option) : readoutOf(option).map((r) => ({ ...r, after: null }));
  const p = { option: option!, after, frame, linked };
  return (
    <section className={k.canvas} aria-live="polite" data-testid={testid} data-layout={option ? layout : "empty"} data-flip={after ? "after" : "now"}>
      <div className={k.chead}>
        <h2>{title}</h2>
        <div className={k.flip} role="group" aria-label="Show your data now or with this choice">
          <button type="button" data-s="now" aria-pressed={!after} onClick={() => setFlip("now")} data-testid="flip-now">
            Your data now
          </button>
          <button type="button" data-s="after" aria-pressed={after} disabled={!option || layout === "refused"} onClick={() => setFlip("after")} data-testid="flip-after">
            With this choice
          </button>
        </div>
      </div>
      {after && n > 0 ? (
        <div className={k.steps}>
          {Array.from({ length: n }, (_, i) => (
            <button
              key={i}
              type="button"
              aria-label={`Step ${i + 1}: ${story[i]?.label ?? ""}`}
              aria-current={frame === i ? "step" : undefined}
              onClick={() => setFrame(i)}
            />
          ))}
          <button type="button" aria-label="The result" aria-current={frame === null ? "step" : undefined} onClick={() => setFrame(null)} />
          <span>{frame !== null ? `Step ${frame + 1} of ${n}: ${plain(story[frame]?.label)}` : "The result"}</span>
        </div>
      ) : null}
      <p className={k.caption} data-testid="caption">
        {captionOf(option, layout, after, primary)}
      </p>
      {readout.length ? (
        <div className={k.readout} data-testid="readout">
          {readout.map((r) => (
            <span key={r.label}>
              {r.label}{" "}
              <b>
                {r.now}
                {r.after ? ` → ${r.after}` : ""}
              </b>
            </span>
          ))}
        </div>
      ) : null}
      {coach ? <div className={k.coach}>{plain(coach)}</div> : null}
      {option && step && layout === "angles" ? <Angles {...p} step={step} /> : null}
      {option && layout === "strip" ? <Strip {...p} rest={<Views views={shown} after={after} frame={frame} linked={linked} />} /> : null}
      {option && (layout === "flow" || layout === "focus" || layout === "routing") ? <Views views={shown} after={after} frame={frame} linked={linked} /> : null}
      {more.length ? (
        <details className={k.more}>
          <summary>More angles ({more.length})</summary>
          <div className={k.panels}>
            {more.map((v, i) => (
              <View key={i} view={v} after={after} frame={frame} linked={linked} />
            ))}
          </div>
        </details>
      ) : null}
      {keyOf(layout, primary)}
      {option?.preview.basis ? <p className={k.basis}>{plain(option.preview.basis)}</p> : null}
    </section>
  );
}

/** The canvas bound to the walk: the open question's active option, or the results. */
export function Canvas({ walk, title }: { walk: WalkApi; title?: string }) {
  const { state, results } = walk;
  if (results && (state.open === "table2" || state.open === "mattered")) {
    return (
      <section className={k.canvas} data-testid="canvas" data-layout="results">
        <div className={k.chead}>
          <h2>{state.open === "table2" ? "Table 2" : "Which of my decisions mattered?"}</h2>
        </div>
        {state.open === "table2" ? <Table2 results={results} /> : <Mattered results={results} />}
      </section>
    );
  }
  return (
    <CanvasFrame
      step={walk.step}
      option={walk.active}
      flip={walk.flip}
      setFlip={walk.setFlip}
      frame={walk.frame}
      setFrame={walk.setFrame}
      title={title}
    />
  );
}
