/**
 * The canvas: the dynamic window where the user's data speaks (FOUNDATION §3, §5). One flip ("Your
 * data now" ⇄ "With this choice") and one storyboard drive every view; the router picks the layout
 * from the option's footprint; at most three views show, the rest behind "More angles". It is never
 * empty: at rest, and under an option that changes nothing or is not available, it draws the
 * question's columns as they are now, in gray, in the layout its options use (rule 8). After the
 * lock it holds Table 2 and "Which of my decisions mattered?".
 */
import { useState, type ReactNode } from "react";
import type { ConsequenceView, Now, Option, Step } from "../fixture";
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

const matrixSize = (l: { nodes: { lane: string; count?: number | null }[] }) =>
  l.nodes.filter((n) => n.lane === "matrix").reduce((c, n) => c + (n.count || 1), 0);

/** The headline numbers pinned beside the views (BLUEPRINT §11.1): people, r, the model's inputs.
 *  With a storyboard frame shown, the "after" side is that frame's (the readout follows the picture). */
export function readoutOfViews(views: ConsequenceView[], restOnly = false, frame: number | null = null): Readout[] {
  const out: Readout[] = [];
  const flow = views.find((v) => v.kind === "row_flow");
  if (flow && flow.kind === "row_flow") {
    const a = flow.before[flow.before.length - 1]?.n ?? 0;
    const b = flow.after[flow.after.length - 1]?.n ?? a;
    out.push({ label: "People", now: fmtInt(a), after: !restOnly && b !== a ? fmtInt(b) : null });
  }
  const rel = views.find((v) => v.kind === "relationship");
  if (rel && rel.kind === "relationship" && rel.r_before !== null && rel.r_after !== null) {
    const shown = frame !== null ? (rel.story?.[frame]?.r ?? rel.r_after) : rel.r_after;
    const changed = fmtR(shown) !== fmtR(rel.r_before);
    out.push({
      label: `Correlation of ${plain(rel.y_label_before)} with ${plain(rel.x_label)}`,
      now: fmtR(rel.r_before),
      after: !restOnly && changed ? fmtR(shown) : null,
    });
  }
  const lin = views.find((v) => v.kind === "lineage");
  if (lin && lin.kind === "lineage" && out.length < 2) {
    const a = matrixSize(lin.before ?? lin.after);
    const at = frame !== null ? lin.story?.[frame]?.lineage : undefined;
    const b = matrixSize(at ?? lin.after);
    if (restOnly) out.push({ label: "Model inputs", now: fmtInt(a), after: null });
    else if (lin.before && a !== b) out.push({ label: "Model inputs", now: fmtInt(a), after: fmtInt(b) });
  }
  return out.slice(0, 2);
}

export function readoutOf(o: Option | null, frame: number | null = null): Readout[] {
  return o ? readoutOfViews(o.preview.views as ConsequenceView[], false, frame) : [];
}

/** Your data now for a question: its own, or the one an earlier answer leaves (the lock's). */
export function restOf(step: Step | null, answers?: Record<string, string>): Now | null {
  if (!step) return null;
  const by = step.now_by;
  const got = by && answers ? answers[by.step] : undefined;
  return (got && by?.options[got]) || step.now;
}

/** The canvas's one line: what it shows, in the card's register. */
function captionOf(o: Option | null, layout: Layout, after: boolean, primary: ConsequenceView | undefined, rest: Now | null): ReactNode {
  if (!o) return rest ? plain(rest.caption) : "";
  if (layout === "refused") {
    // one calm line: what it would do and why not when the fixture says so plainly; else the
    // refusal when it is short; else the option's own line, which says why
    if (o.preview.caption) return plain(o.preview.caption);
    const full = plain(o.refusal ?? o.preview.refusal?.message);
    return full.split(/\s+/).length <= 30 ? full : plain(o.what);
  }
  if (layout === "none") return plain(o.preview.caption ?? o.preview.note ?? o.preview.views[0]?.caption ?? null);
  if (!after) return rest ? plain(rest.caption) : "Your data now, before this choice.";
  return plain(o.preview.caption ?? primary?.caption ?? o.preview.note);
}

function Key({ now, cut }: { now: boolean; cut: boolean }) {
  return (
    <div className={k.key}>
      <span>
        <i style={{ background: "var(--data-context)" }} />
        {cut && !now ? "Stays" : "Your data now"}
      </span>
      {now ? null : (
        <span>
          <i style={{ background: "var(--data-affected)" }} />
          {cut ? "Leaves with this choice" : "What this choice touches"}
        </span>
      )}
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
  /** The recorded answers, for a rest picture that follows one (the lock's). */
  answers?: Record<string, string>;
  title?: string;
  testid?: string;
}

/** The question's columns as they are now, in gray, in the layout its options use. */
function RestPicture({ step, rest, option, linked }: { step: Step; rest: Now; option: Option | null; linked: Linked }) {
  const p = { after: false, frame: null, linked };
  if (rest.layout === "angles")
    return <Angles step={step} angles={rest.angles ?? []} views={rest.views} activeId={option?.id ?? null} {...p} />;
  if (rest.layout === "strip")
    return (
      <Strip
        cols={rest.strip ?? []}
        title={rest.title}
        focusColumn={option?.id ?? null}
        rest={rest.views.length ? <Views views={rest.views} {...p} /> : null}
        {...p}
      />
    );
  return <Views views={rest.views} {...p} />;
}

/** The canvas for one option, or for none (no walk needed: the kit demo draws every layout with it). */
export function CanvasFrame({ step, option, flip, setFlip, frame, setFrame, answers, title = "Your data", testid = "canvas" }: CanvasFrameProps) {
  const [lit, setLit] = useState<string | null>(null);
  const linked: Linked = { lit, setLit };
  const rest = restOf(step, answers);
  const layout: Layout = option ? route(option.preview, { disabled: option.disabled }) : "none";
  // Nothing pointed at, a choice that changes nothing, or one not available: your data now.
  const quiet = !option || layout === "none" || layout === "refused";
  const after = flip === "after" && !quiet;
  const { shown, more } = option && !quiet ? viewsFor(layout, option.preview) : { shown: [], more: [] };
  const primary = shown[0] ?? (option?.preview.views[0] as ConsequenceView | undefined);
  const n = storyLength(option);
  const story = primary && "story" in primary ? (primary.story ?? []) : [];
  const coach = after ? shown.flatMap((v) => v.coach)[0]?.text : undefined;
  const readout = quiet
    ? rest
      ? readoutOfViews(rest.views, true)
      : []
    : layout === "angles"
      ? []
      : after
        ? readoutOf(option, frame)
        : readoutOf(option).map((r) => ({ ...r, after: null }));
  const p = { after, frame, linked };
  const cut = !quiet && (primary?.kind === "row_flow" || (primary?.kind === "distribution" && primary.before_label === "Every measured value"));
  const basis = quiet ? (rest?.basis ?? option?.preview.basis) : option?.preview.basis;
  return (
    <section
      className={k.canvas}
      aria-live="polite"
      data-testid={testid}
      data-layout={option ? layout : "rest"}
      data-rest={quiet && rest ? rest.layout : undefined}
      data-flip={after ? "after" : "now"}
    >
      <div className={k.chead}>
        <h2>{title}</h2>
        <div className={k.flip} role="group" aria-label="Show your data now or with this choice">
          <button type="button" data-s="now" aria-pressed={!after} onClick={() => setFlip("now")} data-testid="flip-now">
            Your data now
          </button>
          <button type="button" data-s="after" aria-pressed={after} disabled={quiet} onClick={() => setFlip("after")} data-testid="flip-after">
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
              aria-label={`Step ${i + 1}: ${plain(story[i]?.label)}`}
              aria-current={frame === i ? "step" : undefined}
              onClick={() => setFrame(i)}
            />
          ))}
          <button type="button" aria-label="The result" aria-current={frame === null ? "step" : undefined} onClick={() => setFrame(null)} />
          <span>{frame !== null ? `Step ${frame + 1} of ${n}: ${plain(story[frame]?.label)}` : "The result"}</span>
        </div>
      ) : null}
      <p className={k.caption} data-testid="caption">
        {captionOf(option, layout, after, primary, rest)}
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
      {quiet && step && rest ? <RestPicture step={step} rest={rest} option={option} linked={linked} /> : null}
      {option && !quiet && step && layout === "angles" ? (
        <Angles step={step} angles={option.preview.angles ?? []} views={option.preview.views as ConsequenceView[]} activeId={option.id} {...p} />
      ) : null}
      {option && !quiet && layout === "strip" ? <Strip cols={option.preview.strip ?? []} {...p} rest={<Views views={shown} {...p} />} /> : null}
      {option && !quiet && (layout === "flow" || layout === "focus" || layout === "routing") ? <Views views={shown} {...p} /> : null}
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
      <Key now={quiet || !after} cut={cut} />
      {basis ? <p className={k.basis}>{plain(basis)}</p> : null}
    </section>
  );
}

/** The canvas bound to the walk: the open question's active option, its data now, or the results. */
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
      answers={walk.state.answers}
      title={title}
    />
  );
}
