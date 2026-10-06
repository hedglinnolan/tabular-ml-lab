/**
 * The canvas: the dynamic window where the user's data speaks (FOUNDATION §3, §5). One flip ("Your
 * data now" ⇄ "With this choice") and one storyboard drive every view; the router picks the layout
 * from the option's footprint; at most three views show, the rest behind "More angles". It is never
 * empty: at rest, and under an option that changes nothing or is not available, it draws the
 * question's columns as they are now, in gray, in the layout its options use (rule 8). After the
 * lock it holds Table 2 and "Which of my decisions mattered?".
 */
import { useState, type ReactNode } from "react";
import type { ConsequenceView, LineageView, Now, Option, Preview, Step, StripColumn } from "../fixture";
import { route, viewsFor, type Layout } from "../router";
import { fmtInt, fmtR, plain } from "../text";
import type { Flip, WalkApi } from "../useWalk";
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

/** What a lineage's model-input lane counts, when it is not the one model ("Model 1's inputs"):
 *  the readout and the lane say the same. */
export const inputsLabel = (v: LineageView): string | undefined => (v as LineageView & { inputs_label?: string }).inputs_label;

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
    const label = inputsLabel(lin) ?? "Model inputs";
    const a = matrixSize(lin.before ?? lin.after);
    const at = frame !== null ? lin.story?.[frame]?.lineage : undefined;
    const b = matrixSize(at ?? lin.after);
    // the count as it is, and where the choice changes it, the count after (as the flow's people)
    if (restOnly) out.push({ label, now: fmtInt(a), after: null });
    else if (lin.before) out.push({ label, now: fmtInt(a), after: a !== b ? fmtInt(b) : null });
  }
  return out.slice(0, 2);
}

/** The checks reported beside the main analysis (`Preview.beside`): the main analysis keeps its
 *  people whatever is chosen; each check runs on the people its rule keeps. */
function besideReadout(views: ConsequenceView[], after: boolean): Readout[] {
  const flows = views.filter((v) => v.kind === "row_flow");
  const first = flows[0];
  if (!first) return [];
  const main: Readout = { label: "People in the main analysis", now: fmtInt(first.before[first.before.length - 1]?.n ?? 0), after: null };
  if (!after) return [main];
  const checks = flows.map((f) => fmtInt(f.after[f.after.length - 1]?.n ?? 0));
  return [main, { label: flows.length > 1 ? "In the checks" : "In the check", now: checks.join(" and "), after: null }];
}

// ── the Strip's focus: every view follows the focused column (FOUNDATION §5 rule 4) ─────────────

/** The focused column: the one asked for, else the one the engine's own views picture, else the
 *  first (the largest change). */
export function stripFocus(cols: StripColumn[] | undefined, views: ConsequenceView[], focus: string | null): StripColumn | null {
  if (!cols?.length) return null;
  const asked = focus ? cols.find((c) => c.column === focus) : undefined;
  if (asked) return asked;
  const rel = views.find((v) => v.kind === "relationship");
  const pictured = rel ? cols.find((c) => c.column === plain(rel.y_label_before)) : undefined;
  return pictured ?? cols[0]!;
}

/** The engine's views with the focused column's own in place of the column it pictures. */
function swapViews(views: ConsequenceView[], own: ConsequenceView[]): ConsequenceView[] {
  const rel = own.find((v) => v.kind === "relationship");
  const dist = own.find((v) => v.kind === "distribution");
  const out = views.flatMap((v): ConsequenceView[] => (v.kind === "relationship" ? (rel ? [rel] : []) : v.kind === "distribution" ? (dist ? [dist] : []) : [v]));
  if (dist && !views.some((v) => v.kind === "distribution")) out.push(dist);
  return out;
}

/** A preview as the canvas draws it: under the Strip, the focused column's views and caption. */
export function focusPreview(p: Preview, focus: string | null): Preview {
  const col = stripFocus(p.strip, p.views as ConsequenceView[], focus);
  if (!col?.views?.length) return p;
  return { ...p, caption: col.caption ?? p.caption, views: swapViews(p.views as ConsequenceView[], col.views) };
}

/** Your data now under the Strip: the focused column's own picture and caption. */
function focusNow(rest: Now, focus: string | null): Now {
  if (rest.layout !== "strip") return rest;
  const col = stripFocus(rest.strip, rest.views, focus);
  if (!col?.views?.length) return rest;
  return { ...rest, caption: col.caption ?? rest.caption, views: swapViews(rest.views, col.views) };
}

/** The readout for an option, as the canvas shows it (and the footer on narrow screens): none for
 *  Angles (each panel answers its own question) or for a choice that changes nothing. */
export function readoutFor(o: Option | null, { after, frame = null, focus = null }: { after: boolean; frame?: number | null; focus?: string | null }): Readout[] {
  if (!o) return [];
  const layout = route(o.preview, { disabled: o.disabled });
  if (layout === "angles" || layout === "none" || layout === "refused") return [];
  const p = focusPreview(o.preview, focus);
  if (p.beside) return besideReadout(p.views as ConsequenceView[], after);
  const r = readoutOfViews(p.views as ConsequenceView[], false, after ? frame : null);
  return after ? r : r.map((x) => ({ ...x, after: null }));
}

/** The readout with this choice (kept for callers that name a frame only). */
export function readoutOf(o: Option | null, frame: number | null = null, focus: string | null = null): Readout[] {
  return readoutFor(o, { after: true, frame, focus });
}

/** Your data now for a question: its own, or the one an earlier answer leaves (the lock's). */
export function restOf(step: Step | null, answers?: Record<string, string>): Now | null {
  if (!step) return null;
  const by = step.now_by;
  const got = by && answers ? answers[by.step] : undefined;
  return (got && by?.options[got]) || step.now;
}

/** The canvas's one line: what it shows, in the card's register. */
function captionOf(o: Option | null, pv: Preview | null, layout: Layout, after: boolean, primary: ConsequenceView | undefined, rest: Now | null): ReactNode {
  if (!o || !pv) return rest ? plain(rest.caption) : "";
  if (layout === "refused") {
    // one calm line: what it would do and why not when the fixture says so plainly; else the
    // refusal when it is short; else the option's own line, which says why
    if (pv.caption) return plain(pv.caption);
    const full = plain(o.refusal ?? pv.refusal?.message);
    return full.split(/\s+/).length <= 30 ? full : plain(o.what);
  }
  if (layout === "none") return plain(pv.caption ?? pv.note ?? pv.views[0]?.caption ?? null);
  if (!after) return rest ? plain(rest.caption) : "Your data now, before this choice.";
  return plain(pv.caption ?? primary?.caption ?? pv.note);
}

/** The key: gray is now; indigo is what the choice touches, said for what each picture draws. */
function Key({ now, cut, lines, beside }: { now: boolean; cut: boolean; lines: boolean; beside: boolean }) {
  if (now)
    return (
      <div className={k.key}>
        <span>
          <i style={{ background: "var(--data-context)" }} />
          Your data now
        </span>
      </div>
    );
  return (
    <div className={k.key}>
      <span>
        <i style={{ background: "var(--data-context)" }} />
        {cut ? (beside ? "In the check" : "Stays") : "Your data now"}
      </span>
      <span>
        <i style={{ background: "var(--data-affected)" }} />
        {cut ? (beside ? "Left out of the check only" : "Leaves with this choice") : "What this choice touches"}
      </span>
      {cut && lines ? (
        <span>
          <i className={k.keyLine} style={{ background: "var(--data-affected)" }} />
          Columns this choice changes
        </span>
      ) : null}
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
  /** The Strip's focused column (the walk's); without it the canvas keeps its own. */
  focus?: string | null;
  setFocus?: (column: string | null) => void;
  title?: string;
  testid?: string;
}

/** A storyboard frame that still draws the data as recorded (the residual's fit step): the views
 *  without a storyboard of their own (the Strip's bars and measures) stay on "now" for it. */
function frameStillBefore(v: ConsequenceView | undefined, frame: number | null): boolean {
  if (!v || frame === null || v.kind !== "relationship") return false;
  const f = v.story?.[frame];
  if (!f) return false;
  const a = f.points;
  const b = v.points_before;
  return a === b || (a.length === b.length && a.every((p, i) => p[0] === b[i]![0] && p[1] === b[i]![1]));
}

/** The question's columns as they are now, in gray, in the layout its options use. */
function RestPicture({
  step,
  rest,
  option,
  linked,
  focus,
  setFocus,
}: {
  step: Step;
  rest: Now;
  option: Option | null;
  linked: Linked;
  focus: string | null;
  setFocus: (c: string | null) => void;
}) {
  const p = { after: false, frame: null, linked };
  if (rest.layout === "angles")
    return <Angles step={step} angles={rest.angles ?? []} views={rest.views} activeId={option?.id ?? null} {...p} />;
  if (rest.layout === "strip") {
    // the option pointed at, when the options are the columns (the exposure); else the focus
    const asked = option && rest.strip?.some((c) => c.column === option.id) ? option.id : focus;
    const col = stripFocus(rest.strip, rest.views, asked);
    return (
      <Strip
        cols={rest.strip ?? []}
        title={rest.title}
        focus={col?.column ?? null}
        onFocus={setFocus}
        rest={rest.views.length ? <Views views={rest.views} keep={rest.columns} {...p} /> : null}
        {...p}
      />
    );
  }
  return <Views views={rest.views} keep={rest.columns} {...p} />;
}

/** The canvas for one option, or for none (no walk needed: the kit demo draws every layout with it). */
export function CanvasFrame({ step, option, flip, setFlip, frame, setFrame, answers, focus: walkFocus, setFocus: setWalkFocus, title = "Your data", testid = "canvas" }: CanvasFrameProps) {
  const [lit, setLit] = useState<string | null>(null);
  const [ownFocus, setOwnFocus] = useState<string | null>(null);
  const focus = walkFocus !== undefined ? walkFocus : ownFocus;
  const setFocus = setWalkFocus ?? setOwnFocus;
  const linked: Linked = { lit, setLit };
  const stored = restOf(step, answers);
  const rest = stored ? focusNow(stored, option && stored.strip?.some((c) => c.column === option.id) ? option.id : focus) : null;
  const layout: Layout = option ? route(option.preview, { disabled: option.disabled }) : "none";
  // Nothing pointed at, a choice that changes nothing, or one not available: your data now.
  const quiet = !option || layout === "none" || layout === "refused";
  const after = flip === "after" && !quiet;
  const pv = option ? focusPreview(option.preview, focus) : null;
  const { shown, more: allMore } = pv && !quiet ? viewsFor(layout, pv) : { shown: [], more: [] };
  // Under the Strip the focused column's values are its large view (beside the strip), not an angle.
  const focusDist = layout === "strip" ? allMore.find((v) => v.kind === "distribution") : undefined;
  const more = focusDist ? allMore.filter((v) => v !== focusDist) : allMore;
  const primary = shown[0] ?? (pv?.views[0] as ConsequenceView | undefined);
  const story = primary && "story" in primary ? (primary.story ?? []) : [];
  const n = pv ? Math.max(0, ...(pv.views as ConsequenceView[]).map((v) => ("story" in v ? (v.story?.length ?? 0) : 0))) : 0;
  const coach = after ? shown.flatMap((v) => v.coach)[0]?.text : undefined;
  const readout = quiet
    ? rest && rest.layout !== "angles"
      ? readoutOfViews(rest.views, true)
      : []
    : readoutFor(option, { after, frame, focus });
  const p = { after, frame, linked };
  const cut = !quiet && (primary?.kind === "row_flow" || (primary?.kind === "distribution" && primary.before_label === "Every measured value"));
  const lines = [...shown, ...more].some((v) => v.kind === "lineage");
  const basis = quiet ? (rest?.basis ?? option?.preview.basis) : pv?.basis;
  const col = layout === "strip" && pv ? stripFocus(pv.strip, option!.preview.views as ConsequenceView[], focus) : null;
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
        {captionOf(option, pv, layout, after, primary, rest)}
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
      {quiet && step && rest ? <RestPicture step={step} rest={rest} option={option} linked={linked} focus={focus} setFocus={setFocus} /> : null}
      {option && pv && !quiet && step && layout === "angles" ? (
        <Angles step={step} angles={pv.angles ?? []} views={pv.views as ConsequenceView[]} activeId={option.id} {...p} />
      ) : null}
      {option && pv && !quiet && layout === "strip" ? (
        <Strip
          cols={pv.strip ?? []}
          {...p}
          // the strip's bars and measures follow the storyboard too: a frame that still draws the
          // data as recorded shows them as recorded
          after={after && !frameStillBefore(primary, frame)}
          focus={col?.column ?? null}
          onFocus={setFocus}
          focusView={focusDist ? <View view={focusDist} after={after} frame={frame} linked={linked} /> : undefined}
          rest={<Views views={shown} {...p} />}
        />
      ) : null}
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
      <Key now={quiet || !after} cut={cut} lines={lines} beside={!!pv?.beside} />
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
      focus={walk.focus}
      setFocus={walk.setFocus}
      title={title}
    />
  );
}
