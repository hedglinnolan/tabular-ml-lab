/**
 * The stage's bar: what the stage shows (a preview, a finding's evidence, the recorded pipeline),
 * the transform player — one flip, *Your data now ⇄ With this choice*, with the storyboard's step
 * dots beside it — the pinned headline numbers, and, for an option, the record button (the touch
 * equivalent of pressing Enter in the Record).
 */
import type { ReactNode } from "react";
import type { ReadoutItem, Storyboard } from "./tracks";
import { Rich } from "./text";
import { usePlayerStore, usePlayerUi } from "./usePlayer";
import s from "./Stage.module.css";

interface BarProps {
  pill: string | null;
  label: string | null;
  aside: ReactNode;
  loading: boolean;
  /** The record button (an option only). */
  action?: ReactNode;
  children?: ReactNode;
}

export function StageBar({ pill, label, aside, loading, action, children }: BarProps) {
  return (
    <div className={s.bar}>
      <div className={s.barTop}>
        {pill ? (
          <span className={pill === "Evidence" ? s.pillEvidence : s.pill} data-testid="stage-pill">
            {pill}
          </span>
        ) : null}
        {label ? (
          <span className={s.title} data-testid="stage-title">
            <Rich text={label} />
          </span>
        ) : null}
        <span className={s.aside}>{aside}</span>
        {action}
      </div>
      {children}
      <div className={s.loading} data-on={loading || undefined} aria-hidden="true" data-testid="stage-loading" />
    </div>
  );
}

interface PlayerProps {
  story: Storyboard;
  readout: ReadoutItem[];
  /** The views show one state only (evidence, or a choice that changes nothing shown). */
  still: boolean;
  /** What that one state is, said truly (as loaded; with this choice; unchanged). */
  stillLabel?: string | null;
}

/** The flip, the step dots, the step's own label, and the pinned readout. */
export function PlayerControls({ story, readout, still, stillLabel = "Your data as loaded" }: PlayerProps) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const withOn = ui.side === "with";
  const stepLabel = story.labels[ui.nearest] ?? "";
  return (
    <div
      className={s.player}
      data-testid="player"
      data-side={ui.side}
      data-step={ui.nearest}
      data-moving={ui.moving || undefined}
    >
      {!still ? (
        <>
          <div className={s.flip} role="group" aria-label="Show the data">
            <button
              type="button"
              className={s.flipSide}
              aria-pressed={!withOn}
              onClick={() => store.dispatch({ type: "show", side: "now" })}
              data-testid="flip-now"
            >
              Your data now
            </button>
            <button
              type="button"
              className={s.flipSwitch}
              aria-label={withOn ? "Flip to your data now (Space)" : "Flip to with this choice (Space)"}
              onClick={() => store.dispatch({ type: "flip" })}
              data-testid="flip"
            >
              <span className={s.flipKnob} data-side={ui.side} />
            </button>
            <button
              type="button"
              className={s.flipSide}
              aria-pressed={withOn}
              onClick={() => store.dispatch({ type: "show", side: "with" })}
              data-testid="flip-with"
            >
              With this choice <span className={s.flipHint}>(preview)</span>
            </button>
          </div>
          {story.last > 1 ? (
            <div className={s.dots} role="group" aria-label="Storyboard steps">
              {story.labels.map((l, i) => (
                <button
                  key={i}
                  type="button"
                  className={s.dot}
                  data-reached={i <= ui.reached || undefined}
                  data-at={i === ui.nearest || undefined}
                  aria-label={`Step ${i + 1} of ${story.labels.length}: ${l}`}
                  aria-current={i === ui.nearest ? "step" : undefined}
                  title={l}
                  onClick={() => store.dispatch({ type: "seek", step: i })}
                />
              ))}
            </div>
          ) : null}
          <span className={s.stepLabel} aria-live="polite" data-testid="step-label" hidden={story.last < 2}>
            {ui.moving && !ui.forward ? (
              <span className={s.stepArrow} aria-hidden="true">
                ←{" "}
              </span>
            ) : null}
            {stepLabel}
            {ui.moving && ui.forward ? (
              <span className={s.stepArrow} aria-hidden="true">
                {" "}→
              </span>
            ) : null}
          </span>
        </>
      ) : stillLabel ? (
        <span className={s.stepLabel} data-testid="still-label">
          {stillLabel}
        </span>
      ) : null}
      <span className={s.spacer} />
      {readout.length ? (
        <span className={s.readout} data-testid="readout" data-purpose="readout">
          {readout.map((r) => (
            <span key={r.key} className={s.readoutItem}>
              <span className={s.readoutName}>{r.name}</span>
              <span className={withOn || still ? s.readoutDim : s.readoutOn}>{r.before}</span>
              {!still ? (
                <>
                  <span className={s.readoutArrow}>→</span>
                  <span className={withOn ? s.readoutOn : s.readoutDim}>{r.after}</span>
                </>
              ) : null}
            </span>
          ))}
        </span>
      ) : null}
    </div>
  );
}
