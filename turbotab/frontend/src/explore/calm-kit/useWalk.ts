/**
 * The walk as a structure uses it: the pure state machine (walk.ts) plus the card's own moment —
 * the option pointed at, the option chosen but not yet recorded, the flip and the storyboard. One
 * hook, one state, so every view on the canvas and every zone on the page agree.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { STEP_BY_ID, optionOf, type Option, type Step } from "./fixture";
import { route, type Layout } from "./router";
import {
  blockedBy,
  chain,
  initial,
  isResult,
  manuscript,
  plan,
  reduce,
  results,
  sentenceCount,
  stepLabel,
  type Action,
  type WalkState,
} from "./walk";

export type Flip = "now" | "after";

export interface WalkApi {
  state: WalkState;
  dispatch: (a: Action) => void;
  /** The open plan step (null on a result moment). */
  step: Step | null;
  /** "Participants · step 2 of 3". */
  label: string;
  pointed: string | null;
  chosen: string | null;
  /** The option the canvas shows: the one pointed at, else the one chosen. */
  active: Option | null;
  layout: Layout;
  flip: Flip;
  /** The storyboard frame shown (null: the final state). */
  frame: number | null;
  point: (id: string | null) => void;
  choose: (id: string) => void;
  setFlip: (f: Flip) => void;
  setFrame: (i: number | null) => void;
  /** Continue: record the chosen option (or move on from a result moment). */
  proceed: () => void;
  canProceed: boolean;
  open: (id: string) => void;
  reset: () => void;
  chain: ReturnType<typeof chain>;
  manuscript: ReturnType<typeof manuscript>;
  sentences: number;
  results: ReturnType<typeof results>;
  plan: ReturnType<typeof plan>;
  blocked: ReturnType<typeof blockedBy>;
}

const reducedMotion = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** The storyboard's frames for an option: the longest story among its views. */
export function storyLength(o: Option | null): number {
  if (!o) return 0;
  return Math.max(0, ...o.preview.views.map((v) => ("story" in v ? (v.story?.length ?? 0) : 0)));
}

interface Moment {
  /** The step this moment belongs to: a new question starts fresh. */
  at: string;
  pointed: string | null;
  flip: Flip;
  /** The storyboard frame shown for `story` (null: the final state). */
  frame: number | null;
  story: string | null;
}

const fresh = (at: string): Moment => ({ at, pointed: null, flip: "now", frame: null, story: null });

export function useWalk(init?: WalkState): WalkApi {
  const [state, dispatch] = useReducer(reduce, init ?? initial());
  const step = isResult(state.open) ? null : STEP_BY_ID[state.open]!;
  const [picked, setPicked] = useState<{ step: string; option: string } | null>(null);
  const [stored, setMoment] = useState<Moment>(() => fresh(state.open));
  // A new question starts on "your data now", with nothing pointed at.
  const moment = stored.at === state.open ? stored : fresh(state.open);
  const update = useCallback(
    (f: (m: Moment) => Partial<Moment>) =>
      setMoment((m) => {
        const base = m.at === state.open ? m : fresh(state.open);
        return { ...base, ...f(base) };
      }),
    [state.open],
  );
  const { pointed, flip } = moment;
  const shownFor = useRef<{ at: string; option: string } | null>(null);

  // The card opens on the recorded answer, or on nothing.
  const chosen = step ? (picked?.step === step.id ? picked.option : (state.answers[step.id] ?? null)) : null;
  const activeId = pointed ?? chosen;
  const active = step ? optionOf(step, activeId) : null;
  const layout: Layout = active ? route(active.preview, { disabled: active.disabled }) : "none";
  const frame = active && moment.story === active.id ? moment.frame : null;

  // Flipping to "with this choice" plays the storyboard once per question; switching options
  // while flipped morphs straight to the new result (BLUEPRINT §11.1).
  useEffect(() => {
    if (flip !== "after" || !active) return;
    const n = storyLength(active);
    const first = !shownFor.current || shownFor.current.at !== state.open;
    shownFor.current = { at: state.open, option: active.id };
    if (!first || n === 0 || reducedMotion()) return;
    const timers: number[] = [];
    for (let i = 0; i <= n; i += 1)
      timers.push(window.setTimeout(() => update(() => ({ story: active.id, frame: i < n ? i : null })), i * 420));
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [flip, active, state.open, update]);

  const point = useCallback(
    (id: string | null) => update(() => (id ? { pointed: id, flip: "after" } : chosen ? { pointed: null } : { pointed: null, flip: "now" })),
    [chosen, update],
  );
  const choose = useCallback(
    (id: string) => {
      if (!step) return;
      setPicked({ step: step.id, option: id });
      update(() => ({ flip: "after" }));
    },
    [step, update],
  );
  const setFlip = useCallback(
    (f: Flip) => {
      if (f === "now") shownFor.current = null;
      update(() => ({ flip: f, frame: null, story: null }));
    },
    [update],
  );
  const setFrame = useCallback((i: number | null) => update(() => ({ frame: i, story: activeId })), [update, activeId]);

  const blocked = useMemo(() => (step ? blockedBy(state, step.id) : []), [state, step]);
  const p = useMemo(() => plan(state), [state]);
  const chosenOption = step ? optionOf(step, chosen) : null;
  const canProceed = step
    ? !!chosenOption && !chosenOption.disabled && !blocked.length && (step.id !== "lock" || p.ok)
    : state.open === "table2";

  const proceed = useCallback(() => {
    if (!step) {
      dispatch({ type: "next" });
      return;
    }
    if (!chosen) return;
    dispatch({ type: "record", step: step.id, option: chosen });
    setPicked(null);
  }, [step, chosen]);

  const open = useCallback((id: string) => dispatch({ type: "open", step: id }), []);
  const reset = useCallback(() => {
    dispatch({ type: "reset" });
    setPicked(null);
  }, []);

  return {
    state,
    dispatch,
    step,
    label: stepLabel(state.open),
    pointed,
    chosen,
    active,
    layout,
    flip: active ? flip : "now",
    frame,
    point,
    choose,
    setFlip,
    setFrame,
    proceed,
    canProceed,
    open,
    reset,
    chain: useMemo(() => chain(state), [state]),
    manuscript: useMemo(() => manuscript(state), [state]),
    sentences: useMemo(() => sentenceCount(state), [state]),
    results: useMemo(() => results(state), [state]),
    plan: p,
    blocked,
  };
}
