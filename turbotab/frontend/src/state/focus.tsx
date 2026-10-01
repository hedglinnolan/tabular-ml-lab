/**
 * What the stage shows (M1_CONTRACT §10): one focus, shared by the Record, the banner and
 * the stage. It lives in a small context that ProjectScreen creates — never a global.
 *
 *   option   an option being previewed (hover, keyboard focus, arrow keys, a tap)
 *   finding  a finding's evidence
 *   banner   a banner segment's full view (a press; pressing it again goes back to live)
 *   live     nothing focused: the Results once fitted, else "your data now"
 *
 * Two layers make rifling through options fast without flicker: a pointer preview (hover)
 * sits on top of what is held (keyboard focus or a press). Leaving the options does not drop
 * the preview at once: the pointer may be crossing the gap to the stage, so the hover ends
 * after a short grace unless something else takes over.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import type { Decision } from "../api/schema";

export type BannerSegment = "rows" | "columns" | "models" | "result";

export type StageFocus =
  | { kind: "option"; decision: Decision; label: string }
  | { kind: "finding"; findingId: string }
  | { kind: "banner"; segment: BannerSegment }
  | { kind: "live" };

export const LIVE: StageFocus = { kind: "live" };

/** A stable identity for a focus, so two equal foci compare equal. */
export function focusKey(f: StageFocus): string {
  switch (f.kind) {
    case "option":
      return `option:${JSON.stringify(f.decision)}`;
    case "finding":
      return `finding:${f.findingId}`;
    case "banner":
      return `banner:${f.segment}`;
    case "live":
      return "live";
  }
}

export const sameFocus = (a: StageFocus | null, b: StageFocus | null): boolean =>
  a === b || (a !== null && b !== null && focusKey(a) === focusKey(b));

// ── the reducer (pure; the provider adds the hover grace timer) ──────────────

export interface FocusState {
  /** Keyboard focus or a press: stays until released or replaced. */
  held: StageFocus;
  /** A pointer preview, on top of what is held. */
  hover: StageFocus | null;
}

export type FocusAction =
  /** The pointer is over something previewable. */
  | { type: "hover"; focus: StageFocus }
  /** The pointer left (after the grace period). */
  | { type: "unhover" }
  /** Keyboard focus, an arrow key, a tap, or a press on a finding. */
  | { type: "hold"; focus: StageFocus }
  /** What was held lost focus: back to live, unless something else is held by now. */
  | { type: "release"; focus: StageFocus }
  /** A banner segment: pressing the focused one again goes back to live. */
  | { type: "toggle"; focus: StageFocus }
  /** Escape, a recorded answer, a new question: back to live. */
  | { type: "reset" };

export const INITIAL_FOCUS: FocusState = { held: LIVE, hover: null };

export function focusReducer(state: FocusState, action: FocusAction): FocusState {
  switch (action.type) {
    case "hover":
      return sameFocus(state.hover, action.focus) ? state : { ...state, hover: action.focus };
    case "unhover":
      return state.hover === null ? state : { ...state, hover: null };
    case "hold":
      // A deliberate act wins over a lingering pointer preview.
      if (sameFocus(state.held, action.focus) && state.hover === null) return state;
      return { held: action.focus, hover: null };
    case "release":
      if (!sameFocus(state.held, action.focus)) return state;
      return { ...state, held: LIVE };
    case "toggle": {
      const on = sameFocus(effectiveFocus(state), action.focus);
      return { held: on ? LIVE : action.focus, hover: null };
    }
    case "reset":
      return state.hover === null && state.held.kind === "live" ? state : INITIAL_FOCUS;
  }
}

export function effectiveFocus(state: FocusState): StageFocus {
  return state.hover ?? state.held;
}

// ── the context ──────────────────────────────────────────────────────────────

/** How long a pointer preview survives the pointer crossing to the stage (ms). */
export const HOVER_GRACE_MS = 220;

export interface StageFocusApi {
  focus: StageFocus;
  /** Hold a focus (the stage's own `onFocus` uses this). */
  setFocus: (f: StageFocus) => void;
  /** Pointer preview: shown at once, on top of what is held. */
  preview: (f: StageFocus) => void;
  /** The pointer left: the preview ends after a short grace. */
  endPreview: () => void;
  /** Keep the current pointer preview (the pointer reached the stage). */
  keepPreview: () => void;
  /** The holder of `f` lost focus. */
  release: (f: StageFocus) => void;
  /** Press a banner segment: on, or off again. */
  toggle: (f: StageFocus) => void;
  /** Back to live. */
  reset: () => void;
}

const noop = () => {};
const FocusContext = createContext<StageFocusApi>({
  focus: LIVE,
  setFocus: noop,
  preview: noop,
  endPreview: noop,
  keepPreview: noop,
  release: noop,
  toggle: noop,
  reset: noop,
});

export function StageFocusProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(focusReducer, INITIAL_FOCUS);
  const timer = useRef<number | null>(null);

  const cancel = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  }, []);
  useEffect(() => cancel, [cancel]);

  const api = useMemo<Omit<StageFocusApi, "focus">>(
    () => ({
      setFocus: (f) => {
        cancel();
        dispatch({ type: "hold", focus: f });
      },
      preview: (f) => {
        cancel();
        dispatch({ type: "hover", focus: f });
      },
      endPreview: () => {
        cancel();
        timer.current = window.setTimeout(() => {
          timer.current = null;
          dispatch({ type: "unhover" });
        }, HOVER_GRACE_MS);
      },
      keepPreview: cancel,
      release: (f) => dispatch({ type: "release", focus: f }),
      toggle: (f) => {
        cancel();
        dispatch({ type: "toggle", focus: f });
      },
      reset: () => {
        cancel();
        dispatch({ type: "reset" });
      },
    }),
    [cancel],
  );

  const focus = effectiveFocus(state);
  const value = useMemo(() => ({ ...api, focus }), [api, focus]);
  return <FocusContext.Provider value={value}>{children}</FocusContext.Provider>;
}

export function useStageFocus(): StageFocusApi {
  return useContext(FocusContext);
}
