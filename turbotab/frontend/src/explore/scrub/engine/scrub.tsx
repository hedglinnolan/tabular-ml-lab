/**
 * The scrub: one position `t` between "your data now" (0) and "with this choice" (1), shared by
 * every view on the stage, plus the option currently previewed.
 *
 *   drag the thumb            t follows the pointer (no animation: the hand is the motion)
 *   release                   t settles to the nearer end; the far end is a labeled preview
 *   ← / →                     flip to your data now / to the choice (animated, 0.3 s)
 *   focus another option      each view morphs from what is on screen to the new "after"
 *
 * Views call `useMorph(sceneOf, apply)`; they never animate on their own. Under reduced motion
 * every automatic change is instant (a drag still follows the hand — the user is moving it).
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { animate, useMotionValue, type AnimationPlaybackControls, type MotionValue } from "motion/react";
import { EASE_OUT, useMotionPrefs } from "../../../motion/prefs";
import { blend, windowed, type Scene } from "./morph";

/** Seconds. The switch between two options' "after" states, and a flip along the scrub. */
export const SCRUB_DUR = { switch: 0.32, flip: 0.34, settle: 0.18 } as const;

export type Phase = "now" | "moving" | "after";

export interface ScrubState {
  t: MotionValue<number>;
  /** The state at t = 0 (always "now" in this prototype). */
  left: string;
  /** The state at t = 1: the previewed option, or "now" when nothing is previewed. */
  right: string;
  active: string | null;
  phase: Phase;
  dragging: boolean;
  /** Preview `key`; `at: "now"` opens it on the user's data instead of playing to the choice. */
  focus: (key: string | null, at?: "now" | "after") => void;
  flip: (side: "now" | "after") => void;
  scrubTo: (v: number) => void;
  beginDrag: () => void;
  release: () => void;
}

const Ctx = createContext<ScrubState | null>(null);

export function useScrub(): ScrubState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useScrub outside a ScrubProvider");
  return v;
}

export function ScrubProvider({
  children,
  initial = null,
}: {
  children: ReactNode;
  initial?: string | null;
}) {
  const t = useMotionValue(initial ? 1 : 0);
  const [active, setActive] = useState<string | null>(initial);
  const [phase, setPhase] = useState<Phase>(initial ? "after" : "now");
  const [dragging, setDragging] = useState(false);
  const anim = useRef<AnimationPlaybackControls | null>(null);
  const { reduced } = useMotionPrefs();

  useEffect(
    () =>
      t.on("change", (v) => setPhase(v <= 0.001 ? "now" : v >= 0.999 ? "after" : "moving")),
    [t],
  );

  const go = useCallback(
    (target: number, duration: number) => {
      anim.current?.stop();
      if (reduced || duration === 0) {
        t.set(target);
        return;
      }
      anim.current = animate(t, target, { duration, ease: EASE_OUT });
    },
    [t, reduced],
  );

  const value = useMemo<ScrubState>(
    () => ({
      t,
      left: "now",
      right: active ?? "now",
      active,
      phase,
      dragging,
      focus: (key, at = "after") => {
        setActive(key);
        if (at === "now") go(0, 0);
        else go(key ? 1 : 0, SCRUB_DUR.flip);
      },
      flip: (side) => go(side === "now" ? 0 : 1, SCRUB_DUR.flip),
      scrubTo: (v) => {
        anim.current?.stop();
        t.set(Math.max(0, Math.min(1, v)));
      },
      beginDrag: () => {
        anim.current?.stop();
        setDragging(true);
      },
      release: () => {
        setDragging(false);
        go(t.get() >= 0.5 ? 1 : 0, SCRUB_DUR.settle);
      },
    }),
    [t, active, phase, dragging, go],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

interface MorphOptions {
  /** The scene a view shows in a given state ("now", or an option key). Memoize it. */
  sceneOf: (state: string) => Scene;
  /** Write a blended scene to the DOM / canvas. Called every animation frame while moving. */
  apply: (scene: Scene) => void;
  /** The slice of the scrub this view moves in, so causes lead and effects follow. */
  span?: readonly [number, number];
}

/**
 * Drive one view from the scrub. When the previewed option changes, the view morphs from exactly
 * what it is showing (the last blended scene) to the new state, so an interrupted change never
 * jumps.
 */
export function useMorph({ sceneOf, apply, span = [0, 1] }: MorphOptions) {
  const { t, left, right } = useScrub();
  const { reduced } = useMotionPrefs();
  const s = useMotionValue(1);
  const opts = useRef({ sceneOf, apply, span });
  const pair = useRef<{ left: string; right: string } | null>(null);
  const last = useRef<Scene | null>(null);
  const snap = useRef<Scene | null>(null);
  const controls = useRef<AnimationPlaybackControls | null>(null);

  const compute = useCallback(() => {
    const { sceneOf: of, apply: write, span: w } = opts.current;
    const p = pair.current;
    if (!p) return;
    const target = blend(of(p.left), of(p.right), windowed(t.get(), w));
    const sp = windowed(s.get(), w);
    const out = snap.current && sp < 1 ? blend(snap.current, target, sp) : target;
    if (sp >= 1) snap.current = null;
    last.current = out;
    write(out);
  }, [t, s]);

  useLayoutEffect(() => {
    opts.current = { sceneOf, apply, span };
    const prev = pair.current;
    pair.current = { left, right };
    if (prev && (prev.left !== left || prev.right !== right)) {
      controls.current?.stop();
      if (last.current && !reduced) {
        snap.current = last.current;
        s.jump(0);
        controls.current = animate(s, 1, { duration: SCRUB_DUR.switch, ease: EASE_OUT });
      } else {
        snap.current = null;
        s.jump(1);
      }
    }
    compute();
  });

  useEffect(() => {
    const a = t.on("change", compute);
    const b = s.on("change", compute);
    return () => {
      a();
      b();
      controls.current?.stop();
    };
  }, [t, s, compute]);
}

/** A ref registry: views render keyed elements once, then `apply` writes to them by key. */
export function useRegistry<E extends Element = HTMLElement>() {
  const map = useRef(new Map<string, E>());
  const cache = useRef(new Map<string, (el: E | null) => void>());
  const reg = useCallback((key: string) => {
    let cb = cache.current.get(key);
    if (!cb) {
      cb = (el: E | null) => {
        if (el) map.current.set(key, el);
        else map.current.delete(key);
      };
      cache.current.set(key, cb);
    }
    return cb;
  }, []);
  return { map, reg };
}

/** Write inline styles to an element the engine owns (per frame, outside React's render). */
export function css(el: Element, styles: Record<string, string>) {
  Object.assign((el as HTMLElement).style, styles);
}

/** Write SVG attributes the same way. */
export function attrs(el: Element, values: Record<string, string | number>) {
  for (const k in values) el.setAttribute(k, String(values[k]));
}

/** Apply opacity + translate to an element from a scene item (hidden when absent). */
export function place(
  el: HTMLElement | SVGElement,
  item: Record<string, number> | undefined,
  { x = true, y = true }: { x?: boolean; y?: boolean } = {},
) {
  if (!item || (item.o ?? 1) <= 0.002) {
    el.style.opacity = "0";
    el.style.visibility = "hidden";
    return;
  }
  el.style.visibility = "visible";
  el.style.opacity = String(Math.min(1, item.o ?? 1));
  const tx = x ? (item.x ?? 0) : 0;
  const ty = (y ? (item.y ?? 0) : 0) + (item.dy ?? 0);
  el.style.transform = `translate(${tx}px, ${ty}px)`;
}
