/**
 * Morph an array of numbers from what is on screen to a new target — the same people, the same
 * cells, moving to where the previewed option puts them (DESIGN_LANGUAGE §05.2: identity
 * continuity, never swap). Values are written straight to the DOM by `apply`, so React does not
 * re-render sixty times a second. Reduced motion jumps to the target in one frame.
 */
import { useLayoutEffect, useRef } from "react";
import { animate } from "motion/react";
import { DUR, EASE_OUT, useMotionPrefs } from "../../motion/prefs";

/** The preview morph: long enough to follow a dot, short enough to flip through options. */
export const MORPH_S = DUR.arrive;

export function useMorph(
  target: Float64Array,
  apply: (values: Float64Array) => void,
  deps: unknown[] = [],
): void {
  const { reduced } = useMotionPrefs();
  const shown = useRef<Float64Array | null>(null);
  const applyRef = useRef(apply);
  useLayoutEffect(() => {
    applyRef.current = apply;
  });

  useLayoutEffect(() => {
    const from = shown.current;
    if (!from || reduced || from.length !== target.length) {
      const now = Float64Array.from(target);
      shown.current = now;
      applyRef.current(now);
      return;
    }
    const start = Float64Array.from(from);
    const cur = new Float64Array(target.length);
    const controls = animate(0, 1, {
      duration: MORPH_S,
      ease: EASE_OUT,
      onUpdate: (t) => {
        for (let i = 0; i < cur.length; i++) cur[i] = start[i]! + (target[i]! - start[i]!) * t;
        shown.current = cur;
        applyRef.current(cur);
      },
      onComplete: () => {
        shown.current = Float64Array.from(target);
        applyRef.current(shown.current);
      },
    });
    return () => controls.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, reduced, ...deps]);
}
