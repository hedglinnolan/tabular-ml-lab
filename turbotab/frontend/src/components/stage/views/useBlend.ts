/**
 * The morph between two options' results (arrow keys in the Record): from what is on screen to
 * what the new option shows at the player's position, without replaying its storyboard. Within a
 * unit the drawing glides; across a change of unit the old drawing fades out as the new one fades
 * in. Instant under reduced motion.
 */
import { useCallback, useEffect, useRef } from "react";
import { EASE_OUT, DUR, useMotionPrefs } from "../../../motion/prefs";
import type { Extent, Transition } from "../tracks";
import { transitionBetween } from "../tracks";

export interface Snapshot {
  values: Float64Array;
  extent: Extent;
}

export interface Blend {
  from: Snapshot;
  mode: Transition;
  /** 0 … 1, eased. */
  m: number;
}

function easeOut(t: number): number {
  // EASE_OUT as a cubic Bézier is close to this; the exact curve matters less than the duration.
  const [, y1, , y2] = EASE_OUT;
  const u = 1 - t;
  return 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t;
}

export function useBlend(redraw: () => void) {
  const { reduced } = useMotionPrefs();
  const state = useRef<{ from: Snapshot; mode: Transition; t0: number } | null>(null);
  const frame = useRef(0);
  const redrawRef = useRef(redraw);
  useEffect(() => {
    redrawRef.current = redraw;
  }, [redraw]);

  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const start = useCallback(
    (from: Snapshot | null, to: Extent, sameShape: boolean) => {
      cancelAnimationFrame(frame.current);
      if (!from || reduced) {
        state.current = null;
        return;
      }
      state.current = { from, mode: transitionBetween(from.extent, to, sameShape), t0: performance.now() };
      const loop = () => {
        redrawRef.current();
        if (state.current) frame.current = requestAnimationFrame(loop);
      };
      frame.current = requestAnimationFrame(loop);
    },
    [reduced],
  );

  /** The blend in progress, or null once it has landed. */
  const current = useCallback((): Blend | null => {
    const s = state.current;
    if (!s) return null;
    const t = (performance.now() - s.t0) / (DUR.arrive * 1000);
    if (t >= 1) {
      state.current = null;
      return null;
    }
    return { from: s.from, mode: s.mode, m: easeOut(Math.max(0, t)) };
  }, []);

  const cancel = useCallback(() => {
    cancelAnimationFrame(frame.current);
    state.current = null;
  }, []);

  return { start, current, cancel };
}
