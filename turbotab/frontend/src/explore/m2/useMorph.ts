/**
 * Option-to-option morphs (BLUEPRINT §11.1 rule 2: arrow keys morph directly between results
 * without replaying the storyboard). The player jumps to the new option's state on the same side;
 * this blends from what was last drawn to the new target over one short beat, so a column folding
 * or a record changing is watched rather than swapped. Instant under reduced motion.
 *
 * `blend(target, key)` is called from a view's draw: when `key` (the option) differs from the
 * last drawing's, the blend starts from that drawing and asks for redraws until it lands.
 */
import { useCallback, useEffect, useRef } from "react";
import { useMotionPrefs } from "../../motion/prefs";

const MORPH_MS = 280;

export function useMorph<T>(lerp: (a: T, b: T, t: number) => T, redraw: () => void) {
  const { reduced } = useMotionPrefs();
  const s = useRef<{ key: string | null; last: T | null; from: T | null; t0: number; frame: number }>({
    key: null,
    last: null,
    from: null,
    t0: 0,
    frame: 0,
  });
  const redrawRef = useRef(redraw);
  useEffect(() => {
    redrawRef.current = redraw;
  }, [redraw]);
  useEffect(() => {
    const st = s.current;
    return () => cancelAnimationFrame(st.frame);
  }, []);

  return useCallback(
    (target: T, key: string): T => {
      const st = s.current;
      if (st.key !== null && st.key !== key && st.last !== null && !reduced) {
        st.from = st.last;
        st.t0 = performance.now();
        cancelAnimationFrame(st.frame);
        const tick = () => {
          redrawRef.current();
          if (st.from !== null) st.frame = requestAnimationFrame(tick);
        };
        st.frame = requestAnimationFrame(tick);
      }
      st.key = key;
      let out = target;
      if (st.from !== null) {
        const t = Math.min(1, (performance.now() - st.t0) / MORPH_MS);
        out = lerp(st.from, target, 1 - Math.pow(1 - t, 3));
        if (t >= 1) st.from = null;
      }
      st.last = out;
      return out;
    },
    [lerp, reduced],
  );
}
