import { useLayoutEffect, useRef, useState, type RefObject } from "react";
import { animate } from "motion/react";
import { EASE_OUT, useMotionPrefs } from "../../../motion/prefs";
import { MORPH_S } from "../motion";

export interface Size {
  w: number;
  h: number;
}

/** The content box of an element, kept current with a ResizeObserver. */
export function useSize<T extends HTMLElement>(): [RefObject<T | null>, Size] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // The observer reports once when observation starts, then on every change. Its
    // contentRect is the layout size, unaffected by the transforms a layout animation
    // applies to an ancestor mid-flight (getBoundingClientRect would read those).
    const ro = new ResizeObserver((entries) => {
      const r = entries[entries.length - 1]!.contentRect;
      const w = Math.round(r.width);
      const h = Math.round(r.height);
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/**
 * Morph a flat array of numbers from what is on screen to `target`, calling `draw` every
 * frame. Values are in data-unit coordinates (0..1 of the plot box), so a resize mid-flight
 * only changes how they are drawn, never where the morph is. Interrupting a morph starts
 * the next one from wherever the last frame left off, so rapid arrow presses never jump.
 * The first morph starts from `start` (the recorded values). Instant under reduced motion.
 */
export function useMorph(
  target: Float64Array | null,
  draw: (values: Float64Array) => void,
  start: Float64Array | null,
) {
  const { reduced } = useMotionPrefs();
  const shown = useRef<Float64Array | null>(null);
  const drawRef = useRef(draw);
  const startRef = useRef(start);

  useLayoutEffect(() => {
    drawRef.current = draw;
    startRef.current = start;
    if (shown.current) draw(shown.current); // the box changed size: redraw where we are
  }, [draw, start]);

  useLayoutEffect(() => {
    if (!target) return;
    const from = shown.current ?? startRef.current ?? null;
    if (reduced || !from || from.length !== target.length) {
      shown.current = target.slice();
      drawRef.current(target);
      return;
    }
    const begin = from.slice();
    const buf = new Float64Array(target.length);
    shown.current = buf;
    buf.set(begin);
    drawRef.current(buf);
    const controls = animate(0, 1, {
      duration: MORPH_S,
      ease: EASE_OUT,
      onUpdate: (t) => {
        for (let i = 0; i < buf.length; i++) buf[i] = begin[i]! + (target[i]! - begin[i]!) * t;
        drawRef.current(buf);
      },
      onComplete: () => {
        buf.set(target);
        drawRef.current(buf);
      },
    });
    return () => controls.stop();
  }, [target, reduced]);
}
