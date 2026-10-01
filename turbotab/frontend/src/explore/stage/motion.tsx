/**
 * The stage's transitions: the app's own (motion/prefs — instant under reduced motion),
 * optionally slowed by a review flag. `?slow=10` plays every stage transition ten times
 * slower so a reviewer (or the frame-strip capture) can watch a morph; it changes nothing
 * else. Durations stay the app's: arrive 300 ms, settle 250 ms, number 350 ms.
 */
import { useLayoutEffect, useMemo, useRef } from "react";
import { animate, type Transition } from "motion/react";
import { DUR, useMotionPrefs, useTransitions } from "../../motion/prefs";

function readSlow(): number {
  if (typeof window === "undefined") return 1;
  const v = Number(new URLSearchParams(window.location.search).get("slow"));
  return Number.isFinite(v) && v >= 1 && v <= 50 ? v : 1;
}

export const SLOW = readSlow();

function scale(t: Transition): Transition {
  if (SLOW === 1) return t;
  const d = (t as { duration?: number }).duration;
  return d ? { ...t, duration: d * SLOW } : t;
}

export function useStageTransitions() {
  const t = useTransitions();
  return useMemo(
    () => ({ reduced: t.reduced, arrive: scale(t.arrive), settle: scale(t.settle) }),
    [t],
  );
}

export const MORPH_S = DUR.arrive * SLOW;

/** A number that tweens to its new value (NumberTween, with the stage's slow flag). */
export function Tween({
  value,
  format,
  className,
}: {
  value: number;
  format: (n: number) => string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const shown = useRef<number | null>(null);
  const { reduced } = useMotionPrefs();
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const from = shown.current;
    if (from === null || reduced || from === value) {
      node.textContent = format(value);
      shown.current = value;
      return;
    }
    const controls = animate(from, value, {
      duration: DUR.number * SLOW,
      ease: "easeOut",
      onUpdate: (v) => {
        node.textContent = format(v);
        shown.current = v;
      },
      onComplete: () => {
        node.textContent = format(value);
        shown.current = value;
      },
    });
    return () => controls.stop();
  }, [value, reduced, format]);
  return <span ref={ref} className={className ? `num ${className}` : "num"} data-value={value} />;
}
