/**
 * A number that changes tweens from its old value to its new one instead of
 * jumping, so the eye can follow that it is the same quantity. Instant under
 * reduced motion. The text is written straight to the node during the tween, so
 * React does not re-render sixty times a second.
 */
import { useLayoutEffect, useRef } from "react";
import { animate } from "motion/react";
import { DUR, useMotionPrefs } from "./prefs";

export const formatInt = (n: number) => Math.round(n).toLocaleString("en-US");

interface Props {
  value: number;
  format?: (n: number) => string;
  className?: string;
  "data-testid"?: string;
}

export function NumberTween({ value, format = formatInt, className, ...rest }: Props) {
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
      duration: DUR.number,
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

  return (
    <span
      ref={ref}
      className={className ? `num ${className}` : "num"}
      data-value={value}
      {...rest}
    />
  );
}
