/**
 * Arrive: a section caused by the answer directly above it grows downward from
 * that cause (a short rise and fade). Under an `AnimatePresence initial={false}`
 * ancestor, content present on first paint does not animate — only arrivals do.
 */
import { motion, type HTMLMotionProps } from "motion/react";
import { useTransitions } from "./prefs";

export function Arrive({ children, ...rest }: HTMLMotionProps<"div">) {
  const t = useTransitions();
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={t.arrive}
      {...rest}
    >
      {children}
    </motion.div>
  );
}
