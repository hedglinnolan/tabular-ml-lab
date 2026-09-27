/**
 * Motion preferences. `reduced` is the system's prefers-reduced-motion unless the
 * /lab toggle overrides it. When reduced, every motion primitive is instant.
 *
 * DESIGN_LANGUAGE §05.2: motion exists to preserve identity across a state change —
 * settle, arrive, propagate, and the working table under a reshape. Nothing else moves.
 */
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { MotionConfig, type Transition } from "motion/react";

/** Seconds, for Motion. The CSS mirrors live in tokens.css. */
export const DUR = {
  settle: 0.25,
  arrive: 0.3,
  /** Per-section stagger when staleness sweeps downstream (≤150 ms per §05). */
  propagateStepMs: 110,
  number: 0.35,
  row: 0.24,
} as const;

export const EASE_OUT: [number, number, number, number] = [0.2, 0.7, 0.3, 1];

const QUERY = "(prefers-reduced-motion: reduce)";

function subscribe(cb: () => void): () => void {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const mq = window.matchMedia(QUERY);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}

function systemPrefersReduced(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia(QUERY).matches;
}

interface MotionPrefs {
  reduced: boolean;
  system: boolean;
  override: boolean | null;
  setOverride: (v: boolean | null) => void;
}

const MotionPrefsContext = createContext<MotionPrefs>({
  reduced: false,
  system: false,
  override: null,
  setOverride: () => {},
});

export function MotionPrefsProvider({ children }: { children: ReactNode }) {
  const system = useSyncExternalStore(subscribe, systemPrefersReduced, () => false);
  const [override, setOverride] = useState<boolean | null>(null);
  const reduced = override ?? system;

  useEffect(() => {
    // CSS transitions (the stale veil) read this; see base.css.
    document.documentElement.dataset.reducedMotion = String(reduced);
  }, [reduced]);

  const value = useMemo(
    () => ({ reduced, system, override, setOverride }),
    [reduced, system, override],
  );
  return (
    <MotionPrefsContext.Provider value={value}>
      <MotionConfig reducedMotion={reduced ? "always" : "never"}>{children}</MotionConfig>
    </MotionPrefsContext.Provider>
  );
}

export function useMotionPrefs(): MotionPrefs {
  return useContext(MotionPrefsContext);
}

const INSTANT: Transition = { duration: 0 };

/** The four transitions the app is allowed, instant under reduced motion. */
export function useTransitions() {
  const { reduced } = useMotionPrefs();
  return useMemo(
    () => ({
      reduced,
      settle: reduced ? INSTANT : ({ duration: DUR.settle, ease: "easeOut" } as Transition),
      arrive: reduced ? INSTANT : ({ duration: DUR.arrive, ease: EASE_OUT } as Transition),
      row: reduced ? INSTANT : ({ duration: DUR.row, ease: "easeOut" } as Transition),
    }),
    [reduced],
  );
}
