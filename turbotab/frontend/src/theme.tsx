/**
 * Light and dark are both first-class. The system decides until the viewer
 * chooses; the choice is remembered per browser and wins in both directions.
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

export type Theme = "light" | "dark";
const KEY = "turbotab.theme";
const QUERY = "(prefers-color-scheme: dark)";

function readStored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function subscribe(cb: () => void): () => void {
  if (!window.matchMedia) return () => {};
  const mq = window.matchMedia(QUERY);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}

const systemDark = () => !!window.matchMedia && window.matchMedia(QUERY).matches;

interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
}

const Ctx = createContext<ThemeCtx>({ theme: "light", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useState<Theme | null>(readStored);
  const dark = useSyncExternalStore(subscribe, systemDark, () => false);
  const theme: Theme = choice ?? (dark ? "dark" : "light");

  useEffect(() => {
    const root = document.documentElement;
    if (choice) root.dataset.theme = choice;
    else delete root.dataset.theme;
    try {
      if (choice) localStorage.setItem(KEY, choice);
      else localStorage.removeItem(KEY);
    } catch {
      /* storage unavailable: the choice lasts for this page only */
    }
  }, [choice]);

  const value = useMemo(
    () => ({ theme, toggle: () => setChoice(theme === "dark" ? "light" : "dark") }),
    [theme],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeCtx {
  return useContext(Ctx);
}
