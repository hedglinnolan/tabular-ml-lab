import type { ReactNode } from "react";
import { Link } from "../router";
import { useTheme } from "../theme";
import styles from "./Header.module.css";

const MOCK = import.meta.env.VITE_MOCK === "1";

export function Header({ children, jobs }: { children?: ReactNode; jobs?: ReactNode }) {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <header className={styles.header}>
      <Link href="/" className={styles.brand}>
        TurboTab
      </Link>
      {MOCK ? (
        <span
          className={styles.mock}
          title="Every number on this page comes from the in-browser mock server."
        >
          mock server
        </span>
      ) : null}
      <div className={styles.context}>{children}</div>
      <div className={styles.jobs}>{jobs}</div>
      <nav className={styles.nav} aria-label="App">
        {/* The review surfaces exist only in dev:mock (INBOX 123, 162). */}
        {MOCK ? (
          <Link href="/lab" className={styles.navLink}>
            Motion lab
          </Link>
        ) : null}
        <button
          type="button"
          className={styles.theme}
          onClick={toggle}
          aria-label={`Switch to ${next} theme`}
          title={`Switch to ${next} theme`}
        >
          <span aria-hidden="true" className={styles.themeIcon} data-theme-icon={theme} />
          {theme === "dark" ? "Dark" : "Light"}
        </button>
      </nav>
    </header>
  );
}
