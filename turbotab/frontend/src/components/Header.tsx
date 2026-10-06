import type { ReactNode } from "react";
import { useHealth } from "../api/queries";
import { Link } from "../router";
import { useTheme } from "../theme";
import styles from "./Header.module.css";

const MOCK = import.meta.env.VITE_MOCK === "1";

export function Header({ children, jobs }: { children?: ReactNode; jobs?: ReactNode }) {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  // Server mode: whose workspace this is, on a machine others may share; and a way out where
  // signing out means something (password sign-in, not the institution's proxy).
  const health = useHealth().data;
  const account = health?.mode === "server" ? health.user : null;
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
        {account ? (
          <span className={styles.account} title="Signed in to this server as">
            {account}
          </span>
        ) : null}
        {account && health?.auth === "password" ? (
          <form method="post" action="/logout" className={styles.signOutForm}>
            <button type="submit" className={styles.theme}>
              Sign out
            </button>
          </form>
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
