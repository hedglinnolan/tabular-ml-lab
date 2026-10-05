/**
 * Two routes do not need a router library: `/` and `/p/:pid` (and, in dev:mock only, `/lab`).
 * The location is the browser's; this only subscribes to it.
 */
import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from "react";

export type Route =
  | { name: "start" }
  | { name: "project"; pid: string }
  | { name: "lab" }
  | { name: "explore-stage" }
  | { name: "stage-lab" }
  | { name: "stage-lab-m2" }
  | { name: "m2-lab" }
  | { name: "m3-lab" }
  | { name: "missing"; path: string };

/**
 * The review surfaces under /lab exist only in npm run dev:mock (INBOX 123, 162): a production
 * build has no /lab route, and the bundler drops their chunks and their fixtures.
 */
export const LAB = import.meta.env.VITE_MOCK === "1";

const EVENT = "turbotab:navigate";

function subscribe(cb: () => void): () => void {
  window.addEventListener("popstate", cb);
  window.addEventListener(EVENT, cb);
  return () => {
    window.removeEventListener("popstate", cb);
    window.removeEventListener(EVENT, cb);
  };
}

const getPath = () => window.location.pathname;

export function parseRoute(path: string, lab = LAB): Route {
  if (path === "/" || path === "") return { name: "start" };
  if (lab) {
    if (path === "/lab" || path === "/lab/") return { name: "lab" };
    if (/^\/lab\/explore\/stage\/?$/.test(path)) return { name: "explore-stage" };
    if (/^\/lab\/stage\/?$/.test(path)) return { name: "stage-lab" };
    if (/^\/lab\/stage\/m2\/?$/.test(path)) return { name: "stage-lab-m2" };
    if (/^\/lab\/m2\/?$/.test(path)) return { name: "m2-lab" };
    if (/^\/lab\/m3\/?$/.test(path)) return { name: "m3-lab" };
  }
  const m = /^\/p\/([^/]+)\/?$/.exec(path);
  if (m) return { name: "project", pid: decodeURIComponent(m[1]!) };
  return { name: "missing", path };
}

export function useRoute(): Route {
  return parseRoute(useSyncExternalStore(subscribe, getPath, () => "/"));
}

export function navigate(to: string): void {
  if (to === window.location.pathname) return;
  window.history.pushState(null, "", to);
  window.dispatchEvent(new Event(EVENT));
  window.scrollTo(0, 0); // a navigation press moves the viewport exactly once (§05.0)
}

export function projectPath(pid: string): string {
  return `/p/${encodeURIComponent(pid)}`;
}

export function Link({
  href,
  onClick,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey)
      return;
    e.preventDefault();
    navigate(href);
  };
  return <a href={href} onClick={handle} {...rest} />;
}
