/**
 * Three routes do not need a router library: `/`, `/p/:pid`, `/lab`.
 * The location is the browser's; this only subscribes to it.
 */
import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from "react";

export type Route =
  | { name: "start" }
  | { name: "project"; pid: string }
  | { name: "lab" }
  | { name: "explore-stage" }
  | { name: "stage-lab" }
  | { name: "m2-lab" }
  | { name: "missing"; path: string };

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

export function parseRoute(path: string): Route {
  if (path === "/" || path === "") return { name: "start" };
  if (path === "/lab" || path === "/lab/") return { name: "lab" };
  if (/^\/lab\/explore\/stage\/?$/.test(path)) return { name: "explore-stage" };
  if (/^\/lab\/stage\/?$/.test(path)) return { name: "stage-lab" };
  if (/^\/lab\/m2\/?$/.test(path)) return { name: "m2-lab" };
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
