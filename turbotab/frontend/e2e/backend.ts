import type { Page } from "@playwright/test";

/**
 * Whether the page is driven against the mock (npm run dev:mock) rather than the real server.
 *
 * The real server answers /api/health with JSON; the mock's version ends in "-mock". A health
 * request the mock's worker has not intercepted (it has not started yet) reaches Vite, which
 * answers with index.html: a reply that is not JSON, or a request that fails, is the mock's dev
 * server too, so a spec skips instead of crashing before its skip guard.
 */
export async function onMock(page: Page): Promise<boolean> {
  const version = await page.evaluate(async () => {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) return null;
      const body = (await res.json()) as { version?: unknown };
      return typeof body.version === "string" ? body.version : null;
    } catch {
      return null;
    }
  });
  return version === null || version.endsWith("-mock");
}
