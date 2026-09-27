/** Starts the mock API in the browser (npm run dev:mock). Never bundled into a real build. */
import { setupWorker } from "msw/browser";
import { MockServer } from "./db";
import { makeHandlers, seed } from "./handlers";

export async function startMocks(): Promise<void> {
  const server = new MockServer();
  seed(server);
  const worker = setupWorker(...makeHandlers(server));
  await worker.start({
    onUnhandledRequest: "bypass",
    quiet: true,
    serviceWorker: { url: "/mockServiceWorker.js" },
  });
}
