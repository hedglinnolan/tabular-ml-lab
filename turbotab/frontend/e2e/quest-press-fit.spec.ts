/**
 * The quest journeys' wait after Fit is pressed (e2e/quest.ts `pressFit`), shown against a page
 * written out here, before a journey relies on it. A fit held for the press computes only after it,
 * so the journey reads the server's answer to "is the fit running?" while the shell may already
 * have moved on: the shell opens Results the moment the fit is computed, and Results has no Fit
 * position to look for. The wait must not hold a server's reading against a page that is newer
 * than it, and must still fail where the page really does not show the job it is told is running.
 *
 * It needs no server. The "app" is a page of the shell's markup (`stage-*` with `aria-current`,
 * `fit-position` with `data-running`, `fit-progress`, `fit-cancel`, `card-*`) served from a made-up
 * origin, and its "server" is a clock: the fit runs for a time chosen here after the press and is
 * computed after it. The project's GET reads the fit's status when the request is made and answers
 * after a delay, which is how a slow answer arrives older than the page.
 *
 *   npx playwright test quest-press-fit
 */
import { expect, test, type Page } from "@playwright/test";
import { pressFit } from "./quest";

const ORIGIN = "http://press.test";
const PID = "p1";
const REASON = "Fit trains the two chosen families once you press it.";

/** How the page draws a fit job that runs: as the shell does, with the position marked running
 *  and its progress and Cancel; with the position still the press (the job is not shown); or with
 *  the position marked running and one of the two missing. */
type Drawing = "shell" | "not shown" | "no progress" | "no Cancel";

interface Server {
  /** Milliseconds from the press until the fit is computed (0: it is computed at once). */
  runMs: number;
  /** Milliseconds the project's GET waits before it answers (its reading is taken when asked). */
  readingDelayMs: number;
  drawing: Drawing;
}

/** The made-up page: it follows the clock's state every 40 ms, as the shell follows its events. */
const html = (drawing: Drawing) => `<!doctype html><html><body>
<ol data-testid="quest-line">
  <li><button data-testid="stage-models" aria-current="step">Models</button></li>
  <li><button data-testid="stage-results">Results</button></li>
</ol>
<main id="card"></main>
<script>
  const drawing = ${JSON.stringify(drawing)};
  const stage = (key, current) => {
    const b = document.querySelector('[data-testid="stage-' + key + '"]');
    if (current) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
  };
  const press = ${JSON.stringify(REASON)};
  let drawn = "";
  function draw(state) {
    const view = state.status === "fresh" && state.pressed ? "results" : state.status === "running" && drawing !== "not shown" ? "running" : "press";
    if (view === drawn) return;
    drawn = view;
    stage("models", view !== "results");
    stage("results", view === "results");
    const card = document.getElementById("card");
    if (view === "results") {
      card.innerHTML = '<section data-testid="card-results"><div data-testid="results-cards">Results</div></section>';
    } else if (view === "running") {
      card.innerHTML = '<section data-testid="card-models"><div data-testid="fit-position" data-running="true">' +
        '<span>Fitting the models</span>' +
        (drawing === "no progress" ? "" : '<span role="progressbar" data-testid="fit-progress" style="display:inline-block;width:80px;height:6px"></span>') +
        (drawing === "no Cancel" ? "" : '<button data-testid="fit-cancel">Cancel</button>') +
        '</div></section>';
    } else {
      card.innerHTML = '<section data-testid="card-models"><div data-testid="fit-position"><button data-testid="fit">Fit</button><p>' + press + '</p></div></section>';
      document.querySelector('[data-testid="fit"]').addEventListener("click", () => fetch("/fake/press", { method: "POST" }));
    }
  }
  (async function tick() {
    draw(await (await fetch("/fake/state")).json());
    setTimeout(tick, 40);
  })();
</script></body></html>`;

/** Serves the page and its API from ``server``'s clock on a made-up origin. */
async function serve(page: Page, server: Server): Promise<void> {
  let pressedAt: number | null = null;
  const status = () => (pressedAt === null ? "idle" : Date.now() - pressedAt < server.runMs ? "running" : "fresh");
  const json = (body: unknown) => ({ contentType: "application/json", body: JSON.stringify(body) });
  await page.route(`${ORIGIN}/**`, async (route) => {
    const { pathname } = new URL(route.request().url());
    if (pathname === `/p/${PID}`) return route.fulfill({ contentType: "text/html", body: html(server.drawing) });
    if (pathname === "/fake/state") return route.fulfill(json({ status: status(), pressed: pressedAt !== null }));
    if (pathname === "/fake/press") {
      pressedAt ??= Date.now();
      return route.fulfill(json({}));
    }
    if (pathname === `/api/projects/${PID}/quest`) return route.fulfill(json({ fit: { pressed: false, held: false, reason: REASON }, stages: [] }));
    if (pathname === `/api/projects/${PID}`) {
      const reading = status(); // taken when the request is made, answered later
      await new Promise((done) => setTimeout(done, server.readingDelayMs));
      return route.fulfill(json({ stages: { fit: { status: reading } } }));
    }
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto(`${ORIGIN}/p/${PID}`);
}

test("a server reading older than the page is not held against it: Results is shown, so the wait ends", async ({ page }) => {
  // The first reading says the fit runs (it does, for 2.5 s after the press) and arrives 5 s later,
  // when the page is on Results and has no Fit position.
  await serve(page, { runMs: 2_500, readingDelayMs: 5_000, drawing: "shell" });
  const ran = await pressFit(page, PID, { shown: 3_000 });
  expect(ran, "the page was not seen showing the job, which had finished before the reading came").toBe(false);
  await expect(page.getByTestId("stage-results")).toHaveAttribute("aria-current", "step");
});

test("a fit seen running on Models is checked for its progress and Cancel, and the wait goes on to Results", async ({ page }) => {
  await serve(page, { runMs: 4_000, readingDelayMs: 0, drawing: "shell" });
  expect(await pressFit(page, PID, { shown: 3_000 }), "the job was seen running, with its progress and Cancel").toBe(true);
  await expect(page.getByTestId("stage-results")).toHaveAttribute("aria-current", "step");
});

test("a fit computed before the first look is not a failure either", async ({ page }) => {
  await serve(page, { runMs: 0, readingDelayMs: 0, drawing: "shell" });
  expect(await pressFit(page, PID, { shown: 3_000 })).toBe(false);
});

test("fails where the server runs the fit and Models does not show the job", async ({ page }) => {
  await serve(page, { runMs: 20_000, readingDelayMs: 0, drawing: "not shown" });
  await expect(pressFit(page, PID, { shown: 2_000 })).rejects.toThrow(/serves the fit as running but Models does not show it/);
});

for (const drawing of ["no progress", "no Cancel"] as const) {
  test(`fails where Models shows the job running with ${drawing}`, async ({ page }) => {
    await serve(page, { runMs: 20_000, readingDelayMs: 0, drawing });
    await expect(pressFit(page, PID, { shown: 2_000 })).rejects.toThrow(
      drawing === "no progress" ? /running without its progress and Cancel \(progress false, Cancel true\)/ : /running without its progress and Cancel \(progress true, Cancel false\)/,
    );
  });
}
