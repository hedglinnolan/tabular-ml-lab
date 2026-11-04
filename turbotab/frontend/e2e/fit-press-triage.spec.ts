/**
 * The journeys' press of Fit (e2e/fit.ts `pressFit`) where the open noticings' triage holds it,
 * shown against a page written out here, before a journey relies on it. Under Estimate and
 * Describe the server refuses a press of Fit (`triage_open`) while a noticing has no recorded
 * disposition (core/fit_press.py `triage_refusal`), and the Results' Fit shows the refusal's words
 * alone. The press takes the refusal's way forward "Keep the recommended dispositions" (its
 * `confirm_sweep`, the decision the server served, posted as it came) and presses Fit again; a
 * blocking noticing's refusal, whose ways forward record nothing, fails the press as before; a
 * press the server takes posts nothing beside it.
 *
 * It needs no server. The "app" is a page of the Results' Fit markup (`fit-press`, `press-fit`, the
 * refusal's `role="alert"`) served from a made-up origin, and its "server" answers the press with
 * the refusal built here until the triage is recorded.
 *
 *   E2E_BASE_URL=http://press.test npx playwright test fit-press-triage
 */
import { expect, test, type Page } from "@playwright/test";
import { pressFit } from "./fit";

const ORIGIN = "http://press.test";
const PID = "p1";
const LEAD = "Fit waits: pressing it locks the analysis plan, and the open noticings are triaged before the plan is locked.";
const CONFIRM = { kind: "confirm_sweep", stage: "models", sweep: "noticings", lines: [] };
const OPEN = {
  error: {
    code: "triage_open",
    message: `${LEAD} 1 noticing is still open: \`gender\` is two-level text. Recommended: doesn't change your numbers here. Triage it in Models first, so it is decided before any estimate is seen.`,
    exits: [
      { label: "Triage the open noticings in Models", decision: null },
      { label: "Keep the recommended dispositions", decision: CONFIRM },
    ],
  },
};
const BLOCKING = {
  error: {
    code: "triage_open",
    message: `${LEAD} 1 noticing must be resolved first: \`glucose\` takes one value.`,
    exits: [{ label: "Decide: `glucose` takes one value", decision: null }],
  },
};

/** The Results' Fit as the classic screen draws it (Results.tsx `FitPress`): a press clears the
 *  last refusal, and the server's answer either shows its words in an alert or, taken, leaves the
 *  Results with no Fit. */
const html = `<!doctype html><html><body>
<section data-testid="fit-press"><button data-testid="press-fit">Fit</button></section>
<script>
  const section = document.querySelector('[data-testid="fit-press"]');
  const button = section.querySelector("button");
  button.addEventListener("click", async () => {
    section.querySelector('[role="alert"]')?.remove();
    button.disabled = true;
    const res = await fetch("/api/projects/${PID}/fit", { method: "POST" });
    button.disabled = false;
    if (res.ok) return section.remove();
    const alert = document.createElement("p");
    alert.setAttribute("role", "alert");
    alert.textContent = (await res.json()).error.message;
    section.append(alert);
  });
</script></body></html>`;

/** Serves the page and its API on a made-up origin: Fit refused with ``held`` (null: taken) until
 *  a decision is recorded; each POST kept, with its body. */
async function serve(page: Page, held: unknown): Promise<{ method: string; path: string; body: unknown }[]> {
  const posts: { method: string; path: string; body: unknown }[] = [];
  let triaged = held === null;
  const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });
  await page.route(`${ORIGIN}/**`, async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    if (request.method() === "POST") posts.push({ method: "POST", path: pathname, body: request.postDataJSON() });
    if (pathname === `/p/${PID}/classic`) return route.fulfill({ contentType: "text/html", body: html });
    if (pathname === `/api/projects/${PID}/fit`) return route.fulfill(triaged ? json({ locked: true, pressed: true }) : json(held, 409));
    if (pathname === `/api/projects/${PID}/decisions`) {
      triaged = true;
      return route.fulfill(json({ decisions: [] }));
    }
    if (pathname === `/api/projects/${PID}`) return route.fulfill(json({ stages: {} }));
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto(`${ORIGIN}/p/${PID}/classic`);
  return posts;
}

const where = (posts: { path: string; body: unknown }[]) => posts.map((p) => [p.path.split("/").pop(), p.body]);

test("a press the open noticings' triage holds keeps the recommended dispositions and presses Fit again", async ({ page }) => {
  const posts = await serve(page, OPEN);
  await pressFit(page, 10_000);
  expect(where(posts)).toEqual([
    ["fit", null],
    ["decisions", CONFIRM],
    ["fit", null],
  ]);
  await expect(page.getByTestId("fit-press")).toHaveCount(0);
});

test("a blocking noticing's refusal records nothing and fails the press, as any refusal does", async ({ page }) => {
  const posts = await serve(page, BLOCKING);
  await expect(pressFit(page, 10_000)).rejects.toThrow(`Fit was refused: ${BLOCKING.error.message}`);
  expect(where(posts)).toEqual([["fit", null]]);
});

test("a press the server takes posts nothing beside it", async ({ page }) => {
  const posts = await serve(page, null);
  await pressFit(page, 10_000);
  expect(where(posts)).toEqual([["fit", null]]);
});
