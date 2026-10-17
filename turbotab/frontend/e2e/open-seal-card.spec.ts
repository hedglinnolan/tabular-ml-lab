/**
 * The journeys' opening of the seal from its card (e2e/fit.ts `openSealCard`), shown against a page
 * written out here, before a journey relies on it: the ways forward the card offers are taken in
 * order, the card is pressed again once nothing is refused, and the waits after a press go on while
 * the app computes and end once nothing has, not after a flat time. A way forward is a decision the
 * stages read, so what it changes can recompute before the opening can be pressed again.
 *
 * It needs no server. The "app" is a page of the card's markup (`open-seal-card` with `open-seal`,
 * `open-seal-refused` and `open-seal-exit`, and the header's `job-<stage>` chip while a stage
 * computes) on a made-up origin at the Classic route, and its "server" is a script chosen here.
 *
 *   npx playwright test open-seal-card
 */
import { expect, test, type Page } from "@playwright/test";
import { openSealCard } from "./fit";

const ORIGIN = "http://seal.test";

/** What the made-up server does. ``exits`` are the ways forward the opening asks for, one after
 *  another. ``exitComputeMs`` and ``openComputeMs`` are how long a stage computes (its chip drawn)
 *  after a way forward and after the opening itself; with ``ignoresPress`` the opening does nothing. */
interface Script {
  exits: string[];
  exitComputeMs: number;
  openComputeMs: number;
  ignoresPress?: boolean;
}

const html = `<!doctype html><html><body>
<header id="chips"></header><main id="card"></main>
<script>
  let drawn = "";
  let last = null;
  let pressed = null; // the way forward pressed: its refusal is gone at once, as the app clears it
  function draw(served) {
    last = served;
    const s = { ...served, refusal: served.refusal && served.refusal !== pressed ? served.refusal : null };
    const key = JSON.stringify(s);
    if (key === drawn) return;
    drawn = key;
    document.getElementById("chips").innerHTML = s.computing ? '<ul><li data-testid="job-findings">Checking the table</li></ul>' : "";
    document.getElementById("card").innerHTML = s.opened ? "" :
      '<section data-testid="open-seal-card"><button data-testid="open-seal"' + (s.computing ? " disabled" : "") + '>Open the seal</button>' +
      (s.refusal ? '<div data-testid="open-seal-refused"><button data-testid="open-seal-exit">' + s.refusal + '</button></div>' : "") + '</section>';
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-testid]");
    const id = t && t.getAttribute("data-testid");
    if (id === "open-seal") fetch("/api/projects/p1/decisions", { method: "POST", body: JSON.stringify({ kind: "open_seal" }) });
    if (id === "open-seal-exit") {
      pressed = t.textContent;
      draw(last);
      fetch("/api/projects/p1/decisions", { method: "POST", body: JSON.stringify({ kind: "exit", label: t.textContent }) });
    }
  });
  (async function tick() {
    draw(await (await fetch("/fake/state")).json());
    setTimeout(tick, 40);
  })();
</script></body></html>`;

/** Serves the page and its API on a made-up origin; returns what the server was asked. */
async function serve(page: Page, script: Script): Promise<{ presses: () => number; taken: () => string[] }> {
  const needs = [...script.exits];
  const taken: string[] = [];
  let presses = 0;
  let refusal: string | null = null;
  let computingUntil = 0;
  let openedAt: number | null = null;
  const json = (body: unknown) => ({ contentType: "application/json", body: JSON.stringify(body) });
  await page.route(`${ORIGIN}/**`, async (route) => {
    const { pathname } = new URL(route.request().url());
    if (pathname === "/p/p1/classic") return route.fulfill({ contentType: "text/html", body: html });
    if (pathname === "/fake/state") {
      const now = Date.now();
      return route.fulfill(json({ computing: now < computingUntil, refusal, opened: openedAt !== null && now >= openedAt }));
    }
    if (pathname === "/api/projects/p1/decisions") {
      const body = JSON.parse(route.request().postData() ?? "{}") as { kind: string; label?: string };
      if (body.kind === "open_seal") {
        presses++;
        if (script.ignoresPress) return route.fulfill(json({}));
        if (needs.length) {
          refusal = needs[0]!;
          return route.fulfill({ status: 409, ...json({ error: { code: "refused" } }) });
        }
        computingUntil = Date.now() + script.openComputeMs;
        openedAt = computingUntil;
        return route.fulfill(json({}));
      }
      taken.push(body.label ?? "");
      needs.shift();
      computingUntil = Date.now() + script.exitComputeMs;
      refusal = needs[0] ?? null; // a way forward refused in turn shows its own
      return route.fulfill(json({}));
    }
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto(`${ORIGIN}/p/p1/classic`);
  return { presses: () => presses, taken: () => taken };
}

test("opens at once when nothing is refused", async ({ page }) => {
  const server = await serve(page, { exits: [], exitComputeMs: 0, openComputeMs: 0 });
  expect(await openSealCard(page)).toEqual([]);
  expect(server.presses()).toBe(1);
});

test("takes each way forward in the order the card offers it, then opens", async ({ page }) => {
  const exits = ["Keep the recommended dispositions", "Ridge regression instead"];
  const server = await serve(page, { exits, exitComputeMs: 0, openComputeMs: 0 });
  expect(await openSealCard(page)).toEqual(exits);
  expect(server.taken(), "the server was given them in that order").toEqual(exits);
  expect(server.presses(), "one press, one refusal, and the press again once nothing is refused").toBe(2);
});

test("waits while a stage computes, past the quiet time, and does not press again", async ({ page }) => {
  // The way forward recomputes for 2.5 s with its chip drawn, the opening for 2.5 s more: far past
  // the 600 ms of quiet allowed, and the card is neither pressed again nor given up on meanwhile.
  const server = await serve(page, { exits: ["Keep the recommended dispositions"], exitComputeMs: 2_500, openComputeMs: 2_500 });
  expect(await openSealCard(page, 16, { quiet: 600 })).toEqual(["Keep the recommended dispositions"]);
  expect(server.presses(), "the opening was pressed twice: refused once, then opened").toBe(2);
});

test("ends each wait once nothing has computed for the quiet time", async ({ page }) => {
  // The opening does nothing and nothing computes: each wait ends after the quiet time (600 ms),
  // the card is pressed again, and after the rounds allowed the call returns, not waiting for ever.
  const server = await serve(page, { exits: [], exitComputeMs: 0, openComputeMs: 0, ignoresPress: true });
  const began = Date.now();
  expect(await openSealCard(page, 2, { quiet: 600 })).toEqual([]);
  expect(server.presses(), "the first press and one more for each of the two rounds").toBe(3);
  expect(Date.now() - began, "two quiet times and no more").toBeLessThan(15_000);
});
