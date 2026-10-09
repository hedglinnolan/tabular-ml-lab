/**
 * No dead end (the M3 presentation shell): six reference journeys, from the upload to the Router's
 * last question, answered through the Record in a real browser against the real server. At every
 * step the open question must render something answerable (a bespoke card, or the generic
 * question composed from the server's words), and no request may fail with an error the app does
 * not handle (a 409 is a refusal the app answers at the control, with its exits).
 *
 *   npm run build   (the server serves turbotab/frontend/dist)
 *   TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 TURBOTAB_HOME=$(mktemp -d) \
 *     venv/bin/python -m turbotab.server --port 8874
 *   E2E_BASE_URL=http://127.0.0.1:8874 npx playwright test m3-no-dead-end
 *
 * The journeys: NHANES dietary under inference (the exposure, its effect and the adjustment set,
 * through the generic question) and under prediction; untargeted metabolomics with pooled QCs; a
 * genomics count matrix; a 40-item survey instrument; a clinical longitudinal table (the
 * follow-up, through the generic question). The Router's state is read from the API; every
 * answer is a press in the Record.
 */
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { onMock } from "./backend";
import { nhanesExport } from "./nhanes";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const SAMPLES = resolve(ROOT, "turbotab/sample_data");
const NHANES = nhanesExport();
const RESULTS = resolve(HERE, "../test-results");

const ANSWERABLE =
  '[role="option"]:not([aria-disabled="true"]), button:not([disabled]), select:not([disabled]), input:not([disabled]), [role="combobox"]';

interface Step {
  key: string;
  status: string;
  waiting_on: string[];
  followup: string | null;
  ask: unknown;
}
interface View {
  interview: Step[];
  decisions: { seq: number; decision: { kind: string } }[];
  stages: Record<string, { status: string; error?: string | null }>;
  state: Record<string, unknown>;
}

type Answer = (page: Page, slot: Locator) => Promise<void>;

interface Journey {
  name: string;
  file: string;
  /** How each question is answered; anything else takes its first option. */
  answers: Record<string, Answer>;
  /** The adjustment set's answers, as the fixture's author knows them (truths.py). */
  adjust?: Record<string, [string, string, string]>;
}

// ── reading the Router, and watching the requests ────────────────────────────

async function view(page: Page, pid: string): Promise<View> {
  return page.evaluate(async (p) => (await (await fetch(`/api/projects/${p}`)).json()) as View, pid);
}

/** Console errors and failed requests; a 409 is an answer the app shows, not a failure. */
function watch(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("status of 409")) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.status() !== 409) problems.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });
  page.on("pageerror", (e) => problems.push(`page: ${e.message}`));
  return problems;
}

// ── answering through the Record ─────────────────────────────────────────────

/** Focus an option (the stage previews it) and record it with Enter. */
const option =
  (key: string): Answer =>
  async (_page, slot) => {
    const o = slot.getByTestId(`option-${key}`);
    await o.focus();
    await o.press("Enter");
  };

/** The first option the question offers that can be recorded. */
const first: Answer = async (_page, slot) => {
  const o = slot.locator('[role="option"]:not([aria-disabled="true"])').first();
  await o.focus();
  await o.press("Enter");
};

/** The generic question: an option that opens its choices takes them, then records. */
const generic =
  (key: string | null, choices: Record<string, string> = {}): Answer =>
  async (page, slot) => {
    const o = key
      ? slot.getByTestId(`option-${key}`)
      : slot.locator('[data-generic] [role="option"]:not([aria-disabled="true"])').first();
    const k = await o.getAttribute("data-key");
    await o.click();
    const fields = slot.getByTestId(`fields-${k}`);
    if (await fields.count()) {
      for (const [field, value] of Object.entries(choices)) {
        const c = fields.getByTestId(`choice-${field}-${value}`);
        if (await c.count()) await c.click();
      }
      await slot.getByTestId(`record-${k}`).click();
    }
    void page;
  };

const lens =
  (name: string): Answer =>
  async (page, slot) => {
    const o = slot.getByTestId(`option-${name}`);
    await o.focus();
    await page.keyboard.press("Space");
    await page.keyboard.press("Enter");
  };

const target =
  (column: string): Answer =>
  async (_page, slot) => {
    const box = slot.getByRole("combobox");
    await box.fill(column);
    await box.press("Enter");
    await slot.getByTestId("record-target").click();
  };

const roles: Answer = async (_page, slot) => {
  await slot.getByTestId("record-roles").click();
};

/** The first family on the shelf. */
const models: Answer = async (page, slot) => {
  const o = slot.locator('[role="option"]').first();
  await o.focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("Enter");
};

const substitution: Answer = async (page) => {
  await page.getByRole("button", { name: /^Move energy from / }).first().click();
};

/** The adjustment set: each group the pack guesses alike in one tap, then the rest by their
 *  declared answers ("not sure" where the journey declares none). */
function adjustment(journey: Journey): Answer {
  return async (page, slot) => {
    const group = slot.locator('[data-generic] [role="option"]').first();
    if (await group.count()) {
      await group.click();
      return;
    }
    const matrix = slot.getByTestId("generic-matrix");
    for (const sel of await matrix.locator("select").all()) {
      const id = (await sel.getAttribute("data-testid"))!; // matrix-<column>-<field>
      const m = /^matrix-(.+)-(causes_exposure|causes_outcome|after_exposure)$/.exec(id)!;
      const fields = ["causes_exposure", "causes_outcome", "after_exposure"];
      const answer = journey.adjust?.[m[1]!]?.[fields.indexOf(m[2]!)] ?? "unknown";
      await sel.selectOption(answer);
    }
    await slot.getByTestId("record-matrix").click();
    void page;
  };
}

const openSeal: Answer = async (page) => {
  const step = page.getByTestId("open-seal-step");
  await step.getByTestId("go-open-seal").click();
  const card = page.getByTestId("open-seal-card");
  await card.getByTestId("open-seal").click();
  const exit = card.getByTestId("open-seal-exit").first();
  if (await exit.isVisible({ timeout: 5_000 }).catch(() => false)) await exit.click();
};

/** The ledger's ask card on the open question: confirm every line as shown, else each line, then
 *  the consumer's own ways forward (an energy column's unit and days). */
async function confirmAsk(slot: Locator): Promise<boolean> {
  const card = slot.getByTestId("ask-card");
  if (!(await card.count())) return false;
  const all = card.getByTestId("ask-confirm-all");
  if (await all.count()) await all.click();
  else if (await card.getByTestId("ask-confirm").count()) await card.getByTestId("ask-confirm").first().click();
  else await card.getByTestId("ask-exits").getByRole("button").first().click();
  return true;
}

/** A refusal at the control (in the Record, or on the stage for a pair it draws): take its
 *  first way forward. */
async function takeExit(page: Page): Promise<boolean> {
  const refusal = page.locator('[data-testid="refusal"]:visible').first();
  if (!(await refusal.count())) return false;
  const exit = refusal.locator('[data-testid="refusal-exit"]', { hasText: "instead" }).first();
  if (!(await exit.count())) return false;
  await exit.click();
  return true;
}

const refusedAnywhere = async (page: Page) =>
  (await page.locator('[data-testid="refusal"]:visible').count()) > 0 ||
  (await page.getByTestId("open-seal-refused").isVisible().catch(() => false));

const PURPOSE = (p: string): Answer => option(p);

/** The keys the generic question renders (generic/compose.ts COMPOSERS). */
const GENERIC_KEYS: Record<string, true> = {
  follow_up: true,
  clusters: true,
  estimand: true,
  adjustment: true,
  time_varying: true,
  causal: true,
};


// ── the journeys ─────────────────────────────────────────────────────────────

const NHANES_ADJUST: Record<string, [string, string, string]> = {
  age: ["yes", "yes", "no"],
  gender: ["yes", "yes", "no"],
  cycle_begin_year: ["yes", "yes", "no"],
  ...Object.fromEntries(
    ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"].map((c) => [c, ["unknown", "unknown", "no"]]),
  ),
  ...Object.fromEntries(["weight", "height", "bmi", "waist"].map((c) => [c, ["unknown", "yes", "unknown"]])),
  ...Object.fromEntries(
    ["bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp", "meds_chol"].map((c) => [c, ["no", "yes", "yes"]]),
  ),
};

const JOURNEYS: Journey[] = [
  {
    name: "nhanes-inference",
    file: NHANES,
    adjust: NHANES_ADJUST,
    answers: {
      lens: lens("dietary"),
      target: target("glucose"),
      purpose: PURPOSE("inference"),
      grain: option("one_row_per_unit"),
      roles,
      exclusions: option("none"),
      split: first,
      estimand: generic("sugar", { contrast: "substitution", effect: "total" }),
      energy_adjustment: option("standard"),
      models,
      substitution,
    },
  },
  {
    name: "nhanes-prediction",
    file: NHANES,
    answers: {
      lens: lens("dietary"),
      target: target("glucose"),
      purpose: PURPOSE("prediction"),
      grain: option("one_row_per_unit"),
      roles,
      exclusions: option("none"),
      split: option("0.2"),
      energy_adjustment: option("residual"),
      models,
      substitution,
      open_seal: openSeal,
    },
  },
  {
    name: "metabolomics",
    file: resolve(SAMPLES, "metabolomics_untargeted.csv"),
    answers: {
      lens: lens("metabolomics"),
      orientation: option("sample_major"),
      target: target("responder"),
      event: option("1"),
      purpose: PURPOSE("prediction"),
      grain: option("one_row_per_unit"),
      roles,
      exclusions: option("none"),
      // Most features have blanks (left-censored): complete cases would leave no row.
      missing: option("impute"),
      split: option("0.2"),
      models,
      open_seal: openSeal,
    },
  },
  {
    name: "genomics",
    file: resolve(SAMPLES, "genomics_expression.csv"),
    answers: {
      lens: lens("genomics"),
      orientation: option("sample_major"),
      target: target("condition"),
      event: option("case"),
      purpose: PURPOSE("prediction"),
      grain: option("one_row_per_unit"),
      roles,
      exclusions: option("none"),
      split: option("0.2"),
      models,
      open_seal: openSeal,
    },
  },
  {
    name: "survey",
    file: resolve(SAMPLES, "survey_instrument.csv"),
    answers: {
      lens: lens("survey"),
      target: target("sought_support"),
      event: option("1"),
      purpose: PURPOSE("prediction"),
      grain: option("one_row_per_unit"),
      roles,
      exclusions: option("none"),
      split: option("0.2"),
      models,
      open_seal: openSeal,
    },
  },
  {
    name: "clinical",
    file: resolve(SAMPLES, "clinical_longitudinal.csv"),
    answers: {
      lens: lens("clinical"),
      target: target("progressed"),
      event: option("1"),
      follow_up: generic("same"),
      purpose: PURPOSE("prediction"),
      grain: option("repeated"),
      repeat_kind: option("time_points"),
      unit: option("row"),
      temporal: option("true"),
      roles,
      exclusions: option("none"),
      split: option("0.2"),
      models,
      open_seal: openSeal,
    },
  },
];

// ── the drive ────────────────────────────────────────────────────────────────

const SCREENS = resolve(ROOT, "docs/turbotab-next/m3/screens");

/** The question as it stands, for review (each under 300 KB). */
async function shot(page: Page, slot: Locator, name: string) {
  mkdirSync(SCREENS, { recursive: true });
  await page.mouse.move(1430, 890);
  await slot.scrollIntoViewIfNeeded();
  await page.waitForTimeout(350);
  const path = resolve(SCREENS, `real-${name}.png`);
  // The viewport with the question in it: the Record scrolls in its own column, so an element
  // shot would be cut where the column is.
  await page.screenshot({ path });
  expect(statSync(path).size, `${name} stays under 300 KB`).toBeLessThan(300 * 1024);
}

const timings: Record<string, unknown> = {};

async function upload(page: Page, file: string): Promise<string> {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(file);
  await expect(page).toHaveURL(/\/p\/[^/]+$/, { timeout: 120_000 });
  return decodeURIComponent(/\/p\/([^/]+)$/.exec(page.url())![1]!);
}

/** The first question the Router has not settled, once nothing it waits on is computing. */
async function settle(page: Page, pid: string, timeout = 600_000): Promise<{ v: View; step: Step | null }> {
  const end = Date.now() + timeout;
  for (;;) {
    const v = await view(page, pid);
    const step = v.interview.find((s) => s.status === "open" || s.status === "waiting") ?? null;
    if (!step || step.status === "open") return { v, step };
    // A stage that failed holds the question with the failure shown, never silently: fail here.
    const failed = step.waiting_on.filter((w) => v.stages[w]?.status === "error");
    if (failed.length)
      throw new Error(`${step.key} waits on ${failed.join(", ")}, which failed: ${failed.map((w) => v.stages[w]?.error).join("; ")}`);
    if (Date.now() > end) throw new Error(`${step.key} waits on ${step.waiting_on.join(", ")} for too long`);
    await page.waitForTimeout(400);
  }
}

test.describe.configure({ mode: "serial" });

for (const journey of JOURNEYS) {
  test(`${journey.name}: every open question renders something to answer, to the last one`, async ({ page }) => {
    test.setTimeout(1_800_000);
    await page.goto("/");
    // The app renders once the mock API (if any) has started: ask the backend after that.
    await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
    test.skip(await onMock(page), "the no-dead-end journeys run against the real server");
    const problems = watch(page);
    const t0 = Date.now();
    const pid = await upload(page, journey.file);
    const asked: string[] = [];
    let last = "";
    let tries = 0;
    for (let i = 0; i < 200; i++) {
      const { v, step } = await settle(page, pid);
      if (!step) break;
      const key = step.key;
      tries = key === last ? tries + 1 : 0;
      last = key;
      expect(tries, `${key} stays open after ${tries} answers`).toBeLessThan(6);
      // The open question renders something answerable (the seal's opening is its own card).
      const slot = key === "open_seal" ? page.getByTestId("open-seal-step") : page.locator(`[data-slot="${key}"]`);
      await expect(slot, `${key} renders`).toBeVisible({ timeout: 60_000 });
      if (key !== "open_seal") await expect(slot.getByTestId(`question-${key}`), `${key} asks`).toBeVisible({ timeout: 60_000 });
      await expect
        .poll(async () => slot.locator(ANSWERABLE).count(), { message: `${key}: nothing to answer`, timeout: 60_000 })
        .toBeGreaterThan(0);
      await expect
        .poll(async () => ((await slot.textContent()) ?? "").trim().length, { message: `${key} is blank` })
        .toBeGreaterThan(20);
      if (!asked.includes(key)) {
        asked.push(key);
        // For review: each question the generic renderer or the ask card shows, once.
        if (key in GENERIC_KEYS || (await slot.getByTestId("ask-card").count())) await shot(page, slot, `${journey.name}-${key}`);
      }
      const before = v.decisions.length;
      if (await takeExit(page)) {
        // a refusal from the last press: its way forward
      } else if (await confirmAsk(slot)) {
        // the ledger's card first: the readings this answer's consumer reads
      } else {
        const answer =
          journey.answers[key] ??
          (key === "adjustment" ? adjustment(journey) : key in GENERIC_KEYS ? generic(null) : first);
        await answer(page, slot);
      }
      // Recorded (a decision more), or answered at the control (a refusal to take).
      await expect
        .poll(
          async () =>
            (await view(page, pid)).decisions.length > before || (await refusedAnywhere(page)),
          { message: `${key}: the press did nothing`, timeout: 120_000 },
        )
        .toBe(true);
    }
    const end = await view(page, pid);
    expect(end.interview.filter((s) => s.status === "open" || s.status === "waiting").map((s) => s.key)).toEqual([]);
    timings[journey.name] = { seconds: Math.round((Date.now() - t0) / 1000), asked, decisions: end.decisions.length };
    mkdirSync(RESULTS, { recursive: true });
    writeFileSync(resolve(RESULTS, "m3-no-dead-end.json"), JSON.stringify(timings, null, 1));
    console.log(journey.name, JSON.stringify(timings[journey.name]));
    expect(problems, "no console errors and no failed requests").toEqual([]);
  });
}

// ── the mock's replay of the captures (dev:mock only) ────────────────────────

test("the mock replays a captured journey: the lab lists it, the follow-up is answered, the replay moves on", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  test.skip(!(await onMock(page)), "the replay is the mock's (npm run dev:mock)");
  const problems = watch(page);
  await page.goto("/lab/m3");
  await expect(page.getByTestId("m3-journey-clinical")).toBeVisible();
  await page.getByTestId("m3-journey-clinical").click();
  await page.getByTestId("m3-step-clinical-3").click();
  await expect(page).toHaveURL(/\/p\/m3~clinical~3$/);
  const slot = page.locator('[data-slot="follow_up"]');
  await expect(slot.getByTestId("generic-follow_up")).toBeVisible({ timeout: 30_000 });
  await expect(slot.locator(ANSWERABLE).first()).toBeVisible();
  // The answer the capture recorded moves the replay to the purpose question.
  await generic("same")(page, slot);
  await expect(page.getByTestId("decision-follow_up")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("question-purpose")).toBeVisible({ timeout: 30_000 });
  // An answer it did not record is refused, with the recorded one as its exit.
  await page.getByTestId("question-purpose").getByTestId("option-inference").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("refusal")).toContainText("replay");
  expect(problems).toEqual([]);
});
