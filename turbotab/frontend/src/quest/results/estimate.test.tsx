/**
 * Estimate Results v0 (the front-end audit's G02 v0, G11 and G18) against what the real server
 * served on a light time-to-event walk under Estimate (fixtures/cox-estimate.json, written by
 * fixtures/capture_estimate.py; nothing in it by hand): before the plan's lock, once Fit is pressed
 * and the plan locked, and once the failed proportional-hazards check's first way forward is
 * recorded. Beside it, what the real effects stage serves where Table 2 draws a declared sequence
 * only in part (fixtures/estimate-sequences.json, written by fixtures/capture_sequences.py from
 * the engine tests' own constructions): a primary refused beside an unadjusted model that reports,
 * a family blocked and recorded, a category's levels; what it serves where the exposure's one row
 * is a column of another unit (fixtures/estimate-terms.json, written by fixtures/capture_terms.py:
 * a density, a cut, a two-level string's indicator, and exposure families with a category or a
 * level beside a number); and the residual method's column (mocks/fixtures/m3-dietary-recalls.json,
 * captured from the real server).
 *
 * The truth is the served field: each printed number is read back off the screen and compared with
 * the field it shows, to the three significant digits the views print (views/format.ts), so a
 * number computed in the browser, or shown from the wrong field, fails; each sentence is the
 * engine's, read from the capture. The plan-lock line's time is derived from the served instant by
 * the platform's own ISO formatter and checked against the engine's own wording of the same lock.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { EffectsArtifact } from "../../api/m3-types";
import type { FitLock } from "../../api/schema";
import { ENGINE_LABELS, SERVED } from "../../components/fixtures/stages";
import { table2FromEffects } from "../../components/views/adapters";
import LAB from "../../components/views/lab/fixtures.json";
import diet from "../../mocks/fixtures/m3-dietary-recalls.json";
import survey from "../../mocks/fixtures/m3-survey-design.json";
import { lockOf } from "./estimate";
import capture from "./fixtures/cox-estimate.json";
import sequences from "./fixtures/estimate-sequences.json";
import terms from "./fixtures/estimate-terms.json";
import predict from "./fixtures/nhanes-predict.json";
import { ResultsTapestry } from "./index";

// ── the capture ──────────────────────────────────────────────────────────────

type Json = Record<string, unknown>;
interface Snapshot {
  view: Json & { id: string; state: Json; decisions: Json[]; stages: Record<string, Json> };
  quest: Json & { fit: FitLock | null };
  stages: Record<string, Json & { artifact: unknown }>;
}
type Diagnostic = EffectsArtifact["families"][number]["diagnostics"][number];
const C = capture as unknown as {
  before: Snapshot;
  locked: Snapshot & { plan: { plan_sha256: string; declared_at: string } };
  responded: Snapshot & { taken: Json };
};
const P = predict as unknown as { opened: Snapshot };
/** The engine's effects, cut unchanged from captured journeys: NHANES under inference (a passed
 *  check, on a difference scale), a time-varying exposure (a conditional odds ratio, components/
 *  views/lab/fixtures.json) and the survey design (a check not assessed; mocks/fixtures, one
 *  version of the stage, so its base is the served artifact). */
const NHANES = (LAB as unknown as { nhanes: { effects: EffectsArtifact } }).nhanes.effects;
const ODDS = (LAB as unknown as { timeVarying: { effects: EffectsArtifact } }).timeVarying.effects;
const SURVEY = (survey as unknown as { artifacts: { effects: { base: EffectsArtifact } } }).artifacts.effects.base;
/** Repeated 24-hour recalls under inference, energy-adjusted by the residual method: the exposure
 *  `protein_g` is served as the column the method made of it (one version of the stage). */
const DIET = (diet as unknown as { artifacts: { effects: { base: EffectsArtifact } } }).artifacts.effects.base;
/** A served row's truth from statsmodels (fixtures/capture_sequences.py). */
interface Truth {
  estimate: number;
  ci_low: number;
  ci_high: number;
  p: number;
}
type Sequenced = { artifact: EffectsArtifact; truth?: { crude: { rows: Record<string, Truth> } }; refused?: Record<string, string> };
const Q = sequences as unknown as Record<"dependency" | "sole" | "blocked" | "levels", Sequenced>;
/** The exposure served as one column of another unit, or a family with a member of several rows,
 *  each unadjusted row's truth from statsmodels (fixtures/capture_terms.py). */
const T = terms as unknown as Record<"density" | "cut" | "two_level" | "family_category" | "family_level", Sequenced>;
/** The line the adapter says when Table 2 has no row: "… has no estimate for what you study yet." */
const NONE_YET = table2FromEffects(NHANES).empty!;
/** The locked snapshot with `artifact` served where the walk's effects were. */
const servedAt = (artifact: EffectsArtifact): Snapshot => ({
  ...C.locked,
  stages: { ...C.locked.stages, effects: { ...C.locked.stages.effects!, artifact } },
});
const effectsOf = (s: Snapshot) => s.stages.effects!.artifact as EffectsArtifact & { withheld?: string | null };
const checkOf = (s: Snapshot): Diagnostic => effectsOf(s).families[0]!.diagnostics[0]!;
const unticked = (s: string) => s.replaceAll("`", "");

// ── reading numbers back off the screen (as results.test.tsx reads them) ─────

/** Every number printed in `text`: "1.73 (1.55 to 1.94)" → [1.73, 1.55, 1.94]. */
function numbersIn(text: string): number[] {
  return [...text.matchAll(/[−-]?\d[\d,]*(?:\.\d+)?/g)].map((m) => Number(m[0].replace("−", "-").replace(/,/g, "")));
}

/** A served value printed to three significant digits (whole numbers from 1,000) reads back within
 *  half a unit of its last printed digit. */
function shows(printed: number | undefined, served: number | null | undefined): boolean {
  if (printed === undefined || served === null || served === undefined) return false;
  if (served === 0) return printed === 0;
  const a = Math.abs(served);
  const unit = a >= 1000 ? 1 : 10 ** (Math.floor(Math.log10(a)) - 2);
  return Math.abs(printed - served) <= unit / 2 + 1e-9 * a;
}

function expectShows(text: string | null | undefined, served: (number | null | undefined)[], where: string) {
  const got = numbersIn(text ?? "");
  const want = served.filter((v): v is number => v !== null && v !== undefined);
  expect(got.length, `${where}: "${text}"`).toBe(want.length);
  want.forEach((v, i) => expect(shows(got[i], v), `${where}: printed ${got[i]} for served ${v} in "${text}"`).toBe(true));
}

/** The table's cell for a row (its <th> text starts with `row`) and a column header `col`. */
function cell(table: HTMLElement, row: string, col: string): string | null {
  const heads = [...table.querySelectorAll("thead th")].map((h) => h.textContent ?? "");
  const j = heads.findIndex((h) => h.startsWith(col));
  const tr = [...table.querySelectorAll("tbody tr")].find((r) => (r.querySelector("th")?.textContent ?? "").startsWith(row));
  if (j < 0 || !tr) return null;
  return tr.children[j]?.textContent ?? null;
}

// ── a server that answers as the real one did ────────────────────────────────

function serve(s: Snapshot) {
  const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const path = url.replace(/^.*\/api\/projects\/[^/?]+/, "").replace(/\?.*$/, "");
      if (method !== "GET") return json(500, { detail: `nothing is recorded from the tapestry: ${method} ${path}` });
      if (path === "") return json(200, s.view);
      if (path === "/quest") return json(200, s.quest);
      const stage = /^\/stages\/(\w+)$/.exec(path)?.[1];
      if (stage && s.stages[stage]) return json(200, s.stages[stage]);
      return json(404, { detail: `not in the capture: ${method} ${path}` });
    }),
  );
}

function mount(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

/** The Estimate view, drawn once what it reads is served (the quest log, and the effects where
 *  the stage holds an artifact). */
async function estimateView(s: Snapshot): Promise<HTMLElement> {
  serve(s);
  mount(<ResultsTapestry pid={s.view.id} />);
  return screen.findByTestId("results-estimate");
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// ── the capture itself ───────────────────────────────────────────────────────

describe("the capture", () => {
  it("is journey B under Estimate: withheld before the lock, the sequence and a failed check after it, the response after its exit", () => {
    expect(C.before.view.state.purpose).toBe("inference");
    expect(C.before.quest.fit!.locked).toBe(false);
    expect(effectsOf(C.before).withheld).toBeTruthy();
    expect(effectsOf(C.before).families).toEqual([]);
    expect(C.locked.quest.fit!.locked).toBe(true);
    // Two served fields agree: the quest log's lock and GET /plan's digest of the locked plan.
    expect(C.locked.quest.fit!.sha256).toBe(C.locked.plan.plan_sha256);
    expect(effectsOf(C.locked).scale).toBe("ratio");
    expect(checkOf(C.locked)).toMatchObject({ check: "proportional_hazards", status: "failed", response: null });
    expect(checkOf(C.locked).exits.length).toBeGreaterThan(0);
    expect(checkOf(C.responded).response).toBe(C.responded.taken.action);
    expect(checkOf(C.responded).exits).toEqual([]);
  });
});

// ── the goal's routing ───────────────────────────────────────────────────────

describe("the goal's routing", () => {
  it("draws the estimate under Estimate, not Predict's exhibits", async () => {
    const view = await estimateView(C.locked);
    expect(within(view).getByTestId("results-table2")).toBeTruthy();
    expect(screen.queryByTestId("results-tapestry")).toBeNull();
    expect(screen.queryByTestId("results-cv")).toBeNull();
  });

  it("leaves Predict's tapestry as it is under Predict", async () => {
    serve(P.opened);
    mount(<ResultsTapestry pid={P.opened.view.id} />);
    expect(await screen.findByTestId("results-cv")).toBeTruthy();
    expect(screen.getByTestId("results-tapestry")).toBeTruthy();
    expect(screen.queryByTestId("results-estimate")).toBeNull();
  });
});

// ── the lock's gate (FOUNDATION §5 rule 6) ───────────────────────────────────

/** Every number the locked effects serve for the exposure, on its display scale. */
const servedNumbers = (s: Snapshot) =>
  effectsOf(s).families.flatMap((f) => f.sequence.flatMap((q) => (q.effects ?? []).flatMap((e) => [e.ratio, e.ratio_low, e.ratio_high])));

describe("the lock's gate", () => {
  it("before the lock shows no estimate and no plan-lock line, in the engine's words for why", async () => {
    const view = await estimateView(C.before);
    expect(view.textContent).toContain(unticked(effectsOf(C.before).withheld!));
    expect(within(view).queryByRole("table")).toBeNull();
    expect(within(view).queryByTestId("results-plan-lock")).toBeNull();
    expect(within(view).queryByTestId("results-diagnostics")).toBeNull();
  });

  it("after the lock prints Table 2 and its forest, every number a served field", async () => {
    const view = await estimateView(C.locked);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    const art = effectsOf(C.locked);
    const fam = art.families[0]!;
    const est = `${art.measure_label![0]!.toUpperCase()}${art.measure_label!.slice(1)}`;
    expect(fam.sequence.length).toBeGreaterThan(1); // the declared sequence, not the primary alone
    for (const fit of fam.sequence) {
      const e = fit.effects!.find((x) => x.feature === art.exposure)!;
      // A hazard ratio is printed on its ratio scale: the ratio and its limits, never the log.
      expectShows(cell(table, unticked(fit.label), est), [e.ratio, e.ratio_low, e.ratio_high], `${fit.label}'s estimate`);
      expectShows(cell(table, unticked(fit.label), "n"), [fit.n_rows], `${fit.label}'s rows`);
      expect(e.p!, "both p-values are below the journals' floor").toBeLessThan(0.001);
      expect(cell(table, unticked(fit.label), "P")).toBe("<0.001");
      // Its forest mark sits on its own table row.
      const row = table.querySelector(`tr[data-row="${fit.key}"]`)!;
      expect(row.querySelector(`svg[data-mark="${fit.key}"]`), `${fit.key}'s mark`).not.toBeNull();
    }
    // A ratio's forest is drawn around no effect at 1.
    expect(within(view).getByRole("img", { name: /: no effect at 1$/ })).toBeTruthy();
  });

  it("fails closed: with no lock in the quest log, the served effects are not drawn", async () => {
    for (const quest of [{ ...C.locked.quest, fit: null }, C.before.quest]) {
      const view = await estimateView({ ...C.locked, quest });
      expect(within(view).queryByRole("table")).toBeNull();
      expect(within(view).queryByTestId("results-table2")).toBeNull();
      expect(within(view).queryByTestId("results-plan-lock")).toBeNull();
      expect(numbersIn(view.textContent ?? "").some((n) => servedNumbers(C.locked).some((v) => shows(n, v)))).toBe(false);
      cleanup();
    }
  });

  it("says the engine's withheld words when the lock is in force but the effects are withheld", async () => {
    const withheld = { ...C.locked.stages.effects!, artifact: effectsOf(C.before) };
    const view = await estimateView({ ...C.locked, stages: { ...C.locked.stages, effects: withheld } });
    expect(view.textContent).toContain(unticked(effectsOf(C.before).withheld!));
    expect(within(view).queryByRole("table")).toBeNull();
  });
});

// ── the plan-lock line (G11; FOUNDATION §7) ──────────────────────────────────

describe("the plan-lock line", () => {
  it("says when the plan was fixed, with the locked plan's SHA-256 as its label", async () => {
    const view = await estimateView(C.locked);
    const lock = C.locked.quest.fit!;
    const iso = new Date(lock.at!).toISOString(); // "2026-11-03T20:28:27.456Z"
    const [day, time] = [iso.slice(0, 10), iso.slice(11, 16)];
    // The engine words the same lock with the same day and minute (core/fit_press.py `_when`).
    expect(lock.reason).toContain(`on ${day} at ${time} UTC`);
    const line = within(view).getByTestId("results-plan-lock");
    expect(line.textContent).toContain(`Plan fixed at ${time} UTC on ${day}`);
    expect(within(line).getByTestId("results-plan-sha256").textContent).toBe(`SHA-256 ${C.locked.plan.plan_sha256}`);
    // It opens Results: nothing comes before it.
    expect(view.firstElementChild?.contains(line)).toBe(true);
  });

  it("is the lock in force: its record in the log, by the digest the quest log serves", () => {
    const records = C.locked.view.decisions as unknown as { id: string; decision: { kind: string; digest?: string } }[];
    const record = records.filter((r) => r.decision.kind === "lock_plan");
    expect(record.length).toBe(1);
    expect(record[0]!.decision.digest).toBe(C.locked.quest.fit!.sha256);
    expect(lockOf(C.locked.quest.fit, C.locked.view.decisions as never)).toEqual({ locked: true, plan_lock: record[0]!.id });
    expect(lockOf(C.before.quest.fit, C.before.view.decisions as never)).toEqual({ locked: false, plan_lock: null });
    expect(lockOf(null, C.locked.view.decisions as never)).toEqual({ locked: false, plan_lock: null });
  });
});

// ── Table 2's caption on its display scale ───────────────────────────────────

/** The primary model's served caption. */
const captionOf = (art: EffectsArtifact) => {
  const seq = art.families[0]!.sequence;
  return (seq.find((q) => q.key === "model_2") ?? seq[0]!).inference!.caption;
};

/** Table 2's footnotes, as printed under it. */
const footnotesIn = (view: HTMLElement) => within(within(view).getByTestId("results-table2")).getByTestId("footnotes").textContent ?? "";

describe("Table 2's caption on its display scale", () => {
  // Every number on a ratio scale's Table 2 is a ratio, exp(estimate) (the lock's gate, above), and
  // it has no estimate column; the engine's caption is the coefficient table's, whose rows are
  // the logs. Its sentence on that scale ends at "exp(estimate)." (core/models/inference.py
  // `_on_scale`), and its other sentences say the intervals and the rows.
  const SCALE_ENDS = "exp(estimate).";
  it.each([
    { name: "a hazard ratio (Cox, the walk's capture)", served: effectsOf(C.locked), at: C.locked },
    { name: "an odds ratio (logistic, the time-varying capture)", served: ODDS, at: servedAt(ODDS) },
  ])("on $name, leaves out the caption's sentence on the coefficients' scale and keeps its intervals and rows", async ({ served, at }) => {
    expect(served.scale).toBe("ratio");
    const caption = captionOf(served);
    const cut = caption.indexOf(SCALE_ENDS) + SCALE_ENDS.length;
    const [scale, rest] = [caption.slice(0, cut), caption.slice(cut).trim()];
    // The served caption says its estimates are logs, and its other sentences are the
    // intervals' and the rows' (the primary's served row count among them).
    expect(cut).toBeGreaterThan(SCALE_ENDS.length);
    expect(scale).toMatch(/^Estimates are log/);
    expect(rest).toContain("interval");
    const primary = served.families[0]!.sequence.find((q) => q.key === "model_2")!;
    expect(rest).toMatch(new RegExp(`Estimated from all ${primary.n_rows!.toLocaleString("en-US")} analyzed rows\\.$`));
    const view = await estimateView(at);
    const notes = footnotesIn(view);
    expect(notes).not.toContain(unticked(scale));
    expect(notes).not.toContain("exp(");
    expect(notes).toContain(unticked(rest));
  });

  it("on a difference scale, which prints the estimate itself, keeps the caption whole", async () => {
    expect(NHANES.scale).toBe("difference");
    const view = await estimateView(servedAt(NHANES));
    expect(footnotesIn(view)).toContain(unticked(captionOf(NHANES)));
  });
});

// ── the diagnostics beside Table 2 (G18) ─────────────────────────────────────

describe("the diagnostics beside Table 2", () => {
  it("say a failed check in the engine's words, and name its ways forward without a press", async () => {
    const view = await estimateView(C.locked);
    const d = checkOf(C.locked);
    const item = within(within(view).getByTestId("results-diagnostics")).getByTestId("results-diagnostic");
    expect(item.dataset).toMatchObject({ check: d.check, status: "failed" });
    // The word the researcher reads for the check's state, not only the served status beneath it.
    expect(within(item).getByText("Proportional hazards: failed")).toBeTruthy();
    expect(item.textContent).toContain(unticked(d.reading));
    expect(item.textContent).toContain("No response is recorded yet.");
    for (const x of d.exits) expect(item.textContent).toContain(unticked(x.label));
    // Nothing is recorded from the tapestry (FOUNDATION §1): no way forward is a press.
    expect(within(view).queryAllByRole("button")).toEqual([]);
  });

  it("say a passed check in the engine's words, ask for no response and name no way forward; a difference prints its estimate", async () => {
    // NHANES under inference, served where the walk's effects were.
    const nhanes = NHANES;
    const view = await estimateView(servedAt(nhanes));
    const d = nhanes.families[0]!.diagnostics[0]!;
    expect(d).toMatchObject({ check: "influence", status: "passed", exits: [], response: null });
    const item = within(view).getByTestId("results-diagnostic");
    expect(item.dataset).toMatchObject({ check: d.check, status: "passed" });
    expect(within(item).getByText("Influence: passed")).toBeTruthy();
    expect(item.textContent).toContain(unticked(d.reading));
    expect(item.textContent).not.toContain("No response is recorded yet.");
    expect(item.querySelector("ul")).toBeNull();
    // On a difference scale the estimate and its limits are printed, around no effect at 0.
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    const m2 = nhanes.families[0]!.sequence.find((q) => q.key === "model_2")!;
    const e = m2.effects!.find((x) => x.feature === nhanes.exposure)!;
    expectShows(cell(table, unticked(m2.label), "Difference"), [e.estimate, e.ci_low, e.ci_high], "Model 2's estimate");
    expect(within(view).getByRole("img", { name: /: no effect at 0$/ })).toBeTruthy();
  });

  it("once a way forward is recorded, say the response as the record words it, and no way forward", async () => {
    const view = await estimateView(C.responded);
    const d = checkOf(C.responded);
    const records = C.responded.view.decisions as unknown as { sentence: string; decision: Json }[];
    const [record] = records.filter((r) => r.decision.kind === "respond_diagnostic");
    expect(record!.decision).toEqual(C.responded.taken);
    const item = within(view).getByTestId("results-diagnostic");
    // A response recorded leaves the check as it read: still failed.
    expect(d.status).toBe("failed");
    expect(within(item).getByText("Proportional hazards: failed")).toBeTruthy();
    expect(item.textContent).toContain(unticked(d.reading));
    expect(item.textContent).toContain(unticked(record!.sentence));
    expect(item.textContent).not.toContain("No response is recorded yet.");
    for (const x of checkOf(C.locked).exits) expect(item.textContent).not.toContain(unticked(x.label));
  });

  it("say a check not assessed as not assessed, in the engine's words for why, asking for no response", async () => {
    // The survey design: its model is design-based, so the unweighted checks are not read.
    const view = await estimateView(servedAt(SURVEY));
    const d = SURVEY.families[0]!.diagnostics[0]!;
    expect(d).toMatchObject({ check: "influence", status: "not_assessed", exits: [], response: null });
    const item = within(view).getByTestId("results-diagnostic");
    expect(item.dataset).toMatchObject({ check: d.check, status: "not_assessed" });
    expect(within(item).getByText("Influence: not assessed")).toBeTruthy();
    expect(within(item).getByText(unticked(d.reading))).toBeTruthy();
    expect(item.textContent).not.toContain("No response is recorded yet.");
    expect(item.querySelector("ul")).toBeNull();
  });
});

// ── a fit refused, or not yet run ────────────────────────────────────────────

describe("a fit refused, or not yet run", () => {
  /** The locked snapshot with the effects stage's status replaced, and no effects served. */
  const withEffects = (status: Json): Snapshot => {
    const { effects: _effects, ...stages } = C.locked.stages;
    return { ...C.locked, view: { ...C.locked.view, stages: { ...C.locked.view.stages, effects: { ...C.locked.view.stages.effects!, ...status } } }, stages };
  };

  it("says the stage's refusal in its words, its way forward that is a decision neither pressed nor said", async () => {
    // The engine's served refusal (components/fixtures/stages.json, captured from it), on the
    // effects stage: a way forward that is a decision and one that is none.
    const refusal = { ...SERVED.R.refusal!, stage: "effects" };
    const view = await estimateView(withEffects({ status: "error", fresh: false, error: SERVED.R.error, refusal }));
    const said = within(view).getByTestId("failure-effects");
    expect(said.dataset).toMatchObject({ stage: "effects", kind: "refused" });
    expect(said.textContent).toContain(refusal.message);
    const [decided, told] = [refusal.exits.find((x) => x.decision)!, refusal.exits.find((x) => !x.decision)!];
    expect(said.textContent).toContain(told.label);
    expect(said.textContent).not.toContain(decided.label);
    expect(within(view).queryAllByRole("button")).toEqual([]);
    expect(within(view).queryByRole("table")).toBeNull();
    // The plan's lock stands: the line still opens Results.
    expect(within(view).getByTestId("results-plan-lock")).toBeTruthy();
  });

  it("says the work under way in the engine's words while the effects are computed", async () => {
    const view = await estimateView(withEffects({ status: "running", fresh: false }));
    expect(view.textContent).toContain(ENGINE_LABELS.effects);
    expect(within(view).queryByRole("table")).toBeNull();
  });
});

// ── a declared sequence Table 2 draws in part ────────────────────────────────

/** The table's drawn rows, by key. */
const drawnRows = (view: HTMLElement) => [...within(view).getByRole("table").querySelectorAll("tbody tr[data-row]")].map((r) => r.getAttribute("data-row"));
/** The line said for a declared model Table 2 does not draw. */
const unreported = (view: HTMLElement, key: string) => view.querySelector<HTMLElement>(`[data-testid="results-unreported"][data-fit="${key}"]`);
/** The keys of the declared models said beside Table 2, in the order they are said. */
const saidOrder = (view: HTMLElement) => [...within(view).getByTestId("results-table2").querySelectorAll<HTMLElement>('[data-testid="results-unreported"]')].map((x) => x.dataset.fit);
const literally = (name: string) => name.replace(/[.*+?^$()|[\]{}\\]/g, "\\$&");
/** A claim that a number is per unit of the column `name` itself: the name whole, never the start
 *  of a longer column's (`fiber` in `fiber_per_kcal`, `arm` in `arm_y`). */
const perUnitOf = (name: string) => new RegExp(`per unit of ${literally(name)}(?![\\w'[])`, "i");
/** All a reader is told: the text, and the plots' accessible names. */
const told = (view: HTMLElement) => [view.textContent ?? "", ...[...view.querySelectorAll("[aria-label]")].map((x) => x.getAttribute("aria-label") ?? "")].join(" ");
/** Table 2's caption: its number and its title. */
const captionIn = (view: HTMLElement) => within(view).getByRole("table").querySelector("caption")?.textContent ?? "";
/** A drawn row's cell by its key (a family's rows repeat each model's label). */
function cellAt(table: HTMLElement, key: string, col: string): string | null {
  const heads = [...table.querySelectorAll("thead th")].map((h) => h.textContent ?? "");
  const j = heads.findIndex((h) => h.startsWith(col));
  const tr = table.querySelector(`tbody tr[data-row="${key}"]`);
  return j < 0 || !tr ? null : (tr.children[j]?.textContent ?? null);
}

describe("the exposure's served rows", () => {
  it("draws the column an energy model made of the exposure (the residual method's) as its estimate, under that column's name", async () => {
    const fam = DIET.families[0]!;
    expect(DIET.exposures).toEqual([DIET.exposure]);
    const adjusted = `${DIET.exposure}_adj`;
    const view = await estimateView(servedAt(DIET));
    expect(view.textContent).not.toContain(NONE_YET);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(drawnRows(view)).toEqual(fam.sequence.map((q) => q.key));
    for (const fit of fam.sequence) {
      // Each model serves the exposure's one row, named for the residual method's column.
      const [e, ...rest] = fit.effects!;
      expect(rest).toEqual([]);
      expect(e!.feature).toBe(adjusted);
      expectShows(cell(table, unticked(fit.label), "Difference"), [e!.estimate, e!.ci_low, e!.ci_high], `${fit.label}'s estimate`);
      expectShows(cell(table, unticked(fit.label), "n"), [fit.n_rows], `${fit.label}'s rows`);
      expect(table.querySelector(`tr[data-row="${fit.key}"] svg[data-mark="${fit.key}"]`), `${fit.key}'s mark`).not.toBeNull();
      expect(unreported(view, fit.key)).toBeNull();
    }
    // The estimate is per unit of the residual, the column the method made: its name is that
    // column's, and nothing says it is per unit of the intake itself (ER2-2).
    expect(captionIn(view)).toMatch(perUnitOf(adjusted));
    expect(told(view)).not.toMatch(perUnitOf(DIET.exposure));
  });

  it.each([
    { name: "a density, `fiber` / `kcal`", of: T.density },
    { name: "a cut, `fiber` above 20", of: T.cut },
    { name: "a two-level string's indicator, `arm`'s y against x", of: T.two_level },
  ])("draws a row on another unit than the exposure's column under the served column's name, never per unit of the exposure: $name", async ({ of }) => {
    const art = of.artifact;
    const seq = art.families[0]!.sequence;
    // The premise: every model serves the exposure as one column that is not the exposure's own.
    const [term] = seq[0]!.effects!.map((e) => e.feature);
    expect(term).not.toBe(art.exposure);
    for (const fit of seq) expect(fit.effects!.map((e) => e.feature)).toEqual([term]);
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(drawnRows(view)).toEqual(seq.map((q) => q.key));
    // The unadjusted row is statsmodels' fit of the outcome on that column, written out from the
    // construction; each other row its served field.
    const truth = of.truth!.crude.rows[term!]!;
    const crude = seq.find((q) => q.key === "crude")!;
    expectShows(cell(table, unticked(crude.label), "Difference"), [truth.estimate, truth.ci_low, truth.ci_high], "the unadjusted estimate");
    for (const fit of seq) {
      const [e] = fit.effects!;
      expectShows(cell(table, unticked(fit.label), "Difference"), [e!.estimate, e!.ci_low, e!.ci_high], `${fit.label}'s estimate`);
    }
    // The number is per unit of the served column, and is never said to be per unit of the
    // exposure: in the caption, the forest's head or the plot's name.
    expect(captionIn(view)).toMatch(perUnitOf(term!));
    expect(told(view)).not.toMatch(perUnitOf(art.exposure));
    expect(within(view).queryAllByTestId("results-unreported")).toEqual([]);
  });

  it("names the exposure's own column as the exposure, the same served rows otherwise", async () => {
    // NHANES: the exposure is its own column, so its row is drawn under the exposure's name.
    const m2 = NHANES.families[0]!.sequence.find((q) => q.key === "model_2")!;
    expect(m2.effects!.map((e) => e.feature)).toContain(NHANES.exposure);
    const view = await estimateView(servedAt(NHANES));
    expect(captionIn(view)).toMatch(perUnitOf(NHANES.exposure));
  });

  it("says a category served as its levels is not drawn here, naming the rows, never that there is no estimate", async () => {
    const art = Q.levels.artifact;
    const view = await estimateView(servedAt(art));
    expect(view.textContent).not.toContain(NONE_YET);
    expect(within(view).queryByRole("table")).toBeNull();
    for (const fit of art.families[0]!.sequence) {
      const said = unreported(view, fit.key)!;
      expect(said, fit.key).not.toBeNull();
      expect(said.textContent).toContain(unticked(fit.label));
      for (const e of fit.effects!) expect(said.textContent).toContain(e.feature);
    }
    // No served number of a level is printed as an estimate.
    const served = art.families[0]!.sequence.flatMap((q) => q.effects!.flatMap((e) => [e.estimate, e.ci_low, e.ci_high]));
    expect(numbersIn(view.textContent ?? "").some((n) => served.some((v) => shows(n, v)))).toBe(false);
    // statsmodels' truth for the unadjusted model is what the engine serves (the capture checks it).
    expect(Object.keys(Q.levels.truth!.crude.rows)).toEqual(art.families[0]!.sequence[0]!.effects!.map((e) => e.feature));
  });

  it("says the primary first, then the others in the served sequence's order (ER2-4)", async () => {
    const art = Q.levels.artifact;
    const keys = art.families[0]!.sequence.map((q) => q.key);
    // The premise: the served sequence does not open on the primary, so its order is not this one.
    expect(keys[0]).not.toBe("model_2");
    expect(keys).toContain("model_2");
    const view = await estimateView(servedAt(art));
    expect(saidOrder(view)).toEqual(["model_2", ...keys.filter((k) => k !== "model_2")]);
  });
});

describe("an exposure family", () => {
  it("draws only the members with a drawn row: no heading for a category it does not draw, and the title per unit of the one it does (ER2-3)", async () => {
    const of = T.family_category;
    const art = of.artifact;
    const seq = art.families[0]!.sequence;
    // The premise: the family {fiber, grp}, `grp` served as its levels by every model.
    expect(art.exposures).toEqual(["fiber", "grp"]);
    for (const fit of seq) expect(fit.effects!.map((e) => e.feature)).toEqual(["grp_b", "grp_c", "fiber"]);
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    // One member is drawn: no group heading, and no heading for the member that has no row.
    expect(table.querySelectorAll('tbody tr[data-group="true"]').length).toBe(0);
    expect(told(view)).not.toMatch(perUnitOf("grp"));
    expect(drawnRows(view)).toEqual(seq.map((q) => q.key));
    // The title is per unit of that member, never of the family's served name for itself.
    expect(captionIn(view)).toMatch(perUnitOf("fiber"));
    expect(captionIn(view)).not.toContain(art.exposure);
    const truth = of.truth!.crude.rows.fiber!;
    const crude = seq.find((q) => q.key === "crude")!;
    expectShows(cell(table, unticked(crude.label), "Difference"), [truth.estimate, truth.ci_low, truth.ci_high], "the unadjusted estimate of fiber");
    // `grp` is named as its rows, beside the table, for each model, the primary first.
    expect(saidOrder(view)).toEqual(["model_2", ...seq.map((q) => q.key).filter((k) => k !== "model_2")]);
    for (const fit of seq) expect(unreported(view, fit.key)!.textContent).toContain("grp is reported as 2 rows (grp_b, grp_c)");
  });

  it("draws a member's level indicator under its own name beside a member's own column", async () => {
    const of = T.family_level;
    const art = of.artifact;
    const seq = art.families[0]!.sequence;
    expect(art.exposures).toEqual(["fiber", "arm"]);
    for (const fit of seq) expect(fit.effects!.map((e) => e.feature)).toEqual(["arm_y", "fiber"]);
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    // Each member's rows under a heading that says the column they are per unit of.
    const heads = [...table.querySelectorAll('tbody tr[data-group="true"]')].map((r) => r.textContent);
    expect(heads).toEqual(["Per unit of fiber", "Per unit of arm_y"]);
    expect(told(view)).not.toMatch(perUnitOf("arm"));
    // The unadjusted rows are statsmodels' fit of glucose on both columns.
    for (const term of ["fiber", "arm_y"]) {
      const truth = of.truth!.crude.rows[term]!;
      expectShows(cellAt(table, `${term}:crude`, "Difference"), [truth.estimate, truth.ci_low, truth.ci_high], `the unadjusted estimate of ${term}`);
      for (const fit of seq) {
        const e = fit.effects!.find((x) => x.feature === term)!;
        expectShows(cellAt(table, `${term}:${fit.key}`, "Difference"), [e.estimate, e.ci_low, e.ci_high], `${fit.label}'s ${term}`);
      }
    }
    expect(within(view).queryAllByTestId("results-unreported")).toEqual([]);
  });
});

// ── an exposure family fitted in one model (ERV3-1) ──────────────────────────

/** Table 2's footnotes, each as printed. */
const notesIn = (view: HTMLElement) => [...within(within(view).getByTestId("results-table2")).getByTestId("footnotes").querySelectorAll("li")].map((li) => li.textContent ?? "");
/** The note that says which members a model holds together. */
const TOGETHER = / are in the model together, /;
/** A name as a whole word: `arm` never as the start of `arm_y`. */
const whole = (name: string) => new RegExp(`(?<![\\w'])${literally(name)}(?![\\w'])`);

describe("an exposure family fitted in one model", () => {
  /** The truth of a fit's row printed in a cell, and not another fit's: printed to three
   *  significant digits, the two read apart. */
  function showsTruth(printed: string | null, truth: Truth, not: Truth, where: string) {
    expectShows(printed, [truth.estimate, truth.ci_low, truth.ci_high], where);
    expect(shows(numbersIn(printed ?? "")[0], not.estimate), `${where}: ${not.estimate} is another fit's`).toBe(false);
  }

  it("says beside Table 2 that each model, the unadjusted one included, holds both members, and names them: statsmodels' joint fit, not arm_y's alone", async () => {
    const of = T.family_level;
    const art = of.artifact;
    const seq = art.families[0]!.sequence;
    const crude = seq.find((q) => q.key === "crude")!;
    // The premise. The unadjusted model has no adjustment set, so its row says no other column is
    // in it; yet its served rows are statsmodels' one fit of glucose on arm_y and fiber together,
    // and arm_y's reads apart from statsmodels' fit of glucose on arm_y alone (the two-level
    // capture, on the same seeded frame: capture_terms.py).
    expect(crude.adjusted_for).toEqual([]);
    expect(crude.effects!.map((e) => e.feature)).toEqual(Object.keys(of.truth!.crude.rows));
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(table.querySelector('tr[data-row="arm_y:crude"]')!.textContent).toContain("No other column in the model");
    showsTruth(cellAt(table, "arm_y:crude", "Difference"), of.truth!.crude.rows.arm_y!, T.two_level.truth!.crude.rows.arm_y!, "the unadjusted arm_y");
    // So the footnotes that give each model's adjustment set say, first, that every model holds
    // both members: each named, as the column it is served as.
    const notes = notesIn(view);
    const joint = notes.filter((t) => TOGETHER.test(t));
    expect(joint).toEqual([
      "In each model of this table, the unadjusted one included, fiber and arm (as arm_y) are in the model together, so each is adjusted for the other, beyond the columns the model's row and footnote name.",
    ]);
    expect(notes[0]).toBe(joint[0]);
    for (const name of [...art.exposures, ...crude.effects!.map((e) => e.feature)]) expect(joint[0]).toMatch(whole(name));
    for (const fit of seq.filter((q) => q.adjusted_for.length)) {
      expect(fit.adjusted_for.some((c) => art.exposures.includes(c))).toBe(false);
      expect(notes.some((t) => t.includes(`${unticked(fit.label)}: adjusted for ${fit.adjusted_for.join(", ")}`))).toBe(true);
    }
  });

  it("says it where the member held is not drawn: fiber's unadjusted number is statsmodels' fit beside grp's levels", async () => {
    const of = T.family_category;
    const art = of.artifact;
    const crude = art.families[0]!.sequence.find((q) => q.key === "crude")!;
    expect(crude.adjusted_for).toEqual([]);
    expect(crude.effects!.map((e) => e.feature)).toEqual(Object.keys(of.truth!.crude.rows));
    // The premise, from statsmodels on the one seeded frame: grp's rows here read apart from its
    // fit alone (the levels capture: glucose on grp_b and grp_c), and fiber's unadjusted number
    // here (beside grp) from fiber's beside arm_y (family_level). Both rows say no other column.
    const [joint, alone] = [of.truth!.crude.rows, Q.levels.truth!.crude.rows];
    for (const level of Object.keys(alone)) expect(shows(Number(joint[level]!.estimate.toPrecision(3)), alone[level]!.estimate), level).toBe(false);
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(table.querySelector('tr[data-row="crude"]')!.textContent).toContain("No other column in the model");
    showsTruth(cell(table, unticked(crude.label), "Difference"), joint.fiber!, T.family_level.truth!.crude.rows.fiber!, "the unadjusted fiber");
    expect(notesIn(view).filter((t) => TOGETHER.test(t))).toEqual([
      "In each model of this table, the unadjusted one included, fiber and grp (as grp_b and grp_c) are in the model together, so each is adjusted for the other, beyond the columns the model's row and footnote name.",
    ]);
  });

  it("names the models that hold the members where not every drawn model does (a family's served shape, one model's member row left out)", async () => {
    // Built in the served shape from family_level: Model 3 serves fiber only, so it holds no other
    // member and the note names the two models that do.
    const art = structuredClone(T.family_level.artifact);
    const m3 = art.families[0]!.sequence.find((q) => q.key === "model_3")!;
    m3.effects = m3.effects!.filter((e) => e.feature === "fiber");
    const view = await estimateView(servedAt(art));
    expect(drawnRows(view)).toContain("fiber:model_3");
    expect(notesIn(view).filter((t) => TOGETHER.test(t))).toEqual([
      "In Unadjusted and Model 2 (primary), fiber and arm (as arm_y) are in the model together, so each is adjusted for the other, beyond the columns the model's row and footnote name.",
    ]);
  });

  it.each([
    { name: "a two-level string alone (arm_y)", at: servedAt(T.two_level.artifact) },
    { name: "a difference (NHANES)", at: servedAt(NHANES) },
    { name: "a density", at: servedAt(T.density.artifact) },
    { name: "the residual method's exposure", at: servedAt(DIET) },
    { name: "a hazard ratio (the walk's capture)", at: C.locked },
    { name: "the primary refused beside the unadjusted model", at: servedAt(Q.dependency.artifact) },
  ])("says nothing of the kind for a single exposure: $name", async ({ at }) => {
    const view = await estimateView(at);
    expect(within(view).getByRole("table")).toBeTruthy();
    expect(told(view)).not.toMatch(TOGETHER);
  });

  it("is silent where the single exposure's unadjusted row is its fit alone (statsmodels), which says so", async () => {
    const of = T.two_level;
    const crude = of.artifact.families[0]!.sequence.find((q) => q.key === "crude")!;
    const view = await estimateView(servedAt(of.artifact));
    const table = within(view).getByRole("table");
    expect(table.querySelector('tr[data-row="crude"]')!.textContent).toContain("No other column in the model");
    showsTruth(cell(table, unticked(crude.label), "Difference"), of.truth!.crude.rows.arm_y!, T.family_level.truth!.crude.rows.arm_y!, "arm_y alone");
    expect(notesIn(view).some((t) => TOGETHER.test(t))).toBe(false);
  });
});

describe("a declared model that reports no estimate", () => {
  it.each([
    { name: "Model 2 refused whole for a dependency, the linear model", of: Q.dependency },
    { name: "Model 2 and Model 3 refused as not estimable given their covariates, feature-wise regression", of: Q.sole },
  ])("says each in the engine's words beside the unadjusted model's row: $name", async ({ of }) => {
    const art = of.artifact;
    const seq = art.families[0]!.sequence;
    const view = await estimateView(servedAt(art));
    // The unadjusted model's row is drawn, and its numbers are statsmodels'.
    expect(drawnRows(view)).toEqual(["crude"]);
    const table = within(view).getByRole("table");
    const crude = seq.find((q) => q.key === "crude")!;
    const truth = of.truth!.crude.rows[art.exposure]!;
    expectShows(cell(table, unticked(crude.label), "Difference"), [truth.estimate, truth.ci_low, truth.ci_high], "the unadjusted estimate");
    expect(truth.p).toBeLessThan(0.001);
    expect(cell(table, unticked(crude.label), "P")).toBe("<0.001");
    // Each model with no estimate is said, in its refusal's words, the primary first; none is a
    // footnote marked on a row that is not there.
    const refused = Object.keys(of.refused!);
    expect(refused[0]).toBe("model_2");
    const section = within(view).getByTestId("results-table2");
    const said = [...section.querySelectorAll<HTMLElement>('[data-testid="results-unreported"]')];
    expect(said.map((x) => x.dataset.fit)).toEqual(refused);
    for (const key of refused) {
      const fit = seq.find((q) => q.key === key)!;
      expect(fit.effects).toBeNull();
      expect(unreported(view, key)!.textContent).toContain(`${unticked(fit.label)}: no estimate. ${unticked(of.refused![key]!)}`);
      expect(footnotesIn(view)).not.toContain(`${unticked(fit.label)}:`);
    }
  });

  it("says a family blocked and recorded in its refusal's words, in place of Table 2", async () => {
    const art = Q.blocked.artifact;
    const [fit] = art.families[0]!.sequence;
    expect(fit).toMatchObject({ key: "model_2", effects: null });
    const view = await estimateView(servedAt(art));
    expect(within(view).queryByRole("table")).toBeNull();
    expect(view.textContent).not.toContain(NONE_YET);
    const section = within(view).getByTestId("results-table2");
    expect(within(section).getByRole("heading").textContent).toBe("Table 2");
    expect(unreported(view, "model_2")!.textContent).toBe(`${unticked(fit!.label)}: no estimate. ${unticked(Q.blocked.refused!.model_2!)}`);
  });

  it.each([
    { name: "a hazard ratio (the walk's capture)", at: C.locked },
    { name: "a difference (NHANES)", at: servedAt(NHANES) },
    { name: "the residual method's exposure", at: servedAt(DIET) },
  ])("says nothing more where every declared model reports: $name", async ({ at }) => {
    const view = await estimateView(at);
    expect(within(view).getByRole("table")).toBeTruthy();
    expect(within(view).queryAllByTestId("results-unreported")).toEqual([]);
  });
});

// ── a recorded response's served estimates (G18) ─────────────────────────────

describe("a recorded response's estimates", () => {
  it("are drawn under the check from the served change rows, on the ratio scale, beside the engine's label", async () => {
    const view = await estimateView(C.responded);
    const d = checkOf(C.responded);
    expect(d.change!.length).toBe(2);
    const change = within(view).getByTestId("results-diagnostic-change");
    expect(change.textContent).toContain(unticked(d.change_label!));
    const rows = within(change).getAllByTestId("results-change-row");
    expect(rows.length).toBe(d.change!.length);
    d.change!.forEach((e, i) => {
      expect(within(rows[i]!).getByTestId("results-change-label").textContent).toBe(unticked(e.feature));
      expectShows(within(rows[i]!).getByTestId("results-change-value").textContent, [e.ratio, e.ratio_low, e.ratio_high], `${e.feature}'s ratio`);
    });
  });

  it("are not drawn where no response shows any (a failed check unanswered, a passed check)", async () => {
    for (const at of [C.locked, servedAt(NHANES)]) {
      const view = await estimateView(at);
      expect(within(view).queryByTestId("results-diagnostic-change")).toBeNull();
      cleanup();
    }
  });
});

// ── a failure the engine did not type as a refusal ───────────────────────────

describe("a failure the engine did not type", () => {
  /** The locked snapshot with stages failed in the engine's untyped shapes (components/fixtures/
   *  stages.json: C fails with an error, AC waits on a stage that failed) and no effects served;
   *  the models' question answered, as the server serves the interview, so a failure has a lever. */
  const failing = (statuses: Record<string, Json>): Snapshot => {
    const { effects: _effects, ...stages } = C.locked.stages;
    const models = (C.locked.view.decisions as unknown as { id: string; decision: { kind: string } }[]).find((r) => r.decision.kind === "select_models")!;
    const interview = [{ key: "models", status: "answered", decision_id: models.id, reason: null, waiting_on: [], deferred_findings: [] }];
    return { ...C.locked, view: { ...C.locked.view, interview, stages: { ...C.locked.view.stages, ...statuses } }, stages };
  };
  const failed = (stage: string) => ({ ...SERVED.C, stage }) as unknown as Json;
  const waiting = (stage: string, on: string) => ({ ...SERVED.AC, stage, error: SERVED.AC.error!.replace("'CH'", `'${on}'`) }) as unknown as Json;

  it.each<{ name: string; statuses: Record<string, Json>; at: string }>([
    { name: "the effects stage's own", statuses: { effects: failed("effects") }, at: "effects" },
    { name: "the fit's, which the effects stage waits on", statuses: { fit: failed("fit"), effects: waiting("effects", "fit") }, at: "fit" },
  ])("says $name in its words, and offers no change the shell's Results cannot reach", async ({ statuses, at }) => {
    const view = await estimateView(failing(statuses));
    const said = within(view).getByTestId("failure-effects");
    expect(said.dataset).toMatchObject({ stage: at, kind: "failed" });
    expect(said.textContent).toContain(`${ENGINE_LABELS[at]} did not finish: ${SERVED.C.error!.replace(/^\w+: /, "")}`);
    expect(within(view).queryAllByRole("button").filter((b) => (b.textContent ?? "").startsWith("Change"))).toEqual([]);
    expect(within(view).queryByRole("table")).toBeNull();
  });
});
