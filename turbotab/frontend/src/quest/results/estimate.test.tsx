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
 * a density, a cut, a two-level string's indicator, exposure families with a category or a level
 * beside a number under the linear model, which fits the members together, and families of two
 * numbers under feature-wise regression, which fits each in its own model and gives each row a
 * Benjamini–Hochberg q-value); and the residual
 * method's column (mocks/fixtures/m3-dietary-recalls.json, captured from the real server).
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
/** A served row's truth from statsmodels (fixtures/capture_sequences.py), with its
 *  Benjamini–Hochberg q-value where each member is tested in its own model (capture_terms.py). */
interface Truth {
  estimate: number;
  ci_low: number;
  ci_high: number;
  p: number;
  q?: number;
}
type Sequenced = { artifact: EffectsArtifact; truth?: { crude: { rows: Record<string, Truth> } }; refused?: Record<string, string> };
const Q = sequences as unknown as Record<"dependency" | "sole" | "blocked" | "levels", Sequenced>;
/** The exposure served as one column of another unit, or an exposure family (fit together, with a
 *  member of several rows or a level, or three numbers; or one fit per member, `family_q` with
 *  q-values that print
 *  apart from their p-values), each unadjusted row's truth from statsmodels
 *  (fixtures/capture_terms.py). */
const T = terms as unknown as Record<"density" | "cut" | "two_level" | "family_category" | "family_level" | "family_three" | "family_apart" | "family_q", Sequenced>;
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

// ── an exposure family (ERV4-1, ERV4-2; pp2-family-effect-words) ─────────────

/** The served row of `feature` in the declared model `key`. */
const rowOf = (art: EffectsArtifact, key: string, feature: string) => art.families[0]!.sequence.find((q) => q.key === key)!.effects!.find((e) => e.feature === feature)!;
/** A served number read as printed (three significant digits) is another fit's truth. */
const readsAs = (served: number, truth: Truth) => shows(Number(served.toPrecision(3)), truth.estimate);
/** The served declared model `key`. */
const fitOf = (art: EffectsArtifact, key: string) => art.families[0]!.sequence.find((q) => q.key === key)!;
/** Table 2's drawn row `key` (`<column>:<model>` with several columns drawn): the line under its
 *  label, and its cell under the column header that starts with `col`. */
function rowAt(table: HTMLElement, key: string, col: string): { line: string; cell: string } {
  const heads = [...table.querySelectorAll("thead th")].map((h) => h.textContent ?? "");
  const j = heads.findIndex((h) => h.startsWith(col));
  const tr = table.querySelector(`tbody tr[data-row="${key}"]`)!;
  return { line: tr.querySelector("th small")?.textContent ?? "", cell: tr.children[j]?.textContent ?? "" };
}
/** A row's line for `n` columns its estimate is adjusted for, `n` from the construction. */
const adjustedFor = (n: number) => (n ? `Adjusted for ${n} column${n === 1 ? "" : "s"}` : "No other column in the model");
/** Table 2's header cells. */
const headsOf = (table: HTMLElement) => [...table.querySelectorAll("thead th")].map((h) => h.textContent ?? "");
/** Table 2's footnote for the model `label`: its mark, and its text after the mark. */
function footnoteOf(view: HTMLElement, label: string): { mark: string; text: string } | null {
  for (const li of within(within(view).getByTestId("results-table2")).getByTestId("footnotes").querySelectorAll("li")) {
    const mark = li.querySelector("sup")?.textContent ?? "";
    const text = (li.textContent ?? "").slice(mark.length);
    if (text.startsWith(`${label}: adjusted for `)) return { mark, text };
  }
  return null;
}
/** What a footnote says its rows are adjusted for: "Model 3: adjusted for a, b; …" → [a, b]. */
const namedIn = (text: string) => text.slice(text.indexOf(": adjusted for ") + ": adjusted for ".length).split(/[;.]/)[0]!.split(", ");
/** The footnote mark a drawn row carries after its label. */
const markOn = (table: HTMLElement, key: string) => table.querySelector(`tbody tr[data-row="${key}"] th > span`)?.textContent ?? "";
/** The construction's covariates in Model 2 (capture_terms.py, the feature-wise tests' frame). */
const FOUR = ["age", "sex", "smoking", "activity"];
/** What a joint model's footnote says of the member beside a row's own (estimate.tsx `asTable2`). */
const OTHER = "the other study factor";
/** A q-value printed as the view prints a p-value (views/format.ts `fmtP`: two significant digits,
 *  "<0.001" below that) reads back as `q`. */
function printsQ(printed: string, q: number): boolean {
  if (q < 0.001) return printed === "<0.001";
  if (q >= 0.995) return printed === "1.0";
  return Math.abs(Number(printed) - q) <= 10 ** (Math.floor(Math.log10(q)) - 1) / 2 + 1e-12;
}
/** The line Results drew in place of a family's Table 2 while what the engine said each estimate
 *  is adjusted for was not yet served true for a joint fit. */
const WITHHELD = /does not draw this Table 2|several members are served/;

describe("an exposure family", () => {
  it("serves what each model holds: a joint fit names its members in each model, one fit per member names none (the premise, from statsmodels)", () => {
    // Under the linear model the members are in one fit: its unadjusted rows are statsmodels' one
    // fit of glucose on both, read apart from each member's fit alone (the two-level and levels
    // captures, on the same seeded frame), so each model names both members among the columns it
    // is adjusted for, and the unadjusted model is "Without covariates" (core/stages/effects.py;
    // capture_terms.py `_holds`).
    const level = T.family_level;
    expect(fitOf(level.artifact, "crude").effects!.map((e) => e.feature)).toEqual(Object.keys(level.truth!.crude.rows));
    expect(shows(rowOf(level.artifact, "crude", "arm_y").estimate!, level.truth!.crude.rows.arm_y!.estimate)).toBe(true);
    expect(readsAs(rowOf(level.artifact, "crude", "arm_y").estimate!, T.two_level.truth!.crude.rows.arm_y!)).toBe(false);
    const category = T.family_category;
    expect(fitOf(category.artifact, "crude").effects!.map((e) => e.feature)).toEqual(Object.keys(category.truth!.crude.rows));
    for (const term of ["grp_b", "grp_c", "fiber"]) expect(shows(rowOf(category.artifact, "crude", term).estimate!, category.truth!.crude.rows[term]!.estimate), term).toBe(true);
    for (const level of Object.keys(Q.levels.truth!.crude.rows)) expect(readsAs(rowOf(category.artifact, "crude", level).estimate!, Q.levels.truth!.crude.rows[level]!), level).toBe(false);
    for (const art of [level.artifact, category.artifact]) {
      for (const fit of art.families[0]!.sequence) expect(art.exposures.every((m) => fit.adjusted_for.includes(m)), fit.key).toBe(true);
      expect(fitOf(art, "crude")).toMatchObject({ label: "Without covariates", adjusted_for: art.exposures });
    }
    // Under feature-wise regression each member is fitted in its own model: its unadjusted rows are
    // statsmodels' glucose on each member alone, and no model names the other member.
    const apart = T.family_apart;
    expect(apart.artifact.exposures).toEqual(Object.keys(apart.truth!.crude.rows));
    for (const m of apart.artifact.exposures) expect(shows(rowOf(apart.artifact, "crude", m).estimate!, apart.truth!.crude.rows[m]!.estimate), m).toBe(true);
    for (const fit of apart.artifact.families[0]!.sequence) expect(fit.adjusted_for.some((c) => apart.artifact.exposures.includes(c)), fit.key).toBe(false);
    expect(fitOf(apart.artifact, "crude")).toMatchObject({ label: "Unadjusted", adjusted_for: [] });
  });

  it("draws a joint fit's Table 2 and forest, each member under its heading, each row's line its own and each footnote what every row it marks is adjusted for (family_level)", async () => {
    const art = T.family_level.artifact;
    const view = await estimateView(servedAt(art));
    expect(told(view)).not.toMatch(WITHHELD);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    // Each member's column under its own heading, in the members' order, every declared model's row.
    const columns = ["fiber", "arm_y"];
    const keys = art.families[0]!.sequence.map((q) => q.key);
    expect(drawnRows(view)).toEqual(columns.flatMap((c) => keys.map((k) => `${c}:${k}`)));
    expect([...table.querySelectorAll('tbody tr[data-group="true"]')].map((r) => r.textContent)).toEqual(columns.map((c) => `Per unit of ${c}`));
    for (const k of drawnRows(view)) expect(table.querySelector(`tr[data-row="${k}"] svg[data-mark="${k}"]`), `${k}'s mark`).not.toBeNull();
    // Each unadjusted row is statsmodels' one fit of glucose on both members, adjusted for the
    // other member alone: 1 column, never "No other column in the model".
    const together = Object.keys(T.family_level.truth!.crude.rows);
    for (const c of columns) {
      const truth = T.family_level.truth!.crude.rows[c]!;
      const crude = rowAt(table, `${c}:crude`, "Difference");
      expectShows(crude.cell, [truth.estimate, truth.ci_low, truth.ci_high], `the unadjusted estimate of ${c}`);
      expect(crude.line).toBe(adjustedFor(together.length - 1));
    }
    expect(told(view)).not.toContain("No other column in the model");
    // Each model's footnote is marked on every member's row, so it says what each row's estimate is
    // adjusted for as the construction's design does: the member beside the row's own, as the
    // other study factor (never a member by name, so never a row's own), and from Model 2 the
    // construction's 4 covariates, Model 3 with bmi beside them. It names as many as each row's
    // line counts, the unadjusted model's too, with the engine's note that the members are in it
    // together (the verifier's FW-V2: Model 2's named 4 beside rows that count 5, and the
    // unadjusted rows had no footnote).
    expect(together.length - 1).toBe(1);
    const design: Record<string, string[]> = { crude: [OTHER], model_2: [...FOUR, OTHER], model_3: [...FOUR, OTHER, "bmi"] };
    expect(Object.keys(design)).toEqual(keys);
    for (const [key, named] of Object.entries(design)) {
      const note = footnoteOf(view, unticked(fitOf(art, key).label));
      expect(note, key).not.toBeNull();
      expect(namedIn(note!.text), key).toEqual(named);
      for (const c of columns) {
        expect(rowAt(table, `${c}:${key}`, "Difference").line, `${c}:${key}`).toBe(adjustedFor(named.length));
        expect(markOn(table, `${c}:${key}`), `${c}:${key}`).toBe(note!.mark);
      }
    }
    expect(footnoteOf(view, "Without covariates")!.text).toBe(`Without covariates: adjusted for ${OTHER}; ${unticked(fitOf(art, "crude").note!).replace(/\.$/, "")}.`);
    expect(footnotesIn(view)).not.toMatch(/adjusted for [^.;]*\b(fiber|arm)\b/);
    // Silent: no row of a joint fit carries a q-value, so no q column is drawn.
    expect(art.families[0]!.sequence.every((q) => (q.effects ?? []).every((e) => e.q === null))).toBe(true);
    expect(headsOf(table)).not.toContain("q");
    // The engine's sentence that the members were fit together, each adjusted for the other.
    expect(art.multiplicity).toMatch(/^2 exposures were fit together in each model, so each estimate is adjusted for the other, /);
    expect(footnotesIn(view)).toContain(unticked(art.multiplicity!));
    // The plan-lock line opens Results, and the primary model's diagnostics stay in the engine's words.
    expect(view.firstElementChild?.contains(within(view).getByTestId("results-plan-lock"))).toBe(true);
    const d = art.families[0]!.diagnostics[0]!;
    const item = within(within(view).getByTestId("results-diagnostics")).getByTestId("results-diagnostic");
    expect(item.dataset).toMatchObject({ check: d.check, status: d.status });
    expect(item.textContent).toContain(unticked(d.reading));
  });

  it("says the 2 other study factors three members fit together are each adjusted for, as many as each row's line counts (family_three)", async () => {
    const art = T.family_three.artifact;
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    // Each unadjusted row is statsmodels' one fit of glucose on the three members (the capture's
    // truth), so each is adjusted for the 2 others; from Model 2 the construction's 4 covariates
    // beside them, Model 3 with bmi.
    const members = Object.keys(T.family_three.truth!.crude.rows);
    expect(members.length).toBe(3);
    for (const c of members) {
      const truth = T.family_three.truth!.crude.rows[c]!;
      expectShows(rowAt(table, `${c}:crude`, "Difference").cell, [truth.estimate, truth.ci_low, truth.ci_high], `the unadjusted estimate of ${c}`);
    }
    const others = `the other ${members.length - 1} study factors`;
    const design: Record<string, string[]> = { crude: [others], model_2: [...FOUR, others], model_3: [...FOUR, others, "bmi"] };
    for (const [key, named] of Object.entries(design)) {
      const note = footnoteOf(view, unticked(fitOf(art, key).label))!;
      expect(namedIn(note.text), key).toEqual(named);
      const n = named.length + members.length - 2; // "the other 2 study factors" is 2 columns
      for (const c of members) {
        expect(rowAt(table, `${c}:${key}`, "Difference").line, `${c}:${key}`).toBe(adjustedFor(n));
        expect(markOn(table, `${c}:${key}`), `${c}:${key}`).toBe(note.mark);
      }
    }
    expect(footnotesIn(view)).not.toMatch(/adjusted for [^.;]*\b(fiber|sugar|supplement)\b/);
  });

  it("draws one fit per member's Table 2, each member under its heading, its unadjusted rows with no other column in the model (family_apart)", async () => {
    const art = T.family_apart.artifact;
    const view = await estimateView(servedAt(art));
    expect(told(view)).not.toMatch(WITHHELD);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    const keys = art.families[0]!.sequence.map((q) => q.key);
    expect(drawnRows(view)).toEqual(art.exposures.flatMap((c) => keys.map((k) => `${c}:${k}`)));
    expect([...table.querySelectorAll('tbody tr[data-group="true"]')].map((r) => r.textContent)).toEqual(art.exposures.map((c) => `Per unit of ${c}`));
    // Each unadjusted row is statsmodels' glucose on its member alone; Model 2's, on its member and
    // the 4 covariates of the construction (age, sex, smoking, activity), the other member not.
    const two = fitOf(art, "model_2");
    expect(two.adjusted_for).toEqual(["age", "sex", "smoking", "activity"]);
    for (const c of art.exposures) {
      const truth = T.family_apart.truth!.crude.rows[c]!;
      const crude = rowAt(table, `${c}:crude`, "Difference");
      expectShows(crude.cell, [truth.estimate, truth.ci_low, truth.ci_high], `the unadjusted estimate of ${c}`);
      expect(crude.line).toBe(adjustedFor(0));
      expect(rowAt(table, `${c}:model_2`, "Difference").line).toBe(adjustedFor(4));
    }
    expect(footnotesIn(view)).toContain(`${unticked(two.label)}: adjusted for ${two.adjusted_for.join(", ")}`);
    expect(art.multiplicity).toMatch(/^2 exposures were tested, each in turn; /);
    expect(footnotesIn(view)).toContain(unticked(art.multiplicity!));
    // Each row's q-value beside its P: the unadjusted rows' statsmodels' Benjamini–Hochberg
    // adjustment of the 2 members' p-values (the capture's truth), every row its served field.
    expect(headsOf(table).indexOf("q")).toBe(headsOf(table).indexOf("P") + 1);
    for (const c of art.exposures) {
      expect(printsQ(rowAt(table, `${c}:crude`, "q").cell, T.family_apart.truth!.crude.rows[c]!.q!), c).toBe(true);
      for (const k of keys) expect(printsQ(rowAt(table, `${c}:${k}`, "q").cell, rowOf(art, k, c).q!), `${c}:${k}`).toBe(true);
    }
  });

  it("prints each row's q-value beside its P where its rows carry them, never its p-value (family_q)", async () => {
    // The footnotes say Benjamini–Hochberg q-values control the false-discovery rate across the
    // members, and feature-wise regression serves each row's: Table 2 drew P and not q (the
    // verifier's FW-V5). The members' p-values are far enough from 0 that sugar's q prints apart
    // from its p (the capture's construction), so a column of p-values would not pass.
    const art = T.family_q.artifact;
    const view = await estimateView(servedAt(art));
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(headsOf(table).indexOf("q")).toBe(headsOf(table).indexOf("P") + 1);
    const keys = art.families[0]!.sequence.map((q) => q.key);
    for (const c of art.exposures) {
      const truth = T.family_q.truth!.crude.rows[c]!;
      expect(printsQ(rowAt(table, `${c}:crude`, "q").cell, truth.q!), `${c}: statsmodels' q`).toBe(true);
      for (const k of keys) expect(printsQ(rowAt(table, `${c}:${k}`, "q").cell, rowOf(art, k, c).q!), `${c}:${k}`).toBe(true);
    }
    expect(rowAt(table, "sugar:crude", "q").cell).not.toBe(rowAt(table, "sugar:crude", "P").cell);
    expect(rowAt(table, "sugar:model_2", "q").cell).not.toBe(rowAt(table, "sugar:model_2", "P").cell);
    expect(footnotesIn(view)).toContain("q-values control the false-discovery rate");
  });

  it("draws a joint fit whose other member is served as several rows as one member's Table 2, the other named among its columns and beside it (family_category)", async () => {
    const art = T.family_category.artifact;
    const seq = art.families[0]!.sequence;
    const view = await estimateView(servedAt(art));
    expect(told(view)).not.toMatch(WITHHELD);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    expect(drawnRows(view)).toEqual(seq.map((q) => q.key));
    expect(table.querySelectorAll('tbody tr[data-group="true"]').length).toBe(0);
    // Its title per unit of the member drawn, never of the family's name for itself.
    expect(captionIn(view)).toMatch(perUnitOf("fiber"));
    expect(captionIn(view)).not.toContain(art.exposure);
    // Its unadjusted number is statsmodels' one fit of glucose on fiber and grp's levels, adjusted
    // for grp, 1 column, as its footnote says in the engine's words, which never say fiber is.
    const truth = T.family_category.truth!.crude.rows.fiber!;
    const crude = rowAt(table, "crude", "Difference");
    expectShows(crude.cell, [truth.estimate, truth.ci_low, truth.ci_high], "the unadjusted estimate of fiber");
    expect(crude.line).toBe(adjustedFor(1));
    const note = fitOf(art, "crude").note!;
    expect(note).toContain("are in it together, so each is adjusted for the other");
    expect(footnotesIn(view)).toContain(`Without covariates: adjusted for grp; ${unticked(note).replace(/\.$/, "")}.`);
    expect(footnotesIn(view)).not.toMatch(/adjusted for [^.;]*\bfiber\b/);
    // grp's levels, which one row per model cannot hold, are named beside it for each model.
    for (const fit of seq) expect(unreported(view, fit.key)!.textContent).toContain("grp is reported as 2 rows (grp_b, grp_c)");
    // Silent: one member is drawn, so no footnote says "the other study factor", and no row
    // carries a q-value, so no q column is drawn.
    expect(footnotesIn(view)).not.toContain(OTHER);
    expect(headsOf(table)).not.toContain("q");
  });

  it.each([
    { name: "a difference (NHANES)", at: servedAt(NHANES), art: NHANES },
    { name: "a two-level string's indicator (arm_y)", at: servedAt(T.two_level.artifact), art: T.two_level.artifact },
    { name: "a density", at: servedAt(T.density.artifact), art: T.density.artifact },
    { name: "the residual method's exposure (DIET)", at: servedAt(DIET), art: DIET },
    { name: "a hazard ratio (the walk's capture)", at: C.locked, art: effectsOf(C.locked) },
  ])("draws Table 2 and its forest for a single exposure, each row's line the adapter's: $name", async ({ at, art }) => {
    expect(art.exposures).toEqual([art.exposure]);
    const view = await estimateView(at);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    const keys = art.families[0]!.sequence.filter((q) => (q.effects ?? []).some((e) => e.estimate !== null)).map((q) => q.key);
    expect(drawnRows(view)).toEqual(keys);
    for (const key of keys) {
      expect(table.querySelector(`tr[data-row="${key}"] svg[data-mark="${key}"]`), `${key}'s mark`).not.toBeNull();
      // Silent: a single exposure is never among its model's columns, so each line counts them all.
      expect(rowAt(table, key, "P").line, key).toBe(adjustedFor(fitOf(art, key).adjusted_for.length));
    }
    expect(told(view)).not.toMatch(WITHHELD);
    // Silent: no footnote speaks of another study factor, and no row's q-value is served, so no q column.
    expect(told(view)).not.toContain(OTHER);
    expect(keys.every((k) => fitOf(art, k).effects!.every((e) => e.q === null))).toBe(true);
    expect(headsOf(table)).not.toContain("q");
  });

  it("draws a family that serves one member as a single exposure's: feature-wise regression's {fiber, grp}, grp adjusted for (its served shape)", async () => {
    // Feature-wise regression tests fiber beside grp's indicators and serves fiber's rows only,
    // naming grp among the columns each model is adjusted for and the unadjusted model "Without
    // covariates" (core/stages/effects.py `CRUDE_HOLDING`; core/tests/
    // test_featurewise_family_counts_its_tests.py). Built in that shape from family_category,
    // whose fiber rows are the same least-squares estimates, glucose on fiber with grp's
    // indicators beside it (their intervals the linear model's HC3, as its truth was computed);
    // fiber is not among the columns its own test is adjusted for. The capture's unadjusted note
    // (the members together) and its multiplicity sentence (the linear model's) are not this
    // shape's, so they are left out, not retyped.
    const art = structuredClone(T.family_category.artifact);
    art.multiplicity = null;
    for (const fit of art.families[0]!.sequence) {
      fit.effects = fit.effects!.filter((e) => e.feature === "fiber");
      fit.adjusted_for = fit.adjusted_for.filter((c) => c !== "fiber");
      expect(fit.adjusted_for).toContain("grp");
      if (fit.key === "crude") Object.assign(fit, { label: "Without covariates", note: null });
    }
    expect(art.exposures).toEqual(["fiber", "grp"]);
    const view = await estimateView(servedAt(art));
    expect(told(view)).not.toMatch(WITHHELD);
    const table = within(within(view).getByTestId("results-table2")).getByRole("table");
    const seq = art.families[0]!.sequence;
    expect(drawnRows(view)).toEqual(seq.map((q) => q.key));
    expect(table.querySelectorAll('tbody tr[data-group="true"]').length).toBe(0);
    // Its title per unit of the member, never of the family's name for itself.
    expect(captionIn(view)).toMatch(perUnitOf("fiber"));
    expect(captionIn(view)).not.toContain(art.exposure);
    // Its unadjusted number is statsmodels' fit of glucose on fiber beside grp's levels, and the
    // row says the model is adjusted for grp, never that no other column is in it.
    const truth = T.family_category.truth!.crude.rows.fiber!;
    expectShows(cell(table, "Without covariates", "Difference"), [truth.estimate, truth.ci_low, truth.ci_high], "the unadjusted estimate of fiber");
    expect(table.querySelector('tr[data-row="crude"]')!.textContent).not.toContain("No other column in the model");
    expect(footnotesIn(view)).toContain("Without covariates: adjusted for grp.");
    for (const fit of seq) expect(table.querySelector(`tr[data-row="${fit.key}"] svg[data-mark="${fit.key}"]`), `${fit.key}'s mark`).not.toBeNull();
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
