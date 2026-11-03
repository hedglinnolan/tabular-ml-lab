/**
 * Estimate Results v0 (the front-end audit's G02 v0, G11 and G18) against what the real server
 * served on a light time-to-event walk under Estimate (fixtures/cox-estimate.json, written by
 * fixtures/capture_estimate.py; nothing in it by hand): before the plan's lock, once Fit is pressed
 * and the plan locked, and once the failed proportional-hazards check's first way forward is
 * recorded.
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
import LAB from "../../components/views/lab/fixtures.json";
import survey from "../../mocks/fixtures/m3-survey-design.json";
import { lockOf } from "./estimate";
import capture from "./fixtures/cox-estimate.json";
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
