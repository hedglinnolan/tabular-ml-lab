/**
 * Predict Results v0 (W1-G) against what the real server served on a light NHANES walk under
 * Predict (fixtures/nhanes-predict.json, written by fixtures/capture.py; nothing in it by hand).
 *
 * The truth is the served field: each printed number is read back off the screen and compared with
 * the field it shows, to the three significant digits the views print (views/format.ts), so a
 * number computed in the browser, or shown from the wrong field, fails. The chain of refusals is
 * walked press by press against the server's own responses, in the order it gave them, and every
 * posted body is the one the real walk posted.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitArtifact } from "../../api/m1-stage-types";
import type { EvaluationArtifact, ExplainArtifact } from "../../api/m3-types";
import capture from "./fixtures/nhanes-predict.json";
import { ResultsCards, ResultsTapestry } from "./index";
import { cvComparison, heldOutTable } from "./model";

// ── the capture ──────────────────────────────────────────────────────────────

type Json = Record<string, unknown>;
interface Snapshot {
  view: Json & { id: string; state: Json; decisions: Json[]; stages: Record<string, Json> };
  quest: Json;
  triage: Json & { items: Json[] };
  stages: Record<string, Json & { artifact: unknown }>;
}
interface Step {
  posted: Json;
  status: number;
  response: Json;
}
const C = capture as unknown as { sealed: Snapshot; opened: Snapshot; chain: Step[]; explain_posted: { posted: Json } };
const PID = C.sealed.view.id;
const fitOf = (s: Snapshot) => s.stages.fit!.artifact as FitArtifact;
const explainOf = (s: Snapshot) => s.stages.explain!.artifact as ExplainArtifact;
const evaluationOf = (s: Snapshot) => s.stages.evaluation!.artifact as EvaluationArtifact;

// ── reading numbers back off the screen ──────────────────────────────────────

/** Every number printed in `text`: "−1,080 (1,050 to 1,110)" → [−1080, 1050, 1110]. */
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

interface Call {
  method: string;
  path: string;
  body: unknown;
}

function serve(start: "sealed" | "opened", patch?: (s: Snapshot) => Snapshot) {
  const calls: Call[] = [];
  let now = patch ? patch(C[start]) : C[start];
  let view = now.view;
  const chain = [...C.chain];
  const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const path = url.replace(/^.*\/api\/projects\/[^/?]+/, "").replace(/\?.*$/, "");
      const body = init?.body ? (JSON.parse(String(init.body)) as unknown) : null;
      calls.push({ method, path, body });
      if (method === "POST" && path === "/decisions") {
        const kind = (body as Json).kind;
        if (kind === "set_explain") return json(200, view);
        const step = chain.shift();
        if (!step) return json(500, { detail: "the capture holds no more answers" });
        expect(body).toEqual(step.posted);
        if (step.status !== 200) return json(step.status, step.response);
        view = { ...view, state: { ...view.state, ...(step.response.state as Json) }, decisions: [...view.decisions, step.response.record as Json] };
        // The opening's answer: from here the server serves what it served after the opening.
        if (view.state.seal_opened) now = { ...C.opened, view };
        return json(200, view);
      }
      if (path === "") return json(200, view);
      if (path === "/quest") return json(200, now.quest);
      if (path === "/triage") return json(200, now.triage);
      const stage = /^\/stages\/(\w+)$/.exec(path)?.[1];
      if (stage && now.stages[stage]) return json(200, now.stages[stage]);
      return json(404, { detail: `not in the capture: ${method} ${path}` });
    }),
  );
  return { calls, left: () => chain.length };
}

function mount(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// ── the capture itself ───────────────────────────────────────────────────────

describe("the capture", () => {
  it("is a light NHANES Predict walk with the linear model and ridge, sealed and then opened", () => {
    expect(fitOf(C.sealed).models.map((m) => m.family)).toEqual(["linear", "ridge"]);
    expect(fitOf(C.sealed).holdout_sealed).toBe(true);
    expect(fitOf(C.sealed).models.every((m) => m.holdout === null && m.holdout_detail === null)).toBe(true);
    expect(fitOf(C.opened).holdout_sealed).toBe(false);
    expect(fitOf(C.opened).final_model).toBe("linear");
    expect(C.opened.view.state.seal_opened).toBeTruthy();
  });
});

// ── the tapestry: every printed number is a served field ─────────────────────

describe("the cross-validated comparison", () => {
  it("prints every family's score on every metric with its interval, the baseline and BBC-CV, as served", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const table = within(await screen.findByTestId("results-cv")).getByRole("table");
    const fit = fitOf(C.opened);
    const labels = Object.entries(fit.metric_labels).filter(([k]) => fit.models.some((m) => m.cv[k]));
    expect(labels.length).toBeGreaterThan(1); // every metric, not the primary alone
    for (const m of fit.models) {
      for (const [k, label] of labels) {
        const s = m.cv[k]!;
        expectShows(cell(table, m.label, label), [s.estimate ?? s.mean, s.ci_low, s.ci_high], `${m.label} ${label}`);
      }
      const v = m.versus_baseline!;
      expectShows(cell(table, m.label, "Against"), [v.gain, v.ci_low, v.ci_high], `${m.label} against the baseline`);
    }
    const base = fit.models[0]!.baseline;
    expectShows(cell(table, base.label[0]!.toUpperCase() + base.label.slice(1), fit.metric_labels[base.metric]!), [base.value], "the baseline");
    const sel = fit.selection!;
    const best = fit.models.find((m) => m.family === sel.best)!.label;
    expectShows(cell(table, `${best}, corrected`, fit.metric_labels[sel.metric]!), [sel.corrected, sel.corrected_low, sel.corrected_high], "BBC-CV");
    for (const [k, label] of labels) {
      if (k === sel.metric || !sel.extras[k]) continue;
      const e = sel.extras[k]!;
      expectShows(cell(table, `${best}, corrected`, label), [e.corrected, e.corrected_low, e.corrected_high], `BBC-CV ${label}`);
    }
  });
});

describe("the held-out rows", () => {
  it("print, once opened, each held-out score with its interval beside the baseline's and the verdict, as served", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const table = within(await screen.findByTestId("results-held-out")).getByRole("table");
    const fit = fitOf(C.opened);
    const final = fit.models.find((m) => m.family === "linear")!;
    const detail = final.holdout_detail!;
    for (const k of ["rmse", "mae", "r2"]) {
      const label = fit.metric_labels[k]!;
      const iv = detail.intervals[k]!;
      expectShows(cell(table, final.label, label), [iv.estimate, iv.ci_low, iv.ci_high], `held-out ${label}`);
      // The interval's estimate is the held-out score (to float precision: they are computed apart).
      expect(iv.estimate).toBeCloseTo(final.holdout![k]!, 12);
      const base = detail.baseline!;
      expectShows(cell(table, base.label[0]!.toUpperCase() + base.label.slice(1), label), [base.scores[k]], `baseline ${label}`);
    }
    const v = detail.versus_baseline!;
    expect(cell(table, final.label, "Against")).toMatch(/^(Better|Not shown to be better|Worse)/);
    expectShows(cell(table, final.label, "Against"), [v.gain, v.ci_low, v.ci_high], "the held-out verdict");
    // Its rule, with the floor the engine interpolated, is the table's own footnote.
    expect(screen.getByTestId("results-held-out").textContent).toContain(v.tolerance_basis.replace(/`/g, ""));
  });

  it("stay sealed before the opening, even where the artifact were to hold held-out numbers", async () => {
    const opened = fitOf(C.opened);
    const leaky = (s: Snapshot): Snapshot => ({
      ...s,
      stages: {
        ...s.stages,
        fit: { ...s.stages.fit!, artifact: { ...fitOf(s), models: fitOf(s).models.map((m, i) => ({ ...m, holdout: opened.models[i]!.holdout, holdout_detail: opened.models[i]!.holdout_detail })) } },
      },
    });
    serve("sealed", leaky);
    mount(<ResultsTapestry pid={PID} />);
    const held = await screen.findByTestId("results-held-out");
    expect(within(held).queryByRole("table")).toBeNull();
    expect(held.textContent).toContain(`The ${fitOf(C.sealed).n_holdout.toLocaleString("en-US")} held-out rows are sealed`);
    expect(screen.getByTestId("results-calibration").textContent).toContain("drawn once they are opened");
    // The explanations wait for their own question (the explain stage is blocked on it).
    expect(C.sealed.view.stages.explain!.missing).toContain("explain");
    expect(screen.getByTestId("results-explain").textContent).toContain("No explanation is asked for yet.");
    // No held-out score of either family is printed anywhere in the tapestry.
    const printed = numbersIn(screen.getByTestId("results-tapestry").textContent ?? "");
    for (const m of opened.models) for (const v of Object.values(m.holdout ?? {})) if (v !== null && Math.abs(v) > 1) expect(printed.some((p) => shows(p, v)), `held-out ${v}`).toBe(false);
    // The pure rule both ways: no table while sealed, the table once opened.
    expect(heldOutTable(opened, "sealed")).toBeNull();
    expect(heldOutTable(opened, "opened")).not.toBeNull();
  });
});

describe("calibration", () => {
  it("draws the final model's held-out calibration through its adapter, every number as served", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const cal = fitOf(C.opened).models.find((m) => m.family === "linear")!.holdout_detail!.calibration!;
    const line = within(await screen.findByTestId("results-calibration")).getByTestId("calibration-line");
    // "Predicted e on average against o observed (calibration intercept i, 95% interval lo to hi; 0 …)
    //  Outcomes moved s times … (calibration slope, 95% interval lo to hi; 1 …). n rows, …"
    // The view's fixed reference words ("0 when they agree", "below 1 when …") and the served level
    // ("95% interval") are not calibration numbers.
    const text = (line.textContent ?? "").replace(/[01] when they agree|below 1 when/g, "").replace(/\d+(\.\d+)?% interval/g, "interval");
    expectShows(text, [cal.expected, cal.observed, cal.intercept.estimate, cal.intercept.ci_low, cal.intercept.ci_high, cal.slope.estimate, cal.slope.ci_low, cal.slope.ci_high, cal.n], "calibration");
  });
});

describe("what drives each model", () => {
  it("lists each explained family's SHAP importance as served, rank by rank", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const tables = await screen.findAllByTestId("results-importance");
    const explained = explainOf(C.opened).families.filter((f) => f.explained);
    expect(tables.map((t) => t.getAttribute("data-family"))).toEqual(explained.map((f) => f.family));
    explained.forEach((f, i) => {
      const table = within(tables[i]!).getByRole("table");
      const rows = [...table.querySelectorAll("tbody tr")].map((r) => r.querySelector("th")!.textContent);
      expect(rows).toEqual([...f.importance].sort((a, b) => a.rank - b.rank).map((x) => x.input));
      for (const x of f.importance) {
        expect(cell(table, x.input, "Rank")).toBe(String(x.rank));
        expectShows(cell(table, x.input, "Mean |SHAP|"), [x.mean_abs], `${f.label} ${x.input}`);
        if (x.in_top !== null && x.in_top !== undefined) expect(cell(table, x.input, "Among the top")).toBe(String(x.in_top));
      }
    });
  });

  it("draws each input's curves at the served grid values, broken only where a segment lacks rows", async () => {
    serve("opened");
    const { container } = mount(<ResultsTapestry pid={PID} />);
    const curves = await screen.findAllByTestId("results-curve");
    const art = explainOf(C.opened);
    expect(curves.map((c) => c.getAttribute("data-input"))).toEqual(art.curves.map((c) => c.input));
    art.curves.forEach((ic, i) => {
      const fig = curves[i]!;
      for (const c of ic.curves.filter((k) => k.drawn)) {
        // A run is a stretch of consecutive segments with enough rows whose edges both carry a value
        // (derived from the served mask, not from the adapter). In this capture every segment with
        // too few rows sits in an upper tail, where the engine already serves the far edge as null;
        // a break between two served values is pinned by construction in curveAdapters.test.tsx.
        let runs = 0;
        let inRun = false;
        for (let k = 0; k < ic.grid.length - 1; k++) {
          const ok = !!ic.supported[k] && (c.values[k] ?? null) !== null && (c.values[k + 1] ?? null) !== null;
          if (ok && !inRun) runs++;
          inRun = ok;
        }
        expect(fig.querySelectorAll(`[data-line="${c.family}"] [data-mark="line"]`).length, `${ic.input} ${c.family}`).toBe(runs);
      }
      // The table alternative prints every served value at its grid value.
      const table = within(fig).getByTestId("table-alternative");
      const drawn = ic.curves.filter((k) => k.drawn);
      const body = [...table.querySelectorAll("tbody tr")];
      expect(body).toHaveLength(ic.grid.length);
      ic.grid.forEach((z, g) => {
        const cells = [...body[g]!.children].map((x) => x.textContent ?? "");
        expectShows(cells[0], [z], `${ic.input} grid`);
        drawn.forEach((c, j) => {
          const v = c.values[g];
          if (v === null || v === undefined) expect(cells[j + 1]).toBe("—");
          else expectShows(cells[j + 1], [v], `${ic.input} ${c.family} at ${z}`);
        });
      });
    });
    expect(container.querySelectorAll('[data-testid="results-curve-not-drawn"]').length).toBe(
      art.curves.reduce((n, ic) => n + ic.curves.filter((c) => !c.drawn && c.reason).length, 0),
    );
  });
});

describe("the interpretable model's cost", () => {
  it("is evaluation's own sentence", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const text = String((evaluationOf(C.opened).interpretable as Json).text);
    expect((await screen.findByTestId("results-interpretable")).textContent).toContain(text.replace(/`/g, ""));
  });
});

// ── the card column ──────────────────────────────────────────────────────────

describe("the triage of the open noticings", () => {
  it("lists each noticing with the engine's disposition and reason, and confirms them as the engine's own exit does", async () => {
    const { calls } = serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-triage");
    const items = C.sealed.triage.items as { id: string; summary: string; label: string; reason: string }[];
    expect([...card.querySelectorAll("[data-item]")].map((x) => x.getAttribute("data-item"))).toEqual(items.map((i) => i.id));
    for (const i of items) {
      const li = card.querySelector(`[data-item="${i.id}"]`)!;
      expect(li.textContent).toContain(i.summary.replace(/`/g, ""));
      expect(li.textContent).toContain(i.label);
      expect(li.textContent).toContain(i.reason.replace(/`/g, ""));
    }
    // "Confirm all" posts what the opening's refusal offers as "Keep the recommended dispositions".
    const keep = C.chain.flatMap((s) => ((s.response.error as Json | undefined)?.exits as { label: string; decision: Json | null }[] | undefined) ?? []).find((x) => x.label === "Keep the recommended dispositions")!;
    await userEvent.click(within(card).getByRole("button", { name: `Confirm all ${items.length}` }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual(keep.decision);
  });
});

describe("the triage once settled", () => {
  it("says what each noticing was recorded as beside its recommendation, and that the rows no longer wait", async () => {
    serve("opened");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-triage");
    const t = C.opened.triage as { passed: boolean; items: { id: string; label: string; recommended: string; recorded: string | null }[] };
    expect(t.passed).toBe(true);
    expect(within(card).getByTestId("triage-settled")).toBeTruthy();
    expect(within(card).queryByRole("button")).toBeNull();
    for (const i of t.items) {
      const li = card.querySelector(`[data-item="${i.id}"]`)!;
      // In this walk each noticing the engine recommended acting on was recorded as a limitation
      // through the opening's exits.
      expect([i.recommended, i.recorded]).toEqual(["act_on_it", "could_bias"]);
      expect(li.textContent).toContain(`Recommended: ${i.label} · recorded as a limitation`);
    }
  });
});

describe("the opening's chain of refusals", () => {
  it("is walked press by press: each exit records, then the opening is tried again, until the rows open", async () => {
    const server = serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    const user = userEvent.setup();
    await user.click(within(card).getByRole("button", { name: "Open the held-out rows" }));
    const codes: string[] = [];
    // Each refusal as served: its message, and its exits as buttons (those with a decision) or as
    // plain words (those without). The walk presses the exit the real walk took.
    for (let guard = 0; guard < 40 && server.left() > 0; guard++) {
      const alert = await screen.findByTestId("results-open-refused");
      const code = alert.getAttribute("data-code")!;
      codes.push(code);
      const served = C.chain.filter((s) => s.status !== 200)[codes.length - 1]!.response.error as { message: string; exits: { label: string; decision: Json | null }[] };
      expect(alert.textContent).toContain(served.message.replace(/`/g, ""));
      const buttons = within(alert).queryAllByRole("button").map((b) => b.textContent);
      expect(buttons).toEqual(served.exits.filter((x) => x.decision).map((x) => x.label.replace(/`/g, "")));
      expect(within(alert).queryAllByTestId("results-open-where").map((x) => x.textContent)).toEqual(served.exits.filter((x) => !x.decision).map((x) => x.label.replace(/`/g, "")));
      // The exit the real walk took next is the next posted body after this refusal.
      const at = C.chain.findIndex((s, i) => s.status !== 200 && C.chain.slice(0, i + 1).filter((t) => t.status !== 200).length === codes.length);
      const next = C.chain[at + 1]!.posted;
      const exit = served.exits.find((x) => JSON.stringify(x.decision) === JSON.stringify(next))!;
      expect(exit, "the walk's next post is one of the served exits").toBeTruthy();
      await user.click(within(alert).getByRole("button", { name: exit.label.replace(/`/g, "") }));
      await waitFor(() => expect(screen.queryByText("Recording…") ?? screen.queryByText("Opening…")).toBeNull());
    }
    expect(server.left()).toBe(0);
    // The engine's order: the final model first, then the noticings (W1_PART2_PLAN §2 names the
    // other order; the code is the truth).
    expect(codes[0]).toBe("final_model_needed");
    expect(codes.slice(1).every((c) => c === "triage_open")).toBe(true);
    const posted = server.calls.filter((c) => c.method === "POST").map((c) => c.body);
    expect(posted).toEqual(C.chain.map((s) => s.posted));
    // Opened: the record's line names the final model, and no refusal is left on screen.
    const done = await screen.findByTestId("results-opened");
    expect(done.textContent).toContain("with Linear model as the final model");
    expect(screen.queryByTestId("results-open-refused")).toBeNull();
  });
});

describe("the explain offer, and the Results declarations", () => {
  it("records the explanations with the engine's defaults, as the real walk did", async () => {
    const { calls } = serve("sealed");
    mount(<ResultsCards pid={PID} />);
    await userEvent.click(await screen.findByTestId("explain-offer"));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual(C.explain_posted.posted);
  });

  it("gives the shrinkage question a line of its own until its card is served, never a blank", async () => {
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const line = await screen.findByTestId("results-line");
    expect(line.getAttribute("data-key")).toBe("set_updating");
    expect(line.textContent).toBe("Decide: the shrinkage question");
  });
});

// ── the pure tables ──────────────────────────────────────────────────────────

describe("the tables' rules", () => {
  const rowOf = (t: ReturnType<typeof cvComparison>, key: string) => {
    const r = t.rows.find((x) => x.kind === "row" && x.key === key);
    if (!r || r.kind !== "row") throw new Error(`no row ${key}`);
    return r;
  };

  it("put the final model first once declared, and every metric the fit scored in its columns", () => {
    const t = cvComparison(fitOf(C.opened));
    expect(t.rows[0]!.kind === "row" && t.rows[0]!.key).toBe("linear");
    expect(t.columns.map((c) => c.key).slice(0, -1).sort()).toEqual(Object.keys(fitOf(C.opened).models[0]!.cv).filter((k) => k in fitOf(C.opened).metric_labels).sort());
  });

  it("carry each served field exactly, below the precision the screen prints", () => {
    const fit = fitOf(C.opened);
    const cv = cvComparison(fit);
    for (const m of fit.models) {
      for (const [k, s] of Object.entries(m.cv)) expect(rowOf(cv, m.family).cells[k]).toEqual({ kind: "estimate", est: s.estimate, lo: s.ci_low, hi: s.ci_high });
    }
    expect(rowOf(cv, "baseline").cells).toEqual({ [fit.models[0]!.baseline.metric]: { kind: "estimate", est: fit.models[0]!.baseline.value, lo: null, hi: null } });
    expect(rowOf(cv, "corrected").cells[fit.selection!.metric]).toEqual({ kind: "estimate", est: fit.selection!.corrected, lo: fit.selection!.corrected_low, hi: fit.selection!.corrected_high });
    const held = heldOutTable(fit, "opened")!;
    for (const m of fit.models) {
      for (const [k, iv] of Object.entries(m.holdout_detail!.intervals)) expect(rowOf(held, m.family).cells[k]).toEqual({ kind: "estimate", est: iv.estimate, lo: iv.ci_low, hi: iv.ci_high });
    }
    const base = fit.models[0]!.holdout_detail!.baseline!;
    for (const [k, v] of Object.entries(base.scores)) expect(rowOf(held, "baseline").cells[k]).toEqual({ kind: "estimate", est: v, lo: null, hi: null });
  });

  it("render nothing until the project view is served", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    const { container } = mount(
      <>
        <ResultsCards pid={PID} />
        <ResultsTapestry pid={PID} />
      </>,
    );
    await act(async () => {});
    expect(container.innerHTML).toBe("");
  });
});
