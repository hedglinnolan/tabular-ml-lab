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
import { useState, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitArtifact, SplitArtifact } from "../../api/m1-stage-types";
import type { EvaluationArtifact, ExplainArtifact, VersusBaseline } from "../../api/m3-types";
import type { ProjectView } from "../../api/schema";
import PAYLOADS from "../cards/predict/payloads.fixture.json";
import { FootSlot } from "../contracts";
import { SealCard } from "./cards";
import capture from "./fixtures/nhanes-predict.json";
import { ResultsCards, ResultsTapestry } from "./index";
import { currentOpening, cvComparison, heldOutTable, intervalClause, verdictCell } from "./model";

/** The models' defaults and choices, as test_w1_declaration_cards.py holds them to decisions.py. */
const DEFAULTS = PAYLOADS.defaults;
const CHOICES = PAYLOADS.choices;

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
const splitOf = (s: Snapshot) => s.stages.split!.artifact as SplitArtifact;

// The engine's note once the served scores are not the opening's (turbotab/core/seal.py
// `reported_result`): its head is the capture's own note, its tail the engine's for each case.
const HEAD = fitOf(C.opened).at_opening!.note.replace(/\.$/, "");
const CHANGED_NOTE = `${HEAD}; the held-out scores shown now are from a fit changed after it, so they are post-seal and not an independent test.`;
const REDRAWN_NOTE = `${HEAD}; the held-out scores shown now are from rows drawn again after it, so they are not an independent test.`;
const RESEALED_NOTE = `${HEAD}; the held-out rows were drawn again after it, and stay withheld until they are opened in turn, as a test that is not independent.`;

/** The verdict's word for each served verdict, as its rule names it (`tolerance_basis`), written
 *  here rather than read from the module, so a cell showing another verdict's word fails. */
const WORD: Record<VersusBaseline["verdict"], string> = { better: "Better", no_better: "Not shown to be better", worse: "Worse" };

/** A snapshot whose split artifact is patched: the basis fields given, and any top-level fields. */
const withSplit =
  (basis: Json, top: Json = {}) =>
  (s: Snapshot): Snapshot => ({
    ...s,
    stages: { ...s.stages, split: { ...s.stages.split!, artifact: { ...splitOf(s), basis: { ...splitOf(s).basis, ...basis }, ...top } } },
  });

/** A snapshot whose served fit is patched. */
const withFit =
  (patch: (f: FitArtifact) => Partial<FitArtifact>) =>
  (s: Snapshot): Snapshot => ({
    ...s,
    stages: { ...s.stages, fit: { ...s.stages.fit!, artifact: { ...fitOf(s), ...patch(fitOf(s)) } } },
  });

/** The served fit once its scores are not the opening's: the server then serves the note as the
 *  final model's too (server/service.py `_served_fit`). */
const notCurrent = (note: string, after: boolean) =>
  withFit((f) => ({ changed_after_seal: after, at_opening: { ...f.at_opening!, current: false, note }, final_note: note }));

// ── a re-seal, and a second opening (audit WP16, RO-05) ──────────────────────
// After the opening a changed split is refused (`seal_opened`) with a Reseal exit; the re-seal
// sets `seal_opened` to null (core/decisions.py), so the seal is closed again over the rows drawn
// next; opening them records a second `open_seal`, which the fit's final model comes from
// (selection.py `declared_family` → seal.py `opening`), while `at_opening` stays the first's.

/** The capture's opening record (#22). */
const FIRST = (C.opened.view.decisions as unknown as ProjectView["decisions"]).find((d) => d.decision.kind === "open_seal")!;
const record = (seq: number, decision: Json): Json => ({ ...FIRST, id: `rec-${seq}`, seq, sentence: `Record #${seq}.`, post_seal: false, decision });
const RESEAL = record(FIRST.seq + 1, { kind: "reseal", reason: null, target: "glucose" });
const SECOND = record(FIRST.seq + 2, { ...FIRST.decision });

/** Sealed again by a re-seal after the opening: the record holds the opening and the re-seal, and
 *  the fit, sealed, serves the first opening's scores with the engine's note for this case. */
const resealed = (): Snapshot => ({
  ...C.sealed,
  view: { ...C.sealed.view, state: { ...C.sealed.view.state, seal_opened: null }, decisions: [...C.opened.view.decisions, RESEAL] },
  stages: {
    ...C.sealed.stages,
    fit: { ...C.sealed.stages.fit!, artifact: { ...fitOf(C.sealed), at_opening: { ...fitOf(C.opened).at_opening!, current: false, note: RESEALED_NOTE } } },
  },
});

/** A snapshot whose quest log has the Results line `key` answered. */
const withAnswered =
  (key: string) =>
  (s: Snapshot): Snapshot => ({
    ...s,
    quest: {
      ...s.quest,
      stages: (s.quest.stages as { key: string; lines: Json[] }[]).map((st) =>
        st.key === "results" ? { ...st, lines: st.lines.map((l) => (l.key === key ? { ...l, status: "answered", decision_id: `rec-${key}` } : l)) } : st,
      ),
    },
  });

/** A snapshot whose triage reads as confirmed (each noticing as recommended). */
const triageConfirmed = (s: Snapshot): Snapshot => ({
  ...s,
  triage: { ...s.triage, answered: true, confirmable: false, items: s.triage.items.map((i) => ({ ...i, recorded: i.recommended })) },
});

/** Sealed, with what the opening waits for done: the shrinkage answered and the triage confirmed. */
const readyToOpen = (s: Snapshot): Snapshot => triageConfirmed(withAnswered("set_updating")(s));

/** Opened a second time over the rows drawn again. */
const reopened = (s: Snapshot): Snapshot => notCurrent(REDRAWN_NOTE, false)({ ...s, view: { ...s.view, decisions: [...C.opened.view.decisions, RESEAL, SECOND] } });

// The bases a split can serve, in the engine's words for each (turbotab/core/seal.py `_basis`).
const UNDETERMINED: Json = {
  state: "undetermined",
  column: null,
  label: "undetermined",
  sentence:
    "Held out by row, because whether a unit can appear in more than one row was answered as not known. This is not a verified clean split: treat held-out scores as exploratory.",
  exploratory: true,
  source: "grain",
};
const ABANDONED: Json = {
  state: "abandoned",
  column: "SEQN",
  label: "repetition found but grouping abandoned",
  sentence:
    "Each row was said to be a different unit, but `SEQN` repeats (`8,000` values over `9,000` rows); the held-out rows were drawn by row as answered. Treat held-out scores as exploratory.",
  exploratory: true,
  source: "stated",
  n_units: 8000,
};
const GROUPED: Json = {
  state: "grouped",
  column: "SEQN",
  label: "grouped by `SEQN`",
  sentence: "Held out by `SEQN`: each of the `9,000` units sits wholly on one side, so no unit is both trained on and scored.",
  exploratory: false,
  source: "grain",
  n_units: 9000,
};
// A clean basis whose draw is still exploratory: time was asked for and no column named it.
const UNTIMED = {
  drawn: false,
  time_column: null,
  boundary: null,
  n_units: null,
  n_undated: 0,
  sentence:
    "Later outcomes were said to be predicted from earlier ones, but no column was named as time, so the held-out rows were drawn at random rather than by time: treat held-out scores as exploratory.",
  n_held_rows: null,
  n_held_earlier: null,
  earlier: [],
};
const unticked = (s: string) => s.replace(/`/g, "");
/** How many times `part` is said in `text`. */
const times = (text: string | null | undefined, part: string) => (text ?? "").split(part).length - 1;

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
      expect(cell(table, m.label, "Against"), `${m.label}: the word for a served "${v.verdict}"`).toMatch(new RegExp(`^${WORD[v.verdict]}: `));
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
    for (const m of fit.models) {
      const w = m.holdout_detail!.versus_baseline!;
      expect(cell(table, m.label, "Against"), `${m.label}: the word for a served "${w.verdict}"`).toMatch(new RegExp(`^${WORD[w.verdict]}: `));
      expectShows(cell(table, m.label, "Against"), [w.gain, w.ci_low, w.ci_high], `${m.label}'s held-out verdict`);
    }
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
    expect(heldOutTable(opened, "sealed", splitOf(C.opened))).toBeNull();
    expect(heldOutTable(opened, "opened", splitOf(C.opened))).not.toBeNull();
  });
});

describe("the held-out exhibit's title", () => {
  const caption = async () => (within(await screen.findByTestId("results-held-out")).getByRole("table").querySelector("caption")?.textContent ?? "");
  const n = fitOf(C.opened).n_holdout.toLocaleString("en-US");
  const changed = CHANGED_NOTE;
  const redrawn = REDRAWN_NOTE;

  it("says the rows were scored once, by no choice seen, while the served scores are the opening's", async () => {
    expect(fitOf(C.opened).at_opening!.current).toBe(true);
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    expect(await caption()).toContain(`Scored once, on ${n} rows no choice has seen`);
  });

  it("says the scores come from a fit changed after the opening, once one is", async () => {
    serve("opened", notCurrent(changed, true));
    mount(<ResultsTapestry pid={PID} />);
    const text = await caption();
    expect(screen.getByTestId("results-tapestry").getAttribute("data-seal-phase")).toBe("post_seal");
    expect(text).not.toMatch(/scored once|no choice has seen/i);
    expect(text).toContain("From a fit changed after the opening");
    expect(text).toContain("not an independent test");
    expect(screen.getByTestId("results-held-out").textContent).toContain(unticked(changed));
  });

  it("says the scores come from rows drawn again after the first opening, once they are", async () => {
    serve("opened", notCurrent(redrawn, false));
    mount(<ResultsTapestry pid={PID} />);
    const text = await caption();
    expect(screen.getByTestId("results-tapestry").getAttribute("data-seal-phase")).toBe("opened");
    expect(text).not.toMatch(/scored once|no choice has seen/i);
    expect(text).toContain(`On ${n} rows drawn again after the first opening`);
    expect(text).toContain("not an independent test");
  });
});

describe("the held-out exhibit's notes", () => {
  const notes = async () => {
    await within(await screen.findByTestId("results-held-out")).findByRole("table");
    return screen.getByTestId("results-held-out").textContent ?? "";
  };

  for (const [name, note, patch] of [
    ["from a fit changed after the opening", CHANGED_NOTE, notCurrent(CHANGED_NOTE, true)],
    ["from rows drawn again and opened in turn", REDRAWN_NOTE, reopened],
  ] as const) {
    it(`say the engine's note once when the scores are ${name}`, async () => {
      serve("opened", patch);
      mount(<ResultsTapestry pid={PID} />);
      expect(times(await notes(), unticked(note))).toBe(1);
    });
  }

  it("say both the final model's note and the opening's, once each, while the scores are the opening's", async () => {
    const fit = fitOf(C.opened);
    expect(fit.final_note).not.toBe(fit.at_opening!.note);
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const text = await notes();
    expect(times(text, unticked(fit.final_note!))).toBe(1);
    expect(times(text, unticked(fit.at_opening!.note))).toBe(1);
  });
});

describe("an exploratory split", () => {
  // Every place a held-out score or the opening is said: the held-out table, the held-out
  // calibration, the opened line.
  const said = () =>
    [screen.getByTestId("results-held-out"), screen.getByTestId("results-calibration"), screen.getByTestId("results-opened")].map((x) => x.textContent ?? "");

  for (const [name, patch, sentences] of [
    ["an undetermined basis", withSplit(UNDETERMINED), [UNDETERMINED.sentence as string]],
    ["an abandoned grouping", withSplit(ABANDONED), [ABANDONED.sentence as string]],
    ["a clean basis drawn at random where time was asked for", withSplit(GROUPED, { exploratory: true, chronology: UNTIMED }), [UNTIMED.sentence]],
  ] as const) {
    it(`labels the opened held-out scores exploratory, in the split's own words, under ${name}`, async () => {
      serve("opened", patch);
      mount(
        <>
          <ResultsCards pid={PID} />
          <ResultsTapestry pid={PID} />
        </>,
      );
      await within(await screen.findByTestId("results-held-out")).findByRole("table");
      await screen.findByTestId("results-opened");
      await waitFor(() => expect(screen.getByTestId("results-held-out").textContent).toContain(unticked(sentences[0]!)));
      const [held, calibration, opened] = said();
      expect(within(screen.getByTestId("results-held-out")).getByRole("table").querySelector("caption")!.textContent).toMatch(/exploratory/);
      expect(calibration).toMatch(/exploratory/);
      for (const s of sentences) {
        expect(held).toContain(unticked(s));
        expect(opened).toContain(unticked(s));
      }
      expect(opened).toMatch(/These scores are exploratory/);
    });
  }

  for (const [name, patch] of [
    ["the capture's one row per unit", undefined],
    ["a grouped basis", withSplit(GROUPED)],
  ] as const) {
    it(`gives no exploratory label under ${name}`, async () => {
      expect(splitOf(C.opened).basis.state).toBe("one_row_per_unit");
      serve("opened", patch);
      const { container } = mount(
        <>
          <ResultsCards pid={PID} />
          <ResultsTapestry pid={PID} />
        </>,
      );
      await within(await screen.findByTestId("results-held-out")).findByRole("table");
      await screen.findByTestId("results-opened");
      // Until the split is read its basis is unknown, which is labeled (seal/phase.ts); once read, nothing.
      await waitFor(() => expect(container.textContent).not.toMatch(/exploratory/i));
      expect(said().every((t) => t.length > 0)).toBe(true);
    });
  }

  it("says on the seal card, before the opening, why the scores will carry the label", async () => {
    serve("sealed", withSplit(UNDETERMINED));
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    await waitFor(() => expect(card.textContent).toContain(unticked(UNDETERMINED.sentence as string)));
    expect(card.textContent).toContain("In the Results the held-out scores are then marked exploratory.");
  });

  it("says nothing of a label on the seal card under a clean basis", async () => {
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    await waitFor(() => expect(card.textContent).not.toMatch(/exploratory/i));
    expect(card.textContent).toContain("Open the held-out rows");
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

describe("a curve the palette cannot color", () => {
  it("names each family the engine drew past the palette's last color, beside the curve it is missing from", async () => {
    // By construction: the capture's first input with four more families drawn, each a copy of
    // the linear model's served values under its own name (the engine caps neither count).
    const more = [
      ["elastic_net", "Elastic net"],
      ["huber", "Huber regression"],
      ["random_forest", "Random forest"],
      ["boosted_trees", "Boosted trees"],
    ];
    const six = (s: Snapshot): Snapshot => {
      const art = explainOf(s);
      const [first, ...rest] = art.curves;
      const linear = first!.curves.find((c) => c.family === "linear")!;
      const curves = [{ ...first!, curves: [...first!.curves, ...more.map(([family, label]) => ({ ...linear, family: family!, label: label! }))] }, ...rest];
      return { ...s, stages: { ...s.stages, explain: { ...s.stages.explain!, artifact: { ...art, curves } } } };
    };
    serve("opened", six);
    mount(<ResultsTapestry pid={PID} />);
    const [fig, ...others] = await screen.findAllByTestId("results-curve");
    expect(fig!.querySelectorAll("[data-line]")).toHaveLength(5);
    expect(fig!.querySelector('[data-line="boosted_trees"]')).toBeNull();
    expect(within(fig!).getByTestId("results-curve-past-palette").textContent).toBe(
      "Not drawn here: Boosted trees. Each curve takes the color of its family's place among the families explained, and there are 5 colors.",
    );
    // Where every drawn family has a color, nothing is said.
    for (const f of others) expect(within(f).queryByTestId("results-curve-past-palette")).toBeNull();
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

describe("the triage's heading", () => {
  const heading = async () => within(await screen.findByTestId("results-triage")).getByRole("heading").textContent;
  const n = C.sealed.triage.items.length;

  it("says the noticings come before the held-out rows open, while they are sealed", async () => {
    expect(C.sealed.triage.passed).toBe(false);
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    expect(await heading()).toBe(`${n} noticings before the held-out rows open`);
  });

  it("says, once the rows are opened, that every noticing was settled before they were", async () => {
    expect(C.opened.triage.passed).toBe(true);
    expect((C.opened.triage.items as { recorded: string | null }[]).every((i) => i.recorded !== null)).toBe(true);
    serve("opened");
    mount(<ResultsCards pid={PID} />);
    expect(await heading()).toBe(`${C.opened.triage.items.length} noticings, settled before the held-out rows were opened`);
    expect(screen.getByTestId("triage-settled")).toBeTruthy();
  });

  it("claims no settling, once the rows are opened, for a noticing with no disposition standing", async () => {
    // By construction: a noticing open after the opening (one a later change raised, or one whose
    // recorded disposition no longer stands). A confirmation is refused once the gate has passed
    // (core/sweep.py `gate_passed`), so it has no recorded disposition.
    const raised = (s: Snapshot): Snapshot => ({
      ...s,
      triage: { ...s.triage, answered: false, items: s.triage.items.map((i, k) => (k === 0 ? { ...i, recorded: null } : i)) },
    });
    serve("opened", raised);
    mount(<ResultsCards pid={PID} />);
    expect(await heading()).toBe(`${C.opened.triage.items.length} noticings, 1 open after the held-out rows were opened`);
    expect(screen.queryByTestId("triage-settled")).toBeNull();
    // Nor is "Confirm all" offered: the engine refuses it once the gate has passed.
    expect(within(screen.getByTestId("results-triage")).queryByRole("button")).toBeNull();
  });
});

describe("the opening's chain of refusals", () => {
  it("is walked press by press: each exit records, then the opening is tried again, until the rows open", async () => {
    // The press waits for the shrinkage answer and the triage's confirmation (the foot says so,
    // below); the real walk pressed before either, and the server answered as the chain holds. So
    // the page is shown them done, and the server still answers as the real one did: the engine's
    // refusals stand behind the page's reading, a noticing to act on included.
    const server = serve("sealed", readyToOpen);
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

describe("the seal card beside a fit that is not current", () => {
  const card = (fresh: boolean) => {
    serve("sealed");
    return mount(<SealCard pid={PID} view={C.sealed.view as unknown as ProjectView} fit={fitOf(C.sealed)} fresh={fresh} split={splitOf(C.sealed)} />);
  };

  it("says the rows can be opened once the fit is current, beside a button that waits for it", () => {
    card(false);
    const button = screen.getByRole("button", { name: "Open the held-out rows" });
    expect(button).toHaveProperty("disabled", true);
    expect(screen.getByTestId("results-open").textContent).toContain("The fit shown is not current; the rows can be opened once it is.");
    expect(screen.getByTestId("results-open").textContent).not.toContain("Nothing is opened until you press it.");
  });

  it("says nothing opens until it is pressed, beside a button that can be, once the fit is current", () => {
    card(true);
    expect(screen.getByRole("button", { name: "Open the held-out rows" })).toHaveProperty("disabled", false);
    expect(screen.getByTestId("results-open").textContent).toContain("Nothing is opened until you press it.");
    expect(screen.getByTestId("results-open").textContent).not.toContain("not current");
  });
});

/** What the first opening's card and line claim, which a re-seal makes untrue. */
const FIRST_CLAIMS = ["Opens once", "no choice has seen", "It happens once", "fixed in the record"];

describe("the seal card after a re-seal", () => {
  it("says the engine's note, and none of the first opening's claims, over the rows drawn again", async () => {
    serve("sealed", resealed);
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    await waitFor(() => expect(card.textContent).toContain(unticked(RESEALED_NOTE)));
    for (const claim of FIRST_CLAIMS) expect(card.textContent, claim).not.toContain(claim);
    expect(within(card).getByRole("button", { name: "Open the held-out rows" })).toBeTruthy();
  });

  it("keeps the first opening's words before the first opening, where no earlier one is served", async () => {
    expect(fitOf(C.sealed).at_opening).toBeNull();
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    for (const claim of FIRST_CLAIMS) expect(card.textContent, claim).toContain(claim);
    expect(card.textContent).not.toContain("At the opening");
  });
});

describe("the opened line", () => {
  const line = async () => (await screen.findByTestId("results-opened")).textContent ?? "";

  it("names the opening and says the scores are fixed in the record, while they are the opening's", async () => {
    serve("opened");
    mount(<ResultsCards pid={PID} />);
    const text = await line();
    expect(text).toContain(`The held-out rows were opened once (#${FIRST.seq}), with Linear model as the final model: their scores are fixed in the record.`);
    expect(text).toContain(unticked(fitOf(C.opened).final_note!));
  });

  it("names the second opening, not the first, and says nothing of once, after the rows are drawn again and opened", async () => {
    serve("opened", reopened);
    mount(<ResultsCards pid={PID} />);
    const text = await line();
    expect(text).toContain(`The held-out rows were opened (#${SECOND.seq as number}), with Linear model as the final model.`);
    expect(text).not.toContain(`(#${FIRST.seq})`);
    expect(text).not.toMatch(/\bonce\b/);
    expect(text).not.toContain("fixed in the record");
    expect(times(text, unticked(REDRAWN_NOTE))).toBe(1);
  });

  it("says nothing of once after a change refits the opened fit, and says the engine's note once", async () => {
    serve("opened", notCurrent(CHANGED_NOTE, true));
    mount(<ResultsCards pid={PID} />);
    const text = await line();
    expect(text).toContain(`The held-out rows were opened (#${FIRST.seq}), with Linear model as the final model.`);
    expect(text).not.toMatch(/\bonce\b/);
    expect(text).not.toContain("fixed in the record");
    expect(times(text, unticked(CHANGED_NOTE))).toBe(1);
  });
});

describe("the post-seal band after a second opening", () => {
  it("names the opening the changes came after, the current one (seal.py `post_seal_changes`)", async () => {
    // After a second opening the engine's note is the rows-drawn-again one, a change or not.
    const changedAfterSecond = (s: Snapshot): Snapshot => withFit(() => ({ changed_after_seal: true }))(reopened(s));
    serve("opened", changedAfterSecond);
    mount(<ResultsTapestry pid={PID} />);
    const band = await screen.findByTestId("post-seal");
    expect(band.textContent).toContain(`(#${SECOND.seq as number})`);
    expect(band.textContent).not.toContain(`(#${FIRST.seq})`);
  });
});

describe("the current opening", () => {
  // The engine's rule (turbotab/core/seal.py `opening`): among the records not reverted, the latest
  // opening or re-seal about the current outcome (or naming none); an opening opens, a re-seal
  // closes. Each case's answer is read off its records by that rule.
  const view = (...extra: Json[]) => ({ ...C.opened.view, decisions: [...C.opened.view.decisions, ...extra] }) as unknown as ProjectView;
  const seqOf = (v: ProjectView) => currentOpening(v)?.seq ?? null;

  it("is the capture's one opening", () => expect(seqOf(view())).toBe(FIRST.seq));
  it("is none after a re-seal", () => expect(seqOf(view(RESEAL))).toBeNull());
  it("is the second opening once the rows drawn again are opened", () => expect(seqOf(view(RESEAL, SECOND))).toBe(SECOND.seq));
  it("is the first again when the re-seal is reverted", () =>
    expect(seqOf(view(RESEAL, record(FIRST.seq + 2, { kind: "revert", decision_id: RESEAL.id as string })))).toBe(FIRST.seq));
  it("is the re-seal's again when the revert is itself reverted", () => {
    const revert = record(FIRST.seq + 2, { kind: "revert", decision_id: RESEAL.id as string });
    expect(seqOf(view(RESEAL, revert, record(FIRST.seq + 3, { kind: "revert", decision_id: revert.id as string })))).toBeNull();
  });
  it("ignores a re-seal of another outcome's seal", () => expect(seqOf(view(record(FIRST.seq + 1, { kind: "reseal", reason: null, target: "hba1c" })))).toBe(FIRST.seq));
});

describe("the Results declarations, through W1-F's cards", () => {
  // The orchestrator's ruling (CARRIED-INT-1): W1-F's DECLARATION_CARDS are the only renderers of
  // set_explain and set_updating, in Results as everywhere; Results draws no offer or line of its own.

  it("records the explanations with the engine's defaults through W1-F's card, as the real walk did", async () => {
    const { calls } = serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("declaration-set_explain");
    expect(screen.queryByTestId("explain-offer")).toBeNull();
    // The engine's default curves, with every other field left as the card starts it.
    await userEvent.click(await within(card).findByTestId(`option-curves:${DEFAULTS.set_explain.curves}`));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    // The real walk posted the kind alone and the server filled in its model's defaults
    // (payloads.fixture.json `defaults`, held to decisions.py by test_w1_declaration_cards.py): the
    // card posts the same decision with each default spelled out.
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ ...C.explain_posted.posted, ...DEFAULTS.set_explain });
  });

  it("draws the shrinkage question as W1-F's card, never a line of Results' own", async () => {
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("declaration-set_updating");
    expect(within(card).getByRole("heading")).toHaveTextContent("The shrinkage question");
    expect(within(card).getAllByRole("option").map((o) => o.getAttribute("data-key"))).toEqual(
      CHOICES.set_updating.method.map((v) => `method:${v}`),
    );
    expect(screen.queryByTestId("results-line")).toBeNull();
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
    const held = heldOutTable(fit, "opened", splitOf(C.opened))!;
    for (const m of fit.models) {
      for (const [k, iv] of Object.entries(m.holdout_detail!.intervals)) expect(rowOf(held, m.family).cells[k]).toEqual({ kind: "estimate", est: iv.estimate, lo: iv.ci_low, hi: iv.ci_high });
    }
    const base = fit.models[0]!.holdout_detail!.baseline!;
    for (const [k, v] of Object.entries(base.scores)) expect(rowOf(held, "baseline").cells[k]).toEqual({ kind: "estimate", est: v, lo: null, hi: null });
  });

  it("give each served verdict its own word, with the gain as served, a negative one included", () => {
    const v = (verdict: VersusBaseline["verdict"], gain: number, lo: number, hi: number): VersusBaseline => ({
      verdict,
      metric: "r2",
      gain,
      tolerance: 0.02,
      tolerance_basis: "Better only when the gain's 95% interval lies above zero and the gain is at least 0.01 (a convention): a gain above 0.020.",
      se: 0.01,
      ci_low: lo,
      ci_high: hi,
      df: null,
      level: 0.95,
    });
    const labels = { r2: "R²" };
    expect(verdictCell(v("better", 0.135, 0.114, 0.156), labels)).toEqual({ kind: "text", text: "Better: a gain in R² of 0.135 (0.114 to 0.156)" });
    expect(verdictCell(v("no_better", 0.004, -0.012, 0.02), labels)).toEqual({ kind: "text", text: "Not shown to be better: a gain in R² of 0.004 (−0.012 to 0.02)" });
    expect(verdictCell(v("worse", -0.031, -0.05, -0.012), labels)).toEqual({ kind: "text", text: "Worse: a gain in R² of −0.031 (−0.05 to −0.012)" });
    // No gain served: the word alone, and no verdict at all where none is served.
    expect(verdictCell({ ...v("no_better", 0, 0, 0), gain: null, ci_low: null, ci_high: null }, labels)).toEqual({ kind: "text", text: "Not shown to be better" });
    expect(verdictCell(null, labels)).toBeNull();
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

// ── the design review's repairs (W1 part 2) ──────────────────────────────────

describe("the comparison's caption on intervals (P2)", () => {
  const caption = async () => within(await screen.findByTestId("results-cv")).getByRole("table").querySelector("caption")?.textContent ?? "";
  const fit = fitOf(C.opened);
  const n = fit.n_train.toLocaleString("en-US");
  const base = fit.models[0]!.baseline.label;
  const bare = <T extends object>(s: T) => ({ ...s, ci_low: null, ci_high: null });
  /** The fit with no cross-validated interval served at all, and no BBC-CV row (the mock's fit; a
   *  fit from before WP9). */
  const noIntervals = withFit((f) => ({
    models: f.models.map((m) => ({ ...m, cv: Object.fromEntries(Object.entries(m.cv).map(([k, s]) => [k, bare(s)])) })),
    selection: null,
  }));

  it("says each score has its interval but the baseline's, where every model's printed score has one", async () => {
    // By the capture: each family's and BBC-CV's printed scores carry their intervals; the
    // no-predictor baseline is served with none.
    expect(fit.models.every((m) => Object.values(m.cv).every((s) => s.ci_low !== null && s.ci_high !== null))).toBe(true);
    expect(fit.selection!.corrected_low).not.toBeNull();
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    expect(await caption()).toBe(`Models compared. Cross-validated on ${n} training rows: each score with its interval, except ${base}`);
  });

  it("claims no interval where no printed score has one", async () => {
    serve("opened", noIntervals);
    mount(<ResultsTapestry pid={PID} />);
    expect(await caption()).toBe(`Models compared. Cross-validated on ${n} training rows`);
  });

  it("names a metric served with no interval (as macro-F1's), beside the baseline", async () => {
    const noMae = withFit((f) => ({ models: f.models.map((m) => ({ ...m, cv: { ...m.cv, mae: bare(m.cv.mae!) } })) }));
    serve("opened", noMae);
    mount(<ResultsTapestry pid={PID} />);
    expect(await caption()).toBe(
      `Models compared. Cross-validated on ${n} training rows: each score with its interval, except ${base} and ${fit.metric_labels.mae}`,
    );
  });

  it("is a rule that holds both ways, on rows built by hand", () => {
    const m = [
      { key: "a", label: "A" },
      { key: "b", label: "B" },
    ];
    const est = (lo: number | null) => ({ kind: "estimate" as const, est: 1, lo, hi: lo === null ? null : 2 });
    const row = (key: string, a: number | null, b: number | null) => ({ kind: "row" as const, key, label: key.toUpperCase(), cells: { a: est(a), b: est(b) } });
    expect(intervalClause([row("x", 0, 0), row("y", 0, 0)], m)).toBe(": each score with its interval");
    expect(intervalClause([row("x", null, null), row("y", null, null)], m)).toBe("");
    expect(intervalClause([row("x", 0, 0), row("y", null, null)], m, { y: "the baseline" })).toBe(": each score with its interval, except the baseline");
    expect(intervalClause([row("x", 0, null), row("y", 0, null)], m)).toBe(": each score with its interval, except B");
    // One score without its interval, neither a whole row nor a whole column: nothing is claimed.
    expect(intervalClause([row("x", 0, null), row("y", 0, 0)], m)).toBe("");
    // A cell with nothing served is not a printed score.
    expect(intervalClause([{ ...row("x", 0, 0), cells: { a: est(0), b: null } }], m)).toBe(": each score with its interval");
  });

  it("starts the held-out caption with a capital, as the comparison's", async () => {
    serve("opened");
    mount(<ResultsTapestry pid={PID} />);
    const held = within(await screen.findByTestId("results-held-out")).getByRole("table").querySelector("caption")!.textContent!;
    expect(held).toMatch(/^Held-out rows\. Scored once, /);
    expect(held).toContain(`except ${fit.models[0]!.holdout_detail!.baseline!.label}`);
  });
});

describe("the opening at the card's foot (P5)", () => {
  const press = () => screen.findByTestId("results-open-press");
  const hint = () => screen.getByTestId("results-open-hint").textContent;
  const results = (C.sealed.quest.stages as { key: string; lines: { key: string; name: string }[] }[]).find((s) => s.key === "results")!;
  const shrinkage = results.lines.find((l) => l.key === "set_updating")!;
  const n = C.sealed.triage.items.length;

  it("waits, in the button's own words, for the Results question before it", async () => {
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    await screen.findByTestId("results-open-hint");
    await waitFor(() => expect(hint()).toBe(`To open the held-out rows, first answer ${shrinkage.name}.`));
    expect(await press()).toHaveTextContent("Open the held-out rows");
    expect(await press()).toBeDisabled();
  });

  it("then for the triage's confirmation", async () => {
    serve("sealed", withAnswered("set_updating"));
    mount(<ResultsCards pid={PID} />);
    await screen.findByTestId("results-open-hint");
    await waitFor(() => expect(hint()).toBe(`To open the held-out rows, first confirm the ${n} noticings above.`));
    expect(await press()).toBeDisabled();
  });

  it("can be pressed once both are done", async () => {
    serve("sealed", readyToOpen);
    mount(<ResultsCards pid={PID} />);
    await screen.findByTestId("results-open-hint");
    await waitFor(() => expect(hint()).toBe("Nothing is opened until you press it."));
    expect(await press()).toBeEnabled();
  });

  it("is drawn in the shell's foot, the card column's one primary action, and nowhere else", async () => {
    serve("sealed", readyToOpen);
    function Shell() {
      const [slot, setSlot] = useState<HTMLDivElement | null>(null);
      return (
        <FootSlot.Provider value={slot}>
          <ResultsCards pid={PID} />
          <div ref={setSlot} data-testid="foot" />
        </FootSlot.Provider>
      );
    }
    mount(<Shell />);
    const card = await screen.findByTestId("results-open");
    await waitFor(() => expect(within(screen.getByTestId("foot")).getByTestId("results-open-press")).toBeEnabled());
    expect(within(card).queryByTestId("results-open-press")).toBeNull();
    expect(screen.getAllByRole("button", { name: "Open the held-out rows" })).toHaveLength(1);
  });
});

describe("the opening's card before the opening (P6)", () => {
  it("says what will happen, and wears no recorded glyph", async () => {
    serve("sealed");
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    expect(card.textContent).toContain("Opens once");
    expect(card.textContent).not.toContain("Opened once");
    expect(within(card).queryByRole("img")).toBeNull();
  });

  it("draws the basis's glyph where the basis is not a verified clean one, as its label says", async () => {
    serve("sealed", withSplit(UNDETERMINED));
    mount(<ResultsCards pid={PID} />);
    const card = await screen.findByTestId("results-open");
    await waitFor(() => expect(within(card).getByRole("img")).toHaveAccessibleName("Sealed by row: basis undetermined"));
  });
});

describe("the triage's heading with no rows held out (CARRIED-G-1)", () => {
  // By construction: the sealed capture with nothing held out (the fit's `n_holdout` 0). Under
  // Predict the triage is served whatever the holdout, and its gate never passes, since the engine
  // refuses to open nothing (core/seal.py, `nothing_sealed`).
  const cvOnly = withFit(() => ({ n_holdout: 0 }));
  const heading = async () => within(await screen.findByTestId("results-triage")).getByRole("heading").textContent;
  const n = C.sealed.triage.items.length;

  it("names no held-out rows, and says what confirming records", async () => {
    serve("sealed", cvOnly);
    mount(<ResultsCards pid={PID} />);
    expect(await heading()).toBe(`${n} noticings, each recorded as listed once confirmed`);
    expect(await screen.findByTestId("seal-none")).toBeInTheDocument();
    expect(screen.getByTestId("results-triage").textContent).not.toMatch(/held-out/);
  });

  it("says each disposition is recorded once confirmed", async () => {
    serve("sealed", (s) => triageConfirmed(cvOnly(s)));
    mount(<ResultsCards pid={PID} />);
    expect(await heading()).toBe(`${n} noticings, each with its disposition recorded`);
  });
});
