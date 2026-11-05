/**
 * The triage of the open noticings at the gate (GET /triage; FOUNDATION §7), under Estimate (G05):
 * the gate is the plan's lock, in Models (core/sweep.py `gate_stage`), so "Confirm all" records the
 * served stage's sweep, and the card speaks of the plan being fixed, never of held-out rows. Each
 * noticing the engine says owes a limitation sentence (`TriageItem.limitation`: it could bias the
 * estimate and nothing was done about it) says so while the disposition that owes it stands, where
 * its reason does not already say it. Once the gate has passed, a triage is settled only where no
 * noticing is open and none recorded "act on it" is still listed (one acted on drops out). A
 * triage with no noticing draws nothing.
 *
 * The triage is built by construction in the shape the engine serves it (no Estimate triage is
 * captured in the fixtures); every expected word and value is read off it. Predict's card, before
 * the held-out rows open, is pinned on the real server's capture in results/results.test.tsx.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Triage, TriageItem } from "../../api/schema";
import { TriageCard } from "./TriageCard";

const PID = "estimate-triage";

/** One open finding at the gate, as `sweep.triage` serves it. */
const item = (id: string, patch: Partial<TriageItem>): TriageItem => ({
  id,
  line: `finding:${id}`,
  stage: "data",
  summary: `The noticing \`${id}\`.`,
  severity: "warning",
  columns: [],
  question: null,
  recommended: "no_change",
  label: "Doesn't change your numbers here",
  reason: "A note about the data: no number rests on it.",
  blocker: false,
  recorded: null,
  family: "finding",
  band: null,
  measure: null,
  calibrated: null,
  limitation: false,
  done: null,
  ...patch,
});

/** Could bias the estimate, nothing done about it: the engine owes a limitation sentence. */
const OWED = item("implausible_bmi", {
  stage: "whos_in",
  columns: ["bmi"],
  recommended: "could_bias",
  label: "Could bias the estimate",
  reason: "It concerns `bmi`, which the analysis reads; left as it is, it could bias the estimate.",
  limitation: true,
});
/** Could bias the estimate, with a sensitivity analysis declared on its column: none is owed. */
const DONE = item("implausible_age", {
  stage: "whos_in",
  columns: ["age"],
  recommended: "could_bias",
  label: "Could bias the estimate",
  reason: "It concerns `age`, which the analysis reads; left as it is, it could bias the estimate.",
  done: 'The sensitivity analysis "Adults only" is declared beside the primary.',
});
const ACT = item("sas_zeros", {
  recommended: "act_on_it",
  label: "Act on it",
  reason: "It is decided in Your data, where its repair is chosen.",
});
const NOTE = item("note__SEQN", {});

/** The triage before the plan's lock under Estimate, as `sweep.triage` serves it. */
const ESTIMATE: Triage = {
  version: 1,
  stage: "models",
  gate: "gate:open-noticings-before-lock",
  items: [OWED, DONE, ACT, NOTE],
  blockers: 0,
  confirmable: true,
  answered: false,
  confirmed_by: null,
  changed: [],
  passed: false,
  rows: [{ family: "finding", name: "Other things noticed", items: [OWED, DONE, ACT, NOTE].map((i) => i.id), count: 4 }],
};

/** The triage with each item `set`, and the triage's fields `patch`ed. */
const triageWith = (set: (i: TriageItem) => TriageItem, patch: Partial<Triage> = {}): Triage => ({
  ...ESTIMATE,
  ...patch,
  items: ESTIMATE.items.map(set),
});
/** The triage once `sas_zeros` is acted on, its repair chosen in Your data: an act-on-it noticing
 *  whose decision is answered drops out (core/sweep.py `triage`: only an open or covered line is
 *  listed; core/materiality.py `in_triage`), so none is left to act on. */
const ACTED: Triage = {
  ...ESTIMATE,
  items: ESTIMATE.items.filter((i) => i.id !== ACT.id),
  rows: [{ ...ESTIMATE.rows[0]!, items: ESTIMATE.rows[0]!.items.filter((id) => id !== ACT.id), count: ESTIMATE.items.length - 1 }],
};
/** How many of a triage's noticings stand recorded "act on it". */
const toAct = (t: Triage) => t.items.filter((i) => i.recorded === "act_on_it").length;

/** A measured noticing that owes a limitation sentence, as `sweep._noticing_item` serves it: its
 *  reason is `materiality.recommend`'s, which says the sentence is drafted (the NHANES Estimate
 *  journey's day-to-day noise in sugar, its words as served). */
const MEASURED = item("diet-day-to-day-variance", {
  line: "noticing:diet-day-to-day-variance",
  stage: "models",
  summary: "Part of `sugar`'s spread is day-to-day noise",
  severity: "noticing",
  columns: ["sugar"],
  recommended: "could_bias",
  label: "Could bias the estimate (not measurable here)",
  reason:
    "How much of `sugar` is day-to-day noise is not measurable here: each person has one recall day and no second measure, so a slope could be weakened by it without a sign. Nothing here can check it, so a limitation sentence is drafted.",
  family: "K3",
  band: 1,
  measure: "attenuation 1 − λ: not measurable (one measure per person)",
  calibrated: false,
  limitation: true,
});

/** Confirmed before the lock: each noticing as recommended. */
const confirmed = (t: Triage): Triage => ({ ...t, answered: true, confirmable: false, items: t.items.map((i) => ({ ...i, recorded: i.recommended })) });
/** The plan fixed after that (Fit pressed): the gate has passed. */
const locked = (t: Triage): Triage => ({ ...t, passed: true, confirmable: false });

interface Call {
  method: string;
  path: string;
  body: unknown;
}

/** A server that answers a decision with an empty view, and records what was posted. */
function serve() {
  const calls: Call[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const path = url.replace(/^.*\/api\/projects\/[^/?]+/, "").replace(/\?.*$/, "");
      calls.push({ method, path, body: init?.body ? (JSON.parse(String(init.body)) as unknown) : null });
      if (method === "POST") return new Response(JSON.stringify({ id: PID, state: {}, decisions: [], stages: {} }), { status: 200, headers: { "Content-Type": "application/json" } });
      return new Promise<Response>(() => {});
    }),
  );
  return calls;
}

function mount(triage: Triage) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  render(
    <QueryClientProvider client={qc}>
      <TriageCard pid={PID} triage={triage} id="t" />
    </QueryClientProvider>,
  );
  return screen.getByTestId("results-triage");
}

const heading = (card: HTMLElement) => within(card).getByRole("heading").textContent;
const itemEl = (card: HTMLElement, id: string) => card.querySelector<HTMLElement>(`[data-item="${CSS.escape(id)}"]`)!;
const unticked = (s: string) => s.replaceAll("`", "");

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the triage before the plan's lock, under Estimate (G05)", () => {
  it("lists each noticing with the engine's disposition and reason, before the plan is fixed", () => {
    serve();
    const card = mount(ESTIMATE);
    const n = ESTIMATE.items.length;
    expect(heading(card)).toBe(`${n} noticings before the plan is fixed`);
    expect([...card.querySelectorAll("[data-item]")].map((x) => x.getAttribute("data-item"))).toEqual(ESTIMATE.items.map((i) => i.id));
    for (const i of ESTIMATE.items) {
      const li = itemEl(card, i.id);
      expect(li.textContent).toContain(unticked(i.summary));
      expect(li.textContent).toContain(i.label);
      expect(li.textContent).toContain(unticked(i.reason));
    }
    // Under Estimate the gate is the plan's lock: nothing on the card names held-out rows.
    expect(card.textContent).not.toMatch(/held-out/i);
  });

  it("records every disposition at the served gate's stage with Confirm all, as the engine's route says", async () => {
    const calls = serve();
    const card = mount(ESTIMATE);
    await userEvent.click(within(card).getByRole("button", { name: `Confirm all ${ESTIMATE.items.length}` }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    // server/routes/projects.py `triage`: {"kind": "confirm_sweep", "stage": <the gate's stage>, "sweep": "noticings"}.
    expect(calls.find((c) => c.method === "POST")).toEqual({
      method: "POST",
      path: "/decisions",
      body: { kind: "confirm_sweep", stage: ESTIMATE.stage, sweep: "noticings", lines: [] },
    });
  });

  it("offers no Confirm all once confirmed, each noticing reading as recorded", () => {
    serve();
    const card = mount(confirmed(ESTIMATE));
    expect(within(card).queryByTestId("triage-confirm")).toBeNull();
    // The one press left is the line recorded "act on it": its change of disposition (below).
    expect(within(card).getAllByRole("button")).toEqual([within(itemEl(card, ACT.id)).getByRole("button")]);
    for (const i of ESTIMATE.items) expect(itemEl(card, i.id).textContent).toContain(`${i.label} · recorded`);
    expect(heading(card)).toBe(`${ESTIMATE.items.length} noticings before the plan is fixed`);
  });

  it("says, once the plan is fixed, that every noticing was settled before it was, none left to act on", () => {
    serve();
    const t = locked(confirmed(ACTED));
    expect(toAct(t)).toBe(0);
    const card = mount(t);
    expect(heading(card)).toBe(`${ACTED.items.length} noticings, settled before the plan was fixed`);
    expect(within(card).queryByRole("button")).toBeNull();
    expect(card.textContent).not.toMatch(/held-out/i);
  });

  it("claims no settling, once the plan is fixed, for a noticing with no disposition standing", () => {
    // A confirmation is refused once the gate has passed (core/sweep.py `gate_passed`), so a
    // noticing with no recorded disposition is open after the lock.
    serve();
    const t = locked({ ...ACTED, items: ACTED.items.map((i) => (i.id === OWED.id ? i : { ...i, recorded: i.recommended })) });
    expect(toAct(t)).toBe(0);
    const card = mount(t);
    expect(heading(card)).toBe(`${ACTED.items.length} noticings, 1 open after the plan was fixed`);
    expect(within(card).queryByRole("button")).toBeNull();
    expect(card.textContent).not.toMatch(/held-out/i);
  });
});

/**
 * A noticing recorded "act on it" and still listed was not acted on: one acted on drops out of the
 * triage (core/sweep.py `triage` lists only an open or covered line, and the triage's record
 * covers no act-on-it line, `cover_noticings`). Under Predict the held-out rows stay sealed while
 * one is listed (core/seal.py: "N marked \"act on it\" is not acted on yet"); under Estimate the
 * plan's lock reads no triage (server/service.py `press_fit`), so the plan can be fixed with one.
 * The triage is in the shape the engine served on the NHANES Estimate journey after the lock: the
 * gate passed, every noticing recorded, `sas_zeros` recorded "act on it".
 */
describe("a noticing marked to act on, not acted on when the gate passed (B1)", () => {
  it("is not called settled once the plan is fixed: the heading counts it still to act on", () => {
    serve();
    const t = locked(confirmed(ESTIMATE));
    expect(t.passed).toBe(true);
    expect(t.items.every((i) => i.recorded !== null)).toBe(true);
    expect(t.items.filter((i) => i.recorded === "act_on_it").map((i) => i.id)).toEqual([ACT.id]);
    const card = mount(t);
    expect(heading(card)).toBe(`${t.items.length} noticings, ${toAct(t)} still to act on after the plan was fixed`);
    expect(heading(card)).not.toMatch(/settled/);
    expect(itemEl(card, ACT.id).textContent).toContain(`${ACT.label} · recorded`);
    expect(within(card).queryByRole("button")).toBeNull();
  });

  it("is counted beside the open ones, each in its own clause", () => {
    serve();
    const t = locked(triageWith((i) => (i.id === OWED.id ? i : { ...i, recorded: i.recommended })));
    expect(toAct(t)).toBe(1);
    const card = mount(t);
    expect(heading(card)).toBe(`${t.items.length} noticings, 1 open and ${toAct(t)} still to act on after the plan was fixed`);
  });

  it("is not called settled under Predict once the held-out rows are open, and the rows are not said to wait no more", () => {
    serve();
    const t: Triage = { ...locked(confirmed(ESTIMATE)), stage: "results", gate: "gate:open-noticings-before-seal" };
    const card = mount(t);
    expect(heading(card)).toBe(`${t.items.length} noticings, ${toAct(t)} still to act on after the held-out rows were opened`);
    expect(within(card).queryByTestId("triage-settled")).toBeNull();
    // Without it, the same triage is settled, and says so.
    cleanup();
    const settled = mount({ ...locked(confirmed(ACTED)), stage: "results", gate: "gate:open-noticings-before-seal" });
    expect(heading(settled)).toBe(`${ACTED.items.length} noticings, settled before the held-out rows were opened`);
    expect(within(settled).getByTestId("triage-settled")).toBeTruthy();
  });
});

/**
 * DE1 (b1-decision-exits; the owner's Q3 and Q8, 2026-11-04): while the gate has not passed, a
 * noticing recorded "act on it" and not acted on holds it, and the gate's refusal offers changing
 * its disposition, "Record as a limitation: …", the decision `{kind: confirm_sweep, stage: <the
 * gate's>, sweep: noticings, lines: [{id: <its line>, key: <its id>, value: could_bias}]}`
 * (core/fit_press.py `triage_refusal`; core/tests/test_fit_press_triage.py pins it). In the shell
 * that way forward opens this card (FOUNDATION §1), so the card offers that same decision on the
 * noticing's own line, and nowhere else.
 */
describe("a noticing recorded to act on, before the gate passes (DE1)", () => {
  const LIMIT = "Record as a limitation";
  /** Confirmed before the gate, as core/sweep.py `triage` serves it: `confirmable` is "items, no
   *  blocker, the gate not passed", whatever is answered, so a confirmed triage stays confirmable
   *  (the file's `confirmed` leaves it false, which no rule of the engine's serves). */
  const kept = (t: Triage): Triage => {
    const c = confirmed(t);
    return { ...c, confirmable: c.items.length > 0 && c.blockers === 0 && !c.passed };
  };
  /** The decision the gate's refusal carries for `i`, written out as core/fit_press.py writes it. */
  const exitDecision = (t: Triage, i: TriageItem) => ({
    kind: "confirm_sweep",
    stage: t.stage,
    sweep: "noticings",
    lines: [{ id: i.line, key: i.id, value: "could_bias" }],
  });

  it("offers Record as a limitation on the line recorded act on it, and on no other", () => {
    serve();
    const t = kept(ESTIMATE);
    expect(t.items.filter((i) => i.recorded === "act_on_it").map((i) => i.id)).toEqual([ACT.id]);
    const card = mount(t);
    expect(within(itemEl(card, ACT.id)).getByRole("button", { name: LIMIT })).toBeEnabled();
    for (const i of [OWED, DONE, NOTE]) expect(within(itemEl(card, i.id)).queryByRole("button"), i.id).toBeNull();
  });

  it("records exactly the decision the gate's refusal carries for that line", async () => {
    const calls = serve();
    const t = kept(ESTIMATE);
    const card = mount(t);
    await userEvent.click(within(itemEl(card, ACT.id)).getByRole("button", { name: LIMIT }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.filter((c) => c.method === "POST")).toEqual([
      { method: "POST", path: "/decisions", body: exitDecision(t, ACT) },
    ]);
    expect(exitDecision(t, ACT)).toEqual({
      kind: "confirm_sweep",
      stage: "models",
      sweep: "noticings",
      lines: [{ id: "finding:sas_zeros", key: "sas_zeros", value: "could_bias" }],
    });
  });

  it("records it at the gate's own stage under Predict, before the held-out rows open", async () => {
    const calls = serve();
    const t: Triage = { ...kept(ESTIMATE), stage: "results", gate: "gate:open-noticings-before-seal" };
    const card = mount(t);
    await userEvent.click(within(itemEl(card, ACT.id)).getByRole("button", { name: LIMIT }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual(exitDecision(t, ACT));
  });

  it("reads as recorded as a limitation once recorded so, with no press left", () => {
    serve();
    // As core/sweep.py `triage` serves it after the decision: recorded could bias, recommended act on it.
    const t = triageWith((i) => ({ ...i, recorded: i.id === ACT.id ? "could_bias" : i.recommended }), { answered: true });
    const card = mount(t);
    expect(itemEl(card, ACT.id).textContent).toContain(`Recommended: ${ACT.label} · recorded as a limitation`);
    expect(within(card).queryByRole("button")).toBeNull();
  });

  it("is offered on no line before the triage is confirmed: a noticing recommended to act on is not recorded so yet", () => {
    serve();
    expect(ESTIMATE.items.find((i) => i.id === ACT.id)!.recorded).toBeNull();
    const card = mount(ESTIMATE);
    expect(within(card).queryByRole("button", { name: LIMIT })).toBeNull();
  });

  it("waits, disabled, while a noticing blocks the triage, as Confirm all does", () => {
    serve();
    const blocker = item("outcome_constant", { blocker: true, severity: "critical" });
    const t: Triage = { ...kept(ESTIMATE), items: [blocker, ...kept(ESTIMATE).items], blockers: 1, confirmable: false, answered: false };
    const card = mount(t);
    expect(within(itemEl(card, ACT.id)).getByRole("button", { name: LIMIT })).toBeDisabled();
  });
});

describe("a triage with no noticing in it", () => {
  it("draws nothing at either gate: no heading, no Confirm all", () => {
    // core/sweep.py `triage` serves items [] where nothing is open: confirmable false, answered true.
    serve();
    for (const stage of ["models", "results"]) {
      const qc = new QueryClient();
      const { container } = render(
        <QueryClientProvider client={qc}>
          <TriageCard pid={PID} triage={{ ...ESTIMATE, stage, items: [], rows: [], confirmable: false, answered: true }} id="t" />
        </QueryClientProvider>,
      );
      expect(container.querySelector("[data-testid='results-triage']"), stage).toBeNull();
      expect(container.textContent, stage).toBe("");
      cleanup();
    }
  });
});

describe("a noticing's limitation sentence (G05; FOUNDATION §7: could bias the estimate, a limitation sentence)", () => {
  const owedIn = (card: HTMLElement) => [...card.querySelectorAll("[data-testid='triage-limitation']")].map((x) => x.closest("[data-item]")!.getAttribute("data-item"));

  it("is said where the engine owes one, and nowhere else: not where something was done, nor beside another disposition", () => {
    serve();
    const card = mount(ESTIMATE);
    expect(owedIn(card)).toEqual(ESTIMATE.items.filter((i) => i.limitation).map((i) => i.id));
    expect(owedIn(card)).toEqual([OWED.id]);
    expect(within(itemEl(card, OWED.id)).getByTestId("triage-limitation").textContent).toMatch(/limitation sentence/);
    for (const i of [DONE, ACT, NOTE]) expect(within(itemEl(card, i.id)).queryByTestId("triage-limitation")).toBeNull();
  });

  it("stands once recorded as the disposition that owes it, and after the plan is fixed", () => {
    serve();
    expect(owedIn(mount(confirmed(ESTIMATE)))).toEqual([OWED.id]);
    cleanup();
    expect(owedIn(mount(locked(confirmed(ESTIMATE))))).toEqual([OWED.id]);
  });

  it("is said for a finding, whose reason does not say it, and not repeated for a measured noticing, whose reason does", () => {
    // A finding's reason (core/sweep.py `recommend`, `_left_as_it_is`) names no limitation
    // sentence; a measured one that owes it (core/materiality.py `recommend`) ends saying it is drafted.
    serve();
    expect(OWED.family).toBe("finding");
    expect(OWED.reason).not.toMatch(/limitation sentence/);
    expect(MEASURED.limitation).toBe(true);
    expect(MEASURED.reason).toMatch(/a limitation sentence is drafted/);
    const card = mount({ ...ESTIMATE, items: [OWED, MEASURED] });
    expect(owedIn(card)).toEqual([OWED.id]);
    const measured = itemEl(card, MEASURED.id);
    expect(measured.textContent).toContain(unticked(MEASURED.reason));
    expect(measured.textContent!.match(/limitation sentence/g)).toHaveLength(1);
  });

  it("is not claimed where the disposition recorded is another: no change, or acting on it", () => {
    // `limitation` is the recommendation's (core/sweep.py `triage`); the ledger owes the sentence
    // only for a noticing left as could bias (core/materiality.py `ledger`, `disposed`).
    serve();
    for (const recorded of ["no_change", "act_on_it"] as const) {
      const card = mount(triageWith((i) => (i.id === OWED.id ? { ...i, recorded } : i)));
      expect(owedIn(card), recorded).toEqual([]);
      cleanup();
    }
  });
});
