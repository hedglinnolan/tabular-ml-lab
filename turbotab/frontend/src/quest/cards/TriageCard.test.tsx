/**
 * The triage of the open noticings at the gate (GET /triage; FOUNDATION §7), under Estimate (G05):
 * the gate is the plan's lock, in Models (core/sweep.py `gate_stage`), so "Confirm all" records the
 * served stage's sweep, and the card speaks of the plan being fixed, never of held-out rows. Each
 * noticing the engine says owes a limitation sentence (`TriageItem.limitation`: it could bias the
 * estimate and nothing was done about it) says so while the disposition that owes it stands.
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
    expect(within(card).queryByRole("button")).toBeNull();
    for (const i of ESTIMATE.items) expect(itemEl(card, i.id).textContent).toContain(`${i.label} · recorded`);
    expect(heading(card)).toBe(`${ESTIMATE.items.length} noticings before the plan is fixed`);
  });

  it("says, once the plan is fixed, that every noticing was settled before it was", () => {
    serve();
    const card = mount(locked(confirmed(ESTIMATE)));
    expect(heading(card)).toBe(`${ESTIMATE.items.length} noticings, settled before the plan was fixed`);
    expect(within(card).queryByRole("button")).toBeNull();
    expect(card.textContent).not.toMatch(/held-out/i);
  });

  it("claims no settling, once the plan is fixed, for a noticing with no disposition standing", () => {
    // A confirmation is refused once the gate has passed (core/sweep.py `gate_passed`), so a
    // noticing with no recorded disposition is open after the lock.
    serve();
    const card = mount(locked(triageWith((i) => (i.id === OWED.id ? i : { ...i, recorded: i.recommended }))));
    expect(heading(card)).toBe(`${ESTIMATE.items.length} noticings, 1 open after the plan was fixed`);
    expect(within(card).queryByRole("button")).toBeNull();
    expect(card.textContent).not.toMatch(/held-out/i);
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
