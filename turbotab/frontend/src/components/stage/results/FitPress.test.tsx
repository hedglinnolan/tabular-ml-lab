/**
 * Fit, pressed and refused, on Classic's Results (FOUNDATION §7). Under Estimate and Describe the
 * engine refuses a press of Fit while the open noticings' triage before the lock is not confirmed
 * (core/fit_press.py `triage_refusal`, code `triage_open`), and Classic draws no triage, so the
 * refusal's ways forward are the way past: one that is a decision is a press that records it as the
 * server served it, after which Fit can be pressed again; one that names no decision is said in
 * words after the refusal's. A refusal with no way forward, and a failure, are said as before:
 * their words alone. Nothing is pressed for the person.
 *
 * The refusals are the served shape (turbotab/server/tests/test_fit_press_holds_api.py): the words
 * built by the engine's rule from a noticing the triage serves (quest/results/fixtures/
 * nhanes-predict.json, `binary_text__gender`), "1 noticing is still open: <summary>. Recommended:
 * <label, lower case>.", and the two ways forward; a blocking noticing's, whose one way forward
 * records nothing ("Decide: <summary>").
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FitArtifact } from "../../../api/m1-stage-types";
import type { Decision, FitLock, ProjectView, Refusal, StageStatus } from "../../../api/schema";
import { Results, type ResultsData } from "./Results";

const PID = "p1";

/** core/fit_press.py LOCK_FIRST: the withheld fit's caption under Estimate. */
const LOCK_FIRST =
  "No estimate is shown until Fit is pressed: pressing it locks the analysis plan, so the plan is declared before any estimate is seen.";
/** core/fit_press.py PRESS_WAITS_FOR_TRIAGE. */
const LEAD = "Fit waits: pressing it locks the analysis plan, and the open noticings are triaged before the plan is locked.";
/** The triage's served noticing (nhanes-predict.json): its summary and its recommended disposition's label. */
const SUMMARY = "`gender` is two-level text (`female`, `male`); which level counts as 1 is not asked yet.";
const LABEL = "Act on it";
const said = (summary: string) => summary.trim().replace(/\.$/, "");

/** The triage's Confirm all, as the refusal serves it. */
const CONFIRM: Decision = { kind: "confirm_sweep", stage: "models", sweep: "noticings", lines: [] };
const TRIAGE_IN_MODELS = "Triage the open noticings in Models";
const KEEP = "Keep the recommended dispositions";

const OPEN: Refusal = {
  error: {
    code: "triage_open",
    message: `${LEAD} 1 noticing is still open: ${said(SUMMARY)}. Recommended: ${LABEL.toLowerCase()}. Triage it in Models first, so it is decided before any estimate is seen.`,
    exits: [
      { label: TRIAGE_IN_MODELS, decision: null },
      { label: KEEP, decision: CONFIRM },
    ],
  },
};

/** A blocking noticing (a summary written here): decided on its own, so no way forward records. */
const BLOCKING_SUMMARY = "`glucose` takes one value in every row.";
const BLOCKING: Refusal = {
  error: {
    code: "triage_open",
    message: `${LEAD} 1 noticing must be resolved first: ${said(BLOCKING_SUMMARY)}.`,
    exits: [{ label: `Decide: ${BLOCKING_SUMMARY}`, decision: null }],
  },
};

/** A refusal with no way forward (written here). */
const ORDINARY: Refusal = { error: { code: "fit_not_yet", message: "There is nothing to fit yet.", exits: [] } };

const PRESSED: FitLock = {
  purpose: "inference",
  locks: true,
  locked: true,
  at: "2026-11-04T10:00:00.000Z",
  sha256: "0".repeat(64),
  pressed: true,
  opened: true,
  held: false,
  estimate_seconds: null,
  reason: null,
} as unknown as FitLock;

const fitStatus: StageStatus = {
  stage: "fit",
  status: "fresh",
  key: "k-fit",
  fresh: true,
  missing: [],
  error: null,
  job_id: null,
  progress: null,
  updated_at: "2026-11-04T10:00:00.000Z",
  cancelled: false,
  held: null,
  refusal: null,
};

const VIEW = {
  summary: { id: PID, name: "NHANES", n_rows: 5000, n_cols: 40 },
  state: { purpose: "inference", target: "glucose", roles: null, models: ["linear"] },
  decisions: [],
  interview: [],
  stages: { fit: fitStatus },
} as unknown as ProjectView;

/** The fit computed and withheld for Fit, as GET /stages/fit serves it before the press. */
const DATA: ResultsData = {
  fit: { artifact: { withheld_for: "fit", withheld: LOCK_FIRST, models: [] } as unknown as FitArtifact, veil: "fresh" },
  shelf: null,
  design: null,
  split: null,
  substitution: { artifact: null, veil: "fresh" },
};

interface Post {
  path: string;
  body: unknown;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** POST /fit refused with `held` until a decision is recorded, then taken; POST /decisions answered
 *  with `decided` (a refusal: 409). Every POST is kept with its body. */
function serve(held: Refusal | { status: number; body: unknown }, decided: Refusal | null = null): Post[] {
  const posts: Post[] = [];
  let recorded = false;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST") posts.push({ path: url, body: typeof init?.body === "string" ? JSON.parse(init.body) : null });
      if (method === "POST" && url === `/api/projects/${PID}/fit`) {
        if (recorded) return json(PRESSED);
        return "error" in held ? json(held, 409) : json(held.body, held.status);
      }
      if (method === "POST" && url === `/api/projects/${PID}/decisions`) {
        if (decided) return json(decided, 409);
        recorded = true;
        return json(VIEW);
      }
      return json({});
    }),
  );
  return posts;
}

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <Results pid={PID} view={VIEW} data={DATA} />
    </QueryClientProvider>,
  );
  return within(screen.getByTestId("fit-press"));
}

const where = (posts: Post[]) => posts.map((p) => [p.path.split("/").pop(), p.body]);
const buttons = (scope: ReturnType<typeof mount>) => scope.getAllByRole("button").map((b) => b.textContent);
/** Text as Rich draws it: backticked data becomes a chip, its backticks gone. */
const shown = (text: string) => text.replaceAll("`", "");

afterEach(() => vi.unstubAllGlobals());

describe("a press of Fit the open noticings' triage holds", () => {
  it("offers the way forward that is a decision as a press that records it as served, and Fit again", async () => {
    const posts = serve(OPEN);
    const press = mount();
    const user = userEvent.setup();
    await user.click(press.getByRole("button", { name: "Fit" }));
    const alert = await press.findByRole("alert");
    // The words, then the way forward that names no decision, said: Classic draws no triage, and
    // the shell's Models does (the refusal's words name it).
    expect(alert.querySelector("p")?.textContent).toBe(`${shown(OPEN.error.message)} ${TRIAGE_IN_MODELS}.`);
    expect(buttons(press)).toEqual(["Fit", KEEP]);
    expect(within(alert).getByRole("button", { name: KEEP })).toBeEnabled();
    expect(press.queryByRole("button", { name: TRIAGE_IN_MODELS })).toBeNull();
    expect(where(posts)).toEqual([["fit", null]]);

    // Keep the recommended dispositions: the decision the server served, posted as it came; the
    // refusal goes with it and Fit is not pressed for the person.
    await user.click(press.getByRole("button", { name: KEEP }));
    await waitFor(() => expect(press.queryByRole("alert")).toBeNull());
    expect(where(posts)).toEqual([
      ["fit", null],
      ["decisions", CONFIRM],
    ]);
    expect(buttons(press)).toEqual(["Fit"]);

    // Fit, pressed again, is taken.
    await user.click(press.getByRole("button", { name: "Fit" }));
    await waitFor(() =>
      expect(where(posts)).toEqual([
        ["fit", null],
        ["decisions", CONFIRM],
        ["fit", null],
      ]),
    );
    await waitFor(() => expect(press.getByRole("button", { name: "Fit" })).toBeEnabled());
    expect(press.queryByRole("alert")).toBeNull();
  });

  it("says the server's refusal of that decision, and keeps the press's refusal", async () => {
    const gone: Refusal = { error: { code: "gate_passed", message: "The triage is already confirmed.", exits: [] } };
    const posts = serve(OPEN, gone);
    const press = mount();
    const user = userEvent.setup();
    await user.click(press.getByRole("button", { name: "Fit" }));
    await user.click(await press.findByRole("button", { name: KEEP }));
    const note = await press.findByTestId("refusal");
    expect(note).toHaveTextContent(gone.error.message);
    expect(press.getByTestId("fit-press-refusal")).toHaveTextContent(shown(OPEN.error.message));
    expect(where(posts)).toEqual([
      ["fit", null],
      ["decisions", CONFIRM],
    ]);
  });

  it("says a blocking noticing's refusal with its way forward in words, and offers no press", async () => {
    const posts = serve(BLOCKING);
    const press = mount();
    const user = userEvent.setup();
    await user.click(press.getByRole("button", { name: "Fit" }));
    const alert = await press.findByRole("alert");
    expect(alert.textContent).toBe(`${shown(BLOCKING.error.message)} ${shown(BLOCKING.error.exits[0]!.label)}`);
    expect(buttons(press)).toEqual(["Fit"]);
    expect(where(posts)).toEqual([["fit", null]]);
  });
});

describe("a press of Fit refused or failed otherwise", () => {
  it("says a refusal with no way forward in its words alone, as before", async () => {
    serve(ORDINARY);
    const press = mount();
    const user = userEvent.setup();
    await user.click(press.getByRole("button", { name: "Fit" }));
    const alert = await press.findByRole("alert");
    expect(alert.textContent).toBe(ORDINARY.error.message);
    expect(buttons(press)).toEqual(["Fit"]);
  });

  it("says a failure in its words alone, as before", async () => {
    serve({ status: 500, body: { detail: "The server stopped." } });
    const press = mount();
    const user = userEvent.setup();
    await user.click(press.getByRole("button", { name: "Fit" }));
    const alert = await press.findByRole("alert");
    expect(alert.textContent).toBe("The server stopped.");
    expect(buttons(press)).toEqual(["Fit"]);
  });
});
