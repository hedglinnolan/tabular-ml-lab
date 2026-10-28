/**
 * The Results ask GET /models only for what a sentence reads from it: the table a family declares
 * it gives under inference, which the coefficients' note reads for a family whose fit gives no
 * coefficient table (ridge, the elastic net: `inference: null`). A Results whose every family has
 * its table (the linear model's intervals), or a Results under prediction, reads no declaration,
 * so it asks the server nothing of the families. The paper prototype's walk (e2e/methods-protos)
 * runs the production Results from a fixture with no server, and holds that no request is made.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { FitArtifact, FittedModel } from "../../../api/m1-stage-types";
import type { FamilyInfo } from "../../../api/m3-types";
import type { ProjectView } from "../../../api/schema";
import { Results, type ResultsData } from "./Results";

class NoResize {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const PID = "p1";
const HC3 = "95% confidence intervals from HC3 (heteroskedasticity-robust) standard errors, t(116).";

const model = (family: string, label: string, inference: FittedModel["inference"]) =>
  ({
    family,
    label,
    cv: { r2: { mean: 0.1, sd: 0.02, folds: [0.1, 0.12, 0.08] } },
    holdout: null,
    concerns: [],
    fit_seconds: 1,
    baseline: { metric: "r2", value: 0, label: "the outcome's average" },
    coefficients: [{ feature: "walks_daily", estimate: -5.4, ci_low: inference ? -8.9 : null, ci_high: inference ? -2.7 : null, p: inference ? 0.0003 : null }],
    inference,
  }) as unknown as FittedModel;

const LINEAR = model("linear", "Linear model", { caption: HC3, refused: null } as unknown as FittedModel["inference"]);
const RIDGE = model("ridge", "Ridge", null);

function fit(models: FittedModel[]): FitArtifact {
  return {
    task: "regression",
    primary_metric: "r2",
    metric_labels: { r2: "R²" },
    n_train: 120,
    n_holdout: 0,
    models,
    holdout_sealed: false,
    changed_after_seal: false,
    post_seal_decisions: [],
  } as unknown as FitArtifact;
}

function view(purpose: string): ProjectView {
  return {
    state: { purpose, target: "sbp_mmhg", roles: { walks_daily: "exposure" }, seal_opened: false },
    stages: {},
    decisions: [],
  } as unknown as ProjectView;
}

const data = (artifact: FitArtifact): ResultsData => ({
  fit: { artifact, veil: "fresh" },
  shelf: null,
  design: null,
  split: null,
  substitution: { artifact: null, veil: "fresh" },
});

// The tables these families declare under inference, as GET /models serves them (ridge.py and
// linear.py's ``inference_decl``).
const FAMILIES = [
  { key: "ridge", inference_table: "shrunk_no_intervals" },
  { key: "linear", inference_table: "intervals" },
] as unknown as FamilyInfo[];

/** Every request the Results make, answered as the server would (GET /models: the families). */
function serve(): string[] {
  const asked: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      asked.push(`${init?.method ?? "GET"} ${url}`);
      const body = url.endsWith("/api/models") ? FAMILIES : {};
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }),
  );
  return asked;
}

function mount(purpose: string, models: FittedModel[]) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <Results pid={PID} view={view(purpose)} data={data(fit(models))} />
    </QueryClientProvider>,
  );
  return screen.getByTestId("coefficients").querySelector("p")!;
}

/** Long enough for a request, were one made on mount, to be made and answered. */
const settled = () => new Promise((r) => setTimeout(r, 50));

afterEach(() => vi.unstubAllGlobals());

describe("the Results' reads of the families", () => {
  beforeAll(() => {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver ??= NoResize;
  });

  it("asks nothing of the families when every family has its table", async () => {
    const asked = serve();
    const note = mount("inference", [LINEAR]);
    await settled();
    expect(asked.filter((a) => a.includes("/api/models"))).toEqual([]);
    expect(note.textContent).toBe(`Change in predicted sbp_mmhg per unit of each input, holding the others; ${HC3}`);
  });

  it("asks nothing of the families under prediction, whose note reads no declaration", async () => {
    const asked = serve();
    const note = mount("prediction", [RIDGE]);
    await settled();
    expect(asked.filter((a) => a.includes("/api/models"))).toEqual([]);
    expect(note.textContent).toBe(
      "Under prediction the coefficients describe the fitted model; they are not interpreted as effects, so no intervals are drawn.",
    );
  });

  it("reads the declaration of a family with no table under inference, and says why from it", async () => {
    const asked = serve();
    const note = mount("inference", [RIDGE]);
    await waitFor(() =>
      expect(note.textContent).toBe(
        "Ridge's coefficients are shrunk toward zero by its penalty and have no standard errors, so they describe the fitted model; they are not interpreted as effects, and no intervals are drawn.",
      ),
    );
    expect(asked.filter((a) => a.includes("/api/models"))).toEqual(["GET /api/models"]);
  });
});
