/**
 * The work each stage does, named in the engine's words (the audit's S2): STAGE_LABEL holds every
 * stage's label as the engine's graph declares it (fixtures/stages.json, captured from
 * core/stages `build_graph` by capture_stages.py), so a chip before its JobView arrives and a
 * stage that is not running, whose status serves no label, say what the engine says.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import type { JobView, StageStatus } from "../api/schema";
import { ENGINE_LABELS, SERVED } from "./fixtures/stages";
import { JobChips, STAGE_LABEL, stageLabel } from "./JobChips";

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

afterEach(() => vi.restoreAllMocks());

describe("STAGE_LABEL", () => {
  it("is every stage's label in the engine's words", () => {
    expect(STAGE_LABEL).toEqual(ENGINE_LABELS);
  });

  it("says a stage the engine does not name by its name, in words", () => {
    expect(stageLabel("time_varying")).toBe(ENGINE_LABELS.time_varying);
    expect(stageLabel("no_such_stage")).toBe("no such stage");
  });
});

describe("JobChips", () => {
  it("names a running stage in the engine's words before its JobView arrives", () => {
    vi.spyOn(api, "job").mockReturnValue(new Promise<JobView>(() => {}));
    const running: StageStatus = { ...SERVED.C, stage: "time_varying", status: "running", error: null, job_id: "j1", progress: 0.5 };
    wrap(<JobChips pid="p" stages={{ time_varying: running }} />);
    expect(screen.getByTestId("job-time_varying")).toHaveTextContent(ENGINE_LABELS.time_varying!);
  });
});
