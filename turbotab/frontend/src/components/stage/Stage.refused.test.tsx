/**
 * A fit the engine refused in words (StageStatus.refusal, the audit's G04), on the stage's Results
 * and on its "now" scene beside the recorded rows and columns (LiveScenes NotFitted ->
 * StageFailure, through ResultsScene and NowScene). Classic's stage offers each way forward that is a
 * decision as a press that records it; the quest shell's (Stage `recordHere={false}`: the card is
 * the one place a choice is recorded, FOUNDATION §1) offers none, as it draws no refusal's ways
 * forward elsewhere (Stage.record.test.tsx). The statuses are the engine's
 * (components/fixtures/stages.json).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { keys } from "../../api/queries";
import type { ProjectView, StageStatus } from "../../api/schema";
import type { StageFocus } from "../../state/focus";
import { SERVED } from "../fixtures/stages";
import { Stage } from "./Stage";

const PID = "p-refused";
/** R's refusal as the fit serves it, its own. */
const FIT: StageStatus = { ...SERVED.R, stage: "fit", refusal: { ...SERVED.R.refusal!, stage: "fit" } };

/** The Results segment's scene, or with `now` the live scene a stage with no fit shows: the
 *  recorded rows and columns (the cohort's, put in the query cache by construction) above Results. */
function mount(recordHere?: boolean, now = false) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  const cohort = { status: "fresh", key: "k" };
  const view = {
    summary: { id: PID, name: "toy", n_rows: 10, n_cols: 3 },
    state: { purpose: "inference", target: "y", roles: null, models: ["ridge"] },
    decisions: [],
    interview: [],
    stages: now ? { cohort, fit: FIT } : { fit: FIT },
  } as unknown as ProjectView;
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  if (now) qc.setQueryData(keys.stage(PID, "cohort"), { key: "k", fresh: true, artifact: { predictors: ["x"], steps: [] } });
  const focus: StageFocus = now ? { kind: "live" } : { kind: "banner", segment: "result" };
  render(
    <QueryClientProvider client={qc}>
      <Stage pid={PID} view={view} focus={focus} onFocus={() => {}} recordHere={recordHere} />
    </QueryClientProvider>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("a refused fit on the stage's Results", () => {
  const take = FIT.refusal!.exits.find((x) => x.decision)!;

  it("offers its decision as a press on Classic's stage", () => {
    mount();
    expect(screen.getByTestId("fit-failure")).toHaveAttribute("data-kind", "refused");
    expect(screen.getAllByTestId("stage-refusal-exit").map((b) => b.textContent)).toEqual([take.label]);
  });

  it("offers no press and says no decision on the shell's stage, where the card records", () => {
    mount(false);
    const note = screen.getByTestId("fit-failure");
    expect(note).toHaveAttribute("data-kind", "refused");
    expect(screen.queryAllByTestId("stage-refusal-exit")).toEqual([]);
    expect(note).not.toHaveTextContent(String(take.label));
  });

  describe("on the now scene, above the recorded rows and columns", () => {
    it("offers its decision as a press on Classic's stage", () => {
      mount(undefined, true);
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "now");
      expect(screen.getByTestId("fit-failure")).toHaveAttribute("data-kind", "refused");
      expect(screen.getAllByTestId("stage-refusal-exit").map((b) => b.textContent)).toEqual([take.label]);
    });

    it("offers no press and says no decision on the shell's stage", () => {
      mount(false, true);
      expect(screen.getByTestId("stage")).toHaveAttribute("data-scene", "now");
      const note = screen.getByTestId("fit-failure");
      expect(note).toHaveAttribute("data-kind", "refused");
      expect(screen.queryAllByTestId("stage-refusal-exit")).toEqual([]);
      expect(note).not.toHaveTextContent(String(take.label));
    });
  });
});
