/**
 * A stage that did not finish (Failure.tsx), on the statuses the engine serves
 * (fixtures/stages.json, captured from the engine by capture_stages.py). A refusal in words is
 * named and said as the engine serves it (StageStatus.refusal: the stage that refused, its words)
 * and its ways forward are the levers (the audit's G04 and S1); a crash keeps its two levers, Try
 * again and the answer to change. The work is named in the engine's words for every stage (S2).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import type { DecisionRecord, ProjectView, StageStatus } from "../api/schema";
import { StageFailure, rootFailure } from "./Failure";
import { ENGINE_LABELS, SERVED } from "./fixtures/stages";

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

/** A served status as the stage `name` would serve it: a refusal of its own is its own. */
function as(name: string, from: StageStatus): StageStatus {
  const own = from.refusal && from.refusal.stage === from.stage;
  return { ...from, stage: name, refusal: own ? { ...from.refusal!, stage: name } : from.refusal };
}

/** The models answered (select_models, the newest record): the answer the fit reads that a crash's
 *  lever offers to change (Failure.tsx causeOf). */
function viewWith(stages: Record<string, StageStatus>): Pick<ProjectView, "stages" | "decisions" | "interview"> {
  const models = { id: "d1", seq: 1, decision: { kind: "select_models", models: ["linear"] } } as unknown as DecisionRecord;
  return {
    stages,
    decisions: [models],
    interview: [{ key: "models", status: "answered" }] as unknown as ProjectView["interview"],
  };
}

describe("rootFailure", () => {
  it("names the stage that refused, with its words, as the engine serves them", () => {
    const failure = rootFailure({ AR: SERVED.AR, RH: SERVED.RH }, "AR")!;
    expect(failure.stage).toBe("RH");
    expect(failure.refusal).toEqual(SERVED.AR.refusal);
    expect(failure.message).toBe(SERVED.RH.refusal!.message);
  });

  it("names it from the served refusal, not the error's text, with that stage out of view", () => {
    const failure = rootFailure({ AR: SERVED.AR }, "AR")!;
    expect(failure.stage).toBe(SERVED.AR.refusal!.stage);
    expect(failure.message).toBe(SERVED.AR.refusal!.message);
  });

  it("names a crash from its error, as before, with no refusal", () => {
    // AC waits on CH; named here as real stages are (the error's form reads lowercase names).
    const fit = { ...as("fit", SERVED.AC), error: SERVED.AC.error!.replace("'CH'", "'design'") };
    const failure = rootFailure({ fit, design: as("design", SERVED.CH) }, "fit")!;
    expect(failure.stage).toBe("design");
    expect(failure.refusal).toBeNull();
    // C and CH fail with the refusal's words, untyped (the toy graph's construction).
    expect(failure.message).toBe(SERVED.R.refusal!.message);
  });
});

describe("StageFailure", () => {
  it("says a refused stage's words and offers its ways forward, not Try again or a guessed answer", () => {
    const fit = as("fit", SERVED.R);
    const refusal = fit.refusal!;
    wrap(<StageFailure pid="p" view={viewWith({ fit })} stage="fit" />);
    const note = screen.getByTestId("failure-fit");
    expect(note).toHaveAttribute("data-stage", "fit");
    expect(note).toHaveAttribute("data-kind", "refused");
    expect(note).toHaveTextContent(`${ENGINE_LABELS.fit} did not finish: ${refusal.message}`);
    const take = refusal.exits.find((x) => x.decision)!;
    expect(screen.getAllByTestId("stage-refusal-exit").map((b) => b.textContent)).toEqual([take.label]);
    expect(screen.queryByTestId("retry-fit")).toBeNull();
    expect(screen.queryByTestId("failure-change-models")).toBeNull();
  });

  it("says a refused stage's way forward that names no decision in words, none a press, where it may not record", () => {
    // The decision is the card's to offer (Stage's recordHere false): neither a press nor a sentence.
    const fit = as("fit", SERVED.R);
    const told = fit.refusal!.exits.find((x) => x.decision === null)!;
    wrap(<StageFailure pid="p" view={viewWith({ fit })} stage="fit" recordHere={false} />);
    const note = screen.getByTestId("failure-fit");
    expect(note.textContent).toBe(`${ENGINE_LABELS.fit} did not finish: ${fit.refusal!.message} ${String(told.label)}.`);
    expect(screen.queryAllByRole("button")).toEqual([]);
  });

  it("offers a crash's two levers as before: Try again and the answer to change", () => {
    const fit = as("fit", SERVED.C);
    wrap(<StageFailure pid="p" view={viewWith({ fit })} stage="fit" />);
    const note = screen.getByTestId("failure-fit");
    expect(note).toHaveAttribute("data-kind", "failed");
    expect(screen.getByTestId("retry-fit")).toHaveTextContent("Try again");
    expect(screen.getByTestId("failure-change-models")).toBeInTheDocument();
    expect(screen.queryByTestId("stage-refusal-exit")).toBeNull();
  });

  it("names the work in the engine's words for a stage beyond the first milestone's", () => {
    const stage = "time_varying";
    wrap(<StageFailure pid="p" view={viewWith({ [stage]: as(stage, SERVED.C) })} stage={stage} />);
    expect(screen.getByTestId(`failure-${stage}`)).toHaveTextContent(`${ENGINE_LABELS[stage]} did not finish:`);
  });
});
