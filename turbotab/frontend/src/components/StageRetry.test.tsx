/**
 * The way forward from a stage that will not restart by itself (StageRetry), on the statuses the
 * engine serves (fixtures/stages.json): Try again for a crash and Recompute for a stopped stage,
 * and nothing for a refusal in words (the audit's G04), which is for the current answers, so
 * running them again refuses again; its ways forward are StageRefusal's.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import type { StageStatus } from "../api/schema";
import { SERVED } from "./fixtures/stages";
import { StageRetry, needsRetry } from "./StageRetry";

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>);
}

describe("StageRetry", () => {
  it("offers no Try again for a refused stage, or one waiting on a refused stage", () => {
    for (const status of [SERVED.R, SERVED.AR, SERVED.U]) {
      const { container, unmount } = wrap(<StageRetry pid="p" status={status} />);
      expect(container).toBeEmptyDOMElement();
      unmount();
    }
  });

  it("offers Try again for a crash and a stage waiting on one, Recompute for a stopped stage", () => {
    const stopped: StageStatus = { ...SERVED.C, stage: "S", status: "stale", error: null, cancelled: true };
    wrap(
      <>
        <StageRetry pid="p" status={SERVED.C} />
        <StageRetry pid="p" status={SERVED.AC} />
        <StageRetry pid="p" status={stopped} />
      </>,
    );
    expect(screen.getByTestId("retry-C")).toHaveTextContent("Try again");
    expect(screen.getByTestId("retry-AC")).toHaveTextContent("Try again");
    expect(screen.getByTestId("retry-S")).toHaveTextContent("Recompute");
  });

  it("still counts a refused stage as one that will not restart by itself", () => {
    // needsRetry says whether a stage waits on the user (Record and AskFor read it so); only the
    // press is withheld for a refusal.
    expect(needsRetry(SERVED.R)).toBe(true);
    expect(needsRetry({ ...SERVED.R, status: "fresh", error: null, refusal: null })).toBe(false);
  });
});
