/**
 * The grain stated, not asked (M2_CONTRACT §10, §12.5): "Not asked: every `SEQN` appears once,
 * so each person is one row", with "Ask me anyway". It is said once, and only when its turn
 * comes: a stated step after the open question waits under "Then".
 */
import { fireEvent, render, screen } from "@testing-library/react";
import type { InterviewStep, QuestionKey } from "../../api/m1-types";
import { layoutFlow } from "./flow";
import { StatedSkip, skipReason } from "./StatedSkip";

const step = (key: QuestionKey, status: InterviewStep["status"], extra: Partial<InterviewStep> = {}) =>
  ({
    key,
    status,
    decision_id: status === "answered" ? `${key}-1` : null,
    reason: null,
    waiting_on: [],
    deferred_findings: [],
    ...extra,
  }) as InterviewStep;

const GRAIN = "Not asked: every `SEQN` appears once, so each person is one row.";

describe("the grain stated by a unique identifier", () => {
  it("says 'Not asked:' once, chips the column, and reopens on 'Ask me anyway'", () => {
    const ask = vi.fn();
    render(<StatedSkip qkey="grain" reason={GRAIN} onAsk={ask} />);
    const row = screen.getByTestId("skip-grain");
    expect(row).toHaveTextContent("Not asked: every SEQN appears once, so each person is one row.");
    expect(row.textContent!.match(/Not asked/g)).toHaveLength(1);
    expect(screen.getByText("SEQN").tagName).toBe("CODE");
    fireEvent.click(screen.getByRole("button", { name: "Ask me anyway" }));
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it("keeps a reason that does not lead with it, and drops an empty one", () => {
    expect(skipReason("Not asked — the dates are 3 to 14 days apart.")).toBe(
      "the dates are 3 to 14 days apart.",
    );
    expect(skipReason("These look like repeats.")).toBe("These look like repeats.");
    expect(skipReason(null)).toBe("");
  });

  it("is said in the flow once the questions before it are answered", () => {
    const interview = [
      step("lens", "answered"),
      step("target", "answered"),
      step("purpose", "answered"),
      step("grain", "skipped", { reason: GRAIN }),
      step("roles", "open"),
      step("exclusions", "waiting", { waiting_on: ["roles"] }),
    ];
    const flow = layoutFlow(interview);
    expect(flow.inline.map((s) => s.key)).toEqual(["lens", "target", "purpose", "grain", "roles"]);
    expect(flow.next.map((s) => s.key)).toEqual(["exclusions"]);
  });

  it("waits under 'Then' while a question before it is open", () => {
    const interview = [
      step("lens", "answered"),
      step("target", "answered"),
      step("task", "skipped"),
      step("purpose", "open"),
      step("grain", "skipped", { reason: GRAIN }),
      step("repeat_kind", "not_applicable", { reason: "Each unit appears once." }),
      step("roles", "waiting", { waiting_on: ["purpose"] }),
    ];
    const flow = layoutFlow(interview);
    // The task, stated before the open question, stays in place; the grain does not run ahead.
    expect(flow.inline.map((s) => s.key)).toEqual(["lens", "target", "task", "purpose"]);
    expect(flow.next.map((s) => s.key)).toEqual(["grain", "repeat_kind", "roles"]);
    // "Ask me anyway" from elsewhere (a finding's lever) brings it into the flow, asked.
    const reopened = layoutFlow(interview, { grain: true });
    expect(reopened.inline.map((s) => s.key)).toContain("grain");
    expect(reopened.next.map((s) => s.key)).not.toContain("grain");
  });

  it("stands in place as a pending row when the first unanswered step waits on a stage", () => {
    const interview = [
      step("lens", "answered"),
      step("target", "waiting", { waiting_on: ["ingest"] }),
      step("grain", "skipped", { reason: GRAIN }),
    ];
    const flow = layoutFlow(interview);
    expect(flow.pendingStep?.key).toBe("target");
    expect(flow.inline.map((s) => s.key)).toEqual(["lens"]);
    expect(flow.next.map((s) => s.key)).toEqual(["grain"]);
  });

  it("says adjacent inapplicable steps once, as a run", () => {
    const interview = [
      step("grain", "skipped", { reason: GRAIN }),
      step("repeat_kind", "not_applicable"),
      step("unit", "not_applicable"),
      step("aggregation", "not_applicable"),
      step("roles", "open"),
    ];
    const { naRuns } = layoutFlow(interview);
    expect(naRuns.get("repeat_kind")?.map((s) => s.key)).toEqual([
      "repeat_kind",
      "unit",
      "aggregation",
    ]);
    expect(naRuns.get("unit")).toBeNull();
    expect(naRuns.has("grain")).toBe(false);
  });
});
