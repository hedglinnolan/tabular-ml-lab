/**
 * The progress bar and the hover panel say what GET /quest says (plans/W1_PART2_PLAN.md §2 "W1-E"
 * acceptance): on the engine's own quest logs (fixtures/quest-contract.json, from the path fuzzer's
 * NHANES fixture under Predict), each segment carries its stage's progress, empty when the stage
 * is not reached, and the panel lists each stage's lines by label with their statuses, an answered
 * line with its answer and a waiting one with what it waits for.
 */
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { QuestLog } from "../api/schema";
import { CONTRACT } from "./fixtures/contract";
import { LABELS, openStageKey, stageOf } from "./log";
import { StageLine } from "./StageLine";

const LOGS = Object.entries(CONTRACT.logs).map(([name, x]) => [name, x.log] as const);

/** A stand-in answer for every decision on the log, so each answered line can be found by it. */
function answersOf(log: QuestLog): Map<string, string> {
  const out = new Map<string, string>();
  for (const s of log.stages) for (const l of s.lines) if (l.decision_id) out.set(l.decision_id, `answer of ${l.decision_id}`);
  return out;
}

describe.each(LOGS)("the bar and the panel on %s", (_name, log) => {
  it("fills each segment with its stage's served progress, and leaves a stage not reached empty", () => {
    render(<StageLine log={log} shown={openStageKey(log)} onOpen={() => {}} answers={new Map()} fitting={null} />);
    expect(screen.getByTestId("progress-bar").children).toHaveLength(log.stages.length);
    for (const s of log.stages) {
      const seg = screen.getByTestId(`segment-${s.key}`);
      if (!s.reached) {
        expect(seg, s.key).not.toHaveAttribute("data-reached");
        expect(seg, s.key).not.toHaveAttribute("data-answered");
        expect(seg.textContent, s.key).toContain("not reached yet");
        expect(seg.textContent, s.key).not.toMatch(/\d+ of \d+/);
        continue;
      }
      expect(seg).toHaveAttribute("data-answered", String(s.progress!.answered));
      expect(seg).toHaveAttribute("data-required", String(s.progress!.required));
      // A stage that asks nothing is done only once the analysis comes to it (finding 25): no
      // stage before it is the first one still working.
      const at = log.stages.findIndex((x) => x.key === openStageKey(log));
      const cameTo = log.stages.indexOf(s) <= at;
      if (s.progress!.complete && (s.progress!.required > 0 || cameTo)) expect(seg, s.key).toHaveAttribute("data-complete");
      else if (s.progress!.required === 0) {
        expect(seg, s.key).not.toHaveAttribute("data-complete");
        expect(seg.textContent, s.key).toContain("not reached yet");
      } else {
        expect(seg, s.key).not.toHaveAttribute("data-complete");
        expect(seg.textContent).toContain(`${s.progress!.answered} of ${s.progress!.required} answered`);
      }
    }
  });

  it("lists each stage's lines by label on pointing, with their served statuses and answers", () => {
    const answers = answersOf(log);
    render(<StageLine log={log} shown={openStageKey(log)} onOpen={() => {}} answers={answers} fitting={null} />);
    for (const s of log.stages) {
      fireEvent.pointerEnter(screen.getByTestId(`stage-${s.key}`));
      const panel = screen.getByTestId(`panel-${s.key}`);
      if (!s.reached) {
        expect(within(panel).getByText("May change as you answer")).toBeInTheDocument();
        expect(panel.textContent).not.toMatch(/\d+ of \d+/);
      } else if (s.lines.length === 0) {
        expect(within(panel).getByText("Nothing to decide here.")).toBeInTheDocument();
      } else {
        const heads = [...panel.querySelectorAll("h4")].map((h) => h.textContent);
        expect(heads).toEqual(LABELS.filter((label) => s.lines.some((l) => l.label === label)));
      }
      const rows = [...panel.querySelectorAll("li[data-status]")];
      expect(rows.map((r) => r.getAttribute("data-status"))).toEqual(
        s.reached
          ? LABELS.flatMap((label) =>
              [...s.lines].sort((a, b) => a.order - b.order).filter((l) => l.label === label).map((l) => l.status),
            )
          : s.lines.map((l) => l.status),
      );
      if (s.reached) {
        for (const l of s.lines) {
          const row = within(panel).getAllByTestId(`panel-line-${l.key}`)[0]!;
          if (l.status === "answered" && l.decision_id) expect(row.textContent).toContain(answers.get(l.decision_id));
          if (l.status === "waiting") {
            expect(row.textContent).toContain("Waiting for:");
            for (const w of l.waiting_for) expect(row.textContent).toContain(w.name);
          }
        }
      }
      fireEvent.pointerLeave(screen.getByTestId(`stage-${s.key}`));
    }
  });
});

describe("the bar's guard", () => {
  it("says why a reopened stage dropped back, beside the bar, and stays quiet when none did", () => {
    const log = CONTRACT.logs.results_reached.log;
    const { unmount } = render(
      <StageLine log={log} shown="models" onOpen={() => {}} answers={new Map()} fitting={null} />,
    );
    expect(log.stages.every((s) => s.reopened.length === 0)).toBe(true);
    expect(screen.queryByTestId("reopened-note")).toBeNull();
    unmount();
    const sentence = "Your data reopened: the join added 12 columns, so what each new column is gets read again.";
    const reopened: QuestLog = {
      ...log,
      stages: log.stages.map((s) =>
        s.key === "data"
          ? { ...s, reopened: [{ changed_in: "data", decision_id: "r9", kind: "join_files", questions: [], results: [], within: true, sentence }] }
          : s,
      ),
    };
    render(<StageLine log={reopened} shown="models" onOpen={() => {}} answers={new Map()} fitting={null} />);
    expect(screen.getByTestId("reopened-note")).toHaveTextContent(sentence);
    expect(screen.getByTestId("segment-data")).toHaveAttribute("data-reopened");
  });
});

/**
 * Finding 25 of the 2026-10-17 walkthrough: under Predict the engine reaches Write-up with Results,
 * at 0 of 0, so Write-up read as done, its segment full, while the analysis was still before it.
 */
describe("a stage that asks nothing, before the analysis comes to it (finding 25)", () => {
  const log = CONTRACT.logs.results_reached.log;
  /** Every stage answered, confirmed and complete, as the engine serves it then. */
  const allDone: QuestLog = {
    ...log,
    stages: log.stages.map((s) => ({
      ...s,
      sweep: s.sweep ? { ...s.sweep, answered: true } : null,
      lines: s.lines.map((l) => (l.label === "Decide" ? { ...l, status: "answered" as const, decision_id: `r-${l.key}` } : l)),
      progress: s.progress ? { answered: s.progress.required, required: s.progress.required, complete: true } : null,
    })),
  };

  it("never reads as done on the line or the bar while an earlier stage is still working", () => {
    const writeup = stageOf(log, "writeup")!;
    expect(writeup).toMatchObject({ reached: true, progress: { required: 0, complete: true } });
    expect(stageOf(log, "results")!.progress?.complete).toBe(false);
    render(<StageLine log={log} shown="results" onOpen={() => {}} answers={new Map()} fitting={null} />);
    expect(screen.getByTestId("stage-writeup")).toHaveAttribute("data-status", "todo");
    const seg = screen.getByTestId("segment-writeup");
    expect(seg).not.toHaveAttribute("data-complete");
    expect(seg.querySelector("span")!.style.width).toBe("0%");
    expect(seg.textContent).toContain("not reached yet");
  });

  it("reads as done once every stage before it is", () => {
    render(<StageLine log={allDone} shown="results" onOpen={() => {}} answers={new Map()} fitting={null} />);
    expect(screen.getByTestId("stage-writeup")).toHaveAttribute("data-status", "done");
    const seg = screen.getByTestId("segment-writeup");
    expect(seg).toHaveAttribute("data-complete");
    expect(seg.querySelector("span")!.style.width).toBe("100%");
  });
});
