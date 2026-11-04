/**
 * How Write-up veils the tables and figures it drew from an earlier read of the export (reads.ts
 * `veilOf`): live only where nothing moved since that read and the export is neither waited for,
 * read again nor refused; else by what moved in the view since: a stage computing, failed or
 * stopped, or a decision recorded. Each state is asked where it applies and where it does not.
 * The view is a real server's captured NHANES walk, changed by hand as each case says.
 */
import type { DecisionRecord, ProjectView, StageStatus } from "../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import { basisOf, veilOf } from "./reads";

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const VIEW = ((): ProjectView => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, "quest-writeup", "2026-11-04T00:00:00Z");
})();

const LIVE = { ready: true, again: false, failed: false };
const withStage = (view: ProjectView, name: string, change: Partial<StageStatus>): ProjectView => ({
  ...view,
  stages: { ...view.stages, [name]: { ...view.stages[name]!, ...change } },
});
const withDecision = (view: ProjectView): ProjectView => {
  const last = view.decisions[view.decisions.length - 1] as DecisionRecord;
  return { ...view, decisions: [...view.decisions, { ...last, id: `${last.id}-next` }] };
};
const drawnFrom = (view: ProjectView) => basisOf(view)!;

describe("veilOf", () => {
  it("the walk the cases change has a fresh fit, and another stage beside it", () => {
    expect(VIEW.stages.fit?.status).toBe("fresh");
    expect(Object.keys(VIEW.stages).length).toBeGreaterThan(1);
  });

  it("is live where nothing moved, nothing is waited for, and no read is in flight or refused", () => {
    expect(veilOf(drawnFrom(VIEW), VIEW, LIVE)).toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, ready: false })).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, again: true })).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: true })).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), withDecision(VIEW), LIVE)).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), LIVE)).not.toBe("fresh");
  });

  it("is a progress tick's: none moves the basis", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { progress: 0.5 }), LIVE)).toBe("fresh");
  });

  it("is \"recomputing\" while the export is read again, or a stage that moved since computes", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), { ...LIVE, again: true })).toBe("recomputing");
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "running" }), { ...LIVE, ready: false })).toBe("recomputing");
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "queued" }), { ...LIVE, failed: true })).toBe("recomputing");
  });

  it("is not \"recomputing\" for a stage that was computing already when the export was read", () => {
    const running = withStage(VIEW, "fit", { status: "running" });
    expect(veilOf(drawnFrom(running), withDecision(running), { ...LIVE, ready: false })).toBe("stale");
  });

  it("is \"failed\" where a stage that moved since failed", () => {
    expect(veilOf(drawnFrom(VIEW), withDecision(withStage(VIEW, "fit", { status: "error" })), { ...LIVE, ready: false })).toBe("failed");
  });

  it("is not \"failed\" for a stage that had failed already when the export was read", () => {
    const failed = withStage(VIEW, "fit", { status: "error" });
    expect(veilOf(drawnFrom(failed), withDecision(failed), { ...LIVE, ready: false })).toBe("stale");
  });

  it("is \"stopped\" where a stage that moved since was stopped", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "stale", cancelled: true }), { ...LIVE, ready: false })).toBe("stopped");
  });

  it("is not \"stopped\" for a stage stopped already when the export was read", () => {
    const stopped = withStage(VIEW, "fit", { status: "stale", cancelled: true });
    expect(veilOf(drawnFrom(stopped), withDecision(stopped), { ...LIVE, ready: false })).toBe("stale");
  });

  it("is \"stale\" where a decision was recorded since, and nothing that moved computes, failed or was stopped", () => {
    expect(veilOf(drawnFrom(VIEW), withDecision(VIEW), { ...LIVE, ready: false })).toBe("stale");
    expect(veilOf(drawnFrom(VIEW), withDecision(withStage(VIEW, "fit", { status: "stale" })), { ...LIVE, failed: true })).toBe("stale");
  });

  it("is \"failed\" where no decision was recorded and no stage moved to computing, failing or stopped, and the export read again refused or failed", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), { ...LIVE, failed: true })).toBe("failed");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: true })).toBe("failed");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, ready: false })).toBe("stale");
  });
});
