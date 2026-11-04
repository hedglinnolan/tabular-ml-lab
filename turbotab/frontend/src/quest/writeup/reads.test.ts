/**
 * How Write-up veils the tables and figures it drew from an earlier read of the export (reads.ts
 * `veilOf`): live only where nothing moved since that read and the export is neither waited for,
 * read again nor refused; else by what moved in the view since: a stage computing, failed or
 * stopped, or a decision recorded, in StaleVeil's words for that state. Where the view shows none
 * of those, the tag names no cause the view does not show: it says only that the export now waits,
 * refused or could not be read. Each state and tag is asked where it applies and where it does not.
 * The view is a real server's captured NHANES walk, changed by hand as each case says.
 */
import type { DecisionRecord, ProjectView, StageStatus } from "../../api/schema";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import { basisOf, EXPORT_TAG, veilOf } from "./reads";

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const VIEW = ((): ProjectView => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, "quest-writeup", "2026-11-04T00:00:00Z");
})();

const LIVE = { ready: true, again: false, failed: null } as const;
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
    expect(veilOf(drawnFrom(VIEW), VIEW, LIVE)).toEqual({ state: "fresh" });
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, ready: false }).state).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, again: true }).state).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: "refused" }).state).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: "unread" }).state).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), withDecision(VIEW), LIVE).state).not.toBe("fresh");
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), LIVE).state).not.toBe("fresh");
  });

  it("is a progress tick's: none moves the basis", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { progress: 0.5 }), LIVE)).toEqual({ state: "fresh" });
  });

  it("is \"recomputing\" while the export is read again, or a stage that moved since computes", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), { ...LIVE, again: true })).toEqual({ state: "recomputing" });
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "running" }), { ...LIVE, ready: false })).toEqual({ state: "recomputing" });
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "queued" }), { ...LIVE, failed: "refused" })).toEqual({ state: "recomputing" });
  });

  it("is not \"recomputing\" for a stage that was computing already when the export was read", () => {
    const running = withStage(VIEW, "fit", { status: "running" });
    expect(veilOf(drawnFrom(running), withDecision(running), { ...LIVE, ready: false })).toEqual({ state: "stale" });
  });

  it("is \"failed\" where a stage that moved since failed", () => {
    expect(veilOf(drawnFrom(VIEW), withDecision(withStage(VIEW, "fit", { status: "error" })), { ...LIVE, ready: false })).toEqual({ state: "failed" });
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "error" }), { ...LIVE, failed: "refused" })).toEqual({ state: "failed" });
  });

  it("is not \"failed\" for a stage that had failed already when the export was read", () => {
    const failed = withStage(VIEW, "fit", { status: "error" });
    expect(veilOf(drawnFrom(failed), withDecision(failed), { ...LIVE, ready: false })).toEqual({ state: "stale" });
  });

  it("is \"stopped\" where a stage that moved since was stopped", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "stale", cancelled: true }), { ...LIVE, ready: false })).toEqual({ state: "stopped" });
  });

  it("is not \"stopped\" for a stage stopped already when the export was read", () => {
    const stopped = withStage(VIEW, "fit", { status: "stale", cancelled: true });
    expect(veilOf(drawnFrom(stopped), withDecision(stopped), { ...LIVE, ready: false })).toEqual({ state: "stale" });
  });

  it("is \"stale\" in its own words where a decision was recorded since, and nothing that moved computes, failed or was stopped", () => {
    expect(veilOf(drawnFrom(VIEW), withDecision(VIEW), { ...LIVE, ready: false })).toEqual({ state: "stale" });
    expect(veilOf(drawnFrom(VIEW), withDecision(withStage(VIEW, "fit", { status: "stale" })), { ...LIVE, failed: "refused" })).toEqual({ state: "stale" });
    expect(veilOf(drawnFrom(VIEW), withDecision(VIEW), { ...LIVE, failed: "unread" })).toEqual({ state: "stale" });
  });

  it("says only that the export now waits where no decision was recorded and nothing that moved computes, failed or was stopped, and the checklist waits", () => {
    // Nothing moved (a resync, the export now refused at its gate and the checklist waiting)...
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, ready: false })).toEqual({ state: "stale", tag: EXPORT_TAG.waits });
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, ready: false, failed: "refused" })).toEqual({ state: "stale", tag: EXPORT_TAG.waits });
    // ...or the fit's result moved and it rests, while the checklist read when it computed still waits.
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), { ...LIVE, ready: false })).toEqual({ state: "stale", tag: EXPORT_TAG.waits });
  });

  it("says only that the export refused, or could not be read, where no decision was recorded and nothing that moved computes, failed or was stopped", () => {
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: "refused" })).toEqual({ state: "stale", tag: EXPORT_TAG.refused });
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), { ...LIVE, failed: "refused" })).toEqual({ state: "stale", tag: EXPORT_TAG.refused });
    expect(veilOf(drawnFrom(VIEW), VIEW, { ...LIVE, failed: "unread" })).toEqual({ state: "stale", tag: EXPORT_TAG.unread });
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { status: "stale" }), { ...LIVE, failed: "unread" })).toEqual({ state: "stale", tag: EXPORT_TAG.unread });
  });

  it("says only that the tables are as last read where a stage moved and rests, with no decision, and no read is in flight, waited for or refused", () => {
    expect(veilOf(drawnFrom(VIEW), withStage(VIEW, "fit", { key: "another" }), LIVE)).toEqual({ state: "stale", tag: EXPORT_TAG.read });
  });

  it("names no answer and no recompute in any tag it gives", () => {
    for (const tag of Object.values(EXPORT_TAG)) expect(tag).not.toMatch(/answer|recompute/);
  });
});
