/**
 * The seal's state machine (Tier A: seal integrity): held-out scores reach the screen only once
 * the seal is opened, whatever an artifact carries, and only a verified basis is a clean lock.
 */
import type { FitArtifact } from "../../../api/m1-stage-types";
import { comparisonOf, metricBasis } from "../results/model";
import { glyphOf, isExploratory, scoresVisible, sealPhase, shownHoldout, type SealPhase } from "./phase";

function fit(over: Partial<FitArtifact> = {}): FitArtifact {
  const model = (family: string, holdout: number | null) => ({
    family,
    label: family,
    cv: { r2: { mean: 0.1, sd: 0.02, folds: [0.1, 0.12, 0.08] } },
    holdout: holdout === null ? null : { r2: holdout },
    concerns: [],
    fit_seconds: 1,
    baseline: { metric: "r2", value: 0, label: "the outcome's average" },
  });
  return {
    task: "regression",
    primary_metric: "r2",
    metric_labels: { r2: "R²" },
    n_train: 240,
    n_holdout: 60,
    models: [model("linear", 0.31), model("boosted_trees", 0.22)],
    holdout_sealed: true,
    changed_after_seal: false,
    post_seal_decisions: [],
    ...over,
  } as unknown as FitArtifact;
}

describe("the seal's phases", () => {
  it("walks none → sealed → opened → post-seal as the fit and the Record change", () => {
    const steps: [FitArtifact | null, boolean, SealPhase][] = [
      [null, false, "none"],
      [fit({ n_holdout: 0, holdout_sealed: false }), false, "cv_only"],
      [fit(), false, "sealed"],
      // open_seal recorded, the fit not yet fetched again: still sealed (no number before it arrives)
      [fit(), true, "sealed"],
      [fit({ holdout_sealed: false }), true, "opened"],
      [fit({ holdout_sealed: false, changed_after_seal: true, post_seal_decisions: ["d12"] }), true, "post_seal"],
    ];
    for (const [f, opened, want] of steps) expect(sealPhase(f, opened)).toBe(want);
  });

  it("never shows a held-out score before the seal is opened, even one an artifact carries", () => {
    // A stray score in an unopened project (an older artifact, a server bug) is not drawn.
    const leaky = fit({ holdout_sealed: false });
    for (const opened of [false]) {
      const phase = sealPhase(leaky, opened);
      expect(phase).toBe("sealed");
      expect(scoresVisible(phase)).toBe(false);
      const cmp = comparisonOf(leaky, null, "r2", phase);
      expect(cmp.rows.map((r) => r.holdout)).toEqual([null, null]);
      expect(metricBasis(leaky, null, phase)).toContain("60 held-out rows sealed");
      expect(metricBasis(leaky, null, phase)).not.toContain("scored");
    }
    // The default is sealed: a caller that forgets the phase shows nothing either.
    expect(comparisonOf(leaky, null).rows.every((r) => r.holdout === null)).toBe(true);
    expect(shownHoldout("cv_only", 0.4)).toBeNull();
    expect(shownHoldout("none", 0.4)).toBeNull();
  });

  it("shows the exact scores once opened, and after a post-seal change", () => {
    const opened = fit({ holdout_sealed: false });
    expect(comparisonOf(opened, null, "r2", sealPhase(opened, true)).rows.map((r) => r.holdout)).toEqual([0.31, 0.22]);
    const post = fit({ holdout_sealed: false, changed_after_seal: true });
    expect(comparisonOf(post, null, "r2", sealPhase(post, true)).rows.map((r) => r.holdout)).toEqual([0.31, 0.22]);
    expect(metricBasis(opened, null, "opened")).toContain("60 held-out rows scored once");
  });
});

describe("the seal's glyph", () => {
  it("draws only a verified basis as a clean lock", () => {
    expect(glyphOf("grouped")).toBe("closed");
    expect(glyphOf("one_row_per_unit")).toBe("closed");
    expect(glyphOf("abandoned")).toBe("abandoned");
    expect(glyphOf("undetermined")).toBe("undetermined");
    // A basis the stage cannot read (missing, or a state it does not know) is never a clean lock.
    for (const s of [null, undefined, "", "repetition_found_grouping_abandoned", "cross_sectional"])
      expect(glyphOf(s)).not.toBe("closed");
  });

  it("labels every basis but a verified one exploratory", () => {
    expect(isExploratory("grouped", false)).toBe(false);
    expect(isExploratory("grouped", true)).toBe(true); // a chronology drawn at random says so
    expect(isExploratory("undetermined", false)).toBe(true);
    expect(isExploratory(undefined)).toBe(true);
  });
});
