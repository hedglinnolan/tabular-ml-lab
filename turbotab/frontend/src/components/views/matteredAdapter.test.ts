/**
 * The served specification curve (core/mattered.py `MatteredCurve`) as the specification curve
 * view reads it, against what the real server served on the NHANES journey under Estimate once the
 * plan was locked (quest/results/fixtures/mattered-estimate.json, written by
 * fixtures/capture_mattered.py; nothing in it by hand): 19 specifications estimated and one not,
 * the departure adding `hdl`, which waits for that column's reading; and the logistic curve the
 * stage makes under a yes/no outcome and the odds ratio (fixtures/mattered-binary.json, by the same
 * script).
 *
 * The truth is the served field: every number passes through unchanged, so each is compared with
 * the field it comes from, and every label is the served one with its backticks dropped.
 */
import { describe, expect, it } from "vitest";
import binary from "../../quest/results/fixtures/mattered-binary.json";
import capture from "../../quest/results/fixtures/mattered-estimate.json";
import { notEstimatedOf, onLogOdds, specCurveFromMattered, type ServedCurve } from "./matteredAdapter";

const C = capture as unknown as {
  nhanes: {
    before: { route: { status: number; body: { error: { message: string } } } };
    locked: { stages: { mattered: { artifact: { curve: ServedCurve } } } };
  };
};
const CURVE = C.nhanes.locked.stages.mattered.artifact.curve;
/** The logistic curve, under a yes/no outcome. */
const LOGISTIC = (binary as unknown as { binary: { artifact: { curve: ServedCurve } } }).binary.artifact.curve;
/** The line GET /mattered refused with before the lock (core/methods/spec_curve.py `SEALED`). */
const SEALED = C.nhanes.before.route.body.error.message;
const unticked = (s: string) => s.replaceAll("`", "");

describe("the capture", () => {
  it("is a curve of the recorded choices with one specification not estimated", () => {
    expect(CURVE.sealed).toBeNull();
    expect(CURVE.specs.length).toBeGreaterThan(1);
    expect(CURVE.specs.filter((s) => s.primary)).toHaveLength(1);
    expect(CURVE.not_estimated).toHaveLength(1);
    expect(CURVE.estimate_label).toContain("`");
    expect(C.nhanes.before.route.status).toBe(409);
  });
});

describe("specCurveFromMattered", () => {
  it("passes every specification through unchanged, its scale key as the view's", () => {
    const d = specCurveFromMattered(CURVE);
    expect(d.specs).toEqual(
      CURVE.specs.map((s) => ({ key: s.key, estimate: s.estimate, low: s.low, high: s.high, n: s.n, primary: s.primary, picks: s.picks, scaleKey: s.scale_key })),
    );
    expect(d.level).toBe(CURVE.level);
    expect(d.sealed).toBeNull();
    // The no-difference line is drawn, as the engine's own view contract draws it (`to_view`).
    expect(d.zero).toBe(true);
  });

  it("names the estimate, each choice and each option in the engine's words without backticks", () => {
    const d = specCurveFromMattered(CURVE);
    expect(d.estimateLabel).toBe(unticked(CURVE.estimate_label));
    expect(d.choices).toEqual(
      CURVE.choices.map((c) => ({ key: c.key, label: unticked(c.label), options: c.options.map((o) => ({ key: o.key, label: unticked(o.label) })) })),
    );
    // A column's name keeps its case: no label is capitalized.
    expect(d.choices.find((c) => c.key === "adjusted:protein")?.label).toBe("protein in the adjustment set");
  });

  it("carries no specification and no level once sealed, and every one while it is not", () => {
    const sealed = specCurveFromMattered({ ...CURVE, sealed: SEALED });
    expect(sealed.sealed).toBe(unticked(SEALED));
    expect(sealed.specs).toEqual([]);
    expect(sealed.level).toBeNull();
    expect(specCurveFromMattered(CURVE).specs).toHaveLength(CURVE.specs.length);
  });
});

describe("notEstimatedOf", () => {
  it("names the choice each specification not estimated changed, and its reason in the engine's words", () => {
    const [held] = CURVE.not_estimated;
    const changed = CURVE.choices.filter((c) => held!.picks[c.key] !== c.primary);
    expect(changed).toHaveLength(1); // one choice at a time
    const option = changed[0]!.options.find((o) => o.key === held!.picks[changed[0]!.key])!;
    expect(notEstimatedOf(CURVE)).toEqual([
      { key: held!.key, changed: [{ choice: unticked(changed[0]!.label), option: unticked(option.label) }], reason: held!.reason },
    ]);
  });

  it("names no choice for a specification that changes none", () => {
    const primary = CURVE.specs.find((s) => s.primary)!;
    const recorded = { key: primary.key, picks: primary.picks, primary: true, reason: CURVE.not_estimated[0]!.reason };
    expect(notEstimatedOf({ ...CURVE, not_estimated: [recorded] })[0]!.changed).toEqual([]);
  });
});

describe("onLogOdds", () => {
  it("holds for the logistic curve, every specification of one log-odds scale, and not for a difference in means", () => {
    expect(LOGISTIC.estimate_label).toMatch(/^Difference in log-odds of /);
    expect(onLogOdds(LOGISTIC)).toBe(true);
    expect(CURVE.estimate_label).toMatch(/^Difference in mean /);
    expect(onLogOdds(CURVE)).toBe(false);
  });

  it("does not hold where no log-odds curve is drawn: sealed, with no specification, or on two scales", () => {
    expect(onLogOdds({ ...LOGISTIC, sealed: SEALED })).toBe(false);
    expect(onLogOdds({ ...LOGISTIC, specs: [] })).toBe(false);
    const [first, ...rest] = LOGISTIC.specs;
    expect(onLogOdds({ ...LOGISTIC, specs: [{ ...first!, scale_key: "log_odds:another" }, ...rest] })).toBe(false);
  });
});
