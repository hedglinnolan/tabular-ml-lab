/**
 * The mock Routers mirror turbotab/core/interview.py on a fit that will not come (the zero-row
 * crash): the seal's step and the substitution waited for good on a failed fit. A fit that failed
 * or was stopped holds no step back; one on its way still does, and opening the seal then says why
 * it cannot be taken (fit_failed), never "wait for the fit".
 */
import type { Decision, ProjectState, StageStatus } from "../api/schema";
import type { MockDataset } from "./datasets";
import type { MockProject } from "./db";
import { findings } from "./findings";
import { holdsBack, route } from "./m1-router";
import { m2Mock } from "./m2-record";

const status = (s: StageStatus["status"], extra: Partial<StageStatus> = {}): StageStatus => ({
  stage: "fit",
  status: s,
  key: null,
  fresh: s === "fresh",
  missing: [],
  error: null,
  job_id: null,
  progress: null,
  updated_at: "2026-10-08T00:00:00Z",
  cancelled: false,
  held: null,
  ...extra,
});

const FAILED = status("error", { error: "Needs 'design', which failed." });
const STOPPED = status("stale", { cancelled: true });

describe("a step that needs the fit fresh", () => {
  it("waits on the fit only while it is on its way", () => {
    expect(holdsBack(status("running"))).toBe(true);
    expect(holdsBack(status("idle"))).toBe(true);
    expect(holdsBack(undefined)).toBe(true);
    expect(holdsBack(status("fresh"))).toBe(false);
    expect(holdsBack(FAILED)).toBe(false);
    expect(holdsBack(STOPPED)).toBe(false);
  });

  it("the substitution waits on its earlier question, never on a failed or stopped fit", () => {
    const state = new Proxy({}, { get: () => null }) as ProjectState; // nothing answered yet
    const waiting = (fit: StageStatus) =>
      route(state, { fit }, null, [], {}, () => true).find((s) => s.key === "substitution")!.waiting_on;
    expect(waiting(status("idle"))).toContain("fit");
    expect(waiting(FAILED)).not.toContain("fit");
    expect(waiting(STOPPED)).not.toContain("fit");
  });
});

/**
 * BEATS_PREDICT ruling 5, as the engine's W1-P11 makes it: under prediction neither the energy
 * model nor the substitution is asked, and the dietary energy noticing does not fire; under
 * estimation, with energy and two energy-bearing exposures recorded, both are asked and it fires.
 * The table is constructed: kcal and glucose, three rows.
 */
describe("the energy model and the substitution under Predict (ruling 5)", () => {
  const roles = { kcal: "energy", protein: "exposure", sugar: "exposure", glucose: "outcome" };
  const asked = (purpose: string) => {
    const state = new Proxy({ purpose, lens: ["dietary"], roles, target: "glucose" } as Record<string, unknown>, {
      get: (t, k) => (k in t ? t[k as string] : null),
    }) as unknown as ProjectState;
    const steps = route(state, { fit: status("fresh") }, null, [], {}, () => true);
    return ["energy_adjustment", "substitution"].map((k) => steps.find((s) => s.key === k)!.status);
  };
  const ds: MockDataset = {
    name: "t.csv",
    nRows: 3,
    sourceBytes: 1,
    columns: [
      { name: "kcal", dtype: "numeric", physical_type: "double", values: [300, 2000, 6000] },
      { name: "glucose", dtype: "numeric", physical_type: "double", values: [90, 100, 110] },
    ],
  };
  const energyNoticed = (purpose: string) => findings(ds, ["dietary"], "glucose", purpose).findings.some((f) => f.id === "energy_adjustment");

  it("asks neither under prediction, and the energy noticing does not fire", () => {
    expect(asked("prediction")).toEqual(["not_applicable", "not_applicable"]);
    expect(energyNoticed("prediction")).toBe(false);
  });

  it("asks both under estimation, and the energy noticing fires", () => {
    expect(asked("inference")).not.toContain("not_applicable");
    expect(energyNoticed("inference")).toBe(true);
  });
});

describe("the M2 mock's seal step", () => {
  const state = {
    grain: { grain: "one_row_per_unit" },
    findings: {},
    split: { holdout: 0.2 },
    seal_opened: null,
  } as unknown as ProjectState;
  const m2 = m2Mock(() => state);
  const p = { records: [] } as unknown as MockProject;
  const seal = (fit: StageStatus) => m2.route(p, [], { fit }).find((s) => s.key === "open_seal")!;

  it("opens over a fit that failed or was stopped, and waits for one on its way", () => {
    expect(seal(status("running"))).toMatchObject({ status: "waiting", waiting_on: ["fit"] });
    expect(seal(FAILED)).toMatchObject({ status: "open", waiting_on: [] });
    expect(seal(STOPPED)).toMatchObject({ status: "open", waiting_on: [] });
  });

  it("refuses opening with the failure, never 'wait for the fit'", () => {
    const open = { kind: "open_seal" } as Decision;
    expect(m2.validate(p, open, { fit: FAILED })?.error.code).toBe("fit_failed");
    expect(m2.validate(p, open, { fit: STOPPED })?.error.code).toBe("fit_failed");
    expect(m2.validate(p, open, { fit: status("running") })?.error.code).toBe("fit_not_fresh");
  });
});
