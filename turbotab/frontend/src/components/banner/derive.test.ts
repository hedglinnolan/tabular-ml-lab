import type { ProjectView, StageResult, StageStatus } from "../../api/schema";
import type {
  CohortArtifact,
  DesignArtifact,
  FitArtifact,
  InterviewStep,
  SplitArtifact,
} from "../../api/m1-types";
import { bestModel, deriveBanner, formatMetric, worstVeil, type BannerInput } from "./derive";

const status = (stage: string, key: string, s: StageStatus["status"] = "fresh"): StageStatus => ({
  stage,
  status: s,
  key,
  fresh: s === "fresh",
  missing: [],
  error: null,
  job_id: null,
  progress: null,
  updated_at: "2026-10-01T12:00:00Z",
  cancelled: false,
});

function result<A>(stage: string, key: string, artifact: A, fresh = true): StageResult<A> {
  return { stage, key, fresh, status: fresh ? "fresh" : "stale", artifact };
}

const step = (key: InterviewStep["key"], s: InterviewStep["status"], waiting_on: string[] = []) =>
  ({ key, status: s, decision_id: null, reason: null, waiting_on }) as InterviewStep;

const cohort: CohortArtifact = {
  steps: [
    { key: "loaded", label: "Rows loaded", n: 21849, dropped: 0, reason: null, decision_id: null },
    {
      key: "outcome_measured",
      label: "glucose measured",
      n: 21849,
      dropped: 0,
      reason: "glucose is missing",
      decision_id: null,
    },
    {
      key: "exclude_kcal",
      label: "kcal within 500–5000",
      n: 21348,
      dropped: 501,
      reason: "implausible intakes",
      decision_id: "d5",
    },
    {
      key: "complete_cases",
      label: "Complete cases",
      n: 2943,
      dropped: 18405,
      reason: "a predictor is missing",
      decision_id: "d6",
    },
  ],
  n_final: 2943,
  predictors: Array.from({ length: 11 }, (_, i) => `p${i}`),
  complete_case_loss: null,
};

const split: SplitArtifact = {
  n_train: 2352,
  n_holdout: 591,
  holdout: 0.2,
  seed: 0,
  folds: 5,
  grouped_by: null,
  n_groups: null,
  stratified: false,
  fold_scheme: "random",
  folds_stratified: false,
  time_ordered_folds: false,
  note: "",
  basis: {
    state: "one_row_per_unit",
    column: null,
    label: "one row per unit",
    sentence: "Held out by row: each row was said to be a different unit, and no identifier repeats.",
    exploratory: false,
    source: "grain",
    n_units: null,
  },
  chronology: null,
  exploratory: false,
};

const node = (id: string, lane: "raw" | "adjusted" | "matrix", count = 1) => ({
  id,
  column: id,
  lane,
  role: null,
  label: id,
  formula: null,
  group: null,
  count,
});

const design: DesignArtifact = {
  lineage: {
    nodes: [
      node("age", "raw"),
      node("kcal", "raw"),
      node("nutrients", "raw", 9),
      node("age_m", "matrix"),
      node("nutrients_adj", "matrix", 9),
    ],
    links: [],
    collapsed: true,
  },
  matrix: { n_rows: 2352, n_cols: 10 },
  models: [],
  estimand: null,
  substitution_pairs: [],
  warnings: [],
  nested: [],
  left_out: [],
  terms: {},
  energy_form: null,
};

const baseline = { metric: "r2", value: 0, label: "the outcome's average" };

// The fields WP9 added (models/performance.py, validation.py): no SE, calibration or resampling.
const noSe = { se: null, ci_low: null, ci_high: null, repeats: 1, repeat_sd: null };
const wp9 = { calibration: null, holdout_detail: null, optimism: null, internal_external: null };

const fit: FitArtifact = {
  task: "regression",
  levels: null,
  withheld: null,
  estimand: null,
  primary_metric: "r2",
  metric_labels: { r2: "R²", rmse: "RMSE" },
  n_train: 2352,
  n_holdout: 591,
  models: [
    {
      family: "linear",
      label: "Linear regression",
      cv: {
        r2: { mean: 0.07, sd: 0.04, folds: [], estimate: 0.076, estimator: "pooled", ...noSe },
        rmse: { mean: 44.8, sd: 1, folds: [], estimate: 44.8, estimator: "pooled", ...noSe },
      },
      holdout: null,
      coefficients: null,
      fit_seconds: 0.03,
      concerns: [],
      baseline,
      versus_baseline: null,
      inference: null,
      coefficients_n: null,
      role: null,
      ...wp9,
      exposure_tests: [],
    },
    {
      family: "elastic_net",
      label: "Elastic net",
      cv: {
        r2: { mean: 0.07, sd: 0.03, folds: [], estimate: 0.077, estimator: "pooled", ...noSe },
        rmse: { mean: 44.7, sd: 1, folds: [], estimate: 44.7, estimator: "pooled", ...noSe },
      },
      holdout: null,
      coefficients: null,
      fit_seconds: 0.8,
      concerns: [],
      baseline,
      versus_baseline: null,
      inference: null,
      coefficients_n: null,
      role: null,
      ...wp9,
      exposure_tests: [],
    },
    {
      family: "boosted_trees",
      label: "Gradient-boosted trees",
      cv: {
        r2: { mean: -0.05, sd: 0.06, folds: [], estimate: -0.039, estimator: "pooled", ...noSe },
        rmse: { mean: 47.5, sd: 1, folds: [], estimate: 47.5, estimator: "pooled", ...noSe },
      },
      holdout: null,
      coefficients: null,
      fit_seconds: 1.7,
      concerns: ["Predicts worse than the outcome's average: CV R² −0.04"],
      baseline,
      versus_baseline: null,
      inference: null,
      coefficients_n: null,
      role: null,
      ...wp9,
      exposure_tests: [],
    },
  ],
  holdout_sealed: true,
  changed_after_seal: false,
  post_seal_decisions: [],
  fold_scheme: "random",
  cv_definition: null,
  selection: null,
  final_model: null,
  final_note: null,
  validation: "kfold",
  repeats: 1,
  ranking: "highest R²",
  se_definition: null,
  comparisons: [],
  precision: null,
  imbalance: null,
};

function input(over: Partial<BannerInput> = {}, viewOver: Partial<ProjectView> = {}): BannerInput {
  const view = {
    summary: {
      id: "p1",
      name: "nhanes",
      created_at: "2026-10-01T00:00:00Z",
      source_kind: "upload",
      source_name: "nhanes.csv",
      n_rows: 21849,
      n_cols: 29,
      ingest: null,
    },
    state: {
      lens: ["dietary"],
      target: "glucose",
      task: null,
      purpose: "prediction",
      roles: { age: "covariate" },
      energy_adjustment: {
        method: "residual",
        energy_column: "kcal",
        nutrients: [],
        log_transform: false,
        strata: null,
      },
      exclusions: [],
      missing: "complete_case",
      split: { holdout: 0.2, seed: 0, folds: 5 },
      models: ["linear", "elastic_net", "boosted_trees"],
      substitution: null,
    },
    stages: {
      cohort: status("cohort", "c1"),
      split: status("split", "s1"),
      design: status("design", "d1"),
      fit: status("fit", "f1"),
    },
    interview: [step("energy_adjustment", "answered"), step("substitution", "open")],
    ...viewOver,
  } as BannerInput["view"];
  return {
    view,
    cohort: result("cohort", "c1", cohort),
    split: result("split", "s1", split),
    design: result("design", "d1", design),
    fit: result("fit", "f1", fit),
    ...over,
  };
}

describe("deriveBanner", () => {
  it("reads the row flow: loaded, every step that removed rows, then the split", () => {
    const [rows] = deriveBanner(input()).segments;
    expect(rows.flow.map((f) => f.n)).toEqual([21849, 21348, 2943]);
    expect(rows.train).toBe(2352);
    expect(rows.holdout).toBe(591);
    expect(rows.summary).toContain("2,352 to train, 591 held out");
  });

  it("says cross-validation only when nothing is held out", () => {
    const [rows] = deriveBanner(
      input({ split: result("split", "s1", { ...split, n_holdout: 0, n_train: 2943 }) }),
    ).segments;
    expect(rows.cvOnly).toBe(true);
    expect(rows.holdout).toBeNull();
  });

  it("counts the column path through the lineage, collapsed groups included", () => {
    const columns = deriveBanner(input()).segments[1];
    expect(columns.from).toBe(11);
    expect(columns.to).toBe(10);
    expect(columns.method).toBe("residual");
  });

  it("before the design exists, counts the file's columns or the predictors", () => {
    const early = deriveBanner(
      input({ cohort: undefined, design: undefined }, { stages: {} } as Partial<ProjectView>),
    ).segments[1];
    expect(early.from).toBe(29);
    expect(early.unit).toBe("in the file");
    const roles = deriveBanner(input({ design: undefined })).segments[1];
    expect(roles.from).toBe(11);
    expect(roles.unit).toBe("predictors");
    expect(roles.to).toBeNull();
  });

  it("reports the best family by its cross-validated primary metric", () => {
    const res = deriveBanner(input()).segments[3];
    expect(res.metric).toBe("R²");
    expect(res.value).toBe(0.077);
    expect(res.family).toBe("elastic net");
    // Lower is better for an error metric.
    expect(bestModel({ ...fit, primary_metric: "rmse" })?.family).toBe("elastic_net");
    expect(formatMetric(-0.0391)).toBe("−0.039");
  });

  it("names how the families were ranked, never 'best' (audit ME-10, WP9)", () => {
    const res = deriveBanner(input()).segments[3];
    expect(res.summary).toBe("Result: R² 0.077 by cross-validation, highest R² for Elastic net.");
    const summary = (estimate: number) => ({
      mean: estimate,
      sd: 0.01,
      folds: [],
      estimate,
      estimator: "fold_mean" as const,
      se: null,
      ci_low: null,
      ci_high: null,
      repeats: 1,
      repeat_sd: null,
    });
    const binary: FitArtifact = {
      ...fit,
      task: "binary",
      primary_metric: "auc",
      metric_labels: { auc: "AUC" },
      ranking: "highest AUC",
      models: fit.models.map((m, i) => ({
        ...m,
        cv: { auc: summary([0.71, 0.74, 0.69][i] ?? 0) },
      })),
    };
    const auc = deriveBanner(input({ fit: result("fit", "f1", binary) })).segments[3];
    expect(auc.summary).toBe("Result: AUC 0.740 by cross-validation, highest AUC for Elastic net.");
    expect(auc.summary).not.toContain("best");
    // Lower is better for log loss, the multiclass primary: the lowest is the one named.
    const multiclass: FitArtifact = {
      ...fit,
      task: "multiclass",
      primary_metric: "log_loss",
      ranking: "lowest log loss",
      models: fit.models.map((m, i) => ({
        ...m,
        cv: { log_loss: summary([0.9, 0.8, 1.1][i] ?? 0) },
      })),
    };
    expect(bestModel(multiclass)?.family).toBe("elastic_net");
  });

  it("veils what an earlier answer made stale, segment by segment", () => {
    const stale = input({ fit: result("fit", "f1", fit, true) }, {
      stages: {
        cohort: status("cohort", "c1"),
        split: status("split", "s1"),
        design: status("design", "d2", "running"),
        fit: status("fit", "f2", "queued"),
      },
    } as Partial<ProjectView>);
    const veils = deriveBanner(stale).segments.map((sg) => sg.veil);
    expect(veils).toEqual(["fresh", "recomputing", "recomputing", "recomputing"]);
    expect(worstVeil("fresh", "stale", "recomputing")).toBe("stale");
  });

  it("marks the segment the open question acts on as now", () => {
    expect(deriveBanner(input()).now).toBe("result");
    const roles = deriveBanner(
      input({}, { interview: [step("purpose", "answered"), step("roles", "open")] }),
    );
    expect(roles.now).toBe("columns");
    expect(roles.nowLabel).toBe("column roles");
    // Waiting on a stage, the next question is still where the user is.
    const waiting = deriveBanner(
      input(
        {},
        {
          interview: [
            step("roles", "answered"),
            step("exclusions", "waiting", ["proposals"]),
            step("missing", "waiting", ["exclusions"]),
          ],
        },
      ),
    );
    expect(waiting.now).toBe("rows");
    expect(waiting.nowWaiting).toBe(true);
    // Everything answered: no marker.
    expect(
      deriveBanner(input({}, { interview: [step("substitution", "answered")] })).now,
    ).toBeNull();
  });

  it("never invents a number: nothing computed yet says what it waits for", () => {
    const empty = deriveBanner(
      input({ cohort: undefined, split: undefined, design: undefined, fit: undefined }, {
        summary: { ...input().view.summary, n_rows: null, n_cols: null },
        stages: { ingest: status("ingest", "i1", "running") },
        state: { ...input().view.state, models: null },
      } as Partial<ProjectView>),
    );
    const [rows, columns, models, res] = empty.segments;
    expect(rows.flow).toEqual([]);
    expect(rows.waiting).toBe("reading the file…");
    expect(columns.waiting).toBe("reading the file…");
    expect(models.waiting).toBe("not chosen yet");
    expect(res.waiting).toBe("after the fit");
  });

  it("before a cohort, counts the table as it stands: turned around, or combined per unit", () => {
    const none = { cohort: undefined, split: undefined, design: undefined, fit: undefined };
    const turned = result("oriented", "o1", { n_rows: 80, n_cols: 397, transposed: true } as never);
    const rows = deriveBanner(input({ ...none, oriented: turned })).segments[0];
    expect(rows.key === "rows" && rows.flow).toEqual([{ n: 80, label: "rows in the table" }]);
    const combined = result("working", "w1", { n_rows: 300, n_source_rows: 600 } as never);
    const both = deriveBanner(input({ ...none, oriented: turned, working: combined })).segments[0];
    expect(both.key === "rows" && both.flow).toEqual([{ n: 300, label: "rows in the table" }]);
    // A working table being recomputed is not read: the file's count stands until it is fresh.
    const stale = result("working", "w0", { n_rows: 300, n_source_rows: 600 } as never, false);
    const file = deriveBanner(input({ ...none, working: stale })).segments[0];
    expect(file.key === "rows" && file.flow).toEqual([{ n: 21849, label: "rows loaded" }]);
  });
});
