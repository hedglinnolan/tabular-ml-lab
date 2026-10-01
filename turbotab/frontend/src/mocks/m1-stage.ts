/**
 * The stage's mock API (dev:mock only): the NHANES project of the §9 journey, every number from
 * the real server (src/mocks/m1-stage-fixture.json, written by
 * docs/turbotab-next/m1/stage/capture_stage_fixture.py).
 *
 * It serves one project, `nhanes-m1`, fitted end to end: its view, the cohort / split / shelf /
 * design / fit / substitution artifacts, the previews of every captured option, and finding
 * evidence. It also emulates what the part 2 backend adds (M1_CONTRACT §12), so the stage can be
 * built against it now: storyboards (residual; partition), labeled marks on a cut, each model's
 * baseline and the concern when a family loses to it, the evidence route, and the refit band
 * (computed by the capture script, as §12.7 specifies; the mock only shortens its wait).
 * Requests for any other project fall through to the M0 handlers.
 */
import { http, HttpResponse, sse, type HttpHandler } from "msw";
import type {
  ConsequenceView,
  DistributionView,
  FitArtifact,
  Lineage,
  LineageView,
  Mark,
  PreviewResult,
  RelationshipView,
  RowFlowView,
  SubstitutionArtifact,
  TableFocusView,
} from "../api/m1-stage-types";
import type { Decision, DecisionRecord, ProjectSummary, ProjectView, Refusal, StageStatus } from "../api/schema";
import raw from "./m1-stage-fixture.json";

export const DEMO_PID = "nhanes-m1";

interface Captured {
  decision: Decision & Record<string, unknown>;
  status: number;
  body: PreviewResult;
  label?: string;
}

interface Fixture {
  previews: Record<string, Captured[]>;
  view: ProjectView;
  cohort: unknown;
  split: unknown;
  shelf: unknown;
  design: unknown;
  roles: unknown;
  findings: { findings: { id: string; affected_columns: string[]; summary: string }[]; basis: string };
  fit_prediction: FitArtifact;
  fit_inference: FitArtifact;
  baseline: { metric: string; value: number };
  substitution: (SubstitutionArtifact & { _seconds?: number })[];
  band: { n_boot: number; rows: number; seconds: number; families: Record<string, { ci_low: (number | null)[]; ci_high: (number | null)[] }> };
  density: { design: unknown; fit: FitArtifact; substitution: SubstitutionArtifact };
  evidence_inputs: Record<
    string,
    {
      table: { columns: string[]; rows: unknown[][] } | null;
      histogram: { column?: string; edges: number[]; counts: number[]; n_missing: number } | null;
      column: string | null;
    }
  >;
}

const F = raw as unknown as Fixture;

// ── small helpers ────────────────────────────────────────────────────────────

function ols(points: [number, number][]): [number, number] {
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (const [x, y] of points) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const n = points.length;
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  return [slope, (sy - slope * sx) / n];
}

function corr(points: [number, number][]): number {
  const n = points.length;
  const mx = points.reduce((a, p) => a + p[0], 0) / n;
  const my = points.reduce((a, p) => a + p[1], 0) / n;
  let sxy = 0,
    sxx = 0,
    syy = 0;
  for (const [x, y] of points) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
    syy += (y - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

// ── storyboards and marks, as the part 2 backend will build them (§12.1, §12.2) ──

/** Residual: fit each nutrient on energy → keep what energy does not explain → add back the average. */
function residualStory(views: ConsequenceView[]): ConsequenceView[] {
  return views.map((v) => {
    if (v.kind === "relationship") {
      const rel = v as RelationshipView;
      const [slope, intercept] = ols(rel.points_before);
      const mean = rel.points_after.reduce((a, p) => a + p[1], 0) / Math.max(1, rel.points_after.length);
      const residuals = rel.points_after.map(([x, y]) => [x, y - mean] as [number, number]);
      return {
        ...rel,
        story: [
          { label: "Fit each nutrient on energy", points: rel.points_before, r: rel.r_before, fit_line: { slope, intercept }, y_label: rel.y_label_before },
          { label: "Keep what energy does not explain", points: residuals, r: corr(residuals), fit_line: { slope: 0, intercept: 0 }, y_label: `${rel.y_label_before} residual` },
        ],
      };
    }
    if (v.kind === "distribution") {
      const d = v as DistributionView;
      const n = d.after.counts.reduce((a, c) => a + c, 0);
      const mean =
        d.after.counts.reduce((a, c, i) => a + c * ((d.after.edges[i]! + d.after.edges[i + 1]!) / 2), 0) / Math.max(1, n);
      return {
        ...d,
        story: [
          { label: "Fit each nutrient on energy", hist: d.before, x_label: d.before_label },
          {
            label: `${d.column} residuals, centered on 0`,
            hist: { ...d.after, edges: d.after.edges.map((e) => +(e - mean).toPrecision(4)) },
            x_label: `${d.column} residual`,
          },
        ],
      };
    }
    return v;
  });
}

/** Partition: each nutrient becomes its kcal first; then kcal itself is split off. */
function partitionStory(views: ConsequenceView[]): ConsequenceView[] {
  return views.map((v) => {
    if (v.kind !== "lineage") return v;
    const lv = v as LineageView;
    const after = lv.after;
    const frame: Lineage = clone(after);
    const other = frame.nodes.find((n) => n.id === "adj:kcal_from_other");
    if (other) {
      other.id = "adj:kcal";
      other.column = "kcal";
      other.label = "kcal";
      other.formula = null;
      const mx = frame.nodes.find((n) => n.id === "mx:kcal_from_other");
      if (mx) Object.assign(mx, { id: "mx:kcal", column: "kcal", label: "kcal" });
      frame.links = frame.links
        .filter((l) => !(l.target === "adj:kcal_from_other" && l.source !== "raw:kcal"))
        .map((l) => ({
          ...l,
          source: l.source.replace("kcal_from_other", "kcal"),
          target: l.target.replace("kcal_from_other", "kcal"),
          operation: l.target === "adj:kcal_from_other" ? "kept" : l.operation,
        }));
    }
    return { ...lv, story: [{ label: "Each nutrient becomes its kcal", lineage: frame }] };
  });
}

/** Labeled marks for a range rule (§12.2): each bound, by level when the rule is by level. */
function marksFor(decision: Record<string, unknown>): Mark[] {
  const rules = (decision.rules as { low: number | null; high: number | null; by: { ranges: Record<string, [number | null, number | null]> } | null }[]) ?? [];
  const out: Mark[] = [];
  for (const r of rules) {
    if (r.by) {
      for (const [level, [lo, hi]] of Object.entries(r.by.ranges)) {
        if (lo !== null) out.push({ value: lo, label: lo.toLocaleString("en-US"), group: level });
        if (hi !== null) out.push({ value: hi, label: hi.toLocaleString("en-US"), group: level });
      }
    } else {
      if (r.low !== null) out.push({ value: r.low, label: r.low.toLocaleString("en-US"), group: null });
      if (r.high !== null) out.push({ value: r.high, label: r.high.toLocaleString("en-US"), group: null });
    }
  }
  return out;
}

function withStory(c: Captured): PreviewResult {
  const body = clone(c.body);
  const d = c.decision;
  if (d.kind === "set_energy_adjustment" && d.method === "residual") body.views = residualStory(body.views);
  if (d.kind === "set_energy_adjustment" && d.method === "partition") body.views = partitionStory(body.views);
  if (d.kind === "set_exclusions") {
    const marks = marksFor(d);
    body.views = body.views.map((v) => (v.kind === "distribution" ? { ...v, marks } : v));
  }
  return body;
}

const sameDecision = (a: Record<string, unknown>, b: Record<string, unknown>) => {
  const norm = (d: Record<string, unknown>) => {
    const o: Record<string, unknown> = { ...d };
    if (Array.isArray(o.nutrients)) o.nutrients = [...(o.nutrients as string[])].sort();
    if (Array.isArray(o.models)) o.models = [...(o.models as string[])].sort();
    delete o.energy_column;
    delete o.log_transform;
    delete o.strata;
    return JSON.stringify(o, Object.keys(o).sort());
  };
  return norm(a) === norm(b);
};

function refusal(code: string, message: string, exits: Refusal["error"]["exits"] = []): Refusal {
  return { error: { code, message, exits } };
}

/** The roles preview, as the server draws it: every column, the predictors entering the matrix. */
function rolesPreview(roles: Record<string, string>): PreviewResult {
  const PREDICTOR = ["exposure", "covariate", "energy"];
  const nodes: Lineage["nodes"] = [];
  const links: Lineage["links"] = [];
  let n = 0;
  for (const [c, role] of Object.entries(roles)) {
    const r = role as Lineage["nodes"][number]["role"];
    nodes.push({ id: `raw:${c}`, column: c, lane: "raw", role: r, label: c, formula: null, group: null, count: 1 });
    if (!PREDICTOR.includes(role)) continue;
    n++;
    nodes.push({ id: `mx:${c}`, column: c, lane: "matrix", role: r, label: c, formula: null, group: null, count: 1 });
    links.push({ source: `raw:${c}`, target: `mx:${c}`, operation: "kept" });
  }
  const total = Object.keys(roles).length;
  return {
    kind: "set_roles",
    views: [
      {
        kind: "lineage",
        title: "Which columns enter the model",
        caption: `\`${n}\` of \`${total}\` columns enter the model.`,
        emphasis: [],
        before: null,
        after: { nodes, links, collapsed: false },
        story: [],
      },
    ],
    basis: "Counts on all 21,849 rows.",
    note: null,
    caution: null,
  };
}

// ── the project ──────────────────────────────────────────────────────────────

const ENERGY_NUTRIENTS = (F.view.state.energy_adjustment?.nutrients ?? []) as string[];

class DemoProject {
  view: ProjectView;
  version = 0;
  listeners = new Set<(type: string, data: unknown) => void>();
  bandPairs = new Set<string>();

  constructor() {
    this.view = clone(F.view);
    this.view.summary = { ...this.view.summary, id: DEMO_PID, name: "_tt_tmp_nhanes" } as ProjectSummary;
    // `?unfit` (review only): the journey one step earlier, models not chosen yet, nothing fitted.
    if (typeof location !== "undefined" && new URLSearchParams(location.search).has("unfit")) {
      const blocked = (stage: string) =>
        ({ ...this.view.stages[stage]!, status: "blocked", key: null, fresh: false, missing: ["models"] }) as StageStatus;
      this.view = {
        ...this.view,
        state: { ...this.view.state, models: null, substitution: null },
        decisions: this.view.decisions.filter((d) => d.decision.kind !== "select_models" && d.decision.kind !== "set_substitution"),
        stages: { ...this.view.stages, design: blocked("design"), fit: blocked("fit"), substitution: blocked("substitution") },
        interview: this.view.interview.map((i) =>
          i.key === "models" ? { ...i, status: "open", decision_id: null } : i.key === "substitution" ? { ...i, status: "waiting", decision_id: null, waiting_on: ["fit"] } : i,
        ),
      };
    }
  }

  emit(type: string, data: unknown) {
    this.listeners.forEach((fn) => fn(type, data));
  }

  status(stage: string): StageStatus {
    return this.view.stages[stage]!;
  }

  setStatus(stage: string, patch: Partial<StageStatus>) {
    const next = { ...this.status(stage), ...patch, updated_at: new Date().toISOString() } as StageStatus;
    this.view = { ...this.view, stages: { ...this.view.stages, [stage]: next } };
    this.emit("stage", next);
  }

  /** Recompute `stages` in order: running, then fresh at a new key after `ms`. */
  recompute(stages: string[], ms: number, progress = false) {
    this.version++;
    const v = this.version;
    stages.forEach((stage, i) => {
      this.setStatus(stage, {
        status: i === 0 ? "running" : "queued",
        fresh: false,
        progress: i === 0 && progress ? 0 : null,
        job_id: i === 0 && progress ? "mock-band" : null,
        cancelled: false,
      });
    });
    let t = 0;
    stages.forEach((stage, i) => {
      const share = ms / stages.length;
      if (progress && i === 0) {
        for (let k = 1; k < 5; k++) {
          window.setTimeout(() => this.version === v && this.setStatus(stage, { status: "running", progress: k / 5 }), t + (share * k) / 5);
        }
      }
      t += share;
      window.setTimeout(() => {
        if (this.version !== v) return;
        this.setStatus(stage, { status: "fresh", fresh: true, key: `${stage}-${v}`, progress: null });
        const nextStage = stages[i + 1];
        if (nextStage) this.setStatus(nextStage, { status: "running" });
      }, t);
    });
  }

  decide(d: Decision & Record<string, unknown>): ProjectView | Refusal {
    const state = { ...this.view.state } as ProjectView["state"] & Record<string, unknown>;
    let sentence: string;
    let downstream: string[];
    let ms: number;
    let progress = false;
    switch (d.kind) {
      case "set_substitution": {
        const pair = F.substitution.find((x) => x.donor === d.donor && x.recipient === d.recipient);
        if (!pair) return refusal("not_a_pair", `\`${d.donor}\` → \`${d.recipient}\` is not a substitution this design offers.`);
        const nBoot = Number(d.n_boot ?? 0);
        state.substitution = { donor: d.donor, recipient: d.recipient, step_kcal: d.step_kcal ?? 100, ...(nBoot ? { n_boot: nBoot } : {}) } as never;
        if (nBoot) this.bandPairs.add(`${d.donor}>${d.recipient}`);
        sentence = nBoot
          ? `An uncertainty band was added: each family was refit on \`${nBoot}\` bootstrap resamples of up to \`2,000\` training rows.`
          : `Energy is moved from \`${d.donor}\` to \`${d.recipient}\` in steps of \`${d.step_kcal ?? 100}\` kcal.`;
        downstream = ["substitution"];
        ms = nBoot ? 3200 : 600;
        progress = nBoot > 0;
        break;
      }
      case "set_energy_adjustment": {
        if (d.method !== "residual" && d.method !== "density") {
          return refusal("mock_only", "The mock fitted only the residual and density methods.");
        }
        state.energy_adjustment = { ...(state.energy_adjustment ?? {}), method: d.method } as never;
        sentence =
          d.method === "density"
            ? "Energy was adjusted by nutrient density: each nutrient is divided by `kcal`, and `kcal` leaves the model."
            : F.view.decisions.find((r) => r.decision.kind === "set_energy_adjustment")?.sentence ?? "";
        downstream = ["design", "fit", "substitution"];
        ms = 2400;
        break;
      }
      case "set_purpose":
        state.purpose = d.purpose as never;
        sentence =
          d.purpose === "inference"
            ? "The analysis was declared for `inference`: associations are estimated with their uncertainty."
            : "The analysis was declared for `prediction`: models are judged on rows they never saw.";
        downstream = ["design", "fit", "substitution"];
        ms = 2400;
        break;
      default:
        return refusal("mock_only", "The stage's mock records substitution, energy method and purpose only.");
    }
    const seq = Math.max(...this.view.decisions.map((r) => r.seq)) + 1;
    const record = {
      id: `mock-${seq}`,
      seq,
      at: new Date().toISOString(),
      note: null,
      sentence,
      decision: d,
    } as DecisionRecord;
    this.view = { ...this.view, state, decisions: [...this.view.decisions, record] };
    this.emit("decision", record);
    this.recompute(downstream, ms, progress);
    return this.view;
  }

  fit(): FitArtifact {
    const base =
      this.view.state.energy_adjustment?.method === "density"
        ? F.density.fit
        : this.view.state.purpose === "inference"
          ? F.fit_inference
          : F.fit_prediction;
    const fit = clone(base);
    // §12.6: each model carries the baseline, and says so in plain words when it loses to it.
    for (const m of fit.models) {
      m.baseline = { ...F.baseline, label: "the outcome's average" };
      const cv = m.cv[F.baseline.metric]?.mean;
      if (cv !== null && cv !== undefined && cv < F.baseline.value) {
        m.concerns = [
          ...m.concerns,
          `Predicts worse than the outcome's average: CV R² ${cv.toFixed(2).replace("-", "−")}.`,
        ];
      }
    }
    return fit;
  }

  substitution(): SubstitutionArtifact | null {
    const s = this.view.state.substitution;
    if (!s) return null;
    const density = this.view.state.energy_adjustment?.method === "density";
    let art: SubstitutionArtifact | undefined =
      density && F.density.substitution.donor === s.donor && F.density.substitution.recipient === s.recipient
        ? F.density.substitution
        : F.substitution.find((x) => x.donor === s.donor && x.recipient === s.recipient);
    if (!art) return null;
    art = clone(art);
    delete (art as { _seconds?: number })._seconds;
    // §12.7: no row-resampling band; the refit band only when it was asked for.
    const key = `${s.donor}>${s.recipient}`;
    const band = F.band;
    const nBoot = (s as { n_boot?: number }).n_boot ?? 0;
    const banded = nBoot > 0 && this.bandPairs.has(key) && key === "fat_total>carb" && !density;
    art.models = art.models.map((m) => ({
      ...m,
      ci_low: banded ? (band.families[m.family]?.ci_low ?? null) : null,
      ci_high: banded ? (band.families[m.family]?.ci_high ?? null) : null,
    }));
    art.carried = art.carried ?? [];
    art.band = banded
      ? { n_boot: band.n_boot, n_rows: band.rows, grouped_by: null, seconds: band.seconds, failed: 0 }
      : null;
    art.band_estimate = banded ? null : { n_boot: band.n_boot, seconds: band.seconds };
    if (nBoot > 0 && !banded) art.note = `${art.note} (The mock holds a refit band for fat_total → carb only.)`;
    if (density && key !== `${F.density.substitution.donor}>${F.density.substitution.recipient}`) {
      art.note = `${art.note} (The mock holds density curves for fat_total → carb only; this one is the residual capture.)`;
    }
    return art;
  }

  artifact(stage: string): unknown {
    if (this.view.stages[stage]?.status === "blocked") return null;
    const density = this.view.state.energy_adjustment?.method === "density";
    switch (stage) {
      case "cohort":
        return F.cohort;
      case "split":
        return F.split;
      case "shelf":
        return F.shelf;
      case "design":
        return density ? F.density.design : F.design;
      case "fit":
        return this.fit();
      case "substitution":
        return this.substitution();
      case "findings":
        return F.findings;
      case "roles":
        return F.roles;
      default:
        return null;
    }
  }

  preview(d: Decision & Record<string, unknown>): PreviewResult | Refusal | null {
    if (d.kind === "set_energy_adjustment" && d.method === "partition") {
      const nutrients = (d.nutrients as string[] | undefined) ?? [];
      const bad = nutrients.filter((n) => !ENERGY_NUTRIENTS.includes(n));
      if (bad.length) {
        const subset = ENERGY_NUTRIENTS.filter((n) => ["protein", "carb", "fat_total"].includes(n));
        return refusal(
          "method_not_applicable",
          `\`${bad[0]}\` carries no energy in a known unit, so \`kcal\` cannot be split by it.`,
          [
            {
              label: `Partition ${subset.join(", ")} instead`,
              decision: { ...d, nutrients: subset } as Decision,
            },
          ],
        );
      }
    }
    const group = Object.values(F.previews).flat();
    const hit = group.find((c) => c.status === 200 && sameDecision(c.decision, d));
    return hit ? withStory(hit) : null;
  }

  /**
   * Any other project (the Record's NHANES-shaped table): the captured preview of the same kind of
   * option, said to be the mock's stand-in; else the real server's answer for a choice with no
   * picture yet.
   */
  previewLike(d: Decision & Record<string, unknown>): PreviewResult | Refusal {
    const exact = this.preview(d);
    if (exact) return exact;
    const same = (c: Captured): boolean => {
      const x = c.decision as Decision & Record<string, unknown>;
      if (x.kind !== d.kind || c.status !== 200) return false;
      switch (d.kind) {
        case "set_energy_adjustment":
          return x.method === d.method && ((x.nutrients as string[]).length > 3) === (((d.nutrients as string[]) ?? []).length > 3);
        case "set_missing":
          return x.strategy === d.strategy;
        case "set_split":
          return x.holdout === d.holdout;
        case "select_models":
          return [...(x.models as string[])].sort().join() === [...(d.models as string[])].sort().join();
        case "set_exclusions": {
          const [a] = (x.rules as { low: number | null; high: number | null; by: unknown }[]) ?? [];
          const [b] = (d.rules as { low: number | null; high: number | null; by: unknown }[]) ?? [];
          if (!a || !b) return !a && !b;
          return !!a.by === !!b.by && (!!a.by || (a.low === b.low && a.high === b.high));
        }
        default:
          return false;
      }
    };
    if (d.kind === "set_roles") return rolesPreview(d.roles as Record<string, string>);
    const near = Object.values(F.previews).flat().find(same);
    if (near) {
      const body = withStory(near);
      return { ...body, note: `${body.note ? `${body.note} ` : ""}(Mock: the captured NHANES preview of this kind of option.)` };
    }
    return { kind: d.kind, views: [], basis: "Nothing is computed for this choice.", note: "Nothing about this choice can be shown on your data yet.", caution: null };
  }

  evidence(fid: string): PreviewResult | null {
    const f = F.findings.findings.find((x) => x.id === fid);
    if (!f) return null;
    const basis = "All 21,849 loaded rows.";
    if (fid === "pack::dietary::energy_adjustment") {
      const residual = F.previews.energy_adjustment!.find((c) => c.decision.method === "residual")!.body;
      const rel = residual.views.find((v) => v.kind === "relationship") as RelationshipView;
      const asRecorded: RelationshipView = {
        ...rel,
        title: `${rel.y_label_before} against ${rel.x_label}, as recorded`,
        caption: `\`${rel.y_label_before}\` rises with \`${rel.x_label}\` at r ${rel.r_before?.toFixed(2)} on the training rows.`,
        points_after: rel.points_before,
        y_label_after: rel.y_label_before,
        r_after: rel.r_before,
      };
      const lineage: LineageView = {
        kind: "lineage",
        title: "Where an energy adjustment acts",
        caption: "Each nutrient would be rewritten in the adjusted lane.",
        emphasis: f.affected_columns,
        before: null,
        after: (F.previews.energy_adjustment!.find((c) => c.decision.method === "none")!.body.views.find((v) => v.kind === "lineage") as LineageView).after,
        story: [],
      };
      return { kind: "evidence", views: [asRecorded, lineage], basis: residual.basis, note: null, caution: null };
    }
    if (fid === "pack::dietary::implausible_intake") {
      const ex = F.previews.exclusions!.find((c) => sameDecision(c.decision, { kind: "set_exclusions", rules: [{ kind: "range", column: "kcal", low: 500, high: 5000, by: null, reason: "implausible intakes (sex-neutral 500–5,000 kcal a day)" }] }));
      const dist = ex?.body.views.find((v) => v.kind === "distribution") as DistributionView | undefined;
      const flow = ex?.body.views.find((v) => v.kind === "row_flow") as RowFlowView | undefined;
      if (!dist || !flow) return null;
      const evidence: DistributionView = {
        ...dist,
        title: "`kcal` on every loaded row",
        caption: "`501` rows fall below `500` or above `5,000` kcal a day.",
        after: dist.before,
        after_label: dist.before_label,
        marks: [
          { value: 500, label: "500", group: null },
          { value: 5000, label: "5,000", group: null },
        ],
      };
      const rows: RowFlowView = { ...flow, title: "Rows, before any exclusion", caption: "", before: flow.before, after: flow.before };
      return { kind: "evidence", views: [evidence, rows], basis, note: null, caution: null };
    }
    const input = F.evidence_inputs[fid];
    const views: ConsequenceView[] = [];
    if (input?.table) {
      const cols = input.table.columns.filter((c) => c !== "SEQN" || f.affected_columns.includes("SEQN"));
      const idCol = input.table.columns.indexOf("SEQN");
      const rows = input.table.rows.map((r, i) => {
        const values = Object.fromEntries(cols.map((c) => [c, r[input.table!.columns.indexOf(c)]]));
        return { row_id: idCol >= 0 ? Number(r[idCol]) || i : i, before: values, after: values };
      });
      const table: TableFocusView = {
        kind: "table_focus",
        title: `${f.affected_columns.map((c) => `\`${c}\``).join(" and ")}, first rows`,
        caption: f.summary,
        emphasis: f.affected_columns,
        columns_before: cols,
        columns_after: cols,
        rows,
        changed: [],
        n_affected_columns: cols.length,
        story: [],
      };
      views.push(table);
    }
    if (input?.histogram && input.column) {
      const h = { edges: input.histogram.edges, counts: input.histogram.counts, n_missing: input.histogram.n_missing };
      views.push({
        kind: "distribution",
        title: `\`${input.column}\` on every loaded row`,
        caption: `\`${h.counts.reduce((a, c) => a + c, 0).toLocaleString("en-US")}\` values; \`${h.n_missing.toLocaleString("en-US")}\` blank.`,
        emphasis: [input.column],
        column: input.column,
        before: h,
        after: h,
        before_label: input.column,
        after_label: input.column,
        marks: [],
        story: [],
      });
    }
    if (!views.length) {
      const lineage = (F.design as { lineage: Lineage }).lineage;
      views.push({
        kind: "lineage",
        title: "Every column entering the model",
        caption: "No column has the survey-design role.",
        emphasis: [],
        before: null,
        after: lineage,
        story: [],
      });
    }
    return { kind: "evidence", views, basis, note: null, caution: null };
  }
}

// ── handlers ────────────────────────────────────────────────────────────────

export function m1StageHandlers(): HttpHandler[] {
  const p = new DemoProject();
  const base = `/api/projects/${DEMO_PID}`;
  const later = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const missing = (message: string) =>
    HttpResponse.json({ error: { code: "not_found", message, exits: [] } }, { status: 404 });
  return [
    http.get(base, () => HttpResponse.json(p.view)),
    http.post(`${base}/decisions`, async ({ request }) => {
      const out = p.decide((await request.json()) as Decision & Record<string, unknown>);
      return "error" in out ? HttpResponse.json(out, { status: 409 }) : HttpResponse.json(out);
    }),
    http.post(`${base}/preview`, async ({ request }) => {
      const d = (await request.json()) as Decision & Record<string, unknown>;
      await later(40 + Math.random() * 60); // the real route answers in 2–80 ms on NHANES
      const out = p.preview(d);
      if (!out)
        return HttpResponse.json(
          { error: { code: "not_captured", message: "The mock holds no preview for this option.", exits: [] } },
          { status: 422 },
        );
      return "error" in out ? HttpResponse.json(out, { status: 409 }) : HttpResponse.json(out);
    }),
    http.get(`${base}/findings/:fid/evidence`, async ({ params }) => {
      await later(30);
      const out = p.evidence(decodeURIComponent(String(params.fid)));
      return out ? HttpResponse.json(out) : missing("No such finding.");
    }),
    http.get(`${base}/stages/:stage`, ({ params }) => {
      const stage = String(params.stage);
      const status = p.view.stages[stage];
      if (!status) return missing("No such stage.");
      return HttpResponse.json({
        stage,
        key: status.key,
        fresh: status.status === "fresh",
        status: status.status,
        artifact: p.artifact(stage),
      });
    }),
    // Every other project (the Record's mock table): previews and evidence from the capture.
    http.post("/api/projects/:pid/preview", async ({ request }) => {
      const d = (await request.json()) as Decision & Record<string, unknown>;
      await later(40 + Math.random() * 60);
      const out = p.previewLike(d);
      return "error" in out ? HttpResponse.json(out, { status: 409 }) : HttpResponse.json(out);
    }),
    http.get("/api/projects/:pid/findings/:fid/evidence", async ({ params }) => {
      await later(30);
      const out = p.evidence(decodeURIComponent(String(params.fid)));
      return HttpResponse.json(
        out ?? { kind: "evidence", views: [], basis: "The mock holds no evidence for this finding.", note: null },
      );
    }),
    http.post(`${base}/jobs/:jid/cancel`, ({ params }) => {
      p.version++;
      p.setStatus("substitution", { status: "stale", cancelled: true, progress: null });
      return HttpResponse.json({ job_id: String(params.jid), state: "cancelled" });
    }),
    sse(`${base}/events`, ({ client, request }) => {
      const send = (type: string, data: unknown) => {
        try {
          client.send({ event: type, data } as never);
        } catch {
          p.listeners.delete(send);
        }
      };
      p.listeners.add(send);
      request.signal.addEventListener("abort", () => p.listeners.delete(send));
    }),
  ];
}
