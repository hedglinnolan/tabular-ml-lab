/**
 * The generic question (pure): what an open step without a bespoke component shows and records,
 * composed from the server's own words — the teaching entry's question and one line, and the
 * options the step's stage serves (the follow-up candidates, the groupings the roles read, the
 * estimand and adjustment cards, the causal card, the time-varying card, the form card and the
 * declared modifiers), each with its labels
 * where the server gives them (customary in a field, sound for the purpose; north star 5).
 *
 * Nothing here invents an answer: an option's decision is the one the card offers, filled with
 * the decision's own defaults; detection is said beside an option, never pre-selected. A press
 * the server refuses answers at the option with its exits (the Record's refusal note), so a
 * question this module composes partly is still answerable: the server's exits finish it.
 *
 * Every Router key has a renderer: a bespoke component in the Record (BESPOKE_KEYS) or a composer
 * here (COMPOSERS). generic.test.tsx fails when interview.py gains a key that has neither.
 */
import type { InterviewStep, QuestionKey, TeachingEntry } from "../../../api/m1-types";
import type { RolesArtifact, ProposalsArtifact } from "../../../api/m1-types";
import type {
  AdjustmentCard,
  CausalDesignArtifact,
  CovariateAnswers,
  EstimandCard,
  FormsArtifact,
  SetCausal,
  SetClusters,
  SetEstimand,
  SetExposureForm,
  SetModification,
  SetTimeVarying,
  TimeVaryingArtifact,
} from "../../../api/m3-types";
import type { ColumnInfo, Decision, ProjectState, TargetInfoArtifact } from "../../../api/schema";
import type { Grammar } from "../Question";

// ── the shape every composer returns ────────────────────────────────────────

/** North star 5: where the field uses an option, and whether it is sound for the purpose. */
export interface TwoLabels {
  customary?: string;
  source?: string;
  sound?: string;
  verdict?: string;
}

export type FieldValues = Record<string, string[]>;

export interface GenericOption {
  key: string;
  label: string;
  /** The consequence, in the app's voice (backticks mark data). */
  line: string;
  /** What pressing records, given the fields' values. Null: nothing to record from here. */
  build: ((values: FieldValues) => Decision) | null;
  /** Not applicable here: the reason, said in place of nothing happening. */
  na?: string;
  tags?: { text: string; tone?: "usual" | "detected" | "suggested" | "na" | "badge" }[];
  labels?: TwoLabels;
  /** Columns the option names, said as data chips. */
  chips?: string[];
  /** A second line: why the option stands as it does (the server's reason). */
  note?: string;
}

/** A choice that shapes the options' decisions (the effect, the measure, the assumptions). */
export interface GenericField {
  name: string;
  label: string;
  kind: "one" | "many";
  choices: { value: string; label: string; line?: string; status?: string }[];
  initial: string[];
  /** The options it shapes; every option when absent. */
  appliesTo?: string[];
}

/** Answers asked of each of several columns, recorded together (the adjustment's unguessed). */
export interface GenericMatrix {
  rows: string[];
  fields: { name: string; label: string; choices: { value: string; label: string }[] }[];
  build: (answers: Record<string, Record<string, string>>) => Decision | null;
  recordLabel: string;
}

export interface GenericQuestion {
  grammar: Grammar;
  /** Lines from the step's own data (their data before theory, §11.6). */
  data: string[];
  options: GenericOption[];
  fields: GenericField[];
  matrix?: GenericMatrix;
  /** The stage whose artifact the options come from, while it is not read yet. */
  waiting?: string;
  /** Said when the composer has nothing to offer (a key this version does not compose). */
  note?: string;
}

export interface ComposeContext {
  key: QuestionKey;
  step: InterviewStep;
  state: ProjectState;
  entry?: TeachingEntry | undefined;
  targetInfo?: TargetInfoArtifact | null | undefined;
  roles?: RolesArtifact | null | undefined;
  proposals?: ProposalsArtifact | null | undefined;
  causalDesign?: CausalDesignArtifact | null | undefined;
  timeVarying?: TimeVaryingArtifact | null | undefined;
  /** FORM: the form question's card (the `forms` stage). */
  forms?: FormsArtifact | null | undefined;
  /** The table's columns as it stands the right way round (no `__row_id`). */
  columns?: ColumnInfo[] | undefined;
}

// ── the keys and their renderers ─────────────────────────────────────────────

/** Keys the Record renders with a bespoke component (Record.tsx `ask`, and the seal's opening). */
export const BESPOKE_KEYS: readonly QuestionKey[] = [
  "lens",
  "orientation",
  "target",
  "event",
  "task",
  "purpose",
  "grain",
  "repeat_kind",
  "unit",
  "aggregation",
  "temporal",
  "roles",
  "survey",
  "exclusions",
  "missing",
  "split",
  "energy_adjustment",
  "models",
  "substitution",
  "open_seal",
];

const taught = (entry: TeachingEntry | undefined, value: string) =>
  entry?.options.find((o) => o.value === value);
const tick = (c: string) => `\`${c}\``;
const NUMERIC = new Set(["numeric", "integer"]);

function listing(items: string[], limit = 4): string {
  const shown = items.slice(0, limit).map(tick);
  const rest = items.length - shown.length;
  if (rest > 0) return `${shown.join(", ")} and ${rest} more`;
  if (shown.length <= 1) return shown.join("");
  return `${shown.slice(0, -1).join(", ")} and ${shown.at(-1)}`;
}

function teachingOptions(ctx: ComposeContext): GenericOption[] {
  return (ctx.entry?.options ?? []).map((o) => ({
    key: o.value,
    label: o.label,
    line: o.consequence,
    build: null,
  }));
}

// ── the study design (P0.6, crosswalk disagreement 10) ────────────────────────

type SetDesign = Extract<Decision, { kind: "set_design" }>;
const DESIGN_LABELS: Record<NonNullable<SetDesign["design"]>, string> = {
  observational: "Observed as they were",
  parallel_trial: "Randomized groups",
  cluster_randomized_trial: "Randomized by group",
  case_control: "Sampled by outcome",
  matched_sets: "Matched sets",
  crossover: "Crossover trial",
  repeated_measures_trial: "Trial, repeated outcomes",
};

/** Observational is stated until answered; every other design is a named value the server
 *  refuses as "Not available yet", with its reason and the observational exit (seam guard 6). */
function design(ctx: ComposeContext): GenericQuestion {
  const keys = Object.keys(DESIGN_LABELS) as NonNullable<SetDesign["design"]>[];
  return {
    grammar: "choice",
    data: ctx.step.reason ? [ctx.step.reason] : [],
    fields: [],
    options: keys.map((key) => ({
      key,
      label: taught(ctx.entry, key)?.label ?? DESIGN_LABELS[key],
      line:
        taught(ctx.entry, key)?.consequence ??
        (key === "observational"
          ? "Confounders are adjusted; the results are worded as associations."
          : "Not available yet: the reason and the way forward are shown when chosen."),
      build: (): SetDesign => ({ kind: "set_design", design: key }),
    })),
  };
}

// ── the follow-up (WP17, RO-03) ──────────────────────────────────────────────

function effectiveTask(ctx: ComposeContext): string | null {
  if (ctx.state.task) return ctx.state.task;
  return ctx.targetInfo && ctx.targetInfo.column === ctx.state.target ? ctx.targetInfo.task : null;
}

function followUp(ctx: ComposeContext): GenericQuestion {
  const target = ctx.state.target;
  if (!target) return { grammar: "fact", data: [], options: [], fields: [], waiting: "target_info" };
  const ti = ctx.targetInfo?.column === target ? ctx.targetInfo : null;
  const candidates = ti?.follow_up ?? [];
  const task = effectiveTask(ctx);
  if (task === "time_to_event") {
    // A time to event's follow-up must be named: the columns that read as one lead, then every
    // other numeric column (a name never decides; the user knows which column it is).
    const named = candidates.map((c) => c.column);
    const others = (ctx.columns ?? [])
      .filter((c) => NUMERIC.has(c.dtype) && c.name !== target && !named.includes(c.name))
      .map((c) => c.name);
    const options: GenericOption[] = [...named, ...others].map((column) => {
      const cand = candidates.find((c) => c.column === column);
      return {
        key: `time:${column}`,
        label: column,
        line: cand
          ? `Each row is followed for its \`${column}\`: ${cand.min} to ${cand.max}.`
          : `Each row is followed for its \`${column}\`.`,
        tags: cand ? [{ text: "reads as follow-up", tone: "detected" }] : undefined,
        build: () => ({
          kind: "set_follow_up",
          column: target,
          time_column: column,
          entry_column: null,
          landmark: null,
          horizon: null,
          prediction_horizon: null,
        }),
      };
    });
    return {
      grammar: "fact",
      data: [`${tick(target)} is a time to event: which column holds each row's follow-up?`],
      options,
      fields: [],
    };
  }
  const same = taught(ctx.entry, "same");
  const varies = taught(ctx.entry, "varies");
  const data = candidates.map((c) =>
    c.varies
      ? `${tick(c.column)} runs from ${c.min} to ${c.max}: follow-up that ended at different times.`
      : `${tick(c.column)} is ${c.min} on every row.`,
  );
  return {
    grammar: "fact",
    data,
    fields: [],
    options: [
      {
        key: "same",
        label: same?.label ?? "Same for everyone",
        line: same?.consequence ?? "The yes/no outcome stands, counted over one period.",
        build: () => ({ kind: "set_censoring", column: target, acknowledged: false }),
      },
      {
        key: "varies",
        label: varies?.label ?? "Follow-up varies",
        line: varies?.consequence ?? "Analyzed as a time to event.",
        tags: candidates.some((c) => c.varies)
          ? [{ text: "the data suggest it", tone: "suggested" }]
          : undefined,
        build: () => ({ kind: "set_task", column: target, task: "time_to_event" }),
      },
    ],
  };
}

// ── the grouping above the person (WP17, RO-08) ──────────────────────────────

/** Columns that read as a group of participants: as estimand.cluster_candidates reads them. */
export function clusterCandidates(ctx: ComposeContext): string[] {
  const recorded = (ctx.state.roles ?? {}) as Record<string, string>;
  const unit = ctx.state.grain?.id_column ?? null;
  const found = Object.entries(recorded)
    .filter(([, r]) => r === "cluster")
    .map(([c]) => c);
  for (const p of ctx.roles?.columns ?? []) {
    if (p.proposed === "cluster" && (recorded[p.column] ?? "cluster") === "cluster")
      found.push(p.column);
  }
  return [...new Set(found)].filter((c) => c !== ctx.state.target && c !== unit);
}

function clusters(ctx: ComposeContext): GenericQuestion {
  if (!ctx.roles) return { grammar: "fact", data: [], options: [], fields: [], waiting: "roles" };
  const found = clusterCandidates(ctx);
  const inference = ctx.state.purpose === "inference";
  const set = (column: string | null, adjust: SetClusters["adjust"]): SetClusters => ({
    kind: "set_clusters",
    column,
    adjust,
    acknowledged: column === null && found.length > 0,
    none_of: [],
  });
  const options: GenericOption[] = [];
  for (const c of found) {
    if (inference) {
      const fe = taught(ctx.entry, "fixed_effects");
      const co = taught(ctx.entry, "cluster_only");
      options.push(
        {
          key: `fixed_effects:${c}`,
          label: `${fe?.label ?? "Adjust and cluster"}: ${c}`,
          line: fe?.consequence ?? "Each group gets its own intercept; intervals cluster by it.",
          build: () => set(c, "fixed_effects"),
        },
        {
          key: `cluster_only:${c}`,
          label: `${co?.label ?? "Cluster only"}: ${c}`,
          line: co?.consequence ?? "Intervals cluster by the group.",
          build: () => set(c, "cluster_only"),
        },
      );
    } else {
      options.push({
        key: `group:${c}`,
        label: `Grouped by ${c}`,
        line: "Validation keeps each group's rows together; the split can hold out whole groups.",
        build: () => set(c, null),
      });
    }
  }
  const none = taught(ctx.entry, "none");
  options.push({
    key: "none",
    label: none?.label ?? "No grouping",
    line: none?.consequence ?? "Every participant is analyzed as independent.",
    build: () => set(null, null),
  });
  const reasons = new Map((ctx.roles.columns ?? []).map((p) => [p.column, p.reason]));
  return {
    grammar: "fact",
    data: found.map((c) => `${tick(c)} reads as a grouping: ${reasons.get(c) ?? "its role says so"}.`),
    options,
    fields: [],
  };
}

// ── the exposure and its effect (WP17, MODELING_SEQUENCE §1 step 2) ──────────

function estimand(ctx: ComposeContext): GenericQuestion {
  const card: EstimandCard | null | undefined = ctx.proposals?.estimand;
  if (!card) return { grammar: "choice", data: [], options: [], fields: [], waiting: "proposals" };
  const fitted = card.measures
    .filter((m) => m.fitted)
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
  const named = card.measures.filter((m) => !m.fitted);
  const energy = card.exposures.filter((e) => e.energy_contrast).map((e) => e.column);
  const family = card.family;
  const familyFitted = (family?.measures ?? []).filter((m) => m.fitted);
  const base = (v: FieldValues): Omit<SetEstimand, "exposure" | "family" | "contrast" | "measure"> => ({
    kind: "set_estimand",
    effect: (v.effect?.[0] as SetEstimand["effect"]) ?? "total",
    multiplicity: null,
    multiplicity_acknowledged: false,
  });
  const options: GenericOption[] = card.exposures.map((e) => ({
    key: e.column,
    label: e.column,
    line: e.energy_contrast
      ? "Its effect with total energy in view: a substitution or an addition."
      : "Its effect on the outcome, adjusted as the next question decides.",
    build: (v) => ({
      ...base(v),
      exposure: e.column,
      family: false,
      contrast: e.energy_contrast
        ? ((v.contrast?.[0] as SetEstimand["contrast"]) ?? null)
        : null,
      measure: (v.measure?.[0] ?? fitted[0]?.measure ?? "mean_difference") as SetEstimand["measure"],
    }),
  }));
  if (family) {
    options.push({
      key: "__family",
      label: `Every exposure in turn (${family.n})`,
      line: family.consequence,
      build: (v) => ({
        ...base(v),
        exposure: null,
        family: true,
        contrast: family.energy_contrast
          ? ((v.contrast?.[0] as SetEstimand["contrast"]) ?? null)
          : null,
        measure: (v.family_measure?.[0] ??
          familyFitted[0]?.measure ??
          "mean_difference") as SetEstimand["measure"],
        multiplicity: (v.multiplicity?.[0] as SetEstimand["multiplicity"]) ?? null,
      }),
    });
  }
  const fields: GenericField[] = [
    {
      name: "effect",
      label: "Effect",
      kind: "one",
      choices: card.effects.map((c) => ({
        value: c.effect ?? "total",
        label: c.label,
        line: c.consequence,
      })),
      initial: [card.effects[0]?.effect ?? "total"],
    },
  ];
  if (card.contrasts.length && (energy.length || family?.energy_contrast)) {
    fields.push({
      name: "contrast",
      label: "Energy",
      kind: "one",
      choices: card.contrasts.map((c) => ({
        value: c.contrast ?? "substitution",
        label: c.label,
        line: c.consequence,
      })),
      // No pre-selection of an estimand: the first press without a contrast is answered by the
      // server ("which contrast?"), with both as exits.
      initial: [],
      appliesTo: [...energy, ...(family?.energy_contrast ? ["__family"] : [])],
    });
  }
  if (fitted.length) {
    fields.push({
      name: "measure",
      label: "Measure",
      kind: "one",
      choices: fitted.map((m) => ({ value: m.measure, label: m.label, line: m.reason })),
      initial: [fitted[0]!.measure],
      appliesTo: card.exposures.map((e) => e.column),
    });
  }
  if (family && familyFitted.length) {
    fields.push({
      name: "family_measure",
      label: "Measure",
      kind: "one",
      choices: familyFitted.map((m) => ({ value: m.measure, label: m.label, line: m.reason })),
      initial: [familyFitted[0]!.measure],
      appliesTo: ["__family"],
    });
  }
  if (family?.multiplicity) {
    fields.push({
      name: "multiplicity",
      label: "Multiplicity",
      kind: "one",
      choices: family.multiplicity.options.map((o) => ({
        value: o.key,
        label: o.label,
        line: o.sound.reason,
      })),
      initial: [],
      appliesTo: ["__family"],
    });
  }
  const data: string[] = [];
  if (named.length)
    data.push(
      `Named, not fitted here: ${named.map((m) => m.label).join("; ")}. Only fitted measures are offered.`,
    );
  if (!card.exposures.length && !family)
    data.push("No predictor is settled as an exposure yet; the card above asks about the roles.");
  return { grammar: "choice", data, options, fields };
}

// ── the adjustment set (WP17, MODELING_SEQUENCE §1 step 3) ───────────────────

const ANSWER3 = [
  { value: "yes", label: "yes" },
  { value: "no", label: "no" },
  { value: "unknown", label: "not sure" },
];
const ASKED = ["causes_exposure", "causes_outcome", "after_exposure"] as const;

function covariate(answers: Record<string, string>): CovariateAnswers {
  return {
    causes_exposure: answers.causes_exposure as CovariateAnswers["causes_exposure"],
    causes_outcome: answers.causes_outcome as CovariateAnswers["causes_outcome"],
    after_exposure: answers.after_exposure as CovariateAnswers["after_exposure"],
    instrument: false,
    proxy: false,
    keep: false,
    acknowledged: false,
    further: false,
    confounds_mediator: null,
    interacts: null,
    interaction_attested: false,
  };
}

function adjustment(ctx: ComposeContext): GenericQuestion {
  const card: AdjustmentCard | null | undefined = ctx.proposals?.adjustment;
  const spec = ctx.state.estimand;
  const exposure = spec?.family ? "*" : spec?.exposure;
  if (!card || (exposure && card.exposure !== exposure))
    return { grammar: "fact", data: [], options: [], fields: [], waiting: "proposals" };
  const options: GenericOption[] = card.groups
    .filter((g) => g.decision)
    .map((g) => ({
      key: g.key,
      label: g.label,
      line: `Answered as the field's guess: each ${g.derived_words ?? "covariate"}.`,
      chips: g.columns,
      tags: g.derived_words ? [{ text: g.derived_words, tone: "detected" as const }] : undefined,
      note: g.estimand_note ?? g.reason,
      build: () => g.decision as unknown as Decision,
    }));
  const unguessed = card.groups.find((g) => !g.decision)?.columns ?? [];
  const matrix: GenericMatrix | undefined = unguessed.length
    ? {
        rows: unguessed,
        fields: ASKED.map((name) => ({
          name,
          label: card.questions[name] ?? name,
          choices: ANSWER3,
        })),
        recordLabel: "Record these answers",
        build: (answers) => {
          const complete = Object.entries(answers).filter(([c, a]) =>
            unguessed.includes(c) && ASKED.every((f) => a[f]),
          );
          if (!complete.length) return null;
          return {
            kind: "set_adjustment",
            exposure: card.exposure,
            answers: Object.fromEntries(complete.map(([c, a]) => [c, covariate(a)])),
          };
        },
      }
    : undefined;
  const answered = Object.entries(card.answered);
  const data = answered.length
    ? [
        `Answered so far: ${answered
          .slice(0, 6)
          .map(([c, d]) => `${tick(c)} ${d.words}`)
          .join("; ")}${answered.length > 6 ? `; and ${answered.length - 6} more` : ""}.`,
      ]
    : [];
  if (card.estimand_note) data.push(card.estimand_note);
  return { grammar: "fact", data, options, fields: [], matrix };
}

// ── the causal lane ──────────────────────────────────────────────────────────

function causal(ctx: ComposeContext): GenericQuestion {
  const design = ctx.causalDesign;
  if (!design) return { grammar: "choice", data: [], options: [], fields: [], waiting: "causal_design" };
  const exposure = design.exposure;
  const decision = (method: SetCausal["method"], v: FieldValues): SetCausal => ({
    kind: "set_causal",
    exposure,
    method,
    learner:
      method === "none" || method === "pds_lasso"
        ? null
        : ((v.learner?.[0] || null) as SetCausal["learner"]),
    population: ((v.population?.[0] as SetCausal["population"]) ?? "all") || "all",
    folds: 5,
    repetitions: 5,
    seed: 0,
    assumptions: method === "none" ? [] : ((v.assumptions ?? []) as SetCausal["assumptions"]),
    trim: null,
    acknowledged: false,
    sample_only: false,
    complete_rows: false,
  });
  if (!design.offered) {
    return {
      grammar: "choice",
      data: design.reason ? [design.reason] : [],
      fields: [],
      options: [
        {
          key: "none",
          label: taught(ctx.entry, "none")?.label ?? "The primary model only",
          line: taught(ctx.entry, "none")?.consequence ?? "No causal estimate beside the primary.",
          build: (v) => decision("none", v),
        },
      ],
    };
  }
  const estimators = design.options.filter((o) => o.key !== "none").map((o) => o.key);
  const options: GenericOption[] = design.options.map((o) => ({
    key: o.key,
    label: o.label,
    line: taught(ctx.entry, o.key)?.consequence ?? o.sound.reason,
    labels: {
      customary: `${o.customary.text} (${o.customary.field})`,
      source: o.customary.source,
      sound: o.sound.reason,
      verdict: `${o.sound.verdict} for ${o.sound.purpose}`,
    },
    tags: o.key === design.recommended ? [{ text: "ranks first", tone: "usual" }] : undefined,
    build: (v) => decision(o.key as SetCausal["method"], v),
  }));
  const fields: GenericField[] = [
    {
      name: "assumptions",
      label: "Declared before any estimate",
      kind: "many",
      choices: design.assumptions.map((a) => ({
        value: a.key,
        label: a.label,
        line: `${a.statement} ${a.diagnostic}`,
        status: a.status,
      })),
      initial: [],
      appliesTo: estimators,
    },
    {
      name: "learner",
      label: "Learner",
      kind: "one",
      choices: [
        { value: "", label: "default for this size" },
        { value: "linear", label: "linear" },
        { value: "lasso", label: "lasso" },
        { value: "nuisance_forest", label: "random forest" },
        { value: "untuned_boosted_trees", label: "boosted trees, untuned" },
      ],
      initial: [""],
      appliesTo: estimators.filter((k) => k !== "pds_lasso"),
    },
  ];
  if (design.exposure_kind === "binary") {
    fields.push({
      name: "population",
      label: "Whose effect",
      kind: "one",
      choices: [
        { value: "all", label: "everyone (average effect)" },
        { value: "exposed", label: "the exposed" },
      ],
      initial: ["all"],
      appliesTo: ["tmle", "dml_irm"],
    });
  }
  const data = [design.leash, design.positivity.violated ? (design.positivity.reason ?? "") : ""]
    .filter((x): x is string => !!x);
  return { grammar: "choice", data, options, fields };
}

// ── the time-varying exposure (V2 causal row) ────────────────────────────────

function timeVarying(ctx: ComposeContext): GenericQuestion {
  const art = ctx.timeVarying;
  if (!art) return { grammar: "choice", data: [], options: [], fields: [], waiting: "time_varying" };
  const exposure = art.exposure ?? ctx.state.estimand?.exposure ?? "";
  const lane = ctx.state.time_varying;
  const lanePart = (method: SetTimeVarying["method"], v: FieldValues): SetTimeVarying => ({
    kind: "set_time_varying",
    exposure,
    method,
    // Never declared for the user: unanswered, the ordering is recorded as not known, and the
    // server's leash for that answer applies.
    ordering: ((v.ordering?.[0] as SetTimeVarying["ordering"]) ??
      "unknown") as SetTimeVarying["ordering"],
    confounders: [],
    baseline: [],
    censoring: null,
    pattern: "switches",
    summary: "cumulative",
    truncation: null,
    simulations: null,
    bootstrap: null,
    acknowledged: false,
    ordering_acknowledged: false,
    diagnostics_seen: null,
  });
  // After the weights' diagnostics were read, the truncation is declared on them.
  const truncation = art.diagnostics?.truncation ?? [];
  if (lane && lane.exposure === exposure && lane.method === "msm_iptw" && truncation.length) {
    return {
      grammar: "choice",
      data: [...(art.diagnostics?.concerns ?? []), art.withheld ?? ""].filter((x) => !!x),
      fields: [],
      options: truncation.map((t) => ({
        key: `truncation:${t.key}`,
        label: t.label,
        line: `Weights the outcome model reads: mean ${t.summary.mean.toFixed(2)}, max ${t.summary.max.toFixed(1)}.`,
        labels: { customary: t.customary, sound: t.sound },
        build: () => ({ ...(lane as unknown as SetTimeVarying), kind: "set_time_varying", truncation: t.key }),
      })),
    };
  }
  const options: GenericOption[] = [...art.options]
    .sort((a, b) => a.order - b.order)
    .map((o) => ({
      key: o.key,
      label: o.label,
      line: taught(ctx.entry, o.key)?.consequence ?? o.sound,
      labels: { customary: o.customary, sound: o.sound },
      tags: o.rung === "recommended" ? [{ text: "recommended", tone: "usual" }] : undefined,
      build: (v) => lanePart(o.key as SetTimeVarying["method"], v),
    }));
  const s = art.setting;
  const data: string[] = [];
  if (s)
    data.push(
      `${tick(s.exposure)} changes within ${s.exposure_changers.toLocaleString("en-US")} of ${s.units.toLocaleString("en-US")} units over ${s.time_points} time points of ${tick(s.time_column)}.`,
    );
  if (art.affected.length) data.push(`Affected by earlier exposure: ${listing(art.affected)}.`);
  if (art.withheld) data.push(art.withheld);
  return {
    grammar: "choice",
    data,
    options,
    fields: [
      {
        name: "ordering",
        label: "Time ordering",
        kind: "one",
        choices: [
          { value: "exposure_precedes_outcome", label: "exposure before the outcome" },
          { value: "same_time", label: "measured at the same time" },
          { value: "unknown", label: "not known" },
        ],
        initial: [],
      },
    ],
  };
}

// ── the functional form (FORM, MODELING_SEQUENCE §1 row 5) ───────────────────

const RUNG_TAG: Record<string, { text: string; tone: "usual" | "na" | "badge" }> = {
  recommended: { text: "recommended", tone: "usual" },
  rank_lower: { text: "ranks lower", tone: "badge" },
  block_and_record: { text: "blocked and recorded", tone: "na" },
};

function form(ctx: ComposeContext): GenericQuestion {
  const card = ctx.forms;
  if (!card || !card.ready)
    return { grammar: "choice", data: [], options: [], fields: [], waiting: "forms" };
  const one = (column: string, value: string): SetExposureForm => ({
    // The server completes a spline's k by the declared rule, and the scale and unit it is on.
    kind: "set_exposure_form",
    column,
    form: value as SetExposureForm["form"],
    knots: null,
    knots_rule: null,
    n_effective: null,
    cuts: null,
    domain: "all",
    acknowledged: false,
    scale: null,
    unit: null,
  });
  const options: GenericOption[] = [];
  if (card.answer) {
    options.push({
      key: "__proposals",
      label: "Every proposed form",
      line: `Each column takes the form the card proposes; ${card.rule}`,
      chips: card.needs.map((n) => n.column),
      build: () => card.answer as unknown as Decision,
    });
  }
  for (const need of card.needs) {
    for (const o of need.options) {
      const tag = RUNG_TAG[o.rung];
      options.push({
        key: `${need.column}:${o.value}`,
        label: `${need.column} (${need.role}): ${o.label}`,
        line: o.consequence,
        labels: { customary: o.customary, sound: o.sound },
        tags: tag ? [tag] : undefined,
        note: need.label ?? undefined,
        build: () => one(need.column, o.value),
      });
    }
  }
  const data = [
    ...card.needs.map(
      (n) => `${tick(n.column)} (${n.role}) on its ${n.scale} scale, per ${n.unit}${n.mass_at_zero ? "; a mass at zero" : ""}.`,
    ),
    ...card.stated.map((x) => `${tick(x.column)}: ${x.why}`),
    ...(card.beside ? [card.beside] : []),
  ];
  return { grammar: "choice", data, options, fields: [] };
}

// ── effect modification and interaction (FORM, MODELING_SEQUENCE §1 row 7) ───

function modification(ctx: ComposeContext): GenericQuestion {
  const exposure = ctx.state.estimand?.family ? null : (ctx.state.estimand?.exposure ?? null);
  const declared = Object.entries(ctx.state.modifications ?? {})
    .filter(([, m]) => m)
    .map(([c]) => c);
  const set = (modifier: string, v: FieldValues, withdraw = false): SetModification => ({
    kind: "set_modification",
    modifier,
    modification: ((v.modification?.[0] as SetModification["modification"]) ??
      "effect_modification") as SetModification["modification"],
    exposure,
    low: null,
    high: null,
    levels: null,
    answers: {},
    post_hoc: false,
    withdraw,
  });
  // Any covariate of the model may be declared a modifier; the server refuses another column
  // with its exits, and asks an interaction's second exposure its adjustment answers.
  const candidates = Object.entries((ctx.state.roles ?? {}) as Record<string, string>)
    .filter(([c, r]) => r === "covariate" && c !== exposure && c !== ctx.state.target)
    .map(([c]) => c)
    .filter((c) => !declared.includes(c));
  const options: GenericOption[] = [
    ...candidates.map((c) => ({
      key: c,
      label: c,
      line: exposure
        ? `The effect of \`${exposure}\` across \`${c}\`, against one reference, on both scales.`
        : "Declared for the exposure once its question is answered.",
      build: (v: FieldValues) => set(c, v),
    })),
    ...declared.map((c) => ({
      key: `withdraw:${c}`,
      label: `Withdraw ${c}`,
      line: "Taken back from the declared modifiers.",
      build: (v: FieldValues) => set(c, v, true),
    })),
  ];
  const data = [
    ...(ctx.step.reason ? [ctx.step.reason] : []),
    ...(declared.length ? [`Declared: ${listing(declared)}.`] : []),
  ];
  return {
    grammar: "choice",
    data,
    options,
    fields: [
      {
        name: "modification",
        label: "Declared as",
        kind: "one",
        choices: [
          { value: "effect_modification", label: "effect modification (the exposure's effect across it)" },
          { value: "interaction", label: "interaction (a second exposure)" },
        ],
        initial: ["effect_modification"],
        appliesTo: candidates,
      },
    ],
  };
}

// ── what the task question still asks (WP18, RO-10) ──────────────────────────

/**
 * The task step held open by its follow-up (`step.followup`): a positive, markedly skewed
 * outcome's scale, or an ordinal text outcome's order. The target stage serves the question with
 * its own composed answers (`scale_question` / `order_question`, each an exit with its decision),
 * so the task card asks it rather than offering the task again.
 */
export function taskFollowup(ctx: ComposeContext): GenericQuestion | null {
  const follow = ctx.step.followup;
  if (!follow) return null;
  const ti = ctx.targetInfo?.column === ctx.state.target ? ctx.targetInfo : null;
  const q = follow === "scale" ? ti?.scale_question : ti?.order_question;
  if (!q) return { grammar: "fact", data: [], options: [], fields: [], waiting: "target_info" };
  const exits = [...q.options, ...("order_options" in q ? q.order_options : [])].filter(
    (e) => e.decision,
  );
  return {
    grammar: "fact",
    data: [q.question, q.evidence].filter((x) => !!x),
    fields: [],
    options: exits.map((e, i) => ({
      key: `${follow}-${i}`,
      label: e.label,
      line: "",
      build: () => e.decision as Decision,
    })),
  };
}

// ── the registry ─────────────────────────────────────────────────────────────

export const COMPOSERS: Partial<Record<QuestionKey, (ctx: ComposeContext) => GenericQuestion>> = {
  design,
  follow_up: followUp,
  clusters,
  estimand,
  adjustment,
  time_varying: timeVarying,
  form,
  modification,
  causal,
};

/** Every key has a renderer: bespoke, or composed here. */
export function hasRenderer(key: QuestionKey): boolean {
  return BESPOKE_KEYS.includes(key) || key in COMPOSERS;
}

/** The question for an open step: its composer's, else the teaching entry's words and options
 *  (a safety net: said, never blank, and answerable only once a composer exists). */
export function compose(ctx: ComposeContext): GenericQuestion {
  const composer = COMPOSERS[ctx.key];
  if (composer) return composer(ctx);
  return {
    grammar: "fact",
    data: [],
    options: teachingOptions(ctx),
    fields: [],
    note: "This version cannot record this answer from here yet; the options are listed so you can see what it asks.",
  };
}

/** The decision an option records under the fields' current values. */
export function decisionOf(option: GenericOption, values: FieldValues): Decision | null {
  return option.build ? option.build(values) : null;
}

/** A field shapes an option when it names it, or names none. */
export const fieldApplies = (field: GenericField, optionKey: string | null) =>
  !field.appliesTo || (optionKey !== null && field.appliesTo.includes(optionKey));
