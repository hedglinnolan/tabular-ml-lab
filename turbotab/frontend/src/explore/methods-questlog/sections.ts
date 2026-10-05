/**
 * The living methods section as an objective list (BLUEPRINT §11.4): the purpose's reporting
 * guideline gives the sections, the Router's questions and the recorded sentences fill them.
 *
 * Every item is one decision in one of the three tiers, plus the two states a quest log needs:
 *   stated   recorded, with its sentence (the engine's, verbatim) and its changeable phrase
 *   engine   stated by the engine without asking (a skip, or a reading its values settled);
 *            neutral, never green (DESIGN_LANGUAGE §09: green means a human recorded it)
 *   asked    a slot to fill now: the objective
 *   waiting  a slot that waits on an earlier answer, which it names
 *   silent   changes no number; in the export only
 *   author   an item only the author can supply (ethics, the instrument, …)
 *
 * The mapping of questions to guideline sections is this prototype's; the section names and
 * numbers are STROBE's methods items (4–12, with the STROBE-nut extensions) and TRIPOD+AI's
 * (5–17).
 */
import type { MethodsLine, Moment, Step } from "./data";

export type Tier = "stated" | "engine" | "asked" | "waiting" | "silent" | "author";
export type Guideline = "STROBE-nut" | "TRIPOD+AI";

export interface Item {
  key: string;
  title: string;
  tier: Tier;
  /** The engine's sentences for this item (stated), or its reason (engine, silent). */
  sentences: string[];
  /** What it waits on, by question title. */
  waitingOn: string[];
  /** Open readings or groups the slot holds (the count the objective shows). */
  count: number;
  /** Author-only: what only the author can supply (repo text, `turbotab/strobe_nut.py`). */
  ask?: string;
  /** An offered slot nothing waits on (a declared secondary analysis). */
  optional?: boolean;
}

export interface Section {
  key: string;
  title: string;
  ref: string;
  items: Item[];
  open: number;
  waiting: number;
  author: number;
  silent: number;
  /** Every item stated or settled by the engine (author-only items do not hold a section open). */
  done: boolean;
}

interface SectionDef {
  key: string;
  title: string;
  ref: string;
  items: string[];
}

export const STROBE: SectionDef[] = [
  { key: "design", title: "Study design", ref: "STROBE 4", items: ["purpose", "lens"] },
  {
    key: "data",
    title: "Data sources and measurement",
    ref: "STROBE 8 · STROBE-nut",
    items: ["roles", "readings", "author:instrument", "author:fcdb"],
  },
  { key: "participants", title: "Participants", ref: "STROBE 6", items: ["grain", "clusters", "exclusions"] },
  { key: "variables", title: "Variables", ref: "STROBE 7", items: ["target", "task", "estimand", "adjustment"] },
  { key: "bias", title: "Bias", ref: "STROBE 9", items: ["author:bias"] },
  { key: "size", title: "Study size", ref: "STROBE 10", items: ["size"] },
  { key: "quantitative", title: "Quantitative variables", ref: "STROBE 11", items: ["energy_adjustment"] },
  {
    key: "methods",
    title: "Statistical methods",
    ref: "STROBE 12",
    items: ["missing", "split", "survey", "models", "model_sequence", "sensitivity", "causal", "lock"],
  },
];

export const TRIPOD: SectionDef[] = [
  { key: "data", title: "Data", ref: "TRIPOD+AI 5", items: ["lens", "author:dates"] },
  { key: "participants", title: "Participants", ref: "TRIPOD+AI 6", items: ["grain", "clusters", "exclusions"] },
  { key: "prep", title: "Data preparation", ref: "TRIPOD+AI 7", items: ["roles", "readings"] },
  { key: "outcome", title: "Outcome", ref: "TRIPOD+AI 8", items: ["target", "task", "author:blinding"] },
  { key: "predictors", title: "Predictors", ref: "TRIPOD+AI 9", items: ["energy_adjustment", "author:timing"] },
  { key: "size", title: "Sample size", ref: "TRIPOD+AI 10", items: ["size"] },
  { key: "missing", title: "Missing data", ref: "TRIPOD+AI 11", items: ["missing"] },
  { key: "analysis", title: "Analytical methods", ref: "TRIPOD+AI 12", items: ["purpose", "split", "survey", "models"] },
  { key: "fairness", title: "Fairness", ref: "TRIPOD+AI 14", items: ["author:fairness"] },
  { key: "evaluation", title: "Training versus evaluation", ref: "TRIPOD+AI 16", items: ["open_seal"] },
  { key: "ethics", title: "Ethical approval", ref: "TRIPOD+AI 17", items: ["author:ethics"] },
];

export const TITLES: Record<string, string> = {
  purpose: "Purpose",
  lens: "Research lens",
  roles: "Column roles",
  readings: "Column readings",
  grain: "One row per person",
  clusters: "Grouping",
  exclusions: "Eligibility",
  target: "Outcome",
  task: "Outcome type",
  estimand: "Exposure and effect",
  adjustment: "Adjustment set",
  energy_adjustment: "Energy adjustment",
  missing: "Missing values",
  split: "Held-out rows",
  survey: "Survey design",
  models: "Model family",
  model_sequence: "Model sequence",
  sensitivity: "Sensitivity analyses",
  causal: "Causal estimators",
  lock: "Analysis plan lock",
  size: "Rows analyzed",
  open_seal: "Held-out score",
  substitution: "Substitution curve",
  "author:instrument": "Dietary instrument",
  "author:fcdb": "Food composition database",
  "author:bias": "Efforts to address bias",
  "author:dates": "Source and dates",
  "author:blinding": "Outcome ascertainment",
  "author:timing": "When predictors were measured",
  "author:fairness": "Fairness across groups",
  "author:ethics": "Ethics and consent",
};

/** What only the author can supply. The STROBE-nut and TRIPOD+AI wording is the repo's own
 *  (`turbotab/strobe_nut.py`, `turbotab/reporting_checklist.py`). */
export const AUTHOR_ASKS: Record<string, string> = {
  "author:instrument":
    "What dietary assessment instrument, administered how many times, covering what time frame, in what mode?",
  "author:fcdb": "Which food composition database and version, and how were non-matching foods handled?",
  "author:bias": "Any efforts to address potential sources of bias.",
  "author:dates": "The source of the data and the dates it covers.",
  "author:blinding": "How the outcome was ascertained and whether the assessor could see the predictors.",
  "author:timing": "When each predictor was measured relative to the index.",
  "author:fairness": "Whether performance was checked across relevant sociodemographic groups.",
  "author:ethics": "Ethics approval and consent.",
};

/** The decision kinds whose sentences an item states. */
export const KINDS: Record<string, string[]> = {
  purpose: ["set_purpose"],
  lens: ["set_lens"],
  roles: ["set_roles"],
  readings: ["confirm_reading", "confirm_readings", "set_column_unit"],
  grain: ["set_grain"],
  clusters: ["set_clusters"],
  exclusions: ["set_exclusions"],
  target: ["set_target"],
  task: ["set_task"],
  estimand: ["set_estimand"],
  adjustment: ["set_adjustment"],
  energy_adjustment: ["set_energy_adjustment"],
  missing: ["set_missing"],
  split: ["set_split"],
  survey: ["set_survey"],
  models: ["select_models"],
  model_sequence: ["set_model_sequence"],
  sensitivity: ["set_sensitivity"],
  causal: ["set_causal"],
  lock: ["lock_plan"],
  open_seal: ["open_seal"],
};

export interface ReadingsState {
  /** Open reading groups (each a slot). */
  open: number;
  /** The readings question is not reached yet (waits on the roles). */
  waiting: boolean;
}

export interface Extras {
  readings: ReadingsState;
  /** Analyzed rows, when the cohort holds them (Study size). */
  analyzed: number | null;
}

function linesOf(lines: MethodsLine[], kinds: string[]): MethodsLine[] {
  return lines.filter((l) => l.in_force && kinds.includes(l.kind));
}

const stepTitle = (k: string) => TITLES[k] ?? null;

function waitingNames(step: Step): string[] {
  const named = step.waiting_on.map(stepTitle).filter((x): x is string => !!x);
  return [...new Set(named)];
}

/** One item at one moment: its tier from the Router's step and the record's sentences. */
export function itemOf(key: string, moment: Moment, extras: Extras): Item {
  const title = TITLES[key] ?? key;
  const lines = moment.methods.lines;
  const base: Item = { key, title, tier: "silent", sentences: [], waitingOn: [], count: 0 };
  if (key.startsWith("author:")) return { ...base, tier: "author", ask: AUTHOR_ASKS[key] };
  if (key === "readings") {
    const said = linesOf(lines, KINDS.readings!).map((l) => l.sentence);
    if (extras.readings.waiting) return { ...base, tier: "waiting", waitingOn: ["Column roles"], sentences: said };
    if (extras.readings.open > 0) return { ...base, tier: "asked", count: extras.readings.open, sentences: said };
    return { ...base, tier: said.length ? "stated" : "engine", sentences: said };
  }
  if (key === "size") {
    if (extras.analyzed === null) return { ...base, tier: "waiting", waitingOn: ["Eligibility", "Missing values"] };
    return { ...base, tier: "engine", sentences: [`\`${extras.analyzed.toLocaleString("en-US")}\` rows were analyzed.`] };
  }
  const kinds = KINDS[key] ?? [];
  const said = linesOf(lines, kinds).map((l) => l.sentence);
  if (key === "model_sequence" || key === "sensitivity" || key === "lock") {
    // Not Router questions: declared beside the primary, or recorded by the server at the first
    // estimate. Before that they wait on the fit's inputs (the lock: on the first estimate).
    if (said.length) return { ...base, tier: "stated", sentences: said };
    if (key === "lock") return { ...base, tier: "waiting", waitingOn: ["the first estimate"] };
    const est = moment.view.interview.find((s) => s.key === "estimand");
    return est?.status === "answered"
      ? { ...base, tier: "asked", count: 0, optional: true }
      : { ...base, tier: "waiting", waitingOn: ["Exposure and effect"] };
  }
  const step = moment.view.interview.find((s) => s.key === key);
  if (!step) return { ...base, tier: "silent" };
  switch (step.status) {
    case "answered":
      return { ...base, tier: "stated", sentences: said };
    case "skipped":
      return { ...base, tier: "engine", sentences: step.reason ? [step.reason] : said };
    case "open":
      return { ...base, tier: "asked", count: 1, sentences: said };
    case "waiting":
      return { ...base, tier: "waiting", waitingOn: waitingNames(step), sentences: said };
    case "not_applicable":
      return { ...base, tier: "silent", sentences: step.reason ? [step.reason] : [] };
  }
}

export function sectionsOf(moment: Moment, extras: Extras, guideline: Guideline): Section[] {
  const defs = guideline === "STROBE-nut" ? STROBE : TRIPOD;
  return defs.map((d) => {
    const items = d.items.map((k) => itemOf(k, moment, extras));
    const open = items.reduce((n, i) => n + (i.tier === "asked" ? i.count : 0), 0);
    const waiting = items.filter((i) => i.tier === "waiting").length;
    const author = items.filter((i) => i.tier === "author").length;
    const silent = items.filter((i) => i.tier === "silent").length;
    const done = items.every((i) => i.tier !== "waiting" && (i.tier !== "asked" || !!i.optional));
    return { key: d.key, title: d.title, ref: d.ref, items, open, waiting, author, silent, done };
  });
}

export interface Progress {
  settled: number;
  open: number;
  waiting: number;
  author: number;
  total: number;
}

/** The objective counter: decisions settled, slots open, slots waiting, items only you supply. */
export function progressOf(sections: Section[]): Progress {
  let settled = 0;
  let open = 0;
  let waiting = 0;
  let author = 0;
  for (const s of sections)
    for (const i of s.items) {
      if (i.tier === "stated" || i.tier === "engine") settled += 1;
      else if (i.tier === "asked") open += i.count;
      else if (i.tier === "waiting") waiting += 1;
      else if (i.tier === "author") author += 1;
    }
  return { settled, open, waiting, author, total: settled + open + waiting };
}

/** The changeable phrase of a stated sentence: the part a click opens (BLUEPRINT §11.4). */
const PHRASES: Record<string, RegExp> = {
  set_purpose: /`(?:inference|prediction)`/,
  set_lens: /`[a-z]+`/,
  set_target: /^`[^`]+`/,
  set_estimand: /total effect|direct effect/,
  set_energy_adjustment: /the standard \(multivariate\) model|the residual method[^:]*|No energy adjustment/,
  set_missing: /complete-case analysis|multiple imputation/,
  set_exclusions: /No rows were excluded/,
  set_split: /`20%`/,
  select_models: /linear regression and gradient-boosted trees|linear regression/,
  set_model_sequence: /Model 1, adjusted for `age`, `gender` and `kcal`/,
  set_sensitivity: /`Plausible energy reporters only`/,
};

export function phraseOf(kind: string, sentence: string): [string, string, string] | null {
  const re = PHRASES[kind];
  const m = re ? re.exec(sentence) : null;
  if (!m) return null;
  return [sentence.slice(0, m.index), m[0], sentence.slice(m.index + m[0].length)];
}

export function kindOfSentence(lines: MethodsLine[], sentence: string): string | null {
  return lines.find((l) => l.sentence === sentence)?.kind ?? null;
}
