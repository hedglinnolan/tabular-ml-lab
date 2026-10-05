/**
 * The living methods section as a document (BLUEPRINT §11.4): the Router's questions and the
 * recorded decisions, arranged by the purpose's reporting guideline — STROBE-nut under inference,
 * TRIPOD+AI under prediction — into sections of paragraphs. Pure: every paragraph's text is a
 * server string (a decision's sentence, an interview step's reason, a stage's methods sentence or
 * a teaching entry), never composed here.
 *
 * Each paragraph is in one tier, decided by the server's own state for its question:
 *   recorded  answered: the decision's sentence, with its changeable phrase marked
 *   stated    skipped: the server stated it from the data (the step's reason), phrase marked
 *   asked     open: a slot with the best guess and its evidence
 *   waiting   not yet reachable: says what it waits on
 *   author    a checklist item only the author can supply (ethics, setting, dates)
 *   silent    not applicable: changes no number; counted, listed only on request (the export)
 */
import type { QuestionKey } from "../../api/m1-types";
import type { DecisionRecord, ProjectView } from "../../api/schema";
import {
  artifactOf,
  teaching,
  type AskCard,
  type EstimandCard,
  type Moment,
} from "./fixture";

export type Tier = "recorded" | "stated" | "asked" | "waiting" | "author" | "silent";
export type Guideline = "STROBE-nut" | "TRIPOD+AI";
export type SlotKind = "readings" | "estimand" | "adjustment" | "question";
export type ConceptKey = "substitution" | "mediator";

export interface Mark {
  /** The exact substring of `text` that is marked. */
  text: string;
  kind: "phrase" | "concept";
  concept?: ConceptKey;
}

export interface Para {
  id: string;
  head: string;
  /** The guideline item this paragraph answers. */
  item: string | null;
  tier: Tier;
  text: string | null;
  marks: Mark[];
  question: QuestionKey | null;
  slot: SlotKind | null;
  /** For waiting: the question it waits on, in words. */
  waitingOn: string | null;
  /** A long sentence's tail, folded: shown on request. */
  fold: { label: string; text: string } | null;
  /** Sentences beneath the paragraph's own (the readings' confirmations). */
  subs: string[];
  seq: number | null;
  afterEstimates: boolean;
}

export interface Section {
  id: string;
  title: string;
  item: string;
  paras: Para[];
  silent: { head: string; reason: string }[];
  results?: boolean;
}

export interface Doc {
  guideline: Guideline;
  purpose: "inference" | "prediction";
  sections: Section[];
  /** The slots in the order "next slot" walks them. */
  objectives: string[];
  counts: { written: number; open: number; waiting: number; author: number; silent: number };
}

// ── the blueprint: which question sits in which section, under which head ─────

type Row =
  | { q: QuestionKey; head: string; item?: string }
  | { kind: string; head: string; item?: string }
  | { author: string; head: string; item: string; prompt: string }
  | { readings: true; head: string; item?: string }
  | { stage: string; head: string; item: string }
  | { results: "flow" | "table2" | "other" | "performance"; head: string; item: string };

interface Blueprint {
  id: string;
  title: string;
  item: string;
  rows: Row[];
  results?: boolean;
}

/** STROBE items 4, 5 and 8 are quoted from the STROBE checklist (von Elm et al. 2007). */
const STROBE_NUT: Blueprint[] = [
  {
    id: "design",
    title: "Study design and setting",
    item: "STROBE 4–5",
    rows: [
      { author: "design", head: "Study design", item: "4", prompt: "Present key elements of study design early in the paper." },
      {
        author: "setting",
        head: "Setting",
        item: "5",
        prompt:
          "Describe the setting, locations, and relevant dates, including periods of recruitment, exposure, follow-up, and data collection.",
      },
    ],
  },
  {
    id: "participants",
    title: "Participants",
    item: "STROBE 6",
    rows: [
      { q: "grain", head: "Unit of analysis", item: "6" },
      { q: "exclusions", head: "Eligibility", item: "nut-13" },
      { kind: "set_sensitivity", head: "Every row, beside", item: "nut-13" },
      { q: "clusters", head: "Clustering", item: "6" },
      { q: "survey", head: "Survey design", item: "6" },
    ],
  },
  {
    id: "variables",
    title: "Variables",
    item: "STROBE 7",
    rows: [
      { q: "target", head: "Outcome", item: "7" },
      { q: "task", head: "Its type", item: "7" },
      { q: "roles", head: "Column roles", item: "7" },
      { q: "estimand", head: "Exposure and estimand", item: "7" },
      { q: "adjustment", head: "Adjustment set", item: "7" },
    ],
  },
  {
    id: "measurement",
    title: "Data sources and measurement",
    item: "STROBE 8",
    rows: [
      {
        author: "assessment",
        head: "Dietary assessment",
        item: "8",
        prompt:
          "For each variable of interest, give sources of data and details of methods of assessment (measurement).",
      },
      { q: "lens", head: "Lens", item: "8" },
      { readings: true, head: "Readings", item: "8" },
    ],
  },
  {
    id: "statistics",
    title: "Statistical methods",
    item: "STROBE 12",
    rows: [
      { q: "purpose", head: "Purpose", item: "12" },
      { q: "energy_adjustment", head: "Energy adjustment", item: "nut-12.2" },
      { q: "missing", head: "Missing data", item: "12c" },
      { q: "split", head: "Validation", item: "12" },
      { q: "causal", head: "Estimator", item: "12" },
      { q: "models", head: "Model", item: "12" },
      { stage: "effects", head: "Estimation and reporting", item: "12" },
      { stage: "secondary", head: "Secondary model", item: "12e" },
      { stage: "sensitivity", head: "Sensitivity analysis", item: "12e" },
      { kind: "lock_plan", head: "The plan, locked", item: "12" },
    ],
  },
  {
    id: "results",
    title: "Results",
    item: "STROBE 13–17",
    results: true,
    rows: [
      { results: "flow", head: "Participants", item: "13" },
      { results: "table2", head: "Main results", item: "16" },
      { results: "other", head: "Other analyses", item: "17" },
      { q: "substitution", head: "Substitution curves", item: "17" },
    ],
  },
];

const TRIPOD_AI: Blueprint[] = [
  {
    id: "data",
    title: "Data",
    item: "TRIPOD+AI 5",
    rows: [
      { author: "sources", head: "Sources", item: "5", prompt: "The sources of data and the dates they span." },
      { q: "lens", head: "Lens", item: "5" },
    ],
  },
  {
    id: "participants",
    title: "Participants",
    item: "TRIPOD+AI 6",
    rows: [
      { q: "grain", head: "Unit of analysis", item: "6" },
      { q: "exclusions", head: "Eligibility", item: "6" },
      { q: "clusters", head: "Clustering", item: "6" },
      { q: "survey", head: "Survey design", item: "6" },
    ],
  },
  {
    id: "preparation",
    title: "Data preparation",
    item: "TRIPOD+AI 7",
    rows: [
      { readings: true, head: "Readings", item: "7" },
      { q: "energy_adjustment", head: "Energy adjustment", item: "7" },
    ],
  },
  {
    id: "outcome",
    title: "Outcome",
    item: "TRIPOD+AI 8",
    rows: [
      { q: "target", head: "Outcome", item: "8" },
      { q: "task", head: "Its type", item: "8" },
    ],
  },
  {
    id: "predictors",
    title: "Predictors",
    item: "TRIPOD+AI 9",
    rows: [{ q: "roles", head: "Column roles", item: "9" }],
  },
  {
    id: "missing",
    title: "Missing data",
    item: "TRIPOD+AI 11",
    rows: [{ q: "missing", head: "Missing data", item: "11" }],
  },
  {
    id: "analysis",
    title: "Analytical methods",
    item: "TRIPOD+AI 12",
    rows: [
      { q: "purpose", head: "Purpose", item: "12" },
      { q: "models", head: "Models", item: "12" },
    ],
  },
  {
    id: "evaluation",
    title: "Training versus evaluation",
    item: "TRIPOD+AI 16",
    rows: [
      { q: "split", head: "The seal", item: "16" },
      { q: "open_seal", head: "The held-out score", item: "16" },
    ],
  },
  {
    id: "results",
    title: "Results",
    item: "TRIPOD+AI 20–23",
    results: true,
    rows: [
      { results: "flow", head: "Participants", item: "20" },
      { results: "performance", head: "Model performance", item: "23" },
      { q: "substitution", head: "Substitution curves", item: "23" },
    ],
  },
  {
    id: "ethics",
    title: "Ethical approval",
    item: "TRIPOD+AI 17",
    rows: [{ author: "ethics", head: "Ethics", item: "17", prompt: "The ethics committee and the approval it gave." }],
  },
];

/** Not-applicable steps a reader of the methods would still want stated (they change numbers). */
const STATED_WHEN_NA: readonly QuestionKey[] = ["survey"];

/** Each question's changeable phrase, found in its server sentence. */
const PHRASE: Partial<Record<QuestionKey, RegExp>> = {
  lens: /`dietary`/,
  target: /^`[^`]+`/,
  purpose: /`inference`|`prediction`/,
  task: /a regression outcome/,
  grain: /each person is one row/,
  clusters: /no column reads as a site, centre, household or batch/,
  survey: /no surveyed population to weight to|not weighted to a population/,
  exclusions: /Willett 2013's sex-specific cut-offs/,
  missing: /complete-case analysis|imputed in each training fold without the outcome/,
  split: /No rows were held out|A random `20%` of the rows/,
  estimand: /total effect of `[^`]+`/,
  energy_adjustment: /standard \(multivariate\) model/,
  causal: /the primary model estimates the declared effect/,
  models: /linear regression/,
};

/** Where a concept is named inside a server sentence. */
const CONCEPTS: { concept: ConceptKey; pattern: RegExp }[] = [
  { concept: "substitution", pattern: /a substitution|at fixed total energy/ },
  { concept: "mediator", pattern: /are mediators/ },
];

/** A select_models sentence carries every value-settled reading; the readings card lists them. */
const READ_FROM_VALUES = "Read from the values, no question asked:";

function firstUpper(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function marksFor(question: QuestionKey | null, text: string): Mark[] {
  const marks: Mark[] = [];
  const pattern = question ? PHRASE[question] : undefined;
  const m = pattern?.exec(text);
  if (m) marks.push({ text: m[0], kind: "phrase" });
  for (const c of CONCEPTS) {
    const hit = c.pattern.exec(text);
    if (hit && !marks.some((x) => x.text.includes(hit[0]) || hit[0].includes(x.text)))
      marks.push({ text: hit[0], kind: "concept", concept: c.concept });
  }
  return marks;
}

function recordById(view: ProjectView, id: string | null | undefined): DecisionRecord | undefined {
  return id ? view.decisions.find((d) => d.id === id) : undefined;
}

function latest(view: ProjectView, kind: string): DecisionRecord | undefined {
  return [...view.decisions].reverse().find((d) => d.decision.kind === kind);
}

/** The adjustment question answers in several records: one sentence each, in order. */
function adjustmentSentences(view: ProjectView): DecisionRecord[] {
  return view.decisions.filter((d) => d.decision.kind === "set_adjustment");
}

/** The open step that carries the ask card (the readings its consumer needs), if any. */
export function openAsk(view: ProjectView): AskCard | null {
  return view.interview.find((s) => (s.status === "open" || s.status === "waiting") && s.ask)?.ask ?? null;
}

/** Confirmations made one reading at a time (a unit and its days, or a single reading). */
export function singleConfirmations(view: ProjectView): DecisionRecord[] {
  return view.decisions.filter(
    (d) => d.decision.kind === "confirm_reading" || d.decision.kind === "set_column_unit",
  );
}

/**
 * The ask card's readings as the slot lists them: a homogeneous family stays one line (§14.2);
 * single readings that share a guess and the same evidence sit on one line too, each column still
 * its own confirmation (a tap per column settles that column alone). Evidence is the roles stage's
 * reason for the column where it has one (more specific than the card's), else the card's.
 */
export interface ReadingLine {
  key: string;
  columns: string[];
  guess: string | null;
  guessWords: string;
  evidence: string;
  family: boolean;
}

export function readingLines(ask: AskCard, proposals: { column: string; reason: string }[]): ReadingLine[] {
  const out: ReadingLine[] = [];
  const reason = (c: string) => proposals.find((p) => p.column === c)?.reason ?? null;
  for (const g of ask.groups) {
    const family = g.columns.length > 1;
    const evidence = reason(g.columns[0]!) ?? g.evidence;
    const same = family ? null : out.find((l) => !l.family && l.guess === g.guess && l.evidence === evidence);
    if (same) {
      same.columns.push(g.columns[0]!);
      same.key = same.columns.join(",");
      continue;
    }
    out.push({ key: g.columns.join(","), columns: [...g.columns], guess: g.guess, guessWords: g.guess_words, evidence, family });
  }
  return out;
}

/** How many single confirmations unlock the block confirm (BLUEPRINT §11.4 rule 4). */
export const MASTERY = 3;

/**
 * The block confirmation the section offers once the user has confirmed `MASTERY` readings one at a
 * time: the ask card's own block exit, which settles exactly the readings it lists, each with the
 * value it shows (§14.2). Null before the unlock, or when the card has no block.
 */
export function unlockedBlock(view: ProjectView): AskCard["exits"][number] | null {
  const ask = openAsk(view);
  if (!ask || singleConfirmations(view).length < MASTERY) return null;
  const block = ask.exits[0];
  return block?.decision && (block.decision as { kind?: string }).kind === "confirm_readings" ? block : null;
}

/** The title a question carries in the teaching (used to say what a paragraph waits on). */
function titleOf(q: string): string {
  return teaching(q)?.title ?? q.replace(/_/g, " ");
}

function waitingText(view: ProjectView, q: QuestionKey): string {
  const step = view.interview.find((s) => s.key === q);
  const on = step?.waiting_on?.[0];
  if (on && view.interview.some((s) => s.key === on)) return titleOf(on);
  const open = view.interview.find((s) => s.status === "open");
  return open ? titleOf(open.key) : "an earlier question";
}

function para(base: Partial<Para> & Pick<Para, "id" | "head" | "tier">): Para {
  return {
    item: null,
    text: null,
    marks: [],
    question: null,
    slot: null,
    waitingOn: null,
    fold: null,
    subs: [],
    seq: null,
    afterEstimates: false,
    ...base,
  };
}

function slotKind(q: QuestionKey): SlotKind {
  if (q === "estimand") return "estimand";
  if (q === "adjustment") return "adjustment";
  return "question";
}

function questionPara(m: Moment, row: { q: QuestionKey; head: string; item?: string }, silent: Section["silent"]): Para | null {
  const { view } = m;
  const step = view.interview.find((s) => s.key === row.q);
  if (!step) return null;
  const base = { id: row.q, head: row.head, item: row.item ?? null, question: row.q };
  switch (step.status) {
    case "answered": {
      if (row.q === "adjustment") {
        const recs = adjustmentSentences(view);
        const text = recs
          .map((r) => r.sentence)
          .filter(Boolean)
          .join(" ");
        return para({ ...base, tier: "recorded", text, marks: marksFor(row.q, text), seq: recs.at(-1)?.seq ?? null });
      }
      const rec = recordById(view, step.decision_id);
      if (!rec) return null;
      let text = rec.sentence ?? "";
      let fold: Para["fold"] = null;
      const cut = text.indexOf(READ_FROM_VALUES);
      if (cut > 0) {
        fold = { label: `${m.readings.read_from_data.length} readings were read from your data`, text: text.slice(cut) };
        text = text.slice(0, cut).trim();
      }
      return para({
        ...base,
        tier: "recorded",
        text,
        fold,
        marks: marksFor(row.q, text),
        seq: rec.seq,
        afterEstimates: rec.after_estimates ?? false,
      });
    }
    case "skipped": {
      const text = firstUpper(step.reason ?? "");
      return para({ ...base, tier: "stated", text, marks: marksFor(row.q, text) });
    }
    case "not_applicable": {
      if (STATED_WHEN_NA.includes(row.q)) {
        const text = step.reason ?? "";
        return para({ ...base, tier: "stated", text, marks: marksFor(row.q, text) });
      }
      silent.push({ head: row.head, reason: step.reason ?? "" });
      return null;
    }
    case "open":
      return para({ ...base, tier: "asked", slot: slotKind(row.q), text: teaching(row.q)?.question ?? null });
    case "waiting":
      return para({ ...base, tier: "waiting", waitingOn: waitingText(view, row.q) });
  }
}

function readingsPara(m: Moment, row: { head: string; item?: string }): Para {
  const { view } = m;
  const confirmations = view.decisions
    .filter((d) => ["set_column_unit", "confirm_reading", "confirm_readings"].includes(d.decision.kind))
    .map((d) => d.sentence)
    .filter((t): t is string => !!t);
  const ask = openAsk(view);
  const fold = m.readings.read_from_data.length
    ? { label: `${m.readings.read_from_data.length} readings were read from your data`, text: m.readings.sentence }
    : null;
  if (ask) {
    return para({
      id: "readings",
      head: row.head,
      item: row.item ?? null,
      tier: "asked",
      slot: "readings",
      text: ask.text,
      subs: confirmations,
      fold,
    });
  }
  return para({
    id: "readings",
    head: row.head,
    item: row.item ?? null,
    tier: confirmations.length ? "recorded" : "waiting",
    waitingOn: confirmations.length ? null : "the question whose answer reads them",
    subs: confirmations,
    fold,
  });
}

function resultsPara(m: Moment, row: { results: string; head: string; item: string }): Para | null {
  const fitted = !!m.stages.fit;
  const base = { id: `results-${row.results}`, head: row.head, item: row.item };
  if (row.results === "flow") {
    return m.stages.cohort ? para({ ...base, tier: "recorded" }) : null;
  }
  if (row.results === "table2") {
    if (m.stages.effects) return para({ ...base, tier: "recorded" });
    // Inference shows no estimate until the effect is named; once it is, the result waits on the fit.
    const named = m.view.interview.some((s) => s.key === "estimand" && s.status === "answered");
    if (!named && m.view.state.purpose !== "prediction")
      return para({ ...base, tier: "waiting", text: teaching("estimand")?.one_liner ?? null });
    return para({ ...base, tier: "waiting", waitingOn: waitingText(m.view, "models") });
  }
  if (row.results === "other") {
    if (!m.stages.secondary && !m.stages.sensitivity) return para({ ...base, tier: "waiting", waitingOn: "the fit" });
    return para({ ...base, tier: "recorded" });
  }
  if (row.results === "performance") {
    return fitted ? para({ ...base, tier: "recorded" }) : para({ ...base, tier: "waiting", waitingOn: "the fit" });
  }
  return null;
}

export function buildDoc(m: Moment): Doc {
  const purpose = m.view.state.purpose === "prediction" ? "prediction" : "inference";
  const blueprint = purpose === "prediction" ? TRIPOD_AI : STROBE_NUT;
  const sections: Section[] = blueprint.map((b) => {
    const silent: Section["silent"] = [];
    const paras: Para[] = [];
    for (const row of b.rows) {
      let p: Para | null = null;
      if ("q" in row) p = questionPara(m, row, silent);
      else if ("author" in row)
        p = para({ id: `author-${row.author}`, head: row.head, item: row.item, tier: "author", text: row.prompt });
      else if ("readings" in row) p = readingsPara(m, row);
      else if ("stage" in row) {
        // A stage's own methods sentence, once it has computed (the estimate stages, after the lock).
        const text = artifactOf<{ methods?: string | null }>(m, row.stage)?.methods ?? null;
        if (text)
          p = para({ id: `stage-${row.stage}`, head: row.head, item: row.item, tier: "recorded", text, marks: marksFor(null, text) });
      }
      else if ("results" in row) p = resultsPara(m, row);
      else {
        const rec = latest(m.view, row.kind);
        if (rec)
          p = para({
            id: row.kind,
            head: row.head,
            item: row.item ?? null,
            tier: "recorded",
            text: rec.sentence,
            seq: rec.seq,
            afterEstimates: rec.after_estimates ?? false,
          });
      }
      if (p) paras.push(p);
    }
    return { id: b.id, title: b.title, item: b.item, paras, silent, results: b.results };
  });
  const all = sections.flatMap((s) => s.paras);
  // The readings a consumer needs come before the question that consumes them.
  const objectives = [
    ...all.filter((p) => p.tier === "asked" && p.slot === "readings"),
    ...all.filter((p) => p.tier === "asked" && p.slot !== "readings"),
  ].map((p) => p.id);
  const count = (t: Tier) => all.filter((p) => p.tier === t).length;
  return {
    guideline: purpose === "prediction" ? "TRIPOD+AI" : "STROBE-nut",
    purpose,
    sections,
    objectives,
    counts: {
      written: count("recorded") + count("stated"),
      open: count("asked"),
      waiting: count("waiting"),
      author: count("author"),
      silent: sections.reduce((n, s) => n + s.silent.length, 0),
    },
  };
}

// ── concepts: taught in full at their first encounter, condensed after (§11.4 rule 2) ──

export interface ConceptTeaching {
  key: ConceptKey;
  term: string;
  /** The question whose slot first uses the concept. */
  taughtAt: QuestionKey;
  heading: string;
  body: string;
  evidence: { status: string; source: string } | null;
  /** The condensed phrase a later encounter shows, expandable. */
  condensed: string;
}

export function conceptTeaching(m: Moment, key: ConceptKey): ConceptTeaching | null {
  if (key === "substitution") {
    const section = teaching("estimand")?.drawer?.sections?.[0];
    const card = artifactOf<{ estimand: EstimandCard | null }>(m, "proposals")?.estimand;
    const condensed = card?.contrasts.find((c) => c.contrast === "substitution")?.consequence;
    if (!section || !condensed) return null;
    return {
      key,
      term: "substitution",
      taughtAt: "estimand",
      heading: section.heading,
      body: section.body,
      evidence: section.evidence ?? null,
      condensed,
    };
  }
  const entry = teaching("adjustment");
  const term = entry?.terms?.find((t) => t.term === "mediator");
  if (!entry || !term) return null;
  return {
    key,
    term: "mediator",
    taughtAt: "adjustment",
    heading: entry.title,
    body: entry.why,
    evidence: null,
    condensed: term.definition,
  };
}

/** A concept is condensed once the question that taught it has been answered. */
export function conceptLearned(m: Moment, c: ConceptTeaching): boolean {
  return m.view.interview.some((s) => s.key === c.taughtAt && s.status === "answered");
}

/** The sentences of a long paragraph: a reader sees the first two and opens the rest (§11 rule 4:
 *  no walls). Splits at a full stop followed by a capital or a data chip. */
export function sentences(text: string): string[] {
  return text.split(/(?<=\.)\s+(?=[A-Z`])/);
}

/** Split a sentence around its marks, in order of appearance. */
export function segments(text: string, marks: Mark[]): ({ text: string; mark: Mark | null })[] {
  const found = marks
    .map((mark) => ({ mark, at: text.indexOf(mark.text) }))
    .filter((x) => x.at >= 0)
    .sort((a, b) => a.at - b.at);
  const out: { text: string; mark: Mark | null }[] = [];
  let i = 0;
  for (const { mark, at } of found) {
    if (at < i) continue;
    if (at > i) out.push({ text: text.slice(i, at), mark: null });
    out.push({ text: mark.text, mark });
    i = at + mark.text.length;
  }
  if (i < text.length) out.push({ text: text.slice(i), mark: null });
  return out;
}
