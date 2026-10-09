/**
 * The methods map's scenario, as pure functions over the captured fixture: the nodes and regions,
 * the answers (client state), each node's tier (stated · asked · recorded · waiting · silent), the
 * record the methods text reads (the engine's sentences, in force), the objectives, and which
 * captured fit a set of answers selects.
 *
 * Nothing here writes engine text: every sentence is looked up in the fixture. A combination the
 * capture did not take (an exclusion screen other than two, an adjustment answer other than the
 * fixture's) is said to be so, never filled in.
 */
import type { PreviewResult } from "../../api/m1-stage-types";
import { INF, PRED, type Answer3, type Fit, type Md, type Preview } from "./fixture";

export type Purpose = "inference" | "prediction";

/** The scenario's exposure (scenario.EXPOSURE; SCENARIO.md). */
export const EXPOSURE = "sugar";

export type NodeId =
  | "source"
  | "lens"
  | "outcome"
  | "purpose"
  | "grain"
  | "readings"
  | "exclusions"
  | "seal"
  | "clusters"
  | "exposure"
  | "adjustment"
  | "energy"
  | "form"
  | "missing"
  | "model1"
  | "family"
  | "lock"
  | "estimate"
  | "matter"
  // prediction only
  | "p_missing"
  | "p_seal"
  | "p_energy"
  | "p_models"
  | "p_score";

export type Tier = "stated" | "asked" | "recorded" | "waiting" | "silent" | "gate" | "result" | "anchor";

export interface RegionDef {
  id: string;
  title: string;
  items: string;
  nodes: NodeId[];
}

/** STROBE-nut under inference (Lachat et al. 2016); TRIPOD+AI under prediction (Collins et al. 2024). */
export const REGIONS: Record<Purpose, RegionDef[]> = {
  inference: [
    { id: "data", title: "Data and variables", items: "STROBE 7–8 · nut-7, nut-8", nodes: ["source", "readings"] },
    { id: "participants", title: "Participants", items: "STROBE 6, 13 · nut-13", nodes: ["exclusions", "seal", "clusters"] },
    { id: "exposure", title: "Exposure and confounding", items: "STROBE 7, 12a", nodes: ["exposure", "adjustment"] },
    { id: "methods", title: "Statistical methods", items: "STROBE 12 · nut-11, nut-12", nodes: ["energy", "form", "missing", "model1", "family"] },
    { id: "results", title: "Results", items: "STROBE 16–17", nodes: ["lock", "estimate", "matter"] },
  ],
  prediction: [
    { id: "data", title: "Source of data and predictors", items: "TRIPOD+AI 5, 8–9", nodes: ["source", "readings"] },
    { id: "participants", title: "Participants", items: "TRIPOD+AI 6", nodes: ["exclusions", "clusters"] },
    { id: "missing", title: "Missing data", items: "TRIPOD+AI 11", nodes: ["p_missing"] },
    { id: "methods", title: "Analytical methods", items: "TRIPOD+AI 12", nodes: ["p_seal", "p_energy", "p_models"] },
    { id: "evaluation", title: "Evaluation", items: "TRIPOD+AI 12e, 16, 23", nodes: ["p_score"] },
  ],
};

export const NODE_TITLE: Record<NodeId, string> = {
  source: "The table",
  lens: "Lens",
  outcome: "Outcome",
  purpose: "Purpose",
  grain: "One row is",
  readings: "Readings",
  exclusions: "Exclusions",
  seal: "Held-out rows",
  clusters: "Grouping",
  exposure: "Exposure and estimand",
  adjustment: "Adjustment set",
  energy: "Energy adjustment",
  form: "Exposure's form",
  missing: "Missing data",
  model1: "Model 1",
  family: "Model family",
  lock: "Plan lock",
  estimate: "Table 2",
  matter: "Which decisions mattered",
  p_missing: "Missing data",
  p_seal: "Held-out rows",
  p_energy: "Energy adjustment",
  p_models: "Model families",
  p_score: "Held-out score",
};

/** A node's name on the map, short enough to sit over it. */
export const MAP_TITLE: Record<NodeId, string> = {
  ...NODE_TITLE,
  grain: "Grain",
  seal: "Holdout",
  exposure: "Exposure",
  adjustment: "Adjustment",
  energy: "Energy",
  form: "Form",
  missing: "Missing",
  family: "Family",
  lock: "Lock",
  matter: "Mattered",
  p_missing: "Missing",
  p_seal: "Holdout",
  p_energy: "Energy",
  p_models: "Models",
  p_score: "Score",
};

/** The teaching entry each node's question comes from. */
export const NODE_TEACH: Partial<Record<NodeId, string>> = {
  lens: "lens",
  outcome: "target",
  purpose: "purpose",
  grain: "grain",
  readings: "roles",
  exclusions: "exclusions",
  seal: "split",
  clusters: "clusters",
  exposure: "estimand",
  adjustment: "adjustment",
  energy: "energy_adjustment",
  missing: "missing",
  family: "models",
  p_missing: "missing",
  p_seal: "split",
  p_energy: "energy_adjustment",
  p_models: "models",
};

/** The concepts each node teaches, in the order they are first met (progressive disclosure). */
export const NODE_TERMS: Partial<Record<NodeId, { teach: string; term: string }[]>> = {
  readings: [
    { teach: "roles", term: "exposure" },
    { teach: "roles", term: "covariate" },
  ],
  exclusions: [{ teach: "exclusions", term: "implausible intake" }],
  seal: [
    { teach: "split", term: "holdout" },
    { teach: "split", term: "cross-validation" },
  ],
  exposure: [
    { teach: "estimand", term: "estimand" },
    { teach: "energy_adjustment", term: "substitution" },
  ],
  adjustment: [{ teach: "adjustment", term: "mediator" }],
  energy: [
    { teach: "energy_adjustment", term: "substitution" },
    { teach: "energy_adjustment", term: "residual method" },
    { teach: "energy_adjustment", term: "nutrient density" },
  ],
  missing: [{ teach: "missing", term: "complete cases" }],
  p_missing: [{ teach: "missing", term: "imputation" }],
  p_seal: [
    { teach: "split", term: "holdout" },
    { teach: "split", term: "cross-validation" },
  ],
};

// ── answers (client state) ──────────────────────────────────────────────────

export type Triple = [Answer3, Answer3, Answer3];

export interface Answers {
  /** Readings confirmed, by key, and how: one at a time or in a block. */
  readings: Record<string, "single" | "block">;
  /** The order the singles came in (the role block unlocks after three role readings). */
  singles: string[];
  unit: boolean;
  exclusions: string | null;
  sensitivity: string[];
  /** Missing data: complete cases, recorded at its question (the scenario's answer). */
  missing: boolean;
  exposure: boolean;
  /** Each adjustment group, once recorded: each column's three answers. */
  adjustment: Record<string, Record<string, Triple>>;
  energy: string;
  /** The exposure's form, once recorded; null: none recorded (the engine fits a straight line,
   *  and the record says nothing of it). */
  form: string | null;
  model1: "guess" | "empty" | null;
  /** The plan as it stood at the lock (the energy method and form an edit after it changes). */
  locked: { energy: string; form: string | null; key: string } | null;
  /** Answers recorded after the lock, in order (their sentences say so). */
  after: { node: NodeId; value: string }[];
  // prediction
  pExclusions: boolean;
  pMissing: boolean;
  pSeal: string | null;
}

export const INITIAL: Answers = {
  readings: {},
  singles: [],
  unit: false,
  exclusions: null,
  sensitivity: [],
  missing: false,
  exposure: false,
  adjustment: {},
  // the energy model the engine ranks first (scenario.energy_default): written in until changed
  energy: INF.energy.ranking.order[0]!,
  form: null,
  model1: null,
  locked: null,
  after: [],
  pExclusions: false,
  pMissing: false,
  pSeal: null,
};

export const SINGLES_TO_UNLOCK = 3;

export type Action =
  | { type: "reading"; key: string }
  | { type: "block"; purpose?: Purpose }
  | { type: "codes" }
  | { type: "unit" }
  | { type: "exclusions"; key: string }
  | { type: "sensitivity"; key: string }
  | { type: "missing" }
  | { type: "exposure" }
  | { type: "adjust"; group: string; answers: Record<string, Triple> }
  | { type: "energy"; method: string }
  | { type: "form"; form: string }
  | { type: "model1"; value: "guess" | "empty" }
  | { type: "lock" }
  | { type: "undo"; node: NodeId }
  | { type: "p_exclusions" }
  | { type: "p_missing" }
  | { type: "p_seal"; holdout: string }
  | { type: "reset" };

export const READING_KEYS = INF.readings.items.map((i) => i.key);
/** The role readings, in the exposure's card's order (the block's mask is over these). */
export const ROLE_KEYS = INF.readings.items.filter((i) => i.reading === "role").map((i) => i.key);
/** The fit's code-or-amount readings under inference (its own card, its own block). */
export const CODE_KEYS = INF.readings.items.filter((i) => i.reading === "code_or_count").map((i) => i.key);
/** Under prediction the code-or-amount readings come in the role readings' one block. */
export const PRED_CODE_KEYS = PRED.readings.codes.map((i) => i.key);

/** The readings a purpose asks, in the cards' order. */
export function readingKeys(purpose: Purpose = "inference"): string[] {
  return purpose === "inference" ? READING_KEYS : [...ROLE_KEYS, ...PRED_CODE_KEYS];
}

/** What is still to confirm, in the cards' order. */
export function unconfirmed(a: Answers, purpose: Purpose = "inference"): string[] {
  return readingKeys(purpose).filter((k) => !a.readings[k]);
}

/** The role readings confirmed one at a time (the mastery the block unlocks after). */
export const roleSingles = (a: Answers) => a.singles.filter((k) => ROLE_KEYS.includes(k));

/** The block's sentence for a set of readings it lists, under a purpose (null: not captured). */
function blockSentence(keys: string[], purpose: Purpose): Md | null {
  const block = purpose === "inference" ? INF.readings.block : PRED.readings.block;
  let mask = 0;
  for (const k of keys) {
    const i = block.items.indexOf(k);
    if (i < 0) return null;
    mask |= 1 << i;
  }
  const idx = block.by_mask[mask.toString(36)];
  return idx === undefined ? null : block.sentences[idx]!;
}

/** The block's readings, its sentence, or null when it is not unlocked (or not captured): under
 *  inference the role readings left; under prediction every reading left (its codes too). */
export function blockOffer(a: Answers, purpose: Purpose = "inference"): { keys: string[]; sentence: Md } | null {
  if (roleSingles(a).length < SINGLES_TO_UNLOCK) return null;
  const keys = (purpose === "inference" ? ROLE_KEYS : readingKeys("prediction")).filter((k) => !a.readings[k]);
  if (keys.length < 2) return null;
  const sentence = blockSentence(keys, purpose);
  return sentence ? { keys, sentence } : null;
}

/** The fit's card's block: every code-or-amount reading, as the card shows it (none confirmed yet). */
export function codesOffer(a: Answers): { keys: string[]; sentence: Md } | null {
  const keys = CODE_KEYS.filter((k) => !a.readings[k]);
  if (keys.length < 2 || keys.length !== CODE_KEYS.length) return null;
  return { keys, sentence: INF.readings.codes.sentence };
}

/** Answers that are part of the plan: once it is locked they stand (only stated phrases change). */
const PLAN: ReadonlySet<Action["type"]> = new Set([
  "reading",
  "block",
  "codes",
  "unit",
  "exclusions",
  "sensitivity",
  "missing",
  "exposure",
  "adjust",
  "model1",
]);

export function reduce(a: Answers, e: Action): Answers {
  const post = (node: NodeId, value: string): Answers["after"] =>
    a.locked ? [...a.after, { node, value }] : a.after;
  if (a.locked && PLAN.has(e.type)) return a;
  switch (e.type) {
    case "reading":
      if (a.readings[e.key]) return a;
      return { ...a, readings: { ...a.readings, [e.key]: "single" }, singles: [...a.singles, e.key] };
    case "block":
    case "codes": {
      const offer = e.type === "block" ? blockOffer(a, e.purpose) : codesOffer(a);
      if (!offer) return a;
      const next = { ...a.readings };
      for (const k of offer.keys) next[k] = "block";
      return { ...a, readings: next };
    }
    case "unit":
      return { ...a, unit: true };
    case "exclusions":
      return { ...a, exclusions: e.key, sensitivity: a.sensitivity.filter((s) => s !== e.key) };
    case "sensitivity":
      return {
        ...a,
        sensitivity: a.sensitivity.includes(e.key)
          ? a.sensitivity.filter((s) => s !== e.key)
          : INF.sensitivity.keys.filter((k) => k === e.key || a.sensitivity.includes(k)),
      };
    case "missing":
      return { ...a, missing: true };
    case "exposure":
      return { ...a, exposure: true };
    case "adjust":
      return { ...a, adjustment: { ...a.adjustment, [e.group]: e.answers } };
    case "energy":
      if (e.method === a.energy) return a;
      return { ...a, energy: e.method, after: post("energy", e.method) };
    case "form":
      if (e.form === a.form) return a;
      return { ...a, form: e.form, after: post("form", e.form) };
    case "model1":
      return { ...a, model1: e.value };
    case "lock": {
      if (a.locked || !lockReady(a).ready) return a;
      return { ...a, locked: { energy: a.energy, form: a.form, key: lockKey(a) } };
    }
    case "undo":
      if (a.locked) return a; // the lock is never undone; changes after it are recorded as such
      switch (e.node) {
        case "readings":
          return { ...a, readings: {}, singles: [], unit: false };
        case "exclusions":
          return { ...a, exclusions: null, sensitivity: [] };
        case "missing":
          return { ...a, missing: false };
        case "exposure":
          return { ...a, exposure: false, adjustment: {} };
        case "adjustment":
          return { ...a, adjustment: {} };
        case "model1":
          return { ...a, model1: null };
        case "energy":
          return { ...a, energy: INITIAL.energy };
        case "form":
          return { ...a, form: INITIAL.form };
        default:
          return a;
      }
    case "p_exclusions":
      return { ...a, pExclusions: true };
    case "p_missing":
      return { ...a, pMissing: true };
    case "p_seal":
      return { ...a, pSeal: e.holdout };
    case "reset":
      return INITIAL;
  }
}

// ── the adjustment set ──────────────────────────────────────────────────────

export const ADJ_FIELDS = ["causes_exposure", "causes_outcome", "after_exposure"] as const;

/** The scenario's adjustment decisions as the engine recorded them, in order: the columns each
 *  answered, their three answers (the fixture's declared truth, truths.py) and its sentence. The
 *  fixture keeps them under `unguessed`: every group was answered column by column from the truth,
 *  the guessed groups included (their truth is the pack's guess). */
const RECORDED = INF.adjustment.unguessed;

/** The fixture's declared causal truth for a column (truths.py), as the scenario answered it. */
export function truthTriple(col: string): Triple | null {
  for (const u of RECORDED) if (u.columns.includes(col)) return u.answers as Triple;
  return null;
}

export function guessTriple(group: string): Triple | null {
  const g = INF.adjustment.groups.find((x) => x.key === group)?.guess;
  return g ? [g.causes_exposure, g.causes_outcome, g.after_exposure] : null;
}

export const derive = (t: Triple) => INF.adjustment.derive[t.join(",")]!;

/** Each recorded group's sentences: the scenario's own (the same answers, so the same decisions),
 *  or one per column (any other answer), each the engine's own. */
export function adjustmentSentences(group: string, answers: Record<string, Triple>): Md[] {
  const g = INF.adjustment.groups.find((x) => x.key === group)!;
  const mine = RECORDED.filter((u) => u.columns.every((col) => g.columns.includes(col)));
  const covered = g.columns.every((col) => mine.some((u) => u.columns.includes(col)));
  if (covered && mine.every((u) => u.columns.every((col) => answers[col]?.join() === u.answers.join())))
    return mine.map((u) => u.sentence);
  return g.columns.map((col) => {
    const idx = INF.adjustment.per_column.by_column[col]![answers[col]!.join(",")]!;
    return INF.adjustment.per_column.sentences[idx]!;
  });
}

/** Whether every recorded answer derives what the scenario's do (only those fits were captured). */
export function adjustmentAsCaptured(a: Answers): boolean {
  for (const g of INF.adjustment.groups) {
    const rec = a.adjustment[g.key];
    if (!rec) return false;
    for (const c of g.columns) {
      const mine = derive(rec[c]!);
      const theirs = derive(truthTriple(c)!);
      if (mine.role !== theirs.role || mine.adjusted !== theirs.adjusted || mine.secondary !== theirs.secondary)
        return false;
    }
  }
  return true;
}

/** Where each covariate goes under the recorded answers (null: not answered yet). */
export function routeOf(a: Answers, col: string): "model" | "secondary" | "left_out" | null {
  for (const g of INF.adjustment.groups) {
    if (!g.columns.includes(col)) continue;
    const rec = a.adjustment[g.key];
    if (!rec) return null;
    const d = derive(rec[col]!);
    return d.adjusted ? "model" : d.secondary ? "secondary" : "left_out";
  }
  return null;
}

// ── tiers and objectives ────────────────────────────────────────────────────

/** The decisions asked of you, in the order "Next asked" walks them. */
export const ASKED: Record<Purpose, NodeId[]> = {
  inference: ["readings", "exclusions", "missing", "exposure", "adjustment", "model1"],
  prediction: ["readings", "exclusions", "p_missing", "p_seal"],
};

export function answered(a: Answers, n: NodeId, purpose: Purpose = "inference"): boolean {
  switch (n) {
    case "readings":
      return a.unit && unconfirmed(a, purpose).length === 0;
    case "exclusions":
      return purpose === "prediction" ? a.pExclusions : a.exclusions !== null;
    case "missing":
      return a.missing;
    case "exposure":
      return a.exposure;
    case "adjustment":
      return INF.adjustment.groups.every((g) => !!a.adjustment[g.key]);
    case "model1":
      return a.model1 !== null;
    case "p_missing":
      return a.pMissing;
    case "p_seal":
      return a.pSeal !== null;
    default:
      return false;
  }
}

export function tierOf(a: Answers, n: NodeId, purpose: Purpose): Tier {
  if (n === "source") return "anchor";
  if (n === "lock") return "gate";
  if (n === "estimate" || n === "matter" || n === "p_score") return "result";
  if (ASKED[purpose].includes(n)) return answered(a, n, purpose) ? "recorded" : "asked";
  if (n === "p_energy" || n === "p_models") return "waiting";
  return "stated";
}

export function objectives(a: Answers, purpose: Purpose): { node: NodeId; done: boolean }[] {
  return ASKED[purpose].map((node) => ({ node, done: answered(a, node, purpose) }));
}

export function lockReady(a: Answers): { ready: boolean; waiting: NodeId[] } {
  const waiting = ASKED.inference.filter((n) => !answered(a, n));
  return { ready: waiting.length === 0, waiting };
}

// ── the captured fit a plan selects ─────────────────────────────────────────

const ENERGY_CODES = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density"];
/** The form in a lock key: d — none recorded (the engine's straight line); l — linear; s — spline. */
const FORM_CODE = (form: string | null) => (form === null ? "d" : form[0]!);

/** trim.py's key for a plan's SHA-256: exclusions · form · energy method · the screens beside it
 *  (a bit mask over the sensitivity keys) · Model 1 (g guess, e none, u unanswered). */
export function lockKey(a: Answers): string {
  const mask = INF.sensitivity.keys.reduce((m, k, i) => (a.sensitivity.includes(k) ? m | (1 << i) : m), 0);
  const m1 = a.model1 === "guess" ? "g" : a.model1 === "empty" ? "e" : "u";
  return `${a.exclusions === "willett_2013_by_sex" ? "w" : "n"}${FORM_CODE(a.form)}${ENERGY_CODES.indexOf(a.energy)}-${mask}${m1}`;
}

export type FitPick =
  | { kind: "fit"; fit: Fit; key: string }
  | { kind: "error"; message: Md; key: string }
  | { kind: "uncaptured"; reason: string };

/** The captured fit for the answers. With no form recorded the engine fits a straight line: the
 *  fixture keeps the scenario's own fit under the linear key (trim.py checks its numbers are the
 *  side project's straight line's), and the side project's for the other states. */
export function fitFor(a: Answers): FitPick {
  if (a.exclusions !== "none" && a.exclusions !== "willett_2013_by_sex")
    return { kind: "uncaptured", reason: "this exclusion screen" };
  if (!adjustmentAsCaptured(a)) return { kind: "uncaptured", reason: "these adjustment answers" };
  if (!ENERGY_CODES.includes(a.energy) || (a.form !== null && a.form !== "linear" && a.form !== "spline"))
    return { kind: "uncaptured", reason: "this form" };
  const key = `${a.exclusions}|${a.form ?? "linear"}|${a.energy}`;
  const fit = INF.fits[key];
  if (!fit) return { kind: "uncaptured", reason: "this combination" };
  if (fit.error) return { kind: "error", message: fit.error, key };
  return { kind: "fit", fit, key };
}

export function lockSentence(a: Answers): Md | null {
  if (!a.locked) return null;
  const dg = INF.lock.digests[a.locked.key];
  return dg ? INF.lock.template.replace("{digest}", dg) : null;
}

// ── the record: the methods text, by region ─────────────────────────────────

export interface Line {
  node: NodeId;
  /** The engine's sentence, or null for a slot still asked / a section still waiting. */
  text: Md | null;
  tier: Tier;
  after?: boolean;
}

function stepReason(key: string, purpose: Purpose): Md | null {
  const steps = purpose === "inference" ? INF.steps : PRED.steps;
  return steps.find((s) => s.key === key)?.reason ?? null;
}

/** The skipped question's reason, as the engine states it (a clause; the record capitalizes it). */
export function statedReason(key: string, purpose: Purpose): Md | null {
  const r = stepReason(key, purpose);
  return r ? r.charAt(0).toUpperCase() + r.slice(1) : null;
}

/** The readings' lines, in the engine's order: the roles as set, the kcal unit, the readings
 *  confirmed one at a time, the block, the fit's codes block (under inference); each sentence the
 *  one the purpose's own drive recorded or would. */
function readingsLines(a: Answers, purpose: Purpose): Line[] {
  const inf = purpose === "inference";
  const out: Line[] = [{ node: "readings", text: inf ? INF.stated.roles : PRED.stated.roles, tier: "stated" }];
  if (a.unit) out.push({ node: "readings", text: inf ? INF.readings.unit.sentence : PRED.readings.unit_sentence, tier: "recorded" });
  const asked = readingKeys(purpose);
  const single = inf ? INF.readings.single : PRED.readings.single;
  for (const k of a.singles) {
    const s = asked.includes(k) ? single[k] : undefined;
    if (s) out.push({ node: "readings", text: s, tier: "recorded" });
  }
  const block = (inf ? ROLE_KEYS : asked).filter((k) => a.readings[k] === "block");
  if (block.length) {
    const s = blockSentence(block, purpose);
    if (s) out.push({ node: "readings", text: s, tier: "recorded" });
  }
  if (purpose === "inference" && CODE_KEYS.length && CODE_KEYS.every((k) => a.readings[k] === "block"))
    out.push({ node: "readings", text: INF.readings.codes.sentence, tier: "recorded" });
  if (!answered(a, "readings", purpose)) out.push({ node: "readings", text: null, tier: "asked" });
  return out;
}

export function record(a: Answers, purpose: Purpose): { region: string; lines: Line[] }[] {
  if (purpose === "prediction") return predictionRecord(a);
  const L = (node: NodeId, text: Md | null, tier: Tier, after = false): Line => ({ node, text, tier, after });
  const sections: { region: string; lines: Line[] }[] = [];
  const declaredEnergy = a.locked ? a.locked.energy : a.energy;
  const declaredForm = a.locked ? a.locked.form : a.form;
  const afterLines = (node: NodeId): Line[] =>
    a.after
      .filter((x) => x.node === node)
      .map((x) =>
        L(
          node,
          (node === "energy" ? INF.energy.after[x.value] : INF.form.after[x.value]) ?? null,
          "recorded",
          true,
        ),
      );
  for (const r of REGIONS.inference) {
    const lines: Line[] = [];
    for (const n of r.nodes) {
      switch (n) {
        case "source":
          lines.push(L("lens", INF.stated.lens, "stated"));
          lines.push(L("outcome", INF.stated.target, "stated"));
          lines.push(L("purpose", INF.stated.purpose, "stated"));
          lines.push(L("grain", statedReason("grain", "inference"), "stated"));
          break;
        case "readings":
          lines.push(...readingsLines(a, "inference"));
          break;
        case "exclusions":
          if (a.exclusions === null) lines.push(L(n, null, "asked"));
          else {
            lines.push(L(n, INF.exclusions.sentences[a.exclusions] ?? null, "recorded"));
            const sk = a.sensitivity.length ? a.sensitivity.join("+") : null;
            if (sk) lines.push(L(n, INF.sensitivity.sentences[sk] ?? null, "recorded"));
          }
          break;
        case "seal":
          lines.push(L(n, INF.stated.split, "stated"));
          break;
        case "clusters":
          lines.push(L(n, statedReason("clusters", "inference"), "stated"));
          break;
        case "exposure":
          lines.push(L(n, a.exposure ? INF.estimand.sentence : null, a.exposure ? "recorded" : "asked"));
          break;
        case "adjustment": {
          if (!a.exposure) {
            lines.push(L(n, null, "waiting"));
            break;
          }
          for (const g of INF.adjustment.groups) {
            const rec = a.adjustment[g.key];
            if (!rec) continue;
            for (const s of adjustmentSentences(g.key, rec)) lines.push(L(n, s, "recorded"));
          }
          if (!answered(a, "adjustment")) lines.push(L(n, null, "asked"));
          break;
        }
        case "energy":
          lines.push(L(n, INF.energy.sentences[declaredEnergy] ?? null, "stated"));
          lines.push(...afterLines("energy"));
          break;
        case "form":
          // No form recorded: the engine fits a straight line and the record says nothing of it.
          if (declaredForm !== null) lines.push(L(n, INF.form.sentences[declaredForm] ?? null, "recorded"));
          lines.push(...afterLines("form"));
          break;
        case "missing":
          lines.push(L(n, a.missing ? INF.missing.sentence : null, a.missing ? "recorded" : "asked"));
          break;
        case "model1":
          lines.push(
            L(n, a.model1 ? INF.model_1.sentences[a.model1] : null, a.model1 ? "recorded" : "asked"),
          );
          break;
        case "family":
          lines.push(L(n, INF.models_sentence, "stated"));
          break;
        case "lock":
          lines.push(L(n, lockSentence(a), a.locked ? "recorded" : "waiting"));
          break;
        default:
          break;
      }
    }
    sections.push({ region: r.id, lines });
  }
  return sections;
}

function predictionRecord(a: Answers): { region: string; lines: Line[] }[] {
  const L = (node: NodeId, text: Md | null, tier: Tier): Line => ({ node, text, tier });
  return REGIONS.prediction.map((r) => {
    const lines: Line[] = [];
    for (const n of r.nodes) {
      if (n === "source") {
        lines.push(L("lens", PRED.stated.lens, "stated"));
        lines.push(L("outcome", PRED.stated.target, "stated"));
        lines.push(L("purpose", PRED.stated.purpose, "stated"));
        lines.push(L("grain", statedReason("grain", "prediction"), "stated"));
      } else if (n === "readings") lines.push(...readingsLines(a, "prediction"));
      else if (n === "exclusions") lines.push(L(n, a.pExclusions ? PRED.exclusions.sentence_none : null, a.pExclusions ? "recorded" : "asked"));
      else if (n === "clusters") lines.push(L(n, statedReason("clusters", "prediction"), "stated"));
      else if (n === "p_missing") lines.push(L(n, a.pMissing ? PRED.missing.sentence_impute : null, a.pMissing ? "recorded" : "asked"));
      else if (n === "p_seal") lines.push(L(n, a.pSeal ? (PRED.seal.sentences[a.pSeal] ?? null) : null, a.pSeal ? "recorded" : "asked"));
      else lines.push(L(n, null, "waiting"));
    }
    return { region: r.id, lines };
  });
}

/** The first sentence of an engine text (the models sentence carries the readings list after it,
 *  which the readings card shows as its own list). */
export function firstSentence(text: Md): Md {
  const i = text.indexOf(". ");
  return i > 0 ? text.slice(0, i + 1) : text;
}

// ── previews: each is true of the state it was taken on ─────────────────────

type PreviewGroup = "exclusions" | "energy" | "split" | "missing";
type RefView = { ref: number; from: string | null };

function basePreviews(group: PreviewGroup): Record<string, Preview | null> {
  return group === "split" ? INF.seal.previews : INF[group].previews;
}

function resolveViews(p: Preview, group: PreviewGroup, key: string): Preview {
  if (!p.result) return p;
  const views = (p.result.views as unknown as (RefView | object)[]).map((v) => {
    if (!("ref" in v)) return v;
    const src = v.from ? INF.previews_var[v.from]?.[group]?.[key] : null;
    const from = src ? resolveViews(src, group, key) : basePreviews(group)[key];
    return from!.result!.views[v.ref]!;
  });
  return { result: { ...p.result, views: views as PreviewResult["views"] } };
}

/** The captured preview of an option, taken on the recordable state nearest the answers (the
 *  exclusions and the exposure's form are the only recorded answers a preview here depends on;
 *  with no form recorded, or a straight line, the base previews are the scenario's own). */
export function previewFor(group: PreviewGroup, key: string, a: Answers): Preview | null {
  const base = basePreviews(group)[key] ?? null;
  const state = `${a.exclusions === "willett_2013_by_sex" ? "willett_2013_by_sex" : "none"}|${a.form === "spline" ? "spline" : "linear"}`;
  const v = INF.previews_var[state]?.[group]?.[key];
  return v ? resolveViews(v, group, key) : base;
}

// ── previews taken on the scenario's own plan ──────────────────────────────

/** Whether the answers keep the scenario's rows and form (every row, no form recorded or a
 *  straight line): the state the estimand's, the adjustment's and Model 1's previews were taken on. */
export function onScenarioPlan(a: Answers): boolean {
  return (a.exclusions === null || a.exclusions === "none") && (a.form === null || a.form === "linear");
}

/** The exposure and its estimand, drawn (null: not on the state it was taken on). */
export function estimandPreview(a: Answers): Preview | null {
  return onScenarioPlan(a) ? INF.estimand.preview : null;
}

/** The scenario's answers for one adjustment group. */
export function scenarioAnswers(group: string): Record<string, Triple> {
  const g = INF.adjustment.groups.find((x) => x.key === group)!;
  return Object.fromEntries(g.columns.map((col) => [col, truthTriple(col)!]));
}

const sameAnswers = (x: Record<string, Triple> | undefined, y: Record<string, Triple>) =>
  !!x && Object.keys(y).every((col) => x[col]?.join() === y[col]!.join());

/** A group's answers, drawn: captured for the scenario's answers, on the state the groups before it
 *  (in the card's order) leave with theirs and the ones after it unanswered. */
export function adjustmentPreview(a: Answers, group: string, answers: Record<string, Triple>): Preview | null {
  if (!onScenarioPlan(a) || !sameAnswers(answers, scenarioAnswers(group))) return null;
  const groups = INF.adjustment.groups;
  const at = groups.findIndex((g) => g.key === group);
  for (const [i, g] of groups.entries()) {
    if (i === at) continue;
    const rec = a.adjustment[g.key];
    if (i < at ? !sameAnswers(rec, scenarioAnswers(g.key)) : rec) return null;
  }
  return INF.adjustment.previews[group] ?? null;
}

/** The declared sequence with or without Model 1, drawn. */
export function model1Preview(a: Answers, value: "guess" | "empty"): Preview | null {
  return onScenarioPlan(a) ? INF.model_1.previews[value] : null;
}

/** An exposure form, drawn on the scenario's rows from no form recorded (the state it was taken on;
 *  or, once that form is recorded, the change that recorded it). */
export function formPreview(a: Answers, form: string): Preview | null {
  const rows = a.exclusions === null || a.exclusions === "none";
  return rows && (a.form === null || a.form === form) ? (INF.form.previews[form] ?? null) : null;
}

/** The rows a fit reports on, and the rows the exclusions keep, for the ribbon. */
export function rowsKept(a: Answers): { n: number; dropped: number } {
  const base = INF.exclusions.n_base;
  if (!a.exclusions || a.exclusions === "none") return { n: base, dropped: 0 };
  const off = INF.exclusions.offered.find((o) => o.key === a.exclusions);
  return off ? { n: base - off.affected, dropped: off.affected } : { n: base, dropped: 0 };
}
