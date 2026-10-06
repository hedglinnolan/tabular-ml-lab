/**
 * The scenario walk: one state machine every structure uses, from the first draft to the locked
 * Table 2 and "Which of my decisions mattered?". Pure: state in, state out. What a structure shows
 * (a card, a slot in the manuscript, a node on a map) is derived from it here, so the four
 * structures cannot disagree about an answer, a sentence or a number.
 *
 * The leash (FOUNDATION §5 rule 6): `results` is null until the plan is locked, so no structure
 * can show an outcome-model estimate before the lock.
 */
import { fmtCI, fmtEst } from "../methods-shared/results";
import { FX, STEPS, STEP_BY_ID, optionOf, type Fit, type SectionId, type StageId, type Step } from "./fixture";

/** The two result moments after the lock. */
export const RESULT_STEPS = ["table2", "mattered"] as const;
export type ResultStep = (typeof RESULT_STEPS)[number];
/** Every moment of the walk, in order. */
export const ORDER: string[] = [...STEPS.map((s) => s.id), ...RESULT_STEPS];
export const PLAN_STEPS: string[] = STEPS.map((s) => s.id).filter((id) => id !== "lock");

export interface WalkState {
  /** step id → the recorded option id. */
  answers: Record<string, string>;
  /** Step ids in the order they were (last) recorded: the newest is last. */
  order: string[];
  /** The step on the card (any of ORDER). */
  open: string;
  locked: boolean;
  /** The plan's fit key at the lock ("<exclusions>|<energy>"). */
  lockedKey: string | null;
  /** Steps whose answer changed after the estimates were seen. */
  afterLock: string[];
}

export type Action =
  | { type: "record"; step: string; option: string }
  | { type: "open"; step: string }
  | { type: "next" }
  | { type: "reset" };

export function initial(): WalkState {
  return { answers: {}, order: [], open: ORDER[0]!, locked: false, lockedKey: null, afterLock: [] };
}

export const isResult = (id: string): id is ResultStep => (RESULT_STEPS as readonly string[]).includes(id);

/** The first plan step not yet answered (the lock once every one is), or Table 2 once locked. */
export function frontier(s: WalkState): string {
  const first = PLAN_STEPS.find((id) => !(id in s.answers));
  if (first) return first;
  return s.locked ? "table2" : "lock";
}

/** A step can be opened when it is answered or no later than the frontier; results once locked. */
export function reachable(s: WalkState, id: string): boolean {
  if (isResult(id)) return s.locked;
  if (id in s.answers) return true;
  return ORDER.indexOf(id) <= ORDER.indexOf(frontier(s));
}

/** The earlier answers a step's captured previews depend on, where they differ. */
export function blockedBy(s: WalkState, id: string): { step: string; got: string }[] {
  const req = STEP_BY_ID[id]?.requires;
  if (!req) return [];
  return Object.entries(req).flatMap(([step, want]) => {
    const got = s.answers[step];
    return got !== undefined && !want.includes(got) ? [{ step, got }] : [];
  });
}

export function reduce(s: WalkState, a: Action): WalkState {
  switch (a.type) {
    case "reset":
      return initial();
    case "open":
      return reachable(s, a.step) ? { ...s, open: a.step } : s;
    case "next": {
      if (s.open === "table2") return { ...s, open: "mattered" };
      return { ...s, open: frontier(s) };
    }
    case "record": {
      const step = STEP_BY_ID[a.step];
      const option = step ? optionOf(step, a.option) : null;
      if (!step || !option || option.disabled || blockedBy(s, a.step).length || !reachable(s, a.step)) return s;
      if (a.step === "lock") {
        const p = plan(s);
        if (!p.ok || s.locked) return s;
        const answers = { ...s.answers, lock: "lock" };
        return { ...s, answers, order: [...s.order, "lock"], locked: true, lockedKey: p.key, open: "table2" };
      }
      const changed = s.answers[a.step] !== a.option;
      const answers = { ...s.answers, [a.step]: a.option };
      const order = [...s.order.filter((x) => x !== a.step), a.step];
      const afterLock = s.locked && changed && !s.afterLock.includes(a.step) ? [...s.afterLock, a.step] : s.afterLock;
      const next = { ...s, answers, order, afterLock };
      return { ...next, open: frontier(next) };
    }
  }
}

// ── the chain ────────────────────────────────────────────────────────────────

export type StageStatus = "done" | "current" | "waiting";

export function stageOfStep(id: string): StageId {
  return isResult(id) ? "results" : STEP_BY_ID[id]!.stage;
}

export function stepsOfStage(stage: StageId): string[] {
  return ORDER.filter((id) => stageOfStep(id) === stage);
}

export function chain(s: WalkState): { id: StageId; label: string; status: StageStatus; first: string }[] {
  const current = stageOfStep(s.open);
  return FX.chain.map((c) => {
    const ids = stepsOfStage(c.id);
    const done = c.id === "results" ? false : ids.every((id) => id in s.answers);
    const status: StageStatus = c.id === current ? "current" : done ? "done" : "waiting";
    return { id: c.id, label: c.label, status, first: ids[0]! };
  });
}

/** "Participants · step 2 of 3". */
export function stepLabel(id: string): string {
  const stage = stageOfStep(id);
  const ids = stepsOfStage(stage);
  const label = FX.chain.find((c) => c.id === stage)!.label;
  return `${label} · step ${ids.indexOf(id) + 1} of ${ids.length}`;
}

// ── the plan, its lock and its results ───────────────────────────────────────

/** The answers a captured fit may differ in; every other answer must be the scenario's. */
const FREE: Record<string, string[] | "any"> = {
  exclusions: ["none", "willett_2013_by_sex"],
  energy: FX.lock.energy_codes,
  sensitivity: "any",
};

export type Plan =
  | { ok: true; key: string; fit: Fit }
  | { ok: false; missing: string[]; differs: { step: string; got: string; want: string }[] };

export function plan(s: WalkState): Plan {
  const missing = PLAN_STEPS.filter((id) => !(id in s.answers));
  const differs = PLAN_STEPS.flatMap((id) => {
    const got = s.answers[id];
    if (got === undefined) return [];
    const free = FREE[id];
    const step = STEP_BY_ID[id]!;
    if (free === "any" || (free && free.includes(got)) || got === step.scenario) return [];
    return [{ step: id, got, want: step.scenario }];
  });
  if (missing.length || differs.length) return { ok: false, missing, differs };
  const key = `${s.answers.exclusions}|${s.answers.energy}`;
  const fit = FX.fits[key];
  if (!fit || fit.error) return { ok: false, missing: [], differs: [{ step: "energy", got: s.answers.energy!, want: "standard" }] };
  return { ok: true, key, fit };
}

const SENSITIVITY_MASK: Record<string, string[]> = {
  both: ["willett_2013_by_sex", "nhs_hpfs_by_sex"],
  willett: ["willett_2013_by_sex"],
  nhs: ["nhs_hpfs_by_sex"],
  none: [],
};

/** The plan's SHA-256 (twelve hex digits), as the engine recorded it for these answers. */
export function digest(s: WalkState): string | null {
  const keys = SENSITIVITY_MASK[s.answers.sensitivity ?? ""] ?? [];
  const mask = FX.lock.sensitivity_keys.reduce((m, k, i) => (keys.includes(k) ? m | (1 << i) : m), 0);
  const excl = s.answers.exclusions === "willett_2013_by_sex" ? "w" : "n";
  const e = FX.lock.energy_codes.indexOf(s.answers.energy ?? "");
  return FX.lock.digests[`${excl}d${e}-${mask}g`] ?? null;
}

export function lockSentence(s: WalkState): string | null {
  const d = s.locked ? digest(s) : null;
  return d ? FX.lock.template.replace("{digest}", d) : null;
}

export interface T2Row {
  key: string;
  label: string;
  adjustedFor: string[];
  n: number;
  estimate: number;
  lo: number | null;
  hi: number | null;
  primary: boolean;
  feature: string;
}

export interface MatteredRow {
  key: string;
  /** Its plain name: a model's, or the exclusion rule's as the card offered it. */
  label: string;
  /** The rule's technical name (the quiet second register), for a check on other rows. */
  term?: string;
  /** What it varies from the primary: the adjustment (the model sequence) or the rows (a screen). */
  varies: "primary" | "adjustment" | "rows";
  n: number;
  estimate: number;
  lo: number | null;
  hi: number | null;
}

export interface Results {
  fit: Fit;
  table2: T2Row[];
  mattered: MatteredRow[];
  /** The primary model's interval note ("95% intervals from HC3 …, on t(21,830). …"). */
  footnote: string;
}

const SCREEN_LABEL: Record<string, string> = {
  willett_2013_by_sex: "Willett 2013, by sex",
  nhs_hpfs_by_sex: "NHS/HPFS, by sex",
};

/** The engine's label for a check's rows ("Willett 2013, by sex", "Every row") → the exclusion
 *  option the card offered for the same rule, so the results use the names the user chose from. */
const SCREEN_OPTION: Record<string, string> = {
  "Willett 2013, by sex": "willett_2013_by_sex",
  "NHS/HPFS, by sex": "nhs_hpfs_by_sex",
  "Every row": "none",
};

export function screenName(label: string): { name: string; term?: string } {
  const o = optionOf(STEP_BY_ID.exclusions!, SCREEN_OPTION[label]);
  return o ? { name: o.name, term: o.term } : { name: label };
}

/** Table 2 and the declared alternatives: null until the plan is locked (the leash). */
export function results(s: WalkState): Results | null {
  if (!s.locked || !s.lockedKey) return null;
  // After the lock an answer may change (marked as made after the estimates were seen); the
  // results follow it when the engine's fit for the new plan was captured.
  const p = plan(s);
  const fit = p.ok ? p.fit : FX.fits[s.lockedKey]!;
  const table2: T2Row[] = fit.sequence.map((row) => {
    const e = row.effects[0]!;
    return {
      key: row.key,
      label: row.label,
      adjustedFor: row.adjusted_for,
      n: row.n_rows,
      estimate: e.estimate,
      lo: e.ci_low,
      hi: e.ci_high,
      primary: row.key === "model_2",
      feature: e.feature,
    };
  });
  const declared = (SENSITIVITY_MASK[s.answers.sensitivity ?? ""] ?? []).map((k) => SCREEN_LABEL[k]);
  const mattered: MatteredRow[] = table2.map((r) => ({
    key: r.key,
    label: r.label,
    varies: r.primary ? "primary" : "adjustment",
    n: r.n,
    estimate: r.estimate,
    lo: r.lo,
    hi: r.hi,
  }));
  for (const a of fit.sensitivity.analyses) {
    if (a.primary || a.refused || !(a.added || declared.includes(a.label))) continue;
    const e = a.effects[0];
    if (!e) continue;
    const { name, term } = screenName(a.label);
    mattered.push({ key: `screen:${a.label}`, label: name, term, varies: "rows", n: a.n_rows, estimate: e.estimate, lo: e.ci_low, hi: e.ci_high });
  }
  const primary = fit.sequence.find((r) => r.key === "model_2")!;
  return { fit, table2, mattered, footnote: primary.inference };
}

/** The marks the cross-structure check reads (e2e/calm-protos.spec.ts). */
export function t2Attrs(r: T2Row): Record<string, string> {
  return { "data-t2-row": r.key, "data-t2-estimate": fmtEst(r.estimate), "data-t2-ci": fmtCI(r.lo, r.hi) };
}

export function matteredAttrs(r: MatteredRow): Record<string, string> {
  return { "data-mattered-row": r.key, "data-estimate": fmtEst(r.estimate), "data-ci": fmtCI(r.lo, r.hi) };
}

// ── the manuscript ───────────────────────────────────────────────────────────

export type EntryKind = "stated" | "recorded" | "blank" | "waiting";

export interface Entry {
  /** The slot (a step's slot, a stated decision, or a result). */
  id: string;
  kind: EntryKind;
  head: string;
  /** The engine's sentence (backticks mark data values); null for a blank. */
  sentence: string | null;
  /** The step a click opens (a blank answers it; a recorded phrase changes it). */
  step: string | null;
  newest: boolean;
  afterLock: boolean;
}

export interface Section {
  id: SectionId;
  title: string;
  item: string;
  entries: Entry[];
}

const ESTIMAND_STEPS = ["exposure", "effect", "contrast"];

function sentenceOf(s: WalkState, step: Step): string | null {
  if (step.slot === "estimand") {
    const [x, e, c] = ESTIMAND_STEPS.map((id) => s.answers[id]);
    return x && e && c ? (FX.estimand_sentences[`${x}|${e}|${c}`] ?? null) : null;
  }
  if (step.id === "lock") return lockSentence(s);
  return optionOf(step, s.answers[step.id])?.sentence ?? null;
}

export function manuscript(s: WalkState): Section[] {
  const newest = s.order[s.order.length - 1] ?? null;
  const seen = new Set<string>();
  return FX.sections.map((sec) => {
    const entries: Entry[] = FX.stated
      .filter((x) => x.section === sec.id)
      .map((x) => ({ id: `stated:${x.id}`, kind: "stated", head: x.head, sentence: x.sentence, step: null, newest: false, afterLock: false }));
    for (const step of STEPS) {
      if (step.section !== sec.id || seen.has(step.slot)) continue;
      seen.add(step.slot);
      const members = STEPS.filter((x) => x.slot === step.slot).map((x) => x.id);
      const sentence = sentenceOf(s, step);
      const open = members.find((id) => !(id in s.answers)) ?? members[members.length - 1]!;
      const kind: EntryKind = sentence ? "recorded" : reachable(s, open) ? "blank" : "waiting";
      entries.push({
        id: step.slot,
        kind,
        head: step.head,
        sentence,
        step: open,
        newest: !!newest && members.includes(newest),
        afterLock: members.some((id) => s.afterLock.includes(id)),
      });
    }
    if (sec.id === "results") {
      const r = results(s);
      entries.push(
        r
          ? { id: "results:table2", kind: "recorded", head: "Main results", sentence: r.fit.caption, step: "table2", newest: false, afterLock: false }
          : { id: "results:table2", kind: "waiting", head: "Main results", sentence: null, step: null, newest: false, afterLock: false },
      );
    }
    return { ...sec, entries };
  });
}

/** How many sentences the manuscript holds now (stated and recorded). */
export function sentenceCount(s: WalkState): number {
  return manuscript(s).reduce((n, sec) => n + sec.entries.filter((e) => e.sentence).length, 0);
}

/** The scenario's answer for every plan step, in order: what each walker clicks. */
export const SCENARIO_ANSWERS: [string, string][] = STEPS.map((s) => [s.id, s.scenario]);
