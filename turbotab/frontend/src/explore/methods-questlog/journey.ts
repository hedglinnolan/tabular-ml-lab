/**
 * The walk: where a person is in the shared scenario's captured journey, and what each press
 * records. The scenario's moments (data.ts ORDER) are the server's states between answers; a
 * card's own control advances to the next one. Answers the scenario records without a moment
 * between them are held here as local steps, each shown with the sentence the engine wrote for it
 * (the adjustment card's groups, and the fit's code readings).
 *
 * Pure: the screen keeps a Walk in state; these functions read the fixture.
 */
import { INF, MOMENTS, ORDER, type MethodsLine, type Moment, type StepAsk } from "./data";

export interface Walk {
  /** Index into ORDER. */
  at: number;
  /** The adjustment card's answers recorded so far (indexes into INF.cards.adjustment_answers). */
  adjusted: number[];
  /** The fit's code-or-amount readings confirmed (as one block). */
  codes: boolean;
}

export const START: Walk = { at: 0, adjusted: [], codes: false };

/** The three role readings the scenario confirms one at a time, in the card's order. */
export const SINGLES = ["bp_di", "bp_sys", "cycle_begin_year"];

/** Single confirmations before the block confirm unlocks (the mastery rule). */
export const MASTERY = 3;

export const idOf = (w: Walk): string => ORDER[Math.min(w.at, ORDER.length - 1)]!;
export const momentOf = (w: Walk): Moment => MOMENTS[idOf(w)]!;
export const isLocked = (w: Walk): boolean => idOf(w) === "locked";

/** The next captured moment. The code readings stay confirmed until the record holds them. */
export function advance(w: Walk): Walk {
  const at = Math.min(w.at + 1, ORDER.length - 1);
  return { at, adjusted: [], codes: w.codes && !hasLine(MOMENTS[ORDER[at]!]!, isCodesLine) };
}

/** The moment `id`, as a walk standing on it (review presets). */
export function walkAt(id: string): Walk {
  const at = ORDER.indexOf(id);
  return at < 0 ? START : { ...START, at };
}

const isCodesLine = (l: MethodsLine) => l.kind === "confirm_readings" && /amounts or counts|codes for categories/.test(l.sentence);
const hasLine = (m: Moment, f: (l: MethodsLine) => boolean) => m.methods.lines.some(f);

/** The open question's ask, if the moment's first open step holds one (readings to settle). */
export function askOf(w: Walk): StepAsk | null {
  const steps = momentOf(w).view.interview;
  const first = steps.find((s) => s.status === "open" || s.status === "waiting");
  if (!first?.ask) return null;
  if (first.key === "models" && w.codes) return null;
  return first.ask;
}

/** The objective a newcomer is walked to: the item whose card advances the journey. */
export function objectiveItem(w: Walk): string | null {
  switch (idOf(w)) {
    case "draft":
      return "roles";
    case "roles":
    case "readings":
    case "single_bp_di":
    case "single_bp_sys":
    case "single_cycle_begin_year":
      return "readings";
    case "exclusions":
      return "exclusions";
    case "missing":
      return "missing";
    case "split":
      return "split";
    case "estimand":
      return "estimand";
    case "adjustment":
      return "adjustment";
    case "energy":
      return "energy_adjustment";
    case "model_sequence":
      return "model_sequence";
    case "models":
      return w.codes ? "models" : "readings";
    case "ready":
      return "lock";
    default:
      return null;
  }
}

/** The scenario's answers to the adjustment card, grouped as the card offers them: one tap for a
 *  group the pack guesses, and the truth's answers for the group it does not (each recorded as
 *  the scenario recorded it). */
export interface AdjustmentAnswer {
  /** Index into INF.cards.adjustment_answers. */
  index: number;
  /** The card's group it answers. */
  group: string;
  columns: string[];
}

export const ADJUSTMENT_ANSWERS: AdjustmentAnswer[] = INF.cards.adjustment_answers.map((a, index) => {
  const columns = Object.keys(a.answers);
  const g =
    INF.cards.adjustment.groups.find((x) => x.decision && JSON.stringify(x.decision) === JSON.stringify(a)) ??
    INF.cards.adjustment.groups.find((x) => columns.some((c) => x.columns.includes(c)));
  return { index, group: g?.key ?? "unguessed", columns };
});

/** The sentences later moments hold that this walk has already recorded locally, so the record
 *  shows each answer the moment it is given: the adjustment's groups, the fit's code readings. */
function localLines(w: Walk): MethodsLine[] {
  const id = idOf(w);
  const have = new Set(momentOf(w).methods.lines.map((l) => l.seq));
  const out: MethodsLine[] = [];
  if (id === "adjustment" && w.adjusted.length) {
    const next = MOMENTS[ORDER[w.at + 1] ?? ""];
    const adj = (next?.methods.lines ?? []).filter((l) => l.kind === "set_adjustment" && !have.has(l.seq));
    for (const i of w.adjusted) if (adj[i]) out.push(adj[i]);
  }
  if (w.codes) {
    for (let i = w.at + 1; i < ORDER.length; i++) {
      const hit = MOMENTS[ORDER[i]!]!.methods.lines.find((l) => isCodesLine(l) && !have.has(l.seq));
      if (hit) {
        out.push(hit);
        break;
      }
    }
  }
  return out;
}

/** The methods text this walk shows: the moment's sentences and the local answers' sentences. */
export function linesOf(w: Walk): MethodsLine[] {
  return [...momentOf(w).methods.lines, ...localLines(w)];
}

/** The sentence a later moment holds for a decision kind this walk has not recorded: what
 *  recording it writes (each card leads with it). */
export function nextSentence(w: Walk, kind: string): string | null {
  const have = new Set(linesOf(w).map((l) => l.seq));
  for (let i = w.at + 1; i < ORDER.length; i++) {
    const hit = MOMENTS[ORDER[i]!]!.methods.lines.find((l) => l.kind === kind && !have.has(l.seq));
    if (hit) return hit.sentence;
  }
  return null;
}

/** Single confirmations recorded so far: the mastery the block confirm unlocks with. */
export function singlesDone(w: Walk): MethodsLine[] {
  return linesOf(w).filter((l) => l.kind === "confirm_reading");
}

/** The single the scenario confirms next, while the role readings are open. */
export function nextSingle(w: Walk): string | null {
  const ask = askOf(w);
  if (!ask || ask.consumer !== "the exposure and its effect") return null;
  return SINGLES[singlesDone(w).length] ?? null;
}

// ── persistence (a convenience: the walk survives a reload; Reset clears it) ──

const STORE = "turbotab.methods-questlog.walk.v2";

export function loadWalk(): Walk | null {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return null;
    const v = JSON.parse(raw) as Partial<Walk>;
    if (typeof v.at !== "number" || v.at < 0 || v.at >= ORDER.length) return null;
    const adjusted = Array.isArray(v.adjusted) ? v.adjusted.filter((x): x is number => typeof x === "number") : [];
    return { at: v.at, adjusted, codes: !!v.codes };
  } catch {
    return null;
  }
}

export function saveWalk(w: Walk | null): void {
  try {
    if (w) localStorage.setItem(STORE, JSON.stringify(w));
    else localStorage.removeItem(STORE);
  } catch {
    /* storage unavailable: the walk lasts for this page only */
  }
}

// ── the scenario's answers, as the locked record holds them ─────────────────────

interface FinalState {
  exclusions: unknown[] | null;
  sensitivity: { label: string }[] | null;
  missing: { strategy: string } | null;
  split: { holdout: number } | null;
  energy_adjustment: { method: string } | null;
  models: string[] | null;
  estimand: { exposure: string; effect: string; contrast: string; measure: string } | null;
  model_sequence: { model_1: string[] } | null;
}

const FINAL = MOMENTS.locked!.view.state as unknown as FinalState;

/** What the shared scenario records at each card (a card's record control takes exactly this;
 *  any other answer is previewed on the canvas, never recorded). */
export const SCENARIO = {
  keepEveryRow: !FINAL.exclusions?.length,
  screens: (FINAL.sensitivity ?? []).map((x) => x.label),
  missing: FINAL.missing?.strategy ?? null,
  holdout: FINAL.split?.holdout ?? null,
  energy: FINAL.energy_adjustment?.method ?? null,
  models: FINAL.models ?? [],
  estimand: FINAL.estimand,
  model1: FINAL.model_sequence?.model_1 ?? [],
};

/** The sentence each of the adjustment card's answers writes (in the answers' order). */
export const ADJUSTMENT_SENTENCES: string[] = (() => {
  const before = new Set((MOMENTS.adjustment?.methods.lines ?? []).map((l) => l.seq));
  const after = MOMENTS[ORDER[ORDER.indexOf("adjustment") + 1] ?? ""]?.methods.lines ?? [];
  return after.filter((l) => l.kind === "set_adjustment" && !before.has(l.seq)).map((l) => l.sentence);
})();
