/**
 * The walk: how a person's clicks move the document along the scenario's captured moments
 * (methods-shared/SCENARIO.md). Each moment has one act — the slot whose own control records the
 * scenario's answer — and the moment that answer leads to. The answers themselves are read from the
 * decisions the server recorded (the locked moment), never restated here, so a slot can say which
 * choice this captured path records and hold any other.
 */
import { artifactOf, FX, moment, PATH, type MomentId } from "./fixture";

export interface Act {
  /** The paragraph whose slot records this step (a Para id). */
  slot: string;
  /** The moment the recorded answer leads to. */
  next: MomentId;
}

export const ACTS: Partial<Record<MomentId, Act>> = {
  draft: { slot: "roles", next: "roles" },
  // The screens ask `kcal`'s unit before the rows can be screened: a reading, in the Data section.
  roles: { slot: "readings", next: "exclusions" },
  exclusions: { slot: "exclusions", next: "missing" },
  missing: { slot: "missing", next: "split" },
  split: { slot: "split", next: "readings" },
  // Three readings one at a time, then the unlocked block (BLUEPRINT §11.4 rule 4).
  readings: { slot: "readings", next: "single-bp_di" },
  "single-bp_di": { slot: "readings", next: "single-bp_sys" },
  "single-bp_sys": { slot: "readings", next: "single-cycle_begin_year" },
  "single-cycle_begin_year": { slot: "readings", next: "estimand" },
  estimand: { slot: "estimand", next: "adjustment" },
  adjustment: { slot: "adjustment", next: "energy" },
  energy: { slot: "energy_adjustment", next: "model_sequence" },
  model_sequence: { slot: "model_sequence", next: "models" },
  // The fit's card: whether `age` and `cycle_begin_year` hold amounts or codes, one block.
  models: { slot: "readings", next: "codes" },
  codes: { slot: "models", next: "ready" },
  // Showing the estimates is the one act left, and it locks the plan.
  ready: { slot: "results-table2", next: "locked" },
};

/** The column a single confirmation records at this moment (the next moment is its result). */
export function nextSingle(id: MomentId): string | null {
  const next = ACTS[id]?.next;
  return next?.startsWith("single-") ? next.slice("single-".length) : null;
}

/** The step a moment is on the walk (1-based), and how many there are. */
export function stepOf(id: MomentId): { at: number; of: number } {
  return { at: PATH.indexOf(id) + 1, of: PATH.length };
}

type Rec = { kind: string } & Record<string, unknown>;

function recorded(kind: string): Rec[] {
  return moment("locked").view.decisions.map((d) => d.decision as Rec).filter((d) => d.kind === kind);
}

export interface Captured {
  /** The primary rows: the exclusions option key ("keep_every_row" when no rule). */
  rows: string;
  /** The screens declared beside the primary, as exclusions option keys. */
  beside: string[];
  missing: string;
  /** The split, as the teaching's option value ("0" for no holdout). */
  split: string;
  estimand: { exposure: string; effect: string; contrast: string };
  energy: string;
  model1: string[];
  models: string[];
  /** The answers to the three questions for each covariate the pack does not guess. */
  adjustment: Record<string, [string, string, string]>;
}

/** The scenario's answers, as the server recorded them on the locked moment. */
export function captured(): Captured {
  const labels = artifactOf<{ labels: { exclusions?: { options: { key: string; label: string }[] } } | null }>(
    moment("exclusions"),
    "proposals",
  )?.labels;
  const options = labels?.exclusions?.options ?? [];
  const exclusions = recorded("set_exclusions").at(-1) as { rules?: unknown[] } | undefined;
  const sensitivity = recorded("set_sensitivity").at(-1) as { analyses?: { label: string }[] } | undefined;
  const missing = recorded("set_missing").at(-1) as { strategy?: string } | undefined;
  const split = recorded("set_split").at(-1) as { holdout?: number } | undefined;
  const estimand = recorded("set_estimand").at(-1) as { exposure?: string; effect?: string; contrast?: string } | undefined;
  const energy = recorded("set_energy_adjustment").at(-1) as { method?: string } | undefined;
  const sequence = recorded("set_model_sequence").at(-1) as { model_1?: string[] } | undefined;
  const models = recorded("select_models").at(-1) as { models?: string[] } | undefined;
  return {
    rows: exclusions?.rules?.length ? "screened" : "keep_every_row",
    beside: (sensitivity?.analyses ?? []).flatMap((a) => options.filter((o) => o.label === a.label).map((o) => o.key)),
    missing: missing?.strategy ?? "",
    split: String(split?.holdout ?? ""),
    estimand: { exposure: estimand?.exposure ?? "", effect: estimand?.effect ?? "", contrast: estimand?.contrast ?? "" },
    energy: energy?.method ?? "",
    model1: sequence?.model_1 ?? [],
    models: models?.models ?? [],
    adjustment: FX.adjustmentAnswers,
  };
}

export const sameSet = (a: Iterable<string>, b: Iterable<string>): boolean => {
  const x = [...new Set(a)].sort();
  const y = [...new Set(b)].sort();
  return x.length === y.length && x.every((v, i) => v === y[i]);
};
