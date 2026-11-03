/**
 * What each Estimate declaration card posts, as pure functions of the card's fields: the Model 1
 * question (SetModelSequence), how many tests are accounted for (SetMultiplicity) and the
 * regression calibration question (SetMeasurementError), each the whole of its model in
 * turbotab/core/decisions.py.
 *
 * The types are the generated contract's, so a field the server adds or drops breaks the build
 * here, and `Exactly` fails to compile if a list misses or invents a value the model accepts. The
 * bound and the defaults are the models'; cards.test.tsx holds the lists, the bound and the
 * defaults to turbotab/server/openapi.json, which the server's own test holds to the models.
 */
import type { ProposalsArtifact } from "../../../api/m1-types";
import type { Decision } from "../../../api/schema";
import type { Bounds } from "../predict/payloads";

type Of<K extends Decision["kind"]> = Extract<Decision, { kind: K }>;
export type ModelSequence = Of<"set_model_sequence">;
export type Multiplicity = Of<"set_multiplicity">;
export type MeasurementError = Of<"set_measurement_error">;

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

// ── the Model 1 question (set_model_sequence) ──

/** Model 1's declaration as the engine serves it (core/estimand.py `model_sequence_card`, the
 *  proposals stage's `model_sequence`): the columns declared, the guess, the columns Model 1 may
 *  hold, the one-tap decision that declares the guess, and the engine's reason. */
export type ModelOneServed = NonNullable<ProposalsArtifact["model_sequence"]>;

export function modelSequence(exposure: string, model1: readonly string[]): ModelSequence {
  return { kind: "set_model_sequence", exposure, model_1: [...model1] };
}

/** The served one-tap decision, as the model it is, or null where it is not one. */
export function servedSequence(served: ModelOneServed): ModelSequence | null {
  const d = served.decision;
  const model1 = d.model_1;
  if (d.kind !== "set_model_sequence" || typeof d.exposure !== "string" || d.exposure === "") return null;
  if (!Array.isArray(model1) || !model1.every((c): c is string => typeof c === "string")) return null;
  return modelSequence(d.exposure, model1);
}

// ── how many tests are accounted for (set_multiplicity) ──

const multiplicityMethods = ["bh", "stated_count", "none"] as const;
export const MULTIPLICITY_METHODS: Exactly<typeof multiplicityMethods, Multiplicity["method"]> =
  multiplicityMethods;

/** `acknowledged` is never asked: it is the attestation that keeps an answer the engine blocks
 *  and records, which the engine's refusal offers as its own way forward. */
export const multiplicity = (method: Multiplicity["method"]): Multiplicity => ({
  kind: "set_multiplicity",
  method,
  acknowledged: false,
});

// ── the regression calibration question (set_measurement_error) ──

const calibrationMethods = ["none", "regression_calibration"] as const;
export const CALIBRATION_METHODS: Exactly<
  typeof calibrationMethods,
  MeasurementError["method"]
> = calibrationMethods;

/** The numeric field's bounds (decisions.py `SetMeasurementError.n_boot`). */
export const BOUNDS = {
  set_measurement_error: { n_boot: { ge: 50, le: 2000 } },
} as const satisfies Record<string, Record<string, Bounds>>;

/** The model's defaults for the fields the card does not ask (decisions.py). */
export const DEFAULTS = {
  set_measurement_error: {
    exposures: [] as string[],
    n_boot: 200,
    interval: "whole_chain_bootstrap",
  },
} as const;

/**
 * The bootstrap's refits are read by regression calibration alone: no calibration sends the
 * model's defaults, as the engine's own "Record no calibration" does (decisions.py). The intakes
 * reported are never chosen here: the empty list is every exposure the recalls measure (the
 * model's default). The interval is always the bootstrap over the whole chain, since the engine
 * refuses the model-based and Rubin-only intervals for a calibrated coefficient. `adjustment` is
 * the server's to fill from the state, never the client's.
 */
export function measurementError(
  method: MeasurementError["method"],
  nBoot: number,
): MeasurementError {
  const d = DEFAULTS.set_measurement_error;
  return {
    kind: "set_measurement_error",
    method,
    exposures: [...d.exposures],
    n_boot: method === "regression_calibration" ? nBoot : d.n_boot,
    interval: d.interval,
    adjustment: null,
  };
}
