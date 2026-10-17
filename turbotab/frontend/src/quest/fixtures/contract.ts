/**
 * The engine's side of the shell's contract (quest-contract.json), written by
 * turbotab/core/tests/test_quest_renderer_contract.py from the path fuzzer's sample datasets: every
 * quest line's (source, key) under every goal, the stages and each question's and declaration's
 * place and name, the sweep words, two quest logs of the NHANES fixture under Predict with what
 * "Confirm all" records in each, and the Fit lock as the engine words it held and pressed. Read by
 * the shell's tests and by the mock server; never edited by hand.
 */
import type { FitLock, QuestLog, SweptLineInput } from "../../api/schema";
import raw from "./quest-contract.json";

export interface ContractPair {
  source: string;
  key: string;
  stages: string[];
  labels: string[];
  /** `<dataset>/<goal>` where the line was seen. */
  seen: string[];
}

export interface Place {
  stage: string;
  label: "Decide" | "Confirm" | "For the record";
  order: number;
  name: string;
}

export interface Contract {
  seeds: number;
  datasets: string[];
  goals: string[];
  pairs: ContractPair[];
  stages: { key: string; name: string }[];
  questions: Record<string, Place>;
  declarations: Record<string, Place & { counted: boolean }>;
  /** Each sweep size's heading and action ("Here are the 3 other choices set for you", "Confirm all 3"). */
  sweep_words: Record<string, [string, string]>;
  logs: Record<"models_open" | "results_reached", { log: QuestLog; swept: Record<string, SweptLineInput[]> }>;
  fit_locks: {
    estimate_seconds: number;
    held: FitLock;
    pressed: FitLock;
    /** Not held, not pressed: under Predict, a fit that computes before the press. */
    ready: FitLock;
    /** Under Estimate, before the press. */
    unlocked: FitLock;
    no_goal: FitLock;
  };
  /** The floor a score's gain over the no-predictor baseline must reach (core/models/baseline.py). */
  min_gain: number;
}

export const CONTRACT = raw as unknown as Contract;
