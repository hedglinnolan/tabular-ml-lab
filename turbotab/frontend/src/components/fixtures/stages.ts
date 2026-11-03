/**
 * The engine's stage words, captured from it by capture_stages.py (its docstring says what each
 * is): every stage's label as the graph declares it, and stage statuses as the engine serves them
 * on the typed fit-time refusal test's toy graph.
 */
import type { StageStatus } from "../../api/schema";
import captured from "./stages.json";

/** Each stage's label, by stage name, as core/stages/__init__.py `build_graph` declares it. */
export const ENGINE_LABELS: Record<string, string> = captured.labels;

/** R refuses in words (a decision and a way forward that is none); RH refuses so in a job worker
 *  and AR waits on it; C and CH fail with the same words untyped and AC waits on CH; U waits on
 *  an unsettled reading, its two confirmations its ways forward. */
export const SERVED = captured.statuses as unknown as Record<
  "R" | "RH" | "AR" | "C" | "CH" | "AC" | "U",
  StageStatus
>;
