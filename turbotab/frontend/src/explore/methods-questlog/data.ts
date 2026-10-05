/**
 * The real-data fixture for /lab/methods-questlog, typed for what the prototype reads.
 *
 * Every sentence, guess, piece of evidence and number below is the server's own, captured by
 * `capture_drive.py` (the NHANES export through the real server, answered from the fixture's
 * declared truth) and cut down by `trim.py`. Nothing here is typed by hand.
 */
import raw from "./fixture.json";
import type { PreviewResult } from "../../api/m1-stage-types";
import type { Decision, ProjectView } from "../../api/schema";

export interface Exit {
  label: string;
  decision: (Decision & Record<string, unknown>) | null;
}

export interface AskGroup {
  kind: string;
  columns: string[];
  guess: string;
  guess_words: string;
  evidence: string;
}

export interface ReadFromData {
  kind: string;
  column: string;
  value: string;
  words: string;
  evidence: string;
  change: Exit[];
}

export interface StepAsk {
  question: string;
  consumer: string;
  text: string;
  groups: AskGroup[];
  exits: Exit[];
  read_from_data: ReadFromData[];
}

export interface Step {
  key: string;
  status: "answered" | "open" | "waiting" | "skipped" | "not_applicable";
  decision_id: string | null;
  reason: string | null;
  waiting_on: string[];
  ask: StepAsk | null;
}

export interface MethodsLine {
  record_id: string;
  seq: number;
  kind: string;
  sentence: string;
  in_force: boolean;
  post_seal: boolean;
  after_estimates: boolean;
}

export interface StageResultFx {
  stage: string;
  key: string | null;
  fresh: boolean;
  status: string;
  artifact: Record<string, unknown>;
}

export interface Moment {
  view: Omit<ProjectView, "interview"> & { interview: Step[] };
  methods: { lines: MethodsLine[]; seen_from: number | null; text: string };
  stages: Record<string, StageResultFx>;
}

export interface RoleProposal {
  column: string;
  linked_to: string | null;
  unit: string | null;
  nested_in: string | null;
  proposed: string;
  confidence: "high" | "medium" | "low";
  reason: string;
  attention: boolean;
}

export interface CapturedPreview {
  status: number;
  decision: Decision & Record<string, unknown>;
  body: PreviewResult | { error: { code: string; message: string; exits: Exit[] } };
}

export interface TeachingTerm {
  term: string;
  definition: string;
}

export interface TeachingEntry {
  key: string;
  title: string;
  question: string;
  one_liner: string;
  why: string;
  consumer: string;
  options: { value: string; label: string; consequence: string }[];
  terms: TeachingTerm[];
  drawer: { sections: { heading: string; body: string; evidence: { status: string; source: string } }[] } | null;
}

export interface Coef {
  feature: string;
  estimate: number;
  ci_low: number;
  ci_high: number;
  p: number | null;
  meaning?: string | null;
  why?: string | null;
}

export interface SequenceFit {
  key: string;
  label: string;
  adjusted_for: string[];
  note: string;
  n_rows: number;
  effects: Coef[];
  concerns: string[];
}

export interface AdjustmentGroup {
  key: string;
  label: string;
  columns: string[];
  guess: Record<string, string> | null;
  reason: string;
  derived: string | null;
  derived_words: string | null;
  decision: (Decision & Record<string, unknown>) | null;
}

export interface Fixture {
  meta: Record<string, string>;
  teaching: TeachingEntry[];
  inference: {
    moments: Record<"m1" | "m2" | "m3" | "m4" | "m5" | "m6", Moment>;
    roles: { columns: RoleProposal[]; needs_confirmation: string[] };
    readings: { read_from_data: ReadFromData[] };
    asks: Record<"models_0" | "models_1" | "sensitivity_0", { code: string; message: string; exits: Exit[] }>;
    singles: { decision: Decision & Record<string, unknown>; record: { sentence: string; seq: number } }[];
    estimand_card: {
      exposures: { column: string; energy_contrast: boolean }[];
      effects: { effect: string; label: string; consequence: string }[];
      contrasts: { contrast: string; label: string; consequence: string }[];
      measures: {
        measure: string;
        label: string;
        fitted: boolean;
        reason: string;
        conditioning: string;
        rank: number | null;
      }[];
      family: { n: number } | null;
    };
    adjustment_card: {
      exposure: string;
      questions: Record<string, string>;
      groups: AdjustmentGroup[];
      source: string;
    };
    previews: Record<string, CapturedPreview>;
    evidence: Record<string, PreviewResult>;
    fit: {
      n_train: number;
      models: { coefficients: Coef[]; adjustment_terms: Coef[]; inference: { caption: string }; concerns: string[] }[];
      estimand: { caption: string; appendix: string; adjusted: string[]; left_out: Record<string, string>; secondary: string[] };
    };
    effects: {
      appendix_title: string;
      rows: string;
      measure_label: string;
      families: { family: string; label: string; sequence: SequenceFit[] }[];
      methods: string;
    };
    secondary_methods: string;
    sensitivity: {
      analyses: { label: string; primary: boolean; rules: string[]; n_rows: number }[];
      fits: { label: string; n_rows: number; coefficient: Coef }[];
      methods: string;
    };
    plan: { declared_at: string; plan_sha256: string; sha256: string; status: string; text: string; through_record: number };
  };
  prediction: {
    moment: Moment;
    fit: Record<string, unknown>;
  };
}

export const FX = raw as unknown as Fixture;
export const INF = FX.inference;

export function teaching(key: string): TeachingEntry | undefined {
  return FX.teaching.find((t) => t.key === key);
}

export function term(name: string): TeachingTerm | undefined {
  for (const t of FX.teaching) {
    const hit = t.terms.find((x) => x.term === name);
    if (hit) return hit;
  }
  return undefined;
}

export function isPreview(b: CapturedPreview["body"]): b is PreviewResult {
  return "views" in b;
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/** A small signed number with a real minus sign, at a fixed number of decimals. */
export function fmtEst(v: number, digits = 4): string {
  return v.toFixed(digits).replace("-", "−");
}
