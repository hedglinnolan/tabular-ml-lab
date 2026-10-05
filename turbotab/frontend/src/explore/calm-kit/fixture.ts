/**
 * The calm kit's data: every question, option, sentence, preview and fit the four structures show,
 * captured from the engine on the one scenario (../methods-shared/SCENARIO.md). capture/capture.py
 * takes what the earlier fixtures lacked; capture/build.py assembles this file from it and from
 * the methods map's and the paper's captures. Every preview names its `source`; the numbers the
 * engine does not serve directly (the Strip's) are described in `derived`.
 */
import type {
  ConsequenceView,
  DistributionView,
  HistogramData,
  LineageView,
  RelationshipView,
  RowFlowView,
  TableFocusView,
} from "../../api/m1-stage-types";
import raw from "./fixture.json";

export type { ConsequenceView, DistributionView, HistogramData, LineageView, RelationshipView, RowFlowView, TableFocusView };

export type StageId = "data" | "participants" | "exposure" | "confounders" | "energy" | "model" | "results";
export type SectionId = "participants" | "variables" | "measurement" | "statistics" | "results";
export type QuietLabel = "Recommended" | "Common practice" | "Not available yet" | "Not available";

/** One column of the Strip: an energy model's change to one nutrient (fixture `derived.strip`). */
export interface StripColumn {
  column: string;
  output: string;
  /** The engine's change score (consequences._shifts): the standardized Wasserstein distance. */
  shift: number;
  r_before: number | null;
  r_after: number | null;
  sd_before: number;
  sd_after: number;
  mean_before: number;
  mean_after: number;
  hist_before: { edges: number[]; counts: number[] };
  hist_after: { edges: number[]; counts: number[] };
}

/** One declared tradeoff: a question, and the view, text or list that answers it. */
export interface Angle {
  question: string;
  /** Index into the preview's views. */
  view?: number;
  text?: string;
  list?: { name: string; ok: boolean; why?: string }[];
  /** This option's answer, for the option-by-question table. */
  cell: string;
  /** Columns the choice touches in this angle's view (drawn in the choice color). */
  touch?: string[];
}

export interface Preview {
  source: string;
  basis: string;
  /** The engine's note when nothing can be drawn (rule 7: one line, no empty chart). */
  note: string | null;
  views: ConsequenceView[];
  strip?: StripColumn[];
  angles?: Angle[];
  refusal?: { code: string; message: string } | null;
  withheld?: string;
}

export interface Option {
  id: string;
  name: string;
  /** Its one-line consequence (≤ 16 words). */
  what: string;
  label?: QuietLabel;
  disabled?: boolean;
  /** The engine's refusal, for a disabled option. */
  refusal?: string;
  /** The engine's methods sentence this answer records. */
  sentence: string | null;
  preview: Preview;
}

export interface Step {
  id: string;
  stage: StageId;
  section: SectionId;
  /** The manuscript's run-in head. */
  head: string;
  question: string;
  lede: string;
  why: string;
  legend: string;
  options: Option[];
  /** The scenario's answer. */
  scenario: string;
  /** The manuscript slot this step fills (the three estimand steps share one). */
  slot: string;
  /** The captured previews hold only when these earlier answers do. */
  requires?: Record<string, string[]>;
}

export interface EffectRow {
  feature: string;
  estimate: number;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
}

export interface SequenceRow {
  key: string;
  label: string;
  note: string;
  n_rows: number;
  adjusted_for: string[];
  effects: EffectRow[];
  inference: string;
  concerns: string[];
}

export interface Fit {
  error?: string;
  rows: string;
  measure_label: string;
  caption: string;
  sequence: SequenceRow[];
  methods: string;
  sensitivity: {
    methods: string | null;
    concerns: string[];
    analyses: { label: string; primary: boolean; added: boolean; rules: string[]; n_rows: number; refused: string | null; effects: EffectRow[] }[];
  };
}

export interface Fixture {
  meta: {
    file: string;
    rows: number;
    cols: number;
    captured: string;
    map_captured: string;
    paper_captured: string;
    scenario: string;
    sources: Record<string, string>;
  };
  derived: Record<string, string>;
  chain: { id: StageId; label: string }[];
  sections: { id: SectionId; title: string; item: string }[];
  stated: { id: string; section: SectionId; head: string; sentence: string }[];
  steps: Step[];
  /** "<exposure>|<effect>|<contrast>" → the engine's estimand sentence. */
  estimand_sentences: Record<string, string>;
  /** "<exclusions>|<energy model>" → the captured fit (the scenario's other answers). */
  fits: Record<string, Fit>;
  lock: { template: string; digests: Record<string, string>; energy_codes: string[]; sensitivity_keys: string[] };
}

export const FX = raw as unknown as Fixture;
export const STEPS: Step[] = FX.steps;
export const STEP_BY_ID: Record<string, Step> = Object.fromEntries(STEPS.map((s) => [s.id, s]));

export function stepOf(id: string): Step {
  const s = STEP_BY_ID[id];
  if (!s) throw new Error(`no step ${id}`);
  return s;
}

export function optionOf(step: Step, id: string | null | undefined): Option | null {
  return (id && step.options.find((o) => o.id === id)) || null;
}
