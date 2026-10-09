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

export type StageId = "data" | "participants" | "columns" | "exposure" | "confounders" | "energy" | "model" | "results";
export type SectionId = "participants" | "variables" | "measurement" | "statistics" | "results";
export type QuietLabel = "Recommended" | "Common practice" | "Not available yet" | "Not available";

/** One column of the Strip: an energy model's change to one nutrient (fixture `derived.strip`). */
export interface StripColumn {
  column: string;
  output: string;
  /** What the engine's recognizer reads the name as ("monounsaturated fat"), shown on hover. */
  desc?: string;
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
  /** The column's own views under this choice (its scatter against total calories with the
   *  storyboard, and its values before and after), as the engine draws the one nutrient it pictures
   *  (fixture `derived.strip_views`): the Strip's focus moves every view to it (FOUNDATION §5 rule 4). */
  views?: ConsequenceView[];
  /** The canvas caption when this column is focused, in the card's register. */
  caption?: string;
}

/** One declared tradeoff: one question and the one picture that answers it (FOUNDATION §5). */
export interface Angle {
  question: string;
  /** Its column in the option table (a word or two). */
  head: string;
  /** The picture: an index into the preview's views … */
  view?: number;
  /** … or a list of marks (what can follow), each with its reason on hover. */
  list?: { name: string; ok: boolean; why?: string }[];
  /** This option's answer, for the option table (absent at rest). */
  cell?: string;
  /** Columns the choice touches in this angle's view (drawn in the choice color). */
  touch?: string[];
}

export interface Preview {
  source: string;
  basis: string;
  /** The canvas caption in the card's plain register, restating the primary view's (absent: the view's own);
   *  for an option not available, what it would do and why not. */
  caption?: string;
  /** The engine's note when nothing can be drawn (rule 7: one line, no empty chart). */
  note: string | null;
  views: ConsequenceView[];
  strip?: StripColumn[];
  angles?: Angle[];
  refusal?: { code: string; message: string } | null;
  withheld?: string;
  /** The views draw analyses reported beside the main one (the checks): the main analysis keeps
   *  everyone, so who a check leaves out never leaves the analysis. */
  beside?: boolean;
}

export interface Option {
  id: string;
  /** Its plain name (the card's register, FOUNDATION §2). */
  name: string;
  /** Its one-line consequence (≤ 16 words); for an option not available, why not. */
  what: string;
  label?: QuietLabel;
  /** Its technical name, the quiet second register: shown when the option is pointed at. */
  term?: string;
  disabled?: boolean;
  /** The engine's refusal, for a disabled option. */
  refusal?: string;
  /** The engine's methods sentence this answer records. */
  sentence: string | null;
  preview: Preview;
}

/** The layouts "your data now" is drawn in: the layout the question's options use. */
export type NowLayout = "focus" | "strip" | "flow" | "routing" | "angles";

/** The canvas at rest (FOUNDATION §5 rule 8): the question's columns as they are now, in gray. */
export interface Now {
  layout: NowLayout;
  caption: string;
  basis: string;
  source: string;
  /** The Strip's heading at rest. */
  title?: string;
  /** Drawn in their before state. */
  views: ConsequenceView[];
  strip?: StripColumn[];
  angles?: Angle[];
  /** The columns the question is about: named in the picture even where others are grouped. */
  columns?: string[];
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
  /** Your data now, for the canvas at rest. */
  now: Now;
  /** Your data now when it follows an earlier answer (the lock: the Model 1 answer's models). */
  now_by?: { step: string; options: Record<string, Now> };
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
    /** When capture.py's in-process numbers (strip_views, block_rows) were last taken. */
    derived_captured?: string | null;
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
  /** Scatter points the Strip's per-column views share, by key (a view names them as "@p:<key>"). */
  points?: Record<string, [number, number][]>;
}

export const FX = raw as unknown as Fixture;

/** A rest view in the fixture names an option's engine view (`ref`) rather than copying it. */
type RawView = ConsequenceView | { ref: [string, number]; title?: string };

function resolveNow(step: Step, now: Now): void {
  now.views = (now.views as RawView[]).map((v) => {
    if (!("ref" in v)) return v;
    const [id, i] = v.ref;
    const view = step.options.find((o) => o.id === id)?.preview.views[i];
    if (!view) throw new Error(`${step.id}: no view ${id}/${i} for its rest picture`);
    return v.title ? ({ ...view, title: v.title } as ConsequenceView) : view;
  });
}
/** A per-column view names its shared points ("@p:<key>") rather than repeating them. */
type Points = [number, number][];
function points(p: Points | string): Points {
  if (typeof p !== "string") return p;
  const got = FX.points?.[p.replace(/^@p:/, "")];
  if (!got) throw new Error(`no points ${p}`);
  return got;
}
function resolveColumns(cols: StripColumn[] | undefined): void {
  for (const c of cols ?? [])
    for (const v of c.views ?? []) {
      if (v.kind !== "relationship") continue;
      v.points_before = points(v.points_before as Points | string);
      v.points_after = points(v.points_after as Points | string);
      for (const f of v.story ?? []) f.points = points(f.points as Points | string);
    }
}
for (const step of FX.steps) {
  resolveNow(step, step.now);
  for (const n of Object.values(step.now_by?.options ?? {})) resolveNow(step, n);
  resolveColumns(step.now.strip);
  for (const o of step.options) resolveColumns(o.preview.strip);
}

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
