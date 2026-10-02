/**
 * The purpose registry (BLUEPRINT §11.2, M2_CONTRACT §11): every element the app puts on screen
 * answers a question the user has at that moment. There are five such questions; each view kind,
 * coach anchor kind, stage element and Record component declares which one it answers.
 *
 * `purposes.test.ts` fails when a rendered kind or component has no entry here: a new kind arrives
 * with a purpose or not at all. The server keeps the same registry for the view kinds it sends
 * (turbotab/core/purposes.py). The pedagogy reviewer audits each entry once; it then holds for every
 * decision that uses the kind (the closed vocabulary, §11 rule 2).
 */
import type { ViewKind } from "../../api/m1-stage-types";
import type { CoachAnchorKind } from "../../api/m2-stage-types";

export type QuestionId = "what" | "change" | "matters" | "data_ok" | "provenance";

export const QUESTIONS: Record<QuestionId, string> = {
  what: "What is this choice?",
  change: "What will it change in my data or my model?",
  matters: "Why does that matter for my result?",
  data_ok: "Is my data okay?",
  provenance: "What did I decide, and can a reviewer reproduce it?",
};

/** What an entry shows, in at most this many words. */
export const ANSWER_WORDS = 16;

export interface Purpose {
  question: QuestionId;
  /** What the element shows, on the user's data. */
  answer: string;
}

/** An element with nothing of its own on screen (a provider, a row of buttons it lays out). */
export interface Structural {
  structural: string;
}

// ── the stage's views ────────────────────────────────────────────────────────

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]] ? ([L[number]] extends [U] ? L : never) : never;

const serverViewKinds = ["row_flow", "lineage", "table_focus", "distribution", "relationship"] as const;
/** Every view kind the server sends (a regenerated union that gains one fails to compile here). */
export const SERVER_VIEW_KINDS: Exactly<typeof serverViewKinds, ViewKind> = serverViewKinds;

const anchorKinds = ["column", "range", "points", "step"] as const;
export const COACH_ANCHOR_KINDS: Exactly<typeof anchorKinds, CoachAnchorKind> = anchorKinds;

/** The pictures the stage composes from the vocabulary's views (composed.ts). */
export type ComposedKind = "reshape_table" | "turn_table" | "seal_fork";

export const VIEW_PURPOSES: Record<ViewKind | ComposedKind, Purpose> = {
  row_flow: { question: "change", answer: "which rows stay in the analysis, step by step, and how many each step removes" },
  lineage: { question: "change", answer: "which columns enter the model, and what each one becomes on the way" },
  table_focus: { question: "change", answer: "what the choice writes into real rows, cell by cell" },
  distribution: { question: "change", answer: "how one column's values move, with the cut-offs marked on the axis" },
  relationship: { question: "matters", answer: "how two columns move together before and after, which is what the model sees" },
  reshape_table: {
    question: "change",
    answer: "how each unit's records gather and become one row, and which file rows it stands for",
  },
  turn_table: { question: "what", answer: "which way round the table is: each sample column becomes one row" },
  seal_fork: { question: "matters", answer: "which rows are held out, and whether one unit sits on both sides of the seal" },
};

export const COACH_ANCHOR_PURPOSES: Record<CoachAnchorKind, Purpose> = {
  column: { question: "data_ok", answer: "a fact about one column of this table the picture cannot say alone" },
  range: { question: "data_ok", answer: "how many rows sit in a stretch of the axis, and what that likely means" },
  points: { question: "matters", answer: "which rows in the picture drive what the choice does" },
  step: { question: "matters", answer: "what one step of the row flow costs, and whom it removes" },
};

/** The stage's other elements, by their `data-purpose` attribute. */
export const STAGE_PURPOSES: Record<string, Purpose> = {
  coach_note: { question: "data_ok", answer: "a fact about the user's rows, pointing at the part of the picture it is about" },
  stage_note: { question: "what", answer: "what a choice does when there is no picture of it to draw" },
  stage_caution: { question: "data_ok", answer: "a concern with this choice, beside the control that resolves it" },
  readout: { question: "change", answer: "the headline numbers, as recorded and with this choice" },
  refusal: { question: "what", answer: "why a choice is not available, and the ways out" },
  live_rows: { question: "provenance", answer: "the participant flow as recorded: rows at each step, and the seal" },
  live_columns: { question: "provenance", answer: "the column lineage as recorded, from raw columns to the model matrix" },
  live_results: { question: "what", answer: "why nothing is fitted yet, and which answers the fit waits on" },
  seal_basis: { question: "provenance", answer: "how the held-out rows were drawn, named on the seal itself" },
  model_comparison: { question: "matters", answer: "how each model family scores against the outcome's own average" },
  coefficients: { question: "matters", answer: "what the fitted linear models say each exposure does to the outcome" },
  substitution: { question: "matters", answer: "what moving energy between two nutrients does to the outcome, by model" },
  open_seal: { question: "what", answer: "what opening the held-out rows does, and that it happens once" },
  seal_opened: { question: "provenance", answer: "when the held-out scores were fixed in the record" },
  post_seal: { question: "provenance", answer: "which numbers changed after the held-out scores were seen" },
};

// ── the Record ───────────────────────────────────────────────────────────────

/** Every component exported under src/components/record, by name. */
export const RECORD_PURPOSES: Record<string, Purpose | Structural> = {
  // ── M1 ──
  Record: { structural: "lays out the questions, the decisions and the findings in asking order" },
  Question: {
    question: "what",
    answer: "the one open question, its one-line why, and its options",
  },
  QuestionBlock: { question: "what", answer: "an open question with its kicker and one-line why" },
  DecisionSentence: {
    question: "provenance",
    answer: "a recorded decision as the sentence a methods section would print",
  },
  SkipRow: {
    question: "provenance",
    answer: "a question not asked, with the reason from the data and a way to ask",
  },
  History: {
    question: "provenance",
    answer: "the earlier answers to a question, kept rather than overwritten",
  },
  Pending: { question: "what", answer: "a question that waits, and what it waits on" },
  Options: {
    question: "change",
    answer: "each option with its one-line consequence; focusing one previews it on the stage",
  },
  ColumnPicker: {
    question: "what",
    answer: "the table's columns to choose from, with each one's summary",
  },
  FindingsCards: {
    question: "data_ok",
    answer: "what the checks found in this table, each with its lever",
  },
  RefusalNote: { question: "what", answer: "why an answer was refused, and the ways forward" },
  FailureNote: {
    question: "data_ok",
    answer: "that a request failed, said plainly, with a way to try again",
  },
  LensAsk: {
    question: "what",
    answer: "which research lens the table belongs to, and what each unlocks",
  },
  TargetAsk: { question: "what", answer: "which column is the outcome" },
  TaskAsk: {
    question: "what",
    answer: "what kind of outcome it is: a number, two levels, or several",
  },
  PurposeAsk: { question: "what", answer: "whether the model is for prediction or for inference" },
  RolesAsk: {
    question: "change",
    answer: "what each column is to the model: exposure, covariate, identifier, left out",
  },
  ExclusionsAsk: {
    question: "change",
    answer: "which rows are eligible, asked in scientific terms",
  },
  MissingAsk: { question: "change", answer: "what happens to rows with a blank predictor" },
  EnergyAsk: { question: "matters", answer: "how nutrients are adjusted for total energy" },
  ModelsAsk: {
    question: "change",
    answer: "which model families are fitted, with each one's concerns",
  },
  SubstitutionAsk: {
    question: "matters",
    answer: "which nutrient pair the substitution curve moves energy between",
  },
  TermsProvider: { structural: "makes the teaching's terms define themselves on hover or focus" },
  Taught: {
    question: "what",
    answer: "a term with a dotted underline that defines itself in one sentence",
  },
  EvidenceBadge: {
    question: "provenance",
    answer: "how settled the science behind a claim is, and its source",
  },
  ConceptDrawer: {
    question: "matters",
    answer: "the deeper concept behind a question, sourced and badged, never required",
  },
  Keep: { question: "provenance", answer: "keeping the recorded answer as it is" },
  Actions: { structural: "lays out a question's action buttons" },
  RecordButton: {
    question: "what",
    answer: "records the chosen option, saying what it will record",
  },

  // ── M2: the opening sequence ──
  OrientationAsk: {
    question: "what",
    answer: "which way round the table is, read from its shape, before any check runs",
  },
  EventAsk: {
    question: "what",
    answer: "which level of a two-level outcome the models give the probability of",
  },
  GrainAsk: {
    question: "what",
    answer: "whether one person can appear in more than one row, and which column names them",
  },
  RepeatKindAsk: {
    question: "what",
    answer: "whether a person's rows repeat one measurement or follow them over time",
  },
  UnitAsk: {
    question: "change",
    answer: "what one analyzed row is, with the rows each answer leads to",
  },
  AggregationAsk: {
    question: "change",
    answer: "how each person's rows become one, the recommended way first with its reason",
  },
  TemporalAsk: {
    question: "matters",
    answer: "whether later rows are predicted from earlier ones, which orders the seal by time",
  },
  StatedSkip: {
    question: "provenance",
    answer: "a question the data already answers, with the reading and a way to be asked",
  },

  // ── M2: the seal ──
  SealAsk: {
    question: "matters",
    answer: "how many rows are sealed, what that many can measure, and the seal's basis",
  },
  SealBasisLine: {
    question: "provenance",
    answer: "how the held-out rows are drawn: grouped, by row, abandoned or undetermined",
  },
  SealGlyph: {
    question: "provenance",
    answer: "the seal's basis at a glance, never a clean lock when it is undetermined",
  },
  OpenSealCard: {
    question: "what",
    answer: "what opening the held-out rows does, and that it happens once",
  },

  // ── M2: findings with repairs ──
  RepairsSection: {
    question: "data_ok",
    answer: "how values are written wrongly, each with its repairs previewed on the rows first",
  },
  Resurfaced: {
    question: "provenance",
    answer: "a finding set aside for this question, back pre-checked with its repair",
  },
};

export function isStructural(p: Purpose | Structural): p is Structural {
  return "structural" in p;
}

/** The purpose of a view, a composed picture or a coach anchor, if it has one. */
export function purposeOf(kind: string): Purpose | undefined {
  return (
    (VIEW_PURPOSES as Record<string, Purpose>)[kind] ??
    (COACH_ANCHOR_PURPOSES as Record<string, Purpose>)[kind] ??
    STAGE_PURPOSES[kind]
  );
}
