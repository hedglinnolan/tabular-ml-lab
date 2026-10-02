/**
 * The Record's half of the purpose registry (BLUEPRINT §11.2, M2_CONTRACT §11): every component
 * the Record exports declares which of the user's five questions it answers, in one short
 * sentence about the user's data. `purposes.test.ts` fails when an exported component has no
 * entry, or an entry names a component that is gone.
 *
 * The stage keeps the other half (src/components/stage/purposes.ts: view kinds, coach anchors,
 * stage elements) with the same shapes; the M1 entries below are worded as it words them, so the
 * two merge as one table. `SplitAsk` is gone: the split question is `SealAsk` (M2 §3).
 */

export type QuestionId = "what" | "change" | "matters" | "data_ok" | "provenance";

export const QUESTIONS: Record<QuestionId, string> = {
  what: "What is this choice?",
  change: "What will it change in my data or my model?",
  matters: "Why does that matter for my result?",
  data_ok: "Is my data okay?",
  provenance: "What did I decide, and can a reviewer reproduce it?",
};

/** What an entry says, in at most this many words. */
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
