/**
 * The calm kit: what every competing structure (the Q&A card, the paper, the quest log, the map)
 * is built from, so the only difference between them is how the analysis is organized on the
 * screen (FOUNDATION §6).
 *
 *   const walk = useWalk();                       // the one walk model and the card's moment
 *   <Shell walk={walk} name="The paper" manuscript="column"
 *          card={<Card walk={walk} />} canvas={<Canvas walk={walk} />} />
 *
 * A structure may arrange the parts its own way (open any reachable step with walk.open, show the
 * manuscript as a rail, a column or its own surface, show the chain or not); it never restyles a
 * part or writes its own copy for a question, an option, a sentence or a number.
 */
import "./tokens.css";
import "./base.css";

export { FX, STEPS, STEP_BY_ID, optionOf, stepOf } from "./fixture";
export type { Angle, Fixture, Option, Preview, QuietLabel, SectionId, StageId, Step, StripColumn } from "./fixture";
export {
  ORDER,
  PLAN_STEPS,
  RESULT_STEPS,
  SCENARIO_ANSWERS,
  blockedBy,
  chain,
  digest,
  frontier,
  initial,
  isResult,
  lockSentence,
  manuscript,
  matteredAttrs,
  plan,
  reachable,
  reduce,
  results,
  sentenceCount,
  stageOfStep,
  stepLabel,
  stepsOfStage,
  t2Attrs,
} from "./walk";
export type { Action, Entry, MatteredRow, Plan, Results, Section, T2Row, WalkState } from "./walk";
export { footprint, route, viewsFor } from "./router";
export type { Footprint, Layout } from "./router";
export { useWalk, storyLength } from "./useWalk";
export type { Flip, WalkApi } from "./useWalk";
export { Canvas, CanvasFrame, readoutOf } from "./canvas/Canvas";
export { Angles, Strip, View, Views } from "./canvas/layouts";
export { Mattered, Table2 } from "./canvas/Results";
export { Cells, FlowBars, Hist, Lineage, Scatter } from "./canvas/views";
export {
  Card,
  Chain,
  Continue,
  Footer,
  Hint,
  ManuscriptBody,
  ManuscriptColumn,
  ManuscriptOverlay,
  ManuscriptRail,
  OptionList,
  Shell,
  ThemeSwitch,
  Why,
  useManuscriptRail,
} from "./parts";
export { Plain, plain } from "./text";
export { default as kit } from "./kit.module.css";
