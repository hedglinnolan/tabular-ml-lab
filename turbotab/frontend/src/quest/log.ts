/**
 * The quest log as the shell reads it (pure): which stage is open, each stage's segment of the
 * progress bar, its lines under their labels, the open line, what "Confirm all" records and the
 * one primary action at the card's foot. Everything here reads GET /quest's fields; nothing counts
 * or scores on its own (FOUNDATION §3; plans/W1_PART2_PLAN.md §2 "W1-E").
 */
import type { FitLock, QuestLine, QuestLog, QuestStage, StageStatus, SweptLineInput } from "../api/schema";

export type Label = QuestLine["label"];
export const LABELS: readonly Label[] = ["Decide", "Confirm", "For the record"];

/** Where the analysis is, and the stage the card opens on: the first reached stage not yet
 *  complete that is not passed for the triage (`passedForTriage`: its only lines still to answer are
 *  open noticings, finding 6), else the last reached (Models, complete, before Fit), else the
 *  first. */
export function openStageKey(log: QuestLog): string {
  const reached = log.stages.filter((s) => s.reached);
  const working = reached.find((s) => !s.progress?.complete && !passedForTriage(log, s));
  return (working ?? reached[reached.length - 1] ?? log.stages[0]!).key;
}

/**
 * Whether a reached stage that is not complete is passed for the triage (finding 6): where open
 * noticings travel (`noticingsTravel`: under Predict, and before the goal is chosen), outside the
 * triage's stage (`triageStage`), its only counted Decides still to answer are noticings and no
 * Confirm sweep waits, so its Continue is enabled (`primaryFor`). Under Predict the engine leaves
 * an open noticing outside Results out of its stage's count (ruling R-a, core/quest.py
 * `quest_log`), so such a stage is served complete; before the goal is chosen the engine counts
 * those noticings in the stage's progress, so the stage reads incomplete, yet it is not where the
 * analysis is. Under Estimate nothing is passed: a noticing is decided in its own stage.
 */
export function passedForTriage(log: QuestLog, stage: QuestStage): boolean {
  if (!noticingsTravel(log) || !stage.reached || stage.key === triageStage(log)) return false;
  const sweep = stage.sweep;
  if (sweep && !sweep.answered && sweep.lines > 0) return false;
  const rest = unanswered(stage);
  return rest.length > 0 && rest.every((l) => l.source === "finding");
}

/**
 * Whether a stage reads as done on the quest line and its segment (finding 25: a stage not reached
 * never reads as done): reached and complete, and, where it asks nothing (0 of 0), come to by the
 * analysis (`cameTo`). Under Predict the engine reaches Write-up with Results, and First look with
 * the stages around it; neither reads as done while the analysis is still before it. A stage
 * whose own objectives are all answered reads as done wherever the analysis is.
 */
export function readsDone(log: QuestLog, stage: QuestStage): boolean {
  if (!stage.reached || !stage.progress?.complete) return false;
  if (stage.progress.required > 0) return true;
  return cameTo(log, stage);
}

/**
 * Whether the analysis has come to a stage (W1-R B1): it is at or before where the analysis is
 * (`openStageKey`), or at or before the furthest reached stage with an objective answered, which
 * the researcher has worked past it to. The analysis is at the first stage still working, so a
 * line left open early (Your question's intended use) would otherwise keep First look from
 * reading as done after Who's in holds answers; Write-up stays not done while Results has none.
 */
function cameTo(log: QuestLog, stage: QuestStage): boolean {
  const index = (key: string) => log.stages.findIndex((s) => s.key === key);
  const worked = log.stages.findLastIndex((s) => s.reached && (s.progress?.answered ?? 0) > 0);
  return index(stage.key) <= Math.max(index(openStageKey(log)), worked);
}

export function stageOf(log: QuestLog, key: string): QuestStage | undefined {
  return log.stages.find((s) => s.key === key);
}

/** The stage after `key` in the seven, or null after Write-up. */
export function nextStageKey(log: QuestLog, key: string): string | null {
  const i = log.stages.findIndex((s) => s.key === key);
  return i >= 0 && i + 1 < log.stages.length ? log.stages[i + 1]!.key : null;
}

/** One segment of the progress bar (FOUNDATION §3): empty when the stage is not reached (never
 *  "0 of N"); else its objectives answered of required; dropped back while a reopen stands.
 *  `complete` is the served field (GET /quest's progress, reached and complete), which the bar
 *  marks as it is (`data-complete`); `done` is whether it reads as done (`readsDone`, marked
 *  `data-done`): a reached stage that asks nothing is empty, not done, until the analysis comes to
 *  it (finding 25). */
export interface Segment {
  key: string;
  name: string;
  reached: boolean;
  answered: number | null;
  required: number | null;
  complete: boolean;
  done: boolean;
  /** Why it dropped back, in the engine's words (one per reopening). */
  reopened: string[];
}

export function segments(log: QuestLog): Segment[] {
  return log.stages.map((s) => ({
    key: s.key,
    name: s.name,
    reached: s.reached,
    answered: s.reached && s.progress ? s.progress.answered : null,
    required: s.reached && s.progress ? s.progress.required : null,
    complete: s.reached && !!s.progress?.complete,
    done: readsDone(log, s),
    reopened: s.reopened.map((r) => r.sentence),
  }));
}

/** The fraction of a segment that is filled: full once it reads as done; a stage that asks
 *  nothing is empty until then. */
export function filled(seg: Segment): number {
  if (!seg.reached || seg.answered === null || seg.required === null) return 0;
  if (seg.done) return 1;
  if (seg.required === 0) return 0;
  return Math.min(1, seg.answered / seg.required);
}

/** A stage's lines under their labels, each label's lines in the engine's order. */
export function byLabel(stage: QuestStage): Record<Label, QuestLine[]> {
  const out: Record<Label, QuestLine[]> = { Decide: [], Confirm: [], "For the record": [] };
  for (const line of [...stage.lines].sort((a, b) => a.order - b.order)) out[line.label].push(line);
  return out;
}

/** Whether a line has a card that can record its answer. The shell passes the dispatcher's rule
 *  (cards/renderers.ts `recordable`); every line can, where none is given. */
export type CanAnswer = (line: QuestLine) => boolean;
const ANY: CanAnswer = () => true;

/** Under Predict, Who's in's draw of the held-out rows: the Decide the stage's sweep comes before. */
export const DRAW = "split";

/**
 * Whether the stage's draw waits for its Confirm sweep (FOUNDATION §3, "two departures";
 * disagreement 6): under a Predict track that holds rows out, Who's in runs its sweep just before
 * the held-out rows are drawn, so the draw is the stage's last Decide. The draw is a counted Decide
 * only there (under Estimate the split is For the record), so its line is the test.
 */
export function drawWaitsForSweep(stage: QuestStage): boolean {
  const sweep = stage.sweep;
  if (!stage.reached || !sweep || sweep.answered || sweep.lines === 0) return false;
  return byLabel(stage).Decide.some((l) => l.key === DRAW && l.counted && l.status !== "answered");
}

/** The line the card shows expanded: the stage's first open Decide that a card can answer, counted
 *  ones first; never the draw while the sweep it reads is unconfirmed. */
export function openLine(stage: QuestStage, canAnswer: CanAnswer = ANY): QuestLine | null {
  const held = drawWaitsForSweep(stage);
  const decide = byLabel(stage).Decide.filter(
    (l) => l.status === "open" && canAnswer(l) && !(held && l.key === DRAW),
  );
  return decide.find((l) => l.counted) ?? decide[0] ?? null;
}

/**
 * The line whose answer settled an answered noticing, when that answer is another line's: the
 * question the noticing is decided by, answered there (core/quest.py `_finding_lines` gives the
 * noticing that question's record). Null for a noticing settled by its own disposition (dismissed,
 * repaired, set aside), whose record speaks of it alone; for a record several lines share (a
 * sweep's confirmation), which no one line can be named for; and for every other line.
 */
export function settledBy(log: QuestLog, line: QuestLine): QuestLine | null {
  if (line.source !== "finding" || line.status !== "answered" || !line.decision_id) return null;
  const by = log.stages.flatMap((s) => s.lines).filter((l) => l.source !== "finding" && l.decision_id === line.decision_id);
  return by.length === 1 ? by[0]! : null;
}

/** The counted Decide lines still to answer (open or waiting). */
export function unanswered(stage: QuestStage): QuestLine[] {
  return byLabel(stage).Decide.filter((l) => l.counted && (l.status === "open" || l.status === "waiting"));
}

/**
 * What "Confirm all" records for a stage as it is shown (core/sweep.py `sweep_lines`): each Confirm
 * line still set for the person, or covered by the stage's last confirmation, with the default as
 * stated (`default_value`: a reading line's readings, else its reason, else its name). The server
 * holds these against its own and refuses, with a way to confirm them as they stand, if they moved.
 */
export function sweptLines(stage: QuestStage): SweptLineInput[] {
  const by = stage.sweep?.confirmed_by ?? null;
  return byLabel(stage)
    .Confirm.filter((l) => l.status === "set_for_you" || (by !== null && l.decision_id === by))
    .map((l) => ({ id: l.id, key: l.key, value: defaultValue(l) }));
}

function defaultValue(line: QuestLine): string {
  if (line.items.length) return line.items.map((i) => `${i.kind}:${i.column}=${i.value}`).join("; ");
  return line.reason ?? line.name;
}

/**
 * How long the held fit is expected to take, in the engine's words ("about 30 minutes"), read
 * from the lock's reason (core/fit_press.py `fit_lock`: "it is expected to take {duration}."). The
 * page never rounds `estimate_seconds` itself; null where the reason states no duration.
 */
export function heldFor(lock: FitLock | null | undefined): string | null {
  if (!lock?.reason) return null;
  const m = /expected to take ([^.]+)\./.exec(lock.reason);
  return m ? m[1]!.trim() : null;
}

/** Fit's position at the analysis flowchart's end (FOUNDATION §7; ruling 3). */
export type FitPosition =
  | { kind: "none" }
  /** Fit, with the engine's estimate when it is held; no Cancel, since nothing runs. */
  | { kind: "press"; held: boolean; estimate: string | null }
  /** The fit job runs: its progress and a Cancel where Fit was. */
  | { kind: "running"; progress: number | null; estimate: string | null }
  | { kind: "pressed" };

/** Where Fit stands: offered once the goal is chosen and Models' objectives are done, until it is
 *  pressed; then the fit's job, while it runs. Under Predict a noticing in Models does not hold
 *  Fit back (ruling R-a): its triage is in Results, before the held-out rows open, and the
 *  engine's press does not wait on noticings (server/service.py `press_fit`, which waits only on
 *  the questions an estimate rests on), so Models passed for the triage offers Fit too. */
export function fitPosition(log: QuestLog, fit: StageStatus | undefined): FitPosition {
  const lock = log.fit;
  if (!lock || lock.purpose === null) return { kind: "none" };
  const running = fit?.status === "running" || fit?.status === "queued";
  if (lock.pressed) {
    return running
      ? { kind: "running", progress: fit?.progress ?? null, estimate: heldFor(lock) }
      : { kind: "pressed" };
  }
  const models = stageOf(log, "models");
  if (!models?.reached || !(models.progress?.complete || passedForTriage(log, models))) return { kind: "none" };
  return { kind: "press", held: lock.held, estimate: lock.held ? heldFor(lock) : null };
}

/**
 * Why the tapestry shows no fit yet, in the engine's words (the lock's reason), or null when it may
 * (FOUNDATION §7, "what waits is what is shown"): under Predict a short fit may be computed before
 * Fit is pressed, and until the quest log says it was pressed none of it is shown. Under Estimate
 * the shell keeps the stage's own rule (W3 designs it).
 */
export function fitWithheld(log: QuestLog): string | null {
  const lock = log.fit;
  return lock && lock.purpose === "prediction" && !lock.pressed ? lock.reason : null;
}

/** The stage the shell opens on: Models while the pressed fit runs, since its analysis flowchart
 *  holds the fit's progress and Cancel where Fit was (FOUNDATION §7; ruling 3); else the open stage. */
export function landingStageKey(log: QuestLog, fit: FitPosition): string {
  return fit.kind === "running" && stageOf(log, "models")?.reached ? "models" : openStageKey(log);
}

/** What a waiting foot says, and where its name leads: "Still to answer: the lens question." or
 *  "Waiting for the outcome question, in Your question." The `name` links to `to`: the stage, and
 *  the line in it to bring into view (null: the stage's own open line). */
export interface Hint {
  lead: string;
  name: string;
  tail: string;
  to: { stage: string; line: string | null } | null;
}

/** The card's one primary action (FOUNDATION §1), for the stage it shows. A disabled Continue
 *  always says why, and links to what it waits for. An enabled one goes to `to`, with `line` (the
 *  next open line, finding 7) brought into view there. `note` says how many open noticings wait
 *  for the triage in Results under Predict (finding 6), and under Estimate, beside a noticing
 *  still to answer, that the stage's noticings are decided in it; before the goal is chosen it
 *  says nothing of them (`noticingsTravel`). */
export type Primary =
  | { kind: "continue"; to: string | null; line?: string; label: string; disabled: boolean; hint: Hint | null; note?: string }
  | { kind: "confirm"; label: string; heading: string }
  | { kind: "download" }
  /** Models before Fit: the card's foot holds no button; Fit is on the flowchart (§7). Its `note`
   *  says how many of Models' noticings wait for the triage in Results (ruling R-a). */
  | { kind: "fit"; note?: string }
  /** Results before the opening: the opening is the foot's action, which W1-G's column draws there
   *  with its chain of refusals (src/quest/contracts.ts `FootSlot`). */
  | { kind: "open" };

/** A line's name as running text: its backticks dropped and no closing period, so a name that is a
 *  whole sentence (the engine names a noticing by its summary) is not given a second one. */
export function runningName(name: string): string {
  return name.replaceAll("`", "").trim().replace(/\.+$/, "");
}

const nameOf = (log: QuestLog, key: string) => stageOf(log, key)?.name ?? key;

/**
 * Where the shell draws the triage of the open noticings, or null where it draws none: under
 * Predict, in Results, before the held-out rows open (core/sweep.py `gate_stage`; FOUNDATION §7).
 * Elsewhere the engine's triage stands before the plan's lock, in Models, which the shell does not
 * draw until W3 (W1-R R3 P5). The goal is the one the Fit lock serves.
 */
export function triageStage(log: QuestLog): string | null {
  return log.fit?.purpose === "prediction" ? "results" : null;
}

/**
 * Whether open noticings travel, holding no stage's Continue but the triage's own (finding 6):
 * under Predict, to the triage in Results; and before the goal is chosen (W1-R M1), since under
 * Predict the goal is chosen in Your question, after Your data's noticings, and where they are
 * decided waits on it. Under Estimate each is decided in its own stage, as before finding 6, and
 * so where a log is served with no Fit lock to name the goal.
 */
export function noticingsTravel(log: QuestLog): boolean {
  const goal = log.fit?.purpose;
  return goal === "prediction" || goal === null;
}

/** A stage's open noticings: its Decide lines from a finding, still to answer. */
const openNoticings = (stage: QuestStage) =>
  byLabel(stage).Decide.filter((l) => l.source === "finding" && (l.status === "open" || l.status === "waiting"));

/** "1 noticing waits …", "2 noticings are …": a count of noticings and its verb. */
const noticings = (n: number, one: string, many: string) => `${n} noticing${n === 1 ? ` ${one}` : `s ${many}`}`;

/** The foot's note on the open noticings travelling from `stage` to the triage the shell draws:
 *  none where none do, or where the triage is not drawn yet (before the goal is chosen). */
function travelNote(log: QuestLog, stage: QuestStage): { note?: string } {
  const gate = triageStage(log);
  if (gate === null || stage.key === gate) return {};
  const n = openNoticings(stage).length;
  return n ? { note: `${noticings(n, "waits", "wait")} for the triage in ${nameOf(log, gate)}.` } : {};
}

/**
 * The next open line a card can answer, outside the stage `from` (finding 7): in the seven stages'
 * order, each reached stage's first counted Decide that is open, a card can answer and is neither
 * a noticing where noticings travel (`noticingsTravel`) nor the draw while it waits for its sweep,
 * else that stage's Confirm sweep still to confirm (line null). Null when none is left. Under
 * Estimate a noticing is decided in its own stage, so it can be the next line.
 */
export function nextOpen(log: QuestLog, from: string, canAnswer: CanAnswer = ANY): { stage: QuestStage; line: QuestLine | null } | null {
  const triaged = noticingsTravel(log);
  for (const s of log.stages) {
    if (s.key === from || !s.reached) continue;
    const held = drawWaitsForSweep(s);
    const line = byLabel(s).Decide.find(
      (l) => l.counted && l.status === "open" && !(triaged && l.source === "finding") && canAnswer(l) && !(held && l.key === DRAW),
    );
    if (line) return { stage: s, line };
    if (s.sweep && !s.sweep.answered && s.sweep.lines > 0) return { stage: s, line: null };
  }
  return null;
}

export function primaryFor(log: QuestLog, stage: QuestStage, fit: FitPosition, canAnswer: CanAnswer = ANY): Primary {
  if (stage.key === "models" && fit.kind === "press") return { kind: "fit", ...travelNote(log, stage) };
  if (stage.key === "writeup" && stage.reached) return { kind: "download" };
  // A stage not reached yet is read, not worked on: its foot leads to the stage the analysis is at.
  if (!stage.reached) {
    const at = openStageKey(log);
    if (at === stage.key) return { kind: "continue", to: null, label: "Continue", disabled: true, hint: null };
    return {
      kind: "continue",
      to: at,
      label: `Go to ${nameOf(log, at)}`,
      disabled: false,
      hint: { lead: `${stage.name} is not reached yet; the analysis is at `, name: nameOf(log, at), tail: ".", to: { stage: at, line: null } },
    };
  }
  if (stage.key === "results" && byLabel(stage).Decide.some((l) => l.key === "open_seal" && l.status === "open"))
    return { kind: "open" };

  const held = drawWaitsForSweep(stage);
  const rest = unanswered(stage).filter((l) => !(held && l.key === DRAW));
  // Under Predict open noticings travel to the triage in Results (FOUNDATION §7), and before the
  // goal is chosen they hold nothing either (W1-R M1): they hold no stage but the triage's own.
  // Under Estimate the shell draws no triage, so each holds its stage.
  const travels = noticingsTravel(log);
  const travel = !travels || stage.key === triageStage(log) ? [] : rest.filter((l) => l.source === "finding");
  const left = rest.filter((l) => !travel.includes(l));
  const next = nextStageKey(log, stage.key);
  const open = left.find((l) => l.status === "open");
  if (open) {
    const here = travels ? 0 : openNoticings(stage).length;
    return {
      kind: "continue",
      to: null,
      label: "Continue",
      disabled: true,
      hint: { lead: "Still to answer: ", name: runningName(open.name), tail: ".", to: { stage: stage.key, line: open.key } },
      ...(here ? { note: `${noticings(here, "is", "are")} decided in this stage.` } : {}),
    };
  }
  if (left.length) {
    // Every line still to answer waits. Where the first waits on an answer in another stage, the
    // foot goes there; a wait on this stage, or on a result still computing, holds it.
    const on = left[0]!.waiting_for[0];
    if (on && on.stage !== stage.key && stageOf(log, on.stage)) {
      return {
        kind: "continue",
        to: on.stage,
        label: on.stage === next ? "Continue" : `Go to ${nameOf(log, on.stage)}`,
        disabled: false,
        hint: { lead: "Waiting for ", name: runningName(on.name), tail: `, in ${nameOf(log, on.stage)}.`, to: { stage: on.stage, line: on.key } },
      };
    }
    return {
      kind: "continue",
      to: null,
      label: "Continue",
      disabled: true,
      hint: on
        ? { lead: "Waiting for ", name: runningName(on.name), tail: ".", to: { stage: on.stage, line: on.key } }
        : { lead: "Waiting for a result still computing.", name: "", tail: "", to: null },
    };
  }
  const sweep = stage.sweep;
  if (sweep && !sweep.answered && sweep.lines > 0) return { kind: "confirm", label: sweep.action, heading: sweep.heading };
  const note = travelNote(log, stage);
  // Continue goes on to the next open line a card can answer, else to the next stage (finding 7),
  // naming where it goes when that is not the next stage.
  const to = nextOpen(log, stage.key, canAnswer);
  if (!to) return { kind: "continue", to: next, label: "Continue", disabled: next === null, hint: null, ...note };
  const where = { stage: to.stage.key, line: to.line?.key ?? null };
  const what = to.line ? runningName(to.line.name) : `the choice${to.stage.sweep!.lines === 1 ? "" : "s"} set for you`;
  return {
    kind: "continue",
    to: to.stage.key,
    ...(to.line ? { line: to.line.key } : {}),
    label: "Continue",
    disabled: false,
    hint: to.stage.key === next ? null : { lead: "Next: ", name: what, tail: `, in ${to.stage.name}.`, to: where },
    ...note,
  };
}
