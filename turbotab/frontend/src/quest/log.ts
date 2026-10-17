/**
 * The quest log as the shell reads it (pure): which stage is open, each stage's segment of the
 * progress bar, its lines under their labels, the open line, what "Confirm all" records and the
 * one primary action at the card's foot. Everything here reads GET /quest's fields; nothing counts
 * or scores on its own (FOUNDATION §3; plans/W1_PART2_PLAN.md §2 "W1-E").
 */
import type { FitLock, QuestLine, QuestLog, QuestStage, StageStatus, SweptLineInput } from "../api/schema";

export type Label = QuestLine["label"];
export const LABELS: readonly Label[] = ["Decide", "Confirm", "For the record"];

/** The stage the card opens on: the first reached stage not yet complete, else the last reached
 *  (Models, complete, before Fit), else the first. */
export function openStageKey(log: QuestLog): string {
  const reached = log.stages.filter((s) => s.reached);
  const working = reached.find((s) => !s.progress?.complete);
  return (working ?? reached[reached.length - 1] ?? log.stages[0]!).key;
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
 *  "0 of N"); else its objectives answered of required; dropped back while a reopen stands. */
export interface Segment {
  key: string;
  name: string;
  reached: boolean;
  answered: number | null;
  required: number | null;
  complete: boolean;
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
    reopened: s.reopened.map((r) => r.sentence),
  }));
}

/** The fraction of a segment that is filled: a reached stage that asks nothing is full. */
export function filled(seg: Segment): number {
  if (!seg.reached || seg.answered === null || seg.required === null) return 0;
  if (seg.complete || seg.required === 0) return 1;
  return Math.min(1, seg.answered / seg.required);
}

/** A stage's lines under their labels, each label's lines in the engine's order. */
export function byLabel(stage: QuestStage): Record<Label, QuestLine[]> {
  const out: Record<Label, QuestLine[]> = { Decide: [], Confirm: [], "For the record": [] };
  for (const line of [...stage.lines].sort((a, b) => a.order - b.order)) out[line.label].push(line);
  return out;
}

/** The line the card shows expanded: the stage's first open Decide, counted ones first. */
export function openLine(stage: QuestStage): QuestLine | null {
  const decide = byLabel(stage).Decide.filter((l) => l.status === "open");
  return decide.find((l) => l.counted) ?? decide[0] ?? null;
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
 *  pressed; then the fit's job, while it runs. */
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
  if (!models?.reached || !models.progress?.complete) return { kind: "none" };
  return { kind: "press", held: lock.held, estimate: lock.held ? heldFor(lock) : null };
}

/** The card's one primary action (FOUNDATION §1), for the stage it shows. */
export type Primary =
  | { kind: "continue"; to: string | null; disabled: boolean; hint: string | null }
  | { kind: "confirm"; label: string; heading: string }
  | { kind: "download" }
  /** Models before Fit: the card's foot holds no button; Fit is on the flowchart (§7). */
  | { kind: "fit" };

export function primaryFor(log: QuestLog, stage: QuestStage, fit: FitPosition): Primary {
  const left = unanswered(stage);
  if (stage.key === "models" && fit.kind === "press") return { kind: "fit" };
  if (stage.key === "writeup" && stage.reached) return { kind: "download" };
  if (left.length) {
    const first = left[0]!;
    return {
      kind: "continue",
      to: null,
      disabled: true,
      hint:
        first.status === "waiting" && first.waiting_for.length
          ? `Waiting for ${first.waiting_for.map((w) => w.name).join(", ")}.`
          : `Still to answer: ${first.name}.`,
    };
  }
  const sweep = stage.sweep;
  if (stage.reached && sweep && !sweep.answered && sweep.lines > 0)
    return { kind: "confirm", label: sweep.action, heading: sweep.heading };
  const to = nextStageKey(log, stage.key);
  return { kind: "continue", to, disabled: to === null, hint: null };
}
