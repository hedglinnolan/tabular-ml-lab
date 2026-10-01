/**
 * The transform player's state machine (BLUEPRINT §11.1, M1_CONTRACT §11) — pure, so it is tested
 * without a browser.
 *
 * A preview is a sequence of real, labeled states: 0 is *your data now*, `last` is *with this
 * choice*, and 1 … last − 1 are the method's own storyboard steps. `pos` is where the views are
 * drawn, `target` where they are heading. Flipping plays the storyboard forward (or back) at one
 * steady pace, so a full flip takes at most TOTAL_MS; any new input starts from `pos`, which is
 * what is on screen, so nothing ever jumps.
 *
 * Switching options (arrow keys in the Record) keeps the flip's side and moves straight to the
 * new option's state on that side: the views morph between results; the storyboard does not replay.
 */

export const TOTAL_MS = 900;
export const SEGMENT_MAX_MS = 300;

export interface PlayerState {
  /** Where the views are drawn, 0 … last. Fractional only while moving. */
  pos: number;
  /** Where the views are heading, always a whole step. */
  target: number;
  /** The index of "with this choice" (the storyboard's length + 1). */
  last: number;
  /** Stopped on a storyboard step by its dot. */
  paused: boolean;
}

export type Side = "now" | "with";

export type PlayerEvent =
  /** The flip: now ⇄ with this choice, from wherever the views are. */
  | { type: "flip" }
  | { type: "show"; side: Side }
  /** A step dot: go to that state and stay there. */
  | { type: "seek"; step: number }
  /** A new option (or evidence) with `last` states: hold the side, do not replay. */
  | { type: "options"; last: number }
  /** Time passes. */
  | { type: "tick"; ms: number };

export function initial(last = 1, side: Side = "now"): PlayerState {
  const l = Math.max(1, Math.round(last));
  const at = side === "with" ? l : 0;
  return { pos: at, target: at, last: l, paused: false };
}

/** Milliseconds per storyboard step: brisk, and never more than TOTAL_MS for a full flip. */
export function segmentMs(last: number): number {
  return Math.min(SEGMENT_MAX_MS, TOTAL_MS / Math.max(1, last));
}

/** Which side of the flip the player is on (a paused step counts as "with this choice"). */
export function sideOf(s: PlayerState): Side {
  return s.target > 0 ? "with" : "now";
}

export function moving(s: PlayerState): boolean {
  return s.pos !== s.target;
}

/**
 * The real state the views are showing or heading to. Labels, numbers and saves read this, never
 * `pos`: a half-way drawing is motion, not data.
 */
export function heading(s: PlayerState): number {
  if (s.pos === s.target) return s.target;
  return s.target > s.pos ? Math.floor(s.pos) + 1 : Math.ceil(s.pos) - 1;
}

/** The last real state the views have fully reached. */
export function reached(s: PlayerState): number {
  if (s.pos === s.target) return s.target;
  return s.target > s.pos ? Math.floor(s.pos) : Math.ceil(s.pos);
}

/** Milliseconds until the player arrives where it is heading. */
export function remainingMs(s: PlayerState): number {
  return Math.abs(s.target - s.pos) * segmentMs(s.last);
}

const clampStep = (n: number, last: number) => Math.max(0, Math.min(last, Math.round(n)));

export function reduce(s: PlayerState, e: PlayerEvent, reduced = false): PlayerState {
  switch (e.type) {
    case "flip": {
      const target = sideOf(s) === "with" ? 0 : s.last;
      return { ...s, target, paused: false, pos: reduced ? target : s.pos };
    }
    case "show": {
      const target = e.side === "with" ? s.last : 0;
      if (target === s.target && !s.paused) return s;
      return { ...s, target, paused: false, pos: reduced ? target : s.pos };
    }
    case "seek": {
      const target = clampStep(e.step, s.last);
      return { ...s, target, paused: target !== 0 && target !== s.last, pos: reduced ? target : s.pos };
    }
    case "options": {
      const last = Math.max(1, Math.round(e.last));
      // Hold the side: now stays now; with this choice lands on the new option's result; a
      // paused step stays on that step when the new storyboard has it.
      let target: number;
      if (sideOf(s) === "now") target = 0;
      else if (s.paused && s.target < last) target = s.target;
      else target = last;
      return { pos: target, target, last, paused: s.paused && target !== last && target !== 0 };
    }
    case "tick": {
      if (s.pos === s.target || e.ms <= 0) return s;
      const step = e.ms / segmentMs(s.last);
      const pos =
        s.target > s.pos ? Math.min(s.target, s.pos + step) : Math.max(s.target, s.pos - step);
      return { ...s, pos };
    }
  }
}

/** Ease within each storyboard step, so every real state is a visible beat. */
export function eased(pos: number): number {
  const i = Math.floor(pos);
  const f = pos - i;
  if (f === 0) return pos;
  const e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
  return i + e;
}
