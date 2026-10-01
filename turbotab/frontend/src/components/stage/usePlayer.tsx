/**
 * The player as a small store: the state machine (player.ts) driven by animation frames.
 *
 * The views read the position every frame without re-rendering React (they draw imperatively);
 * the controls read only what changes at a step boundary (side, heading, paused), so the stage's
 * React tree re-renders a handful of times per flip, not sixty times a second.
 */
import { createContext, useContext, useLayoutEffect, useMemo, useRef, useSyncExternalStore } from "react";
import {
  eased,
  heading,
  initial,
  moving,
  reached,
  reduce,
  sideOf,
  type PlayerEvent,
  type PlayerState,
  type Side,
} from "./player";

export interface PlayerStore {
  get(): PlayerState;
  subscribe(fn: () => void): () => void;
  dispatch(e: PlayerEvent): void;
  /** A new scene: start from this state. */
  reset(s: PlayerState): void;
  setReduced(v: boolean): void;
  /** Review captures only: stop the clock and step it by hand. */
  setManual(v: boolean): void;
  advance(ms: number): void;
}

/** `?slow=10` plays every flip ten times slower so a reviewer can watch it (changes nothing else). */
function readSlow(): number {
  if (typeof window === "undefined") return 1;
  const v = Number(new URLSearchParams(window.location.search).get("slow"));
  return Number.isFinite(v) && v >= 1 && v <= 50 ? v : 1;
}

export function createPlayerStore(start: PlayerState = initial()): PlayerStore {
  let state = start;
  let reduced = false;
  let manual = false;
  let frame = 0;
  let last = 0;
  const slow = readSlow();
  const listeners = new Set<() => void>();
  const notify = () => listeners.forEach((fn) => fn());

  const loop = (now: number) => {
    frame = 0;
    const dt = last ? Math.min(64, now - last) : 16;
    last = now;
    state = reduce(state, { type: "tick", ms: dt / slow }, reduced);
    notify();
    if (moving(state) && !manual) frame = requestAnimationFrame(loop);
    else last = 0;
  };
  const kick = () => {
    if (manual || frame || !moving(state) || typeof requestAnimationFrame === "undefined") return;
    last = 0;
    frame = requestAnimationFrame(loop);
  };

  return {
    get: () => state,
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    dispatch(e) {
      const next = reduce(state, e, reduced);
      if (next === state) return;
      state = next;
      notify();
      kick();
    },
    reset(next) {
      state = next;
      notify();
      kick();
    },
    setReduced(v) {
      reduced = v;
      if (v && moving(state)) {
        state = { ...state, pos: state.target };
        notify();
      }
    },
    setManual(v) {
      manual = v;
      if (v && frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
      if (!v) kick();
    },
    advance(ms) {
      state = reduce(state, { type: "tick", ms }, reduced);
      notify();
    },
  };
}

export const PlayerContext = createContext<PlayerStore | null>(null);

export function usePlayerStore(): PlayerStore {
  const store = useContext(PlayerContext);
  if (!store) throw new Error("usePlayerStore: no <PlayerContext> above this view");
  return store;
}

export interface PlayerUi {
  side: Side;
  heading: number;
  reached: number;
  paused: boolean;
  moving: boolean;
  last: number;
  /** Heading forward (true) or back. */
  forward: boolean;
  /** The real state the drawing is nearest to: what text (labels, numbers) names. */
  nearest: number;
}

function uiOf(s: PlayerState): PlayerUi {
  return {
    side: sideOf(s),
    heading: heading(s),
    reached: reached(s),
    paused: s.paused,
    moving: moving(s),
    last: s.last,
    forward: s.target >= s.pos,
    nearest: Math.round(s.pos),
  };
}

const sameUi = (a: PlayerUi, b: PlayerUi) =>
  a.side === b.side &&
  a.heading === b.heading &&
  a.reached === b.reached &&
  a.paused === b.paused &&
  a.moving === b.moving &&
  a.last === b.last &&
  a.forward === b.forward &&
  a.nearest === b.nearest;

/** The player's step-level state, for controls and labels. Re-renders only at step boundaries. */
export function usePlayerUi(store: PlayerStore): PlayerUi {
  const cache = useRef<PlayerUi | null>(null);
  const read = () => {
    const next = uiOf(store.get());
    if (cache.current && sameUi(cache.current, next)) return cache.current;
    cache.current = next;
    return next;
  };
  return useSyncExternalStore(store.subscribe, read, read);
}

/** Call `draw(easedPosition)` now and on every frame the player moves. */
export function usePlayerFrame(store: PlayerStore, draw: (pos: number, s: PlayerState) => void) {
  const drawRef = useRef(draw);
  useLayoutEffect(() => {
    drawRef.current = draw;
    const s = store.get();
    draw(eased(s.pos), s);
  }, [draw, store]);
  useLayoutEffect(
    () =>
      store.subscribe(() => {
        const s = store.get();
        drawRef.current(eased(s.pos), s);
      }),
    [store],
  );
}

/** A fresh store per mount. */
export function usePlayerStoreInstance(): PlayerStore {
  return useMemo(() => createPlayerStore(), []);
}
