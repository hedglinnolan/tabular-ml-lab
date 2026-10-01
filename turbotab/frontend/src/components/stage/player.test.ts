import {
  heading,
  initial,
  reached,
  reduce,
  remainingMs,
  segmentMs,
  sideOf,
  TOTAL_MS,
  type PlayerEvent,
  type PlayerState,
} from "./player";

const run = (s: PlayerState, ...events: PlayerEvent[]) => events.reduce((a, e) => reduce(a, e), s);
const tick = (ms: number): PlayerEvent => ({ type: "tick", ms });

/** Advance in 16 ms frames until the player arrives; returns the elapsed time. */
function playOut(s: PlayerState): [PlayerState, number] {
  let t = 0;
  while (s.pos !== s.target && t < 10_000) {
    s = reduce(s, tick(16));
    t += 16;
  }
  return [s, t];
}

describe("the transform player", () => {
  it("plays the whole storyboard forward on a flip, within 900 ms", () => {
    for (const last of [1, 2, 3, 6, 12]) {
      const [s, t] = playOut(reduce(initial(last), { type: "flip" }));
      expect(s.pos).toBe(last);
      expect(sideOf(s)).toBe("with");
      expect(t).toBeLessThanOrEqual(TOTAL_MS + 16);
    }
    expect(segmentMs(3)).toBe(300);
    expect(segmentMs(1)).toBe(300);
  });

  it("passes through every real step, in order, on the way", () => {
    let s = reduce(initial(3), { type: "flip" });
    const headings: number[] = [];
    while (s.pos !== s.target) {
      headings.push(heading(s));
      s = reduce(s, tick(16));
    }
    expect([...new Set(headings)]).toEqual([1, 2, 3]);
  });

  it("plays it in reverse on the way back", () => {
    let s = playOut(reduce(initial(3), { type: "flip" }))[0];
    s = reduce(s, { type: "flip" });
    expect(sideOf(s)).toBe("now");
    const headings: number[] = [];
    while (s.pos !== s.target) {
      headings.push(heading(s));
      s = reduce(s, tick(16));
    }
    expect([...new Set(headings)]).toEqual([2, 1, 0]);
    expect(s.pos).toBe(0);
  });

  it("turns around from what is on screen when interrupted, without a jump", () => {
    let s = run(initial(3), { type: "flip" }, tick(450));
    expect(s.pos).toBeCloseTo(1.5);
    const before = s.pos;
    s = reduce(s, { type: "flip" });
    expect(s.pos).toBe(before); // no jump
    expect(s.target).toBe(0);
    s = reduce(s, tick(150));
    expect(s.pos).toBeCloseTo(1.0);
    // Going back takes only as long as the way it came.
    expect(remainingMs(s)).toBeCloseTo(300);
  });

  it("pauses on a step when its dot is pressed, and a flip from there goes back to now", () => {
    let s = run(initial(3), { type: "seek", step: 2 });
    s = playOut(s)[0];
    expect(s.pos).toBe(2);
    expect(s.paused).toBe(true);
    expect(sideOf(s)).toBe("with");
    s = playOut(reduce(s, { type: "flip" }))[0];
    expect(s.pos).toBe(0);
    expect(s.paused).toBe(false);
  });

  it("holds the side across options and lands on the new result without replaying", () => {
    let s = playOut(reduce(initial(3), { type: "flip" }))[0];
    s = reduce(s, { type: "options", last: 1 }); // residual (two steps) → density (none)
    expect(s).toEqual({ pos: 1, target: 1, last: 1, paused: false });
    s = reduce(s, { type: "options", last: 3 }); // and back: straight to the result
    expect(s).toEqual({ pos: 3, target: 3, last: 3, paused: false });
    let now = reduce(initial(3), { type: "options", last: 1 });
    expect(now.pos).toBe(0);
    now = reduce(now, { type: "options", last: 4 });
    expect(now).toEqual({ pos: 0, target: 0, last: 4, paused: false });
  });

  it("keeps a paused step when the new storyboard has it, else shows the result", () => {
    let s = playOut(run(initial(4), { type: "seek", step: 2 }))[0];
    s = reduce(s, { type: "options", last: 4 });
    expect(s).toMatchObject({ pos: 2, paused: true });
    s = reduce(s, { type: "options", last: 1 });
    expect(s).toMatchObject({ pos: 1, target: 1, paused: false });
  });

  it("is instant under reduced motion", () => {
    const s = reduce(initial(3), { type: "flip" }, true);
    expect(s.pos).toBe(3);
    expect(reduce(s, { type: "seek", step: 1 }, true).pos).toBe(1);
  });

  it("names a real state at every moment: heading and reached are whole steps", () => {
    let s = reduce(initial(3), { type: "flip" });
    while (s.pos !== s.target) {
      expect(Number.isInteger(heading(s))).toBe(true);
      expect(Number.isInteger(reached(s))).toBe(true);
      s = reduce(s, tick(7));
    }
  });
});
