/**
 * The coach's placement (M2_CONTRACT §6): a note never leaves its view, its leader starts under the
 * note it belongs to, and two notes keep their leaders clear of each other when they can.
 */
import { bandHeight, LEAD_IN, MAX_NOTES, NOTE_H, NOTES_MIN_W, notesFor, placeNotes, type Span } from "./place";

/** A small seeded generator, so a failure names the case that broke. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

describe("coach notes are placed within their view", () => {
  it("keeps every note inside the view and each leader under its own note, for any anchor", () => {
    const r = rng(42);
    for (let trial = 0; trial < 2000; trial++) {
      const box = 120 + Math.round(r() * 900);
      const n = 1 + Math.floor(r() * MAX_NOTES);
      const widths = Array.from({ length: n }, () => 40 + Math.round(r() * 520));
      const spans: (Span | null)[] = Array.from({ length: n }, () => {
        if (r() < 0.15) return null;
        // Anchors may lie partly or wholly outside the drawn axis (a range past the clip).
        const a = -200 + r() * (box + 400);
        const b = a + r() * 300;
        return { x0: a, x1: b, y: 40 + r() * 200, mark: "bracket" };
      });
      const placed = placeNotes(widths, spans, box);
      expect(placed).toHaveLength(n);
      for (const [i, p] of placed.entries()) {
        const where = `trial ${trial}, note ${i}`;
        expect(p.left, where).toBeGreaterThanOrEqual(0);
        expect(p.left + p.width, where).toBeLessThanOrEqual(box + 1e-9);
        expect(p.width, where).toBeLessThanOrEqual(Math.min(widths[i]!, box));
        expect(p.top, where).toBe(p.line * NOTE_H);
        if (spans[i] === null) {
          expect(p.leaderX, where).toBeNull();
          continue;
        }
        expect(p.leaderX, where).not.toBeNull();
        expect(p.leaderX!, where).toBeGreaterThanOrEqual(0);
        expect(p.leaderX!, where).toBeLessThanOrEqual(box);
        // The leader drops from under its own note (a note at least two lead-ins wide).
        if (p.width >= 2 * LEAD_IN) {
          expect(p.leaderX!, where).toBeGreaterThanOrEqual(p.left);
          expect(p.leaderX!, where).toBeLessThanOrEqual(p.left + p.width);
        }
      }
      expect(new Set(placed.map((p) => p.line)).size).toBe(n); // one note per line
    }
  });

  it("drops a leader from the middle of the visible part of a range", () => {
    const [p] = placeNotes([160], [{ x0: -50, x1: 100, y: 80, mark: "bracket" }], 400);
    expect(p!.leaderX).toBe(50);
    expect(p!.left).toBe(36);
  });

  it("swaps the notes' lines when the first note's leader would cut through the second", () => {
    // Note 0 points at x=300 from line 0; note 1 sits on line 1 right over x=300.
    const spans: Span[] = [
      { x0: 300, x1: 300, y: 120, mark: "tick" },
      { x0: 250, x1: 250, y: 120, mark: "tick" },
    ];
    const placed = placeNotes([120, 200], spans, 600);
    expect(placed.map((p) => p.line)).toEqual([1, 0]);
  });

  it("draws at most two notes, on a primary card and on a thumbnail wide enough to hold them", () => {
    const three = ["a", "b", "c"];
    expect(notesFor(three, false, 200)).toEqual(["a", "b"]);
    expect(notesFor(three, true, NOTES_MIN_W - 1)).toEqual([]);
    expect(notesFor(three, true, NOTES_MIN_W)).toEqual(["a", "b"]);
    expect(notesFor(null, false, 900)).toEqual([]);
  });

  it("reserves a band of one line per note, at most two", () => {
    expect(bandHeight(0)).toBe(0);
    expect(bandHeight(1)).toBeGreaterThan(NOTE_H);
    expect(bandHeight(5)).toBe(bandHeight(2));
  });
});
