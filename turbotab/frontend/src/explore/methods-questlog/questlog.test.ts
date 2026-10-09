/**
 * The prototype's claims that are logic, not pixels: every recorded sentence lands in a section of
 * its guideline (nothing the record holds goes missing from the methods section); the walk reaches
 * the lock through the scenario's captured moments with nothing left open; and the block confirm
 * the mastery rule unlocks settles exactly the readings left after the single confirmations.
 */
import { describe, expect, it } from "vitest";
import { MOMENTS, ORDER, PREDICTION } from "./data";
import { ADJUSTMENT_ANSWERS, advance, askOf, idOf, isLocked, linesOf, MASTERY, nextSingle, objectiveItem, SINGLES, singlesDone, START, type Walk } from "./journey";
import { blockOf, readingSlots } from "./readings";
import { KINDS, sectionsOf, STROBE, TRIPOD } from "./sections";

const placed = (defs: typeof STROBE) => new Set(defs.flatMap((d) => d.items).flatMap((k) => KINDS[k] ?? []));

describe("the methods section as objectives", () => {
  it("places every in-force sentence of both drives in a section of its guideline", () => {
    const strobe = placed(STROBE);
    const tripod = placed(TRIPOD);
    for (const l of MOMENTS.locked!.methods.lines.filter((x) => x.in_force)) expect(strobe.has(l.kind), l.kind).toBe(true);
    for (const l of PREDICTION.methods.lines.filter((x) => x.in_force)) expect(tripod.has(l.kind), l.kind).toBe(true);
  });

  it("holds nothing open once the plan is locked, and states the lock", () => {
    const m = MOMENTS.locked!;
    const sections = sectionsOf(m, m.methods.lines, { readings: { open: 0, waiting: false }, analyzed: 21849 }, "STROBE-nut");
    expect(sections.every((s) => s.open === 0 && s.waiting === 0)).toBe(true);
    expect(sections.flatMap((s) => s.items).find((i) => i.key === "lock")?.tier).toBe("stated");
  });
});

describe("the walk", () => {
  it("has an objective at every captured moment until the lock, and each record moves on", () => {
    let w: Walk = START;
    const seen: string[] = [];
    while (!isLocked(w)) {
      expect(objectiveItem(w), idOf(w)).not.toBeNull();
      seen.push(idOf(w));
      if (idOf(w) === "adjustment") w = { ...w, adjusted: ADJUSTMENT_ANSWERS.map((a) => a.index) };
      if (idOf(w) === "models" && !w.codes) w = { ...w, codes: true };
      w = advance(w);
    }
    expect(seen).toEqual(ORDER.slice(0, -1));
  });

  it("unlocks a block that lists exactly the role readings left after the three singles", () => {
    const w: Walk = { ...START, at: ORDER.indexOf("single_cycle_begin_year") };
    expect(singlesDone(w).map((l) => l.sentence.match(/^`([^`]+)`/)?.[1])).toEqual(SINGLES);
    expect(singlesDone(w).length).toBe(MASTERY);
    expect(nextSingle(w)).toBeNull();
    const ask = askOf(w);
    const listed = blockOf(ask)!.items.map((i) => `${i.reading}:${i.column}`);
    const asked = readingSlots(ask).flatMap((r) => r.columns.map((c) => `${r.kind}:${c}`));
    expect(new Set(listed)).toEqual(new Set(asked));
    expect(listed.some((k) => SINGLES.some((c) => k === `role:${c}`))).toBe(false);
  });

  it("shows each adjustment answer's own sentence as it is recorded", () => {
    const at = ORDER.indexOf("adjustment");
    const before = linesOf({ ...START, at }).length;
    const w: Walk = { ...START, at, adjusted: [0, 2] };
    expect(linesOf(w).length).toBe(before + 2);
    expect(linesOf(w).filter((l) => l.kind === "set_adjustment").length).toBe(2);
  });
});
