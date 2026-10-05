/**
 * The prototype's two claims that are logic, not pixels: every recorded sentence lands in a
 * section of its guideline (nothing the record holds goes missing from the methods section), and
 * the unlocked block confirm settles exactly the readings left after the single confirmations.
 */
import { describe, expect, it } from "vitest";
import { FX, INF } from "./data";
import { readingSlots, unlockedBlock } from "./readings";
import { KINDS, sectionsOf, STROBE, TRIPOD } from "./sections";

const placed = (defs: typeof STROBE) => new Set(defs.flatMap((d) => d.items).flatMap((k) => KINDS[k] ?? []));

describe("the methods section as objectives", () => {
  it("places every in-force sentence of both drives in a section of its guideline", () => {
    const strobe = placed(STROBE);
    const tripod = placed(TRIPOD);
    for (const l of INF.moments.m6.methods.lines.filter((x) => x.in_force)) expect(strobe.has(l.kind), l.kind).toBe(true);
    for (const l of FX.prediction.moment.methods.lines.filter((x) => x.in_force)) expect(tripod.has(l.kind), l.kind).toBe(true);
  });

  it("holds nothing open once the plan is locked, and states the lock", () => {
    const sections = sectionsOf(INF.moments.m6, { readings: { open: 0, waiting: false }, analyzed: 21849 }, "STROBE-nut");
    expect(sections.every((s) => s.open === 0 && s.waiting === 0)).toBe(true);
    const lock = sections.flatMap((s) => s.items).find((i) => i.key === "lock");
    expect(lock?.tier).toBe("stated");
  });

  it("unlocks a block that lists exactly the readings left after three single confirmations", () => {
    const singles = new Set(INF.singles.map((x) => `role:${String(x.decision.column)}`));
    const listed = unlockedBlock().items.map((i) => `${i.reading}:${i.column}`);
    expect(listed.some((k) => singles.has(k))).toBe(false);
    const asked = readingSlots()
      .filter((r) => r.kind !== "unit")
      .flatMap((r) => r.columns.map((c) => `${r.kind}:${c}`))
      .filter((k) => !singles.has(k));
    expect(new Set(listed)).toEqual(new Set(asked));
  });
});
