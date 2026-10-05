/**
 * The document model's real logic, on the captured server answers: the mastery unlock (a block
 * confirm appears only after three single confirmations, and lists exactly what the server's block
 * settles), the reading lines (a family is one line; a shared-evidence line keeps every column its
 * own confirmation), and the tiers (every recorded paragraph is a server sentence, verbatim).
 */
import { artifactOf, moment } from "./fixture";
import { buildDoc, openAsk, readingLines, singleConfirmations, unlockedBlock, MASTERY } from "./model";

describe("the methods document", () => {
  it("offers the block confirm only once three readings were confirmed one at a time", () => {
    const before = moment("m2").view;
    const after = moment("m9").view;
    expect(singleConfirmations(before).length).toBeLessThan(MASTERY);
    expect(unlockedBlock(before)).toBeNull();
    expect(singleConfirmations(after).length).toBe(MASTERY);
    const block = unlockedBlock(after);
    expect(block).not.toBeNull();
    // The block settles exactly the readings the card asks, each with the guess it shows.
    const items = (block!.decision as { items: { column: string; value: string }[] }).items;
    const asked = openAsk(after)!.groups.flatMap((g) => g.columns.map((c) => `${c}:${g.guess}`));
    expect(items.map((i) => `${i.column}:${i.value}`).sort()).toEqual(asked.sort());
  });

  it("keeps a family on one line and every other column its own confirmation", () => {
    const m = moment("m2");
    const ask = openAsk(m.view)!;
    const roles = artifactOf<{ columns: { column: string; reason: string }[] }>(m, "roles")?.columns ?? [];
    const lines = readingLines(ask, roles);
    const listed = lines.flatMap((l) => l.columns).sort();
    expect(listed).toEqual(ask.groups.flatMap((g) => g.columns).sort());
    const families = lines.filter((l) => l.family);
    expect(families).toHaveLength(ask.groups.filter((g) => g.columns.length > 1).length);
    for (const l of lines.filter((x) => !x.family))
      for (const c of l.columns) expect(ask.groups.some((g) => g.columns.length === 1 && g.columns[0] === c)).toBe(true);
  });

  it("writes only the server's sentences, in the guideline the purpose names", () => {
    for (const id of ["m1", "m5", "m6", "m7"] as const) {
      const m = moment(id);
      const doc = buildDoc(m);
      expect(doc.guideline).toBe(id === "m7" ? "TRIPOD+AI" : "STROBE-nut");
      const sentences = new Set(m.view.decisions.map((d) => d.sentence));
      const reasons = new Set(m.view.interview.map((s) => s.reason ?? ""));
      for (const p of doc.sections.flatMap((s) => s.paras)) {
        if (p.tier !== "recorded" || !p.text || p.question === "adjustment" || p.id.startsWith("stage-")) continue;
        const whole = p.fold ? `${p.text} ${p.fold.text}` : p.text;
        expect([...sentences].some((t) => t === whole || t === p.text)).toBe(true);
      }
      for (const p of doc.sections.flatMap((s) => s.paras).filter((x) => x.tier === "stated"))
        expect([...reasons].some((r) => r.toLowerCase() === p.text!.toLowerCase())).toBe(true);
    }
  });
});
