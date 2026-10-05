/**
 * The document model's real logic, on the captured server answers: the mastery unlock (a block
 * confirm appears only after three single confirmations, and lists exactly what the server's block
 * settles), the reading lines (a family is one line; a shared-evidence line keeps every column its
 * own confirmation), the tiers (every recorded paragraph is a server sentence, verbatim), and the
 * walk (each moment's act leads to the next captured moment, from the first draft to the lock).
 */
import { artifactOf, moment, PATH } from "./fixture";
import { buildDoc, openAsk, readingLines, singleConfirmations, unlockedBlock, MASTERY } from "./model";
import { ACTS, captured, nextSingle } from "./walk";

describe("the methods document", () => {
  it("offers the block confirm only once three readings were confirmed one at a time", () => {
    const before = moment("readings").view;
    const after = moment("single-cycle_begin_year").view;
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
    const m = moment("readings");
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
    for (const id of ["draft", "model_sequence", "locked", "prediction"] as const) {
      const m = moment(id);
      const doc = buildDoc(m);
      expect(doc.guideline).toBe(id === "prediction" ? "TRIPOD+AI" : "STROBE-nut");
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

describe("the walk", () => {
  it("leads from the first draft through every captured moment to the lock, one slot's press each", () => {
    const seen: string[] = [];
    let id = PATH[0]!;
    while (ACTS[id]) {
      seen.push(id);
      const act = ACTS[id]!;
      // The press is offered: the act's slot is asked at its moment (a single reading, the block,
      // a lone reading, or the question itself).
      const doc = buildDoc(moment(id));
      expect(doc.objectives, id).toContain(act.slot);
      // Each press adds the server's records and nothing else: the next moment extends this one.
      const before = moment(id).view.decisions.map((d) => d.id);
      const after = moment(act.next).view.decisions.map((d) => d.id);
      expect(after.slice(0, before.length), id).toEqual(before);
      expect(after.length, id).toBeGreaterThan(before.length);
      id = act.next;
    }
    expect([...seen, id]).toEqual(PATH);
    expect(id).toBe("locked");
  });

  it("confirms the scenario's three readings singly, in the card's order", () => {
    const cols = PATH.map((id) => nextSingle(id)).filter(Boolean);
    expect(cols).toEqual(["bp_di", "bp_sys", "cycle_begin_year"]);
  });

  it("reads the scenario's answers from the server's own records", () => {
    const c = captured();
    expect(c.rows).toBe("keep_every_row");
    expect(c.beside.sort()).toEqual(["nhs_hpfs_by_sex", "willett_2013_by_sex"]);
    expect(c.estimand).toEqual({ exposure: "sugar", effect: "total", contrast: "substitution" });
    expect(c.model1).toEqual(["age", "gender", "kcal"]);
    expect(c.models).toEqual(["linear"]);
    expect(Object.keys(c.adjustment)).toHaveLength(7);
  });
});
