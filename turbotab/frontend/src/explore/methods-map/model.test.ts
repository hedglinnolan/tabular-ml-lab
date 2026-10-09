/**
 * The methods map's logic, on the captured fixture (the shared scenario, SCENARIO.md): what a block
 * confirmation settles, that walking the scenario's answers locks the scenario's plan and selects
 * its fit, that the record at the lock is the engine's /methods text and nothing else, that a
 * preview is the one taken on the answers' state, and that the record only ever quotes the engine.
 */
import { describe, expect, it } from "vitest";
import { fmtCI, fmtEst, table2Rows } from "../methods-shared/results";
import { INF, PRED } from "./fixture";
import {
  CODE_KEYS,
  INITIAL,
  PRED_CODE_KEYS,
  ROLE_KEYS,
  SINGLES_TO_UNLOCK,
  blockOffer,
  fitFor,
  guessTriple,
  lockKey,
  lockSentence,
  previewFor,
  readingKeys,
  record,
  reduce,
  statedReason,
  truthTriple,
  type Action,
  type Answers,
} from "./model";
import { effectsOf, matteredOf } from "./Results";

const play = (actions: Action[], from: Answers = INITIAL) => actions.reduce(reduce, from);
const keys = ROLE_KEYS;
const adjust: Action[] = INF.adjustment.groups.map((g) => ({
  type: "adjust",
  group: g.key,
  answers: Object.fromEntries(g.columns.map((c) => [c, guessTriple(g.key) ?? truthTriple(c)!])),
}));
/** The scenario's answers, as a person clicks them on the map. */
const planned = play([
  { type: "reading", key: "role:bp_di" },
  { type: "reading", key: "role:bp_sys" },
  { type: "reading", key: "role:cycle_begin_year" },
  { type: "block" },
  { type: "codes" },
  { type: "unit" },
  { type: "exclusions", key: "none" },
  { type: "sensitivity", key: "willett_2013_by_sex" },
  { type: "sensitivity", key: "nhs_hpfs_by_sex" },
  { type: "missing" },
  { type: "exposure" },
  ...adjust,
  { type: "model1", value: "guess" },
]);
const texts = (a: Answers) =>
  record(a, "inference")
    .flatMap((s) => s.lines)
    .map((l) => l.text)
    .filter((t): t is string => !!t);

describe("the readings' block confirmation", () => {
  it("unlocks only after three single confirmations and lists exactly the rest", () => {
    expect(keys.slice(0, 3)).toEqual(["role:bp_di", "role:bp_sys", "role:cycle_begin_year"]);
    expect(blockOffer(play(keys.slice(0, SINGLES_TO_UNLOCK - 1).map((key) => ({ type: "reading", key }))))).toBeNull();
    const three = play(keys.slice(0, SINGLES_TO_UNLOCK).map((key) => ({ type: "reading", key })));
    const offer = blockOffer(three)!;
    expect(offer.keys).toEqual(keys.slice(SINGLES_TO_UNLOCK));
    const after = reduce(three, { type: "block" });
    expect(keys.every((k) => after.readings[k])).toBe(true);
    expect(CODE_KEYS.some((k) => after.readings[k])).toBe(false);
    expect(keys.slice(0, SINGLES_TO_UNLOCK).every((k) => after.readings[k] === "single")).toBe(true);
  });

  it("names, in its sentence, what it settles (the engine's own sentence for that set)", () => {
    const offer = blockOffer(play([2, 5, 9].map((i) => ({ type: "reading", key: keys[i]! }))))!;
    expect(offer.keys).not.toContain(keys[2]);
    expect(INF.readings.block.sentences).toContain(offer.sentence);
    expect(offer.sentence.startsWith("Confirmed together, each as the question showed it")).toBe(true);
  });
});

describe("under prediction, its own drive's readings", () => {
  it("one block settles the role readings left and the code-or-amount ones, in the drive's sentence", () => {
    const three = play(keys.slice(0, SINGLES_TO_UNLOCK).map((key) => ({ type: "reading", key })));
    const offer = blockOffer(three, "prediction")!;
    expect(offer.keys).toEqual([...keys.slice(SINGLES_TO_UNLOCK), ...PRED_CODE_KEYS]);
    expect(PRED.readings.block.sentences).toContain(offer.sentence);
    const done = play([{ type: "block", purpose: "prediction" }, { type: "unit" }], three);
    expect(readingKeys("prediction").every((k) => done.readings[k])).toBe(true);
    const lines = record(done, "prediction").flatMap((s) => s.lines.filter((l) => l.node === "readings"));
    expect(lines.map((l) => l.text)).toEqual([
      PRED.stated.roles,
      PRED.readings.unit_sentence,
      ...keys.slice(0, SINGLES_TO_UNLOCK).map((k) => PRED.readings.single[k]),
      offer.sentence,
    ]);
  });
});

describe("the scenario, walked on the map", () => {
  const locked = reduce(planned, { type: "lock" });

  it("locks the scenario's plan (its SHA-256 is the engine's) and selects the scenario's fit", () => {
    expect(planned.energy).toBe(INF.energy.ranking.order[0]);
    expect(planned.form).toBeNull();
    expect(lockKey(planned)).toBe("nd0-3g");
    const lockLine = INF.methods_at_lock.find((l) => l.kind === "lock_plan")!.sentence;
    expect(lockSentence(locked)).toBe(lockLine);
    const pick = fitFor(locked);
    expect(pick.kind).toBe("fit");
    if (pick.kind !== "fit") return;
    const t2 = table2Rows(effectsOf(pick.fit, locked));
    expect(t2.map((r) => r.key)).toEqual(["crude", "model_1", "model_2", "model_3"]);
    const m2 = t2.find((r) => r.key === "model_2")!;
    expect(m2.n).toBe(21849);
    expect([fmtEst(m2.estimate), fmtCI(m2.lo, m2.hi)]).toEqual(["−0.0199", "−0.0327 to −0.00718"]);
    // the model sequence and the two screens declared beside the primary
    expect(matteredOf(pick.fit, locked).map((r) => r.key)).toEqual([
      "crude",
      "model_1",
      "model_2",
      "model_3",
      "screen:Willett 2013, by sex",
      "screen:NHS/HPFS, by sex",
    ]);
  });

  it("the methods text at the lock is the engine's /methods lines, and no default it never recorded", () => {
    const engine = INF.methods_at_lock.map((l) => l.sentence);
    // the skipped questions' reasons are stated beside them (the engine's, never recorded as answers)
    const skipped = [statedReason("grain", "inference"), statedReason("clusters", "inference")];
    const shown = texts(locked).filter((t) => !skipped.includes(t));
    expect(new Set(shown)).toEqual(new Set(engine));
    expect(shown).toHaveLength(engine.length);
    for (const form of Object.values(INF.form.sentences)) expect(texts(locked)).not.toContain(form);
    // every line the record marks as recorded is in the engine's record
    const recorded = record(locked, "inference")
      .flatMap((s) => s.lines)
      .filter((l) => l.tier === "recorded");
    for (const l of recorded) expect(engine).toContain(l.text);
  });
});

describe("other plans", () => {
  it("an answer whose fit was not captured says so instead of borrowing another's", () => {
    const other = reduce(planned, { type: "exclusions", key: "nhs_hpfs_by_sex" });
    expect(fitFor(other).kind).toBe("uncaptured");
    const answers = { ...planned.adjustment.unguessed! };
    answers.hdl = ["yes", "yes", "no"];
    expect(fitFor(reduce(planned, { type: "adjust", group: "unguessed", answers })).kind).toBe("uncaptured");
  });

  it("a curve on an energy-dropped residual is the engine's own failure, kept as such", () => {
    const pick = fitFor(play([{ type: "form", form: "spline" }, { type: "energy", method: "residual_energy_dropped" }], planned));
    expect(pick.kind).toBe("error");
  });

  it("a form chosen before the lock is recorded, and locks another plan", () => {
    const linear = reduce(planned, { type: "form", form: "linear" });
    expect(lockKey(linear)).toBe("nl0-3g");
    expect(texts(linear)).toContain(INF.form.sentences.linear);
    expect(lockSentence(reduce(linear, { type: "lock" }))).not.toBe(lockSentence(reduce(planned, { type: "lock" })));
  });

  it("is locked once: a change after it is kept and marked, and nothing is undone", () => {
    const locked = reduce(planned, { type: "lock" });
    const changed = reduce(locked, { type: "energy", method: "residual" });
    expect(changed.locked!.energy).toBe("standard");
    expect(changed.after).toEqual([{ node: "energy", value: "residual" }]);
    expect(reduce(changed, { type: "undo", node: "exposure" })).toBe(changed);
    expect(reduce(changed, { type: "missing" })).toBe(changed);
    const lines = record(changed, "inference").flatMap((s) => s.lines);
    expect(lines.some((l) => l.after && l.text?.startsWith("After the estimates were seen"))).toBe(true);
  });
});

describe("previews and the record", () => {
  it("a preview is the one taken on the answers' state", () => {
    const willett = reduce(planned, { type: "exclusions", key: "willett_2013_by_sex" });
    const base = previewFor("split", "0.0", planned)!.result!;
    const there = previewFor("split", "0.0", willett)!.result!;
    expect(base.views[0]!.caption).toContain("21,849");
    expect(there.views[0]!.caption).not.toBe(base.views[0]!.caption);
    // a view shared with another state resolves to that state's view, whole
    const spline = previewFor("energy", "residual", reduce(planned, { type: "form", form: "spline" }))!.result!;
    expect(spline.views.every((v) => "kind" in v)).toBe(true);
  });

  it("quotes only the engine's sentences", () => {
    const after = play([{ type: "lock" }, { type: "form", form: "spline" }, { type: "energy", method: "residual" }], planned);
    const engine = new Set<string>([
      ...INF.methods_at_lock.map((l) => l.sentence),
      ...Object.values(INF.stated),
      ...Object.values(INF.readings.single),
      ...INF.readings.block.sentences,
      ...Object.values(INF.exclusions.sentences).filter((t): t is string => !!t),
      ...Object.values(INF.sensitivity.sentences),
      ...Object.values(INF.energy.sentences),
      ...Object.values(INF.energy.after),
      ...Object.values(INF.form.sentences),
      ...Object.values(INF.form.after),
      ...Object.values(INF.model_1.sentences),
      ...INF.steps.map((s) => (s.reason ? s.reason.charAt(0).toUpperCase() + s.reason.slice(1) : "")),
    ]);
    for (const t of [...texts(planned), ...texts(after)]) expect(engine.has(t), t).toBe(true);
  });
});
