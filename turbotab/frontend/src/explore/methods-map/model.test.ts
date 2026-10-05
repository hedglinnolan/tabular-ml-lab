/**
 * The methods map's logic, on the captured fixture: what a block confirmation settles, which
 * captured fit and lock a plan selects, that a preview is the one taken on the answers' state, and
 * that the record only ever quotes the engine.
 */
import { describe, expect, it } from "vitest";
import { INF } from "./fixture";
import {
  INITIAL,
  SINGLES_TO_UNLOCK,
  blockOffer,
  fitFor,
  lockKey,
  lockSentence,
  previewFor,
  record,
  reduce,
  truthTriple,
  guessTriple,
  type Action,
  type Answers,
} from "./model";

const play = (actions: Action[], from: Answers = INITIAL) => actions.reduce(reduce, from);
const keys = INF.readings.items.map((i) => i.key);
const adjust: Action[] = INF.adjustment.groups.map((g) => ({
  type: "adjust",
  group: g.key,
  answers: Object.fromEntries(g.columns.map((c) => [c, guessTriple(g.key) ?? truthTriple(c)!])),
}));
const planned = play([
  { type: "reading", key: keys[0]! },
  { type: "reading", key: keys[1]! },
  { type: "reading", key: keys[2]! },
  { type: "block" },
  { type: "unit" },
  { type: "exclusions", key: "none" },
  { type: "exposure" },
  ...adjust,
  { type: "model1", value: "guess" },
]);

describe("the readings' block confirmation", () => {
  it("unlocks only after three single confirmations and lists exactly the rest", () => {
    expect(blockOffer(play(keys.slice(0, SINGLES_TO_UNLOCK - 1).map((key) => ({ type: "reading", key }))))).toBeNull();
    const three = play(keys.slice(0, SINGLES_TO_UNLOCK).map((key) => ({ type: "reading", key })));
    const offer = blockOffer(three)!;
    expect(offer.keys).toEqual(keys.slice(SINGLES_TO_UNLOCK));
    const after = reduce(three, { type: "block" });
    expect(keys.every((k) => after.readings[k])).toBe(true);
    expect(keys.slice(0, SINGLES_TO_UNLOCK).every((k) => after.readings[k] === "single")).toBe(true);
  });

  it("names, in its sentence, what it settles (the engine's own sentence for that set)", () => {
    const offer = blockOffer(play([2, 5, 9].map((i) => ({ type: "reading", key: keys[i]! }))))!;
    expect(offer.keys).not.toContain(keys[2]);
    expect(INF.readings.block.sentences).toContain(offer.sentence);
    expect(offer.sentence.startsWith("Confirmed together, each as the question showed it")).toBe(true);
  });
});

describe("the plan selects a captured fit and lock", () => {
  it("the fixture's answers select the fit captured for them, and the lock's digest is the engine's", () => {
    const pick = fitFor(planned);
    expect(pick.kind).toBe("fit");
    if (pick.kind !== "fit") return;
    const m2 = pick.fit.sequence.find((s) => s.key === "model_2")!;
    expect(m2.n_rows).toBe(21849);
    expect(m2.effects[0]!.feature).toBe("sugar");
    const locked = reduce(planned, { type: "lock" });
    expect(lockKey(planned)).toBe("nl0-0g");
    expect(lockSentence(locked)).toContain(INF.lock.digests["nl0-0g"]);
  });

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

  it("is locked once: a change after it is kept and marked, and nothing is undone", () => {
    const locked = reduce(planned, { type: "lock" });
    const changed = reduce(locked, { type: "energy", method: "residual" });
    expect(changed.locked!.energy).toBe("standard");
    expect(changed.after).toEqual([{ node: "energy", value: "residual" }]);
    expect(reduce(changed, { type: "undo", node: "exposure" })).toBe(changed);
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
    const locked = reduce(planned, { type: "lock" });
    const texts = record(locked, "inference")
      .flatMap((s) => s.lines)
      .map((l) => l.text)
      .filter((t): t is string => !!t);
    const engine = new Set<string>([
      ...Object.values(INF.stated),
      ...Object.values(INF.readings.single),
      ...INF.readings.block.sentences,
      INF.readings.unit.sentence,
      ...Object.values(INF.exclusions.sentences).filter((t): t is string => !!t),
      ...Object.values(INF.sensitivity.sentences),
      INF.estimand.sentence,
      ...Object.values(INF.adjustment.group_sentences),
      ...INF.adjustment.unguessed.map((u) => u.sentence),
      ...Object.values(INF.energy.sentences),
      ...Object.values(INF.form.sentences),
      ...Object.values(INF.missing.sentences),
      ...Object.values(INF.model_1.sentences),
      ...INF.steps.map((s) => (s.reason ? s.reason.charAt(0).toUpperCase() + s.reason.slice(1) : "")),
      ...Object.values(INF.fits).map((f) => f.methods),
      INF.models_sentence.slice(0, INF.models_sentence.indexOf(". ") + 1),
      lockSentence(locked)!,
    ]);
    for (const t of texts) expect(engine.has(t), t).toBe(true);
  });
});
