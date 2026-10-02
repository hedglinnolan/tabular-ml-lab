/**
 * Where a finding stands, read from the decision log as the server folds it (M2_CONTRACT §4):
 * the repair disposition transitions (apply, defer, dismiss, revert, change), the server's
 * `answered_by`, and deferral's other half — what recording the question it was held for records.
 */
import type { Decision, DecisionRecord, Finding } from "../../api/schema";
import {
  defaultChoice,
  findingState,
  followUps,
  heldFor,
  recordedRepair,
  repairId,
} from "./findingState";

let seq = 0;
function rec(decision: Decision, id = `r${seq + 1}`): DecisionRecord {
  seq += 1;
  return {
    id,
    seq,
    at: "2026-10-01T12:00:00Z",
    note: null,
    sentence: null,
    post_seal: false,
    decision,
  } as DecisionRecord;
}

const apply = (finding_id: string, option: string, params: Record<string, unknown> = {}) =>
  ({ kind: "apply_repair", finding_id, option, params }) as Decision;
const defer = (finding_id: string, to: string) =>
  ({ kind: "defer_finding", finding_id, to }) as Decision;
const dismiss = (finding_id: string) =>
  ({ kind: "dismiss_finding", finding_id, reason: null }) as Decision;
const revert = (decision_id: string) => ({ kind: "revert", decision_id }) as Decision;

function finding(id: string, options: [string, Record<string, unknown>][] = []): Finding {
  return {
    id,
    severity: "warning",
    title: id,
    detail: id,
    why_it_matters: null,
    affected_columns: ["kcal"],
    source: "pack",
    lens: "clinical",
    evidence: null,
    summary: id,
    routes_to: "exclusions",
    lever_label: null,
    group: null,
    disposition: null,
    answered_by: null,
    repairs: options.map(([key, params]) => ({
      key,
      label: key,
      consequence: `${key} happens.`,
      row_local: true,
      effect: "values",
      sentence: `${key} was applied.`,
      decision: { kind: "apply_repair", finding_id: id, option: key, params },
    })),
  } as Finding;
}

beforeEach(() => {
  seq = 0;
});

const KCAL = finding("pack::clinical::impossible_vs_extreme", [
  ["set_missing", { column: "kcal" }],
  ["exclude_rows", { column: "kcal" }],
  ["mark_unusable", { column: "kcal" }],
]);

describe("a finding's disposition, from the log", () => {
  it("is open until something is recorded about it", () => {
    expect(findingState(KCAL, []).kind).toBe("open");
    expect(findingState(KCAL, [rec(dismiss("another"))]).kind).toBe("open");
  });

  it("moves through apply, change and revert as the server folds them", () => {
    const a = rec(apply(KCAL.id, "set_missing", { column: "kcal" }));
    const st = findingState(KCAL, [a]);
    expect(st).toMatchObject({ kind: "applied", option: "set_missing", recordId: a.id, seq: 1 });

    // A changed repair: the latest write per finding wins.
    const b = rec(apply(KCAL.id, "exclude_rows", { column: "kcal" }));
    expect(findingState(KCAL, [a, b])).toMatchObject({ kind: "applied", option: "exclude_rows" });

    // Reverting the latest puts the earlier one back; reverting both reopens the finding.
    const r1 = rec(revert(b.id));
    expect(findingState(KCAL, [a, b, r1])).toMatchObject({ option: "set_missing" });
    const r2 = rec(revert(a.id));
    expect(findingState(KCAL, [a, b, r1, r2]).kind).toBe("open");
    // A revert of a revert reinstates what it cancelled.
    const r3 = rec(revert(r2.id));
    expect(findingState(KCAL, [a, b, r1, r2, r3])).toMatchObject({ option: "set_missing" });
  });

  it("holds a deferral with the question it targets, and a dismissal as seen", () => {
    const d = rec(defer(KCAL.id, "exclusions"));
    expect(findingState(KCAL, [d])).toEqual({
      kind: "deferred",
      seq: 1,
      recordId: d.id,
      to: "exclusions",
    });
    const x = rec(dismiss(KCAL.id));
    expect(findingState(KCAL, [d, x])).toMatchObject({ kind: "dismissed", seq: 2 });
  });

  it("folds into 'answered' only when the server names a live record", () => {
    const answer = rec({ kind: "set_exclusions", rules: [] } as unknown as Decision, "ans");
    const served = { ...KCAL, answered_by: "ans" };
    expect(findingState(served, [answer])).toEqual({
      kind: "answered",
      seq: 1,
      recordId: "ans",
    });
    // The answer reverted: the finding is open again, whatever the stale artifact says.
    expect(findingState(served, [answer, rec(revert("ans"))]).kind).toBe("open");
    // Its own disposition wins over another record that answered it.
    const own = rec(dismiss(KCAL.id));
    expect(findingState(served, [answer, own]).kind).toBe("dismissed");
  });

  it("names the recorded option by its params when two options share a key", () => {
    const sex = finding("binary_text__sex", [
      ["level", { column: "sex", one: "F", zero: "M" }],
      ["level", { column: "sex", one: "M", zero: "F" }],
    ]);
    const records = [rec(apply(sex.id, "level", { column: "sex", one: "M", zero: "F" }))];
    const st = findingState(sex, records);
    expect(recordedRepair(sex, st, records)).toBe(repairId(sex.repairs[1]!, 1));
    expect(recordedRepair(sex, findingState(sex, []), [])).toBeNull();
  });
});

describe("deferral resurfaces inside its question, pre-checked", () => {
  const SEX = finding("binary_text__sex", [["level", { column: "sex", one: "F", zero: "M" }]]);
  const NOTE = finding("structural__constant", []);

  it("lists the findings held for a question, in the order they were set aside", () => {
    const records = [
      rec(defer(SEX.id, "exclusions")),
      rec(defer(KCAL.id, "exclusions")),
      rec(defer(NOTE.id, "missing")),
    ];
    const all = [KCAL, SEX, NOTE];
    expect(heldFor("exclusions", all, records).map((f) => f.id)).toEqual([SEX.id, KCAL.id]);
    expect(heldFor("missing", all, records).map((f) => f.id)).toEqual([NOTE.id]);
    // The Router's list narrows it (a server that knows better).
    expect(heldFor("exclusions", all, records, [KCAL.id]).map((f) => f.id)).toEqual([KCAL.id]);
    // Once acted on, it is no longer held.
    const done = [...records, rec(dismiss(SEX.id))];
    expect(heldFor("exclusions", all, done).map((f) => f.id)).toEqual([KCAL.id]);
  });

  it("comes back checked with its first repair", () => {
    expect(defaultChoice(KCAL)).toEqual({ checked: true, repair: repairId(KCAL.repairs[0]!, 0) });
    expect(defaultChoice(NOTE)).toBeNull();
  });

  it("records the checked repair, or a dismissal, when the question is answered", () => {
    const held = [rec(defer(KCAL.id, "exclusions")), rec(defer(SEX.id, "exclusions"))];
    const answer = rec({ kind: "set_exclusions", rules: [] } as unknown as Decision);
    const records = [...held, answer];
    // Untouched: the pre-checked first repair is applied.
    expect(followUps("exclusions", answer, [KCAL, SEX], records, {})).toEqual([
      KCAL.repairs[0]!.decision,
      SEX.repairs[0]!.decision,
    ]);
    // Another repair chosen for one, the other unchecked: that repair, and a dismissal.
    const out = followUps("exclusions", answer, [KCAL, SEX], records, {
      [KCAL.id]: { checked: true, repair: repairId(KCAL.repairs[1]!, 1) },
      [SEX.id]: { checked: false },
    });
    expect(out[0]).toEqual(KCAL.repairs[1]!.decision);
    expect(out[1]).toMatchObject({ kind: "dismiss_finding", finding_id: SEX.id });
  });

  it("acts only for findings held for that question before it was answered", () => {
    const early = rec({ kind: "set_exclusions", rules: [] } as unknown as Decision);
    const held = rec(defer(KCAL.id, "exclusions"));
    const elsewhere = rec(defer(SEX.id, "missing"));
    const records = [early, held, elsewhere];
    // Held after this answer: the next answer to the question acts on it, not this one.
    expect(followUps("exclusions", early, [KCAL, SEX], records, {})).toEqual([]);
    const later = rec({ kind: "set_exclusions", rules: [] } as unknown as Decision);
    expect(followUps("exclusions", later, [KCAL, SEX], [...records, later], {})).toEqual([
      KCAL.repairs[0]!.decision,
    ]);
    // A finding with no repair is answered by the question itself: nothing more is recorded.
    const note = rec(defer(NOTE.id, "missing"));
    const miss = rec({ kind: "set_missing" } as unknown as Decision);
    expect(followUps("missing", miss, [NOTE], [note, miss], {})).toEqual([]);
  });
});
