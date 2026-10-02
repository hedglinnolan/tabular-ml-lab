/**
 * Where each finding stands (M2_CONTRACT §4, §10), read from the decision log the way the
 * server folds it (turbotab/core/decisions.py `fold`, turbotab/core/repairs.py `annotate`):
 *
 *   open       nothing recorded about it yet: it shows its options
 *   applied    one of its repairs was recorded (`apply_repair`)
 *   deferred   held for the question it targets (`defer_finding`); it resurfaces there
 *   dismissed  seen and set aside (`dismiss_finding`)
 *   answered   settled by an answer elsewhere: its routed question, or another finding's repair
 *
 * A revert takes its record out of the log (a revert of a revert puts it back), so a reverted
 * repair reopens the finding. Pure: the Record and the tests share it.
 */
import type { QuestionKey } from "../../api/m1-types";
import type { FindingDecision, RepairOption } from "../../api/m2-types";
import type { Decision, DecisionRecord, Finding } from "../../api/schema";

export type FindingState =
  | { kind: "open" }
  | {
      kind: "applied";
      seq: number;
      recordId: string;
      option: string | null;
      sentence: string | null;
    }
  | { kind: "deferred"; seq: number; recordId: string; to: QuestionKey }
  | { kind: "dismissed"; seq: number; recordId: string; sentence: string | null }
  | { kind: "answered"; seq: number; recordId: string };

const FINDING_KINDS = new Set(["apply_repair", "defer_finding", "dismiss_finding"]);

const isFindingDecision = (d: Decision): d is FindingDecision => FINDING_KINDS.has(d.kind);

/** Record id -> the id of the revert that cancels it (a later revert of a revert reinstates). */
export function cancelledIds(records: readonly DecisionRecord[]): Set<string> {
  const ordered = [...records].sort((a, b) => b.seq - a.seq); // latest first
  const cancelled = new Set<string>();
  for (const r of ordered) {
    if (cancelled.has(r.id)) continue;
    if (r.decision.kind === "revert") cancelled.add(r.decision.decision_id);
  }
  return cancelled;
}

/** Finding id -> the live record that holds its disposition (the latest write per finding). */
export function dispositionRecords(
  records: readonly DecisionRecord[],
): Map<string, DecisionRecord> {
  const cancelled = cancelledIds(records);
  const out = new Map<string, DecisionRecord>();
  for (const r of [...records].sort((a, b) => a.seq - b.seq)) {
    if (cancelled.has(r.id) || !isFindingDecision(r.decision)) continue;
    out.set(r.decision.finding_id, r);
  }
  return out;
}

/** Where a finding stands. The server's `answered_by` (a record id) wins when it names a live
 *  record; the log is read for the disposition, so the answer is the same with or without it. */
export function findingState(f: Finding, records: readonly DecisionRecord[]): FindingState {
  const own = dispositionRecords(records).get(f.id);
  if (own) {
    const d = own.decision as FindingDecision;
    switch (d.kind) {
      case "apply_repair":
        return {
          kind: "applied",
          seq: own.seq,
          recordId: own.id,
          option: d.option,
          sentence: own.sentence,
        };
      case "defer_finding":
        return { kind: "deferred", seq: own.seq, recordId: own.id, to: d.to as QuestionKey };
      case "dismiss_finding":
        return { kind: "dismissed", seq: own.seq, recordId: own.id, sentence: own.sentence };
    }
  }
  const by = f.answered_by ? records.find((r) => r.id === f.answered_by) : undefined;
  if (by && !cancelledIds(records).has(by.id))
    return { kind: "answered", seq: by.seq, recordId: by.id };
  return { kind: "open" };
}

/** A repair option's identity on its card: two options may share a key (`level` for each level). */
export function repairId(option: RepairOption, index: number): string {
  return `${option.key}#${index}`;
}

/** Which option of a finding a recorded repair was: by its key and its params. */
export function recordedRepair(
  f: Finding,
  state: FindingState,
  records: readonly DecisionRecord[],
): string | null {
  if (state.kind !== "applied") return null;
  const rec = records.find((r) => r.id === state.recordId);
  const d = rec?.decision;
  if (!d || d.kind !== "apply_repair") return null;
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  const i = f.repairs.findIndex(
    (o) =>
      o.key === d.option &&
      (same(o.decision.params, d.params) ||
        f.repairs.filter((x) => x.key === d.option).length === 1),
  );
  return i === -1 ? null : repairId(f.repairs[i]!, i);
}

// ── deferral: resurfacing inside the question, pre-checked ───────────────────

/** A deferred finding's place in its question: pre-checked with its first repair, or unchecked. */
export type DeferredChoice = { checked: true; repair: string } | { checked: false };

/** The default for a resurfaced finding: checked, with its first repair (pre-checked, §10). */
export function defaultChoice(f: Finding): DeferredChoice | null {
  return f.repairs.length ? { checked: true, repair: repairId(f.repairs[0]!, 0) } : null;
}

/**
 * What recording an answer to `key` also records for the findings held for it: the chosen repair
 * of each checked one, and a dismissal of each one the user unchecked. A finding with no repair
 * is answered by the question itself, so it adds nothing. Only findings deferred before `answer`.
 */
export function followUps(
  key: QuestionKey,
  answer: DecisionRecord,
  findings: readonly Finding[],
  records: readonly DecisionRecord[],
  choices: Readonly<Record<string, DeferredChoice | undefined>>,
): Decision[] {
  const out: Decision[] = [];
  for (const f of findings) {
    const st = findingState(f, records);
    if (st.kind !== "deferred" || st.to !== key || st.seq > answer.seq) continue;
    const choice = choices[f.id] ?? defaultChoice(f);
    if (!choice) continue;
    if (choice.checked) {
      const i = f.repairs.findIndex((o, j) => repairId(o, j) === choice.repair);
      const option = f.repairs[i] ?? f.repairs[0];
      if (option) out.push(option.decision as Decision);
    } else {
      out.push({
        kind: "dismiss_finding",
        finding_id: f.id,
        reason: `Not applied when ${key.replace(/_/g, " ")} was answered.`,
      });
    }
  }
  return out;
}

/** The findings held for `key`, in the order they were deferred. */
export function heldFor(
  key: QuestionKey,
  findings: readonly Finding[],
  records: readonly DecisionRecord[],
  ids?: readonly string[],
): Finding[] {
  const held = findings
    .map((f) => ({ f, st: findingState(f, records) }))
    .filter(({ f, st }) => st.kind === "deferred" && st.to === key && (!ids || ids.includes(f.id)));
  return held
    .sort((a, b) => (a.st as { seq: number }).seq - (b.st as { seq: number }).seq)
    .map((x) => x.f);
}
