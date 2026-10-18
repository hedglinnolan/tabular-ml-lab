/**
 * The mock's quest log and the reads around it (dev:mock only; plans/W1_PART2_PLAN.md §2 "W1-E"),
 * so the quest shell can be walked on the mock's projects: GET /quest, /record, /triage,
 * /materiality, /checklist, /methods, /methods/sections and /export, and POST /fit and
 * /fit/cancel. Under prediction it also keeps Fit's gate as the engine does: the fit is served
 * withheld until Fit is pressed (`servedFit`, read by m1-record.ts), and the opening of the
 * held-out rows waits for the press and then for the triage (`openingRefusal`).
 *
 * The quest log is built from the project's view as the mock serves it (the Router's steps, the
 * decisions, the findings) and placed by the engine's own registry: each question's and
 * declaration's stage, label, order and name, the sweep words and the Fit lock's sentences come
 * from src/quest/fixtures/quest-contract.json (written by the engine's renderer-contract test); a
 * column's own role line's words, choices and place from src/quest/cards/roles/fixture.json (written
 * by the engine's test of those lines), under the project's goal and, under prediction, its lenses.
 * What the mock decides itself is said where it does: which declarations apply (the Predict and
 * Estimate walks'), which stages are reached, and when a finding is settled. Every number is the
 * mock's.
 */
import { http, HttpResponse, type HttpHandler } from "msw";
import type {
  ChecklistReport,
  DecisionRecord,
  Finding,
  FindingsArtifact,
  FitLock,
  ForTheRecord,
  Ledger,
  ProjectView,
  QuestLine,
  QuestLog,
  QuestStage,
  StageResult,
  Triage,
} from "../api/schema";
import type { MethodsSections, MethodsText } from "../api/m3-types";
import type { RolesArtifact } from "../api/m1-types";
import ROLES from "../quest/cards/roles/fixture.json";
import { predictKind } from "./roleWords";
import { CONTRACT } from "../quest/fixtures/contract";
import type { MockServer } from "./db";

type Label = QuestLine["label"];
type Status = QuestLine["status"];

const C = CONTRACT;
/** Decision kind -> its stage, as the engine places every kind (the quest log's `kinds`). */
const KINDS: Record<string, string> = C.logs.results_reached.log.kinds;
/** A question the Router states rather than asks reads For the record where no other answer can
 *  be offered (core/quest.py STATED); every other stated question is a Confirm line. */
const STATED: Record<string, Label> = { task: "For the record", clusters: "For the record" };

function line(p: Partial<QuestLine> & Pick<QuestLine, "id" | "key" | "source" | "label" | "name" | "status">): QuestLine {
  return {
    counted: p.label === "Decide",
    order: 0,
    decision_id: null,
    reason: null,
    waiting_for: [],
    computing: [],
    reopened_by: null,
    would_change: null,
    changes_nothing: null,
    items: [],
    changed_since: null,
    ...p,
  };
}

const latest = (records: readonly DecisionRecord[], test: (d: DecisionRecord["decision"]) => boolean) =>
  [...records].reverse().find((r) => test(r.decision));

/**
 * A role line's words under the project's goal (core/quest.py `role_line_text`): with no goal and
 * under inference, Estimate's; under prediction each predictor role by the roles card's group for it
 * in a project with these lenses (the roles ruling of 2026-10-17), each change's label the engine's
 * own (a move between the card's groups, which names no step: the ruling of 2026-10-18).
 * `reasons`: why a line waits, by the proposed role, its last clause naming what that
 * role's presses do. `group`: a group line's words (ruling R-g), by role.
 */
function roleWords(purpose: string | null | undefined, lenses: readonly string[] | null | undefined) {
  const T = ROLES.templates;
  const estimate = {
    words: ROLES.words as Record<string, string>,
    reasons: ROLES.reasons as Record<string, string>,
    choices: ROLES.choices as Record<string, string[]>,
    change: (_role: string, other: string) => T.change.replaceAll("{other}", (ROLES.words as Record<string, string>)[other] ?? other),
    group: ROLES.group as GroupWords,
    evidence: (reason: string) => reason,
  };
  if (purpose !== "prediction") return estimate;
  const k = ROLES.under.prediction.kinds[predictKind(lenses)];
  const changes = k.changes as Record<string, Record<string, string>>;
  return {
    words: k.words as Record<string, string>,
    reasons: k.reasons as Record<string, string>,
    choices: k.choices as Record<string, string[]>,
    change: (role: string, other: string) => changes[role]?.[other] ?? T.change.replaceAll("{other}", other),
    group: k.group as GroupWords,
    // A group's shared reason without its study clause (core/readings.py `role_evidence`).
    evidence: (reason: string) =>
      (ROLES.under.prediction.study_clauses as [string, string][]).reduce((said, [estimate, predict]) => said.replaceAll(estimate, predict), reason),
  };
}

interface GroupWords {
  many: Record<string, string>;
  reasons: Record<string, string>;
  reasons_two: Record<string, string>;
  press: Record<string, string>;
}

/** The roles stage's proposal of a column as the mock reads it: its reason, and its confidence
 *  where the artifact is served (a reason alone groups nothing: ruling R-g reads all three). */
export type Proposed = Readonly<Record<string, string | { reason: string; confidence: string }>>;

/** "`a`, `b` and 298 more", as core/voice.py `listing` names a group's first columns. */
function firstColumns(columns: readonly string[]): string {
  const ticked = columns.map((c) => `\`${c}\``);
  if (ticked.length <= 3) return ticked.length < 2 ? (ticked[0] ?? "") : `${ticked.slice(0, -1).join(", ")} and ${ticked.at(-1)}`;
  return `${ticked.slice(0, 2).join(", ")} and ${(ticked.length - 2).toLocaleString("en-US")} more`;
}

/**
 * Your data's role lines (core/quest.py `_role_lines`; the walkthrough of 2026-10-17, finding 1):
 * each column the roles answer recorded below high confidence as proposed (`roles_unconfirmed`) waits
 * in a counted Decide, open until a confirmation of its role, then answered by that record; none
 * while the roles question is asked again. Columns the roles stage proposed with one role, one
 * confidence and one reason word for word are one judgment (ruling R-g): one line, its first item
 * the group's reading with its one press, then each column with its moves; a column confirmed on a
 * press of its own leaves the group for a line of its own. The engine moves a column's recorded role
 * to the role confirmed for it, so a confirmation here is the column's role. The evidence is the
 * roles stage's reason for the proposal (`proposed`), else the ledger's own reading's.
 */
export function mockRoleLines(view: ProjectView, proposed: Proposed = {}): QuestLine[] {
  const { state, decisions, interview } = view;
  const named = new Set(state.roles_unconfirmed ?? []);
  const step = interview.find((s) => s.key === "roles");
  if (!named.size || step?.status === "open" || step?.status === "waiting") return [];
  const T = ROLES.templates;
  const G = ROLES.group;
  const W = roleWords(state.purpose, state.lens);
  const fill = (template: string, column: string, role: string) =>
    template.replaceAll("{column}", column).replaceAll("{role}", W.words[role] ?? role);
  const confirmedBy = (column: string) =>
    latest(
      decisions,
      (d) =>
        (d.kind === "confirm_role" && d.column === column) ||
        (d.kind === "confirm_reading" && d.reading === "role" && d.column === column) ||
        (d.kind === "confirm_readings" && d.items.some((i) => i.reading === "role" && i.column === column)),
    );
  const inGroupPress = (column: string) => {
    const d = confirmedBy(column)?.decision;
    return d?.kind === "confirm_readings" && d.items.filter((i) => i.reading === "role").length > 1;
  };
  const reasonOf = (column: string) => {
    const p = proposed[column];
    return typeof p === "string" ? p : p?.reason;
  };
  const option = (column: string, r: string, label: string) => ({ label, decision: { kind: "confirm_role", column, role: r } as Record<string, unknown> });
  const columns = Object.entries(state.roles ?? {}).filter(([column]) => named.has(column) && column !== state.target);
  // The judgments: the columns recorded as proposed, by role, confidence and reason.
  const judged = new Map<string, string[]>();
  for (const [column, recorded] of columns) {
    const p = proposed[column];
    if (!p || typeof p === "string") continue;
    const key = [String(recorded), p.confidence, p.reason].join("\n");
    judged.set(key, [...(judged.get(key) ?? []), column]);
  }
  const grouped = new Map<string, string[]>();
  for (const members of judged.values()) {
    const kept = members.filter((c) => (state.role_confirmations?.[c] ?? null) === null || inGroupPress(c));
    if (kept.length > 1) for (const c of kept) grouped.set(c, kept);
  }
  const lines: QuestLine[] = [];
  const done = new Set<string>();
  columns.forEach(([column, recorded], i) => {
    const order = ROLES.place.order + i / 1000;
    const members = grouped.get(column);
    if (members) {
      if (done.has(column)) return;
      members.forEach((c) => done.add(c));
      const role = String(recorded);
      const p = proposed[column] as { reason: string; confidence: string };
      const waiting = members.filter((c) => (state.role_confirmations?.[c] ?? null) === null);
      const n = (k: number) => k.toLocaleString("en-US");
      const head = (waiting.length ? G.head.waiting : G.head.settled)
        .replaceAll("{n}", n(members.length))
        .replaceAll("{many}", W.group.many[role] ?? role)
        .replaceAll("{confidence}", p.confidence);
      const press = waiting.length
        ? [
            {
              label: (W.group.press[role] ?? "").replaceAll("{n}", n(waiting.length)),
              decision: { kind: "confirm_readings", items: waiting.map((c) => ({ reading: "role", column: c, value: role })) } as Record<string, unknown>,
            },
          ]
        : [];
      const why = waiting.length === 2 ? W.group.reasons_two[role] : (W.group.reasons[role] ?? "").replaceAll("{n}", n(waiting.length));
      const all = judged.get([role, p.confidence, p.reason].join("\n")) ?? members;
      lines.push(
        line({
          id: ROLES.place.id,
          key: G.key.replace("{column}", all[0] ?? column),
          source: "reading",
          label: "Decide",
          name: G.name.replace("{columns}", firstColumns(members)),
          status: waiting.length ? "open" : "answered",
          counted: true,
          order,
          decision_id: waiting.length ? null : (confirmedBy(members[0]!)?.id ?? null),
          reason: waiting.length ? (why ?? null) : null,
          items: [
            { kind: G.item, column: members[0]!, value: role, words: head, evidence: W.evidence(p.reason), options: press },
            ...members.map((c) => {
              const settled = (state.role_confirmations?.[c] ?? null) !== null;
              return {
                kind: "role",
                column: c,
                value: role,
                words: fill(settled ? T.settled : T.proposed, c, role),
                evidence: "",
                options: (W.choices[role] ?? ["covariate"]).map((other) => option(c, other, fill(W.change(role, other), c, role))),
              };
            }),
          ],
        }),
      );
      return;
    }
    const confirmed = state.role_confirmations?.[column] ?? null;
    const role = String(confirmed ?? recorded);
    const waiting = confirmed === null;
    const options = [
      ...(waiting ? [option(column, role, fill(T.confirm, column, role))] : []),
      ...(W.choices[role] ?? ["covariate"]).map((other) => option(column, other, fill(W.change(role, other), column, role))),
    ];
    lines.push(
      line({
        id: ROLES.place.id,
        key: ROLES.key.replace("{column}", column),
        source: "reading",
        label: "Decide",
        name: fill(T.name, column, role),
        status: waiting ? "open" : "answered",
        counted: true,
        order,
        decision_id: waiting ? null : (confirmedBy(column)?.id ?? null),
        reason: waiting && W.reasons[role] ? fill(W.reasons[role], column, role) : null,
        items: [
          {
            kind: "role",
            column,
            value: role,
            words: fill(waiting ? T.proposed : T.settled, column, role),
            evidence: reasonOf(column) || ROLES.unread_evidence,
            options,
          },
        ],
      }),
    );
  });
  return lines;
}

/** The quest log of a mock project's view, Fit pressed or not; `proposed`: the roles stage's reason
 *  (and confidence) for each column's proposal, where its artifact is served. */
export function mockQuest(
  view: ProjectView,
  findings: readonly Finding[] | null,
  pressed: boolean,
  reasons: Proposed = {},
): QuestLog {
  const { state, decisions, interview } = view;
  const predict = state.purpose === "prediction";
  const estimate = state.purpose === "inference";
  const placed: { stage: string; line: QuestLine }[] = [];
  const stepOf = (key: string) => interview.find((s) => s.key === key);
  const settled = (key: string) => {
    const st = stepOf(key)?.status;
    return st === "answered" || st === "skipped";
  };
  const recordOf = (kind: string) => latest(decisions, (d) => d.kind === kind);
  const waitingOn = (key: string) => [
    { key, stage: C.questions[key]?.stage ?? "data", name: C.questions[key]?.name ?? key },
  ];

  // ── the Router's questions ──
  for (const step of interview) {
    const place = C.questions[step.key];
    if (!place || step.status === "not_applicable") continue;
    let status: Status =
      step.status === "skipped" ? "set_for_you" : (step.status as Exclude<typeof step.status, "skipped" | "not_applicable">);
    const label: Label = step.status === "skipped" && place.label === "Decide" ? (STATED[step.key] ?? "Confirm") : place.label;
    let waiting_for = step.waiting_on.map((k) => waitingOn(k)[0]!);
    // Under Predict the families wait for the predictor-selection question (the engine's shelf
    // waits for it), and the opening waits for Fit itself.
    if (predict && step.key === "models" && status === "open" && !recordOf("set_selection")) {
      status = "waiting";
      waiting_for = [{ key: "set_selection", stage: "models", name: C.declarations.set_selection!.name }];
    }
    if (predict && step.key === "open_seal" && status === "open" && !pressed) status = "waiting";
    placed.push({
      stage: place.stage,
      line: line({
        id: `q:${step.key}`,
        key: step.key,
        source: "question",
        label,
        name: place.name,
        status,
        counted: label === "Decide",
        order: place.order,
        decision_id: step.decision_id,
        reason: status === "set_for_you" ? step.reason : null,
        waiting_for,
      }),
    });
  }

  // ── each column's own role, where the roles answer recorded it below high (core/quest.py) ──
  for (const l of mockRoleLines(view, reasons)) placed.push({ stage: ROLES.place.stage, line: l });

  // ── the declarations the mock's Predict and Estimate walks meet (core/quest.py DECLARATIONS) ──
  const declare = (kind: string, applies: boolean, reads: string | null) => {
    const place = C.declarations[kind];
    if (!place || !applies) return;
    const rec = recordOf(kind);
    const ready = reads === null || settled(reads);
    const status: Status = rec ? "answered" : !ready ? "waiting" : place.label === "Confirm" ? "set_for_you" : "open";
    placed.push({
      stage: place.stage,
      line: line({
        id: `decision:${kind}`,
        key: kind,
        source: "declaration",
        label: place.label,
        name: place.name,
        status,
        counted: place.label === "Decide" && place.counted,
        order: place.order,
        decision_id: rec?.id ?? null,
        waiting_for: !rec && !ready && reads ? waitingOn(reads) : [],
      }),
    });
  };
  const families = state.models ?? [];
  declare("join_files", true, null);
  declare("import_codebook", true, null);
  declare("set_intended_use", predict, "purpose");
  declare("set_selection", predict, "roles");
  declare("set_sensitivity", (predict || estimate) && !!state.exclusions?.length, "exclusions");
  declare("set_model_sequence", estimate, "adjustment");
  declare("set_validation", predict, "split");
  declare("set_levers", predict, "models");
  declare("set_updating", predict && families.includes("linear"), "models");
  declare("set_explain", predict || estimate, "models");

  // ── the noticings: each in the stage that asks its question, settled by its disposition or by
  // that question's answer; one with nowhere to go is For the record ──
  for (const f of findings ?? []) {
    const disposed = latest(
      decisions,
      (d) => (d.kind === "dismiss_finding" || d.kind === "defer_finding" || d.kind === "apply_repair") && d.finding_id === f.id,
    );
    const to = f.routes_to ?? null;
    const routed = to ? stepOf(to) : undefined;
    const answeredBy = disposed?.id ?? (routed?.status === "answered" ? routed.decision_id : null);
    const label: Label = to || f.repairs.length ? "Decide" : "For the record";
    placed.push({
      stage: (to && C.questions[to]?.stage) || "data",
      line: line({
        id: `finding:${f.id}`,
        key: f.id,
        source: "finding",
        label,
        name: f.summary ?? f.title ?? f.id,
        status: answeredBy ? "answered" : "open",
        counted: label === "Decide",
        order: (to && C.questions[to]?.order ? C.questions[to]!.order - 0.25 : 11),
        decision_id: answeredBy ?? null,
      }),
    });
  }

  // ── the stages: their lines, sweeps and progress ──
  const reachedBefore = (() => {
    let at = 0;
    C.stages.slice(0, 5).forEach((s, i) => {
      if (placed.some((p) => p.stage === s.key && p.line.label !== "For the record" && p.line.status !== "waiting")) at = i;
    });
    if (state.target && at < 2) at = 2; // First look, once the outcome is chosen
    return at;
  })();
  const resultsOpen = predict ? pressed : !!state.plan_locked;
  const stages: QuestStage[] = C.stages.map((s, i) => {
    const lines = placed.filter((p) => p.stage === s.key).map((p) => p.line);
    const confirms = lines.filter((l) => l.label === "Confirm");
    const swept = latest(decisions, (d) => d.kind === "confirm_sweep" && d.stage === s.key && d.sweep === "defaults");
    if (swept) for (const l of confirms) if (l.status === "set_for_you") Object.assign(l, { status: "answered", decision_id: swept.id });
    const words = C.sweep_words[String(Math.min(confirms.length, 8))];
    const sweep = confirms.length
      ? {
          lines: confirms.length,
          answered: confirms.every((l) => l.status === "answered"),
          id: `other:confirm-sweep:${s.key}`,
          heading: words?.[0] ?? "",
          action: words?.[1] ?? "",
          confirmed_by: swept?.id ?? null,
          changed: [],
        }
      : null;
    const reached = i <= 4 ? i <= reachedBefore : resultsOpen;
    // Ruling R-a (core/quest.py `quest_log`): under Predict an open noticing waits for the triage in
    // Results, not for the stage it is placed in, so outside Results its stage's count leaves it out.
    const counted = lines.filter(
      (l) => l.label === "Decide" && l.counted && !(predict && s.key !== "results" && l.source === "finding" && l.status === "open"),
    );
    const required = counted.length + (sweep ? 1 : 0);
    const answered = counted.filter((l) => l.status === "answered").length + (sweep?.answered ? 1 : 0);
    return {
      key: s.key,
      name: s.name,
      reached,
      progress: reached ? { answered, required, complete: answered === required } : null,
      sweep,
      lines: lines.sort((a, b) => a.order - b.order),
      reopened: [],
    };
  });
  return { version: 5, stages, kinds: KINDS, fit: lockOf(view, pressed) };
}

/** The Fit lock in the engine's words (the contract's samples); the mock never holds a fit. */
export function lockOf(view: ProjectView, pressed: boolean): FitLock {
  const purpose = view.state.purpose ?? null;
  const locked = !!view.state.plan_locked;
  const said =
    purpose === null
      ? C.fit_locks.no_goal
      : purpose === "prediction"
        ? pressed
          ? C.fit_locks.pressed
          : C.fit_locks.ready
        : C.fit_locks.unlocked;
  return {
    purpose,
    locks: purpose !== null && purpose !== "prediction",
    locked,
    at: locked ? (latest(view.decisions, (d) => d.kind === "lock_plan")?.at ?? null) : null,
    sha256: null,
    pressed,
    opened: pressed,
    held: false,
    estimate_seconds: null,
    reason: locked ? "The plan was locked when Fit was pressed, before any estimate was shown." : said.reason,
  };
}

/** The methods text: each recorded sentence, in record order. */
export function mockMethods(view: ProjectView): MethodsText {
  const lines = view.decisions
    .filter((r) => r.sentence)
    .map((r) => ({
      record_id: r.id,
      seq: r.seq,
      kind: r.decision.kind,
      sentence: r.sentence!,
      in_force: true,
      post_seal: !!r.post_seal,
      after_estimates: !!r.after_estimates,
    }));
  return { lines, seen_from: null, text: lines.map((l) => l.sentence).join(" ") };
}

/** The methods in sections: the mock groups its sentences by the stage that records each kind
 *  (the export bundle's own TRIPOD+AI and STROBE sections are the real server's). */
export function mockSections(view: ProjectView): MethodsSections {
  const methods = mockMethods(view);
  const sections = C.stages
    .map((s) => ({
      key: s.key,
      title: s.name,
      item: "",
      entries: methods.lines
        .filter((l) => (KINDS[l.kind] ?? "data") === s.key)
        .map((l) => ({
          source: "record" as const,
          text: l.sentence,
          record_id: l.record_id,
          seq: l.seq,
          kind: l.kind,
          in_force: true,
          after_estimates: l.after_estimates,
          post_seal: l.post_seal,
        })),
    }))
    .filter((s) => s.entries.length);
  return {
    guideline: view.state.purpose === "prediction" ? "TRIPOD+AI" : "STROBE-nut",
    seen_from: null,
    sections,
    count: sections.reduce((n, s) => n + s.entries.length, 0),
  };
}

/** Each stage's For the record lines: the quest log's, with why each was set. */
export function mockRecord(log: QuestLog): ForTheRecord {
  return {
    version: 1,
    stages: log.stages.map((s) => ({
      key: s.key,
      name: s.name,
      lines: s.lines
        .filter((l) => l.label === "For the record")
        .map((l) => ({
          kind: "set_for_you" as const,
          key: l.key,
          id: l.id,
          text: l.reason ? `${l.name.charAt(0).toUpperCase()}${l.name.slice(1)}: ${l.reason}` : l.name,
          decision_id: l.decision_id,
        })),
    })),
  };
}

/** The triage at the gate (before the held-out rows open under Predict): every open noticing,
 *  each recommended "doesn't change your numbers here" by the mock. */
export function mockTriage(view: ProjectView, log: QuestLog): Triage {
  const stage = view.state.purpose === "prediction" ? "results" : "models";
  const confirmed = latest(view.decisions, (d) => d.kind === "confirm_sweep" && d.stage === stage && d.sweep === "noticings");
  const open = log.stages.flatMap((s) =>
    s.lines.filter((l) => l.source === "finding" && l.status === "open").map((l) => ({ s, l })),
  );
  const items = open.map(({ s, l }) => ({
    id: l.key,
    line: l.id,
    stage: s.key,
    summary: l.name,
    severity: "info",
    columns: [],
    question: null,
    recommended: "no_change" as const,
    label: "Doesn't change your numbers here",
    reason: "The mock server recommends no change for every noticing.",
    blocker: false,
    recorded: confirmed ? ("no_change" as const) : null,
    family: "finding",
    band: null,
    measure: null,
    calibrated: null,
    limitation: false,
    done: null,
  }));
  return {
    version: 1,
    stage,
    gate: stage === "results" ? "gate:open-noticings-before-seal" : "gate:open-noticings-before-lock",
    items,
    blockers: 0,
    confirmable: !confirmed,
    answered: !!confirmed,
    confirmed_by: confirmed?.id ?? null,
    changed: [],
    passed: stage === "results" ? !!view.state.seal_opened : !!view.state.plan_locked,
    rows: [],
  };
}

// ── Fit's gate under Predict (core/fit_press.py, core/seal.py), mirrored ─────────

/** The engine's reason no score is served before Fit under prediction (fit_press.FIT_FIRST). */
const FIT_FIRST =
  "No score or estimate is shown until Fit is pressed, so none is seen before the models' settings are confirmed.";
/** The way on it names (fit_press.PRESS_FIT). */
const PRESS_FIT = "Press Fit on the analysis flowchart";
/** What `estimand.withhold` empties in a fit before Fit is pressed: each model's scores
 *  (MODEL_SCORES, and `cv`) and the comparison's (FIT_SCORES). */
const MODEL_SCORES = [
  "holdout",
  "versus_baseline",
  "calibration",
  "holdout_detail",
  "optimism",
  "internal_external",
  "compared_on",
  "performance",
  "nested_cv",
  "calibration_levels",
  "calibration_horizon",
  "calibration_note",
];
const FIT_SCORES = ["selection", "precision", "comparison", "comparisons_note", "result", "nested_offer", "at_opening"];

/**
 * The fit as the real server serves it under prediction before Fit is pressed
 * (`fit_press.served`, `estimand.withhold` with scores withheld): every score and estimate removed,
 * the reason first, `withheld_for` "fit". Under any other goal the mock serves its fit whole, as
 * before (the Estimate lane's lock is W3's).
 */
export function servedFit<T>(fit: T, purpose: string | null | undefined, pressed: boolean): T {
  if (purpose !== "prediction" || pressed || !fit || typeof fit !== "object") return fit;
  const f = fit as Record<string, unknown>;
  const out: Record<string, unknown> = { ...f, withheld: FIT_FIRST, withheld_for: "fit", comparisons: [] };
  for (const k of FIT_SCORES) if (k in out) out[k] = null;
  out.models = ((f.models as Record<string, unknown>[] | undefined) ?? []).map((m) => {
    const x: Record<string, unknown> = {
      ...m,
      cv: {},
      coefficients: null,
      inference: null,
      exposure_tests: [],
      concerns: [FIT_FIRST, ...((m.concerns as string[] | undefined) ?? [])],
    };
    for (const k of MODEL_SCORES) if (k in x) x[k] = null;
    if (x.baseline && typeof x.baseline === "object") x.baseline = { ...(x.baseline as object), value: null };
    return x;
  });
  return out as T;
}

const refusal = (code: string, message: string, exits: { label: string; decision: unknown }[]) => ({
  error: { code, message, exits },
});

/**
 * Why the opening of the held-out rows waits under prediction, in core/seal.py's order and words:
 * after its own checks (opened already, nothing held out, the fit not fresh: the mock's validator,
 * m2-record.ts, refuses those, so they pass here), until Fit is pressed (`fit_not_yet`), then while
 * a noticing has no recorded disposition (`triage_open`, its exits the triage and its Confirm all).
 * The mock's triage holds no blocker and no "act on it", so the engine's other two refusals never
 * arise. Null when nothing here holds it; under other goals the mock keeps its walk (W3).
 */
export function openingRefusal(view: ProjectView, log: QuestLog, pressed: boolean) {
  const { state, stages } = view;
  if (state.purpose !== "prediction") return null;
  if (state.seal_opened || !state.split || state.split.holdout === 0 || stages.fit?.status !== "fresh") return null;
  if (!pressed)
    return refusal(
      "fit_not_yet",
      `The held-out rows open after Fit is pressed: the final model is named on the cross-validated scores it shows. ${FIT_FIRST}`,
      [{ label: PRESS_FIT, decision: null }],
    );
  const t = mockTriage(view, log);
  if (t.passed) return null;
  const waiting = t.items.filter((i) => i.recorded === null);
  if (!waiting.length) return null;
  const n = waiting.length;
  const named = waiting.map((i) => `${i.summary.trim().replace(/\.+$/, "")} (${i.label.toLowerCase()})`).join("; ");
  const where = C.stages.find((s) => s.key === t.stage)?.name ?? "Results";
  return refusal(
    "triage_open",
    `The held-out rows open once, after every open noticing is triaged: ${n} ${n === 1 ? "is" : "are"} still open, each with its recommended disposition: ${named}. Triage ${n === 1 ? "it" : "them"} in ${where} first, so each is decided before any held-out score is seen.`,
    [
      { label: `Triage the open noticings in ${where}`, decision: null },
      { label: "Keep the recommended dispositions", decision: { kind: "confirm_sweep", stage: t.stage, sweep: "noticings", lines: [] } },
    ],
  );
}

const LEDGER: Ledger = { version: 1, locked: false, gate: "", rows: [], limitations: 0, budget: 0 };

/** The reporting checklist: the mock holds no items, and says what the export waits for. */
export function mockChecklist(view: ProjectView): ChecklistReport {
  const predict = view.state.purpose === "prediction";
  return {
    checklist: predict ? "TRIPOD+AI" : "STROBE-nut",
    title: predict ? "TRIPOD+AI" : "STROBE-nut",
    citation: "",
    source: { table: "", pmcid: "", doi: "", url: "", retrieved: "", sha256: "", license: "" },
    purpose: view.state.purpose ?? null,
    items: [],
    counts: { items: 0, answered: 0, partly_answered: 0, unanswered: 0 },
    unanswered: [],
    note: "The mock server holds no checklist items.",
    waiting: exportWaits(view),
  };
}

function exportWaits(view: ProjectView): string[] {
  if (view.state.purpose === "prediction" && !view.state.seal_opened) return ["The held-out rows are not opened yet."];
  if (view.state.purpose === "inference" && !view.state.plan_locked) return ["The plan is not locked yet."];
  if (!view.state.purpose) return ["The goal is not chosen yet."];
  return [];
}

// ── the handlers ─────────────────────────────────────────────────────────────

/** Fit pressed, by project (the mock's; a real server keeps it with the project). */
const pressedFor = new Map<string, boolean>();

/** Whether Fit was pressed for a mock project (the fit stage's handler, m1-record.ts, reads it). */
export function fitPressed(pid: string): boolean {
  return pressedFor.get(pid) ?? false;
}

async function viewOf(pid: string): Promise<ProjectView | null> {
  // Through the mock's own handlers: each project kind (the mock server's, a replay, a lab's)
  // answers its view as it does for the page.
  const res = await fetch(`/api/projects/${encodeURIComponent(pid)}`);
  return res.ok ? ((await res.json()) as ProjectView) : null;
}

async function findingsOf(pid: string, view: ProjectView): Promise<Finding[] | null> {
  const st = view.stages.findings;
  if (!st || st.status === "idle" || st.status === "blocked") return null;
  const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/stages/findings`);
  if (!res.ok) return null;
  const result = (await res.json()) as StageResult<FindingsArtifact>;
  return result.artifact?.findings ?? null;
}

/** The roles stage's reason and confidence for each column's proposal, where its artifact is
 *  served. */
async function reasonsOf(pid: string, view: ProjectView): Promise<Proposed> {
  const st = view.stages.roles;
  if (!view.state.roles_unconfirmed?.length || !st || st.status === "idle" || st.status === "blocked") return {};
  const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/stages/roles`);
  if (!res.ok) return {};
  const result = (await res.json()) as StageResult<RolesArtifact>;
  return Object.fromEntries(
    (result.artifact?.columns ?? []).map((p) => [p.column, { reason: p.reason, confidence: p.confidence }]),
  );
}

async function questOf(pid: string): Promise<{ view: ProjectView; log: QuestLog } | null> {
  const view = await viewOf(pid);
  if (!view) return null;
  const log = mockQuest(view, await findingsOf(pid, view), pressedFor.get(pid) ?? false, await reasonsOf(pid, view));
  return { view, log };
}

const notFound = () =>
  HttpResponse.json({ error: { code: "not_found", message: "No such project.", exits: [] } }, { status: 404 });

export function questHandlers(server: MockServer): HttpHandler[] {
  const pidOf = (params: Record<string, unknown>) => String(params.pid);
  /** A read the mock server's own projects answer here; a replay keeps its capture's. */
  const own = (pid: string) => !!server.get(pid);
  return [
    http.get("/api/projects/:pid/quest", async ({ params }) => {
      const q = await questOf(pidOf(params));
      return q ? HttpResponse.json(q.log) : notFound();
    }),
    http.get("/api/projects/:pid/record", async ({ params }) => {
      const q = await questOf(pidOf(params));
      return q ? HttpResponse.json(mockRecord(q.log)) : notFound();
    }),
    http.get("/api/projects/:pid/triage", async ({ params }) => {
      const q = await questOf(pidOf(params));
      return q ? HttpResponse.json(mockTriage(q.view, q.log)) : notFound();
    }),
    // The opening's gates under prediction (openingRefusal); anything else, and an opening they let
    // through, goes on to the mock server's own decisions handler (m1-record.ts).
    http.post("/api/projects/:pid/decisions", async ({ params, request }) => {
      const pid = pidOf(params);
      if (!own(pid)) return undefined;
      const d = (await request.clone().json()) as { kind?: string };
      if (d.kind !== "open_seal") return undefined;
      const q = await questOf(pid);
      const refused = q ? openingRefusal(q.view, q.log, fitPressed(pid)) : null;
      return refused ? HttpResponse.json(refused, { status: 409 }) : undefined;
    }),
    http.get("/api/projects/:pid/materiality", () => HttpResponse.json(LEDGER)),
    http.get("/api/projects/:pid/checklist", async ({ params }) => {
      const view = await viewOf(pidOf(params));
      return view ? HttpResponse.json(mockChecklist(view)) : notFound();
    }),
    http.get("/api/projects/:pid/methods/sections", async ({ params }) => {
      const view = await viewOf(pidOf(params));
      return view ? HttpResponse.json(mockSections(view)) : notFound();
    }),
    http.get("/api/projects/:pid/methods", async ({ params }) => {
      const pid = pidOf(params);
      if (!own(pid)) return undefined; // a replay's methods are its capture's (m3.ts)
      const view = await viewOf(pid);
      return view ? HttpResponse.json(mockMethods(view)) : notFound();
    }),
    http.post("/api/projects/:pid/fit", async ({ params }) => {
      const pid = pidOf(params);
      const view = await viewOf(pid);
      if (!view) return notFound();
      // Under Estimate, pressing Fit locks the plan (the system's lock_plan record).
      if (view.state.purpose === "inference" && !view.state.plan_locked && own(pid))
        server.decide(pid, { kind: "lock_plan", seen_target: view.state.target ?? null } as never);
      pressedFor.set(pid, true);
      const after = (await viewOf(pid)) ?? view;
      return HttpResponse.json(lockOf(after, true));
    }),
    http.post("/api/projects/:pid/fit/cancel", async ({ params }) => {
      const pid = pidOf(params);
      const view = await viewOf(pid);
      if (!view) return notFound();
      // The mock's fit is always computed by the press, so Cancel only withdraws an unopened press.
      if (!view.state.seal_opened) pressedFor.set(pid, false);
      return HttpResponse.json(lockOf(view, pressedFor.get(pid) ?? false));
    }),
    http.get("/api/projects/:pid/export", async ({ params }) => {
      const view = await viewOf(pidOf(params));
      if (!view) return notFound();
      const waits = exportWaits(view);
      if (waits.length)
        return HttpResponse.json(
          { error: { code: "export_waits", message: `The bundle waits: ${waits.join(" ")}`, exits: [] } },
          { status: 409 },
        );
      return new HttpResponse(`TurboTab mock bundle for ${view.summary.name}\n`, {
        headers: {
          "Content-Type": "text/plain",
          "Content-Disposition": 'attachment; filename="turbotab-mock-export.txt"',
        },
      });
    }),
  ];
}
