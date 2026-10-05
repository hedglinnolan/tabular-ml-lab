/**
 * No step is ever blank (the presentation shell): every Router question key has a renderer, and
 * every one renders something answerable on the real server's captured views (src/mocks/fixtures,
 * docs/turbotab-next/m3/capture_fixtures.py).
 *
 * The keys are read from the Router itself (turbotab/core/interview.py), so a key the engine adds
 * without a bespoke component or a composer fails here, named.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api } from "../../../api/client";
import { keys } from "../../../api/queries";
import { QUESTION_KEYS, type InterviewStep, type QuestionKey, type TeachingEntry } from "../../../api/m1-types";
import type { Decision, ProjectView, TargetInfoArtifact } from "../../../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { Record } from "../Record";
import { BESPOKE_KEYS, COMPOSERS, compose, hasRenderer, taskFollowup } from "./compose";

const HERE = dirname(fileURLToPath(import.meta.url));
const INTERVIEW = resolve(HERE, "../../../../../core/interview.py");

const FIXTURES = import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-*.json", {
  eager: true,
  import: "default",
});
const SHARED = Object.entries(FIXTURES).find(([p]) => p.endsWith("m3-shared.json"))![1] as unknown as {
  teaching: TeachingEntry[];
};
const JOURNEYS = Object.entries(FIXTURES)
  .filter(([p]) => !p.endsWith("m3-shared.json"))
  .map(([p, f]) => ({ name: /m3-(.+)\.json$/.exec(p)![1]!, f, views: viewsOf(f) }));

/** The Router's keys, as interview.py declares them (QUESTION_KEYS). */
function routerKeys(): string[] {
  const text = readFileSync(INTERVIEW, "utf-8");
  const m = /^QUESTION_KEYS: tuple\[str, \.\.\.\] = \(([\s\S]*?)\n\)/m.exec(text);
  if (!m) throw new Error("interview.py no longer declares QUESTION_KEYS as this test reads it");
  return [...m[1]!.matchAll(/"([a-z_]+)"/g)].map((x) => x[1]!);
}

describe("every Router question key has a renderer", () => {
  it("mirrors interview.py's keys, in its order", () => {
    expect(routerKeys()).toEqual([...QUESTION_KEYS]);
  });

  it("gives each key a bespoke component or a composer, never both and never neither", () => {
    for (const key of routerKeys()) {
      expect(hasRenderer(key as QuestionKey), `${key} has no renderer: no step may be blank`).toBe(true);
      expect(BESPOKE_KEYS.includes(key as QuestionKey) && key in COMPOSERS, key).toBe(false);
    }
  });
});

// ── rendering each key on a captured view ───────────────────────────────────

const PID = "m3-test";

/** A view where `key` is the open question: as captured, else its latest captured state forced
 *  open (a skipped or answered question, as "Ask me anyway" or "change" would open it). */
function openViewFor(key: QuestionKey): { view: ProjectView; f: M3Fixture; journey: string } {
  for (const { name, f, views } of JOURNEYS) {
    const i = views.findIndex((v) => v.interview.some((s) => s.key === key && s.status === "open"));
    if (i !== -1) return { view: expandView(f, views[i]!, PID, "2026-10-05T00:00:00Z"), f, journey: name };
  }
  for (const { name, f, views } of JOURNEYS) {
    for (let i = views.length - 1; i >= 0; i--) {
      const v = views[i]!;
      const step = v.interview.find((s) => s.key === key);
      if (!step || step.status === "not_applicable" || step.status === "waiting") continue;
      const interview: InterviewStep[] = v.interview.map((s) =>
        s.key === key
          ? { ...s, status: "open", waiting_on: [] }
          : s.status === "open" || s.status === "waiting"
            ? { ...s, status: "waiting", waiting_on: [key] }
            : s,
      );
      return { view: expandView(f, { ...v, interview }, PID, "2026-10-05T00:00:00Z"), f, journey: name };
    }
  }
  throw new Error(`no captured journey reaches ${key}; capture one (capture_fixtures.py)`);
}

function renderRecord(view: ProjectView, f: M3Fixture) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.teaching(), SHARED.teaching);
  qc.setQueryData(keys.view(PID), view);
  const columns = f.endpoints.columns;
  if (columns?.status === 200) qc.setQueryData(keys.columns(PID), columns.body);
  for (const [stage, status] of Object.entries(view.stages)) {
    if (!status || status.status === "idle" || status.status === "blocked") continue;
    qc.setQueryData(keys.stage(PID, stage), {
      stage,
      key: status.key,
      fresh: status.status === "fresh",
      status: status.status,
      artifact: artifactAt(f, stage, status.key),
    });
  }
  return render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <Record pid={PID} view={view} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
}

const ANSWERABLE =
  '[role="option"]:not([aria-disabled="true"]), button:not([disabled]), select:not([disabled]), input:not([disabled]), [role="combobox"]';

describe("no step is ever blank", () => {
  it("captures enough journeys to reach every key", () => {
    expect(JOURNEYS.length).toBeGreaterThanOrEqual(8);
  });

  for (const key of QUESTION_KEYS) {
    it(`renders ${key} with something to answer`, () => {
      const { view, f, journey } = openViewFor(key);
      renderRecord(view, f);
      const block =
        key === "open_seal" ? screen.getByTestId("open-seal-step") : screen.getByTestId(`question-${key}`);
      expect(block.textContent!.trim().length, `${key} on ${journey}`).toBeGreaterThan(20);
      expect(block.querySelectorAll(ANSWERABLE).length, `${key} on ${journey}: nothing to press`).toBeGreaterThan(0);
      if (key in COMPOSERS) {
        const generic = within(block).getByTestId(`generic-${key}`);
        const waiting = within(generic).queryByTestId(`generic-waiting-${key}`);
        // Composed from the step's own card: options to record, or a matrix of answers.
        if (!waiting)
          expect(
            generic.querySelectorAll('[role="option"], [data-testid="generic-matrix"]').length,
            `${key} on ${journey}`,
          ).toBeGreaterThan(0);
      }
    });
  }
});

describe("a recorded answer can be taken back (INBOX 41)", () => {
  it("posts the engine's revert of the record the sentence shows; the seal's opening has none", async () => {
    const { view, f } = openViewFor("purpose");
    const decide = vi.spyOn(api, "decide").mockResolvedValue(view);
    renderRecord(view, f);
    fireEvent.click(screen.getByRole("button", { name: "Undo the lens" }));
    await waitFor(() => expect(decide).toHaveBeenCalled());
    const lens = view.interview.find((s) => s.key === "lens")!.decision_id;
    expect(decide.mock.calls[0]![1]).toEqual({ kind: "revert", decision_id: lens });
    decide.mockRestore();
  });

  it("offers no undo on the seal's opening, which happens once", () => {
    const j = JOURNEYS.find((x) => x.name === "clinical")!;
    const v = expandView(j.f, j.views.at(-1)!, PID, "2026-10-05T00:00:00Z");
    expect(v.interview.find((s) => s.key === "open_seal")!.status).toBe("answered");
    renderRecord(v, j.f);
    expect(screen.queryByRole("button", { name: "Undo opening the seal" })).toBeNull();
    expect(screen.getByRole("button", { name: "Undo the purpose" })).toBeInTheDocument();
  });
});

describe("the composers read the server's cards", () => {
  const at = (journey: string, key: QuestionKey) => {
    const j = JOURNEYS.find((x) => x.name === journey)!;
    const i = j.views.findIndex((v) => v.interview.some((s) => s.key === key && s.status === "open"));
    const v = j.views[i]!;
    const art = <T,>(stage: string) => artifactAt(j.f, stage, v.stages[stage]?.key ?? null) as T;
    return compose({
      key,
      step: v.interview.find((s) => s.key === key)!,
      state: v.state,
      entry: SHARED.teaching.find((e) => e.key === key),
      targetInfo: art("target_info"),
      roles: art("roles"),
      proposals: art("proposals"),
      causalDesign: art("causal_design"),
      timeVarying: art("time_varying"),
    });
  };

  it("offers the follow-up's two answers, each recording its own decision", () => {
    const q = at("clinical", "follow_up");
    const kinds = q.options.map((o) => o.build?.({})?.kind);
    expect(kinds).toEqual(["set_censoring", "set_task"]);
  });

  it("offers each exposure the estimand card lists, and never pre-selects a contrast", () => {
    const q = at("nhanes-inference", "estimand");
    expect(q.options.map((o) => o.key)).toContain("sugar");
    expect(q.fields.find((f) => f.name === "contrast")?.initial ?? []).toEqual([]);
    const d = q.options.find((o) => o.key === "sugar")!.build!({ effect: ["total"], measure: ["mean_difference"] });
    expect(d).toMatchObject({ kind: "set_estimand", exposure: "sugar", contrast: null, measure: "mean_difference" });
  });

  it("confirms the adjustment card's guessed groups with their own decisions, and asks the rest", () => {
    const q = at("nhanes-inference", "adjustment");
    expect(q.options.length).toBeGreaterThan(0);
    for (const o of q.options) expect(o.build!({}).kind).toBe("set_adjustment");
    expect(q.matrix?.rows.length ?? 0).toBeGreaterThan(0);
    const [row] = q.matrix!.rows;
    expect(q.matrix!.build({})).toBeNull();
    const d = q.matrix!.build({ [row!]: { causes_exposure: "no", causes_outcome: "yes", after_exposure: "yes" } });
    expect(d).toMatchObject({ kind: "set_adjustment", answers: { [row!]: { after_exposure: "yes" } } });
  });

  it("asks what the task still needs (the outcome's scale) with the target stage's own answers", () => {
    const j = JOURNEYS.find((x) => x.name === "nhanes-prediction")!;
    const v = j.views.at(-1)!;
    const ti = artifactAt(j.f, "target_info", v.stages.target_info!.key) as TargetInfoArtifact;
    const log = { kind: "set_outcome_scale", column: ti.column, scale: "log" } as Decision;
    const withScale = {
      ...ti,
      scale_question: {
        question: "Analyze the outcome on its original scale, or its log?",
        skewness: 3.1,
        log_skewness: 0.2,
        n: 100,
        min: 1,
        median: 5,
        max: 400,
        evidence: "skewed",
        log_column: `${ti.column}_ln`,
        options: [{ label: "On the log scale", decision: log }, { label: "Choose later", decision: null }],
      },
    };
    const step = { ...v.interview.find((s) => s.key === "task")!, status: "open" as const, followup: "scale" as const };
    const q = taskFollowup({ key: "task", step, state: v.state, targetInfo: withScale });
    expect(q?.options.map((o) => o.build?.({}))).toEqual([log]);
    expect(taskFollowup({ key: "task", step: { ...step, followup: null }, state: v.state, targetInfo: ti })).toBeNull();
  });

  it("lays out the time-varying lane's options with their two labels, the ordering never declared for the user", () => {
    const q = at("time-varying", "time_varying");
    expect(q.options.map((o) => o.key)).toEqual(expect.arrayContaining(["standard", "msm_iptw", "gformula"]));
    expect(q.options.every((o) => o.labels?.customary && o.labels.sound)).toBe(true);
    expect(q.options[0]!.build!({})).toMatchObject({ kind: "set_time_varying", ordering: "unknown" });
  });
});
