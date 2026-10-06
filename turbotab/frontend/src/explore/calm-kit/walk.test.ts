import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { fmtCI, fmtEst } from "../methods-shared/results";
import { FX, STEPS, STEP_BY_ID, type ConsequenceView } from "./fixture";
import { focusPreview, readoutFor, restOf } from "./canvas/Canvas";
import {
  ORDER,
  SCENARIO_ANSWERS,
  blockedBy,
  chain,
  digest,
  frontier,
  initial,
  manuscript,
  lockSentence,
  plan,
  reduce,
  refitFor,
  results,
  sentenceCount,
  stepLabel,
  type WalkState,
} from "./walk";

function walk(answers: [string, string][], s: WalkState = initial()): WalkState {
  for (const [step, option] of answers) s = reduce(s, { type: "record", step, option });
  return s;
}

const scenario = () => walk(SCENARIO_ANSWERS);

describe("the scenario walk", () => {
  it("starts at the first draft: kcal's unit open, nothing recorded, no estimate", () => {
    const s = initial();
    expect(s.open).toBe("unit");
    expect(results(s)).toBeNull();
    const sections = manuscript(s);
    expect(sections.flatMap((x) => x.entries).filter((e) => e.kind === "recorded")).toEqual([]);
    expect(sentenceCount(s)).toBe(FX.stated.length);
    expect(chain(s).map((c) => c.label)).toEqual(["Data", "Participants", "Columns", "Exposure", "Confounders", "Energy", "Model", "Results"]);
    expect(chain(s).map((c) => c.status)).toEqual(["current", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"]);
  });

  it("walks the scenario's moments in order: readings singly, then the block, to the lock", () => {
    const ids = STEPS.map((s) => s.id);
    expect(ids.slice(0, 8)).toEqual(["unit", "exclusions", "sensitivity", "missing", "single:bp_di", "single:bp_sys", "single:cycle_begin_year", "block"]);
    expect(ids.slice(-3)).toEqual(["model1", "codes", "lock"]);
    let s = initial();
    for (const [step, option] of SCENARIO_ANSWERS) {
      expect(s.open, `before ${step}`).toBe(step);
      s = reduce(s, { type: "record", step, option });
    }
    expect(s.open).toBe("table2");
    expect(reduce(s, { type: "next" }).open).toBe("mattered");
  });

  it("locks the scenario's plan with the engine's SHA-256 and serves the scenario's Table 2", () => {
    const s = scenario();
    expect(s.locked).toBe(true);
    expect(digest(s)).toBe("c9efee9fb0b2");
    const r = results(s)!;
    expect(r.table2.map((x) => x.key)).toEqual(["crude", "model_1", "model_2", "model_3"]);
    const m2 = r.table2.find((x) => x.primary)!;
    expect([fmtEst(m2.estimate), fmtCI(m2.lo, m2.hi)]).toEqual(["−0.0199", "−0.0327 to −0.00718"]);
    expect(r.footnote).toContain("t(21,830)");
    expect(r.mattered.map((x) => x.key)).toEqual(["crude", "model_1", "model_2", "model_3", "screen:Willett 2013, by sex", "screen:NHS/HPFS, by sex"]);
    // the checks carry the names the card offered them under, the technical name beside
    expect(r.mattered.slice(-2).map((x) => [x.label, x.term])).toEqual([
      ["Ranges by sex, men up to 4,000", "Willett's cut-offs (Willett 2013)"],
      ["Ranges by sex, men up to 4,200", "NHS/HPFS cut-offs (Nurses' Health and Health Professionals studies)"],
    ]);
    const lock = manuscript(s).flatMap((x) => x.entries).find((e) => e.id === "lock")!;
    expect(lock.sentence).toContain("c9efee9fb0b2");
  });

  it("holds the leash: no estimate exists anywhere before the lock", () => {
    let s = initial();
    const printed = Object.values(FX.fits).flatMap((f) => (f.error ? [] : f.sequence.map((r) => fmtEst(r.effects[0]!.estimate))));
    for (const [step, option] of SCENARIO_ANSWERS.slice(0, -1)) {
      s = reduce(s, { type: "record", step, option });
      expect(results(s)).toBeNull();
      const text = JSON.stringify(manuscript(s));
      for (const p of printed) expect(text).not.toContain(p);
    }
    // nor in any option's preview, nor in any question's data now
    for (const st of STEPS) {
      expect(JSON.stringify(st.now), st.id).not.toMatch(/coefficient [+−-]?\d/);
      for (const o of st.options) {
        const text = JSON.stringify(o.preview);
        expect(text, `${st.id}/${o.id}`).not.toMatch(/coefficient [+−-]?\d/);
      }
    }
  });

  it("revisits a recorded step from the chain, and returns to the frontier after a change", () => {
    let s = walk(SCENARIO_ANSWERS.slice(0, 5));
    expect(s.open).toBe("single:bp_sys");
    s = reduce(s, { type: "open", step: "exclusions" });
    expect(s.open).toBe("exclusions");
    expect(stepLabel("exclusions")).toBe("Participants · step 1 of 3");
    s = reduce(s, { type: "record", step: "exclusions", option: "willett_2013_by_sex" });
    expect(s.open).toBe("single:bp_sys");
    expect(s.order[s.order.length - 1]).toBe("exclusions");
    const newest = manuscript(s).flatMap((x) => x.entries).find((e) => e.newest)!;
    expect(newest.id).toBe("exclusions");
    // a step past the frontier cannot be opened
    expect(reduce(s, { type: "open", step: "energy" }).open).toBe("single:bp_sys");
  });

  it("says when an earlier answer leaves a step's captures behind, and refuses to record it", () => {
    let s = walk([["unit", "kj_1"]]);
    expect(blockedBy(s, "exclusions")).toEqual([{ step: "unit", got: "kj_1" }]);
    s = reduce(s, { type: "record", step: "exclusions", option: "none" });
    expect(s.answers.exclusions).toBeUndefined();
    // a disabled option is never recorded
    s = walk([["unit", "kcal_1"]]);
    expect(reduce(s, { type: "record", step: "exclusions", option: "goldberg_schofield" }).answers.exclusions).toBeUndefined();
  });

  it("locks only a plan whose fit the engine served, and names what differs otherwise", () => {
    const answers = SCENARIO_ANSWERS.filter(([id]) => id !== "lock").map(([id, o]): [string, string] => (id === "model1" ? [id, "empty"] : [id, o]));
    const s = walk(answers);
    const p = plan(s);
    expect(p.ok).toBe(false);
    if (!p.ok) expect(p.differs).toEqual([{ step: "model1", got: "empty", want: "guess" }]);
    expect(reduce(s, { type: "record", step: "lock", option: "lock" }).locked).toBe(false);
    // Willett's screen as the primary and the residual model: captured, and a different Table 2
    const other = walk(
      SCENARIO_ANSWERS.map(([id, o]): [string, string] => (id === "exclusions" ? [id, "willett_2013_by_sex"] : id === "energy" ? [id, "residual"] : [id, o])),
    );
    expect(other.locked).toBe(true);
    expect(results(other)!.table2.find((r) => r.primary)!.n).toBe(20235);
  });

  it("marks a change after the lock, and resets to the first draft", () => {
    let s = scenario();
    s = reduce(s, { type: "open", step: "energy" });
    s = reduce(s, { type: "record", step: "energy", option: "residual" });
    expect(s.afterLock).toEqual(["energy"]);
    expect(manuscript(s).flatMap((x) => x.entries).find((e) => e.id === "energy")!.afterLock).toBe(true);
    expect(results(s)!.fit.sequence[0]!.effects[0]!.feature).toBe("sugar_adj");
    s = reduce(s, { type: "reset" });
    expect(s).toEqual(initial());
  });

  it("keeps the fingerprint recorded at the lock when an answer changes after it", () => {
    let s = scenario();
    s = reduce(s, { type: "open", step: "exclusions" });
    s = reduce(s, { type: "record", step: "exclusions", option: "willett_2013_by_sex" });
    // the estimates follow the change, the record marks it, and the declared plan's SHA-256 stands
    expect(results(s)!.table2.find((r) => r.primary)!.n).toBe(20235);
    expect(s.afterLock).toEqual(["exclusions"]);
    expect(lockSentence(s)).toContain("c9efee9fb0b2");
    expect(lockSentence(s)).not.toContain(digest(s)!);
  });

  it("after the lock, records a change only with its estimates", () => {
    const s = scenario();
    // no fit was captured for an addition: the record and Table 2 would disagree, so it is refused
    expect(refitFor(s, "contrast", "addition")?.ok).toBe(false);
    const t = reduce(s, { type: "record", step: "contrast", option: "addition" });
    expect(t.answers.contrast).toBe("substitution");
    expect(t.afterLock).toEqual([]);
    // the density model's fit was captured: recorded, marked, and Table 2 follows it
    expect(refitFor(s, "energy", "density")?.ok).toBe(true);
    const u = reduce(s, { type: "record", step: "energy", option: "density" });
    expect(u.afterLock).toEqual(["energy"]);
    expect(results(u)!.table2[0]!.feature).toBe("sugar_per_kcal");
    expect(refitFor(initial(), "contrast", "addition")).toBeNull();
  });

  it("covers every moment the brief names, and ends with Table 2 and what mattered", () => {
    expect(ORDER.slice(-2)).toEqual(["table2", "mattered"]);
    for (const id of ["unit", "exclusions", "sensitivity", "missing", "block", "exposure", "effect", "contrast", "energy", "model1", "codes", "lock"])
      expect(ORDER).toContain(id);
    expect(ORDER.filter((id) => id.startsWith("single:"))).toHaveLength(3);
    expect(ORDER.filter((id) => id.startsWith("adjust:")).length).toBeGreaterThanOrEqual(4);
    expect(frontier(initial())).toBe("unit");
  });
});

describe("the kit's data", () => {
  it("imports the foundation's tokens verbatim", () => {
    const here = resolve(__dirname);
    const ours = readFileSync(resolve(here, "tokens.css"), "utf8");
    const theirs = readFileSync(resolve(here, "../../../../../docs/turbotab-next/calm/tokens.css"), "utf8");
    expect(ours).toBe(theirs);
  });

  it("never leaves the canvas empty: every question has its data now", () => {
    for (const s of STEPS) {
      const n = s.now;
      expect(n.views.length + (n.strip?.length ?? 0), s.id).toBeGreaterThan(0);
      expect(n.caption, s.id).toBeTruthy();
      for (const a of n.angles ?? []) expect(a.view !== undefined && n.views[a.view], `${s.id}: ${a.question}`).toBeTruthy();
      // the strip at rest is the columns as recorded: nothing in it changes
      for (const c of n.strip ?? []) expect([c.shift, c.output, c.sd_after], `${s.id}: ${c.column}`).toEqual([0, c.column, c.sd_before]);
    }
  });

  it("draws one 'your data now' per question: every option starts from the picture at rest", () => {
    const inputs = (l: { nodes: { lane: string; count?: number | null }[] }) => l.nodes.filter((n) => n.lane === "matrix").reduce((c, n) => c + (n.count || 1), 0);
    for (const s of STEPS) {
      const rest = restOf(s)!;
      const lin = rest.views.find((v) => v.kind === "lineage");
      const flow = rest.views.find((v) => v.kind === "row_flow");
      for (const o of s.options)
        for (const v of o.preview.views as ConsequenceView[]) {
          if (v.kind === "lineage" && lin?.kind === "lineage" && !o.preview.beside) {
            expect(v.before, `${s.id}/${o.id}: a before`).toBeTruthy();
            expect(inputs(v.before!), `${s.id}/${o.id}: inputs now`).toBe(inputs(lin.before ?? lin.after));
            // no storyboard frame draws fewer inputs than either end (the roles frame did: sugar alone)
            for (const f of v.story ?? []) expect(inputs(f.lineage), `${s.id}/${o.id}: ${f.label}`).toBeGreaterThanOrEqual(Math.min(inputs(v.before!), inputs(v.after)));
          }
          if (v.kind === "row_flow" && flow?.kind === "row_flow")
            expect(v.before[v.before.length - 1]!.n, `${s.id}/${o.id}: people now`).toBe(flow.before[flow.before.length - 1]!.n);
        }
    }
  });

  it("reads the people and the inputs as each question finds them", () => {
    const opt = (step: string, id: string) => STEP_BY_ID[step]!.options.find((o) => o.id === id)!;
    const line = (step: string, id: string, focus: string | null = null) =>
      readoutFor(opt(step, id), { after: true, focus }).map((r) => `${r.label} ${r.now}${r.after ? ` → ${r.after}` : ""}`);
    expect(line("block", "confirm")).toEqual(["People 21,849 → 2,996", "Model inputs 14 → 21"]);
    // a check reported beside removes no one from the main analysis
    expect(line("sensitivity", "both")).toEqual(["People in the main analysis 21,849", "In the checks 20,235 and 20,430"]);
    expect(line("model1", "guess")).toEqual(["Model 1's inputs 1 → 4"]);
    expect(line("model1", "empty")).toEqual([]);
    // an addition swaps four inputs for four, counted as the question finds the plan
    const lane = (id: string) => (opt("contrast", id).preview.views[0] as Extract<ConsequenceView, { kind: "lineage" }>).after.nodes.filter((n) => n.lane === "matrix").length;
    expect([lane("substitution"), lane("addition")]).toEqual([11, 11]);
    // the Strip's focus moves every view: the caption, the readout and the scatter follow it
    expect(line("energy", "residual")).toEqual(["Correlation of fat_total with kcal 0.88 → 0.00", "Model inputs 11"]);
    expect(line("energy", "residual", "sugar")).toEqual(["Correlation of sugar with kcal 0.66 → 0.00", "Model inputs 11"]);
    const pv = focusPreview(opt("energy", "residual").preview, "sugar");
    expect(pv.caption).toContain("`sugar` correlates");
    expect((pv.views as ConsequenceView[]).map((v) => (v.kind === "relationship" ? v.y_label_before : v.kind))).toEqual(["sugar", "lineage", "distribution"]);
  });

  it("gives each Angles panel one question and one picture, and the option table at most three short columns", () => {
    const words = (s: string) => s.split(/\s+/).filter(Boolean).length;
    for (const s of STEPS)
      for (const o of s.options) {
        const angles = o.preview.angles ?? [];
        if (!angles.length) continue;
        // the option's name and two answers
        expect(angles.length, `${s.id}/${o.id}`).toBeLessThanOrEqual(2);
        for (const a of angles) {
          expect(Number(a.view !== undefined) + Number(!!a.list), `${s.id}/${o.id}: ${a.question}`).toBe(1);
          expect(words(a.head), a.head).toBeLessThanOrEqual(2);
          expect(words(a.cell ?? ""), a.cell).toBeLessThanOrEqual(4);
        }
      }
  });

  it("speaks plainly on the card and the canvas: no internal reference, and an option not available says why", () => {
    const internal = /NUTRITION_PACK|the engine|engine's|predictor|estimand|the lock\b|range check|\bscreen\b/i;
    for (const s of STEPS) {
      for (const t of [s.question, s.lede, s.now.caption, s.now.basis]) expect(t, s.id).not.toMatch(internal);
      for (const o of s.options) {
        for (const t of [o.name, o.what, o.preview.caption ?? "", o.preview.basis]) expect(t, `${s.id}/${o.id}`).not.toMatch(internal);
        if (o.disabled) expect(o.what, `${s.id}/${o.id}`).toMatch(/^(Not possible|Needs)/);
      }
    }
  });

  it("keeps every card within the calm budget's word limits", () => {
    const words = (s: string) => s.split(/\s+/).filter(Boolean).length;
    for (const s of STEPS) {
      expect(words(s.question), s.id).toBeLessThanOrEqual(14);
      expect(words(s.lede), s.id).toBeLessThanOrEqual(22);
      expect(words(s.why), s.id).toBeLessThanOrEqual(60);
      for (const o of s.options) {
        expect(words(o.what), `${s.id}/${o.id}`).toBeLessThanOrEqual(16);
        expect(o.label === undefined || ["Recommended", "Common practice", "Not available yet", "Not available"].includes(o.label)).toBe(true);
      }
    }
  });
});
