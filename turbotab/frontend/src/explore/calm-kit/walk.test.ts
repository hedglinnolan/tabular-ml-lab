import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { fmtCI, fmtEst } from "../methods-shared/results";
import { FX, STEPS } from "./fixture";
import {
  ORDER,
  SCENARIO_ANSWERS,
  blockedBy,
  chain,
  digest,
  frontier,
  initial,
  manuscript,
  plan,
  reduce,
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
    expect(chain(s).map((c) => c.status)).toEqual(["current", "waiting", "waiting", "waiting", "waiting", "waiting", "waiting"]);
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
