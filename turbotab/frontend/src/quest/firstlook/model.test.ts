/**
 * First look's guide and gallery as data (model.ts), over what the server served at each of the
 * outcome's gates (fixtures/served.json, captured by fixtures/capture.py from the real server on the
 * clinical sample): every word and number expected here is read from that payload, never retyped.
 * Each gate is checked both ways: a view opens only where the server opened it, and an outcome view
 * is drawn only once its look is recorded, though Explore serves its points before then.
 */
import type { components } from "../../api/generated";
import {
  type ExploreArtifact,
  barsOf,
  beats,
  columnLooks,
  curveOf,
  decideNow,
  exploreBeat,
  exploreLine,
  exploreLooks,
  gateMark,
  listedColumns,
  outcomeDoor,
  recordedLook,
  rowsDoor,
  tableMark,
} from "./model";
import { DRAWN, ESTIMATE, FIRST, OUTCOME, RECORDED, type Moment, artifactOf, histogramOf, stageName, summaryOf } from "./fixtures/served";

// The served artifact reads as the gallery's ExploreArtifact (the contract's type is not in src/api's
// stage map yet): a field the contract drops or retypes fails to compile here.
type Served = components["schemas"]["ExploreArtifact"];
const asGallery = (a: Served): ExploreArtifact => a;
const asServed = (a: ExploreArtifact): Pick<Served, keyof ExploreArtifact> => a;
void asGallery;
void asServed;

describe("every column but the outcome (columnLooks)", () => {
  const looks = columnLooks(FIRST.columns.body, FIRST.view.state);

  it("is each served column but the outcome, in the served order", () => {
    const names = FIRST.columns.body.map((c) => c.name).filter((n) => n !== OUTCOME);
    expect(looks.map((l) => l.name)).toEqual(names);
    expect(looks.map((l) => l.name)).not.toContain(OUTCOME);
  });

  it("draws a number as a histogram, a category as its levels, and an identifier, a date or text as one line", () => {
    // Before the roles are recorded (the first visit) the identifier is text; after, an identifier.
    for (const m of [FIRST, DRAWN]) {
      const roles = m.view.state.roles ?? {};
      for (const look of columnLooks(m.columns.body, m.view.state)) {
        const c = m.columns.body.find((x) => x.name === look.name)!;
        expect(look.blank).toBe(c.n_missing);
        expect(look.n).toBe(c.n);
        const distinct = c.n_unique.toLocaleString("en-US");
        if (roles[c.name] === "identifier") {
          expect(look.draw).toEqual({ kind: "line", line: `An identifier: ${distinct} distinct values` });
        } else if (c.dtype === "numeric" || c.dtype === "integer") {
          expect(look.draw).toEqual({ kind: "histogram" });
        } else if (c.dtype === "categorical" || c.dtype === "boolean") {
          expect(look.draw).toEqual({
            kind: "levels",
            levels: c.top!.map((t) => ({ label: String(t.value), count: t.count })),
            more: c.n_unique - c.top!.length,
          });
        } else {
          expect(look.draw).toEqual({ kind: "line", line: `${distinct} distinct values` });
        }
      }
    }
    // The sample holds each kind the rule tells apart, and the identifier both ways.
    const kinds = (m: Moment) => columnLooks(m.columns.body, m.view.state).map((l) => l.draw.kind);
    for (const kind of ["line", "histogram", "levels"]) expect(kinds(DRAWN)).toContain(kind);
    const id = Object.entries(DRAWN.view.state.roles!).find(([, r]) => r === "identifier")![0];
    const lineOf = (m: Moment) => columnLooks(m.columns.body, m.view.state).find((l) => l.name === id)!.draw;
    expect(lineOf(DRAWN)).toMatchObject({ kind: "line", line: expect.stringMatching(/^An identifier: /) });
    expect(lineOf(FIRST)).toMatchObject({ kind: "line", line: expect.not.stringMatching(/^An identifier/) });
  });
});

describe("the outcome alone (outcomeDoor), as GET /columns serves its summary", () => {
  it("is shut, in the server's words, before its gate: nothing to post", () => {
    const s = summaryOf(FIRST);
    expect(s.withheld).toBeTruthy();
    expect(s.record).toBeNull();
    expect(outcomeDoor(s)).toEqual({ kind: "shut", line: s.withheld });
  });

  it("hands back the record to post once its gate is open and the look is not recorded", () => {
    const s = summaryOf(DRAWN);
    expect(s.record).not.toBeNull();
    expect(outcomeDoor(s)).toEqual({ kind: "record", line: s.withheld, record: s.record });
  });

  it("is open on the rows it reads once recorded", () => {
    const s = summaryOf(RECORDED);
    expect(s.withheld).toBeNull();
    expect(outcomeDoor(s)).toEqual({ kind: "open", rows: s.rows });
    expect(s.rows).toBe("training");
  });

  it("stays shut where a summary says neither when it opens nor which rows it read", () => {
    // By construction: the open summary with its rows taken away (a server before the gate, a mock).
    const s = { ...summaryOf(RECORDED), rows: null };
    expect(outcomeDoor(s)).toEqual({ kind: "shut", line: "The server has not said that the outcome is open." });
  });
});

describe("the outcome beside the other columns (rowsDoor), as GET /table serves a window", () => {
  it("is shut, with the window's line for it, before its gate", () => {
    for (const m of [FIRST, ESTIMATE]) {
      const w = m.table.body;
      expect(w.columns).not.toContain(OUTCOME);
      expect(w.record).toBeNull();
      expect(rowsDoor(w, OUTCOME)).toEqual({ kind: "shut", line: w.withheld[OUTCOME] });
    }
    // Under Estimate it waits for the lock while the outcome alone is already open.
    expect(rowsDoor(ESTIMATE.table.body, OUTCOME).kind).toBe("shut");
    expect(outcomeDoor(summaryOf(ESTIMATE)).kind).toBe("record");
  });

  it("hands back the window's record once open and not recorded", () => {
    const w = DRAWN.table.body;
    expect(rowsDoor(w, OUTCOME)).toEqual({ kind: "record", line: w.withheld[OUTCOME], record: w.record });
  });

  it("is open once the window holds the outcome", () => {
    const w = RECORDED.table.body;
    expect(w.columns).toContain(OUTCOME);
    expect(w.withheld[OUTCOME]).toBeUndefined();
    expect(w.labels[OUTCOME]).toBeUndefined();
    expect(rowsDoor(w, OUTCOME)).toEqual({ kind: "open", notes: [] });
  });

  it("keeps both of the server's lines on an open window: the rows left blank and its label", () => {
    // By construction, in the served shape: under Estimate after the lock the service serves the
    // outcome with both a line on the rows it leaves blank and the exploratory label (service.py
    // `table_window`); each is said as served, the blank rows' line first.
    const w = { ...RECORDED.table.body, withheld: { [OUTCOME]: "Blank rows' line, as served" }, labels: { [OUTCOME]: "The label, as served" } };
    expect(rowsDoor(w, OUTCOME)).toEqual({ kind: "open", notes: ["Blank rows' line, as served", "The label, as served"] });
    expect(rowsDoor({ ...w, withheld: {} }, OUTCOME)).toEqual({ kind: "open", notes: ["The label, as served"] });
    expect(rowsDoor({ ...w, labels: {} }, OUTCOME)).toEqual({ kind: "open", notes: ["Blank rows' line, as served"] });
  });
});

describe("Explore's looks (exploreLooks)", () => {
  it("draws no relationship before its look is recorded, though Explore serves its points", () => {
    const a = artifactOf(DRAWN);
    const looks = exploreLooks(a, DRAWN.view.state);
    const served = a.findings.filter((f) => f.kind === "outcome_relationship");
    expect(served.length).toBeGreaterThan(1);
    expect(served.every((f) => f.points.length > 0 && !f.viewed)).toBe(true);
    expect(looks.relationships.map((l) => l.finding.id)).toEqual(served.map((f) => f.id));
    for (const l of looks.relationships) expect(l.state).toEqual({ kind: "record", record: l.finding.record });
  });

  it("draws a relationship once it is recorded, and only that one", () => {
    const looks = exploreLooks(artifactOf(RECORDED), RECORDED.view.state);
    const viewed = artifactOf(RECORDED).viewed.filter((k) => k.startsWith("relationship:"));
    expect(viewed).toHaveLength(1);
    const drawn = looks.relationships.filter((l) => l.state.kind === "drawn").map((l) => `relationship:${l.finding.columns[0]}`);
    expect(drawn).toEqual(viewed);
    expect(looks.relationships.filter((l) => l.state.kind === "record")).toHaveLength(looks.relationships.length - 1);
  });

  it("reads a look as recorded from the folded state, before Explore computes again", () => {
    // The artifact as computed before the record, the state after it: the state is the log's.
    const looks = exploreLooks(artifactOf(DRAWN), RECORDED.view.state);
    const keys = Object.keys(RECORDED.view.state.outcome_views!).filter((k) => k.startsWith("relationship:"));
    expect(looks.relationships.filter((l) => l.state.kind === "drawn").map((l) => `relationship:${l.finding.columns[0]}`)).toEqual(keys);
    // A look at another outcome is not a look at this one (explore.py `_viewed`).
    const other = Object.fromEntries(
      Object.entries(RECORDED.view.state.outcome_views!).map(([k, v]) => [k, { ...v, target: `${OUTCOME}_before` }]),
    );
    const elsewhere = { ...RECORDED.view.state, outcome_views: other };
    expect(keys.every((k) => !recordedLook(elsewhere, k))).toBe(true);
    expect(keys.every((k) => recordedLook(RECORDED.view.state, k))).toBe(true);
  });

  it("says why a relationship waits, and offers nothing to post, where the server withholds it", () => {
    const a = artifactOf(ESTIMATE);
    const looks = exploreLooks(a, ESTIMATE.view.state);
    expect(looks.relationships.length).toBeGreaterThan(1);
    for (const l of looks.relationships) {
      expect(l.finding.record).toBeNull();
      expect(l.finding.points).toEqual([]);
      expect(l.state).toEqual({ kind: "waits", line: l.finding.detail });
    }
  });

  it("keeps the outcome's distribution for its door, and every other finding as an outcome-free look", () => {
    const a = artifactOf(DRAWN);
    const looks = exploreLooks(a, DRAWN.view.state);
    expect(looks.distribution?.kind).toBe("outcome_distribution");
    const free = a.findings.filter((f) => !f.outcome_view);
    expect(looks.free.map((l) => l.finding.id)).toEqual(free.map((f) => f.id));
    expect(free.length).toBeGreaterThan(0);
  });

  it("offers Decide now only with the lever's options that record, and none where no option does", () => {
    for (const f of artifactOf(DRAWN).findings) {
      expect(decideNow(f)).toEqual((f.lever?.options ?? []).filter((o) => o.decision !== null));
    }
    const sex = artifactOf(DRAWN).findings.find((f) => f.id === "explore::quality::sex")!;
    expect(decideNow(sex).length).toBeGreaterThan(0);
    // By construction: the same lever with no option that records.
    const told = { ...sex, lever: { ...sex.lever!, options: sex.lever!.options.map((o) => ({ ...o, decision: null })) } };
    expect(decideNow(told)).toEqual([]);
    expect(decideNow({ ...sex, lever: null })).toEqual([]);
  });

  it("names the near-constant predictors served, and how many more the summary counts", () => {
    // By construction, in the served shape (explore.py: the summary counts every near-constant
    // predictor, its columns the first of them). 15 counted, 12 named: 3 more.
    const named = Array.from({ length: 12 }, (_, i) => `flat_${i + 1}`);
    const flat = {
      ...artifactOf(DRAWN).findings.find((f) => !f.outcome_view)!,
      id: "explore::low_variance",
      kind: "low_variance",
      summary: "15 predictors barely vary on the training rows",
      columns: named,
      groups: [],
      view: "table_focus",
    };
    expect(listedColumns(flat)).toEqual({ names: named, more: 3 });
    // Every one named: none more. A count past a thousand reads with its comma.
    expect(listedColumns({ ...flat, summary: "2 predictors barely vary on the training rows", columns: ["a", "b"] })).toEqual({
      names: ["a", "b"],
      more: 0,
    });
    expect(listedColumns({ ...flat, summary: "1,012 predictors barely vary on the training rows" })).toEqual({ names: named, more: 1000 });
    // A summary whose count cannot be read claims nothing more.
    expect(listedColumns({ ...flat, summary: "Some predictors barely vary" })).toEqual({ names: named, more: 0 });
    // Silent on a look that is not about near-constant predictors.
    for (const f of artifactOf(DRAWN).findings) expect(listedColumns(f)).toBeNull();
  });
});

describe("what Explore's beat says (exploreBeat), from its status, its result and its read", () => {
  const whosIn = stageName(FIRST, "whos_in");
  const fresh = (m: Moment) => {
    const artifact = m.explore.body.artifact ?? null;
    return { artifact, looks: artifact ? exploreLooks(artifact, m.view.state) : null };
  };
  const at = (m: Moment, change: Partial<Parameters<typeof exploreBeat>[0]> = {}) =>
    exploreBeat({
      target: m.view.state.target ?? null,
      status: m.view.stages.explore,
      result: m.explore.body,
      failed: null,
      whosIn,
      ...fresh(m),
      ...change,
    });

  it("counts the looks and those that can be decided only while what it found is current", () => {
    const looks = exploreLooks(artifactOf(RECORDED), RECORDED.view.state);
    const artifact = artifactOf(RECORDED);
    expect(at(RECORDED)).toEqual({
      kind: "fresh",
      looks: looks.free.length,
      decidable: looks.free.filter((l) => l.options.length).length,
      rows: artifact.rows,
      n_rows: artifact.n_rows,
      n_holdout: artifact.n_holdout,
    });
    // By construction: an answer it reads changed (the server marks it stale, then reads again).
    const stale = { ...RECORDED.view.stages.explore!, status: "stale" as const, fresh: false, key: "a key after the change" };
    for (const status of [stale, { ...stale, status: "queued" as const }, { ...stale, status: "running" as const }]) {
      expect(at(RECORDED, { status })).toEqual({ kind: "running", again: true });
    }
    expect(exploreLine({ kind: "running", again: true })).toBe("Explore is reading the rows again; nothing it found can be decided until it has.");
  });

  it("is reading what Explore found, not waiting for it, while the stage is fresh and its result not yet read", () => {
    const status = DRAWN.view.stages.explore!;
    expect(status.status).toBe("fresh");
    const unread = { result: undefined, artifact: null, looks: null };
    expect(at(DRAWN, unread)).toEqual({ kind: "reading" });
    expect(exploreLine({ kind: "reading" })).toBe("Reading what Explore found…");
    // A newer result on its way: the one on screen is for an earlier key.
    expect(at(DRAWN, { status: { ...status, key: "a newer key" } })).toEqual({ kind: "reading" });
  });

  it("says the read failed, with the reason, where it did", () => {
    expect(at(DRAWN, { result: undefined, artifact: null, looks: null, failed: "The server answered 500." })).toEqual({
      kind: "unread",
      message: "The server answered 500.",
    });
    expect(exploreLine({ kind: "unread", message: "The server answered 500." })).toBe("What Explore found was not read: The server answered 500.");
  });

  it("waits for Who's in before Explore has started, and says an error or a stop over what it found before", () => {
    expect(at(FIRST, { result: undefined, artifact: null, looks: null })).toEqual({ kind: "waits", stage: whosIn });
    const status = RECORDED.view.stages.explore!;
    expect(at(RECORDED, { status: { ...status, status: "error", fresh: false, error: "It stopped." } })).toEqual({ kind: "error", error: "It stopped." });
    expect(at(RECORDED, { status: { ...status, status: "stale", fresh: false, key: "k", cancelled: true } })).toEqual({ kind: "stopped" });
    expect(exploreLine({ kind: "stopped" })).toBe("Explore was stopped before it finished reading the rows.");
    expect(at({ ...RECORDED, view: { ...RECORDED.view, state: { ...RECORDED.view.state, target: null } } })).toEqual({ kind: "none" });
  });
});

describe("the card's beats", () => {
  const whosIn = stageName(FIRST, "whos_in");

  it("count the columns drawn and say each gate in the server's words, before Who's in", () => {
    const columns = FIRST.columns.body.length - 1;
    const said = beats({
      columns,
      drawn: columns,
      noticed: 0,
      target: OUTCOME,
      door: outcomeDoor(summaryOf(FIRST)),
      rows: rowsDoor(FIRST.table.body, OUTCOME),
      relationships: [],
      explore: { kind: "waits", stage: whosIn },
    });
    expect(said.map((b) => b.key)).toEqual(["columns", "explore", "outcome"]);
    expect(said[0]!.line).toBe(`${columns.toLocaleString("en-US")} columns, each drawn as it is on every row of the table.`);
    expect(said[1]!.line).toBe(`Explore has not run yet: it reads the rows ${whosIn} keeps.`);
    expect(said[2]!.title).toBe(`The outcome, ${OUTCOME}`);
    expect(said[2]!.line).toBe(summaryOf(FIRST).withheld);
    // The rows wait at the same gate, in the same words: said once.
    expect(FIRST.table.body.withheld[OUTCOME]).toBe(summaryOf(FIRST).withheld);
    expect(said[2]!.also).toEqual([]);
  });

  it("say, under Estimate with the distribution recorded, that its distribution is open and what still waits, in the server's words", () => {
    // By construction from the Estimate moment: the outcome's summary as the server serves it once
    // its distribution is recorded (no line, no record, on the rows analyzed); the window and the
    // relationships as served then, both waiting for Fit.
    const opened = { ...summaryOf(ESTIMATE), withheld: null, record: null, rows: "analyzed" as const };
    const rows = rowsDoor(ESTIMATE.table.body, OUTCOME);
    const { relationships } = exploreLooks(artifactOf(ESTIMATE), ESTIMATE.view.state);
    expect(rows.kind).toBe("shut");
    expect(relationships.every((l) => l.state.kind === "waits")).toBe(true);
    const input = { columns: 3, drawn: 3, noticed: 0, target: OUTCOME, door: outcomeDoor(opened), explore: { kind: "none" as const } };
    const outcome = (b: ReturnType<typeof beats>) => b.find((x) => x.key === "outcome")!;
    const said = outcome(beats({ ...input, rows, relationships }));
    expect(said.line).toBe("Its distribution is open on the rows analyzed; each view of the outcome is recorded as looked at.");
    // The window's gate is the outcome beside another column, the relationships' too: its line once.
    expect(said.also).toEqual([ESTIMATE.table.body.withheld[OUTCOME]]);
  });

  /** Explore's beat as the card reads it at a moment, with its status changed where given. */
  const beatAt = (m: Moment, artifact: ExploreArtifact, status = m.view.stages.explore) =>
    exploreBeat({
      target: OUTCOME,
      status,
      result: { ...m.explore.body, artifact },
      failed: null,
      artifact,
      looks: exploreLooks(artifact, m.view.state),
      whosIn,
    });
  const outcomeOf = (b: ReturnType<typeof beats>) => b.find((x) => x.key === "outcome")!;

  it("say nothing more under Estimate while the window is on its way or was not read, though the relationships wait", () => {
    // Under Estimate the relationships wait with their own line, served while Explore is current.
    const a = artifactOf(ESTIMATE);
    const { relationships } = exploreLooks(a, ESTIMATE.view.state);
    expect(relationships.length).toBeGreaterThan(0);
    expect(relationships.every((l) => l.state.kind === "waits" && !!l.state.line)).toBe(true);
    const explore = beatAt(ESTIMATE, a);
    expect(explore.kind).toBe("fresh");
    const input = { columns: 3, drawn: 3, noticed: 0, target: OUTCOME, door: outcomeDoor(summaryOf(ESTIMATE)), relationships, explore };
    // The window not read (on its way, or its read failed): nothing is said for it.
    expect(outcomeOf(beats({ ...input, rows: null })).also).toEqual([]);
    // Both ways: once read and shut, the window's own line.
    expect(outcomeOf(beats({ ...input, rows: rowsDoor(ESTIMATE.table.body, OUTCOME) })).also).toEqual([ESTIMATE.table.body.withheld[OUTCOME]]);
  });

  it("say nothing more under Predict while Explore reads the rows again, and a relationship's own line only once it is current", () => {
    // By construction, in the served shape: while Explore is not fresh under Predict the service
    // serves its last result with each relationship's points withheld and a line in their place
    // (service.py `_outcome_served`, fit_press.withhold_relationships), though the rows are drawn.
    const line = "The line served in the relationships' place";
    const withheld: ExploreArtifact = {
      ...artifactOf(RECORDED),
      findings: artifactOf(RECORDED).findings.map((f) => (f.kind === "outcome_relationship" ? { ...f, points: [], detail: line, record: null } : f)),
    };
    const { relationships } = exploreLooks(withheld, RECORDED.view.state);
    expect(relationships.length).toBeGreaterThan(0);
    expect(relationships.every((l) => l.state.kind === "waits")).toBe(true);
    const rows = rowsDoor(RECORDED.table.body, OUTCOME);
    expect(rows.kind).toBe("open");
    const input = { columns: 3, drawn: 3, noticed: 0, target: OUTCOME, door: outcomeDoor(summaryOf(RECORDED)), rows, relationships };
    const reading = beatAt(RECORDED, withheld, { ...RECORDED.view.stages.explore!, status: "running", fresh: false, key: "a key after the change" });
    expect(reading).toEqual({ kind: "running", again: true });
    expect(outcomeOf(beats({ ...input, explore: reading })).also).toEqual([]);
    // Both ways: with what Explore found current and the window open, a relationship that waits
    // says its own line once (the shape is constructed: today's server serves every relationship's
    // points then); with the window not read, nothing.
    const current = beatAt(RECORDED, withheld);
    expect(current.kind).toBe("fresh");
    expect(outcomeOf(beats({ ...input, explore: current })).also).toEqual([line]);
    expect(outcomeOf(beats({ ...input, rows: null, explore: current })).also).toEqual([]);
    // As served once current, none waits: nothing more is said.
    const recorded = exploreLooks(artifactOf(RECORDED), RECORDED.view.state).relationships;
    expect(recorded.some((l) => l.state.kind === "waits")).toBe(false);
    expect(outcomeOf(beats({ ...input, relationships: recorded, explore: beatAt(RECORDED, artifactOf(RECORDED)) })).also).toEqual([]);
  });

  it("say what Explore read and what can be decided now, once it has run", () => {
    const a = artifactOf(DRAWN);
    const looks = exploreLooks(a, DRAWN.view.state);
    const decidable = looks.free.filter((l) => l.options.length).length;
    const said = beats({
      columns: 10,
      drawn: 6,
      noticed: 2,
      target: OUTCOME,
      door: outcomeDoor(summaryOf(RECORDED)),
      rows: rowsDoor(RECORDED.table.body, OUTCOME),
      relationships: exploreLooks(artifactOf(RECORDED), RECORDED.view.state).relationships,
      explore: { kind: "fresh", looks: looks.free.length, decidable, rows: a.rows, n_rows: a.n_rows, n_holdout: a.n_holdout },
    });
    expect(said.map((b) => b.key)).toEqual(["columns", "noticed", "explore", "outcome"]);
    // The count and the cap in one sentence: only the first are drawn.
    expect(said[0]!.line).toBe("10 columns; the first 6 are drawn here, each as it is on every row of the table.");
    expect(said[1]!.line).toBe("2 noticings, each decided in the stage whose answer it changes, listed below.");
    expect(said[2]!.line).toBe(
      `${looks.free.length} looks on the ${a.n_rows.toLocaleString("en-US")} ${a.rows} rows; the ${a.n_holdout.toLocaleString("en-US")} held-out rows are never read. ${decidable} can be decided now.`,
    );
    expect(said[3]!.line).toBe("Its distribution is open on the training rows; each view of the outcome is recorded as looked at.");
    expect(said[3]!.also).toEqual([]);
  });

  it("leave the outcome out when none is chosen", () => {
    const said = beats({ columns: 3, drawn: 3, noticed: 1, target: null, door: null, rows: null, relationships: [], explore: { kind: "none" } });
    expect(said.map((b) => b.key)).toEqual(["columns", "noticed"]);
    expect(said[1]!.line).toBe("1 noticing, each decided in the stage whose answer it changes, listed below.");
  });
});

describe("what the gallery reads again, and when", () => {
  const view = RECORDED.view;
  const log = RECORDED.quest;

  it("reads the outcome's gates again after a record, a stage's change or Fit's press, and not on a progress tick", () => {
    const at = gateMark(view, log);
    const last = view.decisions[view.decisions.length - 1]!;
    expect(gateMark({ ...view, decisions: [...view.decisions, { ...last, id: "x", seq: last.seq + 1 }] }, log)).not.toBe(at);
    const split = view.stages.split!;
    expect(gateMark({ ...view, stages: { ...view.stages, split: { ...split, status: "running" } } }, log)).not.toBe(at);
    expect(gateMark(view, { ...log, fit: { ...log.fit!, pressed: true } })).not.toBe(at);
    expect(gateMark({ ...view, stages: { ...view.stages, split: { ...split, progress: 0.5 } } }, log)).toBe(at);
  });

  it("reads each column's histogram again only when the table the routes read changes", () => {
    const at = tableMark(view);
    const working = view.stages.working!;
    expect(tableMark({ ...view, stages: { ...view.stages, working: { ...working, key: "another" } } })).not.toBe(at);
    expect(tableMark({ ...view, stages: { ...view.stages, split: { ...view.stages.split!, key: "another" } } })).toBe(at);
  });
});

describe("the drawings' scale math", () => {
  const box = { width: 240, height: 80, top: 4, right: 4, bottom: 18, left: 4 };

  it("draws each bin over its edges with a height in proportion to its count", () => {
    const h = histogramOf(DRAWN, "age");
    const bars = barsOf(h, box);
    expect(bars.map((b) => b.count)).toEqual(h.counts);
    const top = Math.max(...h.counts);
    const tallest = Math.max(...bars.map((b) => b.h));
    expect(tallest).toBeCloseTo(box.height - box.top - box.bottom);
    for (const b of bars) expect(b.h / tallest).toBeCloseTo(b.count / top, 9);
    for (let i = 1; i < bars.length; i++) expect(bars[i]!.x).toBeGreaterThan(bars[i - 1]!.x);
    expect(bars[0]!.x).toBeCloseTo(box.left);
    expect(barsOf({ ...h, edges: [], counts: [] }, box)).toEqual([]);
  });

  it("places each tenth's point by its value, higher outcomes higher, and skips one with no value", () => {
    const age = artifactOf(RECORDED).findings.find((f) => f.id === "explore::relationship::age")!;
    const pts = curveOf(age.points, box);
    expect(pts.map((p) => p.n)).toEqual(age.points.map((p) => p.n));
    for (let i = 1; i < pts.length; i++) {
      expect(pts[i]!.cx).toBeGreaterThan(pts[i - 1]!.cx);
      const [a, b] = [age.points[i - 1]!.y!, age.points[i]!.y!];
      if (a !== b) expect(pts[i]!.cy < pts[i - 1]!.cy).toBe(b > a);
    }
    const holed = age.points.map((p, i) => (i === 3 ? { ...p, y: null } : p));
    expect(curveOf(holed, box).map((p) => p.n)).toEqual(age.points.filter((_, i) => i !== 3).map((p) => p.n));
  });
});
