/**
 * First look's guide and gallery in the quest shell (G06; the owner's Q9: a robust gallery), driven
 * against what the real server served at each of the outcome's gates (fixtures/served.json, from
 * fixtures/capture.py on the clinical sample): every read is answered as the server answered it, and
 * a press is answered as the server answered that press. Every word and number expected is read
 * from those answers. Each gate is checked both ways: an outcome view opens only where the server
 * opened it, by its own press, which posts exactly the record the server handed back, and is drawn
 * only once recorded; "Decide now" is offered only where the look's lever records, and records with
 * decide_now.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { keys, useProjectView } from "../../api/queries";
import type { QuestLog, TableWindow } from "../../api/schema";
import { plain } from "../../kit";
import { StageFocusProvider } from "../../state/focus";
import { QuestScreen } from "../QuestScreen";
import { DRAWN, ESTIMATE, FIRST, OUTCOME, RECORDED, SERVED, type Moment, artifactOf, histogramOf, stageName, summaryOf } from "./fixtures/served";
import { FirstLookGallery, FirstLookGuide } from ".";

const PID = "first-look";
const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

interface Opts {
  after?: Moment;
  refuse?: unknown;
  /** reads left unanswered, as a read still on its way */
  hold?: string[];
}

/**
 * The server as it answered at `start`: each read the gallery makes; any other stays pending, as
 * does each read in `hold`. A POST is answered as the server answered it: a record moves the server
 * on to `after` and answers with that view; `refuse` answers every POST with that refusal.
 */
function server(start: Moment, opts: Opts = {}) {
  let now = start;
  const posts: { path: string; query: URLSearchParams; body: unknown }[] = [];
  const reads: string[] = [];
  const fetch = vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(input, "http://localhost");
    const path = url.pathname.replace(`/api/projects/${PID}`, "");
    if (init?.method === "POST") {
      posts.push({ path, query: url.searchParams, body: JSON.parse(String(init.body)) });
      if (opts.refuse) return json(409, opts.refuse);
      if (opts.after) now = opts.after;
      return json(200, now.view);
    }
    reads.push(path);
    if (opts.hold?.includes(path)) return new Promise<Response>(() => {});
    if (path === "/columns") return json(now.columns.status, now.columns.body);
    if (path === "/table") return json(now.table.status, now.table.body);
    if (path === "/stages/explore") return json(now.explore.status, now.explore.body);
    if (path === "/stages/target_info") return json(now.target_info.status, now.target_info.body);
    const h = /^\/columns\/([^/]+)\/histogram$/.exec(path);
    const served = h ? now.histograms[decodeURIComponent(h[1]!)] : undefined;
    if (served) return json(served.status, served.body);
    return new Promise<Response>(() => {});
  });
  vi.stubGlobal("fetch", fetch);
  return { posts, reads };
}

function client(m: Moment) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.view(PID), m.view);
  qc.setQueryData(keys.quest(PID), m.quest);
  return qc;
}

/** The gallery, as the tapestry mounts it: from the view the shell holds. */
function Gallery({ log }: { log: QuestLog }) {
  const view = useProjectView(PID).data;
  return view ? <FirstLookGallery pid={PID} view={view} log={log} /> : null;
}

/** The guide on a card and the gallery beside it, sharing one stage focus, as in the shell. */
function shell(m: Moment, opts: Opts = {}) {
  const net = server(m, opts);
  const qc = client(m);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <div data-testid="card">
          <FirstLookGuide pid={PID} log={m.quest} />
        </div>
        <div data-testid="canvas">
          <Gallery log={m.quest} />
        </div>
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  return { ...net, qc, card: screen.getByTestId("card"), canvas: screen.getByTestId("canvas") };
}

/** The sex look's one recording option: score performance within its groups. */
const SEX = "explore::quality::sex";
const DISTRIBUTION = "explore::outcome_distribution";
const sexOption = () => artifactOf(RECORDED).findings.find((f) => f.id === SEX)!.lever!.options.find((o) => o.decision)!;

/**
 * By construction, the moment right after "Decide now" on the sex look recorded its option after
 * the outcome's views were recorded: the log holds the record and the state its subgroup, and
 * Explore, which reads that answer, is stale for its new key (the server then reads the rows
 * again), while the result it served for its earlier key is all GET /stages/explore holds.
 */
function afterSubgroup(): Moment {
  const v = RECORDED.view;
  const last = v.decisions[v.decisions.length - 1]!;
  const decision = sexOption().decision as unknown as (typeof last)["decision"];
  return {
    ...RECORDED,
    view: {
      ...v,
      decisions: [...v.decisions, { ...last, id: "subgroup", seq: last.seq + 1, decision }],
      state: { ...v.state, intended_use: { use: "risk_estimation", threshold_low: null, threshold_high: null, threshold: null, subgroups: ["sex"], fairness: "subgroup_performance" } },
      stages: { ...v.stages, explore: { ...v.stages.explore!, status: "stale", fresh: false, key: "explore's key with the subgroup" } },
    },
  };
}

const inert = (el: HTMLElement) => el.closest("[inert]") !== null;

afterEach(() => {
  vi.unstubAllGlobals();
});

const counts = (el: HTMLElement) => [...el.querySelectorAll("[data-count]")].map((x) => Number(x.getAttribute("data-count")));
const relationships = (m: Moment) => artifactOf(m).findings.filter((f) => f.kind === "outcome_relationship");

describe("First look before Who's in, under Predict", () => {
  it("draws every column but the outcome, each number by the histogram the server served, each category by its levels", async () => {
    const { canvas } = shell(FIRST);
    const names = FIRST.columns.body.map((c) => c.name).filter((n) => n !== OUTCOME);
    await within(canvas).findByTestId(`look-column-${names[0]}`);
    expect([...canvas.querySelectorAll('[data-testid^="look-column-"]')].map((el) => el.getAttribute("data-testid"))).toEqual(
      names.map((n) => `look-column-${n}`),
    );
    for (const c of FIRST.columns.body.filter((x) => x.name !== OUTCOME)) {
      const cell = within(canvas).getByTestId(`look-column-${c.name}`);
      if (c.dtype === "numeric" || c.dtype === "integer") {
        await waitFor(() => expect(counts(cell)).toEqual(histogramOf(FIRST, c.name).counts));
      } else if ((c.dtype === "categorical" || c.dtype === "boolean") && c.top?.length) {
        expect(counts(cell)).toEqual(c.top.map((t) => t.count));
      } else {
        expect(cell).toHaveTextContent(`${c.n_unique.toLocaleString("en-US")} distinct values`);
      }
    }
  });

  it("keeps the outcome shut in the server's words, offers nothing to press, and never asks for its values", async () => {
    const { canvas, card, reads } = shell(FIRST);
    const line = summaryOf(FIRST).withheld!;
    const door = await within(canvas).findByTestId("look-outcome-distribution-line");
    expect(door).toHaveTextContent(plain(line));
    expect(within(canvas).queryByTestId("open-outcome-distribution")).toBeNull();
    const rows = await within(canvas).findByTestId("look-outcome-rows-line");
    expect(rows).toHaveTextContent(plain(FIRST.table.body.withheld[OUTCOME]));
    expect(within(canvas).queryByTestId("open-outcome-rows")).toBeNull();
    const head = [...within(canvas).getByTestId("look-outcome-rows").querySelectorAll("th")].map((th) => th.textContent);
    expect(head).toEqual(FIRST.table.body.columns);
    expect(head).not.toContain(OUTCOME);
    expect(within(canvas).getByTestId("first-look-explore-line")).toHaveTextContent(
      `Explore has not run yet: it reads the rows ${stageName(FIRST, "whos_in")} keeps.`,
    );
    // The card's beats say the same, and the outcome's last.
    expect(await within(card).findByTestId("beat-outcome")).toHaveTextContent(plain(line));
    expect(within(card).getByTestId("beat-columns")).toHaveTextContent(`${FIRST.columns.body.length - 1} columns`);
    expect([...card.querySelectorAll('[data-testid^="beat-"][data-testid$="-show"]')]).toHaveLength(3);
    expect(reads).not.toContain(`/columns/${OUTCOME}/histogram`);
  });
});

describe("after the draw, each of the outcome's views opens by its own press, recorded", () => {
  it("the distribution: the press posts the record GET /columns handed back, and only then is it drawn, on the training rows", async () => {
    const { canvas, posts, reads } = shell(DRAWN, { after: RECORDED });
    const open = await within(canvas).findByTestId("open-outcome-distribution");
    expect(open.tagName).toBe("BUTTON");
    expect(open).toHaveTextContent("Open the outcome's distribution");
    expect(within(canvas).getByTestId("look-outcome-distribution-line")).toHaveTextContent(plain(summaryOf(DRAWN).withheld));
    expect(reads).not.toContain(`/columns/${OUTCOME}/histogram`);
    fireEvent.click(open);
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toMatchObject({ path: "/decisions", body: summaryOf(DRAWN).record });
    expect(posts[0]!.query.get("decide_now")).toBeNull();
    const cell = within(canvas).getByTestId("look-outcome-distribution");
    await waitFor(() => expect(counts(cell)).toEqual(histogramOf(RECORDED, OUTCOME).counts));
    const opened = summaryOf(RECORDED);
    expect(opened.rows).toBe("training");
    expect(within(cell).getByTestId("look-outcome-distribution-rows")).toHaveTextContent(`On the ${opened.n} training rows with a value`);
    expect(within(canvas).queryByTestId("open-outcome-distribution")).toBeNull();
  });

  it("a relationship: drawn only once recorded though Explore served its points; the press posts the finding's own record", async () => {
    const { canvas, posts } = shell(DRAWN, { after: RECORDED });
    const served = relationships(DRAWN);
    for (const f of served) {
      const cell = await within(canvas).findByTestId(`look-${f.id}`);
      expect(f.points.length).toBeGreaterThan(0);
      expect(cell).toHaveAttribute("data-state", "record");
      expect(cell.querySelectorAll("circle")).toHaveLength(0);
    }
    const age = served.find((f) => f.columns[0] === "age")!;
    fireEvent.click(within(canvas).getByTestId(`look-${age.id}-open`));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toMatchObject({ path: "/decisions", body: age.record });
    const cell = within(canvas).getByTestId(`look-${age.id}`);
    await waitFor(() => expect(cell).toHaveAttribute("data-state", "drawn"));
    expect([...cell.querySelectorAll("circle")].map((c) => Number(c.getAttribute("data-n")))).toEqual(age.points.map((p) => p.n));
    // The outcome is yes or no (target_info's task): each tenth's share with its event.
    expect(SERVED.predict.recorded.target_info.body.artifact!.detected_task).toBe("binary");
    expect(cell.querySelector('svg[role="img"]')).toHaveAttribute(
      "aria-label",
      `Share of rows where ${OUTCOME} is ${RECORDED.view.state.event} in each tenth of age`,
    );
    for (const f of served.filter((x) => x.id !== age.id)) {
      expect(within(canvas).getByTestId(`look-${f.id}`)).toHaveAttribute("data-state", "record");
    }
  });

  it("the rows: the press posts the window's record, then the outcome stands beside the other columns", async () => {
    const { canvas, posts } = shell(DRAWN, { after: RECORDED });
    const open = await within(canvas).findByTestId("open-outcome-rows");
    expect(open).toHaveTextContent("Open the outcome beside the other columns");
    fireEvent.click(open);
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toMatchObject({ path: "/decisions", body: DRAWN.table.body.record });
    const cell = within(canvas).getByTestId("look-outcome-rows");
    const w = RECORDED.table.body;
    await waitFor(() => expect([...cell.querySelectorAll("th")].map((th) => th.textContent)).toEqual(w.columns));
    const at = w.columns.indexOf(OUTCOME);
    expect([...cell.querySelectorAll("tbody tr")].map((tr) => tr.querySelectorAll("td")[at]!.textContent)).toEqual(
      w.rows.map((r) => (r[at] === null ? "" : String(r[at]))),
    );
  });
});

describe("under Estimate, after Who's in", () => {
  it("the relationships wait for the lock, said once in the server's words with nothing to press; the outcome alone can be opened", async () => {
    const { canvas } = shell(ESTIMATE);
    const served = relationships(ESTIMATE);
    const lines = [...new Set(served.map((f) => f.detail))];
    expect(lines).toHaveLength(1);
    const waiting = await within(canvas).findByTestId("look-relationships-waiting");
    expect(within(waiting).getByTestId("look-relationships-waiting-line")).toHaveTextContent(plain(lines[0]));
    for (const f of served) expect(waiting).toHaveTextContent(f.columns[0]!);
    for (const f of served) expect(within(canvas).queryByTestId(`look-${f.id}-open`)).toBeNull();
    expect(await within(canvas).findByTestId("look-outcome-rows-line")).toHaveTextContent(plain(ESTIMATE.table.body.withheld[OUTCOME]));
    expect(within(canvas).queryByTestId("open-outcome-rows")).toBeNull();
    expect(within(canvas).getByTestId("open-outcome-distribution")).toBeInTheDocument();
  });

  it("the card's outcome beat says the outcome beside another column waits, in the server's words, as well as its distribution's door", async () => {
    const { card } = shell(ESTIMATE);
    const beat = await within(card).findByTestId("beat-outcome");
    await waitFor(() => expect(within(beat).getAllByTestId("beat-outcome-also").map((p) => p.textContent)).toEqual([plain(ESTIMATE.table.body.withheld[OUTCOME])]));
    expect(beat).toHaveTextContent(plain(summaryOf(ESTIMATE).withheld));
  });

  it("the card's outcome beat says nothing more while the window is on its way or was not read, though the relationships wait", async () => {
    const waits = [...new Set(relationships(ESTIMATE).map((f) => f.detail))];
    expect(waits).toHaveLength(1);
    for (const opts of [{ hold: ["/table"] }, {}] as Opts[]) {
      const m = opts.hold ? ESTIMATE : { ...ESTIMATE, table: { status: 500, body: "Internal Server Error" as unknown as TableWindow } };
      const { card, canvas } = shell(m, opts);
      // Explore's result is read: the relationships wait, in their own line, on the canvas.
      expect(await within(canvas).findByTestId("look-relationships-waiting-line")).toHaveTextContent(plain(waits[0]));
      const beat = await within(card).findByTestId("beat-outcome");
      expect(beat).toHaveTextContent(plain(summaryOf(ESTIMATE).withheld));
      const rows = within(canvas).getByTestId("look-outcome-rows");
      if (opts.hold) expect(rows).toHaveTextContent("Reading the rows…");
      else await waitFor(() => expect(within(rows).getByTestId("look-unread")).toHaveTextContent("Not read: Internal Server Error"));
      expect(within(beat).queryAllByTestId("beat-outcome-also")).toEqual([]);
      expect(beat).not.toHaveTextContent(plain(waits[0]));
      cleanup();
    }
  });
});

describe("under Predict while Explore reads the rows again", () => {
  it("the card's outcome beat says nothing of the relationships, whose line the server serves in their place while Explore is not current", async () => {
    // By construction, in the served shape: an answer Explore reads changed after the record, and
    // while Explore reads the rows again the service serves its last result with each
    // relationship's points withheld and a line in their place (service.py `_outcome_served`).
    const line = "The line served in the relationships' place";
    const withheld = {
      ...artifactOf(RECORDED),
      findings: artifactOf(RECORDED).findings.map((f) => (f.kind === "outcome_relationship" ? { ...f, points: [], detail: line, record: null } : f)),
    };
    const v = RECORDED.view;
    const m: Moment = {
      ...RECORDED,
      view: { ...v, stages: { ...v.stages, explore: { ...v.stages.explore!, status: "running", fresh: false, key: "explore's key after the change" } } },
      explore: { ...RECORDED.explore, body: { ...RECORDED.explore.body, artifact: withheld } },
    };
    const { card, canvas } = shell(m);
    const waiting = await within(canvas).findByTestId("look-relationships-waiting-line");
    expect(waiting).toHaveTextContent(line);
    expect(inert(waiting)).toBe(true);
    const beat = await within(card).findByTestId("beat-outcome");
    await waitFor(() => expect(within(card).getByTestId("beat-explore")).toHaveTextContent("Explore is reading the rows again"));
    // The window is open and read: its rows stand beside the other columns.
    await waitFor(() => expect([...within(canvas).getByTestId("look-outcome-rows").querySelectorAll("th")].map((th) => th.textContent)).toEqual(RECORDED.table.body.columns));
    expect(within(beat).queryAllByTestId("beat-outcome-also")).toEqual([]);
    expect(beat).not.toHaveTextContent(line);
  });
});

describe("a read the server did not answer", () => {
  const failed = { status: 500, body: "Internal Server Error" };
  const NOT_READ = "Not read: Internal Server Error";

  it("the rows window: says it was not read where GET /table failed, and that it is reading while it is on its way", async () => {
    const { canvas } = shell({ ...DRAWN, table: failed as unknown as Moment["table"] });
    const rows = within(canvas).getByTestId("look-outcome-rows");
    await waitFor(() => expect(within(rows).getByTestId("look-unread")).toHaveTextContent(NOT_READ));
    expect(rows).not.toHaveTextContent("Reading");
    cleanup();
    // Both ways: held, it is still on its way.
    const held = shell(DRAWN, { hold: ["/table"] });
    await within(held.canvas).findByTestId("open-outcome-distribution");
    const reading = within(held.canvas).getByTestId("look-outcome-rows");
    expect(reading).toHaveTextContent("Reading the rows…");
    expect(within(reading).queryByTestId("look-unread")).toBeNull();
  });

  it("the columns: where GET /columns failed, the guide, each column, the outcome's distribution and its rows say so", async () => {
    const { canvas, card } = shell({ ...DRAWN, columns: failed as unknown as Moment["columns"] });
    await waitFor(() => expect(within(card).getByTestId("look-unread")).toHaveTextContent(NOT_READ));
    for (const id of ["first-look-columns", "look-outcome-distribution", "look-outcome-rows"]) {
      const cell = within(canvas).getByTestId(id);
      await waitFor(() => expect(within(cell).getByTestId("look-unread")).toHaveTextContent(NOT_READ));
      expect(cell).not.toHaveTextContent("Reading");
    }
    expect(card).not.toHaveTextContent("Reading");
    cleanup();
    // Both ways: held, each is still on its way.
    const held = shell(DRAWN, { hold: ["/columns"] });
    expect(await within(held.card).findByTestId("first-look-guide")).toHaveTextContent("Reading the columns…");
    expect(within(held.canvas).getByTestId("first-look-columns")).toHaveTextContent("Reading the columns…");
    expect(within(held.canvas).getByTestId("look-outcome-distribution")).toHaveTextContent("Reading when it opens…");
    expect(within(held.canvas).getByTestId("look-outcome-rows")).toHaveTextContent("Reading the rows…");
    expect(within(held.card).queryByTestId("look-unread")).toBeNull();
    expect(within(held.canvas).queryByTestId("look-unread")).toBeNull();
  });
});

describe("the rows window once open", () => {
  it("says each of the server's lines on it: the rows it leaves blank and its label", async () => {
    // By construction, in the served shape (service.py under Estimate after the lock).
    const table = { ...RECORDED.table, body: { ...RECORDED.table.body, withheld: { [OUTCOME]: "Blank rows' line" }, labels: { [OUTCOME]: "The label" } } };
    const { canvas } = shell({ ...RECORDED, table });
    const cell = await within(canvas).findByTestId("look-outcome-rows");
    await waitFor(() => expect(within(cell).getAllByTestId("look-outcome-rows-line").map((p) => p.textContent)).toEqual(["Blank rows' line", "The label"]));
  });
});

describe("while Explore's result is read", () => {
  it("says it is reading what Explore found, not that Explore has not run, while the stage is fresh and its result on its way", async () => {
    expect(DRAWN.view.stages.explore!.status).toBe("fresh");
    const { canvas, card } = shell(DRAWN, { hold: ["/stages/explore"] });
    expect(await within(card).findByTestId("beat-explore")).toHaveTextContent("Reading what Explore found…");
    expect(within(canvas).getByTestId("first-look-explore-line")).toHaveTextContent("Reading what Explore found…");
    expect(card).not.toHaveTextContent("has not run yet");
    expect(canvas).not.toHaveTextContent("has not run yet");
  });

  it("says the read failed, with the server's reason, where it failed", async () => {
    const failed = { ...DRAWN, explore: { status: 500, body: "Internal Server Error" as unknown as Moment["explore"]["body"] } };
    const { canvas, card } = shell(failed);
    const line = "What Explore found was not read: Internal Server Error";
    await waitFor(() => expect(within(canvas).getByTestId("first-look-explore-line")).toHaveTextContent(line));
    expect(within(card).getByTestId("beat-explore")).toHaveTextContent(line);
    expect(canvas).not.toHaveTextContent("has not run yet");
  });
});

describe("Decide now", () => {
  const sex = artifactOf(DRAWN).findings.find((f) => f.id === "explore::quality::sex")!;
  const option = sex.lever!.options.find((o) => o.decision)!;

  it("hands a look to the card, which asks its lever's question and records the option with decide_now", async () => {
    const { canvas, card, posts } = shell(DRAWN, { after: RECORDED });
    fireEvent.click(await within(canvas).findByTestId(`look-${sex.id}-decide`));
    const asked = await within(card).findByTestId("decide-now");
    expect(asked).toHaveTextContent("Asked from First look");
    expect(asked.querySelector("h3")).toHaveTextContent(plain(sex.lever!.question));
    expect(within(card).queryByTestId("first-look-guide")).toBeNull();
    fireEvent.click(within(asked).getByTestId(`option-${option.key}`));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toMatchObject({ path: "/decisions", body: option.decision });
    expect(posts[0]!.query.get("decide_now")).toBe("true");
    // Recorded: back to the beats.
    expect(await within(card).findByTestId("first-look-guide")).toBeInTheDocument();
  });

  it("a refusal is said at the option, its way forward said, not offered as a button that does nothing", async () => {
    const refusal = SERVED.predict.decide_now_refused.answer.body;
    expect(SERVED.predict.decide_now_refused.answer.status).toBe(409);
    const { canvas, card, posts } = shell(DRAWN, { refuse: refusal });
    fireEvent.click(await within(canvas).findByTestId(`look-${sex.id}-decide`));
    fireEvent.click(within(await within(card).findByTestId("decide-now")).getByTestId(`option-${option.key}`));
    const said = await within(card).findByTestId("decide-now-refusal");
    expect(posts[0]!.query.get("decide_now")).toBe("true");
    expect(said).toHaveTextContent(plain(refusal.error.message));
    const ways = refusal.error.exits;
    expect(ways.length).toBeGreaterThan(0);
    expect(ways.every((w) => w.decision === null)).toBe(true);
    expect(within(said).getAllByTestId("decide-now-way").map((p) => p.textContent)).toEqual(ways.map((w) => plain(w.label)));
    expect(within(said).queryByTestId("decide-now-exit")).toBeNull();
    const buttons = [...said.querySelectorAll("button")].map((b) => b.textContent ?? "");
    for (const w of ways) expect(buttons.some((b) => b.includes(plain(w.label)))).toBe(false);
    // Still asked: nothing was recorded.
    expect(within(card).getByTestId("decide-now")).toBeInTheDocument();
    fireEvent.click(within(card).getByTestId("decide-now-back"));
    expect(await within(card).findByTestId("first-look-guide")).toBeInTheDocument();
  });

  it("is offered on a look whose lever records, and not on one whose lever records nothing", async () => {
    // By construction: the same served artifact with the sex look's options recording nothing.
    const told = {
      ...DRAWN,
      explore: {
        ...DRAWN.explore,
        body: {
          ...DRAWN.explore.body,
          artifact: {
            ...artifactOf(DRAWN),
            findings: artifactOf(DRAWN).findings.map((f) =>
              f.id === sex.id ? { ...f, lever: { ...f.lever!, options: f.lever!.options.map((o) => ({ ...o, decision: null })) } } : f,
            ),
          },
        },
      },
    };
    const { canvas } = shell(told);
    const cell = await within(canvas).findByTestId(`look-${sex.id}`);
    expect(within(cell).queryByTestId(`look-${sex.id}-decide`)).toBeNull();
    const age = artifactOf(DRAWN).findings.find((f) => f.id === "explore::quality::age")!;
    expect(age.lever!.options.some((o) => o.decision)).toBe(true);
    expect(within(canvas).getByTestId(`look-${age.id}-decide`)).toBeInTheDocument();
  });

  it("is offered on no look while Explore reads the rows again after an answer it reads, the outcome's distribution's among them", async () => {
    // The sequence of the verifier's probe: Decide now on the sex look records its subgroup, which
    // Explore reads; the distribution's lever, computed before, would record the slot without it.
    const { canvas, card, posts } = shell(RECORDED, { after: afterSubgroup() });
    const distribution = await within(canvas).findByTestId(`look-${DISTRIBUTION}-decide`);
    const sex = within(canvas).getByTestId(`look-${SEX}-decide`);
    // Both ways: while what Explore found is current, each is offered.
    expect(inert(distribution)).toBe(false);
    expect(inert(sex)).toBe(false);
    fireEvent.click(sex);
    fireEvent.click(within(await within(card).findByTestId("decide-now")).getByTestId(`option-${sexOption().key}`));
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toMatchObject({ body: sexOption().decision });
    expect(await within(card).findByTestId("first-look-guide")).toBeInTheDocument();
    // Now Explore is stale: every Decide now on the canvas is veiled and inert, the distribution's too.
    const decides = [...canvas.querySelectorAll<HTMLElement>('[data-testid$="-decide"]')];
    expect(decides.map((b) => b.getAttribute("data-testid"))).toEqual(expect.arrayContaining([`look-${DISTRIBUTION}-decide`, `look-${SEX}-decide`]));
    for (const b of decides) expect(inert(b)).toBe(true);
    expect(within(card).getByTestId("beat-explore")).toHaveTextContent("Explore is reading the rows again; nothing it found can be decided until it has.");
    expect(within(card).getByTestId("beat-explore")).not.toHaveTextContent("can be decided now");
    expect(posts).toHaveLength(1);
  });

  it("asks nothing to record while Explore reads the rows again, and says so, then asks again once it has", async () => {
    const { canvas, card, posts, qc } = shell(RECORDED);
    fireEvent.click(await within(canvas).findByTestId(`look-${DISTRIBUTION}-decide`));
    const asked = await within(card).findByTestId("decide-now");
    const distribution = artifactOf(RECORDED).findings.find((f) => f.id === DISTRIBUTION)!;
    const recording = distribution.lever!.options.filter((o) => o.decision);
    expect(recording.length).toBeGreaterThan(0);
    for (const o of recording) expect(within(asked).getByTestId(`option-${o.key}`)).toBeInTheDocument();
    // An answer Explore reads changes elsewhere: the view the shell holds says Explore is stale.
    const stale = afterSubgroup().view;
    act(() => {
      qc.setQueryData(keys.view(PID), stale);
    });
    await waitFor(() => expect(within(card).getByTestId("decide-now-waits")).toHaveTextContent("Explore is reading the rows again; nothing it found can be decided until it has."));
    expect(within(card).getByTestId("decide-now").querySelector("h3")).toHaveTextContent(plain(distribution.lever!.question));
    for (const o of distribution.lever!.options) expect(within(card).queryByTestId(`option-${o.key}`)).toBeNull();
    expect(posts).toHaveLength(0);
    // Fresh again: its options are asked again.
    act(() => {
      qc.setQueryData(keys.view(PID), RECORDED.view);
    });
    await waitFor(() => expect(within(card).queryByTestId("decide-now-waits")).toBeNull());
    for (const o of recording) expect(within(card).getByTestId(`option-${o.key}`)).toBeInTheDocument();
  });
});

describe("a look at near-constant predictors", () => {
  // By construction, in the served shape (explore.py `ExploreFinding` kind low_variance): the
  // summary counts 15, the columns name the first 12, and the lever's first option records.
  const named = Array.from({ length: 12 }, (_, i) => `flat_${i + 1}`);
  const sex = artifactOf(DRAWN).findings.find((f) => f.id === SEX)!;
  const flat = (summary: string, columns: string[]) => ({
    ...sex,
    id: "explore::low_variance",
    kind: "low_variance",
    summary,
    columns,
    groups: [],
    view: "table_focus",
    detail: "The rule, as served.",
    lever: {
      question: "Drop near-constant predictors inside each training fold?",
      options: [{ ...sex.lever!.options[0]!, key: "near_zero", decision: { kind: "set_levers", variance_filter: "near_zero" } }],
    },
  });
  const served = (f: ReturnType<typeof flat>): Moment => ({
    ...DRAWN,
    explore: { ...DRAWN.explore, body: { ...DRAWN.explore.body, artifact: { ...artifactOf(DRAWN), findings: [...artifactOf(DRAWN).findings, f] } } },
  });

  it("names the predictors it is about beside its Decide now, and how many more the summary counts", async () => {
    const { canvas } = shell(served(flat("15 predictors barely vary on the training rows", named)));
    const cell = await within(canvas).findByTestId("look-explore::low_variance");
    expect(within(cell).getByTestId("look-explore::low_variance-columns")).toHaveTextContent(`${named.slice(0, -1).join(", ")}, ${named.at(-1)}, and 3 more`);
    expect(within(cell).getByTestId("look-explore::low_variance-decide")).toBeInTheDocument();
  });

  it("says no more where it names every one", async () => {
    const { canvas } = shell(served(flat("2 predictors barely vary on the training rows", ["flat_1", "flat_2"])));
    const cell = await within(canvas).findByTestId("look-explore::low_variance");
    expect(within(cell).getByTestId("look-explore::low_variance-columns").textContent).toBe("flat_1 and flat_2");
  });
});

describe("the quest shell's First look mounts", () => {
  function questShell(m: Moment) {
    server(m);
    render(
      <QueryClientProvider client={client(m)}>
        <QuestScreen pid={PID} stream="open" />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByTestId("stage-first_look"));
  }

  it("puts the guide on First look's card and the gallery on its canvas, and neither on another stage's", async () => {
    questShell(DRAWN);
    const card = screen.getByTestId("card-first_look");
    expect(await within(card).findByTestId("first-look-guide")).toBeInTheDocument();
    const tapestry = screen.getByTestId("tapestry-first_look");
    expect(within(tapestry).getByTestId("first-look-gallery")).toBeInTheDocument();
    expect(screen.queryByTestId("tapestry")).toBeNull();
    fireEvent.click(screen.getByTestId("stage-whos_in"));
    expect(screen.queryByTestId("first-look-guide")).toBeNull();
    expect(screen.queryByTestId("first-look-gallery")).toBeNull();
    expect(screen.getByTestId("tapestry")).toBeInTheDocument();
  });

  it("previews an option asked from First look on the stage, as any option is, and returns to the gallery", async () => {
    questShell(DRAWN);
    const sex = "explore::quality::sex";
    fireEvent.click(await within(screen.getByTestId("tapestry-first_look")).findByTestId(`look-${sex}-decide`));
    const asked = await within(screen.getByTestId("card-first_look")).findByTestId("decide-now");
    const option = artifactOf(DRAWN).findings.find((f) => f.id === sex)!.lever!.options.find((o) => o.decision)!;
    fireEvent.focus(within(asked).getByTestId(`option-${option.key}`));
    expect(screen.queryByTestId("tapestry-first_look")).toBeNull();
    expect(screen.getByTestId("tapestry")).toBeInTheDocument();
    fireEvent.click(within(asked).getByTestId("decide-now-back"));
    expect(screen.getByTestId("tapestry-first_look")).toBeInTheDocument();
  });
});
