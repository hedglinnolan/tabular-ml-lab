/**
 * Write-up shows what the export holds, read-only (G17; the owner's Q5, 2026-11-04): its results
 * tables, its figures, the analysis plan and the checklist's items with their scope and what the
 * author owes, mounted by the quest shell's card at Write-up (StageCard.tsx).
 *
 * The payloads are what the server serves for two projects built by hand (fixtures/capture.py):
 * the zip GET …/export answers, GET …/checklist, GET …/plan, and while a result is computing the
 * checklist then and the export's 409. Every expected word and number is read from them by Python
 * (zipfile, the engine's Table models, ElementTree, the bundle's own checklist file), never retyped
 * here. The quest log is the engine's (fixtures/quest-contract.json, Write-up reached under
 * Predict); the view is a real server's captured NHANES walk.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, configure, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import { keys } from "../../api/queries";
import type { PlanExport } from "../../api/m3-types";
import type { ChecklistItem, ChecklistReport, ProjectView, QuestLog, Refusal } from "../../api/schema";
import { plain } from "../../kit";
import { expandView, viewsOf, type M3Fixture } from "../../mocks/m3";
import { StageFocusProvider } from "../../state/focus";
import { CONTRACT } from "../fixtures/contract";
import { StageCard } from "../StageCard";
import estimateRaw from "./fixtures/bundle-estimate.json";
import predictRaw from "./fixtures/bundle-predict.json";
import { EXPORT_TAG } from "./reads";

interface Captured {
  purpose: string;
  zip: string;
  checklist: ChecklistReport;
  plan: PlanExport;
  waiting: { checklist: ChecklistReport; refusal: Refusal };
  truth: {
    files: Record<string, string>;
    tables: { file: string; title: string; caption: string; header: string[]; rows: string[][]; empty: string | null }[];
    figures: Record<string, { title: string; desc: string }>;
    checklist_headings: string[];
  };
}
// Each test draws the whole card, every checklist item and the bundle's tables, after inflating
// the zip: on a loaded machine that takes past the 1 s a wait gives by default (and a test's 5 s),
// so the waits on the export's read, and the tests, get a budget of their own.
configure({ asyncUtilTimeout: 15_000 });
vi.setConfig({ testTimeout: 60_000 });

const CAPTURED: [string, Captured][] = [
  ["prediction", predictRaw as unknown as Captured],
  ["estimate", estimateRaw as unknown as Captured],
];
const [, PREDICT] = CAPTURED[0]!;
const [, ESTIMATE] = CAPTURED[1]!;

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-nhanes-prediction.json", { eager: true, import: "default" });
const JOURNEY = Object.values(FIXTURES)[0]!;
const PID = "quest-writeup";
const VIEW = ((): ProjectView => {
  const views = viewsOf(JOURNEY);
  return expandView(JOURNEY, views[views.length - 1]!, PID, "2026-11-04T00:00:00Z");
})();
const LOG: QuestLog = { ...CONTRACT.logs.results_reached.log, fit: CONTRACT.fit_locks.pressed };
const WRITEUP = LOG.stages.find((s) => s.key === "writeup")!;

const bytesOf = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const zipAnswer = (c: Captured) => () =>
  new Response(bytesOf(c.zip), {
    status: 200,
    headers: { "Content-Type": "application/zip", "Content-Disposition": 'attachment; filename="writeup-turbotab-export.zip"' },
  });
const refusalAnswer = (c: Captured) => () =>
  new Response(JSON.stringify(c.waiting.refusal), { status: 409, headers: { "Content-Type": "application/json" } });

/** The card at Write-up with the served checklist and plan in the cache; GET …/export answered by
 *  `exportAnswer`, GET …/checklist by the same checklist until `serve` names another, everything
 *  else left pending. */
function writeUp(c: Captured, opts: { checklist?: ChecklistReport; exportAnswer?: () => Response | Promise<Response> } = {}) {
  const checklist = opts.checklist ?? c.checklist;
  let served = checklist;
  const answer = opts.exportAnswer ?? zipAnswer(c);
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), LOG);
  qc.setQueryData(keys.view(PID), VIEW);
  qc.setQueryData(keys.record(PID), { version: 1, stages: [] });
  qc.setQueryData(keys.checklist(PID), checklist);
  qc.setQueryData(keys.plan(PID), c.plan);
  const fetched = vi.fn((url: string) => {
    if (url.endsWith(`/projects/${PID}/export`)) return Promise.resolve(answer());
    if (url.endsWith(`/projects/${PID}/checklist`)) return Promise.resolve(Response.json(served));
    return new Promise<Response>(() => {});
  });
  vi.stubGlobal("fetch", fetched);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <StageCard pid={PID} log={LOG} stage={WRITEUP} answers={new Map()} fit={{ kind: "pressed" }} onOpenStage={() => {}} arrival={null} onOpenManuscript={() => {}} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  const calls = (route: string) => fetched.mock.calls.filter(([url]) => String(url).endsWith(`/projects/${PID}/${route}`)).length;
  const serve = (next: ChecklistReport) => {
    served = next;
  };
  return { qc, card: screen.getByTestId("card-writeup"), calls, serve };
}

const settle = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
const cellsOf = (row: HTMLElement) => [...row.querySelectorAll("th, td")].map((c) => c.textContent);
/** The words the veil over the drawn tables and figures says (StaleVeil's live tag). */
const tagOf = (card: HTMLElement) => within(card).getByTestId("export-veil").querySelector("[aria-live]")!.textContent;
const EXPORT_TAGS: string[] = Object.values(EXPORT_TAG);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe.each(CAPTURED)("Write-up shows what the %s export holds", (_, c) => {
  it("draws each results table as the export writes it: its title, its caption, its header and every cell, or its line in place of rows", async () => {
    const { card, calls } = writeUp(c);
    const drawn = await within(card).findByTestId("export-tables");
    const sections = within(drawn).getAllByTestId("export-table");
    expect(sections.map((s) => s.getAttribute("data-file"))).toEqual(c.truth.tables.map((t) => t.file));
    c.truth.tables.forEach((t, i) => {
      const section = sections[i]!;
      expect(within(section).getByRole("heading", { name: plain(t.title) })).toBeInTheDocument();
      expect(within(section).getByText(plain(t.caption))).toBeInTheDocument();
      if (t.empty !== null) {
        expect(within(section).queryByRole("table")).toBeNull();
        expect(within(section).getByText(t.empty)).toBeInTheDocument();
        return;
      }
      const table = within(section).getByRole("table", { name: plain(t.title) });
      const [head, ...body] = within(table).getAllByRole("row");
      expect(cellsOf(head!)).toEqual(t.header);
      expect(body.map(cellsOf)).toEqual(t.rows);
    });
    expect(calls("export")).toBe(1);
  });

  it("draws each figure from its own file: the drawing, named by its title, and its caption", async () => {
    const { card } = writeUp(c);
    const drawn = await within(card).findByTestId("export-figures");
    const figures = Object.entries(c.truth.figures);
    expect(within(drawn).getAllByRole("img")).toHaveLength(figures.length);
    for (const [name, figure] of figures) {
      const img = within(drawn).getByRole("img", { name: figure.title });
      const src = img.getAttribute("src")!;
      expect(src.startsWith("data:image/svg+xml;base64,")).toBe(true);
      expect(await sha256(bytesOf(src.slice(src.indexOf(",") + 1)))).toBe(c.truth.files[name]);
      expect(within(img.closest("figure")!).getByText(plain(figure.desc))).toBeInTheDocument();
    }
  });

  it("holds the analysis plan as served: its text, and every decision it holds and made after the estimates were seen", () => {
    const { card } = writeUp(c);
    const plan = within(card).getByTestId("export-plan");
    expect(within(plan).getByText(plain(c.plan.text))).toBeInTheDocument();
    for (const s of c.plan.sentences) expect(within(plan).getAllByText(plain(s)).length).toBeGreaterThan(0);
    for (const a of c.plan.after_estimates) expect(within(plan).getByText(plain(a.sentence as string))).toBeInTheDocument();
    expect(within(plan).getAllByRole("listitem")).toHaveLength(c.plan.sentences.length + c.plan.after_estimates.length);
  });

  it("lists every checklist item by section, headed as the bundle's checklist heads it, with what the author owes and where it is answered", () => {
    const { card } = writeUp(c);
    const list = within(card).getByTestId("checklist-items");
    expect(within(list).getByText(plain(c.checklist.citation))).toBeInTheDocument();
    const items = within(list).getAllByTestId("checklist-item");
    expect(items).toHaveLength(c.checklist.counts.items);
    c.checklist.items.forEach((item, i) => {
      const el = items[i]!;
      expect(el).toHaveAttribute("data-item", item.id);
      // TRIPOD+AI's heading carries the item's scope, STROBE's its table: the bundle's own words.
      expect(within(el).getByRole("heading").textContent).toBe(c.truth.checklist_headings[i]);
      expect(within(el).getByText(plain(item.text))).toBeInTheDocument();
      expect(within(el).getByText(plain(item.note))).toBeInTheDocument();
      for (const w of item.where) expect(within(el).getAllByText(plain(w.quote)).length).toBeGreaterThan(0);
    });
    const sections = within(list).getAllByTestId("checklist-section");
    expect(sections.map((s) => within(s).getAllByRole("heading")[0]!.textContent)).toEqual([...new Set(c.checklist.items.map((i) => i.section))]);
  });

  it("keeps the checklist's counts in its own words", () => {
    const { card } = writeUp(c);
    const n = c.checklist.counts;
    expect(within(card).getByTestId("checklist")).toHaveTextContent(
      `${n.answered} answered, ${n.partly_answered} partly answered, ${n.unanswered} unanswered, of ${n.items} items.`,
    );
  });
});

describe("the export is read only when it can be shown", () => {
  it("is not read while the checklist names what the export waits for, which the card says first", async () => {
    const { card, calls } = writeUp(PREDICT, { checklist: PREDICT.waiting.checklist });
    await settle();
    expect(calls("export")).toBe(0);
    expect(calls("checklist")).toBe(0);
    for (const w of PREDICT.waiting.checklist.waiting) expect(within(card).getByText(plain(w))).toBeInTheDocument();
    expect(within(card).getByTestId("export-later")).toBeInTheDocument();
    expect(within(card).queryByTestId("export-tables")).toBeNull();
    expect(within(card).queryByTestId("export-figures")).toBeNull();
  });

  it("once read, reads the checklist again, which may now quote the scores the export showed", async () => {
    const { card, calls } = writeUp(PREDICT);
    await within(card).findByTestId("export-tables");
    await waitFor(() => expect(calls("checklist")).toBe(1));
    expect(within(card).queryByTestId("export-later")).toBeNull();
  });

  it("says the export's refusal in its words where it refuses, and draws no table", async () => {
    const { card } = writeUp(PREDICT, { exportAnswer: refusalAnswer(PREDICT) });
    const refused = await within(card).findByTestId("export-refused");
    expect(refused).toHaveTextContent(plain(PREDICT.waiting.refusal.error.message));
    expect(within(card).queryByTestId("export-tables")).toBeNull();
  });

  it("is not read in a browser that cannot open a zip, so nothing it would record is recorded unseen", async () => {
    vi.stubGlobal("DecompressionStream", undefined);
    const { card, calls } = writeUp(PREDICT);
    await settle();
    expect(calls("export")).toBe(0);
    // The export waits for nothing, so Download holds its tables and figures now.
    expect(PREDICT.checklist.waiting).toEqual([]);
    expect(within(card).getByTestId("export-unreadable")).toHaveTextContent(
      /^This browser cannot open the export here; Download holds its tables and figures\.$/,
    );
    expect(within(card).queryByTestId("export-tables")).toBeNull();
  });

  it("says, in a browser that cannot open a zip, that Download holds the tables and figures only once the export waits for nothing, while it waits", async () => {
    vi.stubGlobal("DecompressionStream", undefined);
    const { card, calls } = writeUp(PREDICT, { checklist: PREDICT.waiting.checklist });
    await settle();
    expect(calls("export")).toBe(0);
    expect(PREDICT.waiting.checklist.waiting.length).toBeGreaterThan(0);
    for (const w of PREDICT.waiting.checklist.waiting) expect(within(card).getByText(plain(w))).toBeInTheDocument();
    expect(within(card).getByTestId("export-unreadable")).toHaveTextContent(
      /^This browser cannot open the export here; Download will hold its tables and figures once the export waits for nothing\.$/,
    );
    expect(within(card).queryByTestId("export-later")).toBeNull();
  });

  it("is not read in a browser whose DecompressionStream refuses \"deflate-raw\", the format the zip's entries are stored in", async () => {
    // As Chromium 80 to 102 do: the class is there, and it inflates "deflate" and "gzip" only.
    const Real = DecompressionStream;
    vi.stubGlobal(
      "DecompressionStream",
      class {
        constructor(format: CompressionFormat) {
          if (format === "deflate-raw") throw new TypeError("Unsupported compression format: 'deflate-raw'");
          return new Real(format);
        }
      },
    );
    const { card, calls } = writeUp(PREDICT);
    await settle();
    expect(calls("export")).toBe(0);
    expect(within(card).getByTestId("export-unreadable")).toBeInTheDocument();
    expect(within(card).queryByTestId("export-failed")).toBeNull();
  });

  it("is read again when a stage changes state or result, never on a progress tick", async () => {
    const { qc, card, calls } = writeUp(PREDICT);
    await within(card).findByTestId("export-tables");
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, progress: 0.5 } } }));
    await settle();
    expect(calls("export")).toBe(1);
    const before = calls("checklist");
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-other` } } }));
    await waitFor(() => expect(calls("export")).toBe(2));
    // the checklist, which says what the export waits for, is read again with it
    await waitFor(() => expect(calls("checklist")).toBeGreaterThan(before));
  });

  it("keeps the tables and figures it drew veiled and inert while it reads the export again, never removed", async () => {
    let answerNext: (r: Response) => void = () => {};
    let n = 0;
    const { qc, card } = writeUp(PREDICT, {
      exportAnswer: () => (++n === 1 ? zipAnswer(PREDICT)() : new Promise<Response>((resolve) => (answerNext = resolve))),
    });
    const drawn = await within(card).findByTestId("export-tables");
    const veilOf = () => within(card).getByTestId("export-veil");
    expect(veilOf()).toHaveAttribute("data-veil", "fresh");
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-other` } } }));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "recomputing"));
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
    await act(async () => answerNext(zipAnswer(PREDICT)()));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "fresh"));
    expect(within(card).getByTestId("export-tables").closest("[inert]")).toBeNull();
  });

  it("keeps the newest read drawn where an older read, for a record since left behind, answers after it", async () => {
    // The fit's result moves twice while the export is read: the read for the middle result
    // answers last, with another export's tables, so which read is drawn shows in its tables as
    // well as in its veil.
    const pending: ((r: Response) => void)[] = [];
    let n = 0;
    const { qc, card, calls } = writeUp(PREDICT, {
      exportAnswer: () => (++n === 1 ? zipAnswer(PREDICT)() : new Promise<Response>((resolve) => pending.push(resolve))),
    });
    await within(card).findByTestId("export-tables");
    const veilOf = () => within(card).getByTestId("export-veil");
    const files = () => within(card).getAllByTestId("export-table").map((s) => s.getAttribute("data-file"));
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-b` } } }));
    await waitFor(() => expect(calls("export")).toBe(2));
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-c` } } }));
    await waitFor(() => expect(calls("export")).toBe(3));
    const [middle, newest] = pending;
    await act(async () => newest!(zipAnswer(PREDICT)()));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "fresh"));
    expect(files()).toEqual(PREDICT.truth.tables.map((t) => t.file));
    // The read left behind answers now. Once read, it reads the checklist again, as every read
    // does, so that read marks it done.
    const before = calls("checklist");
    await act(async () => middle!(zipAnswer(ESTIMATE)()));
    await waitFor(() => expect(calls("checklist")).toBeGreaterThan(before));
    await settle();
    expect(veilOf()).toHaveAttribute("data-veil", "fresh");
    expect(files()).toEqual(PREDICT.truth.tables.map((t) => t.file));
    expect(within(card).getByTestId("export-tables").closest("[inert]")).toBeNull();
    expect(calls("export")).toBe(3);
  });

  it("keeps what it drew veiled \"stale\" while the export waits after a decision, unread, and live again once it is read", async () => {
    const { qc, card, calls, serve } = writeUp(PREDICT);
    const drawn = await within(card).findByTestId("export-tables");
    const veilOf = () => within(card).getByTestId("export-veil");
    // A decision is recorded and the checklist, read again, names what the export now waits for.
    serve(PREDICT.waiting.checklist);
    act(() => qc.setQueryData(keys.checklist(PID), PREDICT.waiting.checklist));
    const last = VIEW.decisions[VIEW.decisions.length - 1]!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, decisions: [...VIEW.decisions, { ...last, id: `${last.id}-next` }] }));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "stale"));
    await settle();
    // A decision was recorded: the veil says the stale state's own words, none of the export's.
    expect(EXPORT_TAGS).not.toContain(tagOf(card));
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
    expect(within(card).getAllByTestId("export-later")).toHaveLength(1);
    expect(calls("export")).toBe(1);
    // Once the checklist waits for nothing, the export is read for the record as it stands.
    serve(PREDICT.checklist);
    await act(() => qc.invalidateQueries({ queryKey: keys.checklist(PID), exact: true }));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "fresh"));
    expect(calls("export")).toBe(2);
    expect(within(card).queryByTestId("export-later")).toBeNull();
    expect(within(card).getByTestId("export-tables").closest("[inert]")).toBeNull();
  });

  it("keeps what it drew veiled \"recomputing\" while a stage computes and the export refuses, and live again once it is read", async () => {
    let n = 0;
    const { qc, card, calls, serve } = writeUp(PREDICT, {
      exportAnswer: () => (++n === 2 ? refusalAnswer(PREDICT)() : zipAnswer(PREDICT)()),
    });
    const drawn = await within(card).findByTestId("export-tables");
    const veilOf = () => within(card).getByTestId("export-veil");
    // The fit computes: the export, read again at once, refuses (as it was captured doing while a
    // result computes), and the checklist, read again, names what it waits for.
    serve(PREDICT.waiting.checklist);
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, status: "running" } } }));
    await waitFor(() => expect(within(card).getAllByTestId("export-later")).toHaveLength(1));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "recomputing"));
    expect(EXPORT_TAGS).not.toContain(tagOf(card));
    expect(calls("export")).toBe(2);
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
    // The fit is fresh again, with its new result: the export is read again and drawn live.
    serve(PREDICT.checklist);
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-new` } } }));
    await waitFor(() => expect(veilOf()).toHaveAttribute("data-veil", "fresh"));
    expect(calls("export")).toBe(3);
    expect(within(card).queryByTestId("export-later")).toBeNull();
    expect(within(card).getByTestId("export-tables").closest("[inert]")).toBeNull();
  });

  it("keeps what it drew veiled where the export, read again, refuses, with its refusal in its words beside it", async () => {
    let n = 0;
    const { qc, card } = writeUp(PREDICT, { exportAnswer: () => (++n === 1 ? zipAnswer(PREDICT)() : refusalAnswer(PREDICT)()) });
    const drawn = await within(card).findByTestId("export-tables");
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, key: `${fit.key}-other` } } }));
    const refused = await within(card).findByTestId("export-refused");
    expect(refused).toHaveTextContent(plain(PREDICT.waiting.refusal.error.message));
    // Nothing the view holds failed, stopped or computes, and no decision was recorded: the veil
    // says only what the export did, never an answer that changed or a recompute that did not finish.
    expect(within(card).getByTestId("export-veil")).toHaveAttribute("data-veil", "stale");
    expect(tagOf(card)).toBe(EXPORT_TAG.refused);
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
  });

  it("says only that the export now waits where a resync finds nothing moved, the export refusing and the checklist waiting, and its state's words once a decision is recorded", async () => {
    // Nothing in the view moves, and the export's gate now refuses and the checklist waits (as for
    // a data file changed on disk since it was read, core/export/gate.py's input_changed, which
    // records no decision). The stream reconnects and the project is read again, unawaited, as
    // events.ts does.
    let n = 0;
    const { qc, card, calls, serve } = writeUp(PREDICT, { exportAnswer: () => (++n === 1 ? zipAnswer(PREDICT)() : refusalAnswer(PREDICT)()) });
    const drawn = await within(card).findByTestId("export-tables");
    serve(PREDICT.waiting.checklist);
    void qc.invalidateQueries({ queryKey: keys.project(PID) });
    await waitFor(() => expect(calls("export")).toBe(2));
    await waitFor(() => expect(within(card).queryByTestId("export-later")).not.toBeNull());
    await waitFor(() => expect(tagOf(card)).toBe(EXPORT_TAG.waits));
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
    // A decision is recorded: now the view shows a cause, and the veil says the stale state's own words.
    const last = VIEW.decisions[VIEW.decisions.length - 1]!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, decisions: [...VIEW.decisions, { ...last, id: `${last.id}-next` }] }));
    await waitFor(() => expect(EXPORT_TAGS).not.toContain(tagOf(card)));
    expect(within(card).getByTestId("export-veil")).toHaveAttribute("data-veil", "stale");
    expect(drawn.closest("[inert]")).not.toBeNull();
  });

  it("says only that the export refused where a resync finds nothing moved and the export, read again, refuses", async () => {
    let n = 0;
    const { qc, card, calls } = writeUp(PREDICT, { exportAnswer: () => (++n === 1 ? zipAnswer(PREDICT)() : refusalAnswer(PREDICT)()) });
    const drawn = await within(card).findByTestId("export-tables");
    void qc.invalidateQueries({ queryKey: keys.project(PID) });
    await waitFor(() => expect(calls("export")).toBe(2));
    const refused = await within(card).findByTestId("export-refused");
    expect(refused).toHaveTextContent(plain(PREDICT.waiting.refusal.error.message));
    await waitFor(() => expect(tagOf(card)).toBe(EXPORT_TAG.refused));
    expect(within(card).getByTestId("export-tables")).toBe(drawn);
    expect(drawn.closest("[inert]")).not.toBeNull();
    // A stage computes: the veil says so, in its state's words.
    const fit = VIEW.stages.fit!;
    act(() => qc.setQueryData(keys.view(PID), { ...VIEW, stages: { ...VIEW.stages, fit: { ...fit, status: "running" } } }));
    await waitFor(() => expect(within(card).getByTestId("export-veil")).toHaveAttribute("data-veil", "recomputing"));
    expect(EXPORT_TAGS).not.toContain(tagOf(card));
  });
});

/**
 * TRIPOD+AI's scope on a checklist item in plain words, as the source's own legend defines each
 * code (core/export/data/tripod_ai.json, Table 2's footnote, read here with node:fs), below the
 * heading that keeps the code as the bundle writes it. A code the legend does not define, or a
 * scope on another checklist's item, is given no words. The small-fields package said these on the
 * stage card's checklist, which this card replaced at integration. The reports are the captured
 * checklist with items built in the served shape.
 */
describe("Write-up says TRIPOD+AI's scope in plain words", () => {
  const TRIPOD = JSON.parse(
    readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../../core/export/data/tripod_ai.json"), "utf-8"),
  ) as { footnote: string; items: { id: string; scope?: string | null }[] };
  /** The legend, as Table 2's footnote gives it: "*D=…; E=…; D;E=…." */
  const LEGEND = (() => {
    const m = /\*D=([^;]+); E=([^;]+); D;E=([^.]+)\./.exec(TRIPOD.footnote);
    if (!m) throw new Error("TRIPOD+AI's footnote holds no D, E and D;E legend");
    return { D: m[1]!, E: m[2]!, "D;E": m[3]! };
  })();
  const item = (over: Partial<ChecklistItem> & Pick<ChecklistItem, "id">): ChecklistItem => ({
    section: "Methods",
    topic: "Analytical methods",
    text: "The item, quoted from its source.",
    kind: null,
    scope: null,
    status: "partly answered",
    where: [],
    owed: "the thing owed",
    note: "partly answered — the author must supply the thing owed",
    ...over,
  });
  const reportOf = (checklist: ChecklistReport["checklist"], items: ChecklistItem[]): ChecklistReport => ({
    ...PREDICT.checklist,
    checklist,
    items,
    counts: { items: items.length, answered: 0, partly_answered: items.length, unanswered: 0 },
    unanswered: [],
    waiting: [],
  });
  const listOf = (report: ChecklistReport) => within(writeUp(PREDICT, { checklist: report }).card).getByTestId("checklist-items");

  it("says each code as the source's legend defines it, the code kept in the heading", () => {
    // The legend: D only development, E only (solely) evaluation, D;E both.
    expect(LEGEND.D).toMatch(/\bonly\b/);
    expect(LEGEND.D).toMatch(/development/);
    expect(LEGEND.D).not.toMatch(/evaluation/);
    expect(LEGEND.E).toMatch(/\bsolely\b/);
    expect(LEGEND.E).toMatch(/evaluation/);
    expect(LEGEND.E).not.toMatch(/development/);
    expect(LEGEND["D;E"]).toMatch(/\bboth\b/);
    expect(LEGEND["D;E"]).toMatch(/development and evaluation/);
    // Every scope the source gives an item is a code the legend defines.
    expect([...new Set(TRIPOD.items.map((i) => i.scope))].sort()).toEqual(Object.keys(LEGEND).sort());
    const items = Object.keys(LEGEND).map((scope, n) => item({ id: `${n + 1}a`, scope }));
    const list = listOf(reportOf("TRIPOD+AI", items));
    const words = (id: string) => within(list).getByTestId(`checklist-scope-${id}`).textContent ?? "";
    const [d, e, both] = items as [ChecklistItem, ChecklistItem, ChecklistItem];
    expect([d.scope, e.scope, both.scope]).toEqual(["D", "E", "D;E"]);
    expect(words(d.id)).toMatch(/\bonly\b/i);
    expect(words(d.id)).toMatch(/develop/i);
    expect(words(d.id)).not.toMatch(/evaluat/i);
    expect(words(e.id)).toMatch(/\bonly\b/i);
    expect(words(e.id)).toMatch(/evaluat/i);
    expect(words(e.id)).not.toMatch(/develop/i);
    expect(words(both.id)).toMatch(/develop/i);
    expect(words(both.id)).toMatch(/evaluat/i);
    expect(words(both.id)).not.toMatch(/\bonly\b/i);
    const headings = within(list).getAllByTestId("checklist-item").map((el) => within(el).getByRole("heading").textContent);
    expect(headings).toEqual(items.map((i) => `${i.id} (${i.scope}) · ${i.topic}`));
  });

  it("says no words for a code the legend does not define", () => {
    const odd = item({ id: "5x", scope: "X" });
    expect(Object.keys(LEGEND)).not.toContain(odd.scope);
    const list = listOf(reportOf("TRIPOD+AI", [odd]));
    expect(within(list).getAllByTestId("checklist-item")).toHaveLength(1);
    expect(within(list).queryByTestId(`checklist-scope-${odd.id}`)).toBeNull();
  });

  it("says no words for a scope on another checklist's item", () => {
    const strobe = item({ id: "nut-6", kind: "STROBE-nut", topic: "Participants", scope: "D" });
    const list = listOf(reportOf("STROBE-nut", [strobe]));
    expect(within(list).getAllByTestId("checklist-item")).toHaveLength(1);
    expect(within(list).queryByTestId(`checklist-scope-${strobe.id}`)).toBeNull();
  });
});
