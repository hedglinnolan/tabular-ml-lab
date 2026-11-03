/**
 * The data-assembly cards (the front-end audit's G07): joining another file and the data
 * dictionary, each through DECLARATION_CARDS.
 *
 * On the base neither has a card: each falls to the fallback, which records nothing, so the shell
 * lists both as "Not available in this version yet" and a researcher with NHANES's component files
 * or a data dictionary cannot join or import them in the shell, though the client's functions
 * exist (api/queries.ts `useFiles`, `useAddFile`, `useJoinPreview`, `useCodebook`). Here each
 * records.
 *
 * The truth each card is held to is the server's: the assembly journey a real server answered
 * (src/mocks/fixtures/m3-assembly.json: DEMO_J.XPT read, DR1TOT_J.XPT added and joined on SEQN,
 * an NHANES codebook page imported), with its files, its join preview, its codebook previews, the
 * decisions it posted and the views they led to; the quest lines from the engine's contract
 * (src/quest/fixtures/quest-contract.json); and every posted decision held to its model in
 * turbotab/server/openapi.json. A payload the capture holds no case of (a refused join, a field the
 * values contradict, a press the server refuses, a file or codebook the server could not read, a
 * table not read yet, or whose read failed or was stopped) is built in the served shape, its words
 * the test's own construction.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, ApiError, RefusalError } from "../../../api/client";
import type {
  AddedFile,
  CodebookPreview,
  JoinPreview,
  JoinPreviewRequest,
} from "../../../api/m3-types";
import { keys } from "../../../api/queries";
import type { Decision, ProjectView, QuestLine, StageResult, StageStatus } from "../../../api/schema";
import { artifactAt, expandView, viewsOf, type M3Fixture } from "../../../mocks/m3";
import { StageFocusProvider } from "../../../state/focus";
import { CONTRACT } from "../../fixtures/contract";
import { openLine } from "../../log";
import { DECLARATION_CARDS } from "../predict";
import { recordable, rendererOf } from "../renderers";

const PID = "assembly-test";
const AT = "2026-11-03T00:00:00Z";
const KINDS = ["join_files", "import_codebook"] as const;
type Kind = (typeof KINDS)[number];

// ── the server's contract (openapi.json) ─────────────────────────────────────

interface JsonSchema {
  type?: string;
  enum?: unknown[];
  const?: unknown;
  pattern?: string;
  minLength?: number;
  default?: unknown;
  anyOf?: JsonSchema[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean | JsonSchema;
}
const OPENAPI = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../../../server/openapi.json",
);
const SCHEMAS = (
  JSON.parse(readFileSync(OPENAPI, "utf-8")) as {
    components: { schemas: Record<string, JsonSchema> };
  }
).components.schemas;
/** The model a posted decision is parsed as (the request body's variant). */
const MODEL: Record<Kind, JsonSchema> = {
  join_files: SCHEMAS["JoinFiles-Input"]!,
  import_codebook: SCHEMAS["ImportCodebook-Input"]!,
};

/** What keeps a posted decision's top-level fields from parsing as `model`: one line per problem. */
function problems(model: JsonSchema, value: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const k of model.required ?? []) if (!(k in value)) out.push(`${k}: missing`);
  for (const [k, v] of Object.entries(value)) {
    const p = model.properties?.[k];
    if (!p) {
      out.push(`${k}: not a field of the model`);
      continue;
    }
    const variants = p.anyOf ?? [p];
    const fits = variants.some(
      (s) =>
        (s.type === "null" ? v === null : s.type === "string" ? typeof v === "string" : true) &&
        (s.const === undefined || v === s.const) &&
        (s.enum === undefined || s.enum.includes(v)) &&
        (s.pattern === undefined || (typeof v === "string" && new RegExp(s.pattern).test(v))) &&
        (s.minLength === undefined || (typeof v === "string" && v.length >= s.minLength)),
    );
    if (!fits) out.push(`${k}: ${JSON.stringify(v)} does not fit`);
  }
  return out;
}

/** Which rows the joined table keeps: the model's values, and the one it takes when none is sent. */
const HOW = MODEL.join_files.properties!.how!;

// ── the captured server (src/mocks/fixtures/m3-assembly.json) ────────────────

const F = Object.values(
  import.meta.glob<M3Fixture>("../../../mocks/fixtures/m3-assembly.json", {
    eager: true,
    import: "default",
  }),
)[0]!;
const served = <T,>(name: string): T => structuredClone(F.endpoints[name]!.body) as T;
const VIEWS = viewsOf(F);
const viewAt = (i: number): ProjectView => expandView(F, VIEWS[i]!, PID, AT);
/** The capture's accepted press of `kind`: the decision as posted, the snapshot it was posted from
 *  and the one it led to. */
const pressOf = (kind: Kind) => {
  const r = F.records.find((x) => x.decision.kind === kind && x.status === 200)!;
  return { decision: r.decision as unknown as Decision, from: r.from, to: r.to! };
};
const FILES = served<AddedFile[]>("files_list");
const FILE = FILES[0]!;
const PREVIEW = served<JoinPreview>("join_preview");
const BOOK = served<CodebookPreview>("codebook_upload");

/** The engine's own line for `kind`, from Your data in the contract's quest log. */
function lineOf(kind: Kind, over: Partial<QuestLine> = {}): QuestLine {
  const data = CONTRACT.logs.models_open.log.stages.find((s) => s.key === "data")!;
  return { ...data.lines.find((l) => l.key === kind)!, ...over };
}

// ── mounting ────────────────────────────────────────────────────────────────

/** How the table's reading (the ingest stage) stands where it is not fresh at the view's key. */
type Ingest = Partial<StageStatus> & { status: StageStatus["status"] };

/**
 * The table as read, standing `over` at a key of its own while the server serves the older result
 * it holds (`older`), or none: a first read that failed or was stopped leaves none (core/graph.py
 * `Engine.get`). "loading": read, its result not fetched yet.
 */
type NotRead = { over: Ingest; older: boolean } | "loading";
const NEXT_KEY = "ingest-next";

function mount(
  kind: Kind,
  opts: {
    at?: number;
    mode?: "local" | "server";
    line?: Partial<QuestLine>;
    next?: ProjectView;
    ingest?: NotRead;
    /** The files added to the project, as the server lists them. */
    files?: AddedFile[];
  } = {},
) {
  const at = viewAt(opts.at ?? 1);
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  const ingest = at.stages.ingest!.key;
  const read = artifactAt(F, "ingest", ingest);
  const not = typeof opts.ingest === "object" ? opts.ingest : null;
  const view: ProjectView = not
    ? {
        ...at,
        stages: {
          ...at.stages,
          ingest: {
            ...at.stages.ingest!,
            key: NEXT_KEY,
            fresh: false,
            error: null,
            cancelled: false,
            ...not.over,
          },
        },
      }
    : at;
  qc.setQueryData(keys.view(PID), view);
  qc.setQueryData(keys.health(), {
    version: "test",
    mode: opts.mode ?? "local",
    workers: 2,
    user: null,
    auth: "none",
  });
  if (not)
    qc.setQueryData(keys.stage(PID, "ingest"), {
      stage: "ingest",
      key: not.older ? ingest : null,
      fresh: false,
      status: not.over.status,
      artifact: not.older ? read : null,
    });
  else if (opts.ingest !== "loading")
    qc.setQueryData(keys.stage(PID, "ingest"), {
      stage: "ingest",
      key: ingest,
      fresh: true,
      status: "fresh",
      artifact: read,
    });
  qc.setQueryData(keys.columns(PID), served("columns"));
  const files = opts.files ?? FILES;
  qc.setQueryData(keys.files(PID), files);
  // Nothing else is fetched: the stage results and previews these cards do not read never answer.
  vi.spyOn(api, "stage").mockImplementation(() => new Promise(() => {}));
  vi.spyOn(api, "preview").mockImplementation(() => new Promise(() => {}));
  vi.spyOn(api, "files").mockResolvedValue(files);
  const decide = vi.spyOn(api, "decide").mockResolvedValue(opts.next ?? view);
  const Card = DECLARATION_CARDS[kind];
  if (!Card) throw new Error(`no card for ${kind}`);
  render(
    <QueryClientProvider client={qc}>
      <StageFocusProvider>
        <Card pid={PID} line={lineOf(kind, opts.line)} />
      </StageFocusProvider>
    </QueryClientProvider>,
  );
  return { decide, qc };
}

/**
 * The table as the server reads it now: the view's ingest standing `over` at `key`, its event
 * patching the view, and once fresh its result at that key (the same columns: the test's own
 * construction of a table read anew).
 */
function readAs(qc: QueryClient, over: Ingest & { key: string }) {
  const view = qc.getQueryData<ProjectView>(keys.view(PID))!;
  const fresh = over.status === "fresh";
  qc.setQueryData(keys.view(PID), {
    ...view,
    stages: { ...view.stages, ingest: { ...view.stages.ingest!, fresh, ...over } },
  });
  if (fresh) {
    const result = qc.getQueryData<StageResult>(keys.stage(PID, "ingest"))!;
    qc.setQueryData(keys.stage(PID, "ingest"), { ...result, key: over.key, fresh, status: "fresh" });
  }
}

const settle = () => act(() => new Promise((r) => setTimeout(r, 20)));
const said = (text: string) => text.replaceAll("`", "");
const fmt = (n: number) => n.toLocaleString("en-US");
const record = () => screen.getByTestId("card-record");

function deferred<T>() {
  let done!: (value: T) => void;
  const promise = new Promise<T>((r) => (done = r));
  return { promise, done };
}

/** Choose the added file, and the identifier in the table through its column picker. */
function chooseJoin(file: string, on: string) {
  fireEvent.change(screen.getByTestId("field-file"), { target: { value: file } });
  // The picker's search box (each select is a combobox too).
  const box = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
  fireEvent.change(box, { target: { value: on } });
  fireEvent.keyDown(box, { key: "Enter" });
}

/** The counts table's cells, row by row, without each row's own label. */
const countsOf = (id: string) =>
  Array.from(screen.getByTestId(id).querySelectorAll("tbody tr")).map((tr) =>
    Array.from(tr.querySelectorAll("td")).map((td) => td.textContent),
  );

const upload = (testId: string, name: string) => {
  const file = new File(["x"], name);
  fireEvent.change(screen.getByTestId(testId), { target: { files: [file] } });
  return file;
};

// The server's answer to a count or a codebook read asked before the table is read (server/
// service.py `_table_ready`: 409 not_yet, whatever the read's status), in its served shape.
const notYet = () =>
  new RefusalError({
    error: {
      code: "not_yet",
      message: "The table is still being read; try again once it is ready.",
      exits: [],
    },
  });

/** How the table's reading stands, for each way it can stand other than read (`mount`'s ingest). */
const NOT_READ: Record<string, NotRead> = {
  "first read failed": {
    over: { status: "error", error: "ValueError: the test's file is not a SAS transport file" },
    older: false,
  },
  "first read stopped": { over: { status: "idle", cancelled: true }, older: false },
  "first read queued": { over: { status: "queued" }, older: false },
  "first read running": { over: { status: "running" }, older: false },
  "first read not started": { over: { status: "idle" }, older: false },
  "read, its columns not fetched yet": "loading",
  "read anew, running": { over: { status: "running" }, older: true },
  "read anew, failed": { over: { status: "error", error: "the read ran out of memory" }, older: true },
};

afterEach(() => {
  vi.restoreAllMocks();
});

// ── the shell ───────────────────────────────────────────────────────────────

describe("the data-assembly lines the shell can answer (G07)", () => {
  it("draws joining another file and the data dictionary through a card that records", () => {
    for (const key of KINDS) {
      expect(rendererOf({ source: "declaration", key }, DECLARATION_CARDS), key).toBe(
        "declaration",
      );
      expect(recordable({ source: "declaration", key }, DECLARATION_CARDS), key).toBe(true);
    }
  });

  it("opens the join in Your data once its counted lines are answered, where nothing could open before", () => {
    // The engine's Your data: the lens and the roles answered, the join and the dictionary open
    // and uncounted (the contract's quest log).
    const data = CONTRACT.logs.models_open.log.stages.find((s) => s.key === "data")!;
    expect(data.lines.filter((l) => l.counted).every((l) => l.status === "answered")).toBe(true);
    expect(openLine(data, (l) => recordable(l, DECLARATION_CARDS))?.key).toBe("join_files");
  });
});

// ── joining another file (join_files) ───────────────────────────────────────

describe("joining another file (join_files)", () => {
  it("offers each added file not yet joined, and names one joined already as joined, by the state's joins", () => {
    const press = pressOf("join_files");
    mount("join_files", { at: press.from });
    const options = Array.from(screen.getByTestId("field-file").querySelectorAll("option")).map(
      (o) => o.value,
    );
    expect(options).toContain(FILE.id);
    expect(screen.queryByTestId("joined")).toBeNull();
  });

  it("lists the file the state holds as joined, and does not offer it again", () => {
    const press = pressOf("join_files");
    const after = viewAt(press.to);
    const spec = after.state.joins![FILE.id]!;
    mount("join_files", { at: press.to });
    expect(screen.getByTestId("joined")).toHaveTextContent(spec.name);
    expect(screen.queryByTestId("field-file")).toBeNull();
  });

  it("counts the join chosen as the server serves it, before anything is recorded", async () => {
    const press = pressOf("join_files");
    const { decide } = mount("join_files", { at: press.from });
    const ask = vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(1));
    expect(ask.mock.calls[0]).toEqual([PID, { file: FILE.id, on: PREVIEW.on, how: HOW.default }]);
    await screen.findByTestId("join-counts");
    expect(screen.getByTestId("join-counts").querySelector("thead")).toHaveTextContent(
      PREVIEW.file_side.name,
    );
    expect(countsOf("join-counts")).toEqual([
      [fmt(PREVIEW.table.rows), fmt(PREVIEW.file_side.rows)],
      [fmt(PREVIEW.table.keys), fmt(PREVIEW.file_side.keys)],
      [fmt(PREVIEW.table_unmatched), fmt(PREVIEW.file_unmatched)],
    ]);
    expect(screen.getByTestId("join-sentence").textContent).toBe(said(PREVIEW.sentence));
    expect(screen.queryByTestId("join-refused")).toBeNull();
    expect(decide).not.toHaveBeenCalled();
  });

  it("counts the rows with a blank identifier only where a side has any", async () => {
    mount("join_files");
    const blanks: JoinPreview = { ...PREVIEW, table: { ...PREVIEW.table, blank_keys: 4 } };
    vi.spyOn(api, "joinPreview").mockResolvedValue(blanks);
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    expect(countsOf("join-counts")).toEqual([
      [fmt(PREVIEW.table.rows), fmt(PREVIEW.file_side.rows)],
      [fmt(PREVIEW.table.keys), fmt(PREVIEW.file_side.keys)],
      [fmt(4), fmt(PREVIEW.file_side.blank_keys)],
      [fmt(PREVIEW.table_unmatched), fmt(PREVIEW.file_unmatched)],
    ]);
  });

  it("records the join exactly as the real server took it: the file, the identifier and the rows kept, never the name or the counts", async () => {
    const press = pressOf("join_files");
    const { decide } = mount("join_files", { at: press.from, next: viewAt(press.to) });
    vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    expect(record()).toBeEnabled();
    expect(record()).toHaveTextContent(FILE.name);
    fireEvent.click(record());
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    const sent = decide.mock.calls[0]![1] as Record<string, unknown>;
    expect(sent).toEqual(press.decision);
    expect(problems(MODEL.join_files, sent)).toEqual([]);
    expect(sent).not.toHaveProperty("name");
    expect(sent).not.toHaveProperty("counts");
  });

  it("offers which rows to keep as the model's own values, the one it takes unsent chosen", () => {
    mount("join_files");
    vi.spyOn(api, "joinPreview").mockImplementation(() => new Promise(() => {}));
    chooseJoin(FILE.id, PREVIEW.on);
    const how = screen.getByTestId("field-how") as HTMLSelectElement;
    expect(Array.from(how.options).map((o) => o.value)).toEqual(HOW.enum);
    expect(how.value).toBe(HOW.default);
  });

  it("shows no count and records nothing until the preview answers the choice shown", async () => {
    mount("join_files");
    const first = deferred<JoinPreview>();
    const second = deferred<JoinPreview>();
    const ask = vi
      .spyOn(api, "joinPreview")
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);
    chooseJoin(FILE.id, PREVIEW.on);
    // Counting: nothing to show, nothing to record.
    expect(await screen.findByText("Counting the rows the join would give…")).toBeInTheDocument();
    expect(screen.queryByTestId("join-counts")).toBeNull();
    expect(record()).toBeDisabled();
    await act(async () => first.done(PREVIEW));
    await screen.findByTestId("join-counts");
    expect(record()).toBeEnabled();
    // The rows kept change: the counts shown were another join's, so they go until this one's come.
    const inner = HOW.enum!.find((v) => v !== HOW.default) as JoinPreviewRequest["how"];
    fireEvent.change(screen.getByTestId("field-how"), { target: { value: inner } });
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    expect(ask.mock.calls[1]![1]).toMatchObject({ how: inner });
    expect(screen.queryByTestId("join-counts")).toBeNull();
    expect(record()).toBeDisabled();
    const keptInner: JoinPreview = {
      ...PREVIEW,
      how: inner,
      rows: PREVIEW.table.rows - PREVIEW.table_unmatched,
    };
    await act(async () => second.done(keptInner));
    await screen.findByTestId("join-counts");
    expect(record()).toBeEnabled();
  });

  it("does not show a preview served for another identifier than the one chosen", async () => {
    mount("join_files");
    const other = FILE.columns.find((c) => c !== PREVIEW.on)!;
    const ask = vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    fireEvent.change(screen.getByTestId("field-right_on"), { target: { value: other } });
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    expect(ask.mock.calls[1]![1]).toEqual({
      file: FILE.id,
      on: PREVIEW.on,
      right_on: other,
      how: HOW.default,
    });
    // The server answered with its earlier identifier (right_on SEQN): not the join chosen.
    await settle();
    expect(screen.queryByTestId("join-counts")).toBeNull();
    expect(record()).toBeDisabled();
  });

  // A preview that answered another join than the one asked, one part at a time: the other file the
  // capture added (its upload), another column of the table, the other rows kept. The silent side,
  // the preview of the join asked, is "counts the join chosen as the server serves it" above.
  const elsewhere: [string, () => Partial<JoinPreview>][] = [
    ["another file", () => ({ file: served<AddedFile>("files_upload").id })],
    [
      "another identifier in the table",
      () => ({
        on: served<{ name: string }[]>("columns").find((c) => c.name !== PREVIEW.on)!.name,
      }),
    ],
    [
      "the other rows kept",
      () => ({ how: HOW.enum!.find((v) => v !== HOW.default) as JoinPreviewRequest["how"] }),
    ],
  ];
  it.each(elsewhere)(
    "does not show a preview served for %s than the one chosen",
    async (_, part) => {
      const { decide } = mount("join_files");
      const other: JoinPreview = { ...PREVIEW, ...part() };
      // Each case differs from the join asked in its one part, and in no other.
      expect(
        Object.keys(other).filter(
          (k) => other[k as keyof JoinPreview] !== PREVIEW[k as keyof JoinPreview],
        ),
      ).toHaveLength(1);
      const ask = vi.spyOn(api, "joinPreview").mockResolvedValue(other);
      chooseJoin(FILE.id, PREVIEW.on);
      await waitFor(() => expect(ask).toHaveBeenCalledTimes(1));
      expect(ask.mock.calls[0]).toEqual([PID, { file: FILE.id, on: PREVIEW.on, how: HOW.default }]);
      await settle();
      // The server has answered: what is missing was refused by the card, not still being counted.
      expect(screen.queryByText("Counting the rows the join would give…")).toBeNull();
      expect(screen.queryByTestId("join-counts")).toBeNull();
      expect(screen.queryByTestId("join-sentence")).toBeNull();
      expect(record()).toBeDisabled();
      fireEvent.click(record());
      await settle();
      expect(decide).not.toHaveBeenCalled();
    },
  );

  it("posts the file's own name for the identifier only where it is named otherwise", async () => {
    const { decide } = mount("join_files");
    const other = FILE.columns.find((c) => c !== PREVIEW.on)!;
    vi.spyOn(api, "joinPreview").mockImplementation(async (_pid, body) => ({
      ...PREVIEW,
      right_on: body.right_on ?? body.on,
    }));
    chooseJoin(FILE.id, PREVIEW.on);
    fireEvent.change(screen.getByTestId("field-right_on"), { target: { value: other } });
    await screen.findByTestId("join-counts");
    fireEvent.click(record());
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    const sent = decide.mock.calls[0]![1] as Record<string, unknown>;
    expect(sent).toEqual({
      kind: "join_files",
      file: FILE.id,
      on: PREVIEW.on,
      right_on: other,
      how: HOW.default,
    });
    expect(problems(MODEL.join_files, sent)).toEqual([]);
  });

  it("says a join the server would refuse in its own words, with its ways forward, and records nothing", async () => {
    const { decide } = mount("join_files");
    const exits = [
      { label: "Join on a column that names one row per unit in one of the files", decision: null },
    ];
    const refused: JoinPreview = {
      ...PREVIEW,
      relation: "refused",
      matched_keys: 0,
      table_unmatched: 0,
      file_unmatched: 0,
      rows: 0,
      added_columns: [],
      sentence: "",
      refusal: { code: "many_to_many", message: "`SEQN` repeats in both files.", exits },
    };
    vi.spyOn(api, "joinPreview").mockResolvedValue(refused);
    chooseJoin(FILE.id, PREVIEW.on);
    const note = await screen.findByTestId("join-refused");
    expect(note).toHaveTextContent(said(refused.refusal!.message));
    expect(note).toHaveTextContent(exits[0]!.label);
    expect(screen.queryByTestId("join-counts")).toBeNull();
    expect(screen.queryByTestId("join-sentence")).toBeNull();
    expect(record()).toBeDisabled();
    fireEvent.click(record());
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });

  it("says a press the server refuses in the engine's words, with its exits", async () => {
    const { decide } = mount("join_files");
    vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    const refusal = {
      error: {
        code: "sealed",
        message: "A join changes what a row is.",
        exits: [{ label: "Withdraw the seal first", decision: null }],
      },
    };
    decide.mockRejectedValue(new RefusalError(refusal));
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    fireEvent.click(record());
    const note = await screen.findByTestId("refusal");
    expect(note).toHaveTextContent(refusal.error.message);
    expect(within(note).getByTestId("refusal-exit")).toHaveTextContent(
      refusal.error.exits[0]!.label,
    );
  });

  // The server's refusal of a file it could not add, in the shape it serves (server/service.py
  // `ApiError`: 400 unreadable_file and the like), its words the test's own.
  const unreadable = (name: string) => {
    const message = `${name} could not be read: the test's file holds no table`;
    return new ApiError(400, message, { error: { code: "unreadable_file", message } });
  };

  it("says why the server could not add a file, and says it no longer once a file is added", async () => {
    mount("join_files", { mode: "server" });
    const added = served<AddedFile>("files_upload");
    const refused = unreadable(added.name);
    const send = vi
      .spyOn(api, "uploadFile")
      .mockRejectedValueOnce(refused)
      .mockResolvedValueOnce(added);
    upload("join-source-input", added.name);
    expect(await screen.findByTestId("join-source-error")).toHaveTextContent(refused.message);
    expect(screen.getByTestId("field-file")).toHaveValue("");
    expect(record()).toBeDisabled();
    upload("join-source-input", added.name);
    await waitFor(() => expect(send).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByTestId("field-file")).toHaveValue(added.id));
    expect(screen.queryByTestId("join-source-error")).toBeNull();
  });

  it("says no refusal where the file is added", async () => {
    mount("join_files", { mode: "server" });
    const added = served<AddedFile>("files_upload");
    vi.spyOn(api, "uploadFile").mockResolvedValue(added);
    upload("join-source-input", added.name);
    await waitFor(() => expect(screen.getByTestId("field-file")).toHaveValue(added.id));
    expect(screen.queryByTestId("join-source-error")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("says why the server could not count the join, and shows no count and records nothing", async () => {
    const { decide } = mount("join_files");
    // The server's answer to a preview of a file it does not hold (server/service.py
    // `join_preview`: 404 unknown_file), in its served shape; its words the test's own.
    const message = `No file ${FILE.id} is added to this project.`;
    const refused = new ApiError(404, message, { error: { code: "unknown_file", message } });
    vi.spyOn(api, "joinPreview").mockRejectedValue(refused);
    chooseJoin(FILE.id, PREVIEW.on);
    expect(await screen.findByTestId("join-preview-error")).toHaveTextContent(message);
    expect(screen.queryByTestId("join-counts")).toBeNull();
    expect(screen.queryByTestId("join-sentence")).toBeNull();
    expect(record()).toBeDisabled();
    fireEvent.click(record());
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });

  it("says no refusal of the count where the server counted the join", async () => {
    mount("join_files");
    vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    expect(screen.queryByTestId("join-preview-error")).toBeNull();
  });

  it("says what the server noticed reading the file chosen, and nothing where it noticed nothing", async () => {
    // The capture's file was read with nothing to say; a note in the served shape (core/datastore.py
    // `ingest`'s warnings), its words the test's own.
    expect(FILE.warnings).toEqual([]);
    const noted: AddedFile = { ...FILE, warnings: ["the test's file was read as Latin-1"] };
    mount("join_files");
    fireEvent.change(screen.getByTestId("field-file"), { target: { value: FILE.id } });
    expect(screen.getByTestId("field-right_on")).toBeInTheDocument();
    expect(screen.queryByTestId("join-file-warnings")).toBeNull();
    cleanup();
    mount("join_files", { files: [noted] });
    fireEvent.change(screen.getByTestId("field-file"), { target: { value: FILE.id } });
    const notes = screen.getByTestId("join-file-warnings");
    expect(Array.from(notes.querySelectorAll("li")).map((li) => li.textContent)).toEqual(
      noted.warnings,
    );
  });

  // Each option of the rows kept is named as the sentence the server records for it says it
  // (core/assembly.py `join_sentence`: "keeping every row of the table" for left, "keeping only
  // the rows found in both" for inner). The left join's is the captured server's own sentence; the
  // capture holds no inner join, so its sentence is built in the served shape from those words.
  const keeping = (label: string) => label.replace(/^Keep /, "keeping ");
  const INNER_KEEP = "keeping only the rows found in both";
  it("names the rows kept as the server's sentence for each says it", async () => {
    expect(PREVIEW.how).toBe("left");
    const inner: JoinPreview = {
      ...PREVIEW,
      how: "inner",
      rows: PREVIEW.table.rows - PREVIEW.table_unmatched,
      sentence: `\`${PREVIEW.file_side.name}\` was joined to the table on \`${PREVIEW.on}\`, ${INNER_KEEP}`,
    };
    mount("join_files");
    vi.spyOn(api, "joinPreview").mockImplementation(async (_pid, body) =>
      body.how === "inner" ? inner : PREVIEW,
    );
    chooseJoin(FILE.id, PREVIEW.on);
    const how = screen.getByTestId("field-how") as HTMLSelectElement;
    const labels = new Map(Array.from(how.options).map((o) => [o.value, o.textContent!]));
    for (const p of [PREVIEW, inner]) {
      fireEvent.change(how, { target: { value: p.how } });
      await waitFor(() =>
        expect(screen.getByTestId("join-sentence").textContent).toBe(said(p.sentence)),
      );
      const label = labels.get(p.how)!;
      expect(how.selectedOptions[0]!.textContent).toBe(label);
      expect(said(p.sentence)).toContain(keeping(label));
      // And not the other option's words.
      for (const [value, other] of labels)
        if (value !== p.how) expect(said(p.sentence)).not.toContain(keeping(other));
    }
  });

  it("adds a file by upload and chooses the file added", async () => {
    mount("join_files", { mode: "server" });
    const added = served<AddedFile>("files_upload");
    const send = vi.spyOn(api, "uploadFile").mockResolvedValue(added);
    const file = upload("join-source-input", added.name);
    await waitFor(() => expect(send).toHaveBeenCalledWith(PID, file));
    await waitFor(() => expect(screen.getByTestId("field-file")).toHaveValue(added.id));
  });

  it("adds a file by its path in local mode only", async () => {
    mount("join_files", { mode: "local" });
    const send = vi.spyOn(api, "addFile").mockResolvedValue(served<AddedFile>("files_add"));
    const path = "/data/nhanes/DR1TOT_J.XPT";
    fireEvent.change(screen.getByTestId("join-source-path"), { target: { value: path } });
    fireEvent.submit(screen.getByTestId("join-source-path").closest("form")!);
    await waitFor(() => expect(send).toHaveBeenCalledWith(PID, path));
  });

  it("offers no path in server mode, where the server reads only an upload", () => {
    mount("join_files", { mode: "server" });
    expect(screen.queryByTestId("join-source-path")).toBeNull();
    expect(screen.getByTestId("join-source-input")).toBeInTheDocument();
  });
});

// ── the table as read, where the join's identifier is chosen ────────────────

describe("the table the join reads (the ingest stage)", () => {
  const card = () => screen.getByTestId("declaration-join_files");
  const title = () => within(card()).getByRole("heading").textContent!;
  const chooseFile = () =>
    fireEvent.change(screen.getByTestId("field-file"), { target: { value: FILE.id } });
  const STILL = "The table is still being read; its columns appear here when it is done.";

  it("names the reading that did not finish, its reason and its retry where the table's first read failed, never 'still being read'", async () => {
    const reason = "the test's file is not a SAS transport file";
    mount("join_files", {
      ingest: { over: { status: "error", error: `ValueError: ${reason}` }, older: false },
    });
    chooseFile();
    const failure = screen.getByTestId("join-ingest-failure");
    expect(failure).toHaveAttribute("data-stage", "ingest");
    expect(failure).toHaveAttribute("data-kind", "failed");
    expect(failure).toHaveTextContent(reason);
    expect(failure).toHaveTextContent(`${title()} waits on it.`);
    expect(card()).not.toHaveTextContent("still being read");
    expect(card()).not.toHaveTextContent("You stopped");
    expect(screen.queryByTestId("picker-count")).toBeNull();
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(within(failure).getByTestId("retry-ingest"));
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "ingest"));
  });

  it("says the reading was stopped, with a way to run it again, where the table's first read was stopped, never 'still being read'", async () => {
    // The ingest waits on no stage (core/stages `ingest`: deps none), so its stop is its own.
    mount("join_files", { ingest: { over: { status: "idle", cancelled: true }, older: false } });
    chooseFile();
    const stopped = screen.getByTestId("join-ingest-failure");
    expect(stopped).toHaveAttribute("data-stage", "ingest");
    expect(stopped).toHaveAttribute("data-kind", "stopped");
    expect(stopped).toHaveTextContent("You stopped");
    expect(stopped).toHaveTextContent(`${title()} waits on it.`);
    expect(card()).not.toHaveTextContent("still being read");
    expect(screen.queryByTestId("picker-count")).toBeNull();
    const retry = within(stopped).getByTestId("retry-ingest");
    expect(retry).toHaveTextContent("Recompute");
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "ingest"));
  });

  it.each(["queued", "running"] as const)(
    "says the table is still being read, and names no failure, while its first read is %s",
    (status) => {
      mount("join_files", { ingest: { over: { status }, older: false } });
      chooseFile();
      expect(card()).toHaveTextContent(STILL);
      expect(screen.queryByTestId("join-ingest-failure")).toBeNull();
      expect(screen.queryByRole("alert")).toBeNull();
      expect(screen.queryByTestId("picker-count")).toBeNull();
    },
  );

  it("says the table is not read yet, never 'still being read', where its first read has not started", () => {
    mount("join_files", { ingest: { over: { status: "idle" }, older: false } });
    chooseFile();
    expect(card()).toHaveTextContent(
      "The table is not read yet; its columns appear here once it is.",
    );
    expect(card()).not.toHaveTextContent("still being read");
    expect(screen.queryByTestId("join-ingest-failure")).toBeNull();
  });

  it("says the table's columns are loading, never 'still being read', where it is read and its columns not fetched yet", () => {
    mount("join_files", { ingest: "loading" });
    chooseFile();
    expect(card()).toHaveTextContent("Loading the table's columns…");
    expect(card()).not.toHaveTextContent("still being read");
    expect(screen.queryByTestId("join-ingest-failure")).toBeNull();
  });

  // The columns of an older read, where the table is read anew: veiled and inert, never offered as
  // the table's, with the way forward from a read that will not restart by itself.
  it.each([
    ["running", "recomputing", null],
    ["error", "failed", "Try again"],
  ] as const)(
    "veils the older read's columns while the table's read is %s, and offers %s",
    async (status, veil, retry) => {
      mount("join_files", {
        ingest: {
          over: { status, ...(status === "error" ? { error: "the read ran out of memory" } : {}) },
          older: true,
        },
      });
      chooseFile();
      const veiled = screen.getByTestId("join-veil");
      expect(veiled).toHaveAttribute("data-veil", veil);
      expect(screen.getByTestId("picker-count").closest("[inert]")).not.toBeNull();
      const way = within(veiled).queryByTestId("retry-ingest");
      if (retry === null) expect(way).toBeNull();
      else {
        expect(way).toHaveTextContent(retry);
        expect(way!.closest("[inert]")).toBeNull();
      }
    },
  );

  it("offers the table's columns unveiled where it is read", () => {
    mount("join_files");
    chooseFile();
    expect(screen.getByTestId("join-veil")).toHaveAttribute("data-veil", "fresh");
    expect(screen.getByTestId("picker-count").closest("[inert]")).toBeNull();
    expect(screen.queryByTestId("retry-ingest")).toBeNull();
  });

  // The identifier is chosen in the column picker over the table as read: a live one is shown only
  // where the table is read and its columns fetched at the key it has now (an older read's is
  // veiled and inert).
  const pickerLive = () => {
    const picker = screen.queryByTestId("picker-count");
    return picker !== null && picker.closest("[inert]") === null;
  };

  it("asks for the identifier above where the table's columns are offered to choose it from", () => {
    mount("join_files");
    chooseFile();
    expect(pickerLive()).toBe(true);
    expect(record()).toHaveTextContent("Choose the identifier above");
    expect(record()).toBeDisabled();
  });

  it.each(Object.entries(NOT_READ))(
    "never asks for the identifier above where none can be chosen: the table's %s",
    (_, ingest) => {
      mount("join_files", { ingest });
      chooseFile();
      expect(pickerLive()).toBe(false);
      expect(record()).not.toHaveTextContent("identifier");
      expect(record()).toHaveTextContent(`Record: join ${FILE.name}`);
      expect(record()).toBeDisabled();
    },
  );

  it("counts the join chosen again once the table is read, after the server answered it was not read yet", async () => {
    const { qc } = mount("join_files");
    const refused = notYet();
    const ask = vi
      .spyOn(api, "joinPreview")
      .mockRejectedValueOnce(refused)
      .mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    expect(await screen.findByTestId("join-preview-error")).toHaveTextContent(refused.message);
    expect(record()).toBeDisabled();
    // The server's table is being read anew, then is read.
    act(() => readAs(qc, { status: "running", key: NEXT_KEY }));
    await settle();
    expect(ask).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("join-veil")).toHaveAttribute("data-veil", "recomputing");
    act(() => readAs(qc, { status: "fresh", key: NEXT_KEY }));
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(2));
    expect(ask.mock.calls[1]).toEqual(ask.mock.calls[0]);
    await screen.findByTestId("join-counts");
    expect(screen.queryByTestId("join-preview-error")).toBeNull();
    expect(screen.getByTestId("join-veil")).toHaveAttribute("data-veil", "fresh");
    expect(record()).toBeEnabled();
  });

  it("asks again only once the table is read anew, and only for a join chosen", async () => {
    const { qc } = mount("join_files");
    const ask = vi.spyOn(api, "joinPreview").mockRejectedValue(notYet());
    // Read anew with no join chosen: nothing to count.
    act(() => readAs(qc, { status: "fresh", key: NEXT_KEY }));
    await settle();
    expect(ask).not.toHaveBeenCalled();
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-preview-error");
    expect(ask).toHaveBeenCalledTimes(1);
    // The project changes, its table as it was read: nothing new to count.
    act(() => {
      const view = qc.getQueryData<ProjectView>(keys.view(PID))!;
      qc.setQueryData(keys.view(PID), { ...view, stages: { ...view.stages } });
    });
    await settle();
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it("veils the counts shown and records nothing while the table is read anew", async () => {
    const { qc, decide } = mount("join_files");
    vi.spyOn(api, "joinPreview").mockResolvedValue(PREVIEW);
    chooseJoin(FILE.id, PREVIEW.on);
    await screen.findByTestId("join-counts");
    expect(record()).toBeEnabled();
    act(() => readAs(qc, { status: "running", key: NEXT_KEY }));
    await settle();
    expect(screen.getByTestId("join-veil")).toHaveAttribute("data-veil", "recomputing");
    expect(screen.getByTestId("join-counts").closest("[inert]")).not.toBeNull();
    expect(record()).toBeDisabled();
    fireEvent.click(record());
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });
});

// ── the data dictionary (import_codebook) ───────────────────────────────────

describe("the data dictionary (import_codebook)", () => {
  it("reads an uploaded codebook and says what importing it would do, as the server serves it", async () => {
    const press = pressOf("import_codebook");
    const { decide } = mount("import_codebook", { at: press.from });
    const send = vi.spyOn(api, "uploadCodebook").mockResolvedValue(BOOK);
    const file = upload("codebook-source-input", BOOK.name);
    await waitFor(() => expect(send).toHaveBeenCalledWith(PID, file));
    await screen.findByTestId("codebook-counts");
    expect(screen.getByTestId("codebook-counts").querySelector("thead")).toHaveTextContent(
      BOOK.name,
    );
    expect(countsOf("codebook-counts")).toEqual([
      [fmt(BOOK.n_entries)],
      [fmt(BOOK.n_matched)],
      [fmt(BOOK.n_unmatched)],
    ]);
    expect(screen.getByTestId("codebook-sentence").textContent).toBe(said(BOOK.sentence));
    // The capture's codebook names every entry a column, and the values contradict none of it.
    expect(BOOK.n_unmatched).toBe(0);
    expect(BOOK.asked).toEqual([]);
    expect(screen.queryByTestId("codebook-unmatched")).toBeNull();
    expect(screen.queryByTestId("codebook-asked")).toBeNull();
    expect(decide).not.toHaveBeenCalled();
  });

  it("records the codebook by its id alone, exactly as the real server took it", async () => {
    const press = pressOf("import_codebook");
    const { decide } = mount("import_codebook", { at: press.from, next: viewAt(press.to) });
    vi.spyOn(api, "uploadCodebook").mockResolvedValue(BOOK);
    expect(record()).toBeDisabled();
    upload("codebook-source-input", BOOK.name);
    await screen.findByTestId("codebook-counts");
    expect(record()).toBeEnabled();
    expect(record()).toHaveTextContent(BOOK.name);
    fireEvent.click(record());
    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    const sent = decide.mock.calls[0]![1] as Record<string, unknown>;
    expect(sent).toEqual(press.decision);
    expect(problems(MODEL.import_codebook, sent)).toEqual([]);
  });

  it("says why the server could not read a codebook, leaves nothing to record, and says it no longer once one is read", async () => {
    const { decide } = mount("import_codebook", { mode: "server" });
    // The server's refusal in its served shape (server/service.py `codebook`: 400
    // unreadable_codebook, the codebook reader's own reason); its words the test's own.
    const message = "The test's file is not a variable table, a codebook page or an XPT file.";
    const refused = new ApiError(400, message, { error: { code: "unreadable_codebook", message } });
    const send = vi
      .spyOn(api, "uploadCodebook")
      .mockResolvedValueOnce(BOOK)
      .mockRejectedValueOnce(refused)
      .mockResolvedValueOnce(BOOK);
    // A codebook read, then another the server could not read: what the first would import goes.
    upload("codebook-source-input", BOOK.name);
    await screen.findByTestId("codebook-counts");
    expect(screen.queryByTestId("codebook-source-error")).toBeNull();
    upload("codebook-source-input", "notes.txt");
    expect(await screen.findByTestId("codebook-source-error")).toHaveTextContent(message);
    expect(screen.queryByTestId("codebook-counts")).toBeNull();
    expect(screen.queryByTestId("codebook-sentence")).toBeNull();
    expect(record()).toBeDisabled();
    fireEvent.click(record());
    await settle();
    expect(decide).not.toHaveBeenCalled();
    upload("codebook-source-input", BOOK.name);
    await waitFor(() => expect(send).toHaveBeenCalledTimes(3));
    await screen.findByTestId("codebook-counts");
    expect(screen.queryByTestId("codebook-source-error")).toBeNull();
    expect(record()).toBeEnabled();
  });

  it("records nothing before a codebook is read", async () => {
    const { decide } = mount("import_codebook");
    expect(record()).toBeDisabled();
    fireEvent.click(record());
    await settle();
    expect(decide).not.toHaveBeenCalled();
  });

  it("lists the fields the values contradict, asked instead of applied, and the entries naming no column", async () => {
    mount("import_codebook");
    const asked: CodebookPreview["asked"] = [
      {
        column: "RIAGENDR",
        field: "codes",
        says: "1 male, 2 female",
        values: "`3` too",
        checked: true,
        exits: [],
      },
      {
        column: "LBXGLU",
        field: "unit",
        says: "mg/dL",
        values: "no check of a unit runs here",
        checked: false,
        exits: [],
      },
    ];
    const unmatched = ["DR1EXMER", "DRABF"];
    const book: CodebookPreview = {
      ...BOOK,
      asked,
      unmatched,
      n_unmatched: 5,
      n_matched: BOOK.n_matched - 5,
    };
    vi.spyOn(api, "uploadCodebook").mockResolvedValue(book);
    upload("codebook-source-input", book.name);
    const list = await screen.findByTestId("codebook-asked");
    const rows = Array.from(list.querySelectorAll("li"));
    expect(rows).toHaveLength(2);
    // As the server's sentence words each (core/codebook.py `codebook_sentence`): what a check that
    // ran found is what "the values show"; a check that could not run showed nothing, and its
    // `values` say why in their own words.
    const [contradicted, unchecked] = asked as [(typeof asked)[0], (typeof asked)[0]];
    expect(rows[0]!.textContent).toBe(
      `${contradicted.column}'s ${contradicted.field}: the codebook says ${contradicted.says}; ` +
        `the values show ${said(contradicted.values)}`,
    );
    expect(rows[1]!.textContent).toBe(
      `${unchecked.column}'s ${unchecked.field}: the codebook says ${unchecked.says}; ` +
        `${unchecked.values}`,
    );
    expect(rows[1]).not.toHaveTextContent("the values show");
    const names = screen.getByTestId("codebook-unmatched");
    for (const n of unmatched) expect(names).toHaveTextContent(n);
    expect(names).toHaveTextContent(`${fmt(book.n_unmatched - unmatched.length)} more`);
    expect(countsOf("codebook-counts")).toEqual([
      [fmt(book.n_entries)],
      [fmt(book.n_matched)],
      [fmt(book.n_unmatched)],
    ]);
  });

  it("names each entry naming no column, and no more, where the server served every one", async () => {
    // The server serves the first twenty names and the count of all (core/codebook.py
    // `Assessment.to_dict`): up to twenty, the names are all of them.
    mount("import_codebook");
    const unmatched = ["DR1EXMER", "DRABF"];
    const book: CodebookPreview = {
      ...BOOK,
      unmatched,
      n_unmatched: unmatched.length,
      n_matched: BOOK.n_matched - unmatched.length,
    };
    vi.spyOn(api, "uploadCodebook").mockResolvedValue(book);
    upload("codebook-source-input", book.name);
    const names = await screen.findByTestId("codebook-unmatched");
    expect(names.textContent).toBe(`Not in the table: ${unmatched[0]} and ${unmatched[1]}`);
    expect(names).not.toHaveTextContent("more");
  });

  it("says a press the server refuses in the engine's words", async () => {
    const { decide } = mount("import_codebook");
    vi.spyOn(api, "uploadCodebook").mockResolvedValue(BOOK);
    const refusal = {
      error: {
        code: "codebook_names_no_column",
        message: "No variable `DEMO_J.htm` documents is a column of this table.",
        exits: [{ label: "Import the codebook of this table's file", decision: null }],
      },
    };
    decide.mockRejectedValue(new RefusalError(refusal));
    upload("codebook-source-input", BOOK.name);
    await screen.findByTestId("codebook-counts");
    fireEvent.click(record());
    expect(await screen.findByTestId("refusal")).toHaveTextContent(said(refusal.error.message));
  });

  it("names the codebooks imported already, by the state's codebooks", () => {
    const press = pressOf("import_codebook");
    const after = viewAt(press.to);
    const name = after.state.codebooks![BOOK.id]!.name;
    mount("import_codebook", { at: press.to });
    expect(screen.getByTestId("imported")).toHaveTextContent(name);
  });

  it("names none where the state holds no codebook", () => {
    const press = pressOf("import_codebook");
    expect(viewAt(press.from).state.codebooks ?? null).toBeNull();
    mount("import_codebook", { at: press.from });
    expect(screen.queryByTestId("imported")).toBeNull();
  });

  it("reads a codebook by its path in local mode only", async () => {
    mount("import_codebook", { mode: "local" });
    const send = vi
      .spyOn(api, "codebook")
      .mockResolvedValue(served<CodebookPreview>("codebook_nhanes"));
    const path = "/data/nhanes/DEMO_J.htm";
    fireEvent.change(screen.getByTestId("codebook-source-path"), { target: { value: path } });
    fireEvent.submit(screen.getByTestId("codebook-source-path").closest("form")!);
    // The table's own labels are not asked for: the request's `labels` is the model's default.
    const labels = SCHEMAS.CodebookRequest!.properties!.labels!.default;
    await waitFor(() => expect(send).toHaveBeenCalledWith(PID, { path, labels }));
    await screen.findByTestId("codebook-counts");
  });

  it("offers no path in server mode", () => {
    mount("import_codebook", { mode: "server" });
    expect(screen.queryByTestId("codebook-source-path")).toBeNull();
    expect(screen.getByTestId("codebook-source-input")).toBeInTheDocument();
  });
});

// ── the table the codebook is read against ──────────────────────────────────

describe("the table the codebook reads (the ingest stage)", () => {
  const card = () => screen.getByTestId("declaration-import_codebook");
  const title = () => within(card()).getByRole("heading").textContent!;
  const failed = NOT_READ["first read failed"] as Exclude<NotRead, "loading">;
  const reason = failed.over.error!.replace(/^ValueError: /, "");

  // The server reads a codebook only against the table read fresh (server/service.py
  // `stage_codebook`, `_table_ready`), and answers every other read "still being read".
  it.each([false, true])(
    "names the reading that did not finish, its reason and its retry where the table's read failed (an older read held: %s), and offers no codebook read the server would answer 'still being read'",
    async (older) => {
      mount("import_codebook", { ingest: { ...failed, older } });
      const failure = screen.getByTestId("codebook-ingest-failure");
      expect(failure).toHaveAttribute("data-stage", "ingest");
      expect(failure).toHaveAttribute("data-kind", "failed");
      expect(failure).toHaveTextContent(reason);
      expect(failure).toHaveTextContent(`${title()} waits on it.`);
      expect(card()).not.toHaveTextContent("still being read");
      expect(card()).not.toHaveTextContent("You stopped");
      expect(screen.queryByTestId("codebook-source-input")).toBeNull();
      expect(screen.queryByTestId("codebook-source-path")).toBeNull();
      expect(screen.queryByTestId("card-record")).toBeNull();
      const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
      fireEvent.click(within(failure).getByTestId("retry-ingest"));
      await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "ingest"));
    },
  );

  // The press is assessed against the table read fresh too, and refused "still being read" where
  // it is not (core/codebook.py `_assessed`).
  it("offers no press of a codebook read before the table's read failed, and names the failure", async () => {
    const { qc, decide } = mount("import_codebook");
    vi.spyOn(api, "uploadCodebook").mockResolvedValue(BOOK);
    upload("codebook-source-input", BOOK.name);
    await screen.findByTestId("codebook-counts");
    expect(record()).toBeEnabled();
    expect(screen.queryByTestId("codebook-ingest-failure")).toBeNull();
    act(() => readAs(qc, { status: "error", error: failed.over.error!, key: NEXT_KEY }));
    const failure = await screen.findByTestId("codebook-ingest-failure");
    expect(failure).toHaveAttribute("data-kind", "failed");
    expect(failure).toHaveTextContent(reason);
    expect(screen.queryByTestId("codebook-counts")).toBeNull();
    expect(screen.queryByTestId("card-record")).toBeNull();
    expect(card()).not.toHaveTextContent("still being read");
    expect(decide).not.toHaveBeenCalled();
  });

  it("says the reading was stopped, with a way to run it again, where the table's read was stopped, never 'still being read'", async () => {
    mount("import_codebook", { ingest: NOT_READ["first read stopped"] });
    const stopped = screen.getByTestId("codebook-ingest-failure");
    expect(stopped).toHaveAttribute("data-stage", "ingest");
    expect(stopped).toHaveAttribute("data-kind", "stopped");
    expect(stopped).toHaveTextContent("You stopped");
    expect(stopped).toHaveTextContent(`${title()} waits on it.`);
    expect(card()).not.toHaveTextContent("still being read");
    expect(screen.queryByTestId("codebook-source-input")).toBeNull();
    expect(screen.queryByTestId("card-record")).toBeNull();
    const retry = within(stopped).getByTestId("retry-ingest");
    expect(retry).toHaveTextContent("Recompute");
    const run = vi.spyOn(api, "runStage").mockImplementation(() => new Promise(() => {}));
    fireEvent.click(retry);
    await waitFor(() => expect(run).toHaveBeenCalledWith(PID, "ingest"));
  });

  it.each(["queued", "running"] as const)(
    "names no failure while the table's read is %s, and says the server's 'still being read' where it is",
    async (status) => {
      mount("import_codebook", { ingest: { over: { status }, older: false } });
      expect(screen.queryByTestId("codebook-ingest-failure")).toBeNull();
      expect(screen.queryByRole("alert")).toBeNull();
      const refused = notYet();
      vi.spyOn(api, "uploadCodebook").mockRejectedValue(refused);
      upload("codebook-source-input", BOOK.name);
      expect(await screen.findByTestId("codebook-source-error")).toHaveTextContent(refused.message);
      expect(screen.queryByTestId("codebook-ingest-failure")).toBeNull();
    },
  );

  it("names no failure and offers the codebook's read where the table is read", () => {
    mount("import_codebook");
    expect(screen.queryByTestId("codebook-ingest-failure")).toBeNull();
    expect(screen.queryByTestId("retry-ingest")).toBeNull();
    expect(screen.getByTestId("codebook-source-input")).toBeInTheDocument();
  });
});
