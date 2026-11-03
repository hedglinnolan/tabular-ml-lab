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
 * values contradict, a press the server refuses) is built in the served shape, its words the test's
 * own construction.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { api, RefusalError } from "../../../api/client";
import type {
  AddedFile,
  CodebookPreview,
  JoinPreview,
  JoinPreviewRequest,
} from "../../../api/m3-types";
import { keys } from "../../../api/queries";
import type { Decision, ProjectView, QuestLine } from "../../../api/schema";
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

function mount(
  kind: Kind,
  opts: {
    at?: number;
    mode?: "local" | "server";
    line?: Partial<QuestLine>;
    next?: ProjectView;
  } = {},
) {
  const view = viewAt(opts.at ?? 1);
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  });
  qc.setQueryData(keys.view(PID), view);
  qc.setQueryData(keys.health(), {
    version: "test",
    mode: opts.mode ?? "local",
    workers: 2,
    user: null,
    auth: "none",
  });
  const ingest = view.stages.ingest!.key;
  qc.setQueryData(keys.stage(PID, "ingest"), {
    stage: "ingest",
    key: ingest,
    fresh: true,
    status: "fresh",
    artifact: artifactAt(F, "ingest", ingest),
  });
  qc.setQueryData(keys.columns(PID), served("columns"));
  qc.setQueryData(keys.files(PID), FILES);
  // Nothing else is fetched: the stage results and previews these cards do not read never answer.
  vi.spyOn(api, "stage").mockImplementation(() => new Promise(() => {}));
  vi.spyOn(api, "preview").mockImplementation(() => new Promise(() => {}));
  vi.spyOn(api, "files").mockResolvedValue(FILES);
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
    expect(rows[0]).toHaveTextContent(asked[0]!.column);
    expect(rows[0]).toHaveTextContent(asked[0]!.says);
    expect(rows[0]).toHaveTextContent(said(asked[0]!.values));
    expect(rows[1]).toHaveTextContent(asked[1]!.column);
    expect(rows[1]).toHaveTextContent(asked[1]!.values);
    const names = screen.getByTestId("codebook-unmatched");
    for (const n of unmatched) expect(names).toHaveTextContent(n);
    expect(names).toHaveTextContent(`${fmt(book.n_unmatched - unmatched.length)} more`);
    expect(countsOf("codebook-counts")).toEqual([
      [fmt(book.n_entries)],
      [fmt(book.n_matched)],
      [fmt(book.n_unmatched)],
    ]);
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
