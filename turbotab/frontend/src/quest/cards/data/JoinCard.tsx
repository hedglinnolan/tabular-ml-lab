/**
 * `join_files`: join a file added to the project to its table on a shared identifier
 * (core/decisions.py `JoinFiles`, core/assembly.py), an uncounted Decide in Your data (core/quest.py
 * DECLARATIONS). NHANES ships each component as its own file, joined on `SEQN`.
 *
 * The card adds the file as the Start screen opens a table: uploaded, or in local mode by its path
 * (`useAddFile`). The person names the file, the identifier in the table (the column picker over
 * the table as read) and in the file (the same name unless named otherwise), and which rows the
 * joined table keeps. The server's preview of exactly that join (`useJoinPreview`) then says what
 * it would do: each side's rows and identifier values, the rows with no partner, and the sentence
 * it would record. Nothing is recorded until the card's button is pressed.
 *
 * Only a join whose preview, for the choice shown, the server counted and did not refuse can be
 * recorded; a refused one (many-to-many, an identifier one side lacks or writes another way, no
 * value in both) is said in the engine's words, with its ways forward. The press posts the file,
 * the identifier and the rows kept; the file's name and the counts are the server's, recorded with
 * the answer. A file the state holds as joined (its `joins`, as the server's file list reads them)
 * is named as joined and not offered again.
 *
 * The identifier in the table is chosen from the table as read (the ingest stage), and the server
 * counts a join only once that read is fresh (server/service.py `_table_ready`, "not yet" until
 * then). A join changes what the table is, so the table is read anew after each: meanwhile the
 * server serves the older read, whose columns, and any count made on it, stay veiled and inert
 * and record nothing, and once the table is read the join chosen is counted again, since nothing
 * else would ask. Where the read failed or was stopped and no older one is held, the card says
 * which computation did not finish and why, with its retry, as the Model 1 card does
 * (components/Failure.tsx `StageFailure`); it says the table is still being read only while it is.
 */
import { useEffect, useId, useRef, useState, type DragEvent } from "react";
import {
  useAddFile,
  useFiles,
  useHealth,
  useJoinPreview,
  useProjectView,
} from "../../../api/queries";
import type { AddedFile, JoinPreview, JoinPreviewRequest } from "../../../api/m3-types";
import type { Decision, DecisionInput, QuestLine } from "../../../api/schema";
import { StageFailure, rootFailure } from "../../../components/Failure";
import { ColumnPicker } from "../../../components/record/ColumnPicker";
import { Pending } from "../../../components/record/blocks";
import { Actions, RecordButton } from "../../../components/record/ask/common";
import g from "../../../components/record/generic/generic.module.css";
import { Prose, V } from "../../../components/Prose";
import { StageRetry } from "../../../components/StageRetry";
import { StaleVeil, veilFor } from "../../../motion/StaleVeil";
import s from "../../../screens/StartScreen.module.css";
import { useStage } from "../../../state/stages";
import { cx, fmtInt, listJoin } from "../../../util/format";
import type { LineCard } from "../../contracts";
import { Frame, useCard, type CardWiring } from "../predict/card";
import c from "../predict/declaration.module.css";
import { SelectField } from "../predict/fields";

type How = JoinPreviewRequest["how"];
/** What the press posts; the model's `name` and `counts` are the server's, recorded with the answer
 *  (core/assembly.py `_join_records_its_counts`). */
type JoinFiles = Pick<
  Extract<DecisionInput, { kind: "join_files" }>,
  "kind" | "file" | "on" | "right_on" | "how"
>;

/** The rows the joined table keeps, in the engine's words for each (core/assembly.py
 *  `join_sentence`); the model's default, which it takes when none is sent, first. */
const HOWS: readonly How[] = ["left", "inner"];
const HOW_NAMES: Record<How, string> = {
  left: "Keep every row of the table",
  inner: "Keep only the rows found in both",
};
const DEFAULT_HOW: How = "left";

/** The line's name as its card's heading says it (predict/card.tsx `Frame`). */
const capital = (text: string) => `${text.charAt(0).toUpperCase()}${text.slice(1)}`;

/** What the person has chosen: the file, the identifier in the table and in the file (null: the
 *  same name), and the rows kept. */
interface Choice {
  file: string | null;
  on: string | null;
  rightOn: string | null;
  how: How;
}

/** The join to count and record for a choice, or null while the file or the identifier is still
 *  to choose. */
export function joinRequest(choice: Choice): JoinPreviewRequest | null {
  if (!choice.file || !choice.on) return null;
  return {
    file: choice.file,
    on: choice.on,
    ...(choice.rightOn ? { right_on: choice.rightOn } : {}),
    how: choice.how,
  };
}

/** Whether a served preview counts exactly the join asked (the server names the file's identifier
 *  even where the request left it to the table's name). */
export function answers(preview: JoinPreview, asked: JoinPreviewRequest): boolean {
  return (
    preview.file === asked.file &&
    preview.on === asked.on &&
    preview.right_on === (asked.right_on ?? asked.on) &&
    preview.how === asked.how
  );
}

/** The decision a counted join records: what the person chose, and nothing the server fills in. */
export function joinDecision(asked: JoinPreviewRequest): JoinFiles {
  return {
    kind: "join_files",
    file: asked.file,
    on: asked.on,
    ...(asked.right_on ? { right_on: asked.right_on } : {}),
    how: asked.how,
  };
}

/**
 * A file to read, as the Start screen takes a table: dropped or chosen, or in local mode typed as
 * a path on this machine. The server reads it and says what it found, or why it could not.
 */
export function FileSource({
  drop,
  local,
  pending,
  error,
  pathAction,
  onFile,
  onPath,
  testId,
}: {
  drop: string;
  local: boolean;
  pending: boolean;
  error: Error | null;
  /** The path form's button. */
  pathAction: string;
  onFile: (file: File) => void;
  onPath: (path: string) => void;
  testId: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [typed, setTyped] = useState("");
  const pathId = useId();
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };
  return (
    <div className={c.group}>
      <div
        className={cx(s.drop, over && s.dropOver)}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        data-testid={`${testId}-drop`}
      >
        <p className={s.dropText}>{drop}</p>
        <button
          type="button"
          className={s.secondary}
          onClick={() => inputRef.current?.click()}
          disabled={pending}
        >
          {pending ? "Reading the file…" : "Choose a file"}
        </button>
        <input
          ref={inputRef}
          type="file"
          className="visually-hidden"
          tabIndex={-1}
          aria-hidden="true"
          data-testid={`${testId}-input`}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
      </div>
      {local ? (
        <form
          className={s.pathForm}
          onSubmit={(e) => {
            e.preventDefault();
            if (typed.trim()) onPath(typed.trim());
          }}
        >
          <label htmlFor={pathId} className={s.pathLabel}>
            Or type a path
          </label>
          <input
            id={pathId}
            className={s.pathInput}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="/path/to/file"
            spellCheck={false}
            data-testid={`${testId}-path`}
          />
          <button type="submit" className={s.secondary} disabled={!typed.trim() || pending}>
            {pathAction}
          </button>
        </form>
      ) : null}
      {error ? (
        <p className={s.error} role="alert" data-testid={`${testId}-error`}>
          {error.message}
        </p>
      ) : null}
    </div>
  );
}

/** Each side's rows and identifier values, and the rows with no partner, as the server counted
 *  them; the rows whose identifier is blank only where a side has any. */
function Counts({ p }: { p: JoinPreview }) {
  const rows: [string, number, number][] = [
    ["Rows", p.table.rows, p.file_side.rows],
    ["Distinct identifier values", p.table.keys, p.file_side.keys],
    ...(p.table.blank_keys || p.file_side.blank_keys
      ? [
          ["Rows with a blank identifier", p.table.blank_keys, p.file_side.blank_keys] as [
            string,
            number,
            number,
          ],
        ]
      : []),
    ["Rows with no partner", p.table_unmatched, p.file_unmatched],
  ];
  return (
    <div className={g.matrixScroll}>
      <table className={g.table} data-testid="join-counts">
        <thead>
          <tr>
            <td />
            <th scope="col">The table</th>
            <th scope="col">
              <V>{p.file_side.name}</V>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, table, file]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{fmtInt(table)}</td>
              <td>{fmtInt(file)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** What the server's preview says of the join chosen: its refusal, or its counts and sentence. */
function Preview({ p }: { p: JoinPreview }) {
  if (p.refusal) {
    return (
      <div className={c.group} data-testid="join-refused" role="alert">
        <p className={c.blocked}>
          <span className={c.blockedLabel}>Refused</span> <Prose text={p.refusal.message} />
        </p>
        {p.refusal.exits.length ? (
          <ul className={c.words}>
            {p.refusal.exits.map((exit) => (
              <li key={exit.label}>
                <Prose text={exit.label} />
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    );
  }
  return (
    <div className={c.group}>
      <Counts p={p} />
      <p className={c.said} data-testid="join-sentence">
        <Prose text={p.sentence} />
      </p>
    </div>
  );
}

function Body({ pid, line, card }: { pid: string; line: QuestLine; card: CardWiring }) {
  const view = useProjectView(pid).data!;
  const listed = useFiles(pid).data;
  const add = useAddFile(pid);
  const preview = useJoinPreview(pid);
  const local = useHealth().data?.mode === "local";
  const ingest = useStage(pid, view, "ingest");
  const read = view.stages.ingest;
  const labelId = useId();
  const [choice, setChoice] = useState<Choice>({
    file: null,
    on: null,
    rightOn: null,
    how: DEFAULT_HOW,
  });
  // The file just added, until the file list read again holds it.
  const [added, setAdded] = useState<AddedFile | null>(null);

  const joins = card.state?.joins ?? {};
  const files = [
    ...(listed ?? []),
    ...(added && !listed?.some((f) => f.id === added.id) ? [added] : []),
  ];
  const offered = files.filter((f) => !(f.id in joins));
  const joined = Object.values(joins).map((j) => j.name);
  // A file the list no longer offers (joined since) is no choice.
  const file = offered.find((f) => f.id === choice.file) ?? null;
  const asked = file ? joinRequest(choice) : null;
  const shown = asked && preview.data && answers(preview.data, asked) ? preview.data : null;
  const columns = ingest?.artifact?.columns.filter((col) => col.name !== "__row_id");
  // The columns on screen are the table's only while its read is fresh at the key it has now.
  const veil = veilFor(read, ingest);
  // A read that failed or was stopped does not restart by itself (components/StageRetry.tsx).
  const failed = rootFailure(view.stages, "ingest") !== null;
  const stopped = !failed && !!read?.cancelled;
  const reading = read?.status === "queued" || read?.status === "running";
  const loading = read?.status === "fresh" || read?.status === "stale";

  // The table read anew (fresh at another key) is counted again for the join chosen: the server
  // answered "not yet" before it was read, or counted the table as it was, and nothing else asks.
  const table = read?.status === "fresh" ? read.key : null;
  useEffect(() => {
    if (table && asked) preview.mutate(asked);
    // Only the table read anew asks here: each choice asks for itself (`choose`).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table]);

  const choose = (patch: Partial<Choice>, pool: AddedFile[] = offered) => {
    const next = { ...choice, ...patch };
    setChoice(next);
    const ask = pool.some((f) => f.id === next.file) ? joinRequest(next) : null;
    if (ask) preview.mutate(ask);
    else preview.reset();
  };
  const onAdded = (f: AddedFile) => {
    setAdded(f);
    choose({ file: f.id, rightOn: null }, [...offered.filter((x) => x.id !== f.id), f]);
  };

  return (
    <>
      {joined.length ? (
        <p className={c.said} data-testid="joined">
          Joined: {listJoin(joined)}
        </p>
      ) : null}
      <fieldset className={c.group}>
        <legend className={c.legend}>The file to join</legend>
        {offered.length ? (
          <div className={c.fields}>
            <SelectField
              name="file"
              label="Added to the project"
              value={file?.id ?? ""}
              options={[
                { value: "", label: "Choose one" },
                ...offered.map((f) => ({
                  value: f.id,
                  label: `${f.name} (${fmtInt(f.n_rows)} rows, ${fmtInt(f.n_cols)} columns)`,
                })),
              ]}
              onChange={(id) => choose({ file: id || null, rightOn: null })}
            />
          </div>
        ) : null}
        <FileSource
          drop={offered.length ? "Or drop another file to join here" : "Drop the file to join here"}
          local={local}
          pending={add.isPending}
          error={add.error}
          pathAction="Add"
          onFile={(f) => add.mutate({ file: f }, { onSuccess: onAdded })}
          onPath={(path) => add.mutate({ path }, { onSuccess: onAdded })}
          testId="join-source"
        />
        {file?.warnings.length ? (
          <ul className={c.words} data-testid="join-file-warnings">
            {file.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        ) : null}
      </fieldset>
      {file && columns ? (
        <StaleVeil
          state={veil}
          testId="join-veil"
          action={<StageRetry pid={pid} status={read} />}
        >
          <fieldset className={c.group}>
            <legend className={c.legend}>The column that identifies each unit</legend>
            <p className={c.fieldLabel} id={labelId}>
              In the table
            </p>
            <ColumnPicker
              columns={columns}
              value={choice.on}
              onChange={(on) => choose({ on })}
              summaries={new Map(card.summaries)}
              labelId={labelId}
            />
            <div className={c.fields}>
              <SelectField
                name="right_on"
                label="In the file"
                value={choice.rightOn ?? ""}
                options={[
                  { value: "", label: choice.on ? `The same name, ${choice.on}` : "The same name" },
                  ...file.columns.map((col) => ({ value: col, label: col })),
                ]}
                onChange={(col) => choose({ rightOn: col || null })}
              />
              <SelectField
                name="how"
                label="Rows kept"
                value={choice.how}
                options={HOWS.map((h) => ({ value: h, label: HOW_NAMES[h] }))}
                onChange={(how) => choose({ how: how as How })}
              />
            </div>
          </fieldset>
          {preview.isPending ? <Pending>Counting the rows the join would give…</Pending> : null}
          {preview.isError ? (
            <p className={s.error} role="alert" data-testid="join-preview-error">
              {preview.error.message}
            </p>
          ) : null}
          {shown ? <Preview p={shown} /> : null}
        </StaleVeil>
      ) : file && (failed || stopped) ? (
        // Nothing in the quest shell answers a request to reopen an answer (the Record does), so
        // the failure offers its retry alone.
        <StageFailure
          pid={pid}
          view={view}
          stage="ingest"
          changeable={false}
          after={<>{capital(line.name)} waits on it.</>}
          testId="join-ingest-failure"
        />
      ) : file ? (
        <Pending>
          {reading
            ? "The table is still being read; its columns appear here when it is done."
            : loading
              ? "Loading the table's columns…"
              : "The table is not read yet; its columns appear here once it is."}
        </Pending>
      ) : null}
      <Actions>
        <RecordButton
          disabled={
            !asked || !shown || shown.refusal !== null || veil !== "fresh" || card.pending
          }
          onClick={() => {
            if (asked && shown && shown.refusal === null && veil === "fresh")
              card.record(joinDecision(asked) as unknown as Decision, "join");
          }}
          title="Records the join counted above."
          testId="card-record"
        >
          {!file
            ? "Choose the file to join above"
            : !choice.on
              ? "Choose the identifier above"
              : `Record: join ${file.name}`}
        </RecordButton>
      </Actions>
      {card.answerAt?.key === "join" ? card.answerAt.node : null}
    </>
  );
}

export const JoinCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "join_files");
  const view = useProjectView(pid).data;
  return (
    <Frame line={line}>
      {card.state && view ? (
        <Body key={line.decision_id ?? "open"} pid={pid} line={line} card={card} />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
