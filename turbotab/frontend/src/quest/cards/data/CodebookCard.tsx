/**
 * `import_codebook`: import the researcher's own data dictionary (core/decisions.py
 * `ImportCodebook`, core/codebook.py), an uncounted Decide in Your data (core/quest.py
 * DECLARATIONS): a variable table, an NHANES codebook page, or the labels an XPT file carries.
 *
 * The card reads the codebook as the Start screen takes a table: uploaded, or in local mode by its
 * path (`useCodebook`). The server keeps it in the project and answers with what importing it would
 * do, before anything is recorded: how many of its entries name a column of the table and which do
 * not, the fields the values contradict (asked, never applied), and the sentence it would record.
 * The press posts the codebook's id alone; what it settles is the server's, read from the codebook
 * and the values. A press the server refuses (a codebook that names no column of the table) is
 * said in the engine's words, with its exits. The codebooks the state holds are named as imported.
 *
 * A codebook is read, and its press assessed, only against the table read fresh (server/service.py
 * `_table_ready`, core/codebook.py `_assessed`): the server answers any other "still being read".
 * That answer holds only until the table is read: from then on the card no longer says it, and
 * offers the read again. What a codebook read would import is counted against the table as it was
 * read then, at its key: once the table is read anew those counts are another table's, so they stay
 * veiled and inert and record nothing, and once it is read the codebook is read again against it,
 * since nothing else would ask, as the join card counts its join again. The press is offered only
 * for a codebook read against the table as it stands read. A read that failed or was stopped does
 * not restart by itself, so there the card says which computation did not finish and why, with its
 * retry, in place of the read and the press, as the join card does (components/Failure.tsx
 * `StageFailure`).
 */
import { useEffect, useState } from "react";
import { isRefusalError } from "../../../api/client";
import { NOT_YET } from "../../../api/m2-types";
import { useCodebook, useHealth, useProjectView } from "../../../api/queries";
import type { CodebookPreview } from "../../../api/m3-types";
import type { Decision, DecisionInput, ProjectView, QuestLine } from "../../../api/schema";
import { StageFailure, rootFailure } from "../../../components/Failure";
import { Pending } from "../../../components/record/blocks";
import { Actions, RecordButton } from "../../../components/record/ask/common";
import g from "../../../components/record/generic/generic.module.css";
import { Prose, V } from "../../../components/Prose";
import { StaleVeil, veilFor, type VeilState } from "../../../motion/StaleVeil";
import { fmtInt, listJoin } from "../../../util/format";
import type { LineCard } from "../../contracts";
import { Frame, useCard, type CardWiring } from "../predict/card";
import c from "../predict/declaration.module.css";
import { FileSource, capital } from "./JoinCard";

/** What the press posts: the codebook's id. Every other field of the model is the server's, which
 *  discards what a client sends in its place (core/codebook.py `_codebook_records_what_it_settles`). */
type ImportCodebook = Pick<
  Extract<DecisionInput, { kind: "import_codebook" }>,
  "kind" | "codebook"
>;

/** The decision a codebook read records: its id, and nothing the server fills in. */
export function codebookDecision(book: CodebookPreview): ImportCodebook {
  return { kind: "import_codebook", codebook: book.id };
}

/** How many of the codebook's entries name a column of the table, as the server counted them, and
 *  the names of those that do not (the server serves the first of them and their count). */
function Counts({ p }: { p: CodebookPreview }) {
  const rows: [string, number][] = [
    ["Entries", p.n_entries],
    ["Name a column of the table", p.n_matched],
    ["Name no column of the table", p.n_unmatched],
  ];
  const more = p.n_unmatched - p.unmatched.length;
  return (
    <>
      <div className={g.matrixScroll}>
        <table className={g.table} data-testid="codebook-counts">
          <thead>
            <tr>
              <td />
              <th scope="col">
                <V>{p.name}</V>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, n]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{fmtInt(n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {p.n_unmatched > 0 ? (
        <p className={c.words} data-testid="codebook-unmatched">
          Not in the table:{" "}
          {listJoin([...p.unmatched, ...(more > 0 ? [`${fmtInt(more)} more`] : [])])}
        </p>
      ) : null}
    </>
  );
}

/** The fields the values contradict, or that cannot be checked here: asked, never applied. */
function Asked({ asked }: { asked: CodebookPreview["asked"] }) {
  return (
    <fieldset className={c.group}>
      <legend className={c.legend}>Asked instead of applied</legend>
      <ul className={c.words} data-testid="codebook-asked">
        {asked.map((a) => (
          <li key={`${a.column}:${a.field}`}>
            <V>{a.column}</V>
            {"'s "}
            {a.field}: the codebook says <Prose text={a.says} />;{" "}
            {a.checked ? (
              <>
                the values show <Prose text={a.values} />
              </>
            ) : (
              <Prose text={a.values} />
            )}
          </li>
        ))}
      </ul>
    </fieldset>
  );
}

function Body({
  pid,
  line,
  view,
  card,
}: {
  pid: string;
  line: QuestLine;
  view: ProjectView;
  card: CardWiring;
}) {
  const read = useCodebook(pid);
  const local = useHealth().data?.mode === "local";
  const books = Object.values(card.state?.codebooks ?? {}).map((b) => b.name);
  const book = read.data ?? null;
  // A read of the table that failed or was stopped does not restart by itself
  // (components/StageRetry.tsx), and the server reads no codebook until it is read.
  const failed = rootFailure(view.stages, "ingest") !== null;
  const stopped = !failed && !!view.stages.ingest?.cancelled;
  // The server answers a read asked before the table is read "not yet" (server/service.py
  // `_table_ready`), which no longer holds once the table is read: it is cleared then, and never
  // said while the table stands read.
  const ingest = view.stages.ingest;
  const table = ingest?.status === "fresh" ? ingest.key : null;
  const early = isRefusalError(read.error) && read.error.refusal.error.code === NOT_YET;
  useEffect(() => {
    if (table && early) read.reset();
    // Only the table read clears it. One that arrives while the table stands read is held unsaid:
    // the server may have begun reading it anew, which its next event says.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table]);
  const error = early && table ? null : read.error;
  // The key the table stood read at when the codebook shown was asked for (none while it was not
  // read), noted as each read is asked: its counts are the table's only while it stands read there.
  const [asked, setAsked] = useState({ submittedAt: 0, at: null as string | null });
  if (read.submittedAt !== asked.submittedAt) setAsked({ submittedAt: read.submittedAt, at: table });
  const readAt = read.submittedAt === asked.submittedAt ? asked.at : table;
  const veil = veilFor(ingest, book ? { fresh: true, key: readAt, artifact: book } : undefined);
  // A codebook read against another table than the one read now is read again against it.
  useEffect(() => {
    if (table && book && readAt !== table && read.variables) read.mutate(read.variables);
    // Only the table read anew, or a codebook's read answered since, asks here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table, book]);
  return (
    <>
      {books.length ? (
        <p className={c.said} data-testid="imported">
          Imported: {listJoin(books)}
        </p>
      ) : null}
      {failed || stopped ? (
        // Nothing in the quest shell answers a request to reopen an answer (the Record does), so
        // the failure offers its retry alone.
        <StageFailure
          pid={pid}
          view={view}
          stage="ingest"
          changeable={false}
          after={<>{capital(line.name)} waits on it.</>}
          testId="codebook-ingest-failure"
        />
      ) : (
        <Read
          card={card}
          local={local}
          read={read}
          error={error}
          book={book}
          veil={veil}
          current={!!table && veil === "fresh"}
        />
      )}
    </>
  );
}

/** The codebook's source, what importing the codebook read would do, and its press. */
function Read({
  card,
  local,
  read,
  error,
  book,
  veil,
  current,
}: {
  card: CardWiring;
  local: boolean;
  read: ReturnType<typeof useCodebook>;
  /** The read's error the card says: none where it no longer holds. */
  error: Error | null;
  book: CodebookPreview | null;
  /** Whether the codebook's counts are the table's as it stands read (`fresh`), or another's. */
  veil: VeilState;
  /** Whether the codebook shown was read against the table as it stands read, so its press is
   *  assessed: never while the table is not read (the server refuses it "still being read"). */
  current: boolean;
}) {
  return (
    <>
      <fieldset className={c.group}>
        <legend className={c.legend}>The codebook</legend>
        <FileSource
          drop="Drop a codebook here: a variable table, an NHANES codebook page or an XPT file"
          local={local}
          pending={read.isPending}
          error={error}
          pathAction="Read"
          onFile={(file) => read.mutate({ file })}
          onPath={(path) => read.mutate({ path, labels: false })}
          testId="codebook-source"
        />
      </fieldset>
      {read.isPending ? <Pending>Reading the codebook…</Pending> : null}
      {book ? (
        <StaleVeil state={veil} testId="codebook-veil">
          <div className={c.group}>
            <Counts p={book} />
            {book.asked.length ? <Asked asked={book.asked} /> : null}
            <p className={c.said} data-testid="codebook-sentence">
              <Prose text={book.sentence} />
            </p>
          </div>
        </StaleVeil>
      ) : null}
      <Actions>
        <RecordButton
          disabled={!book || !current || card.pending}
          onClick={() => {
            if (book && current)
              card.record(codebookDecision(book) as unknown as Decision, "import");
          }}
          title="Records the codebook read above."
          testId="card-record"
        >
          {book ? `Record: import ${book.name}` : "Read a codebook above"}
        </RecordButton>
      </Actions>
      {card.answerAt?.key === "import" ? card.answerAt.node : null}
    </>
  );
}

export const CodebookCard: LineCard = ({ pid, line }) => {
  const card = useCard(pid, "import_codebook");
  const view = useProjectView(pid).data;
  return (
    <Frame line={line}>
      {card.state && view ? (
        <Body key={line.decision_id ?? "open"} pid={pid} line={line} view={view} card={card} />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
