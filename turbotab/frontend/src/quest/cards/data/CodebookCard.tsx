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
 */
import { useCodebook, useHealth } from "../../../api/queries";
import type { CodebookPreview } from "../../../api/m3-types";
import type { Decision, DecisionInput } from "../../../api/schema";
import { Pending } from "../../../components/record/blocks";
import { Actions, RecordButton } from "../../../components/record/ask/common";
import g from "../../../components/record/generic/generic.module.css";
import { Prose, V } from "../../../components/Prose";
import { fmtInt, listJoin } from "../../../util/format";
import type { LineCard } from "../../contracts";
import { Frame, useCard, type CardWiring } from "../predict/card";
import c from "../predict/declaration.module.css";
import { FileSource } from "./JoinCard";

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

function Body({ pid, card }: { pid: string; card: CardWiring }) {
  const read = useCodebook(pid);
  const local = useHealth().data?.mode === "local";
  const books = Object.values(card.state?.codebooks ?? {}).map((b) => b.name);
  const book = read.data ?? null;
  return (
    <>
      {books.length ? (
        <p className={c.said} data-testid="imported">
          Imported: {listJoin(books)}
        </p>
      ) : null}
      <fieldset className={c.group}>
        <legend className={c.legend}>The codebook</legend>
        <FileSource
          drop="Drop a codebook here: a variable table, an NHANES codebook page or an XPT file"
          local={local}
          pending={read.isPending}
          error={read.error}
          pathAction="Read"
          onFile={(file) => read.mutate({ file })}
          onPath={(path) => read.mutate({ path, labels: false })}
          testId="codebook-source"
        />
      </fieldset>
      {read.isPending ? <Pending>Reading the codebook…</Pending> : null}
      {book ? (
        <div className={c.group}>
          <Counts p={book} />
          {book.asked.length ? <Asked asked={book.asked} /> : null}
          <p className={c.said} data-testid="codebook-sentence">
            <Prose text={book.sentence} />
          </p>
        </div>
      ) : null}
      <Actions>
        <RecordButton
          disabled={!book || card.pending}
          onClick={() => {
            if (book) card.record(codebookDecision(book) as unknown as Decision, "import");
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
  return (
    <Frame line={line}>
      {card.state ? (
        <Body key={line.decision_id ?? "open"} pid={pid} card={card} />
      ) : (
        <Pending>Loading the project…</Pending>
      )}
    </Frame>
  );
};
