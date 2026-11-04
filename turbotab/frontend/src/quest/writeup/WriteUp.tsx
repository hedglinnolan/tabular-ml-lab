/**
 * Write-up's card column, read-only (FOUNDATION §9; the owner's Q5, 2026-11-04: "Write-up shows the
 * export's tables, figures, plan and checklist read-only now"): what the export holds, as the server
 * serves it, in this order (what the export waits for first, as §9 orders it; the rest is the
 * plainest order, for the owner's review):
 *
 * 1. The checklist of the declared purpose (GET …/checklist): its title, how many items the record
 *    answers, and what the export still waits for (§9's first item), each in the server's words;
 *    then, behind one disclosure, every item by its section, headed as the bundle's checklist file
 *    heads it (TRIPOD+AI's with its scope, STROBE's with its table), the item quoted from its
 *    source, its note (answered, or what the author must supply), and the sentences, paragraphs or
 *    captions that answer it.
 * 2. The results tables and 3. the figures, from the bundle itself (reads.ts): each table's title,
 *    caption and cells as its Markdown file writes them, each figure drawn from its SVG with the
 *    title and caption the file holds (bundle.ts). They are read only once nothing waits: until
 *    then, or where the export refuses or this browser cannot open it, one line says so. What was
 *    drawn stays, veiled and inert, until the export is read for the record as it stands (while it
 *    is read again, waits or refuses, that line above it), the veil saying what moved since, or,
 *    where the view shows nothing that moved, only what the export did (reads.ts `veilOf`).
 * 4. The analysis plan (GET …/plan), behind one disclosure: its text, the decisions it holds, and
 *    each made after the estimates were seen.
 *
 * Nothing here is edited or recorded by the card itself; reading the bundle counts its scores and
 * estimates as shown, because they are (reads.ts).
 */
import { useId, useState } from "react";
import { isRefusalError } from "../../api/client";
import { useChecklist, usePlan, useProjectView } from "../../api/queries";
import type { PlanExport } from "../../api/m3-types";
import type { ChecklistItem, ChecklistReport, ProjectView } from "../../api/schema";
import { Plain } from "../../kit";
import { StaleVeil } from "../../motion/StaleVeil";
import q from "../quest.module.css";
import type { ExportFigure, ExportTable } from "./bundle";
import { basisOf, useChecklistFollows, useExportContents, veilOf } from "./reads";
import { canReadZip } from "./zip";
import w from "./writeup.module.css";

type ExportState = ReturnType<typeof useExportContents>;

export function WriteUp({ pid }: { pid: string }) {
  const report = useChecklist(pid).data;
  const plan = usePlan(pid).data;
  const view = useProjectView(pid).data;
  const basis = basisOf(view);
  useChecklistFollows(pid, basis?.key ?? null);
  const ready = !!report && report.waiting.length === 0;
  const [readable] = useState(canReadZip);
  const read = useExportContents(pid, basis, ready && readable);
  return (
    <>
      {report ? <Checklist report={report} /> : null}
      {report ? <Exhibits ready={ready} readable={readable} read={read} view={view} /> : null}
      {plan ? <Plan plan={plan} /> : null}
    </>
  );
}

/** The checklist's title, counts and what the export waits for, then every item behind one disclosure. */
function Checklist({ report }: { report: ChecklistReport }) {
  const { counts } = report;
  const sections = new Map<string, ChecklistItem[]>();
  for (const item of report.items) sections.set(item.section, [...(sections.get(item.section) ?? []), item]);
  return (
    <section className={q.group} data-kit data-testid="checklist">
      <h3 className={q.groupLabel}>{report.title}</h3>
      <p className={q.lineQuiet}>
        {counts.answered} answered, {counts.partly_answered} partly answered, {counts.unanswered} unanswered,
        of {counts.items} items.
      </p>
      {report.waiting.length ? (
        <ul className={q.recordList}>
          {report.waiting.map((x) => (
            <li key={x}>
              <Plain text={x} />
            </li>
          ))}
        </ul>
      ) : null}
      <details className={q.record} data-testid="checklist-items">
        <summary>Every item · {report.items.length}</summary>
        <p className={w.text}>
          <Plain text={report.citation} />
        </p>
        {[...sections].map(([section, items]) => (
          <section key={section} className={w.section} data-testid="checklist-section">
            <h4 className={w.sectionName}>{section}</h4>
            <ol className={w.items}>
              {items.map((item) => (
                <Item key={item.id} report={report} item={item} />
              ))}
            </ol>
          </section>
        ))}
      </details>
    </section>
  );
}

/** An item's label as the bundle's checklist file writes it (core/export/checklists.py `markdown`):
 *  TRIPOD+AI's with its scope, a STROBE item with its table's name, a STROBE-nut item by its id. */
function labelOf(report: ChecklistReport, item: ChecklistItem): string {
  if (report.checklist === "TRIPOD+AI") return `${item.id} (${item.scope ?? ""})`;
  return item.kind === "STROBE" ? `${item.kind} ${item.id}` : item.id;
}

function Item({ report, item }: { report: ChecklistReport; item: ChecklistItem }) {
  return (
    <li className={w.item} data-testid="checklist-item" data-item={item.id} data-status={item.status}>
      <h5 className={w.itemName}>
        {labelOf(report, item)} · {item.topic}
      </h5>
      <p className={w.quoted}>
        <Plain text={item.text} />
      </p>
      <p>
        <Plain text={item.note} />
      </p>
      {item.where.length ? (
        <ul className={w.where}>
          {item.where.map((x, i) => (
            <li key={i}>
              <Plain text={x.quote} />
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

/** The bundle's tables and figures, and the one line that says why they are not drawn, or not
 *  drawn for the record as it stands: what was drawn stays, veiled and inert, below that line until
 *  the export is read for the record as it stands (reads.ts `veilOf`). */
function Exhibits({ ready, readable, read, view }: { ready: boolean; readable: boolean; read: ExportState; view: ProjectView | undefined }) {
  const { drawn, again, error } = read;
  const line = !readable ? (
    <p className={q.lineQuiet} data-kit data-testid="export-unreadable">
      {ready
        ? "This browser cannot open the export here; Download holds its tables and figures."
        : "This browser cannot open the export here; Download will hold its tables and figures once the export waits for nothing."}
    </p>
  ) : !ready ? (
    <p className={q.lineQuiet} data-kit data-testid="export-later">
      {drawn
        ? "The results tables and figures below are the export's as last read; it now waits for what is listed above."
        : "The results tables and figures are shown here once the export waits for nothing."}
    </p>
  ) : error ? (
    isRefusalError(error) ? (
      <p className={q.lineQuiet} data-kit data-testid="export-refused">
        <Plain text={error.refusal.error.message} />
      </p>
    ) : (
      <p className={q.lineQuiet} data-kit data-testid="export-failed">
        The export could not be read here: {error.message}
      </p>
    )
  ) : null;
  if (!drawn) return line;
  const { contents } = drawn;
  const veil = veilOf(drawn.basis, view, { ready, again, failed: error ? (isRefusalError(error) ? "refused" : "unread") : null });
  return (
    <>
      {line}
      <StaleVeil state={veil.state} tag={veil.tag} testId="export-veil">
        {contents.tables.length ? (
          <section className={q.group} data-kit data-testid="export-tables">
            <h3 className={q.groupLabel}>Results tables</h3>
            {contents.tables.map((t) => (
              <Table key={t.file} table={t} />
            ))}
          </section>
        ) : null}
        {contents.figures.length ? (
          <section className={q.group} data-kit data-testid="export-figures">
            <h3 className={q.groupLabel}>Figures</h3>
            {contents.figures.map((f) => (
              <Figure key={f.file} figure={f} />
            ))}
          </section>
        ) : null}
      </StaleVeil>
    </>
  );
}

function Table({ table }: { table: ExportTable }) {
  const id = useId();
  return (
    <section className={w.exhibit} data-testid="export-table" data-file={table.file}>
      <h4 id={id} className={w.title}>
        <Plain text={table.title} />
      </h4>
      <p className={w.caption}>
        <Plain text={table.caption} />
      </p>
      {table.header.length ? (
        <div className={w.scroll}>
          <table className={w.table} aria-labelledby={id}>
            <thead>
              <tr>
                {table.header.map((h, i) => (
                  <th key={i} scope="col">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={q.lineQuiet}>{table.empty}</p>
      )}
    </section>
  );
}

function Figure({ figure }: { figure: ExportFigure }) {
  return (
    <figure className={w.figure} data-file={figure.file}>
      <img className={w.drawing} src={figure.src} alt={figure.title} />
      <figcaption className={w.caption}>
        <Plain text={figure.caption} />
      </figcaption>
    </figure>
  );
}

function Plan({ plan }: { plan: PlanExport }) {
  return (
    <details className={q.record} data-kit data-testid="export-plan">
      <summary>The analysis plan</summary>
      <p className={w.text}>
        <Plain text={plan.text} />
      </p>
      <ol className={q.recordList}>
        {plan.sentences.map((s, i) => (
          <li key={i}>
            <Plain text={s} />
          </li>
        ))}
      </ol>
      {plan.after_estimates.length ? (
        <>
          <p className={w.sublabel}>Made after the estimates were seen</p>
          <ol className={q.recordList}>
            {plan.after_estimates.map((a, i) => (
              <li key={i}>
                <Plain text={typeof a.sentence === "string" ? a.sentence : ""} />
              </li>
            ))}
          </ol>
        </>
      ) : null}
    </details>
  );
}
