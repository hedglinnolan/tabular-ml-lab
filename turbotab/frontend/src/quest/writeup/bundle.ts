/**
 * What Write-up draws from the export's own files (pure): each results table from its Markdown,
 * which holds the table as the paper prints it (its title, caption and rounded cells:
 * turbotab/core/export/tables.py `markdown`), and each figure from its SVG, which names itself
 * (its <title>) and carries its caption (its <desc>: figures.py `_svg`). The files are read as the
 * zip holds them, under the one folder every entry is in; nothing here is computed again.
 */

export interface ExportTable {
  /** The file under the bundle's folder, "results/table2.md". */
  file: string;
  title: string;
  caption: string;
  header: string[];
  rows: string[][];
  /** What the file says in place of rows when there are none ("No rows."); null when it has rows. */
  empty: string | null;
}

export interface ExportFigure {
  file: string;
  title: string;
  caption: string;
  /** The SVG as a data URI, drawn by an <img>, so nothing in the file runs. */
  src: string;
}

export interface ExportContents {
  tables: ExportTable[];
  figures: ExportFigure[];
}

/** One row of a Markdown table: its cells between the pipes, an escaped pipe (`\|`) kept as one. */
function cells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/(?<!\\)\|$/, "")
    .split(/(?<!\\)\|/)
    .map((c) => c.trim().replaceAll("\\|", "|"));
}

/** A results table's Markdown: "### title", its caption, then the table or "_No rows._". */
export function readTable(file: string, markdown: string): ExportTable {
  const lines = markdown.split("\n");
  const head = lines.findIndex((l) => l.startsWith("### "));
  const title = head >= 0 ? lines[head]!.slice(4).trim() : file;
  const body = lines.slice(head + 1);
  const first = body.findIndex((l) => l.startsWith("|"));
  const noRows = body.findIndex((l) => /^_.*_$/.test(l.trim()));
  const stop = first >= 0 ? first : noRows >= 0 ? noRows : body.length;
  const caption = body
    .slice(0, stop)
    .map((l) => l.trim())
    .filter(Boolean)
    .join(" ");
  if (first < 0) {
    return { file, title, caption, header: [], rows: [], empty: noRows >= 0 ? body[noRows]!.trim().slice(1, -1) : null };
  }
  const table = body.slice(first).filter((l) => l.startsWith("|"));
  const [header, , ...rows] = table;
  return { file, title, caption, header: cells(header!), rows: rows.map(cells), empty: null };
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/** A figure's SVG: its title and caption as the file states them, and the file to draw. */
export function readFigure(file: string, svg: Uint8Array): ExportFigure {
  const doc = new DOMParser().parseFromString(new TextDecoder().decode(svg), "image/svg+xml");
  const said = (name: string) =>
    [...doc.documentElement.children].find((e) => e.localName === name)?.textContent?.trim() ?? "";
  return { file, title: said("title"), caption: said("desc"), src: `data:image/svg+xml;base64,${base64(svg)}` };
}

/** The folder every entry is in ("turbotab-export/"), or "" when they share none. */
function folderOf(names: string[]): string {
  const first = names[0]?.split("/")[0];
  return first && names.every((n) => n.startsWith(`${first}/`)) ? `${first}/` : "";
}

/** The bundle's results tables in the order its README lists them (by file), and its figures in
 *  the order they are numbered (by title). */
export function readContents(files: ReadonlyMap<string, Uint8Array>): ExportContents {
  const folder = folderOf([...files.keys()]);
  const named = [...files.entries()].map(([name, bytes]) => [name.slice(folder.length), bytes] as const);
  const tables = named
    .filter(([name]) => /^results\/[^/]+\.md$/.test(name))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([name, bytes]) => readTable(name, new TextDecoder().decode(bytes)));
  const figures = named
    .filter(([name]) => /^figures\/[^/]+\.svg$/.test(name))
    .map(([name, bytes]) => readFigure(name, bytes))
    .sort((a, b) => a.title.localeCompare(b.title, "en", { numeric: true }));
  return { tables, figures };
}
