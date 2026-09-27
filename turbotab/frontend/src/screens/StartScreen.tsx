/**
 * Start: open a table from this computer (local mode) or upload one, or return
 * to a recent project. Opening a file creates a project; reading it starts at once.
 */
import { useId, useRef, useState, type DragEvent } from "react";
import { useFsListing, useHealth, useOpenPath, useProjects, useUpload } from "../api/queries";
import type { FsEntry } from "../api/schema";
import { Header } from "../components/Header";
import { Link, navigate, projectPath } from "../router";
import { cx, fmtBytes, fmtInt, fmtWhen } from "../util/format";
import styles from "./StartScreen.module.css";

const TABLE_EXT = /\.(csv|tsv|parquet|xlsx)$/i;

function Crumbs({ path, onGo }: { path: string; onGo: (p: string) => void }) {
  const parts = path.split("/").filter(Boolean);
  return (
    <nav aria-label="Folder" className={styles.crumbs}>
      <ol>
        <li>
          <button type="button" onClick={() => onGo("/")}>
            /
          </button>
        </li>
        {parts.map((p, i) => {
          const to = "/" + parts.slice(0, i + 1).join("/");
          const last = i === parts.length - 1;
          return (
            <li key={to}>
              {last ? (
                <span aria-current="location">{p}</span>
              ) : (
                <button type="button" onClick={() => onGo(to)}>
                  {p}
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function FileBrowser() {
  const [path, setPath] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const listing = useFsListing(path);
  const open = useOpenPath();
  const inputId = useId();

  const openFile = (p: string) =>
    open.mutate(p, { onSuccess: (proj) => navigate(projectPath(proj.id)) });

  const entries = listing.data?.entries ?? [];
  const dirs = entries.filter((e) => e.is_dir);
  const tables = entries.filter((e) => !e.is_dir && TABLE_EXT.test(e.name));
  const others = entries.length - dirs.length - tables.length;

  const row = (e: FsEntry) =>
    e.is_dir ? (
      <li key={e.path}>
        <button type="button" className={styles.entry} onClick={() => setPath(e.path)}>
          <span className={styles.folder} aria-hidden="true" />
          <span className={styles.entryName}>{e.name}</span>
          <span className={styles.entryMeta}>folder</span>
        </button>
      </li>
    ) : (
      <li key={e.path}>
        <button
          type="button"
          className={cx(styles.entry, styles.file)}
          onClick={() => openFile(e.path)}
          disabled={open.isPending}
          aria-label={`Open ${e.name}`}
        >
          <span className={styles.fileIcon} aria-hidden="true" />
          <span className={styles.entryName}>{e.name}</span>
          <span className={styles.entryMeta}>{fmtBytes(e.size)}</span>
          <span className={styles.openHint}>Open</span>
        </button>
      </li>
    );

  return (
    <section className={styles.panel} aria-labelledby="browse-h">
      <h2 id="browse-h" className={styles.h2}>
        From this computer
      </h2>
      <p className={styles.sub}>
        TurboTab reads the file where it is. Nothing is copied off this machine.
      </p>
      {listing.data ? <Crumbs path={listing.data.path} onGo={setPath} /> : null}
      <div className={styles.browser}>
        {listing.isPending ? <p className={styles.empty}>Reading the folder…</p> : null}
        {listing.isError ? (
          <p className={styles.empty}>This folder could not be read: {listing.error.message}</p>
        ) : null}
        {listing.data ? (
          <ul className={styles.entries} data-testid="fs-entries">
            {listing.data.parent ? (
              <li>
                <button
                  type="button"
                  className={styles.entry}
                  onClick={() => setPath(listing.data.parent)}
                >
                  <span className={styles.up} aria-hidden="true">
                    ..
                  </span>
                  <span className={styles.entryName}>Up one folder</span>
                </button>
              </li>
            ) : null}
            {dirs.map(row)}
            {tables.map(row)}
            {entries.length === 0 ? <li className={styles.empty}>This folder is empty.</li> : null}
          </ul>
        ) : null}
        {others > 0 ? (
          <p className={styles.others}>
            {fmtInt(others)} other {others === 1 ? "file is" : "files are"} not a table TurboTab
            reads (CSV, TSV, Parquet, Excel).
          </p>
        ) : null}
      </div>
      <form
        className={styles.pathForm}
        onSubmit={(e) => {
          e.preventDefault();
          if (typed.trim()) openFile(typed.trim());
        }}
      >
        <label htmlFor={inputId} className={styles.pathLabel}>
          Or type a path
        </label>
        <input
          id={inputId}
          className={styles.pathInput}
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder="/path/to/table.csv"
          spellCheck={false}
        />
        <button
          type="submit"
          className={styles.secondary}
          disabled={!typed.trim() || open.isPending}
        >
          Open
        </button>
      </form>
      {open.isError ? (
        <p className={styles.error} role="alert">
          {open.error.message}
        </p>
      ) : null}
    </section>
  );
}

function UploadZone({ only }: { only: boolean }) {
  const upload = useUpload();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const send = (file: File | undefined) => {
    if (!file) return;
    upload.mutate(file, { onSuccess: (proj) => navigate(projectPath(proj.id)) });
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    send(e.dataTransfer.files[0]);
  };
  return (
    <section className={styles.panel} aria-labelledby="upload-h">
      <h2 id="upload-h" className={styles.h2}>
        {only ? "Upload a table" : "Or upload one"}
      </h2>
      <p className={styles.sub}>
        {only
          ? "This server reads uploaded files. The file is streamed to its disk."
          : "Useful when the file is somewhere this computer cannot browse."}
      </p>
      <div
        className={cx(styles.drop, over && styles.dropOver)}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        data-testid="dropzone"
      >
        <p className={styles.dropText}>Drop a CSV, TSV, Parquet or Excel file here</p>
        <button
          type="button"
          className={styles.secondary}
          onClick={() => inputRef.current?.click()}
          disabled={upload.isPending}
        >
          {upload.isPending ? "Uploading…" : "Choose a file"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.tsv,.txt,.parquet,.xlsx"
          className="visually-hidden"
          tabIndex={-1}
          aria-hidden="true"
          onChange={(e) => send(e.target.files?.[0])}
        />
      </div>
      {upload.isError ? (
        <p className={styles.error} role="alert">
          {upload.error.message}
        </p>
      ) : null}
    </section>
  );
}

function Recent() {
  const projects = useProjects();
  if (!projects.data?.length) return null;
  return (
    <section className={styles.recent} aria-labelledby="recent-h">
      <h2 id="recent-h" className={styles.h2}>
        Recent projects
      </h2>
      <ul className={styles.recentList}>
        {projects.data.map((p) => (
          <li key={p.id}>
            <Link href={projectPath(p.id)} className={styles.recentItem}>
              <span className={styles.recentName}>{p.name}</span>
              <span className={styles.recentSource}>{p.source_name}</span>
              <span className={styles.recentSize}>
                {p.n_rows !== null && p.n_cols !== null
                  ? `${fmtInt(p.n_rows)} × ${fmtInt(p.n_cols)}`
                  : "reading…"}
              </span>
              <span className={styles.recentWhen}>{fmtWhen(p.created_at)}</span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function StartScreen() {
  const health = useHealth();
  const local = health.data?.mode !== "server";
  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.intro}>
          <h1 className={styles.h1}>Open a table to begin.</h1>
          <p className={styles.lede}>
            TurboTab asks one question at a time and writes each answer down as a sentence you could
            publish. The record it builds is the start of your methods section.
          </p>
        </div>
        <div className={cx(styles.grid, !local && styles.single)}>
          {health.isPending ? null : local ? <FileBrowser /> : null}
          {health.isPending ? null : <UploadZone only={!local} />}
        </div>
        <Recent />
      </main>
    </>
  );
}
