/**
 * MSW handlers implementing the M0 contract in memory. Browser-only: MSW's `sse`
 * needs a real EventSource, so these are built by `makeHandlers()` at start-up.
 */
import { http, HttpResponse, sse, type HttpHandler } from "msw";
import type { Decision, Health, Refusal, StageName } from "../api/schema";
import { STAGES } from "../api/schema";
import { parseDelimited } from "./csv";
import { dietaryRecalls, genomicsWide, type MockDataset } from "./datasets";
import { MockServer, type EventType } from "./db";
import { m1RecordHandlers } from "./m1-record";
import { m1StageHandlers } from "./m1-stage";
import { m2StageHandlers } from "./m2-stage";
import { m3Handlers } from "./m3";
import { datasetAt, listDir } from "./fs";
import { columnSummary, findColumn, histogram, isNumericDtype } from "./stats";

const HEALTH: Health = { version: "0.1.0-mock", mode: "local", workers: 3 };

function refusal(code: string, message: string): Refusal {
  return { error: { code, message, exits: [] } };
}

const notFound = (what: string) => HttpResponse.json(refusal("not_found", what), { status: 404 });

export function seed(server: MockServer): void {
  const day = 86_400_000;
  server.createProject(genomicsWide(), "path", {
    instant: true,
    createdAt: new Date(Date.now() - 3 * day).toISOString(),
    decisions: [{ kind: "set_lens", lenses: ["genomics"] }],
  });
  server.createProject(dietaryRecalls(), "upload", {
    instant: true,
    createdAt: new Date(Date.now() - 1 * day).toISOString(),
  });
}

async function datasetFromUpload(file: File): Promise<MockDataset> {
  const name = file.name || "upload.csv";
  if (/\.(csv|tsv|txt)$/i.test(name) && file.size < 20_000_000) {
    const ds = parseDelimited(name, await file.text());
    if (ds.columns.length > 0 && ds.nRows > 0) return ds;
  }
  const ds = /genom|count|omics|rna/i.test(name) ? genomicsWide() : dietaryRecalls();
  return { ...ds, name };
}

export function makeHandlers(server: MockServer): HttpHandler[] {
  return [
    // M3: the replayed reference journeys (m3~<journey>), and the endpoints no other mock answers
    // (readings, methods, plan, models, files, join preview, codebooks) for every project.
    ...m3Handlers(),
    ...m2StageHandlers(), // the M2 stage lab's projects (M2 part 2); literal paths, so first
    // M1 + M2: the Router, teaching, sentences, findings and repairs, the M1 and M2 stages. Each
    // answers only for projects it knows, so the stage lab project falls through to its own.
    ...m1RecordHandlers(server),
    ...m1StageHandlers(), // the stage's NHANES project (M1 part 2), and captured previews
    http.get("/api/health", () => HttpResponse.json(HEALTH)),

    http.get("/api/projects", () => HttpResponse.json(server.listProjects())),

    http.post("/api/projects", async ({ request }) => {
      const body = (await request.json()) as { path?: string };
      const ds = body.path ? datasetAt(body.path) : null;
      if (!ds) {
        return HttpResponse.json(
          refusal("not_a_table", `TurboTab cannot read '${body.path ?? ""}' as a table.`),
          { status: 422 },
        );
      }
      return HttpResponse.json(server.createProject(ds, "path"), { status: 201 });
    }),

    http.post("/api/projects/upload", async ({ request }) => {
      const form = await request.formData();
      const file = form.get("file");
      if (!(file instanceof File)) {
        return HttpResponse.json(refusal("no_file", "The upload carried no file."), {
          status: 422,
        });
      }
      return HttpResponse.json(server.createProject(await datasetFromUpload(file), "upload"), {
        status: 201,
      });
    }),

    http.get("/api/projects/:pid", ({ params }) => {
      const view = server.view(String(params.pid));
      return view ? HttpResponse.json(view) : notFound("No such project.");
    }),

    http.post("/api/projects/:pid/decisions", async ({ params, request }) => {
      const pid = String(params.pid);
      if (!server.get(pid)) return notFound("No such project.");
      const out = server.decide(pid, (await request.json()) as Decision);
      if ("error" in out) return HttpResponse.json(out, { status: 409 });
      return HttpResponse.json(out);
    }),

    http.get("/api/projects/:pid/stages/:stage", ({ params }) => {
      const stage = String(params.stage) as StageName;
      if (!STAGES.includes(stage)) return notFound(`No stage named '${stage}'.`);
      const result = server.stageResult(String(params.pid), stage);
      return result ? HttpResponse.json(result) : notFound("No such project.");
    }),

    http.post("/api/projects/:pid/stages/:stage/run", ({ params }) => {
      const stage = String(params.stage) as StageName;
      if (!STAGES.includes(stage)) return notFound(`No stage named '${stage}'.`);
      const status = server.runStage(String(params.pid), stage);
      return status ? HttpResponse.json(status) : notFound("No such project.");
    }),

    http.get("/api/projects/:pid/table", ({ params, request }) => {
      const pid = String(params.pid);
      const p = server.get(pid);
      if (!p) return notFound("No such project.");
      if (!server.isIngested(pid)) {
        return HttpResponse.json(refusal("not_ingested", "The file is still being read."), {
          status: 409,
        });
      }
      const url = new URL(request.url);
      const offset = Math.max(0, Number(url.searchParams.get("offset") ?? 0));
      const limit = Math.min(1000, Math.max(0, Number(url.searchParams.get("limit") ?? 100)));
      const wanted = url.searchParams.get("columns");
      const cols = wanted
        ? wanted
            .split(",")
            .map((n) => findColumn(p.source, n))
            .filter((c) => c !== undefined)
        : p.source.columns;
      const end = Math.min(p.source.nRows, offset + limit);
      const rows = [];
      for (let i = offset; i < end; i++) rows.push(cols.map((c) => c.values[i] ?? null));
      return HttpResponse.json({
        columns: cols.map((c) => c.name),
        rows,
        total_rows: p.source.nRows,
        offset,
      });
    }),

    http.get("/api/projects/:pid/columns", ({ params }) => {
      const pid = String(params.pid);
      const p = server.get(pid);
      if (!p) return notFound("No such project.");
      if (!server.isIngested(pid)) {
        return HttpResponse.json(refusal("not_ingested", "The file is still being read."), {
          status: 409,
        });
      }
      return HttpResponse.json(p.source.columns.map(columnSummary));
    }),

    http.get("/api/projects/:pid/columns/:name/histogram", ({ params, request }) => {
      const p = server.get(String(params.pid));
      if (!p) return notFound("No such project.");
      const col = findColumn(p.source, decodeURIComponent(String(params.name)));
      if (!col) return notFound("No such column.");
      if (!isNumericDtype(col.dtype)) {
        return HttpResponse.json(refusal("not_numeric", `'${col.name}' is not numeric.`), {
          status: 422,
        });
      }
      const bins = Number(new URL(request.url).searchParams.get("bins") ?? 20) || 20;
      return HttpResponse.json(histogram(col, bins));
    }),

    http.get("/api/projects/:pid/jobs/:jid", ({ params }) => {
      const job = server.job(String(params.pid), String(params.jid));
      return job ? HttpResponse.json(job) : notFound("No such job.");
    }),

    http.post("/api/projects/:pid/jobs/:jid/cancel", ({ params }) => {
      const job = server.cancelJob(String(params.pid), String(params.jid));
      return job ? HttpResponse.json(job) : notFound("No such job.");
    }),

    sse("/api/projects/:pid/events", ({ client, params, request }) => {
      const pid = String(params.pid);
      let unsubscribe = () => {};
      const ping = setInterval(() => send("ping", {}), 15_000);
      function send(type: EventType, data: unknown) {
        try {
          client.send({ event: type, data } as never);
        } catch {
          unsubscribe();
          clearInterval(ping);
        }
      }
      unsubscribe = server.subscribe(pid, send);
      request.signal.addEventListener("abort", () => {
        unsubscribe();
        clearInterval(ping);
      });
    }),

    http.get("/api/fs/list", ({ request }) => {
      const path = new URL(request.url).searchParams.get("path");
      const listing = listDir(path);
      return listing ? HttpResponse.json(listing) : notFound(`No folder at '${path ?? ""}'.`);
    }),
  ];
}
