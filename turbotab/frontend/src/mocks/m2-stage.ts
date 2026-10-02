/**
 * The M2 stage's mock API (dev:mock only): nine projects, each a scenario of the stage lab
 * (/lab/stage/m2), every number from the real server (src/mocks/m2-stage-fixture.json, written by
 * docs/turbotab-next/m2/stage/capture_stage_fixture.py).
 *
 * Each project replays what the real server answered on its journey: its view and the stage
 * artifacts at each captured point ("snapshots"), the preview of every option the lab offers, the
 * finding evidence, and, for the answers the capture recorded, the snapshot the project moves to —
 * so recording an option, opening the seal or changing an answer after the opening shows exactly
 * what the server showed. An option the capture did not record is refused, saying so.
 */
import { http, HttpResponse, sse, type HttpHandler } from "msw";
import type { PreviewResult } from "../api/m1-stage-types";
import type { Decision, ProjectView } from "../api/schema";
import raw from "./m2-stage-fixture.json";

interface Snapshot {
  view: ProjectView;
  artifacts: Record<string, unknown>;
}

export interface M2Preview {
  group: string;
  label: string;
  decision: Decision & Record<string, unknown>;
  status: number;
  body: PreviewResult | { error: unknown };
}

export interface M2Project {
  label: string;
  source: string;
  snapshots: Record<string, Snapshot>;
  start: string;
  previews: M2Preview[];
  records: { decision: Decision & Record<string, unknown>; to: string; label: string }[];
  evidence: Record<string, PreviewResult>;
}

export const M2_STAGE = raw as unknown as { meta: Record<string, string>; projects: Record<string, M2Project> };

/** A decision's identity, independent of key order and of how a number was written (0 or 0.0). */
export function decisionKey(d: unknown): string {
  const canon = (v: unknown): unknown =>
    Array.isArray(v)
      ? v.map(canon)
      : v && typeof v === "object"
        ? Object.fromEntries(
            Object.entries(v as Record<string, unknown>)
              .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
              .map(([k, x]) => [k, canon(x)]),
          )
        : v;
  return JSON.stringify(canon(d));
}

function refusal(code: string, message: string) {
  return { error: { code, message, exits: [] } };
}

class Replay {
  current: string;
  constructor(readonly p: M2Project) {
    this.current = p.start;
  }
  get snap(): Snapshot {
    return this.p.snapshots[this.current]!;
  }
  /** A newer snapshot is newer to the client too: its stage statuses carry this moment's stamp. */
  moveTo(name: string): ProjectView {
    this.current = name;
    const now = new Date().toISOString();
    const view = this.snap.view;
    for (const s of Object.values(view.stages)) if (s) s.updated_at = now;
    return view;
  }
}

export function m2StageHandlers(): HttpHandler[] {
  const replays = new Map(Object.entries(M2_STAGE.projects).map(([pid, p]) => [pid, new Replay(p)]));
  const later = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const handlers: HttpHandler[] = [];
  for (const [pid, r] of replays) {
    const base = `/api/projects/${pid}`;
    const previews = new Map(r.p.previews.map((pv) => [decisionKey(pv.decision), pv]));
    const records = new Map(r.p.records.map((rec) => [decisionKey(rec.decision), rec]));
    handlers.push(
      http.get(base, () => HttpResponse.json(r.snap.view)),
      http.post(`${base}/preview`, async ({ request }) => {
        const d = await request.json();
        await later(30 + Math.random() * 40);
        const hit = previews.get(decisionKey(d));
        if (!hit)
          return HttpResponse.json(refusal("not_captured", "The stage lab holds no preview for this option."), {
            status: 422,
          });
        return HttpResponse.json(hit.body, { status: hit.status });
      }),
      http.post(`${base}/decisions`, async ({ request }) => {
        const d = await request.json();
        const rec = records.get(decisionKey(d));
        if (!rec)
          return HttpResponse.json(
            refusal("not_captured", "The stage lab replays only the answers its capture recorded."),
            { status: 409 },
          );
        await later(60);
        return HttpResponse.json(r.moveTo(rec.to));
      }),
      http.get(`${base}/stages/:stage`, ({ params }) => {
        const stage = String(params.stage);
        const status = r.snap.view.stages[stage];
        if (!status) return HttpResponse.json(refusal("not_found", "No such stage."), { status: 404 });
        return HttpResponse.json({
          stage,
          key: status.key,
          fresh: status.status === "fresh",
          status: status.status,
          artifact: r.snap.artifacts[stage] ?? null,
        });
      }),
      http.get(`${base}/findings/:fid/evidence`, async ({ params }) => {
        await later(25);
        const ev = r.p.evidence[decodeURIComponent(String(params.fid))];
        return ev
          ? HttpResponse.json(ev)
          : HttpResponse.json(refusal("not_found", "The stage lab holds no evidence for this finding."), {
              status: 404,
            });
      }),
      // Nothing streams: every change the lab makes is answered by the request that made it.
      sse(`${base}/events`, () => {}),
    );
  }
  return handlers;
}
