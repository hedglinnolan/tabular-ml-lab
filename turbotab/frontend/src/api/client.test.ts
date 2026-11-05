import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  RefusalError,
  api,
  isRefusalError,
  parseRefusal,
  UNAUTHENTICATED_EVENT,
  signInUrl,
} from "./client";
import type { ProjectView, Refusal } from "./schema";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** The query parameters the served contract gives the decisions route (POST …/decisions). */
const DECIDE_QUERY = (
  JSON.parse(
    readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../server/openapi.json"), "utf-8"),
  ) as { paths: Record<string, { post?: { parameters?: { name: string; in: string }[] } }> }
).paths["/api/projects/{pid}/decisions"]!.post!.parameters!.filter((p) => p.in === "query").map((p) => p.name);

function respond(status: number, body: unknown) {
  const fetchMock = vi.fn(async () => {
    const text = typeof body === "string" ? body : JSON.stringify(body);
    return new Response(text, { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const refusal: Refusal = {
  error: {
    code: "task_mismatch",
    message: "`hba1c` has 57 distinct values; a binary task needs exactly two.",
    exits: [
      {
        label: "Model it as regression",
        decision: { kind: "set_task", column: "hba1c", task: "regression" },
      },
      { label: "Keep the current answer", decision: null },
    ],
  },
};

describe("the methods' sections (P8)", () => {
  it("are read from the project's route", async () => {
    const sections = { guideline: "TRIPOD+AI", seen_from: null, sections: [], count: 0 };
    const fetchMock = respond(200, sections);
    await expect(api.methodsSections("p 1")).resolves.toEqual(sections);
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/projects/p%201/methods/sections");
  });
});

/** W1-R finding 3c: the server words the roles card by the project's goal, asked as a plain query
 *  parameter; with no goal named it serves the words as they stand. */
describe("the teaching words, by goal", () => {
  it("names the project's goal as ?purpose=", async () => {
    const fetchMock = respond(200, []);
    await api.teaching("prediction");
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/teaching?purpose=prediction");
  });

  it("names none where the goal is not chosen", async () => {
    const fetchMock = respond(200, []);
    await api.teaching(null);
    await api.teaching();
    const urls = fetchMock.mock.calls.map((c) => (c as unknown as [string])[0]);
    expect(urls).toEqual(["/api/teaching", "/api/teaching"]);
  });
});

describe("a decision the server refuses (409)", () => {
  it("is thrown as a RefusalError carrying the typed refusal and its exits", async () => {
    respond(409, refusal);
    const err = await api
      .decide("p1", { kind: "set_task", column: "hba1c", task: "binary" })
      .catch((e) => e);
    expect(isRefusalError(err)).toBe(true);
    expect(err).toBeInstanceOf(RefusalError);
    const r = (err as RefusalError).refusal;
    expect(r.error.code).toBe("task_mismatch");
    expect(r.error.message).toContain("binary task needs exactly two");
    expect(r.error.exits).toHaveLength(2);
    expect(r.error.exits[0]!.decision).toEqual({
      kind: "set_task",
      column: "hba1c",
      task: "regression",
    });
    expect(r.error.exits[1]!.decision).toBeNull();
    expect((err as Error).message).toBe(refusal.error.message);
  });

  it("posts the decision as the JSON body to the project's decisions route", async () => {
    const view = { decisions: [] } as unknown as ProjectView;
    const fetchMock = respond(200, view);
    await api.decide("p 1", { kind: "set_target", column: "hba1c" });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/projects/p%201/decisions");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ kind: "set_target", column: "hba1c" });
  });

  it("treats a 409 without the refusal shape as a plain ApiError, not a refusal", async () => {
    respond(409, { detail: "conflict" });
    const err = await api
      .decide("p1", { kind: "set_purpose", purpose: "prediction" })
      .catch((e) => e);
    expect(isRefusalError(err)).toBe(false);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
    expect((err as ApiError).message).toBe("conflict");
  });

  it("reads a refusal with no exits as an empty exit list", async () => {
    respond(409, { error: { code: "lens_empty", message: "Pick one." } });
    const err = (await api
      .decide("p1", { kind: "set_lens", lenses: [] })
      .catch((e) => e)) as RefusalError;
    expect(err.refusal.error.exits).toEqual([]);
  });
});

describe("a decision's reason (the decider's note, G14)", () => {
  const decision = { kind: "set_target", column: "hba1c" } as const;
  const urlOf = (call: unknown) => new URL((call as [string])[0], "http://localhost");

  it("rides as the decisions route's note, in the person's words, and only when given", async () => {
    // The parameter the server takes is the served contract's, not the client's own name.
    expect(DECIDE_QUERY).toContain("note");
    const reason = "The clinic reads 7% & over as \"high\" + flags it.";
    const fetchMock = respond(200, { decisions: [] });
    await api.decide("p 1", decision, { note: reason });
    await api.decide("p 1", decision);
    await api.decide("p 1", decision, { note: "" });
    const [noted, plain, blank] = fetchMock.mock.calls.map(urlOf);
    expect(noted!.pathname).toBe("/api/projects/p%201/decisions");
    expect(noted!.searchParams.get("note")).toBe(reason);
    expect([...noted!.searchParams.keys()].every((k) => DECIDE_QUERY.includes(k))).toBe(true);
    // No reason, no parameter: the record's note is none.
    expect(plain!.search).toBe("");
    expect(blank!.search).toBe("");
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual(decision);
  });

  it("goes with decide_now when both are asked for", async () => {
    const fetchMock = respond(200, { decisions: [] });
    await api.decide("p1", decision, { decideNow: true, note: "Answered ahead." });
    const url = urlOf(fetchMock.mock.calls[0]);
    expect(url.searchParams.get("decide_now")).toBe("true");
    expect(url.searchParams.get("note")).toBe("Answered ahead.");
  });
});

describe("other failures", () => {
  it("surface the server's error message and status", async () => {
    respond(404, { error: { code: "not_found", message: "No such project.", exits: [] } });
    const err = await api.project("nope").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).not.toBeInstanceOf(RefusalError);
    expect((err as ApiError).status).toBe(404);
    expect((err as ApiError).message).toBe("No such project.");
  });

  it("encode table windows as offset, limit and a comma-separated column list", async () => {
    const fetchMock = respond(200, { columns: [], rows: [], total_rows: 0, offset: 0 });
    await api.table("p1", { offset: 50, limit: 25, columns: ["age", "bmi"] });
    const [url] = fetchMock.mock.calls[0] as unknown as [string];
    expect(url).toBe("/api/projects/p1/table?offset=50&limit=25&columns=age%2Cbmi");
  });
});

describe("parseRefusal", () => {
  it("rejects bodies that are not refusals", () => {
    expect(parseRefusal(null)).toBeNull();
    expect(parseRefusal("text")).toBeNull();
    expect(parseRefusal({ error: "x" })).toBeNull();
    expect(parseRefusal({ error: { code: 1, message: "m" } })).toBeNull();
    expect(parseRefusal({ error: { code: "c", message: "m", exits: "no" } })).toBeNull();
    expect(
      parseRefusal({ error: { code: "c", message: "m", exits: [{ nolabel: 1 }] } }),
    ).toBeNull();
    expect(
      parseRefusal({ error: { code: "c", message: "m", exits: [{ label: "x", decision: 3 }] } }),
    ).toBeNull();
  });

  /** G12: a way forward names the card it opens, `target` {stage, key} with a null key for a
   *  stage's open line (openapi.json `ExitTarget`; core/exit_target.py `toward`), as the server's
   *  409 serves it (server/tests/test_fit_press_holds_api.py pins Fit's). */
  const ACT = { label: "Act on it in Your data: `sas_zeros`", decision: null, target: { stage: "data", key: "sas_zeros" } };
  const TRIAGE = { label: "Triage the open noticings in Models", decision: null, target: { stage: "models", key: null } };
  const LIMIT = {
    label: "Record as a limitation: `sas_zeros`",
    decision: { kind: "confirm_sweep", stage: "models", sweep: "noticings", lines: [{ id: "finding:sas_zeros", key: "sas_zeros", value: "could_bias" }] },
    target: { stage: "models", key: null },
  };

  it("keeps each way forward's target, a decision's included", () => {
    const parsed = parseRefusal({ error: { code: "triage_open", message: "m", exits: [ACT, TRIAGE, LIMIT] } });
    expect(parsed?.error.exits).toEqual([ACT, TRIAGE, LIMIT]);
  });

  it("drops a target not in the served shape, keeping its way forward, and adds none where none is served", () => {
    const bad = [
      { label: "a", decision: null, target: { stage: 3, key: null } },
      { label: "b", decision: null, target: { stage: "data", key: 4 } },
      { label: "c", decision: null, target: "data" },
      { label: "d", decision: null, target: null },
      { label: "e", decision: null, target: { key: "x" } },
      { label: "f", decision: null },
    ];
    const parsed = parseRefusal({ error: { code: "c", message: "m", exits: bad } });
    expect(parsed?.error.exits).toEqual(bad.map(({ label }) => ({ label, decision: null })));
  });

  it("reaches a press's RefusalError with its targets, as fetch answers the 409", async () => {
    respond(409, { error: { code: "triage_open", message: "Fit waits.", exits: [ACT, LIMIT] } });
    const err = (await api.pressFit("p1").catch((e) => e)) as RefusalError;
    expect(isRefusalError(err)).toBe(true);
    expect(err.refusal.error.exits.map((x) => x.target)).toEqual([ACT.target, LIMIT.target]);
  });
});

describe("a request without a session (server mode, 401)", () => {
  it("announces that the session ended and still rejects the call", async () => {
    respond(401, { error: { code: "unauthenticated", message: "Sign in.", exits: [] } });
    const signIn = vi.fn();
    window.addEventListener(UNAUTHENTICATED_EVENT, signIn);
    try {
      const err = await api.listProjects().catch((e) => e);
      expect(signIn).toHaveBeenCalledTimes(1);
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(401);
      respond(409, refusal);
      await api.listProjects().catch(() => null);
      expect(signIn).toHaveBeenCalledTimes(1); // only a 401 means the session ended
    } finally {
      window.removeEventListener(UNAUTHENTICATED_EVENT, signIn);
    }
  });

  it("names where to come back to as one encoded parameter", () => {
    expect(signInUrl("/projects/p1?tab=rows&x=1")).toBe(
      "/login?next=%2Fprojects%2Fp1%3Ftab%3Drows%26x%3D1",
    );
  });
});
