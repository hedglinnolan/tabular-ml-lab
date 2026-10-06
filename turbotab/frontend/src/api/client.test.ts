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
