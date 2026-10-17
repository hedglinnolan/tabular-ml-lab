/**
 * The route switch (plans/W1_PART2_PLAN.md ruling 8): /p/:pid is the quest shell and
 * /p/:pid/classic today's screen, for every project id the start screen links to.
 */
import { classicPath, parseRoute, projectPath } from "./router";

describe("the project routes", () => {
  it.each(["abc123", "m3~clinical~3", "p 1"])("%s: the shell at its path, today's screen at /classic", (pid) => {
    expect(parseRoute(projectPath(pid))).toEqual({ name: "project", pid });
    expect(parseRoute(classicPath(pid))).toEqual({ name: "classic", pid });
    expect(parseRoute(`${classicPath(pid)}/`)).toEqual({ name: "classic", pid });
  });

  it("finds nothing under a project but /classic", () => {
    expect(parseRoute("/p/abc/results")).toEqual({ name: "missing", path: "/p/abc/results" });
    expect(parseRoute("/p/abc/classic/more")).toEqual({ name: "missing", path: "/p/abc/classic/more" });
  });
});
