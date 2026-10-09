import { fireEvent, render, screen } from "@testing-library/react";
import { EMBEDDING_ONE, EMBEDDING_PCA, syntheticCloud } from "../fixtures";
import { EMBED, layoutEmbedding, nearest, type EmbedLayout } from "./layout";
import { EmbeddingView } from "./EmbeddingView";
import type { EmbeddingInput } from "./types";

const lay = (input: EmbeddingInput, w = 400, focus: string | null = null): EmbedLayout => {
  const r = layoutEmbedding(input, w, focus);
  if ("empty" in r) throw new Error(r.empty);
  return r.layout;
};

const two: EmbeddingInput = {
  method: "pca",
  axes: [{ label: "Component 1", share: 0.4 }, { label: "Component 2", share: 0.2 }],
  xs: [0, 10],
  ys: [0, 5],
  groups: [0, 1],
  grouping: { name: "batch", levels: ["B1", "B2"] },
};

describe("embedding scale", () => {
  it("draws both axes to one scale, centered", () => {
    const l = lay(two);
    // 400 wide → 248 tall; the plot is 10..390 by 22..224, less a marker's radius each side
    expect(l.height).toBe(248);
    expect(l.k).toBeCloseTo(37.2); // min(372 / 10, 194 / 5)
    expect(l.points[0]).toMatchObject({ px: 14, py: 216, key: "g0" });
    expect(l.points[1]).toMatchObject({ px: 386, py: 30, key: "g1" });
    // the same pixels per unit both ways: 10 across is twice 5 up
    expect(l.points[1]!.px - l.points[0]!.px).toBeCloseTo(2 * (l.points[0]!.py - l.points[1]!.py));
  });

  it("names its axes without implying a unit", () => {
    const l = lay(two);
    expect(l.axisTitles).toEqual(["Component 1 · 40% of the spread", "Component 2 · 20% of the spread"]);
    const u = lay(syntheticCloud(50));
    expect(u.axisTitles).toEqual(["UMAP 1", "UMAP 2"]);
    render(<EmbeddingView input={EMBEDDING_PCA} />);
    const texts = [...document.querySelectorAll("svg text")].map((t) => t.textContent ?? "");
    expect(texts.every((t) => !/^[−-]?\d/.test(t))).toBe(true);
  });

  it("places a single row at the center", () => {
    const l = lay(EMBEDDING_ONE);
    expect(l.points).toHaveLength(1);
    expect(l.points[0]!.px).toBe(200);
    expect(l.points[0]!.py).toBe(123);
  });

  it("draws density above its line, every row in a cell, and isolates a group", () => {
    const cloud = syntheticCloud(6000);
    const l = lay(cloud, 600);
    expect(l.mode).toBe("density");
    expect(l.cells.reduce((a, c) => a + c.count, 0)).toBe(6000);
    expect(l.cells.every((c) => c.step >= 1 && c.step <= 4)).toBe(true);
    const south = cloud.groups!.filter((g) => g === 1).length;
    const only = lay(cloud, 600, "g1");
    expect(only.cells.reduce((a, c) => a + c.count, 0)).toBe(south);
    expect(only.cells.every((c) => c.key === "g1")).toBe(true);
    expect(lay(syntheticCloud(EMBED.densityAbove)).mode).toBe("points");
  });

  it("finds the nearest mark within a hit radius larger than the mark", () => {
    const l = lay(two);
    expect(nearest(l, 14 + 9, 216)).toMatchObject({ i: 0 });
    expect(nearest(l, 200, 120)).toBeNull();
  });

  it("folds levels past the fifth into one gray entry", () => {
    const many: EmbeddingInput = { ...two, xs: [0, 1, 2, 3, 4, 5, 6], ys: [0, 1, 2, 3, 4, 5, 6], groups: [0, 1, 2, 3, 4, 5, null], grouping: { name: "g", levels: ["a", "b", "c", "d", "e", "f"] } };
    const l = lay(many);
    expect(l.legend.map((x) => [x.key, x.slot])).toEqual([["g0", 1], ["g1", 2], ["g2", 3], ["g3", 4], ["g4", 5], ["other", null], ["none", null]]);
  });
});

describe("embedding states", () => {
  it("says why in one line when there are no rows", () => {
    render(<EmbeddingView input={{ ...two, xs: [], ys: [], groups: [] }} />);
    expect(screen.getByRole("note")).toHaveTextContent("No rows to place");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("offers its table alternative", () => {
    render(<EmbeddingView input={EMBEDDING_PCA} />);
    fireEvent.click(screen.getByRole("button", { name: "Show as a table" }));
    expect(screen.getAllByRole("table")).toHaveLength(2);
    expect(screen.getByText("QC01")).toBeInTheDocument();
  });
});
