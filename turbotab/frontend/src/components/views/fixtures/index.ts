/**
 * The views' fixtures: the engine's own artifact where one exists (overlap, from the captured
 * causal journey), else values computed from the repository's sample data (make_fixtures.py), and
 * one seeded synthetic cloud for the density drawing, said as such.
 */
import causal from "../../../mocks/fixtures/m3-causal.json";
import type { EmbeddingInput } from "../embedding/types";
import { fromEngine, type EngineOverlap } from "../overlap/fromEngine";
import type { OverlapInput } from "../overlap/types";
import type { MatrixInput } from "../matrix/types";
import pca from "./embedding-pca.json";
import correlation from "./matrix-correlation.json";
import missingness from "./matrix-missingness.json";

const causalBase = (causal as unknown as { artifacts: { causal: { base: { overlap: EngineOverlap; n_trimmed: number; exposure: string } } } }).artifacts.causal.base;
const plannedTrim = (causal as unknown as { endpoints: { plan: { body: { plan: { causal: { trim: number | null } } } } } }).endpoints.plan.body.plan.causal.trim;

/** The engine's propensity overlap (heavy_user, the causal journey), with the plan's trim. */
export const OVERLAP_ENGINE: OverlapInput = fromEngine(causalBase.overlap, {
  labels: [`${causalBase.exposure} = 1`, `${causalBase.exposure} = 0`],
  trim: plannedTrim,
  n_trimmed: causalBase.n_trimmed,
  keep_state: "choice",
});

export const OVERLAP_RECORDED: OverlapInput = { ...OVERLAP_ENGINE, keep_state: "recorded" };
export const OVERLAP_UNTRIMMED: OverlapInput = { ...OVERLAP_ENGINE, keep: null };
export const ENGINE_N_TRIMMED = causalBase.n_trimmed;

/** One row in each group: the degenerate case. */
export const OVERLAP_ONE_EACH: OverlapInput = {
  x_label: "Each row's chance of being exposed",
  scale: "propensity",
  edges: [0, 0.25, 0.5, 0.75, 1],
  groups: [
    { label: "exposed", counts: [0, 0, 1, 0] },
    { label: "not exposed", counts: [0, 1, 0, 0] },
  ],
  keep: null,
};

export const OVERLAP_ONE_GROUP: OverlapInput = {
  ...OVERLAP_ONE_EACH,
  groups: [
    { label: "exposed", counts: [2, 3, 1, 0] },
    { label: "not exposed", counts: [0, 0, 0, 0] },
  ],
};

export const EMBEDDING_PCA = pca as unknown as EmbeddingInput;

/**
 * The same embedding colored by the outcome before its gate: a First look caller's mistake the
 * view refuses in one line (FOUNDATION §5 rule 6).
 */
export const EMBEDDING_OUTCOME_GATED: EmbeddingInput = {
  ...EMBEDDING_PCA,
  grouping: { name: "responder", levels: ["no", "yes"] },
};

/** A seeded synthetic cloud of 6,000 rows in three overlapping groups: density, not points. */
export function syntheticCloud(n = 6000, seed = 7): EmbeddingInput {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const centers = [
    [-2.2, 0.4],
    [1.6, 1.4],
    [0.8, -1.6],
  ];
  const xs: number[] = [];
  const ys: number[] = [];
  const groups: number[] = [];
  for (let i = 0; i < n; i++) {
    const g = rand() < 0.45 ? 0 : rand() < 0.6 ? 1 : 2;
    xs.push(+(centers[g]![0]! + normal() * 1.1).toFixed(4));
    ys.push(+(centers[g]![1]! + normal() * 0.8).toFixed(4));
    groups.push(g);
  }
  return {
    method: "umap",
    axes: [{ label: "UMAP 1" }, { label: "UMAP 2" }],
    xs,
    ys,
    groups,
    grouping: { name: "site", levels: ["North", "South", "East"] },
    columns: ["synthetic"],
    outcome: null,
    basis: `${n.toLocaleString("en-US")} synthetic rows`,
  };
}

export const EMBEDDING_ONE: EmbeddingInput = {
  method: "pca",
  axes: [
    { label: "Component 1", share: 1 },
    { label: "Component 2", share: 0 },
  ],
  xs: [0.4],
  ys: [-1.2],
  ids: ["S01"],
  groups: [0],
  grouping: { name: "batch", levels: ["B1"] },
  columns: ["mz_0001", "mz_0002"],
  outcome: { name: "responder", gate_open: false },
};

export const MATRIX_CORRELATION = correlation as unknown as MatrixInput;
export const MATRIX_MISSINGNESS = missingness as unknown as MatrixInput;

export const MATRIX_ONE_PAIR: MatrixInput = {
  kind: "correlation",
  method: "Pearson",
  rows: ["energy_kcal", "fat_g"],
  cols: ["energy_kcal", "fat_g"],
  order: "as in your file",
  outcome: { name: "hba1c", gate_open: false },
  symmetric: true,
  values: [
    [1, 0.9],
    [0.9, 1],
  ],
  n: [
    [600, 600],
    [600, 600],
  ],
};
