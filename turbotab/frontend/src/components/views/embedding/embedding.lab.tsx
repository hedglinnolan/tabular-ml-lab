import { EMBEDDING_ONE, EMBEDDING_OUTCOME_GATED, EMBEDDING_PCA, syntheticCloud } from "../fixtures";
import type { LabEntry } from "../lab/entry";
import { EmbeddingView } from "./EmbeddingView";
import { EMBEDDING_PURPOSE } from "./types";

const CLOUD = syntheticCloud();
const base = { kind: "embedding", purpose: EMBEDDING_PURPOSE };

export const entries: LabEntry[] = [
  {
    ...base,
    id: "embedding-pca",
    title: "Embedding · PCA by batch",
    source: "Computed from sample_data/metabolomics_untargeted.csv (make_fixtures.py); no engine artifact yet",
    render: () => <EmbeddingView input={EMBEDDING_PCA} title="How the samples group across every feature" />,
  },
  {
    ...base,
    id: "embedding-density",
    title: "Embedding · density, 6,000 rows",
    source: "A seeded synthetic cloud, for the density drawing only",
    render: () => <EmbeddingView input={CLOUD} />,
  },
  { ...base, id: "embedding-one", title: "Embedding · one row", source: "Hand-made: the degenerate case", render: () => <EmbeddingView input={EMBEDDING_ONE} /> },
  {
    ...base,
    id: "embedding-empty",
    title: "Embedding · no rows",
    source: "Hand-made: the empty case",
    render: () => <EmbeddingView input={{ ...EMBEDDING_ONE, xs: [], ys: [], groups: [] }} />,
  },
  {
    ...base,
    id: "embedding-gated",
    title: "Embedding · colored by the outcome before its gate",
    source: "The PCA fixture with responder, the outcome, as its grouping: refused",
    render: () => <EmbeddingView input={EMBEDDING_OUTCOME_GATED} />,
  },
];
