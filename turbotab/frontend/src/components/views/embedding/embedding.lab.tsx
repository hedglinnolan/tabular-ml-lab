import { EMBEDDING_ONE, EMBEDDING_OUTCOME_GATED, EMBEDDING_PCA, syntheticCloud } from "../fixtures";
import type { LabEntry, LabSample } from "../lab/entry";
import { EmbeddingView } from "./EmbeddingView";
import { EMBEDDING_PURPOSE } from "./types";

const CLOUD = syntheticCloud();

const samples: LabSample[] = [
  { label: "PCA by batch",
    source: "Computed from sample_data/metabolomics_untargeted.csv (make_fixtures.py); no engine artifact yet",
    render: () => <EmbeddingView input={EMBEDDING_PCA} title="How the samples group across every feature" />,
  },
  { label: "Density, 6,000 rows",
    source: "A seeded synthetic cloud, for the density drawing only",
    render: () => <EmbeddingView input={CLOUD} />,
  },
  { label: "One row", source: "Hand-made: the degenerate case", render: () => <EmbeddingView input={EMBEDDING_ONE} /> },
  { label: "No rows",
    source: "Hand-made: the empty case",
    render: () => <EmbeddingView input={{ ...EMBEDDING_ONE, xs: [], ys: [], groups: [] }} />,
  },
  { label: "Colored by the outcome before its gate",
    source: "The PCA fixture with responder, the outcome, as its grouping: refused",
    render: () => <EmbeddingView input={EMBEDDING_OUTCOME_GATED} />,
  },
];

export const entries: LabEntry[] = [{ kind: "embedding", order: 8, purpose: EMBEDDING_PURPOSE, samples }];
