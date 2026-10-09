import { MATRIX_CORRELATION, MATRIX_MISSINGNESS, MATRIX_ONE_PAIR } from "../fixtures";
import type { LabEntry } from "../lab/entry";
import { MatrixView } from "./MatrixView";
import { MATRIX_PURPOSE } from "./types";

const base = { kind: "matrix", purpose: MATRIX_PURPOSE };

export const entries: LabEntry[] = [
  {
    ...base,
    id: "matrix-correlation",
    title: "Matrix · correlations",
    source: "Computed from sample_data/dietary_recalls.csv (make_fixtures.py), the outcome left out; no engine artifact yet",
    render: () => <MatrixView input={MATRIX_CORRELATION} title="Which columns move together" />,
  },
  {
    ...base,
    id: "matrix-missingness",
    title: "Matrix · blanks by batch and sample type",
    source: "Computed from sample_data/metabolomics_untargeted.csv (make_fixtures.py); no engine artifact yet",
    render: () => <MatrixView input={MATRIX_MISSINGNESS} title="Where the blanks fall" />,
  },
  { ...base, id: "matrix-one", title: "Matrix · one pair", source: "Hand-made: the degenerate case", render: () => <MatrixView input={MATRIX_ONE_PAIR} /> },
  {
    ...base,
    id: "matrix-empty",
    title: "Matrix · one column",
    source: "Hand-made: the empty case",
    render: () => <MatrixView input={{ ...MATRIX_ONE_PAIR, rows: ["fat_g"], cols: ["fat_g"], values: [[1]], n: undefined }} />,
  },
];
