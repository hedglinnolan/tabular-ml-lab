import { MATRIX_CORRELATION, MATRIX_MISSINGNESS, MATRIX_ONE_PAIR } from "../fixtures";
import type { LabEntry, LabSample } from "../lab/entry";
import { MatrixView } from "./MatrixView";
import { MATRIX_PURPOSE } from "./types";


const samples: LabSample[] = [
  { label: "Correlations",
    source: "Computed from sample_data/dietary_recalls.csv (make_fixtures.py), the outcome left out; no engine artifact yet",
    render: () => <MatrixView input={MATRIX_CORRELATION} title="Which columns move together" />,
  },
  { label: "Blanks by batch and sample type",
    source: "Computed from sample_data/metabolomics_untargeted.csv (make_fixtures.py); no engine artifact yet",
    render: () => <MatrixView input={MATRIX_MISSINGNESS} title="Where the blanks fall" />,
  },
  { label: "One pair", source: "Hand-made: the degenerate case", render: () => <MatrixView input={MATRIX_ONE_PAIR} /> },
  { label: "One column",
    source: "Hand-made: the empty case",
    render: () => <MatrixView input={{ ...MATRIX_ONE_PAIR, rows: ["fat_g"], cols: ["fat_g"], values: [[1]], n: undefined }} />,
  },
  { label: "No pair recorded together",
    source: "Hand-made: only the diagonal computed",
    render: () => <MatrixView input={{ ...MATRIX_ONE_PAIR, values: [[1, null], [null, 1]] }} />,
  },
];

export const entries: LabEntry[] = [{ kind: "matrix", order: 9, purpose: MATRIX_PURPOSE, samples }];
