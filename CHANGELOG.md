# Changelog

All notable changes to Tabular ML Lab are recorded here, in user terms. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet. `main` is at the 1.0.0 tag.

## [1.0.0] - 2026-09-19

The first tagged release, and the version the journal manuscript describes. It is
the Classic Streamlit workbench: upload to LaTeX manuscript in ten steps, with
the test set sealed at upload, bootstrap confidence intervals, and every cap or
limit disclosed on the page and in the manuscript.

### Added

- Double-click starters for Windows and Mac that install a private Python and
  the libraries once, then run offline; a release zip with a SHA-256 checksum.
- Cohort runs as persistent branches: analyze one group, switch to another
  without losing a fit, and export every group side by side, with the sealed
  test set counted per group.
- Upload admission that measures the parsed table against the memory the
  machine can actually give, instead of a fixed megabyte cap; the same check
  when files are joined or stacked; Excel priced before it is parsed; a
  transpose control for wide assay exports.
- A time and memory estimate, measured on your own machine, above the Train
  button before you click it.
- Compute budgets for very wide tables (correlation screen, VIF, model-agnostic
  SHAP, permutation importance), each named on the page and carried into the
  manuscript as a limitation when it fires.
- AI interpretation configurable from the environment on a shared server
  (backend, Ollama address, server-side API keys, or disabled), with the sidebar
  overriding per session; the OpenAI and Anthropic backends are installed
  rather than optional.
- The app's fonts ship with it, so the first launch is the last time it needs
  the network.
- `CITATION.cff`, so GitHub's "Cite this repository" button works.

### Changed

- The README leads with what the app does, states what it is not, has its own
  privacy section, and was checked claim by claim against the code.
- The optional TDA and UMAP extras are installed by every path with dependency
  overrides, because a plain install of giotto-tda downgrades scikit-learn,
  numpy and scipy for the whole app.
- Streamlit is pinned to a version window (1.60 to 1.63) after two upstream
  releases broke unchanged pages.
- The stale `DEPLOYMENT.md` was removed; `UNIVERSITY_DEPLOYMENT.md` is the
  server guide.

### Fixed

- Classification manuscripts now report the AUC in the performance table, the
  abstract and the results prose; the key eval writes differed from the key
  the report looked up.
- Generated prose uses "analyses" as the noun; an earlier American-English pass
  had produced "analyzes".
- Switching cohorts keeps the preprocessing and selection decisions the pages
  own; the constant column is actually dropped; external validation filters
  by the active group.
- The wide-table warning describes the app that ships rather than an older,
  uncapped one.

[Unreleased]: https://github.com/hedglinnolan/tabular-ml-lab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hedglinnolan/tabular-ml-lab/releases/tag/v1.0.0
