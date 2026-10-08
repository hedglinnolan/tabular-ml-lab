# The model-family contract

**Status: DRAFT 1, for Nolan's review.** The orchestrator wrote it on 2026-10-08, acting as methods expert. It carries Nolan's rulings and ideas of 2026-10-08, listed in §0. Nothing was run. The engine claims come from reading branch `turbotab-next` at b1830787. The theory claims come from the verified bibliography in §7 and from the theory source, Simon et al. (2026), read in full.

**What it governs.**
- **`M1_CONTRACT.md` §7,** the `ModelFamily` protocol. The clauses of §1 replace its list of members.
- **`RECIPES_AND_TUNING.md`** §2.4 (recipes) and §4.1 (`TuningDecl`). Both are kept as written and referenced. Additions are marked "new here".
- **`BLUEPRINT.md` §13,** the method contract. A family becomes a contract-shaped plug-in, with sources, two registers and reference tests. Today the methods reference says a family "is not a method contract" (`reference/methods.py:family_section`).
- **`understanding/UNDERSTANDING_LAYER.md` §1.3,** the legality matrix. Every post-fit item in §2 names its view class (O0 to O4).
- **`crosswalk/CROSSWALK.md`,** "Noticings born after the fit", and **`crosswalk/SIZING.md`.** §5 adds work packages in SIZING's units.

**How to read the references.**
- Code and test paths are relative to `turbotab/core/`. Document paths are relative to `docs/turbotab-next/`.
- An engine claim is cited as `file:symbol`.
- **"Convention"** marks a choice with no verified source. It is a number or rule to be checked on the reference journeys, not a finding.
- **"Preprint"** marks a source with no peer-reviewed venue. It is a lead to watch, never a foundation.

---

## 0 · Purpose, in plain words

### What this document is for

A model family is one kind of model the app can fit: a linear model, a random forest, and later a neural network. TurboTab lets a researcher fit several families to the same table and see why they differ. That comparison is only trustworthy if every family tells the app the same things about itself, in the same form, and passes the same checks.

This document is that list: everything a family must declare, satisfy and pass before it joins TurboTab. It is **one door**. These all come through it:
- today's nine families;
- the four v2 additions: ridge, robust linear regression (Huber), random forest and XGBoost;
- later, neural networks (an MLP, a tabular transformer in the style of FT-Transformer);
- pretrained in-context models in the style of TabPFN, whose prior *is* the model and which have no training loop.

Once a family is through the door, nothing else in the app may treat it specially by its name. Every screen, sentence and check reads what the family declared.

### Nolan's north star

> "The magic of this app is that it presumes some of the 'art' we think of as modeling decisions can actually be decomposed into a science we just haven't bothered to check before."

What that means for model families:
- **Choosing a family is not a matter of taste.** Each family assumes something about how the outcome depends on the inputs. Those assumptions can be stated, drawn on the user's own data, and tested.
- **The app checks the science on the user's own table.** Almost all of the theory below was worked out on idealized inputs: Gaussian, continuous and high-dimensional. Tabular data are skewed, small, correlated and partly categorical. The literature search behind §7 found no study that applies this learning theory to tabular data on purpose. So TurboTab cannot inherit the theory's validity; it checks the theory on its own encoded matrices. That is the opening the north star names.
- **Every phenomenon gets its name.** A card says what happened in plain words, and the technical name rides along quietly with its source. In Nolan's words, the researcher thinks "huh, so that's what I call that problem when I'm discussing it with colleagues."

### The rulings this contract carries (2026-10-08)

| # | Ruling or idea | Where it lands |
|---|---|---|
| 1 | The pre-fit ranking happens live at model selection, after the shared steps are settled. Each family is assessed on the input it would actually receive. Everything is outcome-blind. Nothing is ranked by a score before Fit; the corrected comparison (BBC-CV) decides after Fit. | C4; question 1; MC-3 to MC-6 |
| 2 | Inductive-bias curves (each top predictor's effect per family, on shared axes) are central to explainability. | C10; the registry in §2.4 |
| 3 | Target alignment is never shown before the fit. It becomes post-fit explainability: why the penalized or low-rank families did well or badly. | §2.1 |
| 4 | Phenomena are named in two registers: a plain sentence, and the quiet technical name with its source. | C5; §2.4 |
| 5 | The tapestry shows which directions a penalty shrinks. | §2.3 |
| 6 | A flexible family that clearly beats the linear ones, with explanations that point at a few hidden combinations, yields a labeled exploratory hypothesis under the leash: "prediction plus explainability begets further inference". | §2.2; question 2 |
| 7 | Solvable settings become independent reference tests for neural families. | C13; §4 |
| 8 | The north star above. | throughout |

### The rules that hold everywhere

- **The leash.** Every element states its rung for each purpose (`contracts.py:Rung`; BLUEPRINT §11.3). A pattern found on these rows is a hypothesis, never a finding.
- **No estimate before the lock.** Under Estimate and Describe, no estimate is served until pressing Fit locks the track's plan (CROSSWALK, "After training: every result is an exhibit"). Nothing on the pre-fit shelf is an estimate.
- **Outcome-blind before Fit.** The shelf reads no predictor-by-outcome quantity and no score (C4). Question 1 asks how the outcome's own counts are treated.
- **The corrected comparison decides.** The shelf orders the families; it never picks one. After Fit, BBC-CV decides (`models/selection.py:selection_optimism`, `models/selection.py:declared_result`).
- **Calm.** Each new element is one quiet line with a plain sentence. Numbers sit under "More angles". Every new view gets a purpose-registry entry (BLUEPRINT §11.2).

### How the theory maps onto the app

Simon et al. (2026) argue that a scientific theory of deep learning is forming along five strands. Each has a home here.

| Strand (Simon et al. 2026) | In TurboTab |
|---|---|
| §2.1 Solvable settings: deep linear networks, kernel regression, multi-index models | Reference tests with a known answer (C13); the hidden-combinations readout (§2.2) |
| §2.2 Limits: lazy versus rich learning | A neural family declares its regime, and its fit reports which regime it was in (C4, C8) |
| §2.3 Simple laws: edge of stability, the neural feature ansatz | Training diagnostics (C8); the average gradient outer product (§2.2) |
| §2.4 Hyperparameters can be disentangled | Structural settings are identity, searched ones are tuning, and which knobs matter is measured (C1, C6) |
| §2.5 Universal behavior across architectures | Curves on shared axes show where families agree on a shape, and where equally good families disagree (C10; the registry in §2.4) |

---

## 1 · The contract clauses

### The contract on one page

| Clause | What a family declares or satisfies | Enforced by |
|---|---|---|
| **C1** Identity | Key, label, library, defaults version. Neural: learning rule, initialization, parameterization and seed policy. In-context: the prior's checkpoint. | `register_family`; provenance; replay |
| **C2** Tasks, purposes, inference eligibility | Tasks; purposes; whether it predicts; what kind of table it can give under inference | `register_family`; refusals; reference tests |
| **C3** Inputs and recipe | RECIPES §2.4 recipe slots; blank routing; the transformations it is invariant to | `register_family`; invariance probes (C13) |
| **C4** Assess on the actual input | The input-profile fields its `assess` reads; a sourced sample-efficiency prior; its regime | The shelf stage; an outcome-permutation test |
| **C5** Inductive bias in two registers | The plain statement; its quiet names with sources; the shape its curves should take | Word budgets; the citation registry; a curve-shape test |
| **C6** Tuning | `TuningDecl`; structural versus searched settings; transfer; cost | RECIPES T1–T17; `register_family` |
| **C7** Complexity controls | Each knob that sets effective complexity, its direction, and a formula where one exists | A formula test per knob |
| **C8** Training diagnostics | The checks it reports about its own fit | A fixture where each fires and one where it stays silent |
| **C9** Calibration | Its output scale and the recalibration it supports | The generic calibration path; a fixture |
| **C10** Explanation paths | A raw score on a declared scale; its attribution and architecture views | Curves for every predicting family; reference tests |
| **C11** Bootstrap and validation soundness | Whether Harrell's bootstrap is sound; whether it is flexible; its internal splits | A soundness test against fresh data |
| **C12** Replay | Seeds, threads, library versions, tolerance | The export replay test, for every family |
| **C13** Reference tests | An independent implementation; invariance probes; solvable settings | The fold-in gate |
| **C14** The fold-in gate | Passes every item of the checklist | One parametrized acceptance test over the registry, plus expert review |

### How it is declared (`models/base.py`, new here)

These members extend `models/base.py:FamilyBase` beside RECIPES §2.4's `recipe`, `tuning` and `defaults_version`. Existing duck-typed members (`purposes`, `predicts`, `ordered_levels`, `bootstrap_optimism`, `linear_in_values`, `pools_imputations`, `preprocess`, `build_for`, `describe_step`, `inference`, `inference_matrix`) become declared members of the protocol, so `register_family` can check them.

```python
@dataclass(frozen=True)
class Source:
    key: str                 # a key into the verified citation registry (SIZING X4)
    where: str = ""          # "§3.4.1, eq. 3.47"

@dataclass(frozen=True)
class Named:                 # a term or phenomenon in two registers (C5, §2.4)
    plain: str               # the card's sentence, ≤ 22 words
    known_as: str            # "double descent"
    source: Source

@dataclass(frozen=True)
class Identity:              # C1
    kind: Literal["estimator", "trained_network", "pretrained_prior"]
    library: str             # its version is recorded at every fit
    estimator: str           # a class name, for provenance only: nothing switches on it
    learning_rule: str = ""  # trained_network: optimizer, schedule, batch size, epochs or stopping
    initialization: str = "" # trained_network: scheme and scale
    parameterization: str = ""   # trained_network: "standard" · "NTK" · "muP", and the output multiplier
    seed_policy: str = ""    # how its seeds derive from the split's seed (RECIPES §4.7)
    prior: str = ""          # pretrained_prior: checkpoint name and SHA-256

@dataclass(frozen=True)
class InferenceDecl:         # C2
    table: Literal["intervals", "shrunk_no_intervals", "description_only"]
    intervals: tuple[str, ...] = ()   # "HC3", "CR2", "sandwich", "Satterthwaite", "Taylor"
    design_based: bool = False        # replaces the signature check in survey.has_design_estimator
    product_terms: bool = False       # replaces methods/interaction.py:SUPPORTED
    matrix_table: bool = False        # replaces stages/effects.py:SEQUENCE_FAMILIES

@dataclass(frozen=True)
class Prior:                 # C4: a sample-efficiency statement, never a forecast
    says: str                # ≤ 22 words
    kind: Literal["bound", "empirical", "convention"]
    source: Source | None

@dataclass(frozen=True)
class Knob:                  # C7
    setting: str             # the estimator's parameter, or "time" for early stopping
    more_means: Literal["simpler", "more_flexible"]
    formula: str = ""        # "df(λ) = Σ d²/(d² + λ)", when one exists
    source: Source | None = None

class FamilyBase:
    identity: Identity
    purposes: tuple[Purpose, ...]
    predicts: bool
    flexible: bool                          # declared, no longer derived from bootstrap_optimism
    bootstrap_optimism: bool
    inference: InferenceDecl | None = None  # None: not offered as an inference table
    reads: tuple[str, ...] = ()             # InputProfile fields its assess reads (C4)
    sample_efficiency: tuple[Prior, ...] = ()
    regime: Literal["n/a", "lazy", "rich"] = "n/a"
    bias_terms: tuple[Named, ...] = ()      # C5
    invariances: tuple[str, ...] = ()       # "monotone_per_column", "rotation", "column_scale"
    curve_shape: Literal["straight", "steps", "smooth", "any"] = "any"
    complexity: tuple[Knob, ...] = ()       # C7
    diagnostics: tuple[str, ...] = ()       # keys into the diagnostics registry (C8)
    output: Literal["value", "margin", "probability"] = "value"   # C9
    updating: tuple[str, ...] = ()          # recalibration paths it supports (C9)
    attribution: Literal["linear", "trees", "none"] = "none"      # C10
    architecture: tuple[str, ...] = ()      # "equation", "trees", "shrinkage", "spectrum"
    solvable: tuple[str, ...] = ()          # keys into the solvable-settings harness (C13)
    replay_tolerance: float = 1e-12         # C12
    sources: tuple[Source, ...] = ()        # its primary sources, as a method contract has
```

`models/base.py:FamilyInfo` and `models/artifacts.py:ShelfFamily` gain the user-facing parts of these: the bias terms, invariances, inference table kind, `flexible`, `bootstrap_optimism`, the profile fields read, and the structured concerns of C4.

---

### C1 · Identity

**What.**
- Every family declares its key, label, library and estimator class, and RECIPES §2.4's `defaults_version`. The library's version is recorded at every fit (RECIPES §4.7).
- **A trained network's identity includes how it learns.** It declares:
  - its learning rule: optimizer, schedule, batch size, and the number of epochs or the stopping rule;
  - its initialization, by scheme and scale;
  - its parameterization (standard, NTK or muP) and its output multiplier;
  - its seed policy.

  Two MLPs that differ in any of these are two versions of the family, with different version keys (RECIPES §3.2).
- **A pretrained in-context model's identity is its prior.** It declares the checkpoint's name and SHA-256, and any input handling the checkpoint applies internally.
- **Nothing switches on a key or a class name.** Code reads declarations. Today twelve places switch on family keys or estimator class names; §3.3 lists them and the declarations that replace them.

**Why.**
- Simon et al. (2026, §2) list a deep learning system's components as architecture, data, task and learning rule, where the learning rule includes the initialization and the optimization settings. The learning rule is part of what the model *is*.
- The output multiplier alone moves a network between lazy (kernel-like) and rich (feature-learning) training (Chizat et al. 2019; Simon et al. 2026, §2.2). A deep linear network trained by gradient descent converges to the minimum-norm solution only as the initialization scale goes to zero, and drifts away from it at larger scales (Yun et al. 2021). So initialization and parameterization change which function is learned. They belong to identity, not to tuning.
- An in-context model has no fitted penalty; its inductive bias is the prior it was trained on (Hollmann et al. 2023; Müller et al. 2022).

**How the engine enforces it.**
- `register_family` refuses a `trained_network` whose learning rule, initialization, parameterization or seed policy is empty, and a `pretrained_prior` without a checkpoint hash.
- The identity enters provenance and the version key. Replay (C12) refits from it.
- A test fails when a family key or estimator class name appears as a literal outside its own module, the registry and the tests. Its allowlist starts with the twelve places of §3.3 and must shrink to zero (MC-2).
- **Today:** key and label only (`models/base.py:ModelFamily`). The registry's own docstring says "nothing else in the app switches on family keys", which §3.3 shows is not yet true.

### C2 · Tasks and purposes, and inference eligibility

**What.**
- **Tasks** are a subset of `models/base.py:TASKS`; `ordered_levels` marks a family that respects an ordered outcome.
- **Purposes** come from one checked vocabulary: prediction and inference. Describe uses no model family in v2.0.0; its estimator is a descriptive one (SIZING D1). A family that later serves Describe declares it.
- **`predicts`** says whether it makes predictions; a family that only tests has no cross-validated score.
- **Inference eligibility** is an `InferenceDecl`:
  - `intervals`: a table with intervals, naming which kinds (HC3, CR2, sandwich, Satterthwaite, Taylor linearization);
  - `shrunk_no_intervals`: a labeled shrunk table with no intervals, as for the elastic net and ridge (RECIPES §5);
  - `description_only`: curves only, labeled as description.

  A family with no declaration is not offered as an inference table. Its flags for design-based estimation, product terms and the matrix table replace the three key lists that decide these today (§3.3).

**Why.**
- Under inference the coefficient table is the locked primary's estimate (BLUEPRINT §12, ruling 3). Only a family whose intervals were checked against an independent reference may produce one (V2_DEFINITION_OF_DONE §2).
- Conventional intervals on coefficients penalized by cross-validation are "impossible by construction" (RECIPES §5). The declaration makes that a property of the family, not a rule each stage must remember.

**How the engine enforces it.**
- `register_family` checks the purposes vocabulary and the interval kinds against a known set.
- Every interval kind a family declares needs a reference test (C13).
- `models/survey.py:has_design_estimator` reads `design_based` instead of inspecting the `inference` signature for a `survey` parameter.
- **Today:** `purposes` and `predicts` are read with `getattr` (`models/base.py:info`), and `register_family` checks neither.

### C3 · Inputs and recipe slots

**What.**
- **The recipe** is RECIPES §2.4 unchanged: five slots (missing, scale, encoding, transform, outliers), each option with its plain label, quiet term, consequence, customary and sound labels and leash rung. Steps declare `reads(spec)` and `passes_blanks`, so blank routing is computed (RECIPES §2.3). `needs_scaling` and `handles_missing` become derived.
- **Invariances (new here).** A family declares the input transformations its predictions do not change under:
  - `monotone_per_column`: a monotone change of one column (trees);
  - `rotation`: an orthogonal rotation of the input matrix as the family scales it (ridge, an MLP without per-column embeddings);
  - `column_scale`: rescaling one column (unpenalized least squares).
- **For a neural family, the recipe adds a numeric-embedding slot** (§4): none, piecewise-linear or periodic (Gorishniy et al. 2022), with rank or quantile scaling as an option (Beyazit et al. 2023).

**Why.**
- **Rotation is a real dividing line.** Random rotations of the inputs hurt trees and FT-Transformer but not ResNet-style networks, and reverse their ranking (Grinsztajn et al. 2022). Any rotation-invariant algorithm has a worst-case sample complexity that grows at least linearly in the number of irrelevant features, while L1-regularized logistic regression's grows only logarithmically (Ng 2004). Declaring the invariance tells the shelf which input measures matter for a family (C4), and gives the reference test something to check (C13).
- **Blanks.** For prediction, filling blanks before learning is consistent when missingness is not informative. Trees that route blanks themselves handle informative missingness too (Josse et al. 2024). After imputation, the best regression function is generally discontinuous and hard for smooth models to learn (Le Morvan et al. 2021). Missing indicators help when missingness is informative, and overfit when many uninformative ones meet few rows (Van Ness et al. 2023). Better imputation buys little on real outcomes once indicators are present (Le Morvan & Varoquaux 2024).
- **Embeddings for neural families.** Piecewise-linear and periodic embeddings of numeric columns gave large gains in tabular MLPs (Gorishniy et al. 2022). The explicit spectral argument for why they help is Beyazit et al. (2023): tabular target functions are irregular, and transformations such as ranking reduce that irregularity.

**How the engine enforces it.**
- `register_family` runs RECIPES §2.4's checks, including that a family may declare native blanks only if its estimator fits a three-row frame with one blank.
- Each declared invariance gets a probe in C13. A family that does not declare `rotation` must show a change under rotation on the probe's fixture, so the declaration cannot be wrong in either direction.
- **Today:** only `needs_scaling` and `handles_missing` exist, and trees' `handles_missing` is unreachable (RECIPES F1).

### C4 · Assess on the actual input

**What.**

*When the shelf ranks (ruling 1).*
- The shelf ranks once the shared steps that change a family's input are settled: the missing-values answer, categories, energy adjustment, batch, scales, column units, Explore's levers and the selection step. "Settled" means answered, or confirmed in the stage's Confirm sweep (SIZING P0.5).
- Until then the Models stage shows the shelf as "Waiting for" those answers (CROSSWALK disagreement 20). It never shows a provisional ranking.
- When a shared answer changes, the shelf re-ranks, because the stage graph re-runs a stage whose reads change.
- A recipe edit on the card is re-assessed by the consequence preview, not by re-running the stage (MC-6). Stages stay pure functions of the decision log.

*What `assess` reads.* `Situation` gains `inputs: Mapping[family key, InputProfile] | None = None`. The default keeps every test that builds a `Situation` directly working. The profile is computed for every registered family that can model the task, on its **default** recipe, by RECIPES §2.4's `family_spec`. The steps are fitted only up to the last step that does not read the outcome:

| Profile field | In plain words | Why it is there | Read by |
|---|---|---|---|
| rows, units | How many rows and people the plan fits on (RECIPES §4.2's n_plan) | Every criterion | every family |
| columns | Columns after encoding: indicators, the energy step, scale scores, blanks as a level, every level of a category | p and p/n are the outcome-blind core (Dobriban & Wager 2018; McElfresh et al. 2023) | every family |
| parameters | Exact parameter count | Sample-size criteria | regression families |
| blanks routed, blanks filled | Blank cells the family takes as blanks, and cells it is given filled | Native blanks versus fill (Josse et al. 2024) | tree families; every family's caution |
| indicator columns | Missing-indicator columns added | Many indicators with few rows overfit (Van Ness et al. 2023) | every family |
| condition number | Belsley's scaled condition number, reusing `models/linear.py:collinearity_concern` | Collinearity, before the fit instead of only after it | linear families |
| spectrum | The top eigenvalues of the family's own input covariance, scaled as that family scales it | The input spectrum and p/n govern ridge's limiting risk (Dobriban & Wager 2018) | rotation-invariant families |
| effective rank | r₀ = the trace divided by the largest eigenvalue: how many directions the spread fills | The sourced definition (Bartlett et al. 2020) | rotation-invariant families |
| smallest eigenvalue; skewness and kurtosis spread | How irregular the columns are | Part of the irregularity measure on which boosted trees beat neural nets (McElfresh et al. 2023) | trees; neural and kernel families |
| outlying share | Rows far from the centroid in principal-component space | Such rows degrade neural nets faster than boosted trees (Jeffares et al. 2024) | neural and kernel families |
| measured, by rule | Which widths were measured and which come from a step's declared size rule | Some steps read the outcome, so their width can only be a rule | the card |

- **Where a width can only be a rule.** These steps read the outcome: `methods/omics.py:UnivariateScreen`, `models/variable_selection.py:Selector`, `methods/levers.py:InnerCVForms`, `methods/levers.py:RuleSplines` (whose k comes from the effective size) and `methods/levers.py:ImbalanceCorrected`. After the last outcome-free step, a width comes from the step's declared size rule, and the card says "by rule" beside it.
- **Rows.** Under prediction: the training rows, at the plan level. Under inference: every analyzed row, as `stages/modeling.py:design_stage` uses. Column summaries come from these rows, not from the whole table (§3.4).
- **Cost.** The spectrum comes from the smaller of the covariance and the Gram matrix, or from a randomized top-k decomposition when both are large. It uses the sampling caps of `models/cost.py:sample_shape`, and the card says when it did.
- **Trees read width, blanks and the irregularity measures, not the spectrum.** Their splits follow the columns, so the eigen-directions of their input are not what they see (Grinsztajn et al. 2022).

*What `assess` returns.*
- `Assessment` gains `measures: Mapping[str, float]`, and each concern becomes a `Concern(text, known_as, source, field)`. The `field` names the profile field that raised it, and `reads` lists every field the family looked at, so the card can say "Looked at: 412 columns for 2,400 rows; spread along about 31 directions".
- The score stays the family's own judgment of fit to the situation. Every change to it names a profile field and a source, or says "convention". As the engine already says, "no validated rule predicts which family will perform best" (the prediction review, quoted above `models/selection.py:FLEXIBLE_REASON`).

*The sample-efficiency prior.* Each family declares sourced statements of how data-hungry it is, each marked bound, empirical or convention. Examples:
- **Lasso-type families:** with many columns that may not matter, rows needed grow only with the logarithm of their number (a bound, for L1 logistic regression; Ng 2004).
- **Ridge and neural nets:** in the worst case, rows needed grow at least linearly in the number of irrelevant columns (a lower bound; Ng 2004).
- **Boosted trees:** they win on large, irregular tables with many rows per column (empirical; McElfresh et al. 2023).
- **NTK-type kernel machines:** a small edge over random forests on small classification tables (empirical; the margin is small; Arora et al. 2020, a protocol Wainberg et al. 2016 criticized).
- **Pretrained in-context models:** dominant up to 10,000 rows and 500 features (empirical, at the limits of their pretraining; Hollmann et al. 2025).

A prior is shown under "More angles" as a caution. It is never a forecast and never moves a score by itself.

*The regime.* A trained network declares the regime its parameterization implies: lazy under the NTK parameterization, rich under muP or mean-field scaling (Simon et al. 2026, §2.2 and §2.4). The shelf states it. After the fit, the fit reports which regime it was actually in (C8).

*Outcome-blindness.*
- The profile builder takes no outcome argument, so no predictor-by-outcome quantity (view class O3) or score (O4) can enter it.
- The outcome's own counts (O1) still feed Riley's minimum, EPV and Whitehead's effective size. The library-size check (`stages/modeling.py:_assay_concern`) is a design count (O2), shown as a separate shelf line, not part of any `assess`. Question 1 asks Nolan to confirm this reading of "outcome-blind".

**Why.** Ruling 1, with:
- the meta-features that carry signal about which family fits (McElfresh et al. 2023; Ye et al. 2024), restricted to the outcome-blind ones (McElfresh et al. also used some that read the outcome);
- the spectrum and p/n, which govern ridge's limiting risk (Dobriban & Wager 2018);
- effective rank (Bartlett et al. 2020);
- the irregularity and outlyingness cautions (McElfresh et al. 2023; Jeffares et al. 2024).

Whether a column is uninformative depends on the outcome, so it cannot be checked before Fit. Only width, p/n and redundancy can (Grinsztajn et al. 2022; Ng 2004).

**How the engine enforces it.**
- **The shelf stage** (`stages/modeling.py:shelf_stage`) depends on a new `trunk` stage split out of `design` (MC-3). Its reads widen to energy adjustment, batch, scales, column units and selection. It requires the shared answers listed above.
- **An outcome-permutation test.** Permuting the outcome across rows keeps its counts but destroys every predictor-by-outcome relation. The ranking, scores, fits, measures and concerns must stay identical, byte for byte.
- **`register_family` checks** that every field in `reads` exists in `InputProfile`.
- **Today:**
  - `Situation` holds only scalars (`models/base.py:Situation`). No `assess` reads collinearity, the spectrum, blanks or the width after encoding. Belsley's number runs only after the fit (`stages/modeling.py:fit_stage`, for `linear` only).
  - The shelf does not read energy adjustment, batch, scales, column units or selection, so it neither waits for them nor re-ranks when they change (`stages/__init__.py:build_graph`, the `shelf` stage).

### C5 · The inductive-bias statement, in two registers

**What.**
- **The plain statement** stays `inductive_bias`, at most 20 words (`models/base.py:INDUCTIVE_BIAS_WORDS`). Example: "Straight-line effects all shrunk toward zero together; correlated predictors share weight; none is dropped."
- **The quiet names** are `bias_terms`: each a `Named(plain, known_as, source)`. Examples:
  - ridge: "Known as L2 shrinkage" (ESL §3.4.1);
  - random forest: "Known as implicit regularization" (Mentch & Zhou 2020);
  - an MLP: "Known as spectral bias" (Rahaman et al. 2019; Beyazit et al. 2023).
- **The curve shape** is `curve_shape`: what the family's inductive-bias curves should look like on a fixture. Linear families draw straight lines, tree families steps, and smooth families smooth curves.
- **The card shows the plain statement.** The quiet name rides beside it as "Known as …" with its source, in the quiet style of RECIPES §6.5.

**Why.**
- Ruling 4.
- Different interpretable models can learn different, even contradictory, shapes for the same predictor while being equally accurate, because "inductive bias plays a crucial role in what interpretable models learn" (Chang et al. 2021).
- Drawing several families' fits of the same toy functions side by side shows each family's bias at a glance: ridge fits only lines, boosting is piecewise-constant (Hollmann et al. 2025, Fig. 3a).

**How the engine enforces it.**
- Word budgets: plain ≤ 22 words, `known_as` ≤ 6 words.
- Every `known_as` names a phenomenon or term in the registry of §2.4, or a term with its own source. Every source key resolves in the citation registry (SIZING X4), whose entries are verified.
- The curve-shape test (C13) draws the family's curve on a fixture and checks the declared shape.
- **Today:** the plain statement exists. `models/base.py:Assessment.concerns` are plain strings, and the only quiet terms are settings (RECIPES §6.5) and `teaching/__init__.py:TeachingTerm`.

### C6 · Tuning

**What.**
- **RECIPES §4.1's `TuningDecl`, unchanged.** Kind (none, path, search), searched dimensions, settings by hand, fixed settings, standard settings as candidate 0, early stopping, out-of-bag scoring, space version. The plan, nesting and replay follow RECIPES §4.2–4.7.
- **New here: structural versus searched.** A structural setting is part of identity (C1): the architecture, the parameterization and output multiplier, the loss, the booster type. Only searched settings may be dimensions. So the regime is fixed by structural settings, and the fit reports the regime it actually reached (C8).
- **New here: tunability.** Each dimension may cite a measured tunability, meaning how much tuning it gained on benchmark data (Probst et al. 2019). The tuning line shows the one or two knobs that matter, not all of them.
- **New here: transfer and the validated range.** A dimension declares whether its best value is expected to transfer across sizes, with a source. Standard settings declare the row range they were validated on. Defaults meta-tuned on 1,000 to 500,000 rows (Holzmüller et al. 2024) are labeled unvalidated below 1,000.
- **New here: a cost model.** How one fit's seconds grow with rows, columns and the budget (trees, rounds, epochs), so `models/cost.py:estimate_fits` can scale a timing (RECIPES RT-8).

**Why.**
- Simon et al. (2026, §2.4): hyperparameters can be disentangled. Width-dependent factors separate from scale-free coefficients, so some optima stay stable across sizes (Yang et al. 2021). For small tabular networks, which are cheap to tune directly, that transfer gives little leverage, and the search found no tabular muP study.
- Tunability can be measured, so "which knobs are worth turning" becomes a quantity (Probst et al. 2019). That is the north star in one line.
- Searching a cocktail of regularizers let plain MLPs win on 40 datasets (Kadra et al. 2021). That win did not hold in larger later benchmarks (McElfresh et al. 2023; Erickson et al. 2025).
- With the penalty tuned, the risk curve can be monotone, so tuned users rarely see double descent (Nakkiran et al. 2021).

**How the engine enforces it.**
- RECIPES T1–T17.
- `register_family` refuses a structural setting listed as a dimension, and a dimension without a label, quiet term, scale or source.
- **Today:** none of `TuningDecl` is built. The elastic nets tune by scikit-learn's CV estimators (RECIPES F4, F5), and boosted trees are not tuned (RECIPES F2).

### C7 · Complexity controls

**What.** Each family declares the knobs that set its effective complexity (`Knob`): which direction makes it simpler, and a formula where one exists. These place families on one shrinkage axis in the tapestry (§2.3).

| Family | Knob | Formula or equivalence | Source |
|---|---|---|---|
| Ridge | penalty λ | Effective degrees of freedom df(λ) = Σⱼ dⱼ²/(dⱼ² + λ); direction j is shrunk by dⱼ²/(dⱼ² + λ) | ESL §3.4.1, eqs. 3.47 and 3.50 |
| Elastic net | λ and the mix | The ridge part as above; the lasso part drops columns (the path view) | ESL §3.4.1 for the ridge part |
| Linear model or network trained by gradient descent | training time t | With t = 1/λ, gradient flow's risk on least squares is at most 1.69 times ridge's | Ali et al. 2019 |
| Random forest | columns tried per split (mtry) | mtry acts as the penalty knob | Mentch & Zhou 2020 |
| Random forest | leaf size; trees | A forest is smoother than its trees and adapts its smoothing at test time; an "effective smoothing" number | Curth et al. 2024 (preprint) |
| Boosted trees, XGBoost | rounds, learning rate, leaf size, L2 pull | No sourced closed form; double descent may be named only along one named capacity axis | Curth et al. 2023 |
| Every family with filled blanks | the fill itself | Under MCAR in a high-dimensional linear model, zero imputation acts like ridge | Ayme et al. 2023 |
| Every family with noisy inputs | measurement error | Training with input noise equals a Tikhonov penalty | Bishop 1995 |

A single complexity axis shared by all families has been proposed (Allerbo & Schön 2026, preprint). It is a lead to test, not a foundation.

**Why.**
- Ruling 5. The user should see that a penalty, a forest's random column choice, early stopping and even filled blanks all pull the fit in the same way.
- Theory sometimes gives the exchange rate between them (Ali et al. 2019; Mentch & Zhou 2020; Patil & Du 2023 for subsampling and ridge).

**How the engine enforces it.**
- A declared formula has a test. For example, df(λ) must equal the trace of ridge's hat matrix on a fixture to 1e-10.
- A knob with no formula says so on its card line, instead of borrowing one.

### C8 · Training diagnostics

**What.** A family declares the checks it reports about its own fit, as keys into a diagnostics registry. Each check is a label or a disclosure after the fit (O4), never a change to the plan.

| Family kind | Diagnostics |
|---|---|
| Linear | Separation (Firth), collinearity, the HC3 residual-spread check, Cook's influence (today) |
| Mixed, GEE, Cox | Boundary or singular fits, convergence, proportional hazards (today) |
| Penalized | The penalty at a grid edge; the calibration slope per fold (RECIPES §4.6) |
| Tree ensembles | Early-stopping round; out-of-bag error (forest) |
| Trained networks (§4) | **Loss curve and convergence.** **Lazy or rich:** the first layer's relative weight change from its initialization. **Edge of stability:** under full-batch gradient descent only, the sharpness (the largest Hessian eigenvalue) against 2/η. **Neural feature ansatz:** how closely the first layer's Gram matrix matches the average gradient outer product. |

**Why.**
- In the lazy regime the weights and hidden representations change only negligibly while the loss drops; in the rich regime they reorganize (Simon et al. 2026, §2.2; Chizat et al. 2019).
- Under full-batch gradient descent with learning rate η, the sharpness rises and then hovers near 2/η: the edge of stability (Simon et al. 2026, §2.3, summarizing Cohen et al. 2021). The rule is stated for full-batch gradient descent, so the diagnostic never runs for minibatch or adaptive training.
- A trained layer's Gram matrix is roughly proportional to the average gradient outer product (Radhakrishnan et al. 2024). Simon et al. (2026, §2.3) call the rule heuristic and inexact but often strikingly accurate.

**How the engine enforces it.**
- Each diagnostic has a fixture where it fires and one where it stays silent, as a noticing does (SIZING T2).
- Warnings caught during a fit already become concerns (`stages/modeling.py:_concerns`). A declared diagnostic must not depend on a warning's wording.
- The thresholds for "lazy" and "at the edge" are conventions, stated as such.

### C9 · Calibration

**What.**
- A family declares its **output scale**: a value, a margin (log-odds) or a probability. A predicting family for a yes/no outcome must expose a margin or probability through `predict_proba` and a `decision_function`. The random forest has no `decision_function`, so its wrapper supplies one: the log-odds of the clipped probability, labeled (RECIPES RT-5d).
- It declares the **recalibration it supports** (`updating`), such as shrinkage by the calibration slope. Today that is offered for `linear` only (`stages/evaluation.py:_shrinkage`).
- **Every predicting family gets the generic checks:** out-of-fold and held-out calibration (`models/performance.py:calibration`), by level (`models/performance.py:level_calibration`), and at a horizon for Cox (`models/performance.py:horizon_calibration`).
- A pretrained in-context model's probabilities are a posterior predictive under its prior (Müller et al. 2022). Whether they are calibrated on the user's data is what the generic check measures. Nothing assumes it.

**Why.**
- A model fitted on predictors measured one way and used on predictors measured another can be miscalibrated badly enough to be clinically useless (Luijken et al. 2019).
- RECIPES §4.6 gives the small-sample reasons, with their sources, for showing the calibration slope of penalized families per fold.

**How the engine enforces it.**
- A fixture with a deliberately miscalibrated model must be flagged for every predicting family.
- `register_family` refuses a predicting family for a yes/no outcome whose model step lacks `decision_function`, because `models/explain.py:Anatomy.raw_score` needs it.

### C10 · Explanation paths, and inductive-bias curves on shared axes

**What.**
- **Required of every predicting family:** a raw score on a declared scale, which `models/explain.py:Anatomy.raw_score` reads. The prediction for a numeric outcome, and the margin for a yes/no one.
- **Inductive-bias curves need nothing more** (ruling 2). For each top predictor, every family's curve is drawn on one quantile grid shared by every family (`models/explain.py:ale_grid`, `models/explain.py:_curves`). The default is accumulated local effects, because nutrition predictors are strongly correlated and partial dependence then extrapolates outside the data (Apley & Zhu 2020). The contract adds three things:
  - **Every predicting family draws.** Today `_curves` and `_interactions` run only for families with a SHAP path, because `models/explain.py:_work` returns early when `models/explain.py:model_kind` is None. Both need only the raw score.
  - **A spread band** from the reseeded refits (`models/explain.py:_refits`, `RESEEDS`), so seed-to-seed differences show (D'Amour et al. 2022).
  - **The data envelope** is drawn, as the grid's `supported` mask already computes.
- **Optional, declared:**
  - `attribution`: exact SHAP for linear models, path-dependent TreeSHAP for trees, or none. A family with none still gets curves and interactions.
  - `architecture`: the equation, the tree structure, the shrinkage path (`models/explain.py:shrinkage_path`), and the new spectrum view of §2.3.
  - Interaction readouts: Friedman and Popescu's H statistic is model-agnostic and exists (`models/explain.py:h_statistics`). An MLP may add a weight-based screen (Tsang et al. 2018).
  - The average gradient outer product, for any family with a raw score (§2.2).
- **The floor stays.** A family that does not beat the no-predictor baseline draws no curve (`models/explain.py:floor_of`). Under inference, a curve is drawn for the declared exposure only, as description.

**On the card:**
- "Each model's idea of how sodium relates to blood pressure, on the same axes. They differ because each assumes a different shape." Known as accumulated local effects (Apley & Zhu 2020).
- When the corrected comparison ties two families whose curves disagree: "These models score the same but tell different stories about sodium." Known as the Rashomon effect (Fisher et al. 2019).

**Why.**
- Showing the same predictor's curve per family is informative precisely because inductive bias shapes what each learns (Chang et al. 2021). Fig. 3a of Hollmann et al. (2025) is a published precedent on toy functions; TurboTab draws it on the user's own top predictors.
- Where many well-performing models rely on different covariates, report the range, not one model's story (Fisher et al. 2019). Equivalent held-out scores can hide very different behavior (D'Amour et al. 2022).
- In Simon et al.'s terms (2026, §2.5), where families agree the shape is close to universal for this table, and where they disagree it is a property of the family.

**How the engine enforces it.**
- A parametrized test over the registry: every predicting family draws a curve on a fixture, and the curve equals an ALE computed by hand from its raw score.
- The curve-shape test of C5.
- **Today:**
  - Curves exist for the linear families and boosted trees only (`models/explain.py:LINEAR_MODELS`, `models/explain.py:model_kind`). The mixed and GEE estimators have `coef_` and a raw score but are blocked by the class-name gate.
  - The explain stage supports regression and yes/no outcomes only (`stages/explain.py:SUPPORTED_TASKS`).
  - **Plausible:** under Explore's imbalance lever the model step is `methods/levers.py:ImbalanceCorrected`, `model_kind` returns None, and no family is explained.

### C11 · Bootstrap and validation soundness

**What.**
- `bootstrap_optimism`: whether Harrell's bootstrap optimism correction is sound for the family.
- `flexible`: whether the family is a flexible learner, ranked after the regression families below Riley's minimum. It is declared, not derived. Today `models/selection.py:is_flexible` falls back to `not bootstrap_optimism`, and no family declares it.
- **Internal splits.** Any internal split (early stopping, out-of-bag, a nested search) is drawn by unit, by time when the folds follow time, and by whole PSU under the population answer (RECIPES §4.3).
- **For an in-context family:** "refitting" in a fold means conditioning on that fold's training rows only. The context must never hold a validation or held-out row. Its bootstrap soundness is unknown, so it declares `bootstrap_optimism = False` (a convention) and keeps its cross-validated score.

**Why.**
- The engine's own replication: a learner that nearly memorizes its rows scores the original rows inside each resample almost perfectly, so the bootstrap understates its optimism (`models/base.py:FamilyBase`, the `bootstrap_optimism` comment).
- The validation method strongly affects which family looks best (Erickson et al. 2025). Tuning without a held-out set biased a widely cited benchmark (Wainberg et al. 2016).

**How the engine enforces it.**
- **A soundness test per family.** On a null fixture, a family declaring `bootstrap_optimism = True` must have a bootstrap-corrected score within 2 Monte Carlo standard errors of its fresh-data score. This generalizes the repair round's replication. It also settles the screened elastic net at p ≫ n, which is declared sound but unverified.
- **Today:**
  - `featurewise` skips the loop (`stages/modeling.py:fit_stage`), yet the methods reference prints "sound: yes" for it (`reference/methods.py:family_section`). It should print "not applicable".
  - `flexible` is read by `models/selection.py:is_flexible` and declared by no family.

### C12 · Replay determinism and tolerance

**What.**
- Seeds derive from SHA-256 of canonical JSON, and thread counts and library versions are recorded (RECIPES §4.7).
- A family declares `replay_tolerance`. The default is DoD gate 6's 1e-12. A looser tolerance needs Nolan's approval as an amendment to gate 6, as RECIPES §4.7 already says for XGBoost.
- **Trained networks** run on CPU with deterministic operations, a fixed thread count, and seeds for initialization and batch order. Local compute is the baseline; a server option runs the same pinned plan.
- **In-context families** record the checkpoint hash and fix the context's row order from the plan's seed.

**Why.** V2_DEFINITION_OF_DONE gate 6: "Replaying the record reproduces the model matrix and the estimates."

**How the engine enforces it.**
- The export replay test (`tests/acceptance/test_export.py`) runs for every registered family, not only `linear` and `linear + elastic_net` as today.
- RECIPES T13 adds pinned and re-run replay for tuned families.

### C13 · Reference tests, including the solvable settings

**What.** Every family ships three kinds of reference test.

1. **An independent implementation.** As today: statsmodels, lifelines, glmnet, R, or a hand implementation written from the primary source (§3.1 lists them).
2. **Invariance probes** for C3:
   - **The rotation probe** rotates the scaled input matrix by a random orthogonal matrix. A family declaring `rotation` keeps its predictions to tolerance. A family not declaring it must change them. Trees and FT-Transformer change, and ResNet-style networks do not (Grinsztajn et al. 2022). L2-penalized models and backpropagation-trained networks are rotationally invariant (Ng 2004), so ridge on its scaled matrix must not change.
   - **The monotone probe** applies a monotone change to one column. A family declaring `monotone_per_column` keeps its predictions.
   - Both double as teaching exhibits ("why trees and ridge disagree on this table").
3. **Known answers from solvable settings,** declared in `solvable`:

| Setting | The known answer | Tolerance and conditions | Source |
|---|---|---|---|
| Ridge, closed form | The SVD solution at each λ; df(λ) equals the hat matrix's trace | 1e-8 and 1e-10 | ESL §3.4.1; RECIPES T11 |
| Deep linear network from small initialization | Gradient flow reaches the minimum-norm solution pinv(X)y on TurboTab's encoded matrix; OLS when X has full column rank | Report the initialization scale; the answer holds only as that scale goes to zero | Yun et al. 2021; Hastie et al. 2022 |
| Least squares trained by gradient descent | Its path tracks ridge with λ = 1/t, with risk within 1.69 times ridge's along the whole path | A bound, checked on fixtures | Ali et al. 2019 |
| Deep linear network, dynamics | With whitened inputs and small task-aligned initialization, singular modes are learned one after another, largest first | Order of appearance, not exact times | Saxe et al. 2014 |
| Wide network in the lazy regime | The mean prediction over random initializations follows Θ(x,X)Θ⁻¹(I − e^(−ηΘt))Y, which is ridgeless NTK kernel regression at convergence | Average over seeds, or zero the initial output first; the output scale is set explicitly to enter the lazy regime; tolerances allow for finite width | Lee et al. 2019; Jacot et al. 2018; Chizat et al. 2019 |
| Wide network, independent oracle | The NTK computed by the Neural Tangents library | A test-only dependency (it is JAX-based) | Novak et al. 2019 |

   A ridge term in the lazy oracle appears only with explicit regularization or early stopping (Ali et al. 2019). "Kernel ridge regression" is a loose description of the lazy limit, which Simon et al. (2026, §2.1) also use.
4. **The curve-shape test** of C5.

**Why.**
- Simon et al. (2026, §2.1): solvable settings are "analytically tractable cornerstones" that "reveal phenomena and mechanisms to look for".
- For TurboTab, they are references no implementation bug can share. A deep linear network that misses pinv(X)y on the encoded matrix has a bug in its training loop, its encoding or its initialization, whatever its score.
- No test uses a solvable setting today. The harness is new infrastructure (MC-14).

**How the engine enforces it.** The fold-in gate (C14) requires item 1 always, item 2 for every declared invariance, item 3 for every declared solvable setting, and item 4 for every declared curve shape.

### C14 · The fold-in gate

A family joins the shelf only when every blocking item passes. One parametrized acceptance test, `tests/acceptance/test_family_contract.py` (MC-12), runs the automatic items for every registered family. A family added later fails CI until it passes.

| # | Item | How it is checked | Blocking |
|---|---|---|---|
| 1 | Registers cleanly: identity complete, vocabularies, word budgets, recipe and tuning checks | `register_family` | yes |
| 2 | No key or class-name switch outside its module | The no-switch test (C1) | yes |
| 3 | Its profile fields exist, and its `assess` is unchanged when the outcome is permuted | The outcome-permutation test (C4) | yes |
| 4 | Every quiet name and source resolves in the verified citation registry | Registry test (C5) | yes |
| 5 | Tuning: RECIPES T1–T17 where it tunes | RECIPES §8 | yes |
| 6 | Each complexity formula matches its closed form | Formula tests (C7) | yes |
| 7 | Each diagnostic fires on one fixture and stays silent on another | Diagnostic fixtures (C8) | yes |
| 8 | Calibration: the generic checks run; a miscalibrated fixture is flagged | Calibration fixture (C9) | yes |
| 9 | Curves: it draws one on a fixture, matching a hand-computed ALE, with its declared shape | Explanation tests (C10) | yes |
| 10 | Bootstrap soundness as declared | Soundness test (C11) | yes |
| 11 | Export replay reproduces its matrix and estimates at its tolerance | `tests/acceptance/test_export.py` (C12) | yes |
| 12 | Independent implementation, invariance probes and solvable settings | Reference tests (C13) | yes |
| 13 | The methods reference regenerates with its contract-shaped entry | `reference/methods.py:family_section` | yes |
| 14 | Its methods sentence names it and its settings | Voice tests | yes |
| 15 | The shelf's estimate is within a factor of 3 of the measured time | RECIPES T6 | no (Tier B) |
| 16 | Every journey's first pick is unchanged, or the change is approved | RECIPES RT-13 (a heavy run, scheduled with Nolan) | yes, on approval |
| 17 | Plain words and purpose-registry entries for any new element | P0.12 word list; purpose registry | yes |
| 18 | The prediction and inference reviewers' packets carry its row | SIZING R4 | before release |

---

## 2 · Post-fit intelligence from theory

Everything in this section exists only after Fit. Each item names its view class (UNDERSTANDING_LAYER §1.3).
- **Under Predict,** it reads the training rows only. The held-out rows stay sealed.
- **Under Estimate,** it appears only after the lock.

Its effects are the ones the legality matrix allows: labels, disclosures and sensitivity analyses, plus the one candidate question 2 asks about. None of it prunes a family, changes the trunk or re-ranks the shelf.

### 2.1 Target alignment: why the penalized and low-rank families did well or badly (ruling 3)

**What it measures.**
- Take the family's input matrix Z on the fit's rows, centered and scaled as the family scales it (the explain stage's `anatomy.matrix(A_all)`), and its decomposition Z = UDVᵀ.
- The share of the centered outcome's power along principal direction j is aⱼ = (uⱼᵀy)²/‖y‖².
- **The alignment curve** C(ρ) = a₁ + … + a_ρ, with directions in order of spread, is the cumulative power distribution, the sample version of Canatar et al.'s (2021) task-model alignment. For ridge and principal-component regression the eigen-directions are the input's principal directions. For a kernel family they are the kernel's.
- For a yes/no outcome, y is the 0/1 code. That is an approximation for logistic fits, and the card says so.

**Why it explains results.**
- Ridge shrinks direction j by dⱼ²/(dⱼ² + λ), so it shrinks the low-spread directions most (ESL §3.4.1). Kernel and NTK regression learn high-eigenvalue modes first, and higher alignment means lower error at every sample size (Canatar et al. 2021).
- **So when the outcome's signal lies along the widest directions, penalized and low-rank families do well.** When it lies along narrow ones, they do badly. Small-variance components "can be as important as those with large variance" (Jolliffe 1982), which is exactly why alignment cannot be assumed before the fit.
- **When the tuned penalty sits at the bottom of its grid** with fewer rows than columns, high alignment explains it. The many narrow directions already act as an implicit ridge, so the best explicit penalty can be zero (Kobak et al. 2020; Wu & Xu 2020). RECIPES' `penalty_at_edge` concern then gains this explanation instead of only a warning.
- **For kernel families,** the eigenlearning framework's conservation law explains underperformance: a kernel can learn only so much in total, and here it spent it where the outcome had little signal (Simon et al. 2023).

**On the card** (after Fit; plain, then quiet):
- "Most of the outcome's signal lay along your columns' widest directions, which the penalty barely shrinks." Known as task-model alignment (Canatar et al. 2021).
- "Much of the outcome's signal lay along narrow directions, which the penalty shrinks most, so ridge gave some of it up." Known as kernel-target alignment (Cortes et al. 2012).
- **Under Estimate** no score is served (BLUEPRINT §12, ruling 13), so the card describes where the signal lay and makes no claim about doing well.

**View class and placement.**
- **View class:** O3, a predictor-by-outcome quantity. Under prediction it may label and disclose. After the lock under inference it may diagnose, label and disclose. It is never on the shelf and never before Fit.
- **Where:** computed in the explain stage beside `models/explain.py:_architecture`, and stored on `models/explain.py:FamilyExplanation` as `alignment`. Drawn on the tapestry with §2.3's view.
- **A cross-check, v2.x:** KARE estimates kernel ridge regression's risk from training data alone (Jacot et al. 2020). It may sit beside BBC-CV after the fit for ridge and kernel families. It must never rank families before Fit.
- **Thresholds:** "most" means C at the effective rank is 0.8 or more (a convention). The literature search found no study that measures task-model alignment on real tabular benchmarks, so the reference journeys will be the first check.

### 2.2 When a flexible family wins and its explanations concentrate: a labeled hypothesis (ruling 6)

**What fires it.** Two conditions, both on the training rows of a Predict track:
1. **A flexible family clearly beats the linear ones.** The selection-corrected difference between the regression-with-splines benchmark and the best flexible family (`models/selection.py:interpretable_cost`, computed in `stages/evaluation.py:evaluation_stage`) has an interval that excludes zero in the flexible family's favor. "Flexible" is the family's declared `flexible` (C11).
2. **Its explanations point at a few hidden combinations.** Either of:
   - **the average gradient outer product** (AGOP) of the winning family's raw score has its top one to three eigenvalues holding at least 80% of its trace (a convention), and its leading directions mix several inputs;
   - **an interaction pair** stays in the top across reseeds (`models/explain.py:Interaction`, `in_top`).

**How the readout is computed.**
- **The AGOP** is the average over rows of ∇f(x)∇f(x)ᵀ on the model's inputs. Its top eigenvectors are a family-agnostic estimate of the few combinations the outcome depends on (Radhakrishnan et al. 2024). The idea predates neural networks as the outer product of gradients (Xia et al. 2002), and the statistician's name for the structure is a multi-index model (Li 1991). Forests get it the same way (Rauniyar 2025, preprint).
- **Gradients are local differences across each input's quantile intervals at observed rows,** as ALE takes them. Trees and smooth models are then treated alike, and nothing is evaluated outside the data (a convention, following Apley & Zhu 2020).
- **A model-free cross-check:** sliced inverse regression estimates the same directions without fitting the model (Li 1991). Agreement between SIR and the AGOP raises the card's confidence line; disagreement is said.
- **Main effects present.** Interactions that SGD-trained networks learn efficiently typically build on lower-order effects (Abbe et al. 2022). A combination or pair whose inputs show no main effect in the curves is reported as weaker.

**Why it is a hypothesis, not a finding.**
- The theory says flexible models beat kernel-like ones when the outcome depends on a few hidden directions and there are enough rows to find them (Ghorbani et al. 2020; Damian et al. 2022; Bietti et al. 2022 for the single-direction case).
- But those results hold for Gaussian or spherical inputs in high dimension. No study tests them on mixed categorical and continuous tabular data. On these rows the pattern is a hypothesis (the leash).
- A pretrained in-context family blurs the line between linear and flexible that this comparison relies on (Zhang et al. 2025). When it is the winner, the card says so.

**On the card** (the plain sentence stays within 22 words):
- "Boosted trees beat the regression benchmark, and their predictions move mostly along one mix of sodium and potassium."
- "A question for new data, not a finding: does that mix matter on its own?"
- Label: "Exploratory: suggested by these rows". Known as a multi-index model (Li 1991); the readout is the average gradient outer product (Radhakrishnan et al. 2024).
- One click, "Check it across refits", runs the explanations' reseeded refits when they have not run (calm: one obvious action).

**What it may lead to: prediction plus explainability begets further inference.** Using machine-learned patterns to generate hypotheses, with testing as a separate step, is a published procedure (Ludwig & Mullainathan 2024). The leash rows:

| What | Predict track | Estimate track |
|---|---|---|
| The noticing itself | Labels the explanations; a disclosure in Write-up | Never fires: no score is served there |
| A term for the interpretable family (the combination, or the pair's product) | Question 2. Recommended: only as a version added after scores (RECIPES §3), disclosed, the earlier version kept and the choice corrected by BBC-CV, labeled "suggested by the explanations". Otherwise not offered. | n/a |
| An effect question on the same rows | n/a | Only as an exploratory secondary, recorded as "suggested by data inspection" (`methods/interaction.py:POST_HOC`, set by `methods/interaction.py:_modifier_records_when` once the plan is locked) and counted in the family of tests. Never the locked primary. |
| A check on rows no model has seen | When a holdout was sealed before any score, once at the opening, as a labeled secondary (`opened_change_is_secondary`, RECIPES §7.2) | n/a |
| A question for new data | A Write-up sentence | The same |

- **A properly powered follow-up** on held-out or new data can use algorithm-agnostic variable importance with valid intervals (Williamson et al. 2023).
- **New data must use the same measurement protocol,** or the model's predictions may not carry over (Luijken et al. 2019). The Write-up sentence says so.
- It never prunes, never changes the trunk, and never re-ranks the shelf.

**View class and placement.**
- **View class:** the trigger reads scores (O4). The SIR cross-check is O3, on training rows.
- **The O4 cell under prediction** allows labels, disclosures and a baseline comparison, but not a new candidate. Question 2 asks to amend that one cell narrowly.
- **Where:** a post-fit noticing stage, `late_notices`, with deps fit, design and evaluation, under prediction. It computes the AGOP on the final fits itself (`fit.objects["fitted"]`, as `stages/explain.py:explain_stage` reads them), by finite differences on capped rows, with no refits. `evaluation` and `explain` are siblings today, and `explain` runs only when asked, so the noticing cannot wait for it.
- **Threads:** this is the catalog thread `thread:shared-learned-interaction` (CROSSWALK, "Noticings born after the fit"), widened from pairs to combinations.
- **Fixtures:** a single-index generator where it fires; an additive nonlinear generator, whose AGOP is spread across the inputs, where it stays silent; a linear generator where condition 1 fails.

### 2.3 What the penalty shrinks, on the tapestry (ruling 5)

**The view.** A Focus on the penalty (FOUNDATION §5), drawn with the curve view kind (SIZING P0.3b):
- **x:** the family's principal directions, widest first;
- **bars:** each direction's spread, dⱼ²;
- **line:** the shrink factor dⱼ²/(dⱼ² + λ) at the chosen penalty, read from the tuning record (RECIPES RT-5f);
- **one number:** df(λ) (ESL §3.4.1, eqs. 3.47 and 3.50);
- **after Fit:** the alignment bars aⱼ of §2.1 on the same axis, so the reader sees whether the outcome's signal sits where the penalty barely shrinks or where it shrinks most.

**On the card:**
- "Ridge keeps most of the widest directions and shrinks the narrow ones hard." Known as L2 shrinkage (ESL §3.4.1).
- "This setting uses about 14 of your 31 effective directions." Known as effective degrees of freedom.

**Per family.**
- **Ridge:** exact.
- **Elastic net:** the line shows its ridge part only, and the card says so. The lasso part drops columns, which the existing path view shows (`models/explain.py:shrinkage_path`).
- **Logistic ridge:** the factors use the weighted matrix at the fitted probabilities, labeled an approximation (a convention).
- **Families trained by gradient descent with early stopping (v2.x):** placed at λ ≈ 1/t, labeled approximate (Ali et al. 2019).
- **Random forest:** no per-direction line, because its splits follow the columns, not these directions. Its knob appears on the tuning line: "Choosing from a random handful of columns at each split is this forest's penalty knob." Known as implicit regularization (Mentch & Zhou 2020).
- **Filled blanks:** a note that the fill adds shrinkage of its own under MCAR, so the effective penalty is larger than the tuned one (Ayme et al. 2023).
- **Error-prone inputs,** such as self-reported intake: a note that input noise acts like a penalty (Bishop 1995). Known as regression dilution (Hutcheon et al. 2010).

**Before and after Fit.**
- **Before Fit,** the bars alone are the input's shape, an O0 measure, and may appear under the shelf's "More angles" as part of the input profile.
- **The shrink line needs the tuned penalty,** which was chosen with the outcome, so it appears only after Fit. The alignment bars appear only after Fit (§2.1).

**Where:** `models/explain.py:Architecture` gains a `spectrum` part beside `kind="shrinkage"`, computed from the SVD of the scaled Z.

### 2.4 The named-phenomena registry (ruling 4)

**The registry.**
- `phenomena.py` holds one `Phenomenon` per entry: its key, the plain card sentence (≤ 22 words), its quiet name, its sources, where it is detected, its view class, the families it applies to, and its guard: when the label must *not* be used.
- Every `known_as` on a concern, curve, card or thread points into it.
- A test checks that every source key resolves in the verified citation registry (SIZING X4), and that every entry has a fixture where its detector fires and one where it stays silent.

| Phenomenon | The plain card sentence | Known as (source) | Where it is detected | Guard |
|---|---|---|---|---|
| Double descent | "Error rose, then fell again, as the model grew past the size that fits every training row exactly." | Double descent (Belkin et al. 2019; Hastie et al. 2022) | A teaching exhibit only: a learning curve over capacity with the penalty off | Tuned penalties make risk monotone, so users rarely see it (Nakkiran et al. 2021). For trees and boosting, name the capacity axis being varied, or do not use the label (Curth et al. 2023). Kanoh (2026, preprint) is watched, not used. |
| Smoothness, or spectral, bias | "The neural net's curve is smoother than the trees': it learns broad trends before sharp steps." | Spectral bias (Rahaman et al. 2019; for tabular data, Beyazit et al. 2023) | After Fit, on the inductive-bias curves, when a neural or kernel family's curve is visibly smoother than the tree curves on the same axes | Neural nets are biased toward overly smooth solutions (Grinsztajn et al. 2022). The card may name the remedy slot: numeric embeddings or rank scaling (§4). |
| Rotation invariance and columns that may not matter | "With many columns that may not matter, a lasso-type penalty needs far fewer rows than ridge or a neural net." | Rotational invariance (Ng 2004) | Before Fit, from width and p/n only (O0); the rotation probe as a teaching exhibit | A worst-case bound, not a forecast. Whether a column matters depends on the outcome, so it is never judged before Fit (Grinsztajn et al. 2022). |
| Lazy versus rich training | Lazy: "This network barely changed its inner weights while learning, so it behaved like a fixed-feature model." Rich: "This network reshaped its inner weights around a few directions: it built its own features." | Lazy versus rich training (Chizat et al. 2019); feature learning (Ba et al. 2022) | After Fit: the C8 weight-movement diagnostic, trained networks only. Before Fit, the declared regime (C4). | The threshold is a convention. |
| Edge of stability | "Training took the largest steps the loss surface allowed; the loss wobbled but kept falling." | Edge of stability (Cohen et al. 2021, as summarized in Simon et al. 2026, §2.3) | After Fit: the C8 sharpness diagnostic | Full-batch gradient descent only. |
| Task-model alignment | §2.1's sentences | Task-model alignment (Canatar et al. 2021); kernel-target alignment (Cortes et al. 2012) | After Fit, explain stage (O3) | Never before Fit. Describes only, under Estimate. |
| Effective rank, and implicit ridge | Before Fit: "Your 412 columns spread along about 31 directions; the rest add little new." After Fit, penalty at the floor: "The best penalty here was almost none: your many narrow directions already act as one." | Effective rank (Bartlett et al. 2020); implicit ridge regularization (Kobak et al. 2020) | Before Fit, input profile (O0); after Fit, with `penalty_at_edge` and high alignment | The second sentence only with fewer rows than columns. "Benign overfitting" (Bartlett et al. 2020) is named only for a fit that interpolates. |
| Implicit regularization in a forest | "Choosing from a random handful of columns at each split is this forest's penalty knob." | Implicit regularization (Mentch & Zhou 2020) | The tuning line and §2.3 | none |
| Hidden combinations | §2.2's sentences | Multi-index model (Li 1991); average gradient outer product (Radhakrishnan et al. 2024) | After Fit: `late_notices` (§2.2) | Exploratory label always. |
| Same score, different stories | "These models score the same but tell different stories about sodium." | Rashomon effect (Fisher et al. 2019); underspecification (D'Amour et al. 2022) | After Fit: BBC-CV's tie and the curves' disagreement (`thread:shared-rashomon-disagreement`) | Report a range, not one family's story. |
| Measurement error flattens curves | "Error in measuring intake flattens its curve in every model; a flat curve here is expected, not a model failing." | Regression dilution (Hutcheon et al. 2010) | A measurement-error reading (`stages/modeling.py:_measurement_error_line`) plus flat curves after Fit | Random error only. |
| Filling blanks shrinks the fit | "Filling blanks with typical values already pulls the fit toward zero, on top of the penalty." | Implicit regularization by imputation (Ayme et al. 2023) | Before Fit, from blank counts (O0); after Fit, on §2.3's view | MCAR and a linear family only. |

---

## 3 · Today's and v2's families against the contract

### 3.1 Today's nine families

**Gaps all nine share:**
- **C1:** key and label only, with no library, version or defaults version. `screened_elastic_net` is registered outside `models/__init__.py` (`methods/omics.py:_register_screened_family`) and has no `voice.py:_FAMILY_LABEL` entry, so its sentence falls back to the bare key.
- **C3:** no recipe slots and no invariances. Trees' native blanks are unreachable (RECIPES F1).
- **C4:** no input profile. No `assess` reads collinearity, the spectrum, blanks or the width after encoding.
- **C5:** plain statements only, with no quiet names, sources or curve shape.
- **C7:** no declared complexity knobs.
- **C12:** export replay is tested only with `linear` and `linear + elastic_net` (`tests/acceptance/test_export.py`).
- **C13:** no invariance probes, curve-shape tests or solvable settings.

**Where they differ** ("part" means implemented with gaps; "decl" means a flag is set and nothing verifies it):

| Clause | linear | elastic_net | boosted_trees | featurewise | proportional_odds | mixed | gee | cox | screened_elastic_net |
|---|---|---|---|---|---|---|---|---|---|
| C2 tasks and purposes | 4 tasks; both purposes | 4; both | 4; both | reg, bin; inference only; no predictions | ordinal; ordered levels | reg | reg, bin | time to event | reg, bin; prediction only |
| C2 inference table | intervals (HC3, CR2, Firth, survey) | shrunk, no intervals | none | intervals (classical t, CR2 to 200 exposures, BH; no survey) | intervals (Wald, sandwich, survey) | intervals (Satterthwaite; no survey) | intervals (CR2; no survey) | intervals (Wald, Lin–Wei, survey) | refused |
| C4 assess reads | rows, parameters, events, outcome mean and SD, units | rows, columns, purpose | rows, columns, purpose | purpose, lenses, columns vs rows | class counts, columns | units, columns, purpose | units, EPV | events, units | purpose, p vs n, then the elastic net's |
| C6 tuning | n/a | part (F4, F5, F13) | no (F2) | n/a | n/a | n/a | n/a | n/a | part (F4) |
| C8 diagnostics | yes | no | no | part | yes | part | part | yes | no |
| C9 calibration | generic, plus shrinkage updating | generic | generic | n/a | generic, by level | generic | generic | at a horizon | generic |
| C10 explanation | SHAP, equation | SHAP, path | TreeSHAP, trees | n/a | none | blocked by the class-name gate | blocked by the class-name gate | none | SHAP, path |
| C11 bootstrap | decl yes | decl yes | decl no | n/a, printed "yes" | decl yes | decl yes | decl yes | decl yes | decl yes, unverified at p ≫ n |
| C13 independent reference | yes | path only | TreeSHAP only | yes | yes | yes | yes | yes | none |

The inventory behind this table cites each cell, for example `models/linear.py:Linear.assess` and `models/boosted_trees.py:BoostedTrees.assess`. The independent references are:
- linear: HC3 and CR2 against their definitions, Firth against Haldane, the equation against R (`tests/acceptance/test_wp2_intervals.py`, `tests/acceptance/test_explain.py`);
- elastic net: its path against glmnet only, with the chosen penalty unchecked;
- boosted trees: TreeSHAP against shap, and stopping by whole unit;
- featurewise: statsmodels with Benjamini–Hochberg (`tests/acceptance/test_wp11_omics.py`);
- proportional odds: MASS::polr and statsmodels (`tests/acceptance/test_wp12a_ordinal.py`);
- mixed, GEE and Cox: REML, sandwiches and lifelines (`tests/acceptance/test_wp12b_cox_mixed_gee.py`);
- screened elastic net: none; chain tests only (`tests/acceptance/test_ms7_chains.py`).

### 3.2 What the four v2 families must ship with

RECIPES §2.2, §4.1 and §9 already specify their recipes, tuning, wrappers and RT tests. This contract adds the rest:

| Clause | Ridge | Robust linear (Huber) | Random forest | XGBoost |
|---|---|---|---|---|
| C1 | scikit-learn `Ridge`; logistic with an L2 penalty | statsmodels `RLM` wrapper | scikit-learn forest | `xgboost`, tree booster; threads recorded |
| C2 | reg, bin, multiclass; prediction; inference as a shrunk table with no intervals | reg; prediction only | reg, bin, multiclass; prediction; description only under inference | as the forest |
| C3 invariances | rotation, on its scaled matrix | column scale | monotone per column | monotone per column |
| C4 reads | rows, columns, p/n, effective rank, spectrum, condition number, indicators | rows, columns, outlying share | rows, columns, blanks routed, irregularity | as boosted trees |
| C4 prior | Ng 2004 (lower bound) | convention | convention | McElfresh et al. 2023 |
| C5 quiet name | L2 shrinkage (ESL §3.4.1) | Huber loss (Friedman 2001; RECIPES §2.2 for Huber 1964) | implicit regularization (Mentch & Zhou 2020) | gradient boosting (Friedman 2001) |
| C5 curve shape | straight | straight | steps | steps |
| C6 | path (RECIPES §4.1); the grid reaches near zero for wide data (Kobak et al. 2020) | by hand only | search, out of bag where allowed | search |
| C7 knobs | λ, df(λ) | threshold t | mtry, leaf size; effective smoothing (Curth et al. 2024, preprint) | rounds, learning rate, depth, L2; no double-descent label without a named axis |
| C8 | penalty at the edge; calibration slope per fold | IRLS convergence | out-of-bag error | early-stopping round |
| C9 | margin and probability | value | probability; the wrapper adds a margin `decision_function` | margin `decision_function` (RECIPES §2.2) |
| C10 | linear SHAP; equation; shrinkage path; §2.3's spectrum view; §2.1's alignment | linear SHAP; equation | compiled TreeSHAP (RECIPES RT-5d); curves via the margin | SHAP from `pred_contribs` |
| C11 | bootstrap sound (path families are bootstrapped, RECIPES §4.3); not flexible | sound; not flexible | not sound; flexible | not sound; flexible |
| C12 | 1e-12 | 1e-12 | 1e-12, prediction single-threaded (RECIPES §4.7) | 1e-12, or a declared tolerance with Nolan's approval (RECIPES §4.7) |
| C13 | closed form and df(λ); rotation probe keeps predictions | IRLS from Holland and Welsch; MASS::rlm (RECIPES T11) | direct library fits; TreeSHAP to 1e-6; monotone probe keeps predictions, rotation probe changes them | native `xgb.train`; `pred_contribs`; the same probes |

### 3.3 The switches on family keys to retire

Each of these is a place a new family must be added by hand today. Each becomes a read of a declaration.

| Where | What it switches on | Replaced by |
|---|---|---|
| `voice.py:_FAMILY_LABEL`, `voice.py:_family_label` | methods labels by key, `linear` by task | `describe(task, purpose)`'s label in the methods register |
| `methods/omics.py:model_clause` | elastic net and screened elastic net | `tuning.kind` and the plan's sentence |
| `stages/modeling.py:fit_stage` | the collinearity concern for `linear` only | `"collinearity" in diagnostics` |
| `stages/effects.py:SEQUENCE_FAMILIES`, `stages/effects.py:matrix_table` | families with a matrix table | `inference.matrix_table` |
| `stages/effects.py:_Run.diagnostics` | diagnostics for `cox` and `linear` only | `diagnostics` |
| `stages/evaluation.py:_shrinkage` | shrinkage updating for `linear` only | `updating` |
| `methods/interaction.py:SUPPORTED` | families that test product terms | `inference.product_terms` |
| `models/survey.py:_DESIGN_FAMILY`, `models/survey.py:has_design_estimator` | the design-based family, and a signature check | `inference.design_based` |
| `models/explain.py:LINEAR_MODELS`, `models/explain.py:model_kind` | estimator class names | `attribution` and `architecture` |
| `estimand.py`, the `family_needs_featurewise` refusal | `featurewise` by key | `predicts` and `purposes` |
| `decisions.py:model_families` | fallback keys | the registry, always importable |
| `models/selection.py:is_flexible` | `flexible` derived from `bootstrap_optimism` | `flexible`, declared |

### 3.4 Count mismatches the profile fixes

The shelf's counts today diverge from what a family receives (`stages/modeling.py:shelf_stage`, `stages/modeling.py:predictor_parameters`, `stages/modeling.py:rule_spline_terms`):
- missing indicators are not counted;
- the energy step's dropped or added columns are not counted;
- scale items are counted, though they collapse to one score;
- blanks as a level add a level per category that is not counted;
- the batch drop and the in-fold filters are not reflected;
- category levels are read from summaries over the whole table, held-out rows included;
- the spline rule is applied to every family, though RECIPES §2.3 has tree families skip it.

The input profile is computed from each family's own steps, so each of these is counted where it applies. The timing estimate also stops counting bootstrap refits for families that are never bootstrapped (RECIPES F7, in `stages/modeling.py:_estimates`).

---

## 4 · Neural and in-context families (v2.x)

These come after v2.0.0. Each enters through the same door. Below are the clauses that differ for each.

### 4.1 An MLP family (first)

**Which MLP.**
- An MLP that efficiently imitates an ensemble of MLPs was the best tabular deep model in its authors' evaluation, and MLP-based models beat attention models there (Gorishniy et al. 2025). That makes it the first neural candidate, before a transformer.
- Meta-tuned defaults (Holzmüller et al. 2024) are a reasonable candidate 0, labeled unvalidated below 1,000 rows. Many nutrition cohorts are smaller.

**C1 identity:** optimizer and schedule, batch size or full batch, epochs or the stopping rule, initialization scheme and scale, parameterization and output multiplier, ensemble size, and seeds for initialization and batch order.

**C3 recipe:**
- `scale` defaults to every column on one scale. Rank or quantile scaling is an option, principled because it reduces the target's irregularity (Beyazit et al. 2023).
- A numeric-embedding slot: none, piecewise-linear or periodic (Gorishniy et al. 2022). Random Fourier features are labeled experimental (Sergazinov et al. 2025, preprint).
- **Invariance:** `rotation` only without per-column embeddings. Whether an embedded MLP is rotation-invariant is not sourced, so the probe decides and the declaration follows it.

**C4 assess reads:** rows, columns, effective rank, the irregularity measures and the outlying share.
- Heavy tails and rows far from the centroid lower its fit to the situation (McElfresh et al. 2023; Jeffares et al. 2024).
- Its prior states that deep models catch up mainly under larger time budgets (Erickson et al. 2025).

**C6 tuning:**
- Searched: learning rate, weight decay, dropout, width and epochs. A regularization search is supported (Kadra et al. 2021), with its caveat.
- Structural, never searched: the parameterization and output multiplier.
- Width transfer under muP is not used; small tabular networks are cheap to tune directly (Yang et al. 2021).

**C8 diagnostics:** loss and convergence, lazy versus rich, the edge of stability under full batch, and the neural feature ansatz (Radhakrishnan et al. 2024).

**C10 explanations:** curves via the raw score. The AGOP comes free from its gradients. A weight-based interaction screen (Tsang et al. 2018) is triangulated with the H statistic.

**C11:** flexible; `bootstrap_optimism = False`, following the engine's reasoning for near-interpolating learners (a convention until its soundness test runs).

**C13 solvable settings:** deep linear to minimum norm, gradient descent to ridge, deep linear dynamics, and the lazy wide network against Neural Tangents (C13's table). They are run on the family's own training loop with its nonlinearity removed or its width raised, so they test the loop, the encoding and the initialization that ship.

### 4.2 A tabular transformer (later, if at all)

- **FT-Transformer** tokenizes each feature (Gorishniy et al. 2021), so it is **not** rotation-invariant (Grinsztajn et al. 2022). Its probe must show a change.
- **Against boosted trees** there is "still no universally superior solution" (Gorishniy et al. 2021). MLP-based models beat attention in the more recent evaluation (Gorishniy et al. 2025).
- **Recommendation:** register it only if the MLP family's journeys show a gap a transformer could plausibly close. muP transfer would matter only if it were scaled up (Yang et al. 2021).

### 4.3 A pretrained in-context family (TabPFN-style)

**C1 identity is the prior:** the checkpoint's name and SHA-256, and any input handling it applies internally. It has no training loop and no learning rule.

**C2:**
- **Tasks:** classification, regression, categorical columns and missing values within its validated limits, 10,000 rows and 500 features (Hollmann et al. 2025).
- **Limits:** beyond them the family is refused with exits, never silently subsampled.
- **Inference:** description only.

**C4 assess reads:** rows and columns against those limits. Its prior: dominant on small data (Hollmann et al. 2025; Erickson et al. 2025).

**C5 inductive bias:** "Predicts by approximate Bayesian inference under a prior learned from simulated datasets." Known as a prior-data fitted network (Müller et al. 2022). It is a prior, not a fitted penalty, and the card contrasts it with ridge's explicit penalty. For statisticians it reads as approximate Bayesian inference (Zhang et al. 2025).

**C6 and C7:** `TuningDecl(kind="none")`. Its complexity control is the context: which rows it conditions on.

**C9:** its probabilities are a posterior predictive under its prior (Müller et al. 2022), and the generic check measures their calibration on the user's data.

**C10:** curves via the raw score. Its authors interpret it with SHAP (Hollmann et al. 2025).

**C11:**
- In each fold the context holds that fold's training rows only. That is the fold's "fit".
- `bootstrap_optimism = False` (a convention).
- It is `flexible`. It blurs the linear-versus-flexible contrast that §2.2 relies on (Zhang et al. 2025), so §2.2's card names it when it wins.

**C12:** the checkpoint hash and the context's row order, fixed by the plan's seed.

**C13:**
- **No solvable oracle is sourced.** The known-answer check is Hollmann et al.'s (2025) Fig. 3a toy functions as a sanity test, not a correctness oracle.
- **The leak check:** the context holds no validation or held-out row, checked by perturbation like RECIPES T4.

**Where it runs:** it needs pretrained weights and is heavy. It is offered on the server option first, with local compute when the weights are present.

### 4.4 A feature-learning kernel machine (watch)

xRFM combines kernel machines that learn features through the AGOP with a tree partition. It reports strong results on 100 regression and 200 classification datasets, with native interpretability from the AGOP (Beaglehole et al. 2025, preprint). It would feed §2.2's readout directly. It is a candidate to watch until it has a peer-reviewed venue.

---

## 5 · Engine work packages

**Sizes** follow SIZING: S = 1, S–M = 2, M = 3, M–L = 5.5, L = 8, XL = 20. Every estimate is relative and uncertain by about a third either way. "Brought in by" uses SIZING's column; everything here comes from the rulings of 2026-10-08, except where it is DoD §2's existing requirement that each family enter "through the method contract, with a reference test and an explanation path".

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **MC-1** Declarations | §1's members in `models/base.py`: `Identity`, `InferenceDecl`, `Prior`, `Knob`, `Named`, `Source`, invariances, curve shape, diagnostics keys, output, updating, attribution, architecture, solvable keys, tolerance, sources. `register_family`'s checks. `FamilyInfo` and `ShelfFamily` extended. The nine families declare theirs. | RT-2 (recipes) | M | engine | DoD §2 |
| **MC-2** No switches on keys | §3.3's twelve places read declarations; the no-switch test with its allowlist at zero | MC-1 | M–L | engine | DoD §2 |
| **MC-3** The trunk stage | `trunk` split out of `design` (the shared-step fit, lineage, matrix file, warnings). `shelf` and `design` depend on it. The shelf's reads widen and it requires the shared answers (C4). Versions bumped: shelf 15→16, design 24→25. | P0.4 | M–L | engine | rulings of 2026-10-08 |
| **MC-4** The input profile | `InputProfile` per family on its default recipe through `family_spec`, fitted up to the last outcome-free step, with size rules after it. Spectrum, effective rank, condition number, irregularity, outlying share, with cost caps. `Situation.inputs`; `Assessment.measures`; structured concerns; `reads`. §3.4's count fixes. | MC-1, MC-3, RT-2, RT-3 | L | engine | rulings of 2026-10-08 |
| **MC-5** Assess rewritten | The nine families and the four new ones read their profiles, with two-register concerns and sourced priors; the outcome-permutation test | MC-4 | M | engine | rulings of 2026-10-08 |
| **MC-6** Live re-assessment | `models/previews.py:models_preview` returns each family's profile and its `assess` under an edited recipe. It stops before outcome-reading steps, which today would plausibly raise inside `consequences.py:plan`. | MC-4 | S–M | engine | rulings of 2026-10-08 |
| **MC-7** Curves for every family | `_curves` and `_interactions` decoupled from `model_kind`; the raw-score contract; the spread band; the data envelope; the Rashomon label | MC-1 | M | engine | rulings of 2026-10-08 |
| **MC-8** Alignment and the shrinkage view | The SVD of the scaled Z; shrink factors and df(λ); C(ρ) and its bars; the `spectrum` part of `Architecture`; `alignment` on `FamilyExplanation`; the Focus view's spec | MC-7, RT-5b, RT-5f | M | both | rulings of 2026-10-08 |
| **MC-9** The hypothesis noticing | `late_notices` (deps fit, design, evaluation); the AGOP by local differences; the SIR cross-check; main effects; the trigger on `interpretable_cost`; the card, the leash rows and the sentences; fixtures that fire and stay silent. Widens `thread:shared-learned-interaction`, already counted at 0.5 in T2. | MC-7, U1, U2 | L | both | rulings of 2026-10-08 |
| **MC-10** The phenomena registry | `phenomena.py`; `known_as` and `source` on concerns, curves, cards and thread sentences; the registry test against X4's citation registry | MC-1, X4 (the registry may land first as a list) | S–M | engine | rulings of 2026-10-08 |
| **MC-11** Probes and replay for every family | Rotation and monotone probes; curve-shape tests; export replay over the registry; soundness tests per family | MC-1 | M | engine | DoD §2 |
| **MC-12** The fold-in gate | `tests/acceptance/test_family_contract.py` over the registry (§1, C14) | MC-1 to MC-11 | M | engine | DoD §2 |
| **MC-13** The methods reference | `reference/methods.py:family_section` prints the contract: sources, two registers, invariances, inference table, "not applicable" where sound does not apply | MC-1 | S | engine | DoD §2 |

**v2.x packages** (after v2.0.0):

| Package | What it is | Size |
|---|---|---|
| **MC-14** The solvable-settings harness | Deep linear to minimum norm; gradient descent to ridge; deep linear dynamics; the lazy wide network against Neural Tangents (a test-only dependency) | L |
| **MC-15** Neural training diagnostics | Lazy versus rich, sharpness against 2/η, the neural feature ansatz | M |
| **MC-16a** The MLP family | §4.1 | L |
| **MC-16b** The in-context family | §4.3, on the server option first | L |

**The total.**
- **v2 packages:** about 50 units, of which about 0.5 (part of `shared-learned-interaction`) is already in T2. About **47 new units**, which is about 6% on top of SIZING's 734 remaining.
- **v2.x:** about 27 units.
- **DoD §2's own requirement** (MC-1, MC-2, MC-11, MC-12, MC-13) is about 15.5 of the 47. It is arguably already implied by C6a's "each family through the method contract", so it may be double-counted against C6a; the overlap is at most a few units.

**How they fit the road** (SIZING "The order, and why").
- **MC-1, MC-2 and MC-13 go first in C6a,** before RT-5b to RT-5e. The four new families then enter through the door instead of adding to §3.3's list.
- **MC-3** needs the stage registry (P0.4) and can run beside the C6 engine work.
- **MC-4, MC-5 and MC-6 follow RT-2 and RT-3** (C6b), because the profile needs `family_spec`.
- **MC-7 and MC-8 land with C7d** (Results under Predict). MC-8's view needs the curve view kind (P0.3b).
- **MC-9 rides with T1/T2** as the widened `shared-learned-interaction`, and needs the thread machinery (P0.9's U1 and U2).
- **MC-10 lands with P0.12** (plain words) and before X4 closes.
- **MC-11 and MC-12 close with RT-14 and G1.**
- **Heavy runs are scheduled with Nolan.** These are RT-13's regenerated journeys and the soundness tests' Monte Carlo runs. The dev machine is beside the bed.

**Not taken.**
- Ranking families by a pre-fit risk estimate such as KARE: it reads the outcome (§2.1).
- Showing target alignment on the shelf: ruling 3.
- Making a pretrained in-context model the default: its limits and dependency weight, and it blurs §2.2's contrast.
- muP for tabular networks: little leverage at these widths.

---

## 6 · Open questions for Nolan

1. **What does "outcome-blind" mean for the shelf?** Ruling 1 says everything is outcome-blind. Today linear, proportional odds, GEE and Cox read the outcome's own counts (events, class sizes, the outcome's mean and SD) for Riley's minimum, EPV and Whitehead's effective size. Those are O1 under the legality matrix, which already lets O1 "set the df budget".
   - **Recommendation:** read "outcome-blind" as no predictor-by-outcome quantity and no score (no O3, no O4). Keep the O1 counts for the sample-size criteria, and keep the O2 library-size check as its own shelf line. The outcome-permutation test (C4) enforces exactly this.
   - The strict alternative would drop TRIPOD+AI item 10's sample-size check from the shelf.
2. **Under Predict, may the hypothesis of §2.2 add a candidate?** The legality matrix's O4 cell under prediction allows labels, disclosures and a baseline comparison, not a new candidate.
   - **Recommendation:** amend that one cell narrowly. A term suggested by the explanations may be added to the interpretable family only as a version added after scores, through RECIPES §3's existing path: disclosed, the earlier version kept, the choice among versions corrected by BBC-CV, and labeled "suggested by the explanations".
   - Otherwise the hypothesis stays a label, an exploratory secondary on an Estimate track, and a question for new data.
3. **Do these rulings enter v2.0.0 by amendment, or by displacing something?** The definition of done admits a new idea only by displacing something, though the 2026-10-07 amendment added without displacing, at your direction. These packages add about 47 units (6%).
   - **Recommendation:** amend v2.0.0 to include them. They are the north star made concrete, and a third of them is DoD §2's own requirement.
   - If the date matters, the displacement candidate is the kept tuning groups (C6c, 33 units). The rulings serve explainability more directly than successive halving, Hyperband, TPE and BOHB do.

---

## 7 · Sources

Only entries verified for this spec are listed, with the theory source. Engine conventions are marked "convention" where they appear.

**The theory source**
- Simon, J., Kunin, D., Atanasov, A., Boix-Adserà, E., Bordelon, B., Cohen, J., Ghosh, N., Guth, F., Jacot, A., Kamb, M., Karkada, D., Michaud, E. J., Ottlik, B., Turnbull, J. (2026). There Will Be a Scientific Theory of Deep Learning. arXiv 2604.21691. https://arxiv.org/abs/2604.21691 (read in full for this spec: §2.1–2.5).

**Learning theory: solvable settings, limits, laws**
- Ali, A., Kolter, J. Z., Tibshirani, R. J. (2019). A Continuous-Time View of Early Stopping for Least Squares. AISTATS 2019 (PMLR 89). https://arxiv.org/abs/1810.10082
- Belkin, M., Hsu, D., Ma, S., Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. PNAS 116(32):15849–15854. https://doi.org/10.1073/pnas.1903070116
- Bartlett, P. L., Long, P. M., Lugosi, G., Tsigler, A. (2020). Benign overfitting in linear regression. PNAS 117(48):30063–30070. https://doi.org/10.1073/pnas.1907378117
- Canatar, A., Bordelon, B., Pehlevan, C. (2021). Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications 12:2914. https://doi.org/10.1038/s41467-021-23103-1
- Chizat, L., Oyallon, E., Bach, F. (2019). On Lazy Training in Differentiable Programming. NeurIPS 2019. https://arxiv.org/abs/1812.07956
- Cortes, C., Mohri, M., Rostamizadeh, A. (2012). Algorithms for Learning Kernels Based on Centered Alignment. JMLR 13:795–828. https://arxiv.org/abs/1203.0550
- Curth, A., Jeffares, A., van der Schaar, M. (2023). A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning. NeurIPS 2023. https://arxiv.org/abs/2310.18988
- Dobriban, E., Wager, S. (2018). High-dimensional asymptotics of prediction: Ridge regression and classification. Annals of Statistics 46(1):247–279. https://doi.org/10.1214/17-AOS1549
- Hastie, T., Montanari, A., Rosset, S., Tibshirani, R. J. (2022). Surprises in high-dimensional ridgeless least squares interpolation. Annals of Statistics 50(2):949–986. https://doi.org/10.1214/21-AOS2133
- Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning, 2nd ed. Springer. §3.4.1, eqs. 3.47 and 3.50. https://hastie.su.domains/ElemStatLearn/
- Jacot, A., Gabriel, F., Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. NeurIPS 2018. https://arxiv.org/abs/1806.07572
- Jacot, A., Şimşek, B., Spadaro, F., Hongler, C., Gabriel, F. (2020). Kernel Alignment Risk Estimator: Risk Prediction from Training Data. NeurIPS 2020. https://arxiv.org/abs/2006.09796
- Jolliffe, I. T. (1982). A Note on the Use of Principal Components in Regression. Applied Statistics 31(3):300–303. https://doi.org/10.2307/2348005
- Kobak, D., Lomond, J., Sanchez, B. (2020). The Optimal Ridge Penalty for Real-world High-dimensional Data Can Be Zero or Negative due to the Implicit Ridge Regularization. JMLR 21(169):1–16. https://arxiv.org/abs/1805.10939
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., Pennington, J. (2019). Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. NeurIPS 2019; J. Stat. Mech. (2020) 124002. https://doi.org/10.1088/1742-5468/abc62b
- Mentch, L., Zhou, S. (2020). Randomization as Regularization: A Degrees of Freedom Explanation for Random Forest Success. JMLR 21(171):1–36. https://arxiv.org/abs/1911.00190
- Nakkiran, P., Venkat, P., Kakade, S., Ma, T. (2021). Optimal Regularization Can Mitigate Double Descent. ICLR 2021. https://arxiv.org/abs/2003.01897
- Novak, R., Xiao, L., Hron, J., Lee, J., Alemi, A. A., Sohl-Dickstein, J., Schoenholz, S. S. (2019). Neural Tangents: Fast and Easy Infinite Neural Networks in Python. https://arxiv.org/abs/1912.02803
- Patil, P., Du, J.-H. (2023). Generalized equivalences between subsampling and ridge regularization. NeurIPS 2023. https://arxiv.org/abs/2305.18496
- Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y., Courville, A. (2019). On the Spectral Bias of Neural Networks. ICML 2019 (PMLR 97). https://arxiv.org/abs/1806.08734
- Saxe, A. M., McClelland, J. L., Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR 2014. https://arxiv.org/abs/1312.6120
- Simon, J. B., Dickens, M., Karkada, D., DeWeese, M. R. (2023). The Eigenlearning Framework: A Conservation Law Perspective on Kernel Regression and Wide Neural Networks. TMLR 2023. https://arxiv.org/abs/2110.03922
- Wu, D., Xu, J. (2020). On the Optimal Weighted L2 Regularization in Overparameterized Linear Regression. NeurIPS 2020. https://arxiv.org/abs/2006.05800
- Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., Gao, J. (2021). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. NeurIPS 2021. https://arxiv.org/abs/2203.03466
- Yun, C., Krishnan, S., Mobahi, H. (2021). A Unifying View on Implicit Bias in Training Linear Neural Networks. ICLR 2021. https://arxiv.org/abs/2010.02501

**Feature learning and hidden combinations**
- Abbe, E., Boix-Adserà, E., Misiakiewicz, T. (2022). The merged-staircase property. COLT 2022 (PMLR 178). https://arxiv.org/abs/2202.08658
- Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., Wu, D., Yang, G. (2022). High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation. NeurIPS 2022. https://arxiv.org/abs/2205.01445
- Bietti, A., Bruna, J., Sanford, C., Song, M. J. (2022). Learning Single-Index Models with Shallow Neural Networks. NeurIPS 2022. https://arxiv.org/abs/2210.15651
- Damian, A., Lee, J. D., Soltanolkotabi, M. (2022). Neural Networks can Learn Representations with Gradient Descent. COLT 2022 (PMLR 178). https://arxiv.org/abs/2206.15144
- Ghorbani, B., Mei, S., Misiakiewicz, T., Montanari, A. (2020). When Do Neural Networks Outperform Kernel Methods? NeurIPS 2020; J. Stat. Mech. (2021) 124009. https://doi.org/10.1088/1742-5468/ac3a81
- Li, K.-C. (1991). Sliced Inverse Regression for Dimension Reduction. JASA 86(414):316–327. https://doi.org/10.1080/01621459.1991.10475035
- Radhakrishnan, A., Beaglehole, D., Pandit, P., Belkin, M. (2024). Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science 383(6690):1461–1467. https://doi.org/10.1126/science.adi5639
- Xia, Y., Tong, H., Li, W. K., Zhu, L.-X. (2002). An Adaptive Estimation of Dimension Reduction Space. JRSS-B 64(3):363–410. https://doi.org/10.1111/1467-9868.03411

**Tabular benchmarks and inductive bias**
- Arora, S., Du, S. S., Li, Z., Salakhutdinov, R., Wang, R., Yu, D. (2020). Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks. ICLR 2020. https://arxiv.org/abs/1910.01663
- Beyazit, E., Kozaczuk, J., Li, B., Wallace, V., Fadlallah, B. (2023). An Inductive Bias for Tabular Deep Learning. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html
- Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Desai, P. M., Salinas, D., Hutter, F. (2025). TabArena: A Living Benchmark for Machine Learning on Tabular Data. NeurIPS 2025 Datasets and Benchmarks. https://arxiv.org/abs/2506.16791
- Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021. https://arxiv.org/abs/2106.11959
- Gorishniy, Y., Rubachev, I., Babenko, A. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. NeurIPS 2022. https://arxiv.org/abs/2203.05556
- Gorishniy, Y., Kotelnikov, A., Babenko, A. (2025). TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. ICLR 2025. https://arxiv.org/abs/2410.24210
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022 Datasets and Benchmarks. https://arxiv.org/abs/2207.08815
- Hollmann, N., Müller, S., Eggensperger, K., Hutter, F. (2023). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. ICLR 2023. https://arxiv.org/abs/2207.01848
- Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. Nature 637(8045):319–326. https://doi.org/10.1038/s41586-024-08328-6
- Holzmüller, D., Grinsztajn, L., Steinwart, I. (2024). Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data. NeurIPS 2024. https://arxiv.org/abs/2407.04491
- Jeffares, A., Curth, A., van der Schaar, M. (2024). Deep Learning Through A Telescoping Lens. NeurIPS 2024. https://arxiv.org/abs/2411.00247
- Kadra, A., Lindauer, M., Hutter, F., Grabocka, J. (2021). Well-tuned Simple Nets Excel on Tabular Datasets. NeurIPS 2021. https://arxiv.org/abs/2106.11189
- McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Feuer, B., Hegde, C., Ramakrishnan, G., Goldblum, M., White, C. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2305.02997
- Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., Hutter, F. (2022). Transformers Can Do Bayesian Inference. ICLR 2022. https://arxiv.org/abs/2112.10510
- Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. ICML 2004, p. 78. https://doi.org/10.1145/1015330.1015435
- Probst, P., Boulesteix, A.-L., Bischl, B. (2019). Tunability: Importance of Hyperparameters of Machine Learning Algorithms. JMLR 20(53):1–32. https://arxiv.org/abs/1802.09596
- Wainberg, M., Alipanahi, B., Frey, B. J. (2016). Are Random Forests Truly the Best Classifiers? JMLR 17(110):1–5. https://jmlr.org/papers/v17/15-374.html
- Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., Zhan, D.-C. (2024). A Closer Look at Deep Learning Methods on Tabular Datasets. https://arxiv.org/abs/2407.00956
- Zhang, Q., Tan, Y. S., Tian, Q., Li, P. (2025). TabPFN: One Model to Rule Them All? https://arxiv.org/abs/2505.20003

**Explanation**
- Apley, D. W., Zhu, J. (2020). Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models. JRSS-B 82(4):1059–1086. https://doi.org/10.1111/rssb.12377
- Chang, C.-H., Tan, S., Lengerich, B., Goldenberg, A., Caruana, R. (2021). How Interpretable and Trustworthy are GAMs? KDD 2021. https://arxiv.org/abs/2006.06466
- D'Amour, A., et al. (2022). Underspecification Presents Challenges for Credibility in Modern Machine Learning. JMLR 23(226):1–61. https://arxiv.org/abs/2011.03395
- Fisher, A., Rudin, C., Dominici, F. (2019). All Models are Wrong, but Many are Useful. JMLR 20(177):1–81. https://arxiv.org/abs/1801.01489
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics 29(5). https://doi.org/10.1214/aos/1013203451
- Friedman, J. H., Popescu, B. E. (2008). Predictive learning via rule ensembles. Annals of Applied Statistics 2(3):916–954. https://doi.org/10.1214/07-AOAS148
- Ludwig, J., Mullainathan, S. (2024). Machine Learning as a Tool for Hypothesis Generation. Quarterly Journal of Economics 139(2):751–827. https://doi.org/10.1093/qje/qjad055
- Tsang, M., Cheng, D., Liu, Y. (2018). Detecting Statistical Interactions from Neural Network Weights. ICLR 2018. https://arxiv.org/abs/1705.04977
- Williamson, B. D., Gilbert, P. B., Simon, N. R., Carone, M. (2023). A General Framework for Inference on Algorithm-Agnostic Variable Importance. JASA 118(543):1645–1658. https://doi.org/10.1080/01621459.2021.2003200

**Missing values and measurement**
- Ayme, A., Boyer, C., Dieuleveut, A., Scornet, E. (2023). Naive imputation implicitly regularizes high-dimensional linear models. ICML 2023 (PMLR 202). https://arxiv.org/abs/2301.13585
- Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. Neural Computation 7(1):108–116. https://doi.org/10.1162/neco.1995.7.1.108
- Hutcheon, J. A., Chiolero, A., Hanley, J. A. (2010). Random measurement error and regression dilution bias. BMJ 340:c2289. https://doi.org/10.1136/bmj.c2289
- Josse, J., Chen, J. M., Prost, N., Scornet, E., Varoquaux, G. (2024). On the consistency of supervised learning with missing values. Statistical Papers. https://doi.org/10.1007/s00362-024-01550-4
- Le Morvan, M., Josse, J., Scornet, E., Varoquaux, G. (2021). What's a good imputation to predict with missing values? NeurIPS 2021. https://arxiv.org/abs/2106.00311
- Le Morvan, M., Varoquaux, G. (2024). Imputation for prediction: beware of diminishing returns. https://arxiv.org/abs/2407.19804
- Luijken, K., Groenwold, R. H. H., Van Calster, B., Steyerberg, E. W., van Smeden, M. (2019). Impact of predictor measurement heterogeneity across settings on the performance of prediction models. Statistics in Medicine 38:3444–3459. https://doi.org/10.1002/sim.8183
- Van Ness, M., Bosschieter, T. M., Halpin-Gregorio, R., Udell, M. (2023). The Missing Indicator Method: From Low to High Dimensions. KDD 2023, pp. 5004–5015. https://doi.org/10.1145/3580305.3599911

**Preprints, watched and never built on**
- Allerbo, O., Schön, T. B. (2026). A Rigorous, Tractable Measure of Model Complexity. https://arxiv.org/abs/2605.21167
- Beaglehole, D., Holzmüller, D., Radhakrishnan, A., Belkin, M. (2025). xRFM: Accurate, scalable, and interpretable feature learning models for tabular data. https://arxiv.org/abs/2508.10053
- Curth, A., Jeffares, A., van der Schaar, M. (2024). Why do Random Forests Work? Understanding Tree Ensembles as Self-Regularizing Adaptive Smoothers. https://arxiv.org/abs/2402.01502
- Kanoh, R. (2026). Double Descent in Gradient Boosting Decision Trees via Split-Candidate Scaling. https://arxiv.org/abs/2608.03111
- Rauniyar, S. (2025). Jacobian Aligned Random Forests. https://arxiv.org/abs/2512.08306
- Sergazinov, R., Wu, J., Yin, S.-A. (2025). Random at First, Fast at Last: NTK-Guided Fourier Pre-Processing for Tabular DL. https://arxiv.org/abs/2506.02406

**Cited through the theory source, not verified on their own:** Cohen et al. (2021) on the edge of stability, as summarized in Simon et al. (2026, §2.3). The registry entry cites Simon et al. for it. The edge-of-stability label is used only in the full-batch setting that summary describes.
