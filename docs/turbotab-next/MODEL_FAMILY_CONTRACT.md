# The model-family contract

**Status: DRAFT 2, for Nolan's review.** The orchestrator wrote draft 1 on 2026-10-08, acting as methods expert, and revised it the same day after two reviews: a methods review and an engine review. Every finding was checked against its source before it was acted on. "What changed after review" and "Review notes not taken" close the document. It carries Nolan's rulings and ideas of 2026-10-08, listed in §0. Nothing was run. The engine claims come from reading branch `turbotab-next` at b1830787. The theory claims come from the verified bibliography in §7 and from the theory source, Simon et al. (2026), read in full.

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
- **The theory source is itself a preprint** (arXiv 2604.21691, no venue). It frames the five strands below. Every claim the contract rests on cites its primary source as well.

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
- **The app checks the science on the user's own table.** Almost all of the theory below was worked out on idealized inputs: Gaussian, continuous and high-dimensional, and mostly for neural networks and kernels. Tabular data are skewed, small, correlated and partly categorical, and today's flexible families are trees. The literature search behind §7 found no study that applies this learning theory to tabular data on purpose. So TurboTab cannot inherit the theory's validity. It checks what it can on its own encoded matrices, and it never carries a deep-learning result over to trees or tabular data without saying so. That is the opening the north star names.
- **Every phenomenon gets its name.** A card says what happened in plain words, and the technical name rides along quietly with its source. In Nolan's words, the researcher thinks "huh, so that's what I call that problem when I'm discussing it with colleagues."

### The rulings this contract carries (2026-10-08)

| # | Ruling or idea | Where it lands |
|---|---|---|
| 1 | The pre-fit ranking happens live at model selection, after the shared steps are settled. Each family is assessed on the input it would actually receive. Everything is outcome-blind. Nothing is ranked by a score before Fit; the corrected comparison (BBC-CV) decides after Fit. | C4; questions 1 and 4; MC-3 to MC-6 |
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
- **Calm.** Each new element is one quiet line with a plain sentence. Numbers sit under "More angles". A technical name appears on point or focus, never as a second label (calm/FOUNDATION §2). Every new view gets a purpose-registry entry (BLUEPRINT §11.2).

### How the theory maps onto the app

Simon et al. (2026, a preprint) argue that a scientific theory of deep learning is forming along five strands. Each has a home here, and each home says how far the strand reaches on tabular data.

| Strand (Simon et al. 2026) | In TurboTab |
|---|---|
| §2.1 Solvable settings: deep linear networks, kernel regression, multi-index models | Reference tests with a known answer for neural families (C13). The multi-index results motivate §2.2's readout for neural and kernel families only (v2.x); for trees, §2.2 rests on the empirical readout and on Ludwig & Mullainathan (2024). |
| §2.2 Limits: lazy versus rich learning | A neural family declares its parameterization and output multiplier; its fit reports the regime it measured (C1, C8) |
| §2.3 Simple laws: edge of stability, the neural feature ansatz | Training diagnostics for trained networks (C8), each citing its primary source (Cohen et al. 2021; Radhakrishnan et al. 2024) |
| §2.4 Hyperparameters can be disentangled | Structural settings are identity, searched ones are tuning, and which knobs matter is measured (C1, C6) |
| §2.5 Universal behavior across architectures | A motivating analogy only. Curves on shared axes show where families agree on a shape, and the card says that agreement is not evidence the shape is true (C10). |

---

## 1 · The contract clauses

### The contract on one page

| Clause | What a family declares or satisfies | Enforced by |
|---|---|---|
| **C1** Identity | Key, label, library, defaults version. Neural: learning rule, initialization, parameterization and output multiplier, seed policy. In-context: the prior's checkpoint. | `register_family`; provenance; replay |
| **C2** Tasks, purposes, inference eligibility | Tasks; purposes; whether it predicts; what kind of table it can give under inference | `register_family`; refusals; reference tests |
| **C3** Inputs and recipe | RECIPES §2.4 recipe slots; blank routing; the transformations it is invariant to | `register_family`; invariance probes (C13) |
| **C4** Assess on the actual input | The input-profile fields its `assess` reads, on the recipe it would actually receive; a sourced sample-efficiency prior | The shelf stage; an outcome-permutation test |
| **C5** Inductive bias in two registers | The plain statement; its quiet names with sources; the shape its curves should take | Word budgets; the citation registry; a curve-shape test |
| **C6** Tuning | `TuningDecl`; structural versus searched settings; transfer; cost | RECIPES T1–T17; `register_family` |
| **C7** Complexity controls | Each knob that sets effective complexity, its direction, and a formula in the estimator's own parameterization where one exists | A formula test per knob |
| **C8** Training diagnostics | The checks it reports about its own fit, as keys into a diagnostics registry | A fixture where each fires and one where it stays silent |
| **C9** Calibration | Its output scale and the recalibration it supports | The generic calibration path; a fixture |
| **C10** Explanation paths | A raw score on a declared scale for each task; its attribution and architecture views | Curves for every predicting family, by task; reference tests |
| **C11** Bootstrap and validation soundness | Whether Harrell's bootstrap is sound; whether it is flexible; its internal splits | An equivalence test against fresh data, run as a scheduled heavy run |
| **C12** Replay | Seeds, threads, library versions, tolerance | The export replay test, for every family |
| **C13** Reference tests | An independent implementation; invariance probes; solvable settings | The fold-in gate |
| **C14** The fold-in gate | Passes every item of the checklist | One parametrized acceptance test over the registry, plus expert review |

### How it is declared (`models/base.py`, new here)

These members extend `models/base.py:FamilyBase`. RECIPES §2.4's `recipe`, `tuning` and `defaults_version` arrive with RECIPES RT-2, not here, so this package can land first (§5). Existing duck-typed members (`purposes`, `predicts`, `ordered_levels`, `bootstrap_optimism`, `linear_in_values`, `pools_imputations`, `preprocess`, `build_for`, `describe_step`, `inference`, `inference_matrix`) become declared members of the protocol, so `register_family` can check them. The five methods a family may add are declared as None where it adds nothing, and read that way, never with `getattr` or `hasattr`. Because `inference` is the family's table method, the C2 declaration is named `inference_decl` (amended while building MC-1).

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
    parameterization: str = ""   # trained_network: "standard" · "NTK" · "muP", and the output
                                 # multiplier; no regime is implied by it (C8 measures the regime)
    seed_policy: str = ""    # how its seeds derive from the split's seed (RECIPES §4.7)
    prior: str = ""          # pretrained_prior: checkpoint name and SHA-256

@dataclass(frozen=True)
class InferenceDecl:         # C2
    table: Literal["intervals", "shrunk_no_intervals", "description_only"]
    intervals: tuple[str, ...] = ()   # "HC3", "CR2", "sandwich", "Satterthwaite", "Taylor"
    design_based: bool = False        # replaces the signature check in survey.has_design_estimator
    product_terms: bool = False       # replaces methods/interaction.py:SUPPORTED
    matrix_table: bool = False        # replaces stages/effects.py:SEQUENCE_FAMILIES
    default_for: tuple[str, ...] = () # tasks it is the default inference family for
                                      # (replaces stages/scales.py:FAMILY_FOR)

@dataclass(frozen=True)
class Prior:                 # C4: a sample-efficiency statement, never a forecast
    says: str                # ≤ 22 words
    kind: Literal["bound", "empirical", "convention"]
    source: Source | None

@dataclass(frozen=True)
class Knob:                  # C7
    setting: str             # the estimator's parameter, or "time" for early stopping
    more_means: Literal["simpler", "more_flexible"]
    formula: str = ""        # a key into models/formulas.py, whose callable takes the estimator's
                             # own parameters (alpha = nλ for ridge), when a formula exists
    source: Source | None = None

class FamilyBase:
    identity: Identity
    purposes: tuple[Purpose, ...]
    predicts: bool
    flexible: bool                          # declared, no longer derived from bootstrap_optimism
    bootstrap_optimism: bool | None         # None: not applicable, it makes no predictions (C11)
    inference_decl: InferenceDecl | None = None   # None: not offered as an inference table
    reads: tuple[str, ...] = ()             # InputProfile fields its assess reads (C4)
    sample_efficiency: tuple[Prior, ...] = ()
    same_kind_as: tuple[str, float] | None = None   # (family key, rank offset): XGBoost beside
                                                    # boosted trees, ("boosted_trees", -1.0). The
                                                    # shelf reads that family's assessment, its
                                                    # score plus the offset, at least 0.5 but never
                                                    # above it (RECIPES §2.2), not this family's own
    bias_terms: tuple[Named, ...] = ()      # C5
    invariances: tuple[str, ...] = ()       # "linear_maps", "rotation_after_scaling",
                                            # "monotone_per_column", "column_scale" (C3)
    curve_shape: Literal["straight", "piecewise_constant", "any"] = "any"
    complexity: tuple[Knob, ...] = ()       # C7
    diagnostics: tuple[str, ...] = ()       # keys into the diagnostics registry (C8)
    output: Literal["value", "margin", "probability"] = "value"   # C9
    updating: tuple[str, ...] = ()          # recalibration paths it supports (C9)
    raw_scale: Mapping[str, str] = {}       # C10, per task: "value", "margin", "latent",
                                            # "log_hazard", or "not drawn" with its reason
    attribution: Literal["linear", "trees", "none"] = "none"      # C10
    architecture: tuple[str, ...] = ()      # "equation", "trees", "shrinkage", "spectrum"
    review_lenses: tuple[str, ...] = ()     # replaces reference/catalog.py:FAMILY_LENSES
    solvable: tuple[str, ...] = ()          # keys into the solvable-settings harness (C13)
    replay_tolerance: float = 1e-12         # C12
    sources: tuple[Source, ...] = ()        # its primary sources, as a method contract has
    cost_model: Literal["cells", "cross_product"] = "cells"   # C6: how one fit's time grows
    card_label: str = ""                    # the card's plain name ("Mixed model"); `label` is the methods register's
    preprocess = build_for = describe_step = inference = inference_matrix = None  # or methods

    def methods_label(self, task) -> str: ...   # what the methods text calls it ("linear
                                                # regression"); describe()'s label names the step
```

`models/base.py:FamilyInfo` and `models/artifacts.py:ShelfFamily` gain the user-facing parts of these: the bias terms, invariances, inference table kind, `flexible`, `bootstrap_optimism`, the profile fields read (`reads`), the measures, and a `terms` list beside `concerns` (C4). `concerns` stays `list[str]`. The measures arrive with MC-4's `Assessment.measures`.

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
- **Nothing switches on a key or a class name.** Code reads declarations. Today at least eighteen places switch on family keys or estimator class names; §3.3 lists them and the declarations that replace them.

**Why.**
- Simon et al. (2026, §2; a preprint) list a deep learning system's components as architecture, data, task and learning rule, where the learning rule includes the initialization and the optimization settings. The learning rule is part of what the model *is*.
- The output scale moves a network toward lazy (kernel-like) or rich (feature-learning) training (Chizat et al. 2019). That is a statement about scaling limits, so the regime is measured after the fit (C8), not declared.
- Initialization changes which function is learned. In a deep linear network with more columns than the rank of X, the part of the solution outside the row space of X comes only from the initialization, so gradient flow reaches the minimum-norm solution only as the initialization scale goes to zero (the derivation in C13). For orthogonally decomposable linear networks, the norm that gradient flow minimizes "interpolates between weighted ℓ1 and ℓ2 norms" (Yun et al. 2021, abstract). So initialization and parameterization belong to identity, not to tuning.
- An in-context model has no fitted penalty; its inductive bias is the prior it was trained on (Hollmann et al. 2023; Müller et al. 2022).

**How the engine enforces it.**
- `register_family` refuses a `trained_network` whose learning rule, initialization, parameterization or seed policy is empty, and a `pretrained_prior` without a checkpoint hash.
- The identity enters provenance and the version key. Replay (C12) refits from it.
- **The no-switch test reads the syntax tree, not literals.** A literal scan cannot reach zero: `"linear"` is also an exposure form (`methods/exposure_form.py`), the MI substantive model (`methods/missing.py:SUBSTANTIVE`) and a causal learner (`decisions.py:CausalLearner`); `"elastic_net"` is a selection method (`models/variable_selection.py:PREDICTION_METHODS`). The test flags:
  - an equality or membership test against `family.key` or a value of `state.models`;
  - an `isinstance` check or a `__name__` comparison against an estimator class.

  It allows exits that recommend a family by key (the lever exit in `stages/explore.py`, the exit to `mixed` in `models/inference.py`). Its allowlist starts with §3.3's eighteen places and must reach zero (MC-2).
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

  A family with no declaration is not offered as an inference table. Its flags for design-based estimation, product terms, the matrix table and the default family per task replace the four key lists that decide these today (§3.3).

**Why.**
- Under inference the coefficient table is the locked primary's estimate (BLUEPRINT §12, ruling 3). Only a family whose intervals were checked against an independent reference may produce one (V2_DEFINITION_OF_DONE §2).
- Conventional intervals on coefficients penalized by cross-validation are "impossible by construction" (RECIPES §5). The declaration makes that a property of the family, not a rule each stage must remember.

**How the engine enforces it.**
- `register_family` checks the purposes vocabulary and the interval kinds against a known set.
- `register_family` refuses a second default for a task (`default_for`, which replaced the one-to-one `stages/scales.py:FAMILY_FOR`), and a default whose table has no intervals: the scales stage refits the default family to correct a coefficient.
- Every interval kind a family declares needs a reference test (C13).
- `models/survey.py:has_design_estimator` reads `design_based` instead of inspecting the `inference` signature for a `survey` parameter.
- **Today:** `purposes` and `predicts` are read with `getattr` (`models/base.py:info`), and `register_family` checks neither.

### C3 · Inputs and recipe slots

**What.**
- **The recipe** is RECIPES §2.4 unchanged: five slots (missing, scale, encoding, transform, outliers), each option with its plain label, quiet term, consequence, customary and sound labels and leash rung. Steps declare `reads(spec)` and `passes_blanks`, so blank routing is computed (RECIPES §2.3). `needs_scaling` and `handles_missing` become derived.
- **Invariances (new here).** A family declares the input transformations its predictions do not change under. There are four, and a family declares each that holds:
  - `linear_maps`: any invertible linear map of the inputs, applied where the family receives them. The fit depends only on the span of the inputs and an intercept, so predictions do not change. Unpenalized least squares, logistic regression (Firth's included), Huber's M-estimator (`RLM`, whose scale is computed from residuals), proportional odds, Cox, GEE, and the mixed model's fixed effects with random intercepts.
  - `rotation_after_scaling`: an orthogonal rotation applied after the family's own scaling step. Ridge and logistic ridge (the L2 penalty is rotation-invariant). An MLP only in distribution over seeds, with a rotation-invariant initialization and plain gradient descent (Ng 2004's setting); never for Adam or AdamW unless the probe shows it.
  - `monotone_per_column`: a monotone change of one column, on the training rows (trees; split thresholds are midpoints, so rows between them can move).
  - `column_scale`: rescaling one column before the family's steps. Every family that standardizes its inputs, and trees.
  - `featurewise` has no predictions, so its invariances are "not applicable".
- **For a neural family, the recipe adds a numeric-embedding slot** (§4): none, piecewise-linear or periodic (Gorishniy et al. 2022), with rank or quantile scaling as an option (Beyazit et al. 2023).

**Why.**
- **Rotation is a real dividing line.** Random rotations of the inputs hurt trees and FT-Transformer but not ResNet-style networks, and reverse their ranking (Grinsztajn et al. 2022). Any rotation-invariant algorithm has a worst-case sample complexity that grows at least linearly in the number of irrelevant features, while L1-regularized logistic regression's grows only logarithmically (Ng 2004). Declaring the invariance tells the shelf which input measures matter for a family (C4), and gives the reference test something to check (C13).
- **Blanks.** For prediction, filling blanks before learning is consistent when missingness is not informative. Trees that route blanks themselves handle informative missingness too (Josse et al. 2024). After imputation, the best regression function is generally discontinuous and hard for smooth models to learn (Le Morvan et al. 2021). Missing indicators help when missingness is informative, and overfit when many uninformative ones meet few rows (Van Ness et al. 2023). Better imputation may buy little on real outcomes once indicators are present (Le Morvan & Varoquaux 2024, a preprint).
- **Embeddings for neural families.** Piecewise-linear and periodic embeddings of numeric columns gave large gains in tabular MLPs (Gorishniy et al. 2022). The explicit spectral argument for why they help is Beyazit et al. (2023): tabular target functions are irregular, and transformations such as ranking reduce that irregularity.

**How the engine enforces it.**
- `register_family` runs RECIPES §2.4's checks, including that a family may declare native blanks only if its estimator fits a three-row frame with one blank.
- **Each declared invariance gets a two-sided probe in C13.** The family must keep its predictions under what it declares, and must change them under the next larger transformation it does not declare. So the declaration cannot be wrong in either direction.
- **Today:** only `needs_scaling` and `handles_missing` exist, and trees' `handles_missing` is unreachable (RECIPES F1).

### C4 · Assess on the actual input

**What.**

*When the shelf ranks (ruling 1).*
- **It waits only for the Decide answers that come before the families question.** Those are who is kept, the missing-values answer and categories from the earlier stages; the energy model and the forms where they apply (CROSSWALK §5, Decide 4–5); scales, batch and the omics normalization where they apply (Decide 6–8, ruled ahead of the families on 2026-10-08; CROSSWALK disagreement 21); and, under Predict, the selection question when it is asked (Decide 19, which the Predict order puts before the families). Until they are in, the shelf shows "Waiting for: [question]", with a link (CROSSWALK disagreement 20).
- **Everything else is ranked on the defaults in force.** The levers, the validation scheme, the in-fold omics steps, the recipes, the trees' "Try both" and tuning all sit in the Models Confirm sweep, which runs last, after the families are chosen. The shelf ranks on the defaults now set and says so once: "Ranked on the defaults now set." When the sweep or a later answer changes one, the shelf re-ranks. That is legal because the shelf reads no outcome relation and no score: re-ranking is a recomputation, not a choice made after seeing results.
- **Scales, batch and the omics normalization** change width and spectrum more than anything else, so they come before the families question (DoD amendment of 2026-10-08, "Order"; question 4; CROSSWALK §5, Decide 6–8). Where they apply, the shelf waits for them like the other Decide answers above, and shows "Waiting for: [question]"; where none applies, nothing waits.
- **Readiness is a predicate, not `Stage.requires`.** `graph.py:_compute_keys` blocks a stage while a required slot is None. Energy adjustment, batch and scales stay None wherever they do not apply, so `requires` would block the shelf forever on a table without them. Readiness is built from P0.4's stage registry, which knows which slots apply to this table.

*What `assess` reads: the input it would actually receive.* `Situation` gains `inputs: Mapping[family key, InputProfile] | None = None`. The default keeps every test that builds a `Situation` directly working. The profile is computed for every registered family that can model the task, through RECIPES §2.4's `family_spec`, on:
- **an unselected family:** its stated default recipe and tuning (RECIPES §2.2);
- **a selected family:** its current `set_recipe` and `set_tuning` slots. The shelf reads those slots, so a recipe edit re-ranks it, and the stage stays a function of the decision log;
- **a `choose` slot** (the trees' default "Try both"): one profile per option. The card shows both options' measures. The score is the lower of the two assessments, because the shelf cannot know which option the folds will pick.

*Where the profile stops.* Each step declares `reads_outcome` (new here): `"none"`, `"counts"` (O1) or `"values"` (O3).
- **O1 steps continue.** `methods/levers.py:RuleSplines` (its k comes from the effective size) and `methods/levers.py:ImbalanceCorrected` (class counts) read only the outcome's counts. The profile feeds them those counts, which the permutation test keeps fixed, and continues past them.
- **The profile stops at the first O3 step.** These are `methods/omics.py:UnivariateScreen`, `models/variable_selection.py:Selector`, `methods/levers.py:InnerCVForms`, and, under inference, multiple imputation by chained equations and SMC-FCS, whose imputation model includes "the outcome and total energy" (`methods/imputation.py`, module docstring; BLUEPRINT §12 ruling 4).
  - The spectrum, concentration and condition number are computed on the matrix entering that step, with the family's scaler fitted there, and labeled "before selection" or "before the fill".
  - Widths after the step come from its declared size rule, and the card says "by rule".
  - The family's own scaler runs after Explore's steps and the screen (`models/pipeline.py:family_steps`), so this is the only way to describe a scaled input when an O3 step comes first.
- **Blanks under inference.** The X-side measures are computed on one fill that ignores the outcome: the prediction path's in-fold fill, which "imputes in-fold without the outcome" (`methods/imputation.py`). It is used only for the profile and labeled so. Blank counts come from the blank mask, never from a fill. Under prediction the in-fold fill is already outcome-free, so nothing changes.

| Profile field | In plain words | Why it is there | Read by |
|---|---|---|---|
| rows, units, effective size | Training rows (prediction) or analyzed rows (inference); units; Kish's effective size under unequal survey weights; PSUs under a cluster design | Every criterion. p/n uses the effective size wherever weights or clusters apply (Kish 1965). | every family |
| columns | Columns after encoding: indicators, the energy step, scale scores, blanks as a level, every level of a category | p and p/n are the outcome-blind core (Dobriban & Wager 2018) | every family |
| parameters | Exact parameter count, and the trunk's reference-coded count | Riley's minimum reads the trunk's reference-coded count, and the card says so | regression families |
| blanks routed, blanks filled | Blank cells the family takes as blanks, and cells it is given filled, from the blank mask | Native blanks versus fill (Josse et al. 2024) | tree families; every family's caution |
| indicator columns | Missing-indicator columns added | Many indicators with few rows overfit (Van Ness et al. 2023) | every family |
| condition number | Belsley's scaled condition number, reusing `models/linear.py:collinearity_concern` | Collinearity, before the fit instead of only after it | linear families |
| spectrum | The top eigenvalues of the family's own input covariance, scaled as that family scales it | With no particular alignment between outcome and directions, the input spectrum and p/n govern ridge's limiting risk (Dobriban & Wager 2018). That assumption is checked after the fit (§2.1). | rotation-invariant families |
| concentration | The participation ratio (Σλ)²/Σλ²: how many unrelated columns would vary as much. Quiet beside it: r₀ = tr(Σ)/λ₁, total spread relative to the largest direction, with Bartlett et al.'s (2020) meaning | The plain number describes how concentrated the spread is. r₀ does not: one strong energy factor makes r₀ small while the rest spreads over hundreds of directions. | rotation-invariant families |
| smallest eigenvalue; skewness and kurtosis spread | How irregular the columns are | Boosted trees handled "skewed or heavy-tailed feature distributions" better than neural nets (McElfresh et al. 2023, abstract) | trees; neural and kernel families |
| outlying share | Rows far from the centroid in principal-component space | A convention. Jeffares et al. (2024) is the lead; its tabular case study may concern test-time inputs, and it is checked before this field appears on a card. | neural and kernel families |
| measured, by rule | Which widths were measured and which come from a step's declared size rule | Some steps read the outcome, so their width can only be a rule | the card |

- **Rows.** The shelf counts the training rows, so its number matches the basis line "Ranked for N training rows". RECIPES §4.2's n_plan (one outer fold's effective size) stays the tuning plan's size and is not the shelf's row count (§3.4).
- **Cost.** The spectrum comes from the smaller of the covariance and the Gram matrix, or from a randomized top-k decomposition when both are large. It uses the sampling caps of `models/cost.py:sample_shape`, and the card says when it did.
- **Trees read width, blanks and the irregularity measures, not the spectrum.** Their splits follow the columns, so the eigen-directions of their input are not what they see (Grinsztajn et al. 2022).

*What `assess` returns.*
- `Assessment` gains `measures: Mapping[str, float]` and `reads`, the fields the family looked at. `concerns` stays `list[str]`, and a parallel `terms: list[Named | None]` carries each concern's quiet name and source. The engine functions that prepend concerns (`models/base.py:rank`, `models/selection.py:shelf_order`, `models/survey.py:population_shelf`) and the frontend's dedupe (`Shelf.tsx`) keep working unchanged.
- **One shared "Looked at" line, then differences.** The card states once what every family saw ("Looked at: 412 columns for 2,400 training rows; they vary like about 40 unrelated columns"). A family line adds only where its input differs ("Boosted trees: 31 blanks kept as blanks").
- The score stays the family's own judgment of fit to the situation. Every change to it names a profile field and a source, or says "convention". As the engine already says, "no validated rule predicts which family will perform best" (the prediction review, quoted above `models/selection.py:FLEXIBLE_REASON`).
- **XGBoost** reads boosted trees' assessment through its declared `same_kind_as`, not by key (RECIPES §2.2: "Boosted trees' score less 1.0").

*The sample-efficiency prior.* Each family declares sourced statements of how data-hungry it is, each marked bound, empirical or convention. They appear only under "More angles", as a caution. A prior is never a forecast and never moves a score by itself. Examples:
- **Lasso-type families:** with many columns that may not matter, an L1 penalty can need far fewer rows, in the worst case (a bound, for L1 logistic regression; Ng 2004).
- **Ridge and neural nets:** in the worst case, rows needed grow at least linearly in the number of irrelevant columns (a lower bound; Ng 2004).
- **Boosted trees:** they handled skewed or heavy-tailed columns better than neural nets (empirical; McElfresh et al. 2023).
- **NTK-type kernel machines:** a small edge over random forests on small classification tables (empirical; the margin is small; Arora et al. 2020, a protocol Wainberg et al. 2016 criticized).
- **Pretrained in-context models:** dominant up to 10,000 rows and 500 features (empirical, at the limits of their pretraining; Hollmann et al. 2025).

*The parameterization (trained networks).* A network declares its parameterization and output multiplier in its identity (C1), and the shelf states them. It implies no regime. Lazy versus rich is a statement about scaling limits (Chizat et al. 2019), and at tabular widths no parameterization guarantees either one. PyTorch's default "standard" parameterization has no regime value at all. After the fit, the fit reports the regime it measured (C8).

*Outcome-blindness.*
- The profile builder takes no outcome argument, so no predictor-by-outcome quantity (view class O3) or score (O4) can enter it.
- The outcome's own counts (O1) still feed Riley's minimum, EPV, Whitehead's effective size and the O1 steps above. The library-size check (`stages/modeling.py:_assay_concern`) is a design count (O2), shown as a separate shelf line, not part of any `assess`. Question 1 asks Nolan to confirm this reading of "outcome-blind".

**Why.** Ruling 1, with:
- the outcome-blind meta-features that carry signal about which family fits: width, p/n and irregularity (McElfresh et al. 2023; Ye et al. 2024, a preprint). McElfresh et al. also used some that read the outcome; those are left out;
- the spectrum and p/n, which govern ridge's limiting risk under no particular alignment (Dobriban & Wager 2018);
- the irregularity caution (McElfresh et al. 2023).

Whether a column is uninformative depends on the outcome, so it cannot be checked before Fit. Only width, p/n and redundancy can (Grinsztajn et al. 2022; Ng 2004).

**How the engine enforces it.**
- **The shelf stage** (`stages/modeling.py:shelf_stage`) depends on a new `trunk` stage (MC-3). Its reads widen to the recipe and tuning slots, energy adjustment, batch, scales, column units, selection, levers and the validation scheme. Its readiness is P0.4's predicate above.
- **An outcome-permutation test.** Permuting the outcome across rows keeps its counts but destroys every predictor-by-outcome relation. The ranking, scores, fits, measures and concerns must stay identical, byte for byte. Its fixtures include an inference track with multiple imputation (the profile must not move when the draws do) and an O1 step.
- **`register_family` checks** that every field in `reads` exists in `InputProfile`, and every step declares `reads_outcome`.
- **Today:**
  - `Situation` holds only scalars (`models/base.py:Situation`). No `assess` reads collinearity, the spectrum, blanks or the width after encoding. Belsley's number runs only after the fit (`stages/modeling.py:fit_stage`, for `linear` only).
  - The shelf does not read energy adjustment, batch, scales, column units, selection or recipes, so it neither waits for them nor re-ranks when they change (`stages/__init__.py:build_graph`, the `shelf` stage).

### C5 · The inductive-bias statement, in two registers

**What.**
- **The plain statement** stays `inductive_bias`, at most 20 words (`models/base.py:INDUCTIVE_BIAS_WORDS`). Example: "Straight-line effects all shrunk toward zero together; correlated predictors share weight; none is dropped."
- **The quiet names** are `bias_terms`: each a `Named(plain, known_as, source)`. Examples:
  - ridge: "Known as L2 shrinkage" (ESL §3.4.1);
  - random forest: "Known as randomization as regularization" (Mentch & Zhou 2020);
  - an MLP: "Known as spectral bias" (Rahaman et al. 2019; Beyazit et al. 2023).
- **The curve shape** is `curve_shape`: `straight` for families linear in their inputs, `piecewise_constant` for trees, otherwise `any`.
- **The card shows the plain statement.** The quiet name appears on point or focus, in the quiet style of RECIPES §6.5, never as a second label (calm/FOUNDATION §2). Each element carries one quiet name at most.

**Why.**
- Ruling 4.
- Different interpretable models can learn different, even contradictory, shapes for the same predictor while being equally accurate, because "inductive bias plays a crucial role in what interpretable models learn" (Chang et al. 2021).
- Drawing several families' fits of the same toy functions side by side shows each family's bias at a glance: ridge fits only lines, boosting is piecewise-constant (Hollmann et al. 2025, Fig. 3a).

**How the engine enforces it.**
- Word budgets: plain ≤ 22 words, `known_as` ≤ 6 words.
- Every `known_as` names a phenomenon or term in the registry of §2.4, or a term with its own source. Every source key resolves in the citation registry (SIZING X4), whose entries are verified. Until X4 lands, the test checks §7's list.
- **The curve-shape test** (C13) checks only what a grid can tell apart:
  - `straight`: on a fixture with no forms and no spline lever, the curve's values are exactly collinear;
  - `piecewise_constant`: on a fine grid, predictions are constant between consecutive split thresholds.

  "Smooth" is not tested: on a 20-interval ALE grid, a 500-tree forest looks smooth too.
- **Today:** the plain statement exists. `models/base.py:Assessment.concerns` are plain strings, and the only quiet terms are settings (RECIPES §6.5) and `teaching/__init__.py:TeachingTerm`.

### C6 · Tuning

**What.**
- **RECIPES §4.1's `TuningDecl`, unchanged,** with one field added. Kind (none, path, search), searched dimensions, settings by hand, fixed settings, standard settings as candidate 0, early stopping, out-of-bag scoring, space version. The plan, nesting and replay follow RECIPES §4.2–4.7.
- **New here: structural versus searched.** `TuningDecl.structural` names the settings that are part of identity (C1): the architecture, the parameterization and output multiplier, the loss, the booster type. Only searched settings may be dimensions.
- **New here: tunability.** Each dimension may cite a measured tunability, meaning how much tuning it gained on benchmark data (Probst et al. 2019). The tuning line shows the one or two knobs that matter, not all of them.
- **New here: transfer and the validated range.** A dimension declares whether its best value is expected to transfer across sizes, with a source. Standard settings declare the row range they were validated on. Defaults meta-tuned on 1,000 to 500,000 rows (Holzmüller et al. 2024) are labeled unvalidated below 1,000.
- **New here: a cost model.** How one fit's seconds grow with rows, columns and the budget (trees, rounds, epochs), so `models/cost.py:estimate_fits` can scale a timing (RECIPES RT-8). It replaces `models/cost.py:fit_cost`'s class check (§3.3).

**Why.**
- Simon et al. (2026, §2.4; a preprint) argue that hyperparameters can be disentangled. The primary result: width-dependent factors separate from scale-free coefficients, so some optima stay stable across sizes (Yang et al. 2021). For small tabular networks, which are cheap to tune directly, that transfer gives little leverage, and the search found no tabular muP study.
- Tunability can be measured, so "which knobs are worth turning" becomes a quantity (Probst et al. 2019). That is the north star in one line.
- Searching a cocktail of regularizers let plain MLPs win on 40 datasets (Kadra et al. 2021). That win did not hold in larger later benchmarks (McElfresh et al. 2023; Erickson et al. 2025).
- With the penalty tuned, the risk curve can be monotone, so tuned users rarely see double descent (Nakkiran et al. 2021).

**How the engine enforces it.**
- RECIPES T1–T17.
- `register_family` refuses a setting listed both in `structural` and as a dimension, and a dimension without a label, quiet term, scale or source.
- **Today:** none of `TuningDecl` is built. The elastic nets tune by scikit-learn's CV estimators (RECIPES F4, F5), and boosted trees are not tuned (RECIPES F2).

### C7 · Complexity controls

**What.** Each family declares the knobs that set its effective complexity (`Knob`): which direction makes it simpler, and a formula where one exists. A formula is declared in the estimator's own parameterization and held as a callable in `models/formulas.py`. The tuning record stores λ per row (RECIPES §4.1), and the callable converts. These place families on one shrinkage axis in the tapestry (§2.3).

| Family | Knob | Formula or equivalence | Source |
|---|---|---|---|
| Ridge | penalty λ per row (scikit-learn's `alpha` = nλ) | Direction j, with singular value dⱼ of the n × p scaled matrix, is shrunk by dⱼ²/(dⱼ² + nλ); effective degrees of freedom df(λ) = Σⱼ dⱼ²/(dⱼ² + nλ), from 0 to rank(Z) | ESL §3.4.1, eqs. 3.47 and 3.50, with λ rescaled per row |
| Logistic ridge | C = 1/(nλ) | The penalty is (nλ/2)‖w‖² on the summed log-loss; the factors use the weighted matrix at the fitted probabilities, labeled an approximation (a convention) | scikit-learn's objective; ESL §4.4 |
| Elastic net | λ and the mix | The ridge part as above; the lasso part drops columns (the path view) | ESL §3.4.1 for the ridge part |
| Linear least squares trained by gradient flow from zero (v2.x MLP harness only) | training time t | The exact path β(t) = (XᵀX)⁺(I − exp(−tXᵀX/n))Xᵀy. With t = 1/λ its risk is at most about 1.69 times ridge's (Theorem 1). This holds for linear least squares only; a network gets no position on this line. | Ali et al. 2019 |
| Random forest | columns tried per split (mtry) | mtry acts as the penalty knob | Mentch & Zhou 2020 |
| Random forest | leaf size; trees | A forest is smoother than its trees and adapts its smoothing at test time; an "effective smoothing" number | Curth et al. 2024 (preprint) |
| Boosted trees, XGBoost | rounds, learning rate, leaf size, L2 pull | No sourced closed form; double descent may be named only along one named capacity axis | Curth et al. 2023 |
| A linear family after a mean fill and centering | the fill itself | If blanks are missing completely at random, in a high-dimensional linear model, the fill acts like ridge | Ayme et al. 2023 |
| A differentiable family trained with added input noise (v2.x, an MLP option) | the noise level | To first order, for smooth models, training with added noise equals a Tikhonov penalty. It says nothing about measurement error already in the data, and nothing about trees. | Bishop 1995 |

A single complexity axis shared by all families has been proposed (Allerbo & Schön 2026, preprint). It is a lead to test, not a foundation.

**Why.**
- Ruling 5. The user should see that a penalty, a forest's random column choice and a mean fill can all pull the fit in the same way, each under its stated conditions.
- Theory sometimes gives the exchange rate between them (Ali et al. 2019 for linear least squares; Mentch & Zhou 2020; Patil & Du 2023 for subsampling and ridge).
- The arXiv abstract of Ali et al. reads "no less than 1.69", which reverses the bound. The registry quotes Theorem 1 of the AISTATS version, never the abstract.

**How the engine enforces it.**
- A declared formula has a test that calls its callable with the recorded λ. For example, ridge's df(λ) must equal the trace of the hat matrix built from scikit-learn's `alpha` = nλ on a fixture, to 1e-10.
- `register_family` refuses a `Knob.formula` key with no callable in `models/formulas.py`.
- A knob with no formula says so on its card line, instead of borrowing one.

### C8 · Training diagnostics

**What.** A family declares the checks it reports about its own fit, as keys into a diagnostics registry (`models/diagnostics.py`, new; MC-18). Each check is a label or a disclosure after the fit (O4), never a change to the plan.

| Family kind | Diagnostics |
|---|---|
| Linear | Separation (Firth), collinearity, the HC3 residual-spread check, Cook's influence (today) |
| Mixed, GEE, Cox | Boundary or singular fits, convergence, proportional hazards (today) |
| Penalized | The penalty at a grid edge; the calibration slope per fold (RECIPES §4.6) |
| Tree ensembles | Early-stopping round (XGBoost, boosted trees); out-of-bag error (forest) |
| Trained networks (§4) | **Loss curve and convergence.** **Lazy or rich, as measured:** the first layer's relative weight change from its initialization, against a threshold that scales as 1/√width. **Edge of stability:** under full-batch gradient descent only, the sharpness (the largest Hessian eigenvalue) against 2/η. **Neural feature ansatz:** how closely the first layer's Gram matrix matches the average gradient outer product. |

**Why.**
- In the lazy regime the weights and hidden representations change only negligibly while the loss drops; in the rich regime they reorganize (Chizat et al. 2019; Simon et al. 2026, §2.2). Even in a fixed parameterization the relative change shrinks with width, so a fixed threshold would mix up width and regime. The threshold scales with width, and it is a convention.
- Under full-batch gradient descent, "the maximum eigenvalue of the training loss Hessian hovers just above the numerical value 2 / (step size)": the edge of stability (Cohen et al. 2021). The diagnostic never runs for minibatch or adaptive training.
- A trained layer's Gram matrix is roughly proportional to the average gradient outer product (Radhakrishnan et al. 2024). Simon et al. (2026, §2.3) call the rule heuristic and inexact but often strikingly accurate.

**How the engine enforces it.**
- Each diagnostic has a fixture where it fires and one where it stays silent, as a noticing does (SIZING T2).
- Warnings caught during a fit already become concerns (`stages/modeling.py:_concerns`). A declared diagnostic must not depend on a warning's wording.
- The diagnostics registry gathers what today is spread across `stages/modeling.py:fit_stage`, `stages/effects.py:_Run.diagnostics` and the warnings in `stages/modeling.py:_concerns`, and adds the forest's out-of-bag and XGBoost's early-stopping checks.
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
- **Required of every predicting family: a raw score on a declared scale, per task** (`raw_scale`), which `models/explain.py:Anatomy.raw_score` reads:
  - a numeric outcome: the prediction;
  - yes/no: the margin;
  - ordinal (proportional odds): the latent linear predictor, its `decision_function`;
  - time to event (Cox): the log relative hazard;
  - several classes: per-class curves, or "not drawn" with its reason said, in v2.0.

  The explain stage today supports regression and yes/no only (`stages/explain.py:SUPPORTED_TASKS`). MC-7 widens it to ordinal and time to event, with the performance floor read from each task's own cross-validated metric.
- **Inductive-bias curves need nothing more** (ruling 2). For each top predictor, every family's curve is drawn on one quantile grid shared by every family (`models/explain.py:ale_grid`, `models/explain.py:_curves`). The default is accumulated local effects, because nutrition predictors are strongly correlated and partial dependence then extrapolates outside the data (Apley & Zhu 2020). The contract adds:
  - **Every predicting family draws.** Today `_curves` and `_interactions` run only for families with a SHAP path, because `models/explain.py:_work` returns early when `models/explain.py:model_kind` is None.
  - **The grid comes from observed values.** Today it is built from the first family's adjusted inputs (`ale_grid(works[0].A[a])`), and the adjusted lane includes the fill (`models/pipeline.py:ADJUST_STEPS`). A tree family that keeps blanks has blanks there. So the grid is built from each input's observed (non-blank) values in the training rows, before any fill. Each family's ALE averages over the rows observed in that column, and the caption gives the blank count (RECIPES §2.3).
  - **The top predictors are chosen without SHAP.** Today they come from the best family's SHAP importance (`models/explain.py:_exposure_inputs`). They are ranked instead by the spread of each input's ALE curve under the best-scoring family (a convention; it stays inside the data, as ALE does). SHAP importance remains its own view where `attribution` exists.
  - **The refits are decoupled from SHAP.** Today a refit is kept only when it has attributions (`models/explain.py:_refits`). Every predicting family is refit and redrawn.
  - **A spread band** from those refits, captioned for what it is: "spread over 5 refits on resampled rows". The refits are `RESEEDS` = 5 bootstrap resamples of whole units, each with its own seed (`models/explain.py`, module docstring). The band is mostly sampling spread, not seed spread, and five refits do not make an interval.
  - **The data envelope** is drawn, as the grid's `supported` mask already computes.
- **Optional, declared:**
  - `attribution`: exact SHAP for linear models, path-dependent TreeSHAP for trees, or none. A family with none still gets curves and interactions.
  - `architecture`: the equation, the tree structure, the shrinkage path (`models/explain.py:shrinkage_path`), and the new spectrum view of §2.3.
  - Interaction readouts: Friedman and Popescu's H statistic is model-agnostic and exists (`models/explain.py:h_statistics`). An MLP may add a weight-based screen (Tsang et al. 2018).
- **The floor stays.** A family that does not beat the no-predictor baseline draws no curve (`models/explain.py:floor_of`). Under inference, a curve is drawn for the declared exposure only, as description.

**On the card:**
- "Each model's idea of how sodium relates to blood pressure, on the same axes. They differ because each assumes a different shape." Known as accumulated local effects (Apley & Zhu 2020), on focus.
- When families agree, the caption adds: "The models agree on this shape. Agreement is not evidence the shape is true or causal." Families can share attenuation from measurement error, shared confounding, or smoothing toward a line at small n.
- **Under Predict only,** when the corrected comparison ties two families whose curves separate: "These models predict equally well but tell different stories about sodium." Known as the Rashomon effect (Breiman 2001).
  - It fires only when the two families' bands separate over at least 50 refits on resampled units (a convention). With five refits the card is not drawn, and the one action "Check it across refits" offers the larger run with its minutes.
  - Under Estimate no score is served, so "predict equally well" cannot be said.

**Weights and clusters.** Under the population answer the ALE averages use the survey weights, and every refit resamples PSUs within strata. Otherwise rows are unweighted and resampled by unit.

**Why.**
- Showing the same predictor's curve per family is informative precisely because inductive bias shapes what each learns (Chang et al. 2021). Fig. 3a of Hollmann et al. (2025) is a published precedent on toy functions; TurboTab draws it on the user's own top predictors.
- Where many well-performing models rely on different covariates, report the range, not one model's story (Fisher et al. 2019, on Rashomon sets). Equivalent held-out scores can hide very different behavior (D'Amour et al. 2022).
- Simon et al. (2026, §2.5) describe universality: large models trained on large data converge on similar functions. Two to four families agreeing on one table's curve is a different thing, and the card does not call it universal.

**How the engine enforces it.**
- **A parametrized test over the registry, scoped by task.** For each task a family declares, it either draws a curve on a fixture that equals an ALE computed by hand from its raw score, or states a registered reason for not drawing one.
- The curve-shape test of C5.
- **Today:**
  - Curves exist for the linear families and boosted trees only (`models/explain.py:LINEAR_MODELS`, `models/explain.py:model_kind`). The mixed and GEE estimators have `coef_` and a raw score but are blocked by the class-name gate.
  - The explain stage supports regression and yes/no outcomes only (`stages/explain.py:SUPPORTED_TASKS`), so Cox and proportional odds are refused.
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

**How the engine enforces it: an equivalence test, run as a scheduled heavy run (MC-19).**
- **The criterion is one-sided and stated.** A family declaring `bootstrap_optimism = True` passes when its bootstrap-corrected score does not overstate the fresh-data score by more than a stated margin. The test is an equivalence test (TOST), so a failure to find a difference never counts as soundness. The margin is a convention, set with the prediction reviewer.
- **The replicate count follows from the margin.** It is chosen so that a sound family fails with probability at most 0.05/k, for k registered families, so the whole registry fails falsely at most 5% of the time.
- **The fixtures:**
  - a null fixture;
  - a signal fixture in which a flexible learner nearly memorizes its rows, the regime where Harrell's bootstrap understates optimism;
  - a clustered fixture resampled by unit;
  - plus a p ≫ n signal fixture for the screened elastic net, which is declared sound but unverified.
- **Seeds are fixed,** so a run is deterministic.
- **It does not run in CI.** It is a Monte Carlo against fresh data for every family, scheduled with Nolan, because the dev machine is beside the bed. Each run writes a results file per family version. The CI gate reads that file and fails only when a family's identity or defaults version has changed since its last recorded pass.
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
- The export replay test (`tests/acceptance/test_export.py`) runs for every registered family, not only `linear` and `linear + elastic_net` as today. RECIPES RT-12 replays versions; MC-11 extends the same test over the registry, and the overlap is netted in §5.
- RECIPES T13 adds pinned and re-run replay for tuned families.

### C13 · Reference tests, including the solvable settings

**What.** Every family ships three kinds of reference test.

1. **An independent implementation.** As today: statsmodels, lifelines, glmnet, R, or a hand implementation written from the primary source (§3.1 lists them).
2. **Invariance probes** for C3. A probe injects its transformation as a step right after the family's own scaling step, or at the model's input when it has none, so the family's scaler cannot undo it. Each is two-sided:

   | Declared | Must keep its predictions under | Must change them under |
   |---|---|---|
   | `linear_maps` | a random invertible linear map | a monotone non-linear change of one column (a log) |
   | `rotation_after_scaling` | a random orthogonal rotation | a random invertible map that is not orthogonal (a shear) |
   | `monotone_per_column` | a monotone change of one column, compared on the training rows | a random orthogonal rotation |
   | `column_scale` only | rescaling one column before the family's steps | a random orthogonal rotation |

   - Stochastic families are compared in distribution over seeds, or with the rotation applied to the initialization too. A fixed-seed refit of an MLP changes under rotation, so a seed-by-seed comparison would wrongly fail it.
   - `featurewise` is marked not applicable.
   - Trees and FT-Transformer change under rotation, and ResNet-style networks do not (Grinsztajn et al. 2022).
   - The probes double as teaching exhibits ("why trees and ridge disagree on this table").
3. **Known answers from solvable settings,** declared in `solvable`:

| Setting | The known answer | Tolerance and conditions | Source |
|---|---|---|---|
| Ridge, closed form | The SVD solution at each λ; df(λ) equals the hat matrix's trace, with `alpha` = nλ | 1e-8 and 1e-10 | ESL §3.4.1; RECIPES T11 |
| Deep linear network, full column rank | Whenever gradient flow reaches a global minimum, the end-to-end map is the OLS solution, from any initialization | X centered, or the bias handled explicitly; checked at convergence | elementary: the loss has one minimizer in the end-to-end map |
| Deep linear network, rank-deficient (p > rank X) | The first layer's updates lie in the row space of X, so the null-space part comes only from the initialization; the minimum-norm solution pinv(X)y is reached as the initialization scale goes to zero | Reported at three shrinking initialization scales; the distance must fall with the scale | the row-space derivation, written in the test's docstring; Hastie et al. 2022 for the one-layer case |
| Linear least squares trained by gradient flow from zero | The exact path β(t) = (XᵀX)⁺(I − exp(−tXᵀX/n))Xᵀy | 1e-6 along the path, for the discretized loop with a small step | Ali et al. 2019 |
| Deep linear network, dynamics | With whitened inputs, small task-aligned initialization and squared loss, the singular modes of Σyx are learned one after another, largest first | A multi-output squared-loss fixture with at least three distinct singular values; with one outcome Σyx has rank one and the test would be empty. Order of appearance, not exact times. | Saxe et al. 2014 |
| Wide network in the lazy regime, stage 1 | The trained network matches its own linearization at initialization: regression with the empirical NTK, Θ₀(x,X)Θ₀⁻¹(I − e^(−ηΘ₀t))Y | A tight tolerance; the initial output zeroed first (or its term kept), the output scale set explicitly; computed with `nt.linearize` or the empirical NTK | Lee et al. 2019; Chizat et al. 2019 |
| Wide network, stage 2 | The empirical NTK converges to the analytic infinite-width kernel as width grows | The distance must fall with width, about as 1/√width, under one matched parameterization (NTK parameterization in both the network and Neural Tangents) | Jacot et al. 2018; Novak et al. 2019 (Neural Tangents, a test-only dependency, JAX-based) |

   A ridge term in the lazy oracle appears only with explicit regularization or early stopping. "Kernel ridge regression" is a loose description of the lazy limit, which Simon et al. (2026, §2.1) also use.
4. **The curve-shape test** of C5.

**Why.**
- Simon et al. (2026, §2.1): solvable settings are "analytically tractable cornerstones" that "reveal phenomena and mechanisms to look for".
- For TurboTab, they are references no implementation bug can share. A deep linear network that misses OLS on a full-rank encoded matrix has a bug in its training loop, its encoding or its initialization, whatever its score.
- No test uses a solvable setting today. The harness is new infrastructure (MC-14).

**How the engine enforces it.** The fold-in gate (C14) requires item 1 always, item 2 for every declared invariance, item 3 for every declared solvable setting, and item 4 for every declared curve shape.

### C14 · The fold-in gate

A family joins the shelf only when every blocking item passes. One parametrized acceptance test, `tests/acceptance/test_family_contract.py` (MC-12), runs the automatic items for every registered family. A family added later fails CI until it passes.

**It lands early.** MC-12 lands with MC-1, before the four new families. An item whose machinery is not built yet is a declared expected failure (`xfail`) with the package that will make it blocking. It turns blocking the day that package lands, so no family meets eighteen blocking items at once, late.

| # | Item | How it is checked | Blocking |
|---|---|---|---|
| 1 | Registers cleanly: identity complete, vocabularies, word budgets, recipe and tuning checks | `register_family` | yes |
| 2 | No key or class-name switch outside its module | The syntax-tree no-switch test (C1) | yes, once MC-2b lands |
| 3 | Its profile fields exist, and its `assess` is unchanged when the outcome is permuted, under prediction and under multiple imputation | The outcome-permutation test (C4) | yes, once MC-5 lands |
| 4 | Every quiet name and source resolves in the verified citation registry | Registry test (C5); §7's list until X4 lands | yes |
| 5 | Tuning: RECIPES T1–T17 where it tunes | RECIPES §8 | yes |
| 6 | Each complexity formula matches its closed form with the recorded λ | Formula tests (C7) | yes |
| 7 | Each diagnostic fires on one fixture and stays silent on another | Diagnostic fixtures (C8) | yes, once MC-18 lands |
| 8 | Calibration: the generic checks run; a miscalibrated fixture is flagged | Calibration fixture (C9) | yes |
| 9 | Curves, per declared task: drawn on a fixture, matching a hand-computed ALE, with its declared shape; or a registered reason | Explanation tests (C10) | yes, once MC-7 lands |
| 10 | Bootstrap soundness as declared | The heavy run's results file for this family version (C11) | yes, once MC-19's first run is recorded |
| 11 | Export replay reproduces its matrix and estimates at its tolerance | `tests/acceptance/test_export.py` (C12) | yes |
| 12 | Independent implementation, invariance probes and solvable settings | Reference tests (C13) | yes |
| 13 | The methods reference regenerates with its contract-shaped entry | `reference/methods.py:family_section` | yes |
| 14 | Its methods sentence names it and its settings | Voice tests | yes |
| 15 | The shelf's estimate is within a factor of 3 of the measured time | RECIPES T6 | no (Tier B) |
| 16 | Every journey's first pick is unchanged, or the change is approved | RECIPES RT-13 (a heavy run, scheduled with Nolan; CI reads its results file) | yes, on approval |
| 17 | Plain words and purpose-registry entries for any new element | P0.12 word list; purpose registry | yes |
| 18 | The prediction and inference reviewers' packets carry its row | SIZING R4 | before release |

---

## 2 · Post-fit intelligence from theory

Everything in this section exists only after Fit. Each item names its view class (UNDERSTANDING_LAYER §1.3).
- **Under Predict,** it reads the training rows only. The held-out rows stay sealed.
- **Under Estimate,** it appears only after the lock, and it describes; it never claims a family did well, because no score is served.
- **Weights and clusters.** Under the population answer, every post-fit average here (ALE's interval means, §2.2's local-effect matrix, §2.1's projections) uses the survey weights, and every band or refit resamples PSUs within strata. Otherwise rows are unweighted and resampled by unit.

Its effects are the ones the legality matrix allows: labels, disclosures and sensitivity analyses. None of it adds a candidate (question 2), prunes a family, changes the trunk or re-ranks the shelf.

### 2.1 Target alignment: why the penalized and low-rank families did well or badly (ruling 3)

**What it measures.**
- Take the family's input matrix Z on the fit's rows, centered and scaled as the family scales it (the explain stage's `anatomy.matrix(A_all)`), and its decomposition Z = UDVᵀ. The columns of U are the input's principal patterns, widest first.
- **The raw power** of the centered outcome along pattern j is pⱼ = (uⱼᵀy)².
- **The noise floor.** Noise puts about σ² of power along every pattern, whatever the signal. Summed over many narrow patterns, it can outweigh real signal. So the signal along pattern j is estimated as âⱼ = max(0, pⱼ − σ̂²), where σ̂² is:
  - under Predict, the mean squared out-of-fold residual of the family being explained, from the fit's cross-validated predictions. It includes estimation error, so the floor is conservative;
  - under Estimate, after the lock, the family's own residual variance with its degrees of freedom, RSS/(n − df).
- **The alignment curve** is C(ρ) = (â₁ + … + â_ρ) / Σⱼ âⱼ, summed over all patterns up to the rank of Z. It is the share of what the inputs can explain that lies along the ρ widest patterns. Normalizing by ‖y‖² instead would cap the curve at the in-sample R² and count noise as signal.
- **When it stays silent.** The readout is not shown when Σⱼ âⱼ is not clearly above zero: below the 95th percentile of the same sum with the outcome permuted across rows (a convention; legal after Fit as O3).
- **Its source.** This is a sample, noise-corrected version of the cumulative power distribution of Canatar et al. (2021), whose task-model alignment is defined on the target function itself ("the alignment of the target function with the kernel's eigenfunctions"), not on noisy labels. The exact equation is cited in MC-8's docstring once it has been read in the full text; neither review nor this revision could read it.
- For ridge and principal-component regression the eigen-directions are the input's principal directions. For a kernel family (v2.x) they are the kernel's.
- For a yes/no outcome, y is the 0/1 code. That is an approximation for logistic fits, and the view says so.

**Why it explains results.**
- Ridge shrinks pattern j by dⱼ²/(dⱼ² + nλ), so it shrinks the low-spread patterns most (ESL §3.4.1; C7).
- In kernel regression, targets whose power lies in the top eigenfunctions can be estimated accurately at small sample sizes (Canatar et al. 2021).
- **So when what the inputs can explain lies along the widest patterns, penalized and low-rank families do well.** When it lies along narrow ones, they give some of it up. Small-variance components "can be as important as those with large variance" (Jolliffe 1982), which is exactly why alignment cannot be assumed before the fit.
- **The shelf's spectrum reading assumes no particular alignment** (Dobriban & Wager 2018). This readout is where that assumption is checked.
- **When the tuned penalty sits at the bottom of its grid** with fewer rows than columns, high alignment can explain it. The many narrow patterns already act as an implicit ridge, so the best explicit penalty can be zero (Kobak et al. 2020; Wu & Xu 2020). RECIPES' `penalty_at_edge` concern then gains this explanation instead of only a warning.
- **For kernel families (v2.x),** the eigenlearning framework's conservation law explains underperformance: a kernel can learn only so much in total, and here it spent it where the outcome had little signal (Simon et al. 2023).

**On the card** (after Fit; plain, with the quiet name on focus). "Pattern" is defined in place: "A pattern: a way your columns vary together."
- **Under Predict, mostly aligned:** "Most of what your columns can explain about the outcome lies along their main patterns, which the penalty barely shrinks."
- **Under Predict, poorly aligned:** "Much of what your columns can explain lies along minor patterns the penalty shrinks most, so ridge gave some up."
- **Under Estimate:** the same sentences without the claim about ridge, for example "Much of what your columns can explain lies along minor patterns, which the penalty shrinks most."
- Known as task-model alignment (Canatar et al. 2021). One name for one readout.

**View class and placement.**
- **View class:** O3, a predictor-by-outcome quantity. Under prediction it may label and disclose. After the lock under inference it may diagnose, label and disclose. It is never on the shelf and never before Fit.
- **Where:** computed in the explain stage beside `models/explain.py:_architecture`, and stored on `models/explain.py:FamilyExplanation` as `alignment`. Drawn behind "More angles" on §2.3's view.
- **A cross-check, v2.x:** KARE estimates kernel ridge regression's risk from training data alone (Jacot et al. 2020). It may sit beside BBC-CV after the fit for ridge and kernel families. It must never rank families before Fit.
- **Thresholds:** "most" means C(ρ) ≥ 0.8 at the number of patterns that hold 90% of the input's spread (both conventions). The literature search found no study that measures task-model alignment on real tabular benchmarks. So the thresholds are set on the reference journeys before any card fires; until then the readout is computed and stored, and the card is not shown.

### 2.2 When a flexible family wins and its explanations concentrate: a labeled hypothesis (ruling 6)

**What it reads, in plain words.** The benchmark gives each predictor its own curve and adds them up (`models/selection.py:interpretable_cost`'s regression-with-splines model). If a flexible family clearly beats it, the extra must come from something the benchmark cannot carry: predictors acting together, or thresholds and blanks it handles differently. The noticing asks whether that extra concentrates on a few predictors acting together, and names them.

**What fires it.** Three conditions, all on the training rows of a Predict track:
1. **A flexible family clearly beats the benchmark.** The selection-corrected difference between the benchmark and the best flexible family (`models/selection.py:interpretable_cost`, computed in `stages/evaluation.py:evaluation_stage`) has an interval that excludes zero in the flexible family's favor. "Flexible" is the family's declared `flexible` (C11). This condition is what gives the non-additive part predictive weight: if the truth were additive, the benchmark would not be clearly beaten.
2. **The winner's non-additive part concentrates on a few predictors.** Computed as below, the non-additive share is at least 0.2 of the gradient's power, and the smallest set of predictors that carries 80% of it has two to four members (all conventions).
3. **Enough rows.** Riley's minimum for the benchmark is met (`models/selection.py:shelf_order` already computes it under prediction). Below it the noticing stays silent.

The H statistic's stability across refits is not part of the trigger. It needs the explain stage's refits, which run only when asked; it appears after "Check it across refits".

**How the readout is computed: the local-effect deviations.**
- **Local effects, as ALE takes them.** For each numeric input i, on a per-SD scale, and each training row x observed in that input, with xᵢ in grid interval k: the local difference δᵢ(x) = [f(z_k, x₋ᵢ) − f(z_{k−1}, x₋ᵢ)] / (z_k − z_{k−1}) of the raw score f. Its mean over the rows in interval k is the ALE slope (Apley & Zhu 2020).
- **The deviation** eᵢ(x) = δᵢ(x) minus that interval mean. For an additive raw score, eᵢ is exactly zero at every row, whatever the correlation between inputs, because f(z_k, x₋ᵢ) − f(z_{k−1}, x₋ᵢ) then does not depend on x₋ᵢ. So any deviation is the part of the model that is not additive. This is what the benchmark lacks, by construction.
- **The local-effect matrix** G_ij = the mean of eᵢ(x)eⱼ(x) over rows observed in both inputs. It is an average gradient outer product (Radhakrishnan et al. 2024) of the non-additive part. Its trace over the trace of the full local-difference matrix is the non-additive share. Each input's diagonal entry is its part of that share.
- **Scale and scope.** Inputs are measured per SD, so a column recorded in mg instead of g changes nothing. One-hot and two-valued inputs have no quantile intervals; they are left out, and their contrasts are reported separately. Rows blank in an input contribute nothing for it. For cost, the readout covers the 30 inputs with the widest ALE curves (a convention), and the methods text says so.
- **Main effects.** A pair whose inputs show no main effect in the curves is reported as weaker. This is a convention following the effect-heredity principle for interactions (Chipman 1996).

**Why it is a hypothesis, and what it is not.**
- **A tree win does not show a hidden oblique combination.** The theory that feature learning beats kernel-like methods when the outcome depends on a few hidden directions compares neural networks with kernels or random features, on Gaussian, spherical or binary inputs (Ghorbani et al. 2020; Damian et al. 2022; Bietti et al. 2022; Abbe et al. 2022). Axis-aligned trees are a different learner, and they "struggle on … rotated or interaction-dependent decision boundaries" (Rauniyar 2025, a preprint). So for trees the card speaks of predictors acting together, not of a hidden combination.
- **That framing is reserved for neural and kernel families (v2.x),** and even there it carries Ghorbani et al.'s own condition: the kernel's disadvantage becomes milder when the covariates share the target's low-dimensional structure. Tabular inputs with concentrated spread often do.
- **The procedure's support is empirical.** Using machine-learned patterns to generate hypotheses, with testing as a separate step, is a published procedure (Ludwig & Mullainathan 2024).
- **On these rows the pattern is a hypothesis** (the leash).
- A pretrained in-context family blurs the line between the benchmark and a flexible learner that this comparison relies on (Zhang et al. 2025, a preprint; a lead). When it is the winner, the card says so.

**On the card** (each plain sentence within 22 words):
- "Boosted trees beat the one-curve-per-predictor benchmark, and the extra comes mostly from sodium and potassium acting together."
- "A question for new data, not a finding: do sodium and potassium act jointly on blood pressure?"
- Label: "Exploratory: suggested by these rows". Known as an interaction, on focus. For a neural or kernel winner (v2.x), when the leading direction of G mixes inputs: known as a multi-index model (Li 1991).
- The average gradient outer product is named in the methods text, not on the card.
- One action, "Check it across refits", runs the explanations' refits when they have not run, and then shows the pair's H statistic across them.

**What it may lead to: prediction plus explainability begets further inference.** The leash rows:

| What | Predict track | Estimate track |
|---|---|---|
| The noticing itself | Labels the explanations; a disclosure in Write-up. Its output counts as an estimate shown: `late_notices` joins `estimand.ESTIMATE_STAGES`, so `server/service.py:_lock_when_shown` records it, and a later Estimate track is locked as declared after estimates were seen (`server/service.py:_lock_after_prediction`). | Never fires: no score is served there |
| A term for the benchmark (the pair's product) | Not offered in v2.0 (question 2) | n/a |
| An effect question on the same rows | n/a | Only as an exploratory secondary, recorded as "suggested by data inspection" (`methods/interaction.py:POST_HOC`). It is reported as "selected among N candidates on these rows; its p-value and interval are not valid", with no p-value shown, where N is the number of inputs and pairs the readout scanned. Selection on the same rows invalidates the p-value whatever the multiplicity count, so it is named beside the family of tests, not counted as one more test in it. Never the locked primary. |
| A check on rows no model has seen | When a holdout was sealed before any score: once, at the opening, the pair is tested on the held-out rows alone, in the benchmark, as a labeled secondary (`opened_change_is_secondary`, RECIPES §7.2). Those rows played no part in finding it, so this one test's p-value is valid. | n/a |
| A question for new data | A Write-up sentence | The same |

- **A properly powered follow-up** on held-out or new data can use algorithm-agnostic variable importance with valid intervals (Williamson et al. 2023).
- **New data must use the same measurement protocol,** or the model's predictions may not carry over (Luijken et al. 2019). The Write-up sentence says so.
- Only rows no model has seen, or new data, can change how much the hypothesis is believed. Nothing computed on these rows does.
- It never prunes, never changes the trunk, and never re-ranks the shelf.

**View class and placement.**
- **View class:** the trigger reads scores (O4). The local-effect matrix reads only the fitted model's raw score at training rows.
- **Where:** a post-fit noticing stage, `late_notices`, with deps fit, design and evaluation, under prediction. It computes the local differences on the final fits itself (`fit.objects["fitted"]`, as `stages/explain.py:explain_stage` reads them), on capped rows, with no refits. `evaluation` and `explain` are siblings today, and `explain` runs only when asked, so the noticing cannot wait for it.
- **Threads:** this is the catalog thread `thread:shared-learned-interaction` (CROSSWALK, "Noticings born after the fit"), whose pairs it now finds by the local-effect matrix.
- **Fixtures:**
  - a two-input single-index generator, g(w₁x₁ + w₂x₂) with g non-linear, where it fires and names x₁ and x₂;
  - an additive non-linear generator with correlated inputs (sodium and potassium both tracking energy, each with its own monotone effect), where it stays silent. The local-effect matrix of an additive raw score is checked to be exactly zero;
  - a linear generator, where condition 1 fails;
  - a unit change of one column (g to mg), where the readout does not move;
  - rows below Riley's minimum, where it stays silent.

**Not in v2.0.** Sliced inverse regression as a model-free cross-check is proposed for INBOX, for the v2.x multi-index readout. It estimates all the directions the outcome depends on, additive ones included, so it does not check this readout. When it comes, it reads as "consistent with", never as more confidence. It is skipped for categorical-heavy inputs, for a yes/no outcome with more than one direction (SIR finds at most one there), and when p/n is not small. It also needs the linearity condition and misses symmetric dependence such as y = x² (Li 1991).

### 2.3 What the penalty shrinks, on the tapestry (ruling 5)

**The shrink view.** A Focus on the penalty (FOUNDATION §5), drawn with the curve view kind (SIZING P0.3b):
- **x:** the family's top 30 principal patterns, widest first;
- **bars:** each pattern's spread, dⱼ²;
- **line:** the shrink factor dⱼ²/(dⱼ² + nλ) at the chosen penalty, read from the tuning record (RECIPES RT-5f) and converted by C7's callable;
- **one number:** df(λ) (ESL §3.4.1, eqs. 3.47 and 3.50).
- **Behind "More angles":** the alignment bars âⱼ of §2.1 on the same axis, after Fit. They are not on the main view, which would otherwise stack two bar series, a line and a number.

**On the card** (the quiet name on focus):
- "Ridge barely shrinks the main patterns in your columns and shrinks the minor ones hard." Known as L2 shrinkage (ESL §3.4.1).
- "This penalty leaves the fit about 14 columns' worth of freedom, out of 412." Known as effective degrees of freedom. The count is out of rank(Z), the most a plain fit could use, never out of the concentration number.

**Per family.**
- **Ridge:** exact.
- **Elastic net:** the line shows its ridge part only, and the card says so. The lasso part drops columns, which the existing path view shows (`models/explain.py:shrinkage_path`).
- **Logistic ridge:** the factors use the weighted matrix at the fitted probabilities, labeled an approximation (a convention).
- **Networks trained by gradient descent (v2.x):** no position on this line. Early stopping is ridge-like only for linear least squares (Ali et al. 2019), and even a lazy network's early stopping acts on the NTK's eigenbasis, not on the input's principal patterns. If anything, a separate NTK view is drawn, labeled approximate, for a measured lazy fit only.
- **Random forest:** no per-pattern line, because its splits follow the columns, not these patterns. Its knob appears on the tuning line: "Choosing from a random handful of columns at each split is this forest's penalty knob." Known as randomization as regularization (Mentch & Zhou 2020).
- **A mean fill followed by centering, in a linear family:** a note, "Part of this shrinkage comes from filling blanks with the average, if blanks are missing by chance alone." Known as regularization by naive imputation (Ayme et al. 2023). The tuned penalty was chosen on the filled data, so it already reflects the fill; the note does not say the penalty is larger than tuned.
- **Error-prone inputs:** not on this view. See §2.4's measurement-error entry, which is an Estimate-track card.

**Before and after Fit.**
- **Before Fit,** the bars alone are the input's shape, an O0 measure, and may appear under the shelf's "More angles" as part of the input profile.
- **The shrink line needs the tuned penalty,** which was chosen with the outcome, so it appears only after Fit. The alignment bars appear only after Fit (§2.1).

**Where:** `models/explain.py:Architecture` gains a `spectrum` part beside `kind="shrinkage"`, computed from the SVD of the scaled Z.

### 2.4 The named-phenomena registry (ruling 4)

**The registry.**
- `phenomena.py` holds one `Phenomenon` per entry: its key, the plain card sentence (≤ 22 words), its quiet name, its sources, where it is detected, its view class, the families it applies to, and its guard: when the label must *not* be used.
- Every `known_as` on a concern, curve, card or thread points into it. Each name is used for one phenomenon only.
- A test checks that every source key resolves in the verified citation registry (SIZING X4; §7's list until then).
- Every entry with a detector has a fixture where it fires and one where it stays silent. An entry marked **teaching only** has no detector: it appears in teaching exhibits and in "Why does this matter?", and it is exempt from the fire-and-silent rule.

| Phenomenon | The plain card sentence | Known as (source) | Where it is detected | Guard |
|---|---|---|---|---|
| Double descent | "Error rose, then fell again, as the model grew past the size that fits every training row exactly." | Double descent (Belkin et al. 2019; Hastie et al. 2022) | Teaching only: a learning curve over capacity with the penalty off | Tuned penalties make risk monotone, so users rarely see it (Nakkiran et al. 2021). For trees and boosting, name the capacity axis being varied, or do not use the label (Curth et al. 2023). Kanoh (2026, preprint) is watched, not used. |
| Smoothness, or spectral, bias | "The neural net's curve is smoother than the trees': it learns broad trends before sharp steps." | Spectral bias (Rahaman et al. 2019; for tabular data, Beyazit et al. 2023) | Teaching only in v2.x: no measure of "smoother" is defined yet | Neural nets are biased toward overly smooth solutions (Grinsztajn et al. 2022). The card may name the remedy slot: numeric embeddings or rank scaling (§4). |
| Rotation invariance and columns that may not matter | "With many columns that may not matter, an L1 penalty can need far fewer rows, in the worst case." | Rotational invariance (Ng 2004) | Before Fit, under "More angles" only, from width and p/n (O0); the rotation probe as a teaching exhibit | A worst-case bound, not a forecast. Whether a column matters depends on the outcome, so it is never judged before Fit (Grinsztajn et al. 2022). |
| Lazy versus rich training | Lazy: "This network barely changed its inner weights while learning, so it behaved like a fixed-feature model." Rich: "This network reshaped its inner weights around a few directions: it built its own features." | Lazy versus rich training (Chizat et al. 2019) | After Fit: the C8 weight-movement diagnostic, trained networks only. Nothing before Fit. | The threshold is a convention that scales with width. |
| Edge of stability | "Training took the largest steps the loss surface allowed; the loss wobbled but kept falling." | Edge of stability (Cohen et al. 2021) | After Fit: the C8 sharpness diagnostic | Full-batch gradient descent only. |
| Task-model alignment | §2.1's sentences | Task-model alignment (Canatar et al. 2021) | After Fit, explain stage (O3) | Never before Fit. Describes only, under Estimate. Silent below the permutation floor. |
| Concentration of spread | Before Fit: "Your 412 columns vary together like about 40 unrelated columns would." | Effective dimension (the participation ratio; a convention). Effective rank r₀ (Bartlett et al. 2020) is a quiet measure, not this sentence. | Before Fit, input profile (O0), under "More angles" | r₀ is never read as "how many directions the spread fills". |
| Implicit ridge | After Fit, penalty at the floor: "The best penalty here was almost none: your many minor patterns already act as one." | Implicit ridge regularization (Kobak et al. 2020) | After Fit, with `penalty_at_edge` and high alignment | Only with fewer rows than columns. "Benign overfitting" (Bartlett et al. 2020) is named only for a fit that interpolates. |
| A forest's random column choice | "Choosing from a random handful of columns at each split is this forest's penalty knob." | Randomization as regularization (Mentch & Zhou 2020) | The tuning line and §2.3 | none |
| Predictors acting together | §2.2's sentences | An interaction; for neural and kernel winners (v2.x), a multi-index model (Li 1991) | After Fit: `late_notices` (§2.2) | Exploratory label always. For trees, never "a hidden combination". |
| Same score, different stories | "These models predict equally well but tell different stories about sodium." | Rashomon effect (Breiman 2001) | After Fit, under Predict only: BBC-CV's tie and bands that separate over at least 50 refits (`thread:shared-rashomon-disagreement`) | Report a range, not one family's story (Fisher et al. 2019; D'Amour et al. 2022). |
| Measurement error flattens a curve | "Random error in measuring sodium tends to flatten its curve; errors shared with energy intake can push either way." | Regression dilution (Hutcheon et al. 2010) | After the lock, on an Estimate track with one declared error-prone exposure and a stated reliability (`stages/modeling.py:_measurement_error_line`) | Never under Predict: with the same instrument at deployment, dilution is not a failure. Never as an excuse for a null. Self-reported intake has errors correlated with energy and across nutrients, which can bias effects either way in a multivariable model (Kipnis et al. 2003); Hutcheon et al. cover the single-predictor case only. |
| Filling blanks shrinks a linear fit | "Part of this shrinkage comes from filling blanks with the average, if blanks are missing by chance alone." | Regularization by naive imputation (Ayme et al. 2023) | After Fit, on §2.3's view | A mean fill followed by centering, in a linear family only. Blank counts cannot show that blanks are missing by chance, so the sentence keeps its "if". |

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
| C2 inference table | intervals (HC3, CR2, Firth, survey) | shrunk, no intervals | description only (curves, no table) | intervals (classical t, CR2 to 200 exposures, BH; no survey) | intervals (Wald, sandwich, survey) | intervals (Satterthwaite; no survey) | intervals (CR2; no survey) | intervals (Wald, Lin–Wei, survey) | refused |
| C3 invariances (to declare) | linear maps | column scale | monotone per column | not applicable | linear maps | linear maps (random intercepts) | linear maps | linear maps | column scale |
| C4 assess reads | rows, parameters, events, outcome mean and SD, units | rows, columns, purpose | rows, columns, purpose | purpose, lenses, columns vs rows | class counts, columns | units, columns, purpose | units, EPV | events, units | purpose, p vs n, then the elastic net's |
| C6 tuning | n/a | part (F4, F5, F13) | no (F2) | n/a | n/a | n/a | n/a | n/a | part (F4) |
| C8 diagnostics | yes | no | no | part | yes | part | part | yes | no |
| C9 calibration | generic, plus shrinkage updating | generic | generic | n/a | generic, by level | generic | generic | at a horizon | generic |
| C10 explanation | SHAP, equation | SHAP, path | TreeSHAP, trees | n/a | none (task refused) | blocked by the class-name gate | blocked by the class-name gate | none (task refused) | SHAP, path |
| C11 bootstrap | decl yes | decl yes | decl no | n/a, printed "yes" (declared None, printed "not applicable", by MC-1) | decl yes | decl yes | decl yes | decl yes | decl yes, unverified at p ≫ n |
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
| C1 | scikit-learn `Ridge`; logistic with an L2 penalty | statsmodels `RLM` wrapper | scikit-learn forest | `xgboost`, tree booster; threads recorded; `same_kind_as = ("boosted_trees", −1.0)` |
| C2 | reg, bin, multiclass; prediction; inference as a shrunk table with no intervals | reg; prediction only | reg, bin, multiclass; prediction; description only under inference | as the forest |
| C3 invariances | rotation after scaling (column scale implied) | linear maps | monotone per column | monotone per column |
| C4 reads | rows, columns, p/n, concentration, spectrum, condition number, indicators | rows, columns, outlying share | rows, columns, blanks routed, irregularity | boosted trees' reads, through `same_kind_as` |
| C4 prior | Ng 2004 (lower bound) | convention | convention | McElfresh et al. 2023 (irregularity) |
| C5 quiet name | L2 shrinkage (ESL §3.4.1) | Huber loss (Friedman 2001; RECIPES §2.2 for Huber 1964) | randomization as regularization (Mentch & Zhou 2020) | gradient boosting (Friedman 2001) |
| C5 curve shape | straight | straight | piecewise constant | piecewise constant |
| C6 | path (RECIPES §4.1); the grid reaches near zero for wide data (Kobak et al. 2020) | by hand only | search, out of bag where allowed | search |
| C7 knobs | λ per row (`alpha` = nλ), df(λ) | threshold t | mtry, leaf size; effective smoothing (Curth et al. 2024, preprint) | rounds, learning rate, depth, L2; no double-descent label without a named axis |
| C8 | penalty at the edge; calibration slope per fold | IRLS convergence | out-of-bag error | early-stopping round |
| C9 | margin and probability | value | probability; the wrapper adds a margin `decision_function` | margin `decision_function` (RECIPES §2.2) |
| C10 | linear SHAP; equation; shrinkage path; §2.3's spectrum view; §2.1's alignment; multiclass curves not drawn in v2.0, said | linear SHAP; equation | compiled TreeSHAP (RECIPES RT-5d); curves via the margin | SHAP from `pred_contribs` |
| C11 | bootstrap sound (path families are bootstrapped, RECIPES §4.3); not flexible | sound; not flexible | not sound; flexible | not sound; flexible |
| C12 | 1e-12 | 1e-12 | 1e-12, prediction single-threaded (RECIPES §4.7) | 1e-12, or a declared tolerance with Nolan's approval (RECIPES §4.7) |
| C13 | closed form and df(λ); keeps predictions under rotation after scaling, changes them under a shear | IRLS from Holland and Welsch; MASS::rlm (RECIPES T11); keeps predictions under a linear map, changes them under a log | direct library fits; TreeSHAP to 1e-6; keeps predictions under a monotone change on training rows, changes them under rotation | native `xgb.train`; `pred_contribs`; the same probes as the forest |

### 3.3 The switches on family keys to retire

Each of these is a place a new family must be added by hand today. Each becomes a read of a declaration. The first four are the ones the four new families hit, so they go first (MC-2a); the rest follow (MC-2b).

| Where | What it switches on | Replaced by | Package |
|---|---|---|---|
| `models/explain.py:LINEAR_MODELS`, `models/explain.py:model_kind` | estimator class names | `attribution`, `architecture` and `raw_scale` | MC-2a |
| `models/cost.py:fit_cost` | `isinstance` checks for `LinearRegression` and Newton–Cholesky `LogisticRegression` | the declared cost model (C6) | MC-2a |
| `voice.py:_FAMILY_LABEL`, `voice.py:_family_label` | methods labels by key, `linear` by task | `methods_label(task)`, the methods register's name for the family (`describe(task, purpose)`'s label names the model step instead) | MC-2a |
| `models/selection.py:is_flexible` | `flexible` derived from `bootstrap_optimism` | `flexible`, declared | MC-2a |
| `methods/omics.py:model_clause` | elastic net and screened elastic net | `tuning.kind` and the plan's sentence | MC-2b |
| `stages/modeling.py:fit_stage` | the collinearity concern for `linear` only | `"collinearity" in diagnostics` | MC-2b |
| `stages/effects.py:SEQUENCE_FAMILIES`, `stages/effects.py:matrix_table` | families with a matrix table | `inference_decl.matrix_table` | MC-2b |
| `stages/effects.py:_Run.diagnostics` | diagnostics for `cox` and `linear` only | `diagnostics` | MC-2b |
| `stages/evaluation.py:_shrinkage` | shrinkage updating for `linear` only | `updating` | MC-2b |
| `methods/interaction.py:SUPPORTED` | families that test product terms | `inference_decl.product_terms` | MC-2b |
| `models/survey.py:_DESIGN_FAMILY`, `models/survey.py:has_design_estimator` | the design-based family, and a signature check | `inference_decl.design_based` | MC-2b |
| `estimand.py`, the `family_needs_featurewise` refusal | `featurewise` by key | `predicts` and `purposes` | MC-2b |
| `decisions.py:model_families` | fallback keys | the registry, always importable | MC-2b |
| `stages/scales.py:FAMILY_FOR` | the family a scale's analysis uses, by task | `inference_decl.default_for` | MC-2a (done early, with the scales stage) |
| `stages/class_substitution.py:_plain_multinomial` | an `isinstance` check for an unpenalized multinomial `LogisticRegression` | `linear_in_values` and the declared output | MC-2b |
| `method_previews.py` (the measurement-error preview, about line 664) | `"linear" in after.models` | `linear_in_values` and `inference_decl.table == "intervals"` | MC-2b |
| `teaching/content.py` (the models question's options) | family-keyed teaching text | each family's `describe()` and `bias_terms` | MC-2b |
| `reference/catalog.py:FAMILY_LENSES` | family-keyed review lenses | `review_lenses` | MC-2b |

**Found while building MC-2a.** The syntax-tree test widened to dictionaries keyed by family, lists named by family keys, lookups in them, `type(x) is`, `issubclass`, `match`, and any name whose words include `family`, `fam` or `key`. It found more places than this table holds: `models/survey.py`'s `_DESIGN_LABEL`, `_ESTIMATOR_WORDS` and `_BLOCKED_WORDS` with `models_sentence`, which reads them; `scales.py:methods_sentence`'s model by key; `reference/catalog.py:lenses_of_family`; and V2X_SEAMS row 21's `methods/interaction.py:_measure`. Its `NOT_YET` list, which pins each place's family keys and switch count, is the census of record for MC-2b. `models/survey.py:models_sentence` now names a family neither of its tables holds by its `methods_label`, never by its key. `models/wide.py`'s class check stays: the module declares the elastic net's wide model step, so the switch is the family's own.

**Also renamed:** the causal lane's learner key `random_forest` (`models/causal.py:LEARNERS`, `decisions.py:CausalLearner`: 200 trees, at least 5 rows per leaf) differs from the RECIPES forest (500 trees, searched leaf and mtry). It becomes `nuisance_forest` until C6c makes the causal lane's learners read the registry's declarations, so the methods text and the no-switch test never confuse two forests.

**Allowed:** exits that recommend a family by key, such as the lever exit `select_models(['featurewise'])` in `stages/explore.py` and the exit to `mixed` in `models/inference.py`.

### 3.4 Count mismatches the profile fixes

The shelf's counts today diverge from what a family receives (`stages/modeling.py:shelf_stage`, `stages/modeling.py:predictor_parameters`, `stages/modeling.py:rule_spline_terms`):
- missing indicators are not counted;
- the energy step's dropped or added columns are not counted;
- scale items are counted, though they collapse to one score;
- blanks as a level add a level per category that is not counted;
- the batch drop and the in-fold filters are not reflected;
- category levels are read from summaries over the whole table, held-out rows included;
- the spline rule is applied to every family, though RECIPES §2.3 has tree families skip it;
- a recipe change is not reflected at all.

The input profile is computed from each family's own steps on its current recipe, so each of these is counted where it applies. Its row count is the training rows, not RECIPES' n_plan, so the card and the basis line agree. Riley's minimum reads the trunk's reference-coded parameter count, and says so. The timing estimate also stops counting bootstrap refits for families that are never bootstrapped (RECIPES F7, in `stages/modeling.py:_estimates`).

---

## 4 · Neural and in-context families (v2.x)

These come after v2.0.0. Each enters through the same door. Below are the clauses that differ for each.

### 4.1 An MLP family (first)

**Which MLP.**
- An MLP that efficiently imitates an ensemble of MLPs was the best tabular deep model in its authors' evaluation, and MLP-based models beat attention models there (Gorishniy et al. 2025). That makes it the first neural candidate, before a transformer.
- Meta-tuned defaults (Holzmüller et al. 2024) are a reasonable candidate 0, labeled unvalidated below 1,000 rows. Many nutrition cohorts are smaller.

**C1 identity:** optimizer and schedule, batch size or full batch, epochs or the stopping rule, initialization scheme and scale, parameterization and output multiplier (standard is allowed, with no implied regime), ensemble size, and seeds for initialization and batch order.

**C3 recipe:**
- `scale` defaults to every column on one scale. Rank or quantile scaling is an option, principled because it reduces the target's irregularity (Beyazit et al. 2023).
- A numeric-embedding slot: none, piecewise-linear or periodic (Gorishniy et al. 2022). Random Fourier features are labeled experimental (Sergazinov et al. 2025, preprint).
- **Invariance:** `rotation_after_scaling` only in distribution over seeds, without per-column embeddings, with a rotation-invariant initialization and plain gradient descent. An Adam- or AdamW-trained MLP does not declare it unless the probe shows it; Adam's per-coordinate scaling breaks the invariance even in distribution. Whether an embedded MLP is rotation-invariant is not sourced, so the probe decides and the declaration follows it.

**C4 assess reads:** rows, columns, concentration, the irregularity measures and the outlying share (a convention, pending the check of Jeffares et al.).
- Heavy tails lower its fit to the situation (McElfresh et al. 2023).
- Its prior states that deep models catch up mainly under larger time budgets (Erickson et al. 2025).

**C6 tuning:**
- Searched: learning rate, weight decay, dropout, width and epochs. A regularization search is supported (Kadra et al. 2021), with its caveat.
- Structural, never searched: the parameterization and output multiplier.
- Width transfer under muP is not used; small tabular networks are cheap to tune directly (Yang et al. 2021).

**C7:** input-noise augmentation, if offered, is to first order a Tikhonov penalty (Bishop 1995). Early stopping gets no position on §2.3's shrink line.

**C8 diagnostics:** loss and convergence, lazy versus rich as measured with a width-scaled threshold, the edge of stability under full batch, and the neural feature ansatz (Radhakrishnan et al. 2024).

**C10 explanations:** curves via the raw score. The local-effect matrix comes from its gradients, and the multi-index name may be used when its leading direction mixes inputs (§2.2). A weight-based interaction screen (Tsang et al. 2018) is triangulated with the H statistic.

**C11:** flexible; `bootstrap_optimism = False`, following the engine's reasoning for near-interpolating learners (a convention until its soundness run).

**C13 solvable settings:** both deep linear rows, the gradient-flow path, deep linear dynamics on a multi-output fixture, and both stages of the lazy wide network (C13's table). They are run on the family's own training loop with its nonlinearity removed or its width raised, so they test the loop, the encoding and the initialization that ship.

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

**C5 inductive bias:** "Predicts by approximate Bayesian inference under a prior learned from simulated datasets." Known as a prior-data fitted network (Müller et al. 2022). It is a prior, not a fitted penalty, and the card contrasts it with ridge's explicit penalty. The reading for statisticians as approximate Bayesian inference is a lead (Zhang et al. 2025, a preprint).

**C6 and C7:** `TuningDecl(kind="none")`. Its complexity control is the context: which rows it conditions on.

**C9:** its probabilities are a posterior predictive under its prior (Müller et al. 2022), and the generic check measures their calibration on the user's data.

**C10:** curves via the raw score. Its authors interpret it with SHAP (Hollmann et al. 2025).

**C11:**
- In each fold the context holds that fold's training rows only. That is the fold's "fit".
- `bootstrap_optimism = False` (a convention).
- It is `flexible`. It may blur the contrast between the benchmark and a flexible learner that §2.2 relies on (Zhang et al. 2025, a preprint), so §2.2's card names it when it wins.

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
| **MC-1** Declarations | §1's members in `models/base.py`, without RECIPES' recipe members (RT-2 adds those): `Identity`, `InferenceDecl`, `Prior`, `Knob` with `models/formulas.py`, `Named`, `Source`, invariances, curve shape, `same_kind_as`, `raw_scale`, diagnostics keys, output, updating, attribution, architecture, review lenses, solvable keys, tolerance, sources; `TuningDecl.structural`. `register_family`'s checks. `FamilyInfo` and `ShelfFamily` extended, with `terms` beside `concerns`. The nine families declare theirs. | P0.2 | M (3) | engine | DoD §2 |
| **MC-2a** The switches the new families hit | `explain.model_kind` and `LINEAR_MODELS`, `cost.fit_cost`, `voice._FAMILY_LABEL`, `selection.is_flexible` read declarations | MC-1 | S–M (2) | engine | DoD §2 |
| **MC-2b** The rest, and the test | §3.3's other fourteen places; the causal learner key renamed; the syntax-tree no-switch test with its allowlist at zero | MC-1 | M–L (5.5) | engine | DoD §2 |
| **MC-3** The trunk stage and readiness | `trunk` holds only the fitted shared transformer and the profile's inputs, as objects. `design` keeps the lineage, the matrix file and the warnings, which the export (`export/bundle.py:matrix_bytes`), the gate's "The model matrix" and the screens read. `shelf` and `design` depend on `trunk`. The readiness predicate from P0.4's registry; "Ranked on the defaults now set" and the open items; the shelf's reads widened. Versions bumped: shelf 15→16, design 24→25. | P0.4, P0.5 | M–L (5.5) | engine | rulings of 2026-10-08 |
| **MC-4** The input profile | `InputProfile` per family on its current recipe (defaults for unselected families, one profile per option of a `choose` slot) through `family_spec`. Every step declares `reads_outcome` (about twenty step classes); O1 steps fed the counts; the stop at O3 with "before selection"; the outcome-free fill under inference and blanks from the mask; spectrum, participation ratio, r₀, condition number, irregularity, outlying share, Kish's effective size, with cost caps. `Situation.inputs`; `Assessment.measures`, `reads` and `terms`. §3.4's count fixes. | MC-1, MC-3, RT-2, RT-3 | L + S–M (10) | engine | rulings of 2026-10-08 |
| **MC-5** Assess rewritten | The nine families and the four new ones read their profiles, with two-register concerns and sourced priors; XGBoost through `same_kind_as`; the outcome-permutation test with its multiple-imputation and O1 fixtures | MC-4 | M (3) | engine | rulings of 2026-10-08 |
| **MC-6** Live re-assessment | The `set_recipe` preview (RECIPES RT-6) returns the edited family's profile and `assess`, stopping before outcome-reading steps | MC-4, RT-6 | S–M (2) | engine | rulings of 2026-10-08 |
| **MC-7** Curves for every family | `_curves` and `_interactions` decoupled from `model_kind`; raw scales per task, with the explain stage widened to ordinal (latent score) and time to event (log relative hazard), multiclass said; the grid from observed values and ALE over observed rows with blank counts; top inputs by ALE spread; refits decoupled from SHAP; the band and its caption; PSUs within strata and weights under the population answer; the data envelope; the Rashomon label under Predict, at 50 refits | MC-1 | M–L (5.5) | engine | rulings of 2026-10-08 |
| **MC-8** Alignment and the shrinkage view | The SVD of the scaled Z; shrink factors and df(λ) through C7's callables; the noise floor, C(ρ), the permutation floor and the alignment bars; the `spectrum` part of `Architecture`; `alignment` on `FamilyExplanation`; the Focus view's spec, with alignment behind "More angles" | MC-7, RT-5b, RT-5f | M (3) | both | rulings of 2026-10-08 |
| **MC-9** The hypothesis noticing | `late_notices` (deps fit, design, evaluation), registered in `estimand.ESTIMATE_STAGES`; the local-effect matrix on per-SD inputs; the trigger on `interpretable_cost`, the non-additive share and Riley's minimum; the card, the leash rows and the sentences; the held-out test at the opening; "selected among N" on the Estimate secondary, recorded on the modifier decision; fixtures that fire and stay silent (correlated additive, unit change, below Riley). Widens `thread:shared-learned-interaction`, already counted at 0.5 in T2. | MC-7, U1, U2 | L (8) | both | rulings of 2026-10-08 |
| **MC-10** The phenomena registry | `phenomena.py`; `known_as` and `source` on concerns, curves, cards and thread sentences; one name per phenomenon; teaching-only entries exempt from fire-and-silent; the registry test against X4's citation registry (§7's list until then) | MC-1, MC-18 | M (3) | engine | rulings of 2026-10-08 |
| **MC-11** Probes and replay for every family | The two-sided invariance probes injected after the family's scaler, in distribution for stochastic families; the curve-shape tests; export replay over the registry | MC-1 | M (3) | engine | DoD §2 |
| **MC-12** The fold-in gate | `tests/acceptance/test_family_contract.py` over the registry (§1, C14), landing early with declared expected failures that turn blocking as their packages land | MC-1 | M (3) | engine | DoD §2 |
| **MC-13** The methods reference | `reference/methods.py:family_section` prints the contract: sources, two registers, invariances, inference table, "not applicable" where sound does not apply | MC-1 | S (1) | engine | DoD §2 |
| **MC-17** The shelf card's two registers | The shared "Looked at" line and per-family differences; "Known as" on focus; "Waiting for" and "Ranked on the defaults now set"; both options of a "Try both" family; the priors and concentration under "More angles" | MC-5, P0.7 | M (3) | interface | rulings of 2026-10-08 |
| **MC-18** The diagnostics registry | `models/diagnostics.py`, gathering today's diagnostics from `fit_stage`, `effects._Run.diagnostics` and `_concerns`; the forest's out-of-bag and XGBoost's early-stopping checks; a fire and a silent fixture for each | MC-1 | M (3) | engine | DoD §2 |
| **MC-19** The soundness runs | The equivalence test with its margin and replicate count; null, signal, clustered and p ≫ n fixtures; fixed seeds; a results file per family version; the CI reader. The runs themselves are scheduled with Nolan. | MC-1 | M (3) | engine | DoD §2 |

**v2.x packages** (after v2.0.0):

| Package | What it is | Size |
|---|---|---|
| **MC-14** The solvable-settings harness | Both deep linear rows; the exact gradient-flow path; deep linear dynamics on a multi-output fixture; the lazy wide network in two stages against `nt.linearize` and Neural Tangents (a test-only dependency) | L (8) |
| **MC-15** Neural training diagnostics | Lazy versus rich with a width-scaled threshold, sharpness against 2/η, the neural feature ansatz | M (3) |
| **MC-16a** The MLP family | §4.1 | L (8) |
| **MC-16b** The in-context family | §4.3, on the server option first | L (8) |

**The total.**
- **v2 packages:** 66.5 units before overlaps.
- **Overlaps, netted:**
  - MC-1 with RT-2's `register_family` and `FamilyInfo` work: about 1;
  - MC-11 with RT-12's replay of versions: about 1;
  - MC-12 with RT-14's acceptance harness: about 1;
  - MC-7 with the new families' explanation paths in RT-5b to RT-5e (margins, wrappers): about 1;
  - `shared-learned-interaction`, already in T2: 0.5.
- **New units:** about **62**, which is about **8.4%** on top of SIZING's 734 remaining. Draft 1 said 47; its own arithmetic was 49.5, and the review added the step declarations, the diagnostics registry, the soundness runs, the shelf card's interface work and the wider switch count.
- **v2.x:** about 27 units.
- **DoD §2's own requirement** (MC-1, MC-2a, MC-2b, MC-11, MC-12, MC-13, MC-18, MC-19) is about 23.5 of the 66.5, or about 20.5 after its three overlaps. It is arguably already implied by C6a's "each family through the method contract", whose 32 units are the RT packages' own sizes and carry no separate allowance for it.

**How they fit the road** (SIZING "The order, and why").
- **First, in C6a, before RT-5b to RT-5e:** MC-1 (without recipe members, so it does not wait for RT-2 in C6b), MC-2a, MC-12 with its expected failures, and MC-13. The four new families then enter through the door instead of adding to §3.3's list. MC-2b can run any time after MC-1, before G1.
- **MC-3** needs the stage registry and the Confirm sweep (P0.4, P0.5), and can run beside the C6 engine work.
- **MC-4 and MC-5 follow RT-2 and RT-3** (C6b) and MC-3, because the profile needs `family_spec` and the recipe slots. **MC-6** follows RT-6.
- **MC-7 and MC-8 land with C7d** (Results under Predict). MC-8's view needs the curve view kind (P0.3b).
- **MC-9 rides with T1/T2** as the widened `shared-learned-interaction`, and needs the thread machinery (P0.9's U1 and U2) and MC-7.
- **MC-18 and MC-10 land with P0.12** (plain words), before X4 closes.
- **MC-11 follows MC-1. MC-19** follows MC-1; its runs are heavy and scheduled with Nolan.
- **MC-17 lands with C6d** (Models under Predict).
- **Gate items turn blocking** as their packages land: item 2 with MC-2b, item 3 with MC-5, item 7 with MC-18, item 9 with MC-7, item 10 with MC-19's first recorded run, item 4 against X4 once it lands.
- **Heavy runs are scheduled with Nolan.** These are RT-13's regenerated journeys and MC-19's soundness runs. The dev machine is beside the bed.

**Not taken in this design.**
- Ranking families by a pre-fit risk estimate such as KARE: it reads the outcome (§2.1).
- Showing target alignment on the shelf: ruling 3.
- Stating a regime on the shelf: lazy versus rich is measured after the fit (C4, C8).
- Making a pretrained in-context model the default: its limits and dependency weight, and it blurs §2.2's contrast.
- muP for tabular networks: little leverage at these widths.
- Proposed for INBOX: sliced inverse regression for the v2.x multi-index readout (§2.2); a `terms` recipe slot for the benchmark (question 2).

---

## 6 · Open questions for Nolan

1. **What does "outcome-blind" mean for the shelf?** Ruling 1 says everything is outcome-blind. Today linear, proportional odds, GEE and Cox read the outcome's own counts (events, class sizes, the outcome's mean and SD) for Riley's minimum, EPV and Whitehead's effective size. Those are O1 under the legality matrix, which already lets O1 "set the df budget".
   - **Recommendation:** read "outcome-blind" as no predictor-by-outcome quantity and no score (no O3, no O4). Keep the O1 counts for the sample-size criteria and the O1 steps, and keep the O2 library-size check as its own shelf line. Under inference the multiple-imputation fill reads the outcome, so the profile measures the input on a fill that ignores it (C4). The outcome-permutation test enforces exactly this.
   - The strict alternative would drop TRIPOD+AI item 10's sample-size check from the shelf.
2. **May a hypothesis from §2.2 ever add a candidate model?** The legality matrix's O4 cell under prediction allows labels, disclosures and a baseline comparison, not a new candidate. Draft 1 recommended adding the term as a later version corrected by BBC-CV. That was wrong on two counts:
   - RECIPES has no such path. A version is a family's recipe slots plus its tuning. An added term is a form, which is shared trunk and creates no version (RECIPES §3.3; the crosswalk's ruling of 2026-10-08 that a shared-step change keeps a read-only earlier row).
   - The term would come from the final fit, trained on every outer test fold, so its version's out-of-fold scores would be optimistic. BBC-CV corrects only the choice among fixed versions; how a later version was designed "is not corrected" (RECIPES §3.3). RECIPES §3.5 already blocks this kind of quiet leak for values from tuning results.
   - **Recommendation:** not in v2.0. The hypothesis stays a label, an exploratory secondary on an Estimate track, one valid test on sealed held-out rows at the opening, and a question for new data. The candidate route goes to INBOX.
   - **If you want it in v2:** a `terms` recipe slot for the benchmark, routed as RECIPES §3.5 routes data-derived values. Under holdout `none` or `after_scores` it is fitted and shown, labeled "suggested by these rows: not an honest estimate", and excluded from BBC-CV and the declared result. Under `sealed` it is allowed, because the holdout absorbs it. Under `opened` it is a secondary. Its honest exit is "Discover inside each fold": the readout, the term and the refit run on each outer training fold and are scored on that fold's test rows. About L (8 units), sized with RT-7.
3. **Do these rulings enter v2.0.0 by amendment, or by displacing something?** The definition of done admits a new idea only by displacing something, though the 2026-10-07 amendment added without displacing, at your direction. These packages add about 62 units (8.4%).
   - **Recommendation:** amend v2.0.0 to include them. They are the north star made concrete, and about a third of them is DoD §2's own requirement.
   - If the date matters, the displacement candidate is the kept tuning groups (C6c, 33 units). The rulings serve explainability more directly than successive halving, Hyperband, TPE and BOHB do.
4. **Should scales, batch and the omics normalization move ahead of the families question?** They change a family's width and spectrum more than anything else. Yet the Models Decide order puts them after `q:models` (CROSSWALK §5, Decide 10–12), so the shelf must rank before they are answered.
   - **Recommendation:** move them ahead of `q:models`, recorded as crosswalk disagreement 21. They have no Router key, so the move costs only the stage registry's order (P0.4).
   - The alternative keeps the order. The shelf then ranks on their current state, labeled "scales not yet answered", until they are answered, and re-ranks when they are.
   - **Ruled 2026-10-08** (DoD amendment, "Order"): moved ahead, as recommended. C4 reads the ruled order.

---

## 7 · Sources

Only entries verified for this spec are listed, with the theory source. Engine conventions are marked "convention" where they appear. Entries added in revision say how they were checked.

**The theory source (a preprint)**
- Simon, J., Kunin, D., Atanasov, A., Boix-Adserà, E., Bordelon, B., Cohen, J., Ghosh, N., Guth, F., Jacot, A., Kamb, M., Karkada, D., Michaud, E. J., Ottlik, B., Turnbull, J. (2026). There Will Be a Scientific Theory of Deep Learning. arXiv 2604.21691, no venue. https://arxiv.org/abs/2604.21691 (read in full for this spec: §2.1–2.5). It frames the strands; the primary sources carry each claim.

**Learning theory: solvable settings, limits, laws**
- Ali, A., Kolter, J. Z., Tibshirani, R. J. (2019). A Continuous-Time View of Early Stopping for Least Squares. AISTATS 2019 (PMLR 89). https://arxiv.org/abs/1810.10082. Theorem 1 bounds gradient flow's risk at most about 1.69 times ridge's; the arXiv abstract's "no less than" reverses it. The prediction-risk result holds "in an average sense over the underlying signal β₀" (abstract).
- Belkin, M., Hsu, D., Ma, S., Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. PNAS 116(32):15849–15854. https://doi.org/10.1073/pnas.1903070116
- Bartlett, P. L., Long, P. M., Lugosi, G., Tsigler, A. (2020). Benign overfitting in linear regression. PNAS 117(48):30063–30070. https://doi.org/10.1073/pnas.1907378117
- Canatar, A., Bordelon, B., Pehlevan, C. (2021). Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications 12:2914. https://doi.org/10.1038/s41467-021-23103-1 (intro verified by the review; the exact definition of C(ρ) not yet read)
- Chizat, L., Oyallon, E., Bach, F. (2019). On Lazy Training in Differentiable Programming. NeurIPS 2019. https://arxiv.org/abs/1812.07956
- Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., Talwalkar, A. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR 2021. https://arxiv.org/abs/2103.00065 (added in revision; abstract read)
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
- Novak, R., Xiao, L., Hron, J., Lee, J., Alemi, A. A., Sohl-Dickstein, J., Schoenholz, S. S. (2019). Neural Tangents: Fast and Easy Infinite Neural Networks in Python. https://arxiv.org/abs/1912.02803 (software; a test-only dependency)
- Patil, P., Du, J.-H. (2023). Generalized equivalences between subsampling and ridge regularization. NeurIPS 2023. https://arxiv.org/abs/2305.18496
- Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y., Courville, A. (2019). On the Spectral Bias of Neural Networks. ICML 2019 (PMLR 97). https://arxiv.org/abs/1806.08734
- Saxe, A. M., McClelland, J. L., Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR 2014. https://arxiv.org/abs/1312.6120
- Simon, J. B., Dickens, M., Karkada, D., DeWeese, M. R. (2023). The Eigenlearning Framework: A Conservation Law Perspective on Kernel Regression and Wide Neural Networks. TMLR 2023. https://arxiv.org/abs/2110.03922
- Wu, D., Xu, J. (2020). On the Optimal Weighted L2 Regularization in Overparameterized Linear Regression. NeurIPS 2020. https://arxiv.org/abs/2006.05800
- Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., Gao, J. (2021). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. NeurIPS 2021. https://arxiv.org/abs/2203.03466
- Yun, C., Krishnan, S., Mobahi, H. (2021). A Unifying View on Implicit Bias in Training Linear Neural Networks. ICLR 2021. https://arxiv.org/abs/2010.02501 (cited only for its abstract's weighted ℓ1/ℓ2 result on orthogonally decomposable networks)

**Feature learning, interactions and hidden combinations**
- Abbe, E., Boix-Adserà, E., Misiakiewicz, T. (2022). The merged-staircase property. COLT 2022 (PMLR 178). https://arxiv.org/abs/2202.08658 (cited only for what two-layer networks on binary inputs can learn)
- Bietti, A., Bruna, J., Sanford, C., Song, M. J. (2022). Learning Single-Index Models with Shallow Neural Networks. NeurIPS 2022. https://arxiv.org/abs/2210.15651
- Chipman, H. (1996). Bayesian variable selection with related predictors. Canadian Journal of Statistics 24(1):17–36. https://doi.org/10.2307/3315687 (added in revision; checked against Crossref)
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
- Jeffares, A., Curth, A., van der Schaar, M. (2024). Deep Learning Through A Telescoping Lens. NeurIPS 2024. https://arxiv.org/abs/2411.00247 (its abstract does not state the outlying-rows claim; checked in full before any card uses it)
- Kadra, A., Lindauer, M., Hutter, F., Grabocka, J. (2021). Well-tuned Simple Nets Excel on Tabular Datasets. NeurIPS 2021. https://arxiv.org/abs/2106.11189
- McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Feuer, B., Hegde, C., Ramakrishnan, G., Goldblum, M., White, C. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2305.02997 (abstract re-read: "GBDTs are much better than NNs at handling skewed or heavy-tailed feature distributions and other forms of dataset irregularities")
- Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., Hutter, F. (2022). Transformers Can Do Bayesian Inference. ICLR 2022. https://arxiv.org/abs/2112.10510
- Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. ICML 2004, p. 78. https://doi.org/10.1145/1015330.1015435
- Probst, P., Boulesteix, A.-L., Bischl, B. (2019). Tunability: Importance of Hyperparameters of Machine Learning Algorithms. JMLR 20(53):1–32. https://arxiv.org/abs/1802.09596
- Wainberg, M., Alipanahi, B., Frey, B. J. (2016). Are Random Forests Truly the Best Classifiers? JMLR 17(110):1–5. https://jmlr.org/papers/v17/15-374.html

**Explanation**
- Apley, D. W., Zhu, J. (2020). Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models. JRSS-B 82(4):1059–1086. https://doi.org/10.1111/rssb.12377
- Breiman, L. (2001). Statistical Modeling: The Two Cultures. Statistical Science 16(3):199–231. https://doi.org/10.1214/ss/1009213726 (added in revision for the term "Rashomon effect"; checked against Crossref)
- Chang, C.-H., Tan, S., Lengerich, B., Goldenberg, A., Caruana, R. (2021). How Interpretable and Trustworthy are GAMs? KDD 2021. https://arxiv.org/abs/2006.06466
- D'Amour, A., et al. (2022). Underspecification Presents Challenges for Credibility in Modern Machine Learning. JMLR 23(226):1–61. https://arxiv.org/abs/2011.03395
- Fisher, A., Rudin, C., Dominici, F. (2019). All Models are Wrong, but Many are Useful. JMLR 20(177):1–81. https://arxiv.org/abs/1801.01489 (Rashomon sets and model reliance)
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics 29(5). https://doi.org/10.1214/aos/1013203451
- Friedman, J. H., Popescu, B. E. (2008). Predictive learning via rule ensembles. Annals of Applied Statistics 2(3):916–954. https://doi.org/10.1214/07-AOAS148
- Ludwig, J., Mullainathan, S. (2024). Machine Learning as a Tool for Hypothesis Generation. Quarterly Journal of Economics 139(2):751–827. https://doi.org/10.1093/qje/qjad055
- Tsang, M., Cheng, D., Liu, Y. (2018). Detecting Statistical Interactions from Neural Network Weights. ICLR 2018. https://arxiv.org/abs/1705.04977
- Williamson, B. D., Gilbert, P. B., Simon, N. R., Carone, M. (2023). A General Framework for Inference on Algorithm-Agnostic Variable Importance. JASA 118(543):1645–1658. https://doi.org/10.1080/01621459.2021.2003200

**Missing values, measurement and survey design**
- Ayme, A., Boyer, C., Dieuleveut, A., Scornet, E. (2023). Naive imputation implicitly regularizes high-dimensional linear models. ICML 2023 (PMLR 202). https://arxiv.org/abs/2301.13585
- Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. Neural Computation 7(1):108–116. https://doi.org/10.1162/neco.1995.7.1.108 (cited only for noise added during training)
- Hutcheon, J. A., Chiolero, A., Hanley, J. A. (2010). Random measurement error and regression dilution bias. BMJ 340:c2289. https://doi.org/10.1136/bmj.c2289 (cited for the single-predictor case)
- Josse, J., Chen, J. M., Prost, N., Scornet, E., Varoquaux, G. (2024). On the consistency of supervised learning with missing values. Statistical Papers. https://doi.org/10.1007/s00362-024-01550-4
- Kipnis, V., Subar, A. F., Midthune, D., Freedman, L. S., Ballard-Barbash, R., Troiano, R. P., Bingham, S., Schoeller, D. A., Schatzkin, A., Carroll, R. J. (2003). Structure of Dietary Measurement Error: Results of the OPEN Biomarker Study. American Journal of Epidemiology 158(1):14–21. https://doi.org/10.1093/aje/kwg091 (added in revision; abstract read and checked against Crossref)
- Kish, L. (1965). Survey Sampling. Wiley. (added in revision for the effective sample size under unequal weights; a standard text, not re-read here)
- Le Morvan, M., Josse, J., Scornet, E., Varoquaux, G. (2021). What's a good imputation to predict with missing values? NeurIPS 2021. https://arxiv.org/abs/2106.00311
- Luijken, K., Groenwold, R. H. H., Van Calster, B., Steyerberg, E. W., van Smeden, M. (2019). Impact of predictor measurement heterogeneity across settings on the performance of prediction models. Statistics in Medicine 38:3444–3459. https://doi.org/10.1002/sim.8183
- Van Ness, M., Bosschieter, T. M., Halpin-Gregorio, R., Udell, M. (2023). The Missing Indicator Method: From Low to High Dimensions. KDD 2023, pp. 5004–5015. https://doi.org/10.1145/3580305.3599911

**Preprints, watched and never built on**
- Allerbo, O., Schön, T. B. (2026). A Rigorous, Tractable Measure of Model Complexity. https://arxiv.org/abs/2605.21167
- Beaglehole, D., Holzmüller, D., Radhakrishnan, A., Belkin, M. (2025). xRFM: Accurate, scalable, and interpretable feature learning models for tabular data. https://arxiv.org/abs/2508.10053
- Curth, A., Jeffares, A., van der Schaar, M. (2024). Why do Random Forests Work? Understanding Tree Ensembles as Self-Regularizing Adaptive Smoothers. https://arxiv.org/abs/2402.01502
- Kanoh, R. (2026). Double Descent in Gradient Boosting Decision Trees via Split-Candidate Scaling. https://arxiv.org/abs/2608.03111
- Le Morvan, M., Varoquaux, G. (2024). Imputation for prediction: beware of diminishing returns. https://arxiv.org/abs/2407.19804
- Rauniyar, S. (2025). Jacobian Aligned Random Forests. https://arxiv.org/abs/2512.08306
- Sergazinov, R., Wu, J., Yin, S.-A. (2025). Random at First, Fast at Last: NTK-Guided Fourier Pre-Processing for Tabular DL. https://arxiv.org/abs/2506.02406
- Ye, H.-J., Liu, S.-Y., Cai, H.-R., Zhou, Q.-L., Zhan, D.-C. (2024). A Closer Look at Deep Learning Methods on Tabular Datasets. https://arxiv.org/abs/2407.00956
- Zhang, Q., Tan, Y. S., Tian, Q., Li, P. (2025). TabPFN: One Model to Rule Them All? https://arxiv.org/abs/2505.20003

**Removed in revision:** Cortes, Mohri & Rostamizadeh (2012). Kernel-target alignment is a single scalar, not the per-pattern curve shown, so it no longer names any readout.

---

## What changed after review

Two reviews read draft 1: a methods review, which checked the citations on arXiv and against the theory source, and an engine review, which checked the code citations. Both found the structure sound and the leash respected. Every finding was checked against its source; the ones applied are below, and the parts not applied are in the next section.

**Blockers.**
- **Alignment no longer counts noise as signal** (§2.1). Draft 1 divided each pattern's power by ‖y‖², which caps the curve at the in-sample R² and lets noise along hundreds of narrow patterns outweigh real signal. Each pattern's power now has a noise floor subtracted, the curve is normalized by what the inputs can explain, it stays silent below a permutation floor, and its thresholds are set on the reference journeys before any card fires. The kernel-target alignment name is dropped, and "lower error at every sample size" is softened to the source's claim.
- **The hypothesis no longer leaks into BBC-CV** (§2.2, question 2). Draft 1 let a term found on the final fit join the corrected comparison. RECIPES has no path for it, and RECIPES §3.5 already blocks that kind of leak. Question 2 now recommends no new candidate in v2.0, with the §3.5 routing and "Discover inside each fold" as the stated alternative.
- **The shelf no longer deadlocks against the Confirm sweep** (C4). It waits only for Decide answers before the families question, ranks on the defaults in force with a label, and re-ranks when one changes. Readiness is a predicate from the stage registry, not `Stage.requires`. Question 4 asks to move scales, batch and the omics normalization ahead of the families question.
- **The profile reads the recipe the family would actually receive** (C4). Selected families use their current recipe, a "Try both" slot is profiled per option, and re-assessment moved to the `set_recipe` preview (MC-6). The purity argument is gone.

**Methods.**
- **Multiple imputation reads the outcome under inference.** It is declared outcome-reading, the profile uses an outcome-free fill and the blank mask, and the permutation test gains a multiple-imputation fixture (C4).
- **Steps declare whether they read the outcome**, so the profile continues past count-only steps and stops, labeled, at the first value-reading one (C4).
- **The hidden-combination trigger now reads only what the additive benchmark cannot carry** (§2.2). It uses the local-effect deviations, which are exactly zero for any additive model whatever the correlation, on per-SD inputs, with a row floor and the correlated-additive and unit-change fixtures.
- **Deep-learning results are no longer carried over to trees** (§0, §2.2). Multi-index framing is reserved for neural and kernel families, with Ghorbani et al.'s covariate condition. A tree win is described as predictors acting together.
- **The same-rows effect question** carries "selected among N candidates; its p-value and interval are not valid", and the noticing counts as an estimate shown, so a later Estimate track locks as declared after estimates were seen (§2.2).
- **The measurement-error card** is restricted to an Estimate track with one declared error-prone exposure and a stated reliability, and says correlated errors can push either way (Kipnis et al. 2003). Bishop (1995) now covers only noise added in training (C7, §2.4).
- **Invariances are four nested declarations, probed two-sided** after the family's scaler. Unpenalized linear and GLM families and Huber declare invariance to all linear maps. MLPs are probed in distribution, and Adam-trained ones declare nothing unproven (C3, C13).
- **Gradient flow is placed on the shrink line for linear least squares only**, with the exact path as its test and Theorem 1 cited instead of the abstract (C7, C13, §2.3).
- **Effective rank is no longer read as concentration.** The plain sentence uses the participation ratio, r₀ stays a quiet measure, and df(λ) is stated against rank(Z) (C4, §2.3, §2.4).
- **Bootstrap soundness is an equivalence test**: one-sided, with a stated margin, null, signal, clustered and p ≫ n fixtures, fixed seeds, and a family-wise false-fail rate (C11).
- **Agreement across families is not called universal**, and the card says agreement is not evidence (C10).
- **Smaller items:** the refit band is captioned as resampling; the Rashomon card is Predict-only at 50 refits; the solvable settings are split, given two stages or made multi-output; the main-effects rule cites effect heredity; the spectrum reading states its no-alignment assumption; Kish's effective size and the weighting rule per purpose are added; the fill note is restricted to a mean fill and loses "larger than the tuned one"; regime is measured, never declared.

**Engine and sequencing.**
- **Ridge's formulas** are declared in the estimator's parameterization (`alpha` = nλ), held as callables and tested with the recorded λ (C7).
- **The curves** use a grid from observed values, ALE over observed rows, inputs ranked without SHAP, refits without SHAP, and a raw scale per task, with ordinal and time-to-event outcomes drawn (C10, MC-7).
- **The no-switch test reads the syntax tree.** The switch count rose from twelve to eighteen, with allowed exits, and the causal forest key is renamed (C1, §3.3).
- **The trunk stage** keeps the lineage, matrix file and warnings in `design`, so the export, replay and screens are untouched (MC-3).
- **Concerns stay strings,** with a parallel `terms` list, so no API breaks (C4).
- **New packages:** a diagnostics registry (MC-18), the soundness runs outside CI (MC-19), and the shelf card's interface (MC-17).
- **The sequencing circle is broken.** MC-1 lands without recipe members, only the four switches the new families hit go before them, and the fold-in gate lands early with expected failures.
- **The total is recounted:** 66.5 units, about 62 after the overlaps are netted.

**Copy and calm.** Directions became "patterns", defined in place. Ng's bound reads "in the worst case" and sits under "More angles". "Known as" appears on focus only, one name per element. The "Looked at" line is shared. Alignment moved behind "More angles". The forest, imputation and Kobak phenomena now have distinct names.

**Citations.** The theory source is marked a preprint, and the claims on it now cite their primary sources. Cohen et al. 2021 is cited directly. The venue-less entries moved to the preprint list. Breiman (2001) is credited for "Rashomon effect", and Kipnis et al. (2003), Chipman (1996) and Kish (1965) are added. McElfresh et al. now back irregularity only, and Jeffares et al. wait for a check before reaching a card.

## Review notes not taken

- **Alignment's normalization by ‖P_Z y‖² (methods review, §2.1 fix 2) is not taken as written.** With p ≥ n the column space holds all of y, so that normalization alone removes no noise. The noise floor does that work, and the curve is normalized by the floored total. The review's other §2.1 fixes are applied.
- **Sliced inverse regression is removed, not kept as "consistent with"** (methods review). The readout now measures the non-additive part, and SIR estimates every direction the outcome depends on, additive ones included, so it does not check this readout. The review's conditions on when SIR is uninformative are kept with the proposal for v2.x (§2.2, "Not in v2.0").
- **The AGOP trigger is fixed by a different route** than the three the methods review offered: projecting out the benchmark's direction, comparing with its AGOP, or requiring rank two. The local-effect deviations remove the additive part exactly, which serves the same purpose without depending on the benchmark's own fit.
- **"Gradient flow reaches OLS from any initialization where it converges"** (methods review, solvable settings) is taken with a qualifier. Deep linear networks have saddles, so the row reads "whenever gradient flow reaches a global minimum".
- **Ali et al.'s Theorem 1 constant, 1.6862,** comes from the methods review. This revision could not read the PDF, so the spec quotes "about 1.69 (Theorem 1)". The exact path, not the bound, is now the test.
- **The engine review's plain sentence "Your 412 columns carry about as much separate information as 31 unrelated ones"** is not used as worded. It paired the plain reading with r₀, which the methods review showed misreads concentration. The sentence keeps its form and uses the participation ratio.
- **"The spec states which option drives the score"** (engine review, recipes) is answered with the lower of the two assessments, not a chosen option, because the shelf cannot know which option the folds will pick.
- **Canatar et al.'s exact C(ρ) and Ludwig & Mullainathan's held-out procedure** were not confirmed by either review, and this revision could not read them either. The spec cites Canatar's definition only after MC-8 reads it. It attributes to Ludwig & Mullainathan only the separation of generation from testing, and grounds the held-out test's validity in its own logic.
- **Rashomon's 50-refit floor** is taken as the methods review proposed, but marked a convention. It costs 50 refits per family, so it runs only behind "Check it across refits", with its minutes shown.

## Amended while building MC-1 and MC-2a (2026-10-09)

The builder and the verifier of MC-1 and MC-2a found these places where the text above and the code had to differ. Each is applied in the text above.
- **`inference_decl`, not `inference`.** `inference` is already the family's table method, which §1 makes a declared member, so C2's declaration takes the other name.
- **Boosted trees declare a description-only table.** §3.1 said "none". Under inference they give curves and no coefficients, which is C2's `description_only`, as §3.2 declares for the forest and XGBoost.
- **The methods text reads `methods_label(task)`.** `describe(task, purpose)`'s label names the model step ("Histogram gradient boosting"), not what the methods register calls the family ("gradient-boosted trees").
- **`bootstrap_optimism` may be None,** for a family that makes no predictions. The feature-wise tests declare it, and the methods reference prints "not applicable" (C11). Their curve shape stays the default, `any`, which the curve-shape test does not check.
- **`same_kind_as` is read by the shelf now** (`models/base.py:assessment`), with RECIPES §2.2's floor of 0.5 held as one constant beside it, not in each declaration. A family it names must model every task of the family that reads it, and must not read another family itself.
- **`cost_model`** is a family member until RT-1's tuning declaration can hold the cost model (C6). `TuningDecl` and its `Dimension` are declared in `models/tuning.py`, with `structural` and C6's checks made when a declaration is created; RT-1 builds the engine that reads them, and RT-2 gives each family its `tuning`.
- **A knob names the estimator's own parameter,** so the elastic net's are its grids, `alphas` and `Cs`, from which its inner cross-validation chooses `alpha_` and `C_`.
- **The no-switch test pins each listed place** to the family keys and classes it names and its switch count, so a switch added inside a listed function fails as a new place would.
