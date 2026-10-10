Planned 2026-10-10 on turbotab-next @ c49b05f9 with E1c read from feat/wave-e1c-int; line numbers marked E1c refer to that branch.

# Plan for the next engine wave: C6a, MC-2b and the four quest-log items

This was read-only. I edited, committed and branched nothing. I ran one small test file, `acceptance/test_mc2_no_family_switches.py`: 6 passed and 37 xfailed in 4 s. That is 36 `NOT_YET` places holding 52 switches, plus the `has_design_estimator` xfail.

Line numbers are on `turbotab-next` @ c49b05f9 unless marked "E1c", which means `feat/wave-e1c-int` read with `git show`.

---

## 1. Already done (do not rebuild)

**Tuning declarations (MC-1)**
- `models/tuning.py` already holds the declarations only:
  - `Dimension` at 21–32;
  - `TuningDecl` at 35–57, including `structural` at 51;
  - `tuning_problems` at 60–77, which runs from `__post_init__`.
- There is no plan, candidate list, engine or record yet.

**What base.py already declares**
- `same_kind_as` at 698, and `assessment` (622–632) reads it, with `SAME_KIND_FLOOR = 0.5` at 391.
- `cost_model` at 712; `cost.fit_cost` (95–105) reads it.
- `attribution`, `raw_scale`, `architecture`, `output` and `review_lenses` are declared.
- `register_family` checks that a yes/no model step has `decision_function` (555–562).
- `FamilyBase` (659–719) has **no** `tuning`, `defaults_version` or `recipe` members.

**MC-2a: four switches retired**
- `explain.model_kind` (353–372) reads `attribution` and `raw_scale`. `LINEAR_MODELS` no longer exists, so RT-5b's "Ridge in `explain.LINEAR_MODELS`" is moot: ridge simply declares `attribution="linear"`.
- `cost.fit_cost` reads `cost_model`.
- `voice` reads `methods_label` (voice.py:1262).
- `selection.is_flexible` reads `flexible` (933–935).
- `stages/scales.py` reads `inference_default` (367, 502).

**The findings (F-numbers from RECIPES §1)**

| Finding | Status | Evidence |
|---|---|---|
| F11 | Fixed | `inner_cv.fit_pipeline`'s early-stopping branch (296–318); `tests/test_stopping_rows.py` |
| F12 (RT-4) | Fixed in E1c | `levers.ImbalanceCorrected`'s `stopping_setting` and `fit(X, y, X_val, y_val, groups, order)` (E1c levers.py 447–522); `inner_cv.stopping_setting` (E1c); `tests/test_imbalance_stopping_rows.py`; RECIPES §1 note added by E1c. RT-4 is done. |
| F5 | Partly fixed | Pooled loss with 1e-9 rounding (`elastic_net.lowest_rounded`, 68–81) and the exact path (`PooledElasticNetCV` 92–190, `PooledLogisticRegressionCV` 193–306, `exact_path.py`). **Still open:** the logistic `Cs` grid is fixed (`LOGISTIC_CS`, 47) rather than scaled per row; no survey-weighted inner loss; one absolute grid on the fit's rows rather than each split's own λ_max. |
| F13 | Open | `_stops_early` (215–222) reads each fit's own `len(y)` against `EARLY_STOPPING_ROWS = 10_000` (31). `elastic_net.inner_folds(n_rows)` (64–65) is read at build time from `n_train` (modeling.py:587), so it is per design but sized on training rows, not n_plan. `pinned_to_full_fit` (3276–3291) pins only early stopping, and only for bands. |
| F15 | Open | `fit_pipeline(..., seed=0)` takes no design (271–272). The fit closure (modeling.py:1714–1718) and `evaluation.fit_rows` (224–226) pass no seed and no design. |
| F2 | Open | `boosted_trees.build` (75–82) uses scikit-learn's defaults; `seed_policy` is "random_state 0, fixed" (47). |
| F4 | Open | The omics screen is fit before the elastic net's inner cross-validation (`omics.py` ~1573–1635). |
| F9 | Open | `causal.make_learner` draws `KFold(5, shuffle)` by row (407), and `L1LogisticCV` draws `StratifiedKFold` (377). |
| F8 | Partly fixed | `explain.attributions` (389) and `tree_structure` (948–990) read HGB internals (`hgb_ensemble` 189–198 and `_predictors` at 954). |
| F6, F7 | Open | P0.8 (RT-8): `cost.time_one_fit` (82–92) and `_estimates` (modeling.py:315). |

**Other pieces already in place**
- The causal learner keys are renamed: `causal.py:79–80` (`nuisance_forest`, `untuned_boosted_trees`), `decisions.py:1479` (Tombstone) and `decisions.py:1509`. RT-11 is now only the folds.
- Fit records the lock: `service.press_fit` (1638–1671).
- The hold exists: `fit_press.HOLD_SECONDS = 120` (51), read by `holds` (210).
- The stage registry exists: `quest.py`, `QUEST_VERSION = 3` (97).
- `formulas.ridge_shrinkage` exists (43–54), but `FORMULAS` (57–59) has no ridge key.
- `validation.loss_rows` (690) and `PER_ROW_LOSSES` (679) exist, as does `design_cv.design_folds` (58).
- `test_explain.py:300` already checks v2's TreeSHAP against the `shap` package. Use it as the pattern for T11.

**Dependencies**
- `requirements.txt:14–15` already installs `lightgbm>=4.3` and `xgboost>=2.1` at runtime.
- `shap` is test-only (`requirements-dev.txt:7`).
- `constraints.txt` pins `xgboost==3.4.1` (102), `shap==0.53.0` (84), `threadpoolctl==3.7.0` (90) and scikit-learn 1.9.1 / scipy 1.18.1 (82–83).
- **numba and llvmlite are not pinned, although shap requires them.**
- CI fast and full tiers install runtime + dev with constraints (`v2.yml:70–71`, `v2-full.yml:59–60`). Docker installs `requirements.txt` only (`deploy/Dockerfile:35–36`).

---

## 2. The interface that must land first (RT-1a)

### `models/tuning.py`: public surface

Everything down to `choose` is RT-1a (types and pure functions). From `Head` on is RT-1b (the engine), declared here so the family packages can code against it.

```python
TuningKind = Literal["none", "path", "search"]
Scale = Literal["log", "linear", "int", "log_int", "choice", "share_of_units"]
StrategyName = Literal["sobol"]          # seam guard 3: v2.x adds names, never renames
Mode = Literal["automatic", "lighter", "standard", "manual"]     # what SetTuning (RT-6) records
Loss = Literal["mse", "log_loss", "rps"]  # validation.PER_ROW_LOSSES' strictly proper primaries
SizeUnit = Literal["units", "events", "rarest_class"]
SMALL_N, MID_N, STOP_FROM_N, STANDARD_STOP_ROWS = 300, 1_000, 1_500, 10_000
SEARCHED_K, STOP_SHARE, PER_FOLD_FLOOR, LOSS_PRECISION = 3, 0.1, 2, 1e-9

@dataclass(frozen=True)
class Dimension:                          # §4.1's fields unchanged; three defaulted fields added
    name: str; label: str; term: str; low: float; high: float; scale: Scale
    choices: tuple = (); source: str = ""
    points: int = 0                       # a path grid's points (0 for a searched dimension)
    active: Literal["always", "without_early_stopping"] = "always"   # number of trees / rounds
    tunability: float | None = None       # C6 (Probst et al. 2019), where measured

# TuningDecl: unchanged. early_stopping documented as
#   {"param", "rounds", "patience_param", "patience", "share"}

@dataclass(frozen=True)
class PathFit:          # one split's whole grid, from a path family's `path` member
    values: np.ndarray        # (M, G) penalties used on these rows
    coefs: np.ndarray         # (M, G, K, p)
    intercepts: np.ndarray    # (M, G, K)

@dataclass(frozen=True)
class Candidate:
    index: int                                   # ties go to the lower
    values: Mapping[str, Any]                    # unit-free: shares stay shares; symbolic standards kept
    options: Mapping[str, str] = field(default_factory=dict)   # "Try both" slot -> option (RT-3)
    standard: bool = False

@dataclass(frozen=True)
class FitDesign:        # fixes F15: the population answer, aligned with one fit's rows
    strata: np.ndarray | None = None; psu: np.ndarray | None = None; weights: np.ndarray | None = None
    def take(self, rows: np.ndarray) -> "FitDesign": ...

@dataclass(frozen=True)
class TuningPlan:
    family: str; kind: TuningKind; task: str; loss: Loss
    strategy: StrategyName = "sobol"; mode: Mode = "automatic"
    n_plan: int = 0; unit: SizeUnit = "units"
    plan_rows: int = 0                   # rows of one outer training fold (the standard 10,000 rule)
    inner_k: int = 0                     # after the floor; 0 = no inner folds in any fit (§4.6)
    early_stopping: bool = False         # Sobol candidates, decided once (n_plan >= 1,500)
    standard_stops: bool = False         # standard candidates: plan_rows > 10,000
    stop_share: float = STOP_SHARE
    split_seed: int = 0
    seed: int = 0                        # derive_seed(split_seed, family, space_version)
    space_version: str = ""; defaults_version: str = "1"
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()   # empty until RT-3
    manual: Mapping[str, Any] = field(default_factory=dict)
    candidates: tuple[Candidate, ...] = ()
    out_of_bag: bool = False; weighted: bool = False; imbalance: bool = False; threads: int = 1
    def fits(self) -> int: ...                  # F per outer fit: STRATEGIES[self.strategy].fit_count(self)
    def to_dict(self) -> dict[str, Any]: ...    # canonical JSON-safe (no MappingProxy: it must pickle)
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TuningPlan": ...

class Strategy(Protocol):
    name: str
    def candidates(self, decl, *, task, n_plan, seed, mode, manual, options, early_stopping,
                   searched_size) -> tuple[Candidate, ...]: ...
    def fit_count(self, plan: TuningPlan) -> int: ...
    def order(self, plan: TuningPlan) -> Sequence[int]: ...    # evaluation order
STRATEGIES: Mapping[str, Strategy]                  # {"sobol": ...}; keyed by strategy, never by family

def derive_seed(*parts) -> int: """sha256 of canonical JSON -> 32-bit int; never hash()."""
def tuning_for(family, task) -> TuningDecl | None: """The family's declaration for `task`."""
def effective_size(task, y, units=None) -> tuple[int, SizeUnit]: """§4.2 n_eff (rarest class by unit's modal class)."""
def plan_size(task, y, units, *, folds, order=None) -> tuple[int, int, SizeUnit]: """(n_plan, plan_rows, unit)."""
def inner_k(kind, n_plan, *, rarest=None, psus=None) -> int: """3 searched / inner_folds(n_plan) path, capped at floor(m/2); 0 below 2."""
def sobol_sample(d: int, n: int, seed: int) -> np.ndarray: """qmc.Sobol(d, scramble=True, rng=seed).random_base2(log2 n)."""
def map_unit(dim: Dimension, u: float, *, n_plan: int) -> Any: """One documented scale (T2(b) recomputes it)."""
def make_plan(family, *, task, loss, n_plan, plan_rows, unit, split_seed, rarest=None, psus=None,
              mode="automatic", manual=None, options=(), out_of_bag=False, weighted=False,
              imbalance=False, threads=1) -> TuningPlan | None: """The one plan; None: nothing to tune."""
def pooled_loss(task, loss, y, prediction, *, classes, weights=None) -> float: """sum(w * loss_rows) / sum(w) over every inner validation row."""
def choose(losses, precision=LOSS_PRECISION) -> int: """lowest_rounded, moved here (elastic_net re-exports it)."""

@dataclass(frozen=True)
class PathCurve:
    names: tuple[str, ...]; grid: np.ndarray; losses: np.ndarray; chosen: tuple[int, ...]; at_edge: bool

@dataclass(frozen=True)
class TuningRecord:                  # `tuning_` on a fitted TunedPipeline
    plan: TuningPlan
    libraries: Mapping[str, str]; threads: Mapping[str, int]
    order: tuple[int, ...]           # candidates in evaluation order (V2X row 2)
    losses: tuple[float | None, ...] # every candidate's pooled inner loss (V2X row 4)
    chosen: int
    chosen_params: Mapping[str, Any] # the estimator's own parameters at the refit
    chosen_options: Mapping[str, str]
    path: PathCurve | None
    inner_k_used: int; below_floor: bool
    n_fits: int; seconds: float

# ── RT-1b: the engine ──
@dataclass
class Head:          # one fit's steps before the model, fit on its rows less the stopping units
    fitted: Any | None; Z: Any; y: np.ndarray; groups: Any; order: Any
    Z_stop: Any | None; y_stop: np.ndarray | None

def fit_head(pipeline, X, y, rows, *, groups=None, order=None, seed, stop_share, classify) -> Head
def fit_model(model, head, *, seed) -> Any               # nested recalibration cv redrawn on head rows
def fit_parts(pipeline, X, y, rows, *, groups=None, order=None, design=None, seed,
              stop_share=None) -> Any                    # §4.3 steps 1–4; rows stay an argument (V2X row 1)
def inner_splits_for(plan, X, y, rows, *, groups=None, order=None, design=None, seed)
    -> list[tuple[np.ndarray, np.ndarray]] | None        # units / forward chain / GroupKFold(stratum×PSU) / keys

class TunedPipeline(Pipeline):
    def __init__(self, steps, *, search: TuningPlan | None = None, transform_input=None,
                 memory=None, verbose=False): ...
    def fit(self, X, y, *, groups=None, order=None, design: FitDesign | None = None, **params): ...
        # always searches when `search` is set; search=None behaves as Pipeline (slicing)
    def at(self, chosen: Candidate | Mapping[str, Any], *, units: int | None = None) -> Pipeline: ...

def center(plan) -> Candidate        # midpoints, first option of each slot (RT-8 timing)
@contextmanager
def cancel_scope(check: Callable[[], bool]): ...   # checked before every candidate fit; raises jobs.Cancelled
@contextmanager
def observing(callback): ...         # test seam: every (train, validation, stopping) set drawn
```

### Edits at the boundary

- **`inner_cv.fit_pipeline`** becomes `fit_pipeline(pipeline, X, y, *, groups=None, order=None, design=None, seed=0)`. It dispatches a `TunedPipeline` first; the plain path calls `fit_parts`.
- **`pipeline.DesignSpec`** gains `plans: dict[str, dict] | None = None`. `build_pipeline(..., *, for_timing=False)` wraps a family whose `tuning_for(task)` is not None in a `TunedPipeline` from `spec.plans`, and raises `MissingPlan` when the plan is absent (unless `for_timing`). Then none of the 7 call sites can silently skip the search: modeling.py:587, secondary.py:142, effects.py:765, interaction.py:801, evaluation.py:250 and :636, cost.py:155.
- **Wrappers** declare `inner_param = "model"` (one attribute on `ImbalanceCorrected`), so candidate settings reach the wrapped estimator.

### What base.py must gain (RT-1a owns base.py; nothing else in the wave edits it)

```python
# ModelFamily / FamilyBase / MEMBERS
tuning: TuningDecl | Mapping[Task, TuningDecl] | None = None   # per-task grids (elastic net logistic)
defaults_version: str = "1"
consequence: str            # <= 20 words: the families card line; replaces teaching MODELS options (MC-2b-1)
# OPTIONAL_MEMBERS += four, each None when the family adds nothing:
settings: Callable | None   # (values, *, task, n_units, n_rows, y, plan) -> estimator params for this fit
                            #   (shares to rows, symbolic standards, the leaf cap, XGBoost's hessian scaling)
path: Callable | None       # (Z, y, grid, *, task, weights=None) -> PathFit
trees: Callable | None      # fitted step -> explain.TreeEnsemble (leaf paths + HGB-dtype node tables)
tree_shap: Callable | None  # (fitted step, Z) -> (phi (n,p,K), expected (K,)) | None: compiled TreeSHAP
```

**New `register_family` checks**
- Every dimension, by-hand, `standard` and `fixed` name is an estimator parameter (or is resolved by `settings`).
- Kind `"path"` means `path` is set and every dimension has `points` or `choices`.
- `attribution == "trees"` means `trees` is set.
- `consequence` fits its word budget.
- Each `Dimension.source` resolves in the citation registry. Today nothing checks those strings: `family_source_keys` only sees `Source` objects.

**`explain.TreeEnsemble`** gains `tables` (the node arrays `tree_structure` reads), `rule: "le" | "lt"` (XGBoost uses `<`) and `scale: "margin" | "probability"` (the forest). `attributions` and `tree_structure` read `family.trees` / `family.tree_shap`. `hgb_ensemble` moves to `boosted_trees.trees` and stays exported, because `test_explain.py:300` uses it.

### Each family's `tuning` declaration (shape)

**Boosted trees (RT-5a)**
```python
TuningDecl("search", dimensions=(
  Dimension("learning_rate", ..., 0.01, 0.3, "log"),
  Dimension("max_leaf_nodes", ..., 4, 128, "log_int"),
  Dimension("min_samples_leaf", ..., 2, 200, "log_int"),       # settings() caps at plan units / 20
  Dimension("l2_regularization", ..., 1e-3, 10, "log"),
  Dimension("max_features", ..., 0.3, 1.0, "linear"),
  Dimension("max_iter", ..., 25, 500, "log_int", active="without_early_stopping")),
  standard={"learning_rate": .1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
            "l2_regularization": 0., "max_features": 1., "max_iter": 100},
  standard_source="scikit-learn's defaults",
  early_stopping={"param": "max_iter", "rounds": 1000, "patience_param": "n_iter_no_change",
                  "patience": 20, "share": .1},
  space_version="boosted_trees/1", structural=("loss",), reason=...)
# defaults_version = "2"
```

**XGBoost (RT-5e)**
```python
"search": learning_rate [.01, .3] log; max_depth [2, 10] int;
  min_child_weight [1, 64] log  (settings() multiplies by the mean hessian at the base score);
  subsample [.5, 1]; colsample_bytree [.3, 1]; reg_lambda [1e-3, 100] log;
  n_estimators [25, 1000] log_int (active without early stopping)
standard: XGBoost's defaults (.3, 6, 1, 1, 1, 1, 100)
fixed: {"booster": "gbtree", "reg_alpha": 0, "tree_method": "hist"}; nthread from plan.threads
early_stopping: rounds 2000, patience 50
structural=("booster", "objective"); space_version="xgboost/1"
```

**Random forest (RT-5d)**
```python
"search": max_features [.05, 1] linear; max_samples [.2, 1] linear (bootstrap);
  min_samples_leaf scale "share_of_units", high=.1, low = one unit of the plan
standard symbolic: max_features "default" (sqrt(p) / p/3), min_samples_leaf "default" (10 / 5),
  max_samples None   — resolved by settings() per task
fixed: {"n_estimators": 500, "bootstrap": True}; out_of_bag=True; space_version="random_forest/1"
```

**Ridge (RT-5b)**
```python
TuningDecl("path", dimensions=(Dimension("lambda", ..., 1e-5, 1e2, "log", points=50),),
           space_version="ridge/1")
# settings: alpha = n·λ (Ridge), C = 1/(n·λ) (LogisticRegression, l1_ratio=0)
```

**Elastic net (RT-5f)**, one declaration per task:
- regression: mix `choice (0.1, 0.5, 0.7, 0.9, 0.95, 1)` × `ratio [1e-3, 1] log, points=100` of each split's own λ_max;
- classes: mix `(0.2, 0.6, 1)` × 8 ratios.

**Huber (RT-5c)**
```python
TuningDecl("none", by_hand=(Dimension("t", ..., 1, 3, "linear"),), standard={"t": 1.345},
           standard_source="Huber 1964", fixed={"scale": "MAD", "penalty": "none"},
           space_version="huber/1")
```

**Linear and the other existing families:** `tuning = None`.

---

## 3. Packages

Sizes: S = 1, S–M = 2, M = 3, M–L = 5.5, L = 8. Opus means an Opus builder; "Sonnet + Opus verifier" means plumbing built by Sonnet and checked by Opus.

| Package | Size, tier | Files it owns | Consumes → provides | Acceptance tests and their independent reference |
|---|---|---|---|---|
| **RT-1a** Interface and tree hook | M (3), Opus | `models/tuning.py` (types and pure functions); `models/base.py`; `models/explain.py` (`attributions`, `tree_structure`, `TreeEnsemble`; `linear_equation`'s "shrunk to zero" read from `architecture` rather than `hasattr(alpha_)`); `models/boosted_trees.py` (`trees` member, `consequence`); `tests/acceptance/test_mc1_family_declarations.py`; new `tests/test_tuning_plan.py` | MC-1 → §2's interface | T2(b): candidates against `scipy.qmc.Sobol` with the mapping and the sha256 seed rewritten in the test. T2(e): ties go to the lower index (hand-built losses). T9 and T17 plan arithmetic against hand computations (n_eff, S, K floor, early-stopping switches). Boosted trees' SHAP and tree structure bit-equal before and after the hook; `test_explain.py:300` (the `shap` package) stays green. `register_family` refuses a dimension that is not an estimator parameter. `test_mc1` rewritten as rules (`trees` set iff attribution is "trees"; `path` set iff kind is "path"), with §3.2's four rows added, so no later package edits it. |
| **SRC** Sources | S (1), Sonnet + Opus verifier | `export/data/citations.json`, `export/data/refs.bib`, `models/sources.py` | E1q's records → registry records and SOURCES keys | Records for Huber 1964; Holland & Welsch 1977; Breiman 2001 (forests); Liaw & Wiener 2002; Wright & Ziegler 2017; Probst, Wright & Boulesteix 2019; Probst, Boulesteix & Bischl 2019; Chen & Guestrin 2016; Bergstra & Bengio 2012; Bischl et al. 2023; Cawley & Talbot 2010; Riley et al. 2021; Van Calster et al. 2020; Martin et al. 2021; Kruppa et al. 2014; Josse et al. 2024; Perez-Lebel et al. 2022; Van Ness et al. 2023; Gelman 2008; Ng 2004; Mentch & Zhou 2020; McElfresh et al. 2023; Kobak et al. 2020; Curth et al. 2024. Skip any E1q already added. Reference: `citations_check` against Crossref/DataCite/ISBN; `test_citations` green; the exact `cited_as` strings the families will use are listed in the brief. |
| **MC-2b-1** Registry-read lists | S–M (2), Sonnet + Opus verifier | `reference/catalog.py` (`FAMILY_LENSES`, `lenses_of_family` → `review_lenses`); `reference/methods.py:137`; `reference/packet.py:112, 517`; `teaching/content.py` (`MODELS` options built from the registry); `decisions.py` (`model_families` reads the registry only); `consequence` strings in `linear.py`, `elastic_net.py`, `featurewise.py`, `ordinal.py`, `repeated.py`, `survival.py`, `omics.py` (screened), copied verbatim from today's teaching options; `tests/test_reference.py`, `tests/test_word_budgets.py`, `test_mc1` (lines 82–87), `test_mc2` (4 entries) | RT-1a's `consequence` → new families need no catalog or teaching edits | The 4 places' strict xfails flip. Reference: the tables as they were, copied into the tests (lenses, labels, consequences identical); packets with `--reuse` regenerate identically. |
| **Q-a** Triage dispositions | S (1), Opus | `sweep.py` (`recommend`, E1c 472–499); `repairs.py` (the repair `Family` gains `reparameterizes: bool`; `binary_text` is True) | E1c's triage → findings fixed upstream point to their fix | Rules, in order: blocker; a reparameterization of a column that is not focal (not the exposure, outcome or a modifier) → `no_change` ("doesn't change your numbers here" for the focal estimate); a finding with a repair, or routed to any question (open or answered), → `act_on_it` naming the stage and question; then today's rules. On the design fixture's three items: `sas_zeros` → `act_on_it` (Your data's repair); `pack::dietary::implausible_intake`, with exclusions answered → `act_on_it` (Who's in); `binary_text__gender` → `no_change`, but not when gender is the exposure or a declared modifier. Reference: statsmodels OLS with gender coded both ways gives the same exposure coefficient and SE to 1e-12 and a flipped gender coefficient; with gender × exposure the exposure coefficient changes. |
| **Q-b** Substitution estimand sentence | S–M (2), Opus | `methods/energy.py` (a helper deriving the partner from `describe_model`, 1584–1756, using `_in_place_of` 1531–1534 and the nesting terms 1721–1723); `estimand.py` (`caption`'s `what`, 1121 / E1c 1128; the card consequence 1100 / E1c 1107); `voice.py` (`_set_estimand`, 1689–1691); `plan_previews.py:382` | Adjustment set and `readings.nesting` → one wording in caption, voice, preview and Table 2 | With carb and kcal held: "1 g more sugar in place of other carbohydrate (total carbohydrate and energy fixed)". Without carb: the energy wording, naming the omitted sources (ME-04). Reference: statsmodels identity. In the model reparameterized with `other = carb − sugar`, β′_sugar − β′_other equals the original β_sugar to 1e-10. |
| **Q-c** Quest progress after Fit | S (1), Sonnet + Opus verifier | `quest.py` (`Progress` docstring at E1c 783–790 states the "0 of 0 is complete" rule this replaces; computation at E1c 1574–1581); `QUEST_VERSION` bump; quest tests | → Results counts exhibits | Under Estimate after Fit, Results reads 0 of N, N = the exhibit-bearing stages served (E1c 586–599 map); not complete. Write-up is not reached (progress None) until Results is placed (C7a). Predict is unchanged. Reference: the design's state (`design/quest-models-stage` `screens.ts:52–60`; FOUNDATION §3, §8, §9) and the hand-listed exhibits of the fixture. |
| **Q-d** Near-collinear adjustment noticing | S–M (2), Opus | `materiality.py` (a new predicted instrument `invariance`, "exact by theorem (Frisch–Waugh–Lovell)", **not floored**; the noticing joins `noticings_for`); `materiality_calibration.json` (an entry marked theorem-based, which `calibrate` must preserve); `quest_noticings.json` (reconcile the tier of `shared-collinear-predictors`) | → a K5 pre-Fit noticing | Belsley variance-decomposition proportions (as in `linear.collinearity_concern`, 60–90, without editing `linear.py`). Exposure proportion < 0.5 → band 0, "doesn't change your numbers here", with the disclosure that the other coefficients are adjustment terms. Exact Atwater identity → T1 blocker. Exposure inside the dependency → changes the question (band 2, decided at the adjustment question). Outcome-blind: permuting y changes nothing. Reference: proportions computed by hand from an SVD; statsmodels shows that swapping kcal for kcal − (4P + 4C + 9F + 7A) leaves the exposure's coefficient and SE identical to 1e-10. |
| **RT-1b** The engine (with RT-8's core) | L + S (≈9), Opus | `models/tuning.py` (engine half); `models/inner_cv.py`; `models/pipeline.py` (`DesignSpec.plans`, `build_pipeline`); `stages/modeling.py` (plans made in `design_stage`; the fit, `fit_table`, `refit_resample` and `nested_fit` closures pass `design`; `pinned_to_full_fit` → `at(record.chosen_params)`; `_estimates`); `stages/evaluation.py` (`fit_rows` design); `stages/secondary.py` (passes plans); `methods/levers.py` (`inner_param`); `models/cost.py` (times `at(center)` × `plan.fits()`); `stages/__init__.py` (design and fit version bumps); `tests/test_tuning_engine.py` | RT-1a → the search runs nested in every fit | T3: whole persons, time order, whole PSUs with K = min(3, ⌊P/2⌋), the weighted inner loss computed by hand, the oversample lever. T4: perturbation, holding class counts. T9 engine parts (record counts below-floor fits). T13 (same seed reproduces; another seed changes the candidates; pinned replay). T17 (one plan everywhere, observed through `observing()`). F13: folds of 9,999 and 10,001 rows under one plan stop alike. Cancel within ~2 s. T6 partial: an instrumented fit counter equals `plan.fits()`. Reference: partitions computed by hand from fixture IDs, times and PSUs; scikit-learn `GroupKFold`/`StratifiedGroupKFold` identity; a direct HGB fit at the chosen parameters. `test_heldout_guarantee`, `test_stopping_rows` and `test_imbalance_stopping_rows` must stay green. |
| **RT-5b** Ridge | S (1), Opus | new `models/ridge.py`; `formulas.py` (a "ridge" knob formula); one import line in `models/__init__.py` | interface, SRC → family | T11: numpy closed form (centered, unpenalized intercept, α = nλ) at every grid point to 1e-8. df(λ) against the explicit hat matrix (ESL 3.50). Logistic ridge against scipy L-BFGS on the explicit penalized likelihood to 1e-6. Linear SHAP against the closed form. Rotation-after-scaling keeps predictions; a shear changes them. T2(a) runs in phase 3. |
| **RT-5c** Robust linear (Huber) | M (3), Opus | new `models/huber.py`; one import line | → family | T11: an independent IRLS written in the test from Holland & Welsch (1977), MAD = median\|r\|/0.6745 re-estimated each step, t = 1.345, to 1e-6. R `MASS::rlm(stack.loss ~ ., stackloss)` published coefficients, copied in with their source, to 4 decimals. Linear-map invariance (a log transform changes predictions). SHAP closed form. `purposes=("prediction",)` with no key check. statsmodels' RLM scale is `mad(resid, center=0)`, as MASS uses (checked). |
| **RT-5d** Random forest | L (8), Opus | new `models/forest.py` (wrapper with margin `decision_function` = logit of the clipped probability; `trees` from sklearn's `tree_`, values / n_trees; `tree_shap` via `shap`); `requirements.txt` (shap at runtime, if T11 passes); `constraints.txt` (numba and llvmlite pins); one import line | → family | Predictions exactly equal to a direct `RandomForest*` fit at the same parameters, seed and threads. OOB loss computed by hand from `oob_prediction_` / `oob_decision_function_`. v2's numpy TreeSHAP against `shap.TreeExplainer` (path-dependent) to 1e-6, blanks included. Local accuracy. Monotone-per-column probe. Standard settings asserted against ranger's and randomForest's documentation. |
| **RT-5e** XGBoost | M–L (5.5), Opus | new `models/xgboost_family.py` (early-stopping interface, `stopping_setting()`, `fit(X, y, X_val, y_val)`, margin `decision_function`, labels 0…K−1, safe names, pinned nthread, `same_kind_as`); one import line | → family | Equal to native `xgb.train` with the same stopping rows (same best iteration, exact margins). Equal to direct `XGBRegressor`/`XGBClassifier` at the same nthread. `pred_contribs` against v2's TreeSHAP on converted trees (`<` rule, default-left blanks) to 1e-6. Name round trip. T13 at pinned nthread. Assessment equals boosted trees' − 1.0, floored at 0.5. |
| **RT-11** Causal lasso folds by unit | S (1), Opus | `models/causal.py` (`_Sklearn` and `L1LogisticCV` fit with groups, through `inner_splits`) | → F9 fixed | With 3 rows per person, no person sits on both sides of any lasso fold (a recording splitter). Without repeats, folds and DML estimates are bit-equal to today's. Reference: `LassoCV` with explicit `GroupKFold` folds gives the same α. |
| **MC-2b-2** Inference declarations | M (3), Opus | `models/survey.py` (5 places + `has_design_estimator` → `design_based`); `scales.py`; `methods/interaction.py` (4 → `product_terms`, `raw_scale`); `plan_previews.population_block`; `method_previews.calibration_numbers`, `stages/calibration.calibration_stage` and `quest._linear_family` → one shared predicate; `stages/class_substitution._plain_multinomial`; `estimand._models_fit_the_family`; `test_mc2` | → 17 places + 1 xfail retired | Each predicate pinned to select exactly the families the old switch named (copied from the code). Methods sentences unchanged on the captures. Rewrite `test_mc2`'s probe on `interaction._measure` (`_rescanned`) on synthetic code. |
| **RT-5a** Boosted trees tuned | S (1), Opus | `models/boosted_trees.py` (`tuning`, `settings`, `describe`, `defaults_version = "2"`, `seed_policy`) | engine → tuned family | Below n_plan 300, bit-equal to a direct HGB default fit (scikit-learn identity). T2(c) pinned replay. T1 is slow and scheduled. |
| **RT-5f** Elastic net on the path search | M (3), Opus | `models/elastic_net.py` (`path` kernel through `exact_path`; refit plain at r·λ_max of the fit's rows; per-row logistic grid); `models/wide.py`; `methods/omics.py` (the screen refits per split through the head); `models/explain.py` (`shrinkage_path` reads `tuning_.path`); `tests/test_elastic_net_penalty.py`, `test_wide_data.py`, `acceptance/test_explain.py`, `test_wp4_*`, `test_wp7` pins; `stages/__init__.py` (one bump covering RT-5a and RT-5f) | engine, ridge → F4, F5 fixed | T2(a): the diabetes data against numpy's closed form with the pooled argmin. T2(d): independent `ElasticNet`/`LogisticRegression` refits on the same ratios and splits. Weighted loss computed by hand. T12 is slow and scheduled. The Windows numerics job (`v2.yml:283–291`) re-pinned. |
| **MC-2b-3** Stage switches | S–M (2), Opus | `stages/effects.py` (8 places → `matrix_table`, `diagnostics`, `raw_scale`, `predicts`); `stages/evaluation._shrinkage` (`updating`); `stages/modeling.py` (`_pooled_curve`, `_tests_only`, `fit_stage` → `"collinearity" in diagnostics`); `test_mc2` | → 12 places retired | Pins as above. Rewrite the `evaluation._shrinkage` probe. Table 2 and diagnostics identical on fixtures. |
| **MC-2b-4** Omics places and close | S (1), Sonnet + Opus verifier | `methods/omics.py` (4 places; `model_clause` reads `tuning.kind` and the plan sentence); `test_mc2` (`NOT_YET` becomes `{}`) | → no-switch allowlist at zero | All strict xfails gone; fold-in gate item 2 turns blocking. |

**Shared files, and who holds each**
- `models/base.py`: RT-1a only.
- `test_mc1`: RT-1a, then MC-2b-1.
- `test_mc2`: one MC-2b package per phase.
- `models/explain.py`: RT-1a (phase 1), then RT-5f (phase 3).
- `stages/modeling.py`: RT-1b (phase 2), then MC-2b-3 (phase 4).
- `stages/__init__.py`: RT-1b (phase 2), then RT-5f (phase 3).
- `models/__init__.py`: one import line per family. The integrator orders them as linear, elastic_net, ridge, huber, boosted_trees, random_forest, xgboost, then the rest.
- `requirements.txt` and `constraints.txt`: RT-5d.
- `formulas.py`: RT-5b.
- `sources.py`, `citations.json`, `refs.bib`: SRC.
- **Generated files** (`METHODS_REFERENCE.md`, `review-packets/*.md` with `--reuse`, `openapi.json`, `generated.ts`): regenerated by the integrator after every merge, never hand-merged.

**Totals:** C6a ≈ 34.5 (spec 32: the interface split and RT-8's core are added). MC-2b ≈ 8 (spec 5.5: the census grew from 14 to 36 places). Q-items 6. SRC 1.

---

## 4. Sequencing

Every package waits for **both E1c and E1q** to land.

- **E1c conflicts:** `inner_cv.py`, `levers.py`, `stages/modeling.py`, `explain.py`, `causal.py`, `omics.py`, `effects.py`, `stages/calibration.py`, `interaction.py`, `estimand.py`, `quest.py`, `voice.py`, `plan_previews.py`, `method_previews.py`, `scales.py`, `models/survey.py`, `decisions.py`, `teaching/content.py`, `stages/__init__.py`. E1c also introduces `sweep.py`'s triage and `materiality.py`, which Q-a and Q-d build on.
- **E1q conflicts:** `citations.json`, `refs.bib` (SRC), `reference/catalog.py` (MC-2b-1), plus `METHODS_REFERENCE.md` and the packets.
- Untouched by both: `base.py`, `tuning.py`, `boosted_trees.py`, `elastic_net.py`, `cost.py`, `selection.py`, `metrics.py`, `formulas.py`, `sources.py`, `evaluation.py`, `secondary.py`, `class_substitution.py`, `requirements.txt`.

| Phase | Packages, run in parallel | Why this order |
|---|---|---|
| 1 | RT-1a, SRC, Q-a, Q-b, Q-c, Q-d | Q-items only need E1c landed. |
| 1.5 | MC-2b-1, serial right after RT-1a | It needs `consequence`. It must merge before any family, or each family would have to edit `catalog.py` and `teaching/content.py` (`test_reference.py:67`, `test_word_budgets.py:107`). |
| 2 | RT-1b, RT-5b, RT-5c, RT-5d, RT-5e, RT-11, MC-2b-2 | Families need RT-1a, SRC and MC-2b-1. MC-2b-2's files follow Q-b (`estimand.py`, `plan_previews.py`) and Q-c (`quest.py`). |
| 3 | RT-5a, RT-5f | Both need the engine. RT-5f also needs ridge for T2(a). |
| 4 | MC-2b-3, then MC-2b-4 | MC-2b-3's files follow RT-1b's `stages/modeling.py` and `stages/evaluation.py` edits. MC-2b-4 follows RT-5f's `omics.py`. |

**Hard gate:** RT-5a must not merge before RT-1b's estimate counts `plan.fits()`. `fit_press.holds` (fit_press.py:210) reads the estimate, so without the count a ~30-minute tuned fit would be estimated at about a minute and start live. That is why RT-8's core sits inside RT-1b. F7, the counts across stages, preview estimates and the job payload stay in P0.8.

---

## 5. Risks to brief the builders on

**Leakage**
- `fit_parts` takes its rows as an argument.
- Stopping units are drawn first in every inner fit too.
- One head per inner split is shared by every candidate. In a plan where the standard candidate does not stop early but the Sobol ones do, the standard candidate trains on the split less its stopping rows.
- Inner splits are redrawn inside each outer training fold: whole units; forward chaining by unit under time; `GroupKFold` over stratum × PSU, never stratified within strata.
- Early stopping always draws whole units.
- Under the population answer only the inner loss is weighted; models are fit unweighted.
- XGBoost's mean hessian and the elastic net's λ_max are computed on each fit's own rows.
- The plan reads the outcome's counts (an allowed outcome-count read). So T4 must perturb held-out rows while keeping class counts.
- Out-of-bag scoring needs "no outcome-reading step", but `reads_outcome` only arrives with MC-4. Interim rule, read from `DesignSpec`: no selection step, no levers and no family `preprocess` → out of bag allowed; otherwise inner folds.

**Determinism**
- Every seed comes from `derive_seed` (sha256), threaded from `split.seed`.
- Sobol with `rng=` and `random_base2` (S = 8, 16 and halves are powers of two).
- Pinned `nthread` for XGBoost; forest `n_jobs` for fit, single-threaded predict.
- Threads recorded through `threadpoolctl`. CI sets `OMP_NUM_THREADS=2` and `-n 2`, so tests pass thread counts explicitly.
- Losses rounded before the argmin.
- Keep the elastic net's refit exact (through `exact_path`) even though the final estimator class is plain.
- RT-5f changes elastic-net penalties: each split's own λ_max, n_plan-based K, per-row logistic C. Pinned values (`test_elastic_net_penalty.py`, the WP7 1e-9 pins, the Windows job) must be re-derived. Seeding the selection step and the forms (F15) also changes prediction results with those steps.
- The local `venv` is not on the constraints: scikit-learn 1.9.0, xgboost 3.3.0, shap 0.52 against pinned 1.9.1, 3.4.1 and 0.53.0. Builders run `pip install -c turbotab/server/constraints.txt`.

**Dependencies**
- Making shap a runtime dependency adds numba and llvmlite to Docker and the launchers, and **constraints.txt pins neither**. Re-resolve with `uv pip compile` and run the full tier on a `ci/full-*` branch.
- `import xgboost` has never run in the macOS launcher job. Classic notes libomp may be needed. `register_family` builds the model at import time, so an import failure would break the whole engine. Import lazily and degrade to "XGBoost could not load" rather than crash.

**CI time**
- The acceptance suite runs only in the full tier; the fast "core" shard is about 260 s on a dev machine and 2.5× that on CI.
- Leakage guarantees (T3, T4) belong in `core/tests` so they stay in the fast tier.
- Fast fixtures: n ≤ 400, forests ≤ 50 trees, XGBoost ≤ 50 rounds. Only one smoke test uses 500 trees.
- T1 and T12 are marked `slow` and live in `acceptance/`.

**Tests that will break, and who fixes them**
- `test_mc1`'s exact `sorted(found) == sorted(EXPECTED)` and its `adds` dict: RT-1a.
- `test_reference.py:67` and `test_word_budgets.py:107`: MC-2b-1.
- `test_mc2`'s probes on `interaction._measure` and `evaluation._shrinkage`: MC-2b-2 and MC-2b-3.
- The `alpha_`/`C_` readers (`explain.shrinkage_path`, `test_explain`, `test_wide_data`, `test_wp4`): RT-5f.
- The packets and `METHODS_REFERENCE.md`: regenerated at every merge.
- `test_each_journey_took_what_the_packet_says_the_app_offers_first`: check the shelf ties after the new families' assessments.

**Ambiguities, and the reading I recommend**

*Spec against code or scope*
1. **"Try both" in C6a.** RT-5a's "Try both" default needs RT-2/RT-3 (C6b). C6a's engine carries `options = ()` (s = 1). T9 and T17's "two standard candidates" become strict xfails labeled RT-3. Set `defaults_version "2"` in RT-5a: no versions file exists until RT-7, so no recorded key can collide.
2. **Who adds `tuning`.** The contract's amendment says RT-2 gives each family `tuning`; §9 has RT-5a declare it. RT-1a adds `tuning` and `defaults_version` now; `recipe` waits for RT-2.
3. **`cost_model`.** It stays a family member; RT-8 decides whether it moves into `TuningDecl`.
4. **RT-5b's `LINEAR_MODELS`.** Moot (removed by MC-2a).
5. **RT-11's renames.** Already landed. RT-11 changes folds only when units repeat, so existing DML fixtures stay bit-equal.
6. **`at()` and share settings.** Spec: `at()` returns a plain Pipeline, but share settings rescale per fit. Pinned refits hold the final fit's resolved `chosen_params`; timing resolves `center` for the timing sample.
7. **Plans at stages that build a fresh spec.** "Every fit uses this plan" conflicts with stages that build a new spec (secondary, effects, interaction). Copy `plans`, and have `build_pipeline` refuse a tuned family without one.
8. **MC-2b's size.** It is ≈8, not 5.5 (36 places and 52 switches, not 14).

*Reading the spec itself*
1. **Early stopping, standard against Sobol.** Standard candidates follow scikit-learn's own rule (plan rows > 10,000). Sobol candidates stop early from n_plan ≥ 1,500, up to 1,000 trees with patience 20, and below that the number of trees is searched. The plan therefore needs `plan_rows`.
2. **`share_of_units`.** Its low end ("1 row") is in other units. Map log-uniformly between 1/n_plan and `high`, store as a share, resolve on each fit's rows.
3. **Per-task grids and standards.** The elastic net's logistic mix grid and the forest's mtry and leaf size depend on the task. Use `tuning_for(family, task)`, a per-task mapping, plus symbolic standards resolved by `settings`.
4. **Grid points and conditional dimensions.** §4.1's `Dimension` cannot express path-grid points or the "number of trees only without early stopping" rule. Add `points` and `active`.
5. **"Column share" for XGBoost.** Read it as `colsample_bytree`.
6. **Diagnostics vocabulary.** It lacks out-of-bag and early-stopping keys (MC-18). The forest and XGBoost declare none for now.
7. **Ridge `updating`.** None: uniform calibration-slope shrinkage of a ridge is not sourced.
8. **The calibration predicate.** MC-2b-2's shared predicate must select exactly `{linear}` today; check whether `linear_in_values` holds for proportional odds.
9. **Path families with no inner folds.** At K = 0, `make_plan` raises a stated refusal; the shelf wording is MC-5/RT-13's.

*Quest-log items*
1. **Theorem-backed zeros.** Materiality floors every uncalibrated proxy at band 1 (`band_of`, E1c materiality.py 164–178). Item (d) and Q-a's reference-level rule need an exact "invariance" instrument that is not floored, and `calibrate` must keep it. This amends SURFACING_POLICY §2.6.
2. **Item (d)'s tier.** `quest_noticings.json:286` places `shared-collinear-predictors` as Decide in Models, but band 0 gives For the record. Q-d reconciles them (Decide only when the exposure is in the dependency).
3. **Item (c)'s count.** Count the engine's exhibit-bearing stages, not the design's four. The design lists "unmeasured", "diagnostics" and "appendix", which the engine's map lacks; C7a defines the exhibits.
4. **Item (c)'s rule.** The fix contradicts the documented "0 of 0 is complete" rule in `quest.Progress` (E1c). Follow FOUNDATION §3 and §8.

**The four quest-log items, placed**
- **(a)** Lives in `sweep.recommend`. Today `act_on_it` fires only for a critical finding or one routed to an *open* question, so `sas_zeros` (a repair, no route) and the answered exclusions route both fall through to `could_bias`.
- **(b)** Lives in `estimand.caption`, `voice._set_estimand` and `plan_previews:382`. The correct per-term meaning already exists in `energy.describe_model` and only needs to reach those sentences.
- **(c)** Lives in `quest.quest_log`'s progress, which needs C7a's placement kind to ever complete.
- **(d)** Belongs to noticing family K5, Structure among variables (SURFACING_POLICY.md:371): thread `shared-collinear-predictors`, with `diet-nested-parts` and `shared-compositional-parts` for "a total beside its parts". Materiality is band 0 on the focal estimate by theorem: disposition "doesn't change your numbers here", decided by disclosure, tier For the record, with the other coefficients labeled as adjustment terms. It escalates to band 2 (changes the question) when the exposure is in the dependency, and to a T1 blocker for an exact identity.

---

## 6. Heavy runs to schedule with Nolan

Times are rough estimates for the dev machine.
- **T1** (RT-5a): 200 datasets, flat against nested tuning, with ridge for the corrected estimate, plus 20,000 fresh rows each. About 30–60 minutes.
- **T12** (RT-5f): 50 datasets at n = 120, p = 2,000, with the screen refit per split. About 30–90 minutes.
- **Journey recapture** (RT-13, C6b, but the captures go numerically stale once RT-5a and RT-5f land). Metabolomics prediction took about 27 minutes and survey prediction about 31 today. Tuned at NHANES size, RECIPES' worked example gives about 30 minutes per tuned family, so hours in all. `--reuse` regeneration keeps CI green until then.
- **A budget measurement** at NHANES size for the forest's TreeSHAP branch (v2's numpy TreeSHAP against `shap`): about 5–15 minutes.
- **Local fast tier:** about 17 CPU-minutes across 4 shards. Prefer targeted files locally and the full tier on `ci/full-*`, which runs on GitHub.
- **Light, no scheduling needed:** the Crossref check (about 1 minute, network-bound), packet and methods-reference regeneration with `--reuse`, re-deriving the penalty pins, and `uv pip compile`.

---

## 7. The orchestrator's rulings on this plan (2026-10-10)

1. **Q-a and Q-d are one package (Q-ad).** Both need the exact, theorem-backed "invariance" instrument, and both write in `materiality.py` and `sweep.py`.
2. **SURFACING_POLICY §2.6 is amended by Q-ad.** An instrument that is exact by theorem is not floored at band 1. Examples are Frisch–Waugh–Lovell, and a reparameterization of a column that is not focal. Such an instrument may say "doesn't change your numbers here" without calibration cases. `calibrate` keeps it, and the ledger names the theorem. Every proxy that is not exact keeps the floor.
3. **SURFACING_POLICY §7.2 is amended by Q-c:**
   - A stage's answered count never falls without a Reopened record naming the cause.
   - Its required count may grow as answers make new lines apply within the stage being worked.
   - A complete stage that gains a line has been reopened, and says why.
4. **`consequence` is optional in RT-1a.** MC-2b-1 copies every existing family's wording verbatim, then makes it required in `register_family`.
5. **The split sentence goes into Q-b.** E1c's integration found that a split recorded while the seal plan recomputes is stored with a methods sentence that leaves out the grouping. E1c fixed the test's timing, not the product.
6. **The trials' `MIXED_MODEL` and `GEE_MODEL` constants** pass the MC-2 check only by naming. MC-2b-4 lists the trial core explicitly as not a model family in the check's rules.
7. **Recommended readings.** Every ambiguity in §5 is ruled as the reading this plan recommends.
8. **Phase 1:** RT-1a, then MC-2b-1 on its branch; SRC; Q-ad; Q-b; Q-c. Opus builds everything except SRC, MC-2b-1 and Q-c, which Sonnet builds. Every package gets an Opus verifier.
