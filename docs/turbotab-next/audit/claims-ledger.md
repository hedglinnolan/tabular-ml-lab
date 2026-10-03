> **Provenance.** Copied unchanged from the area G auditor's working file (`/private/tmp/turbotab-audit/G/claims-ledger.md`) on 2026-10-02 for the validation audit (`AUDIT_REPORT.md` in this folder). Finding IDs G1–G20 here are area G's raw IDs; `findings.json` maps each to its deduplicated ID and the skeptic's verdict. After the skeptic pass, G4 and G12 were lowered to minor, G13 was raised to critical, and G10's "order of magnitude" was corrected to 2–4×. Row 88 cites a "G22" that was never filed as a finding (see the report, §7.3).

# Area G: the claims ledger (intelligence layer)

Repository `/Users/nhedglin/tabular-ml-lab`, branch `turbotab-next`, HEAD `bf3db6c`. Read-only audit, 2026-10-02.

Status key:
- **VERIFIED**: a primary source was read and quoted, or the arithmetic was reproduced.
- **CONSISTENT**: matches the primary source, with a caveat noted.
- **OVERSTATED**: the claim has some support, but its wording or badge goes further than the evidence.
- **WRONG**: contradicted by a primary source or by running the code.
- **SELF-CONTRADICTED**: the app's own code or its other claims contradict it.
- **UNSOURCED**: no primary source behind it, or none was found or read in this audit.

Repros are in `/private/tmp/turbotab-audit/G/*.py`. Run them with `PYTHONPATH=. ./venv/bin/python <file>` from the repo root.

## Sources read in this audit

| id | source | how it was read |
|---|---|---|
| S1 | Tomova, Arnold, Gilthorpe, Tennant. AJCN 2022;115(1):189–198. doi:10.1093/ajcn/nqab266 (PMC8755101) | full text, Europe PMC XML (`tomova2022.txt`) |
| S2 | McCullough & Byrd. AJE 2023;192(11):1801. doi:10.1093/aje/kwac071 | WebFetch, two sentences quoted |
| S3 | Banna, McCrory, Fialkowski, Boushey. Front Nutr 2017;4:45. doi:10.3389/fnut.2017.00045. Quotes Willett, *Nutritional Epidemiology* 3rd ed. (2013) | WebFetch. The Willett textbook itself was **not accessible** |
| S4 | Yamamoto, Ejima, Zoh, Brown. eLife 2023;12:e83616. doi:10.7554/eLife.83616 | abstract, Europe PMC |
| S5 | Louie & Bhowmik. Eur J Clin Nutr 2026. doi:10.1038/s41430-026-01712-7 | abstract, Europe PMC |
| S6 | van den Goorbergh, van Smeden, Timmerman, Van Calster. JAMIA 2022. doi:10.1093/jamia/ocac093 | abstract |
| S7 | Shi et al. (MAQC-II). Nat Biotechnol 2010. doi:10.1038/nbt.1665 | abstract |
| S8 | Varoquaux. NeuroImage 2018;180:68–77. doi:10.1016/j.neuroimage.2017.06.061 | abstract |
| S9 | Sperrin, Martin, Sisk, Peek. J Clin Epidemiol 2020. doi:10.1016/j.jclinepi.2020.03.028 | abstract |
| S10 | Sisk, Sperrin, Peek, van Smeden, Martin. Stat Methods Med Res 2023. doi:10.1177/09622802231165001 | abstract |
| S11 | Josse, Chen, Prost, Varoquaux, Scornet. "On the consistency of supervised learning with missing values", arXiv:1902.06931 | abstract, arXiv API |
| S12 | Groenwold et al. CMAJ 2012. doi:10.1503/cmaj.110977 (PMC3414599) | WebFetch, Key points |
| S13 | FDA Bacteriological Analytical Manual, Ch. 3 Aerobic Plate Count (fda.gov/media/178943) | PDF text streams |
| S14 | Keogh et al. (STRATOS Part 1). Stat Med 2020. doi:10.1002/sim.8532 (PMC7450672), §3.1.2–3.1.3 | WebFetch |
| S15 | CDC NHANES tutorials: weighting, variance estimation. DR2TOT_J documentation | WebFetch |
| S16 | Van Calster et al. BMC Med 2019. doi:10.1186/s12916-019-1466-7 | abstract |
| S17 | Collins, Ogundimu, Altman. Stat Med 2016;35:214–226. doi:10.1002/sim.6787 | abstract |
| S18 | Steyerberg et al. J Clin Epidemiol 2001. doi:10.1016/S0895-4356(01)00341-9 | abstract |
| S19 | Scholbeck et al. Data Min Knowl Disc 2024;38:2997–3042. doi:10.1007/s10618-023-00993-x | Crossref metadata (citation only) |
| S20 | Wei et al. "Missing value imputation approach for MS-based metabolomics data" (bioRxiv 171967; Sci Rep 2018) | abstract |
| S21 | Ozarda. Biochem Med 2016. doi:10.11613/BM.2016.001 (PMC4783089) | WebFetch |
| S22 | Ambroise & McLachlan. PNAS 2002. doi:10.1073/pnas.102102699 | abstract |
| S23 | Lachat et al. (STROBE-nut). PLoS Med 2016. doi:10.1371/journal.pmed.1002036 | WebFetch, items nut-12.2 and nut-13 |
| S24 | Chalmers. Educ Psychol Meas 2018. doi:10.1177/0013164417727036 | Crossref abstract |

Key quotes:
- **S1**: "The 'standard model' and the mathematically identical 'residual model' estimate the average relative causal effect … but provide biased estimates even in the absence of confounding."
- **S1, residual model specification**: "regressing the nutrient exposure on total energy, and entering the model residual into a second unadjusted model". The simulation has no other covariates.
- **S1**: "The residual model is algebraically identical to the standard model, and as such suffers the same problems while offering no additional benefits."
- **S1, on the density models**: "the coefficient represents an obscure quantity that conflates both the effect of the nutrient exposure and that of the reciprocal of total energy… We present models with (3a) and without (3b) adjustment for [energy]". Table 2 labels the multivariable nutrient density model "Obscure", and the text says its estimate is "still biased".
- **S1, on total energy**: "Adjusting for TE opens conditional dependencies between the exposure and all competing energy sources".
- **S1**: "Accurate estimates of both the total and average relative causal effects may instead be derived by simultaneously adjusting for all dietary components … the 'all-components model.'"
- **S2**: "A variation on the simple nutrient residual model proposed by Willett and Stampfer includes the nutrient residual plus a term for total energy intake."
- **S3**: "the range of 500–3,500 kcal/day may be applied to data from women", and "an allowable range of 800–4,000 kcal/day for men may be used, as intakes of more than 4,000 kcal/day are unlikely to be true for even relatively active men". The source is ref. 13, Willett 2013.
- **S4**: "Significant bias in simulated associations using self-reported NI was reduced but not completely eliminated by Goldberg cutoffs in 14 of 24 nutrition-outcome pairs; bias was not reduced for the remaining 10 cases… Whether one uses Goldberg cutoffs should therefore be decided based on research purposes and not general rules."
- **S10**: "When missingness is allowed at deployment, omitting the outcome from the imputation model at the development was preferred. Missing indicators improved model performance in many cases but can be harmful under outcome-dependent missingness."
- **S11**: "the widely-used method of imputing with a constant, such as the mean prior to learning is consistent when missing values are not informative. This contrasts with inferential settings where mean imputation is pointed at for distorting the distribution of the data."
- **S12**: "In nonrandomized studies … the missing-indicator method typically results in biased estimates." It adds that in randomized trials it "will give unbiased estimates".
- **S13**: "When number of CFU per plate exceeds 250, for all dilutions, record the counts as too numerous to count (TNTC) for all but the plate closest to 250…", and "Estimate the APC as greater than 100 times the highest dilution plated".
- **S14 §3.1.3**: "the estimated coefficients in model (11) may be larger or smaller than the true target values in a rather unpredictable manner."
- **S15**: "variance estimates computed using standard statistical software packages that assume simple random sampling are generally too low (i.e., significance levels are overstated)". Also "use the weight that is appropriate for the variable of interest…", "dividing the two-year sample weights by the number of two-year cycles" (2001–2002 onward, with a 1999–2002 4-year exception), and "the second interview is collected by telephone 3 to 10 days later."
- **S17**: "externally validating a prognostic model requires a minimum of 100 events and ideally 200 (or more) events."
- **S18**: "split-sample analyses gave overly pessimistic estimates of performance, with large variability… recommend bootstrapping".
- **S23, nut-13**: "Describe the number of missing values, cut-offs for implausible data leading to exclusion, characteristics of those excluded, and any method used to handle missing values."

## Ledger

### `turbotab/core/teaching/content.py` (drawer, why, options, terms)

| # | claim (abridged) | location | badge | status | source / evidence |
|---|---|---|---|---|---|
| 1 | Energy is a strong determinant of nutrient intake; energy adjustment is the field's signature; Atwater 4/4/9/7 | content.py:102-108 | SETTLED | CONSISTENT | S1 background. The Atwater general factors are standard |
| 2 | A reference interval is the central 95% of healthy people and is for annotation only; plausibility bounds are a separate thing | :109-114, :204-208 | CONVENTION | VERIFIED | S21: "2.5th … 97.5th percentile" |
| 3 | Half-minimum is the de facto default for values below detection; it deflates variance | :115-119 | CONVENTION | CONSISTENT | S20 lists HM among the common methods; QRILC is favored for MNAR |
| 4 | p > n: an unpenalized model is degenerate; "Regularization is mandatory" | :120-124, :933-937 | SETTLED | OVERSTATED (minor) | True for least squares and logistic regression. Dimension reduction and tree ensembles are alternatives, so "mandatory" is too strong |
| 5 | Likert ordinal versus interval is disputed; run the other treatment as a sensitivity analysis | :125-129, :318-322 | DISPUTED | CONSISTENT | Not checked against a primary source; the badge is appropriately weak |
| 6 | XCMS/MZmine/MS-DIAL exports are features in rows; MetaboAnalyst's own table is samples in rows | :159-163 | SETTLED | UNSOURCED (minor) | MetaboAnalyst's upload accepts either orientation; not checked here |
| 7 | An expression matrix has genes in rows; an undetected transpose gives a PCA of genes | :164-168 | SETTLED | CONSISTENT | Domain convention |
| 8 | A sentinel 9 on a 1–5 item is a non-answer; recode it, never automatically | :200-203 | SETTLED | CONSISTENT | |
| 9 | **TNTC and QNS are measurement failures, not censoring; treat them as missing** | :209-212; clinical.py:76-79, :135-140; packs.py:5573 | SETTLED | **WRONG for TNTC** | S13: TNTC means the count is above the countable range, reported as an estimate or as ">". That is right-censored, not a failure. QNS is a failure. Finding G8 |
| 10 | Energy is on the causal path to adiposity; conditioning on it can be both over-adjustment and collider bias; present adjusted and unadjusted models | :236-241, :877-880 | DISPUTED | CONSISTENT, but the remedy is broken | Following "present adjusted and unadjusted" gives two identical models, because "No adjustment" keeps energy in the model (G1). No detector surfaces this dispute (G5) |
| 11 | **A flow diagram is "the single most-checked figure in a nutrition methods review"** | :242-246 | SETTLED | OVERSTATED / UNSOURCED | S23 nut-13 requires reporting exclusions, but nothing supports the superlative. Finding G14 |
| 12 | Rank models on calibration and clinical utility, not on AUC alone | :269-273 | SETTLED | VERIFIED; SELF-CONTRADICTED | S16. The app ranks "best" by AUC alone and computes no calibration (G11) |
| 13 | Undersampling, oversampling and SMOTE overestimate minority probability without improving AUC | :274-278 | SETTLED | VERIFIED (for logistic regression) | S6. The study covers standard and ridge logistic regression only |
| 14 | Multiclass is "scored by accuracy" | :299-301 | none | SELF-CONTRADICTED (minor) | metrics.py:21: the primary metric is macro-F1 (G16) |
| 15 | Ordinal: a cumulative link model is best; this version fits none | :308-317 | SETTLED | CONSISTENT | Honest about the gap |
| 16 | **A missing indicator is legitimate for prediction and biased for inference; MI or a principled model is required for inference** | :334-360 | SETTLED | CONSISTENT; SELF-CONTRADICTED | S9, S12 (the RCT exception is not mentioned). S10: indicators "can be harmful under outcome-dependent missingness". The app offers no MI, and its only imputer is single median (G3) |
| 17 | Split by participant; fit everything learned inside the training fold | :361-365, :399-402, :476-480, :555-558, :780-788 | SETTLED | VERIFIED | S22 |
| 18 | Twin columns such as `DR1TKCAL`/`DR2TKCAL` are wide-form repeats | :403-407 | SETTLED | CONSISTENT | |
| 19 | One recall measures one day; two or more non-consecutive recalls separate day-to-day variation | :438-442 | SETTLED | CONSISTENT | |
| 20 | Consecutive days have correlated errors; NHANES uses non-consecutive days by design | :443-447 | SETTLED | VERIFIED | S15 DR2TOT_J: "3 to 10 days later" |
| 21 | **The mean of recalls is "attenuated, but unbiased in direction"** | :491-492, :513-517 | CONVENTION | OVERSTATED | S14 §3.1.3: with several error-prone covariates, bias is "larger or smaller … in a rather unpredictable manner". The pack (§10.11) itself says FFQ error is not classical. Finding G9 |
| 22 | Prevalence and percentiles, episodic foods and unbiased magnitudes need usual-intake modeling (not fitted here) | :518-522 | SETTLED | CONSISTENT | Honest about the gap |
| 23 | The cohort standard is the cumulative average with a lag | :523-527 | CONVENTION | CONSISTENT | Not checked against a primary source |
| 24 | Validation in time is a distinct check, and reporting guidelines treat it so | :538-541 | none | CONSISTENT | TRIPOD splits validation types; not read here |
| 25 | Resample the entire pipeline (imputation, selection, tuning) | :559-562 (SETTLED); :789-793 and :1032-1036 (CONVENTION) | mixed | VERIFIED; the badge is inconsistent | S18, S22. The same claim carries two badges (G18) |
| 26 | NHANES oversamples; unweighted means are biased; SEs without the design are too small; use WTDRD1/WTDR2D, not WTMEC2YR | :609-615; finding_words.py:661-672 | SETTLED | VERIFIED; SELF-CONTRADICTED in use | S15. With design columns present, inference stays unweighted with SRS SEs and no concern is pushed (G13) |
| 27 | Pooled cycles: divide two-year weights by the number of cycles | :616-619; finding_words.py:575-576 | SETTLED | CONSISTENT (minor omission) | S15. The 1999–2002 4-year-weight exception is omitted (G20) |
| 28 | 1 kcal = 4.184 kJ; `_pct_kcal` is a share | :620-624 | SETTLED | VERIFIED | Thermochemical calorie |
| 29 | **"Willett, by sex: women 500–3,500, men 800–4,200"** | options :643-644; proposals.py:345-349; packs.py:2795 | CONVENTION | **WRONG attribution** | S3, quoting Willett 2013: men 800–**4,000**. 800–4,200 is a cohort-specific variant. The methods sentence writes "(Willett's sex-specific cut-offs)". Finding G4 |
| 30 | Screens in circulation: 4,000/5,000 for men, sex-neutral 500–5,000 and 500–3,500; conventions differ | :669-674 | CONVENTION | CONSISTENT | The 4,000 variant is named but not offered as a preset |
| 31 | **Excluding misreporters reduces bias but does not remove it; "only 14 of 24 pairs improved"; "insufficient is settled"** | :675-679 | DISPUTED | OVERSTATED / mis-situated | S4 is about **Goldberg** cut-offs, not the fixed-kcal screens offered here. In 10 of 24 pairs bias was not reduced at all. "Settled" rests on one simulation study (G15) |
| 32 | A common default: primary analysis on the full sample; exclusion as a sensitivity analysis | :680-685 | CONVENTION | UNSOURCED | Pack recommendation; no primary source |
| 33 | Under-reporting concentrates in higher BMI; report who is excluded | :686-690 | SETTLED | CONSISTENT | S23 nut-13 asks for the "characteristics of those excluded" |
| 34 | The Goldberg cut-off is "the field's standard for misreporting", not offered | term :659-661 | none | OVERSTATED (minor) | S4: use "should therefore be decided based on research purposes and not general rules" |
| 35 | An FFQ blank often means "never", which is not at random; MI "an improvement, not a solution" | :722-726 | SETTLED | UNSOURCED | Not checked against a primary source |
| 36 | A zero on a 24HR is a zero for that day | :727-730 | SETTLED | CONSISTENT | |
| 37 | **Mean or median filling is "indefensible in a manuscript"** | :731-734 | SETTLED | OVERSTATED (purpose-blind); SELF-CONTRADICTED | S11: constant imputation is consistent for prediction. The app's only imputer is median (pipeline.py:302), and its methods sentence hides that (G3) |
| 38 | **The outcome belongs in the imputation model; leaving it out biases toward the null** | :735-738 | SETTLED | OVERSTATED (purpose-blind) | True for MI and inference. For prediction, S10 says omitting the outcome "was preferred" when missingness is allowed at deployment. The app's imputer never uses the outcome (G3) |
| 39 | A small holdout measures little | :751-752; seal.py:459-470 | none | VERIFIED | S17: "minimum of 100 events and ideally 200" |
| 40 | Selecting features on all samples can report near-zero error with no signal | :784-788 | SETTLED | VERIFIED | S22 |
| 41 | A single split is weakest; bootstrap optimism correction or repeated CV is preferred | :789-793 | CONVENTION | VERIFIED; under-badged; not offered | S18 recommends bootstrapping. The app offers neither method (G10) |
| 42 | **Below about 50 rows, a single 5-fold estimate cannot resolve 0.05 AUC** | :794-798 (GENOMICS_PACK §08) | SETTLED | WRONG threshold / UNSOURCED | S8: about ±10% at n = 100. Repro `cv_auc_sd.py`: SD 0.084 at n = 100 and 0.053 at n = 200 (G10) |
| 43 | Energy confounds every nutrient association; errors in nutrients and energy move together | :809-813, :847-852 | SETTLED | OVERSTATED | S1: adjusting for total energy changes the estimand. It is not a confounder removal for every question (G5) |
| 44 | Standard and residual "ask about swapping calories … at fixed total energy" | :810-812, :820-824 | none | WRONG for the app's residual when covariates are present | Repro `residual_vs_standard.py`. G2 |
| 45 | **Density plus energy means "diet composition"** | option :825-826 | none | OVERSTATED | S1 labels model 3b "Obscure" and biased (G6) |
| 46 | Density alone is "a rescaled effect whose meaning is obscure" | :827-829 | none | VERIFIED | S1 |
| 47 | Partition means "adding calories, not substituting" | :830-832 | none | VERIFIED | S1: "estimates the total causal effect" |
| 48 | Residual and standard agree when energy is in the model, or with no covariates; the app's residual drops energy, so with covariates they agree only if none correlates with energy | :853-860 | SETTLED | VERIFIED (coefficient) | Repros: identical with E in the model; −32% with a sex covariate. Not stated: the CIs differ even with no covariates (`residual_se.py`: 0.270–0.336 vs 0.291–0.315) |
| 49 | An energy-adjusted coefficient is a substitution estimate | :861-865 | SETTLED | VERIFIED | S1 |
| 50 | Standard and residual are biased even without confounding (composite variable bias); all four only partly control dietary confounding | :866-871 | SETTLED | VERIFIED | S1. Omission: S1's recommended all-components model is never named (G6) |
| 51 | The field default is Willett residual, within the analytic sample and within sex, on log intakes | :872-876 | CONVENTION | UNSOURCED; not producible | log_transform is hard-coded false in the UI (ChoiceQuestions.tsx:415,423) (G17) |
| 52 | The residual regression is fit on training rows only | :881-885 | SETTLED | VERIFIED in code | energy.py:495-520 runs inside the Pipeline |
| 53 | An elastic net "shrinks correlated nutrients together" | :897-898, :906-908 | none | CONSISTENT | Grouping effect of the L2 term |
| 54 | Boosted trees "handle missing values" | :909-911 | none | SELF-CONTRADICTED (minor) | The shared imputer runs before every family (pipeline.py:297-309) (G16) |
| 55 | MAQC-II: >30,000 models from 36 teams; endpoint and proficiency mattered more than algorithm | :898-900, :927-932 | DISPUTED | VERIFIED (with a scope caveat) | S7. The study is microarray classification; the "why" text generalizes it without that context |
| 56 | Automatic selection among correlated nutrients picks one marker; report selection frequency | :922-926 | SETTLED | CONSISTENT | GENOMICS_PACK cites Michiels 2005 (not read here) |
| 57 | Boosted trees are often miscalibrated; report the calibration curve or recalibrate | :938-942 | SETTLED | CONSISTENT; not delivered | The app computes no calibration curve (G11) |
| 58 | Avoid stepwise selection | :943-946 | SETTLED | CONSISTENT, primary not read | Springer and Smith 2018 were inaccessible (redirect/auth). The app offers no stepwise option |
| 59 | Adjusting for energy does not make a substitution isocaloric; components must be in kcal | :981-985 | SETTLED | CONSISTENT | The app enforces kcal factors (substitution.py) |
| 60 | Closed parts: a %E coefficient has no meaning until you say what it replaced | :986-990 | SETTLED | CONSISTENT | |
| 61 | Three designs (leave-one-out, ilr, difference of coefficients) | :991-995 | CONVENTION | CONSISTENT | |
| 62 | Review of 100 substitution studies: 53% unvalidated, r 0.12–0.77 | :996-1000 | SETTLED | VERIFIED; uncited in the UI (minor) | S5. The drawer gives no author or year, so a reader cannot cite it |
| 63 | Report the C-statistic with its interval, calibration intercept, slope and curve, and the Brier score | :1027-1031 | SETTLED | CONSISTENT; not delivered | Only AUC, Brier and log loss are computed (metrics.py:18); no interval and no calibration (G11) |

### `turbotab/core/methods/energy.py` (estimand sentences shown next to coefficients)

| # | claim | location | status | evidence |
|---|---|---|---|---|
| 64 | **none: "Not energy-adjusted: a nutrient coefficient describes absolute intake"** | energy.py:75-85, :320 | **WRONG in use** | Under `none`, every column passes through (energy.py:641) and the energy-role column is a predictor (pipeline.py:43,50-59). The UI's none decision (ChoiceQuestions.tsx:408-416) gives the same matrix as `standard`. Repro `none_vs_standard.py`: both give β = 0.5007, while the absolute-intake β is 0.6351. G1 |
| 65 | standard: "more of the nutrient with total energy held fixed … substitution for the average of all other energy sources" | :86-96 | VERIFIED | S1 |
| 66 | **residual ("Willett residual model", Y ~ N_adj + C, E leaves): "The same substitution as the standard model"** | :97-107, :352-354, :545 | **WRONG with energy-correlated covariates; mislabeled** | S2: the Willett–Stampfer variant keeps an energy term. Repro: −32% versus standard with sex; CI 2.8× wider without covariates. G2 |
| 67 | density_multivariate: "Diet composition…", no caveats | :108-117 | OVERSTATED | S1: "Obscure", biased (G6) |
| 68 | density: obscure; standing "CONVENTION (weakest)" | :118-127 | VERIFIED | S1 |
| 69 | partition: total causal effect; unbiased only without confounding or with equal effects | :128-140 | VERIFIED | S1, verbatim |
| 70 | Fiber at 2 kcal/g is "the value some systems use" | :240-247, :280 | CONSISTENT | Hedged |
| 71 | The 5% slack is the "general Atwater factors' own error" | :395-396 | UNSOURCED (minor) | No citation |

### `turbotab/core/voice.py` (methods sentences that go into the Record and the manuscript)

| # | sentence | location | status | evidence |
|---|---|---|---|---|
| 72 | **"No energy adjustment was applied: `N` enter the models as absolute intakes"** | voice.py:698-700 | **WRONG** | G1: energy is in the model |
| 73 | **Residual: "… regressed on `E` … replaced by the residual plus the nutrient's mean"** | :704-709 | INCOMPLETE / imprecise | It does not say that energy leaves the model (S23 nut-12.2: "describe and justify"). In the log case, what is added back is the mean of the log, a geometric mean (G2, G17) |
| 74 | Standard: "… so each nutrient's effect is at fixed total energy" | :710-713 | CONSISTENT | For tree families "effect" has no single value; minor |
| 75 | **Missing: "missing predictor values were imputed, learned from training rows only…"** | :625-633 | **INCOMPLETE** | The method (single median; most frequent for categories) is never named. S23 nut-13: "any method used to handle missing values" (G3) |
| 76 | **Exclusions: "… were excluded as implausible intakes (Willett's sex-specific cut-offs)"** | :584-600 + proposals.py:349 | **WRONG attribution** | G4 |
| 77 | Split, seal, grain, unit, aggregation, temporal and orientation sentences | :638-680, :832-990 | CONSISTENT | They report what was drawn. Exploratory status is stated |
| 78 | Substitution: "in steps of k kcal with `E` held fixed; band from n bootstrap refits of training rows" | :761-772 | CONSISTENT | Matches modeling.py:511-634 |

### `turbotab/core/coach.py` (data-grounded notes)

| # | note | location | status | evidence |
|---|---|---|---|---|
| 79 | **"`N` tracks `E` at r X: mostly how much people eat" for any \|r\| ≥ 0.3** | coach.py:301-303 | **WRONG** (quantitative) | At r = 0.31, energy explains r² = 10%. Repro `coach_mostly.py` prints the note at r = 0.31, 0.45 and 0.60 (G12) |
| 80 | "Unadjusted, `N` still carries how much people eat" | :307-308 | Literally true of N, but misleads | Energy is in the model under none (G1) |
| 81 | "With this method r 0.00: what is left is composition" | :316 | CONSISTENT for residual | Fires for density too, where it is the obscure estimand (G6) |
| 82 | "Averaging k replicates cuts within-person variance k-fold" | :407-408 | VERIFIED (math) | Var(mean) = σ²_w/k under independent replicates |
| 83 | "`n` rows below 500 kcal: likely under-reporting" | :186-195, :535-539 | CONSISTENT (hedged) | One recall day below 500 kcal can be a real low day; "likely" is fair |
| 84 | "Excluded rows' median BMI is X, against Y kept" | :245-246 | VERIFIED (computed) | Delivers S23's "characteristics of those excluded" |
| 85 | "Top tenth by `E` has R× the `N` of the bottom" | :477-478 | VERIFIED (computed) | |

### `turbotab/core/stages/proposals.py` and `finding_words.py` (findings and card reasons)

| # | claim | location | badge | status | evidence |
|---|---|---|---|---|---|
| 86 | **"…every nutrient association is confounded by total intake; that adjustment is needed is not in dispute… standard and residual estimate the same substitution, … residual's advantages are practical"** | finding_words.py:517-521; packs.py:2809-2812, :2938-2945 | SETTLED (claim) | **OVERSTATED / SELF-CONTRADICTED** | S1: adjusting for TE changes the estimand, and the partition and all-components models estimate the total effect without it. The app's own DISPUTED card covers BMI outcomes. Prediction makes "confounding" moot. The residual half is wrong for the app's implementation (G2, G5) |
| 87 | "`E` is total energy; every nutrient association is confounded by it until adjusted" (summary) | finding_words.py:234-235 | none | OVERSTATED | as #86 |
| 88 | "A yes/no answer blank on ≥50% of rows: blank usually means not asked" | proposals.py:445-458 | none | UNSOURCED heuristic (minor) | G22 |
| 89 | "`n` count columns against `n` samples: an unpenalized model has no unique fit" | finding_words.py:260-261 | none | VERIFIED (math) | Rank deficiency |
| 90 | "`k` strata have a single PSU, which breaks design-based variance estimates" | :287-289 | none | CONSISTENT | |
| 91 | "`E` holds kilojoules: … convert by 4.184" | :293-294 | none | VERIFIED | |
| 92 | "No survey weights or design columns: results describe these participants, not the US population" | :661-672 | SETTLED | VERIFIED | S15. The mirror case (design present but unused) has no finding (G13) |
| 93 | "Missing values cluster in the lowest-abundance features: likely below detection, not MAR" | :314-315 | none | CONSISTENT | S20 (MNAR, left-censored) |
| 94 | Repeats: "a split that puts one participant's rows on both sides … every score looks better" | :599-608 | SETTLED | VERIFIED | S22 logic; standard |
| 95 | Pooled cycles: "methods may differ across them" | :688-696 | SETTLED | CONSISTENT | S15 (minor omission, G20) |

### Other emitted claims touched

| # | claim | location | badge | status | evidence |
|---|---|---|---|---|---|
| 96 | "SETTLED that polychoric correlations are the appropriate choice for ordinal items"; "every correlation below is nearer zero than the polychoric one" | survey.py:63-69, :339-347 | SETTLED | OVERSTATED (minor) | S24: ordinal alpha "should not be used in routine reliability analyses". An item–rest correlation pairs an item with a sum score, which calls for polyserial, not polychoric (G19) |
| 97 | Binary seal floor of 100 events and 100 non-events | seal.py:461-506 | cited | CONSISTENT | S17 says "minimum of 100 … ideally 200" |
| 98 | Substitution curve = forward marginal effect averaged (Scholbeck 2024) | substitution.py:1-35 | cited | VERIFIED citation | S19 |
| 99 | The 50% support floor is "a practitioner convention, not a sourced threshold" | substitution.py:222-225 | self-labeled | CONSISTENT (honest) | |
| 100 | "Total energy was held fixed by assumption … a modeling choice, not a property of the data" | substitution.py:215-217 | none | CONSISTENT | S1 framing |
| 101 | Purpose preview: "With inference … a missing-value indicator would bias them" | fact_previews.py:281-287 | none | VERIFIED (observational) | S12. The warning appears only on the purpose card, not on the missing-values card where the indicator is chosen |

## North star 5 check (custom and soundness labels)

`grep -rn "customary\|soundness\|sound_for"` over `turbotab/core` and `turbotab/frontend/src` returns nothing. No option carries the two labels "customary in <field>" and "sound for <purpose>". The energy card orders by `usual` (residual) first, with "No adjustment" last (ChoiceQuestions.tsx:400-407), regardless of purpose. The missing card's "recommended" tag (ChoiceQuestions.tsx:254) is not purpose-dependent either (G7).

## WP15 re-check (2026-10-03)

*Added by the claims package (AUDIT_REPORT §5, WP15, acceptance test 1); the rows above are unchanged
as the auditor left them.* Every row marked WRONG, OVERSTATED or SELF-CONTRADICTED (27), and three
rows marked INCOMPLETE or misleading (73, 75, 81), re-checked against the text the app serves today.
Each line gives what the app now serves beside the primary source it rests on. Rows the methods
layer had already corrected (WP6: 44, 64, 66, 72, 73; WP7: 37, 38, 75; WP9: 12 in part, 42; WP10: 26)
are re-checked in the same way, so a later edit cannot quietly restore the old claim. The replay is
`turbotab/core/tests/acceptance/test_wp15_claims.py` (`RECHECK`, `ALSO`): it reads this ledger's
marked rows, requires one entry for each, and asserts each corrected phrase is served and each old
one is gone. Where a pack carried the old claim, the pack now carries the correction with a dated
note of what it read before.

| # | was | served now (the corrected sentence, or its load-bearing part) | primary source |
|---|---|---|---|
| 4 | OVERSTATED | "an unpenalized least-squares or logistic model is degenerate"; "Penalization, dimension reduction or one test per feature are the ways through" | closed form: rank(X) ≤ n < p, so XᵀX is singular and least squares has a (p − n)-dimensional set of exact fits (test_4_p_over_n_is_degenerate_for_least_squares_only); the app's own feature-wise family is a way through |
| 9 | WRONG | "the count is above that limit, right-censored"; "QNS, quantity not sufficient" | FDA Bacteriological Analytical Manual, ch. 3, Aerobic Plate Count: "When number of CFU per plate exceeds 250, for all dilutions, record the counts as too numerous to count (TNTC) for all but the plate closest to 250"; "Estimate the APC as greater than 100 times the highest dilution plated, times the area of the plate." |
| 11 | OVERSTATED | "STROBE-nut asks for the number excluded for missing, incomplete or implausible dietary data, and STROBE suggests a flow diagram" | Lachat et al. 2016 (STROBE-nut), PLoS Med (PMC4896435), nut-13: "Report the number of individuals excluded based on missing, incomplete, or implausible dietary/nutritional data." STROBE 13(c): "Consider use of a flow diagram." |
| 12 | SELF-CONTRADICTED | "Judge models on calibration as well as discrimination"; "reports its calibration intercept and slope beside the AUC that ranks the families" | Van Calster et al. 2019, BMC Med 17:230 (PMC6912996): "poor calibration may make an algorithm less clinically useful than a competitor algorithm that has a lower AUC but is well calibrated"; the app ranks binary fits by AUC (models/metrics.py PRIMARY) and reports calibration on every fit |
| 14 | SELF-CONTRADICTED | "scored by log loss" | the code: models/metrics.py PRIMARY['multiclass'] == 'log_loss' (audit ME-10) |
| 16 | SELF-CONTRADICTED | "in a randomized trial it is valid for baseline covariates"; "multiple imputation, offered here" | Groenwold et al. 2012, CMAJ 184:1265 (PMC3414599): the method "typically results in biased estimates in nonrandomized studies"; "In randomized trials, the missing-indicator method is a valid method to handle missing baseline covariate data" |
| 21 | OVERSTATED | "attenuated toward zero only as the model's one error-prone exposure"; "a coefficient can be attenuated, inflated or change sign" | Freedman et al. 2011, JNCI 103:1086 (PMC3143422): "In multivariable disease models with two or more mismeasured exposures, estimated relative risks may become attenuated, inflated, or can even change direction"; Keogh et al. 2020 (STRATOS Part 1, PMC7450672) §3.1.3: "the estimated coefficients in model (11) may be larger or smaller than the true target values in a rather unpredictable manner" |
| 26 | SELF-CONTRADICTED | "Under inference the survey question then asks whether the estimates describe the surveyed population" | CDC NHANES variance tutorial: estimates "computed using standard statistical software packages that assume simple random sampling are generally too low"; the contradiction in use closed by WP10 (the survey question, test_wp10_survey_design.py) |
| 29 | WRONG | "Willett's textbook (2013) gives 500–3,500 kcal a day for women and 800–4,000 for men"; "Women outside 500–3,500 (NHS) and men outside 800–4,200 (HPFS)" | Banna et al. 2017, Front Nutr 4:45, quoting Willett, Nutritional Epidemiology 3rd ed. (2013): "the range of 500–3,500 kcal/day may be applied to data from women" and "an allowable range of 800–4,000 kcal/day for men may be used"; Pan et al. 2011, AJCN 94:1088 (PMC3173026), NHS, NHS II and HPFS: "daily energy intake <800 or >4200 kcal/d for men and <500 or >3500 kcal/d for women" |
| 31 | OVERSTATED | "Goldberg cut-offs reduced but did not remove bias in 14 of 24"; "fixed kcal screens were not evaluated" | Yamamoto et al. 2023, eLife 12:e83616: "Significant bias in simulated associations using self-reported NI was reduced but not completely eliminated by Goldberg cutoffs in 14 of 24 nutrition-outcome pairs; bias was not reduced for the remaining 10 cases. … Whether one uses Goldberg cutoffs should therefore be decided based on research purposes and not general rules." |
| 34 | OVERSTATED | "widely used, and chosen by research purpose" | Yamamoto et al. 2023, eLife 12:e83616: "Significant bias in simulated associations using self-reported NI was reduced but not completely eliminated by Goldberg cutoffs in 14 of 24 nutrition-outcome pairs; bias was not reduced for the remaining 10 cases. … Whether one uses Goldberg cutoffs should therefore be decided based on research purposes and not general rules." |
| 37 | OVERSTATED / SELF-CONTRADICTED | "For inference, mean or median filling understates variance"; "For prediction, a fill learned in each training fold is the deployable choice" | Josse et al., arXiv:1902.06931: "the widely-used method of imputing with a constant, such as the mean prior to learning is consistent when missing values are not informative" (purpose-scoped by WP7) |
| 38 | OVERSTATED | "The outcome's place in the imputation model depends on the purpose"; "without the outcome" | Sisk et al. 2023, Stat Methods Med Res 32:1461: "When missingness is allowed at deployment, omitting the outcome from the imputation model at the development was preferred" (WP7's RULE) |
| 42 | WRONG | "computed on these rows" | Varoquaux 2018, NeuroImage 180:68: "sample sizes of many neuroimaging studies inherently lead to large error bars, eg±10% for 100 samples" (the threshold was 2–4× too low; WP9 replaced it with intervals on the user's rows) |
| 43 | OVERSTATED | "adjusting for it changes what a nutrient's coefficient means"; "Adjusting for it changes the question" | Tomova et al. 2022, AJCN 115:189 (PMC8755101): "It remains underappreciated that adjusting for total energy and adjusting for remaining energy intake evaluate very different causal estimands." |
| 44 | WRONG | "the standard model's swap"; "differs when covariates track energy" | McCullough & Byrd 2023, AJE 192:1801: "A variation on the simple nutrient residual model proposed by Willett and Stampfer includes the nutrient residual plus a term for total energy intake." (WP6) |
| 45 | OVERSTATED | "obscure, and still biased" | Tomova et al. 2022, model 3b: "the multivariable nutrient density model returns a more accurate estimate than the (unadjusted) nutrient density model, but one which is still biased"; Table 2 estimand "Obscure" |
| 54 | SELF-CONTRADICTED | "gives no coefficients" | the code: models/pipeline.py shared_steps fills or drops blanks before every family (test_1_the_trees_never_see_a_blank) |
| 64 | WRONG | "total energy is not in the model" | the code: under 'none' every energy-role column leaves the matrix (WP6, test_wp6_energy_estimands.py test_1) |
| 66 | WRONG | "the coefficient is the standard model's exactly"; "only when no other covariate correlates with energy" | McCullough & Byrd 2023 (as row 44); WP6 test_2 |
| 67 | OVERSTATED | "but one which is still biased"; "the all-components model is the paper's recommended route" | Tomova et al. 2022 (as row 45) and its Discussion: "we would recommend the all-components model as the more intuitive and transparent option and the least susceptible to misinterpretation" |
| 72 | WRONG | "`kcal` was left out of the models" | the code (WP6): under 'none' the energy-role column leaves the models |
| 73 | INCOMPLETE | "with total energy kept in the outcome model" | STROBE-nut nut-12.2: "Describe and justify the method for energy adjustments" (WP6) |
| 75 | INCOMPLETE | "the median for numbers" | STROBE-nut nut-13: "any method used to handle missing values" (WP7) |
| 76 | WRONG | "Willett 2013's sex-specific cut-offs"; "the Nurses' Health Study and Health Professionals Follow-up Study cut-offs" | Banna et al. 2017, Front Nutr 4:45, quoting Willett, Nutritional Epidemiology 3rd ed. (2013): "the range of 500–3,500 kcal/day may be applied to data from women" and "an allowable range of 800–4,000 kcal/day for men may be used"; Pan et al. 2011, AJCN 94:1088 (PMC3173026), NHS, NHS II and HPFS: "daily energy intake <800 or >4200 kcal/d for men and <500 or >3500 kcal/d for women" |
| 79 | WRONG | "energy explains `10%` of it"; "energy explains `36%` of it"; "energy explains most, `64%`" | arithmetic: the share of a nutrient's variance its line on energy explains is r² (0.31² = 0.096; 0.60² = 0.36; 0.80² = 0.64) |
| 81 | misleading | "it no longer tracks energy" | Tomova et al. 2022: the density coefficient is "an obscure quantity" |
| 86 | OVERSTATED / SELF-CONTRADICTED | "makes a nutrient's coefficient a substitution"; "The two answer different questions (Tomova et al. 2022)" | Tomova et al. 2022, AJCN 115:189 (PMC8755101): "It remains underappreciated that adjusting for total energy and adjusting for remaining energy intake evaluate very different causal estimands." |
| 87 | OVERSTATED | "adjusting for it makes each nutrient's effect a swap at fixed energy" | Tomova et al. 2022, AJCN 115:189 (PMC8755101): "It remains underappreciated that adjusting for total energy and adjusting for remaining energy intake evaluate very different causal estimands." |
| 96 | OVERSTATED | "polyserial correlation, not a polychoric one"; "ordinal alpha should not be used in routine reliability analyses" | Chalmers 2018, Educ Psychol Meas 78:1056 (PMC6293415): "ordinal alpha should not be used in routine reliability analyses and reports"; a polyserial correlation is the one between an ordinal and a continuous variable (the item–rest sum). The legacy turbotab/survey.py that printed it is not imported by Next (test_1_the_polychoric_claim_is_not_served_by_next) |
