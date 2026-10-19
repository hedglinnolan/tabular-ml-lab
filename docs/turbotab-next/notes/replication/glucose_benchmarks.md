# Published benchmarks for Nolan's NHANES glucose model (search of 2026-10-19)

A Sonnet 5.5 research agent searched for recent peer-reviewed papers that predict fasting plasma
glucose, prediabetes or undiagnosed diabetes from non-glycemic information on NHANES. Every figure
below came from a small model's reading of the paper's page, not the PDF: check each against the
paper before citing it.

## The finding

No peer-reviewed paper found predicts fasting plasma glucose as a number from non-glycemic
predictors on NHANES. The continuous-outcome papers it found use glycemic predictors (HbA1c,
fasting glucose) or are preprints. So the external benchmark for a glucose model is a
classification: its predictions thresholded at 126 mg/dL (undiagnosed diabetes) and 100 mg/dL
(prediabetes), on the benchmark's own population and outcome.

## The three best benchmarks

| Paper | Data and validation | Outcome | Predictors | Reported performance | Reproducible? |
|---|---|---|---|---|---|
| Bang et al. 2009, *Ann Intern Med* 151:775 ([PMC3633111](https://pmc.ncbi.nlm.nih.gov/articles/PMC3633111/)) | NHANES 1999–2004 to develop (n 5,258); temporal validation on NHANES 2005–06 (n 1,640); external on ARIC/CHS (n 19,728) | undiagnosed diabetes: FPG ≥126 mg/dL, no known diabetes; adults 20+, pregnancy and diagnosed diabetes excluded | age, sex, family history, hypertension history, BMI category, physical activity | AUC 0.79 development, 0.83 NHANES 2005–06, 0.74 ARIC/CHS; no intervals | highest: the score is published and uses plain NHANES variables. Already a pilot replication (SWARM_PLAN §5) |
| Yoo, Maggiore & Jolliet 2026, *J Clin Epidemiol* 195:112266 (doi 10.1016/j.jclinepi.2026.112266; abstract only) | NHANES 1988–2018 to develop (n 17,458); temporal external on NHANES 2021–23 (n 3,043) and Korea NHANES 2023 | prediabetes or diabetes (exact cutoffs and fasting rule not verified) | age, waist, systolic BP, sex, upper leg length, arm circumference, BMI | AUC 0.804 (0.792–0.816) internal; 0.773 (0.756–0.790) US temporal; 0.780 Korea | partly: outcome definition unverified, a neural network |
| De Silva, Jönsson & Demmer 2020, *JAMIA* 27:121 ([PMC7647289](https://pmc.ncbi.nlm.nih.gov/articles/PMC7647289/)) | NHANES 2013–14 random split; external on NHANES 2011–12 (an earlier cycle) | prediabetes by ADA criteria (FPG, OGTT or HbA1c); diabetes excluded | 25 of 46, including non-glycemic labs | AUROC 0.70–0.72 internal, 0.69–0.70 external; CDC screening tool 0.64 and 0.63 | moderate |

## What this means for the glucose project

- **Bang is the north star:** the same NHANES, a published score, and a later-cycle validation
  (AUC 0.83). The glucose model thresholded at 126 mg/dL, on Bang's population, gives a directly
  comparable AUC.
- **A realistic ceiling** for non-laboratory prediction of dysglycemia on a later cycle is about
  0.77–0.80 (Yoo 2026); prediabetes alone is harder, about 0.70 (De Silva 2020).
- **Random-split results are optimistic:** several other candidates report AUCs of 0.82–0.91 on
  random splits (Qin 2022, Riveros Perez 2025, Liu 2025), which a temporal test will not match.

Other candidates and what could not be verified are in the agent's report (session of 2026-10-19).
