# The driver's dry run: each check against its construction

Written by `python -m research.swarm.dry_run` (DRIVER_PLAN D6): each run's expected result per check, written in `dry_run.py` before the run, beside the check's result on the run's record.

| Run | Dataset | Check | Expected | Got | Agrees |
|---|---|---|---|---|---|
| clinical_explore | clinical_readmission_predict | seal_integrity | pass | pass | yes |
| clinical_explore | clinical_readmission_predict | logging | pass | pass | yes |
| clinical_explore | clinical_readmission_predict | single_pass | pass | pass | yes |
| clinical_explore | clinical_readmission_predict | known_answer | not_checked | not_checked | yes |
| clinical_explore | clinical_readmission_predict | null_decoy | not_checked | not_checked | yes |
| clinical_explore | clinical_readmission_predict | robustness | pass | pass | yes |
| survey_single | survey_support_estimate | seal_integrity | pass | pass | yes |
| survey_single | survey_support_estimate | logging | pass | pass | yes |
| survey_single | survey_support_estimate | single_pass | fail | fail | yes |
| survey_single | survey_support_estimate | known_answer | pass | pass | yes |
| survey_single | survey_support_estimate | null_decoy | not_checked | not_checked | yes |
| survey_single | survey_support_estimate | robustness | pass | pass | yes |
| decoy_single | outpatient_admission_predict | seal_integrity | pass | pass | yes |
| decoy_single | outpatient_admission_predict | logging | pass | pass | yes |
| decoy_single | outpatient_admission_predict | single_pass | fail | fail | yes |
| decoy_single | outpatient_admission_predict | known_answer | not_checked | not_checked | yes |
| decoy_single | outpatient_admission_predict | null_decoy | pass | pass | yes |
| decoy_single | outpatient_admission_predict | robustness | pass | pass | yes |

- `clinical_explore`: record `research/swarm/dryrun/clinical_explore-6d1b37d008afc715265af12eb7af020f.json`, 60 driver commands, 209.0 s.
- `survey_single`: record `research/swarm/dryrun/survey_single-80da5ef0902ba80242b557663b8d8d41.json`, 52 driver commands, 60.7 s.
- `decoy_single`: record `research/swarm/dryrun/decoy_single-b57ad1aca6d46a510d3dbe51df2564a0.json`, 55 driver commands, 297.8 s.
