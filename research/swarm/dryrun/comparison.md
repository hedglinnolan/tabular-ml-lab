# The driver's dry run: each check against its construction

Written by `python -m research.swarm.dry_run` (DRIVER_PLAN D6): each run's expected result per check, written in `dry_run.py` before the run, beside the check's result on the run's record.

| Run | Dataset | Check | Expected | Got | Agrees |
|---|---|---|---|---|---|
| clinical_explore | inpatient_readmission_predict | seal_integrity | pass | pass | yes |
| clinical_explore | inpatient_readmission_predict | logging | pass | pass | yes |
| clinical_explore | inpatient_readmission_predict | single_pass | pass | pass | yes |
| clinical_explore | inpatient_readmission_predict | known_answer | not_checked | not_checked | yes |
| clinical_explore | inpatient_readmission_predict | null_decoy | not_checked | not_checked | yes |
| clinical_explore | inpatient_readmission_predict | robustness | pass | pass | yes |
| survey_single | walking_bp_estimate | seal_integrity | pass | pass | yes |
| survey_single | walking_bp_estimate | logging | pass | pass | yes |
| survey_single | walking_bp_estimate | single_pass | fail | fail | yes |
| survey_single | walking_bp_estimate | known_answer | pass | pass | yes |
| survey_single | walking_bp_estimate | null_decoy | not_checked | not_checked | yes |
| survey_single | walking_bp_estimate | robustness | pass | pass | yes |
| decoy_single | outpatient_admission_predict | seal_integrity | pass | pass | yes |
| decoy_single | outpatient_admission_predict | logging | pass | pass | yes |
| decoy_single | outpatient_admission_predict | single_pass | fail | fail | yes |
| decoy_single | outpatient_admission_predict | known_answer | not_checked | not_checked | yes |
| decoy_single | outpatient_admission_predict | null_decoy | pass | pass | yes |
| decoy_single | outpatient_admission_predict | robustness | pass | pass | yes |

- `clinical_explore`: record `research/swarm/dryrun/run_01-f84bb3e05a3989b9afe5ff95cede4fda.json`, 60 driver commands, 364.6 s.
- `survey_single`: record `research/swarm/dryrun/run_02-bb676eb7dc6cdabb911d5748e8c0d1cb.json`, 32 driver commands, 38.3 s.
- `decoy_single`: record `research/swarm/dryrun/run_03-bd56f686e2bd089462954f32fbf13cc9.json`, 55 driver commands, 337.4 s.
