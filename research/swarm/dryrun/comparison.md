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

- `clinical_explore`: record `research/swarm/dryrun/clinical_explore-18aebb257587732c517c624d59e3662d.json`, 60 driver commands, 210.3 s.
- `survey_single`: record `research/swarm/dryrun/survey_single-b1411971f16554c645cfb0a1c447025f.json`, 52 driver commands, 62.8 s.
- `decoy_single`: record `research/swarm/dryrun/decoy_single-e91810f5c78f8ec4fe15445a9e1679c2.json`, 55 driver commands, 299.3 s.
