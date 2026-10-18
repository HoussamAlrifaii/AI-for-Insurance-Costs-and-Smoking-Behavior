# Stability check: original versus log-target random forest

Run `python -m medcost.stability_experiment`. Uses only the original 801 fitting observations. Both target versions use identical folds within each repeat; forest settings and seed stay fixed. Five-fold CV is repeated with split seeds 7, 19, 42, 73 and 101. Seed 42 reproduces the preceding experiment; the other four split arrangements are additional checks.

| Fold seed | Original CV MAE | Log-target CV MAE | Reduction |
|---|---:|---:|---:|
| 7 | $3,017 | $2,471 | 18.1% |
| 19 | $2,969 | $2,462 | 17.1% |
| 42 | $2,943 | $2,452 | 16.7% |
| 73 | $2,982 | $2,470 | 17.2% |
| 101 | $3,021 | $2,493 | 17.5% |

Across the five arrangements, mean CV MAE was $2,986 for original charges and $2,470 for log charges. Averaged out-of-fold rates within $1,000 were 31.7% and 50.1%; rates within $2,000 were 57.6% and 75.6%. These summarize repeated predictions of the same fitting observations, not newly collected people.

The expensive-case trade-off persists: average out-of-fold MAE among records with actual charges >= $20,000 was $6,128 versus $6,257. Actual charge groups are retrospective diagnostics, not known deployment inputs. Log transformation consistently improves overall MAE in these splits but does not solve expensive-case errors.

Caveats: repeated CV folds overlap, so these are not independent experiments or confidence intervals. The follow-up was motivated by previously inspected test errors; no fresh independent final-performance claim is made. Calibration and test partitions were not scored here. The existing application model, its calibrated margin, and test report remain unchanged. Replacing the application requires a deliberate model decision and recalibration; any stronger generalization claim needs independent confirmation.

Validation: all five paired repeats completed; both seed-42 scores independently matched the previous logged experiment to numerical tolerance. Machine-readable results and protocol are saved beside this report.
