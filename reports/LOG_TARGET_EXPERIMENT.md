# Experiment: log-transformed charges

Reproduce with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m medcost.log_experiment` (omit environment prefixes on Windows).

Only the 801 fitting records are scored, using the same five folds as the original comparison. Every preprocessing step fits on the training fold. Log versions train on `log1p(charges)` and invert predictions with `expm1`, so all reported errors are in dollars. Inverse log predictions are not bias-corrected estimates of the conditional mean.

| Random forest version | Mean CV MAE | Out-of-fold predictions within $1,000 | Within $2,000 |
|---|---:|---:|---:|
| Original target | $2,943 | 32.7% | 58.2% |
| Log target | $2,452 | 50.4% | 76.0% |

MAE improved by approximately 16.7%. This is a fitting-data cross-validation comparison, not a new test-set result. Overall out-of-fold RMSE changed from approximately $5,087 to $5,057.

The benefit differs by actual charge range:

| Actual charges | Records | Original OOF MAE | Log OOF MAE |
|---|---:|---:|---:|
| Below $10,000 | 420 | $1,730 | $880 |
| $10,000 to below $20,000 | 214 | $2,813 | $2,513 |
| $20,000 or more | 167 | $6,158 | $6,327 |

Log transformation reduces the influence of large target values during training. Here it improves lower-charge accuracy while slightly worsening the high-charge group. Grouping uses actual charges for retrospective diagnosis; those group labels are not known at prediction time.

This is an exploratory follow-up motivated by already inspected test diagnostics. Calibration and test predictions were not rescored or used for candidate selection in this experiment, but it is not a fully independent confirmation. Before replacing the application model, decide acceptable error criteria, validate stability using development data, and plan independent final confirmation. Changing the model also requires recalculating its calibrated range using the separate calibration partition. The existing application, model artifact, calibration and original test report remain unchanged.

Full candidate results: `log_target_experiment.csv`. Group results: `log_target_segments.csv`. Protocol: `log_target_protocol.json`.
