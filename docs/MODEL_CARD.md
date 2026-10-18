# MedCost Explorer model card

## Purpose and boundaries

Estimate the `charges` field in a small tabular dataset as an educational regression project. This is not a premium quote, clinical recommendation, causal estimate, or validated present-day cost estimator. The main app does not infer smoking status. Sex and smoking are source attributes; their inclusion here is not a recommendation for production insurance pricing.

## Data and provenance

The original repository links `insurance.csv` to https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance. That link does not establish the original collection process, dates, population representativeness, or whether all records are authentic observed cases. We have not independently verified those facts. Do not describe this as verified operational insurer data. Redistribution and upstream license terms need verification before broader distribution or commercial reuse. No license is invented for third-party data.

The checked-in CSV is preserved unchanged: 1,338 rows, seven columns, three missing BMI values, and one exact duplicate. The pipeline removes the repeated row, retains missing BMI records, and trains on 1,337 records. Exact equality does not prove duplicated identities; the exclusion is documented. File SHA-256 is saved in `reports/metrics.json`. Charges are treated as dollar amounts following source descriptions; the exact billing period and price year are not verified, so the interface does not label them annual or inflation-adjusted.

## Inputs and target

Inputs: age (18–64 in the source), sex (female/male in the source), BMI, children/dependents (0–5), self-reported smoker flag, US region. Target: charges, in original source units. No target normalization is used. Region is one-hot encoded rather than discarded. Missing numeric values use training-fold medians; categorical values use training-fold modes. Unknown categories are handled by the encoder, although the interface constrains inputs to source categories. Accepted individual ranges do not establish coverage of every combination.

## Evaluation protocol

A fixed random seed of 42 partitions deduplicated observations, stratifying by smoking group: 801 fitting rows, 268 calibration rows, and 268 test rows. All preprocessing fits inside each CV fold. Five-fold shuffled cross-validation on fitting rows compares a median-only baseline, two Ridge penalties, a random forest, and two histogram gradient boosting configurations. The model with lowest mean CV MAE is fitted on all fitting rows only. Calibration and test targets do not select the model.

The separate calibration partition computes a finite-sample split-conformal absolute residual quantile for nominal 90% prediction coverage. The model is NOT refitted on calibration rows, which would invalidate this calibration procedure. The lower endpoint is clipped at zero because charges are nonnegative. The assumptions are exchangeable examples and valid model/data handling, not arbitrary future-population similarity. Report empirical overall and subgroup coverage; do not promise conditional or individual coverage.

The final test reports MAE, RMSE and R², median-baseline comparison, coverage and width, subgroup errors and feature permutation importance. The held-out set is evaluated for reporting and diagnostics only; subsequent model selection based on its results would require a fresh final evaluation set. Permutation importance explains predictive reliance, not causation. Group metrics use small samples and do not prove fairness.

## Results and limitations

Read `reports/metrics.json` for reproducible current results and `reports/subgroup_errors.csv` for group counts, error, bias and interval coverage. Lower MAE is better; R² is not a percentage accuracy measure. Cross-validation fold standard deviations are not confidence intervals. Predictions can still be substantially wrong for individual records. Random splitting cannot rule out shared people or household dependence because no person identifiers are supplied.

The original coursework used a different split, dropped missing rows, and normalized the target using full-dataset maxima. New scores must not be presented as a controlled algorithm-only improvement over that experiment. Source uncertainty, small sample size, unmeasured medical history, distribution shift, and wide intervals prevent operational readiness claims.
