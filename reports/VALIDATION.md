# Validation record

The pipeline ran successfully with the package versions in `requirements.txt`. All four tests passed: disjoint partitioning and target exclusion; training-only imputation and unknown-category handling; finite-sample conformal quantile; and interactive estimation with changed inputs. The EDA notebook executed top-to-bottom and exported an HTML copy with embedded charts.

Independently recomputed held-out MAE and interval coverage from `artifacts/test_predictions.csv`; both match `reports/metrics.json`. Dataset and legacy notebook/report remain unchanged. Cross-validation selected the random forest without using calibration or test scores. Test R² is calculated from that model's predictions, correcting the old coursework's wrong-object issue.

Results are shareable as an educational experiment with the model-card caveats. Independent data collection provenance, current-day validity, external-population validation, and third-party redistribution terms remain unverified. The app's local behavior is verified; public hosting is not performed. GitHub workflow execution is separate from locally passing tests.
