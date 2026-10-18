# MedCost Explorer — Explainable Medical Charge Estimation

An educational regression project with reproducible preprocessing, cross-validated model selection, calibrated prediction ranges, error analysis and a Streamlit interface. It extends the original **AI for Insurance: Costs and Smoking Behavior** coursework, focusing the new application on estimating medical charges.

**Scope:** this dataset's original collection dates, authenticity and population representativeness are not independently verified. The app is not an insurance quote or a validated predictor of current medical costs. No public deployment is claimed yet.

## What is included

- Six fixed candidate configurations compared using five-fold training-only CV.
- Median imputation, scaling and one-hot encoding fitted inside training folds.
- Separate fitting, calibration and test partitions; the app trains on log-transformed charges and converts estimates back to dollars.
- Original-dollar MAE/RMSE, R², median-baseline comparison and nominal 90% calibrated ranges.
- Subgroup error, bias and coverage; original-feature permutation importance.
- Executed EDA notebook, HTML export, automated tests and an interactive app.

## Quick start — Python 3.12

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m medcost.calibrate_log
python -m pytest -q
python -m streamlit run app.py
```

On Windows activate `.venv\Scripts\Activate.ps1`. See [the learning and execution guide](docs/LEARNING_GUIDE.md) for VS Code and Colab. To execute EDA and export HTML:

```bash
python scripts/execute_notebook.py
```

## Current experiment

The fixed split contains 801 fitting, 268 calibration and 268 test records after removing one exact duplicate. Missing BMI records are retained and imputed using training-only statistics. See [machine-readable metrics](reports/metrics.json) for results and [the model card](docs/MODEL_CARD.md) for the evaluation protocol and limitations. Results use a different split and cleaning strategy from the original coursework, so they are not a controlled before-and-after comparison.

## Original experiment results

| Held-out test measure | Result |
|---|---:|
| Selected model | Random forest, minimum leaf size 5 |
| MAE | $2,309 |
| RMSE | $3,592 |
| R² | 0.911 |
| Median-only baseline MAE | $8,144 |
| Nominal 90% interval: observed test coverage | 91.0% |
| Mean interval width | $10,005 |

MAE was 71.6% below the median baseline on this fixed test sample. Wide intervals and individual errors limit use despite the average improvement. Neither R² nor MAE reduction is classification accuracy. This experiment does not establish reliable current-day pricing.

## Current app: log-target model

The app now uses the log-target random forest, fitted on 801 rows and calibrated on 268 separate rows. It creates this fixed model once per app process and caches it in memory; no saved pickle is loaded, model search is not repeated, and test data is not scored at startup.

The log-target extension improved mean CV MAE across five fold arrangements (approximately $2,470 versus $2,986). Its follow-up comparison on the already inspected test set gave MAE **$1,753**, **79.9%** within $2,000 and overall prediction-range coverage **91.4%**. The calibrated margin is **±$3,917**. These are exploratory follow-up results, not fresh independent confirmation.

**High-charge limitation:** among 52 test records with actual charges >= $20,000, MAE was $4,144 and range coverage was only 71.2%. Overall coverage does not guarantee subgroup or individual coverage. The app cannot know a person's actual charge group beforehand.

Read [the comparison](reports/LOG_TEST_COMPARISON.md), [stability check](reports/STABILITY_EXPERIMENT.md), and [calibration explanation](reports/LOG_CALIBRATION.md). Original notebook and original model reports remain historical experiment evidence; they do not describe the current app model. Reproduce the original experiment using `python -m medcost.training` and the follow-up using the commands in those reports.

## Use the interface

- **Estimate charges:** enter a profile or choose **Try an example profile**, then submit. The estimate and calibrated range appear alongside the form, labeled with the profile used.
- **How it works:** read a plain-language explanation of the inputs, model and prediction range.
- **Results & limitations:** inspect overall results and the weaker high-charge performance.

The example is an illustrative input profile, not a known bill or a test record. It is not added to the training data. Editing inputs requires a new submission; the displayed profile identifies the last submitted estimate.

## Explore

- [Executed EDA and modeling notebook](notebooks/01_medical_costs.ipynb)
- [HTML notebook export](reports/eda.html) — download and open in a browser if GitHub displays source
- [Subgroup errors](reports/subgroup_errors.csv)
- [Permutation importance](reports/permutation_importance.csv)
- [Learning guide](docs/LEARNING_GUIDE.md)
- [Model card and source limitations](docs/MODEL_CARD.md)

## Deployment

After publishing the code to GitHub, create a Streamlit Community Cloud app with this repository, branch `main`, entry point `app.py`, and Python 3.12. On first startup the app fits and calibrates the fixed log-target model and caches it in memory; no GPU, API key or external dataset download is required. Generated model files are not committed. Pickle artifacts must never be loaded from untrusted sources. Restart/redeploy after changing code or data to avoid stale cached artifacts. A public hosting account and its deployment action are still required.

## Data and original work

The checked-in `insurance.csv` is preserved unchanged. The original README attributed it to [Kaggle healthcare-insurance](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance). Original collection details and upstream redistribution terms remain unverified; no third-party license or authenticity claim is invented. Records contain no names or direct contact identifiers in this CSV, but should still be treated as dataset records rather than real-time customer inputs.

`NoteBook.ipynb` and `Report.docx` are retained unchanged for historical reference. Their metrics contain documented errors and are superseded by the new pipeline and reports. The app does not implement smoking classification.
