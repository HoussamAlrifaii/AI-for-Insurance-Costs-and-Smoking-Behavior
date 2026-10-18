# How to understand and run MedCost Explorer

## What changed from the original coursework?

The original `NoteBook.ipynb` and `Report.docx` are preserved unchanged as historical work. They contain known issues: Ridge R² accidentally used the decision tree object; scaling was computed before splitting; classification feature importance used regression column labels; reported tables disagreed. Do not use their metrics as the current experiment results.

The new cost-only pipeline fixes those problems through explicit features, fold-contained preprocessing, original-unit metrics, separate model selection and final evaluation, and checked output files. Smoking classification is retained only in the old coursework. It was using charges as an input and needs a separate well-defined decision context before further development.

## Training, validation, calibration and test

Training adjusts a model from examples. Cross-validation repeatedly fits on part of the fitting partition and scores the remaining fold to compare fixed candidates. We then fit the selected candidate on the full fitting partition. Calibration uses separate observations to measure residual sizes and build prediction ranges. Test evaluates the frozen process. Because calibration must remain independent, we do not refit the model on fitting plus calibration records.

## Metrics

MAE is the average absolute prediction error, in dollars. An MAE of $2,309 is not 2,309% and does not guarantee each prediction is within that amount. RMSE squares errors before averaging and taking a square root, making it more sensitive to large misses. R² compares squared error with predicting the test mean; it can be negative and is not classification accuracy.

The median baseline ignores individual features. The selected random forest combines many trees; minimum leaf size limits tiny leaf groups. Ridge is a regularized linear model; histogram gradient boosting adds trees sequentially. We select on CV MAE, not model reputation.

A nominal 90% calibrated prediction range is not a confidence interval for the mean and does not promise 90% coverage for each subgroup or a future population. Read the empirical coverage and average width together. Wide ranges are useful information about uncertainty, not something to hide.

## Read the EDA

The raw charge distribution is right-skewed; a few large costs influence RMSE strongly. The log chart reveals structure while preserving the original-dollar target in the actual pipeline. Age/BMI scatter plots are colored by reported smoking status to show associations and possible interactions. They do not establish that changing one factor causes a particular savings amount.

The CV comparison is the model-selection evidence. Test predicted-versus-actual and signed-error charts expose over- and underestimation. Largest-error rows reveal failures an average can hide. Subgroup tables report sample sizes; small counts should temper conclusions. Permutation importance asks how predictions worsen after shuffling a feature, not why a medical outcome occurred.

## Run locally or in VS Code

Use Python 3.12. VS Code is optional; the same commands work in a terminal.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m medcost.training
python -m pytest -q
python scripts/execute_notebook.py
python -m streamlit run app.py
```

Windows PowerShell activation is `.venv\Scripts\Activate.ps1`. Choose the `.venv` interpreter and notebook kernel in VS Code. A GPU is unnecessary. In resource-constrained Linux/macOS environments prefix training with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.

## Run in Google Colab

Use a CPU runtime and save your notebook copy in Drive. In a new notebook, run:

```python
!git clone https://github.com/HoussamAlrifaii/AI-for-Insurance-Costs-and-Smoking-Behavior.git /content/medcost
%cd /content/medcost
!pip install -r requirements.txt
!OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m medcost.training
```

Use a fresh runtime if you already cloned this directory. Colab package compatibility may differ from the tested Python 3.12 environment; inspect errors rather than assume success. `python scripts/execute_notebook.py` writes an executed notebook and HTML report in the checkout, which you can download through Colab's file browser. Sessions are temporary; downloaded files and saved Drive copies persist separately. Never paste private medical information into the demonstration.

## Current application update

The application now uses the log-target extension rather than the original-dollar target forest. Run `python -m medcost.calibrate_log` to reproduce its fit and calibration; run `python -m streamlit run app.py` to start the interface. The app builds this model itself on first startup, so saved artifacts are optional. The original training command and executed notebook still reproduce the original experiment.

The follow-up candidate's MAE was $1,753, but its test set had already informed the research direction. This is exploratory. High-charge errors and coverage are worse than the overall figures suggest; the app displays this explicitly. See `reports/LOG_TEST_COMPARISON.md`.
