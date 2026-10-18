# Calibrating the log-target candidate

Reproduce with `python -m medcost.calibrate_log`. The candidate is fitted on the original 801 fitting records. It is not refitted on calibration rows. Predictions are converted back to dollars before computing absolute residuals on the separate 268 calibration records.

The same finite-sample rank rule selects residual 243 for nominal 90% coverage. The resulting dollar margin is **$3,916.74**, compared with **$5,141.97** for the original model. At a hypothetical point prediction of $10,000, these margins would produce ranges of approximately $6,083–$13,917 and $4,858–$15,142 respectively. The point predictions themselves can differ between models; this illustration holds them equal solely to explain the margins.

A smaller calibration margin is not proof of better coverage, individual certainty, or current-day validity. Log-target CV improved overall MAE but slightly worsened expensive-case errors. The calibration partition is small, and subgroup coverage is not guaranteed. This is an exploratory follow-up after inspection of earlier test diagnostics, not an independent final performance claim.

This step did not evaluate the candidate on the test partition. It did not replace the app's model or change the original metrics. The separate local artifact is saved as `artifacts/log_candidate.pkl`; never load untrusted pickle artifacts. Rank and quantile equality were checked independently against sorted calibration residuals.
