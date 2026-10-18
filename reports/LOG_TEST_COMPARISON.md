# Follow-up test comparison: original and log-target models

Run `python -m medcost.evaluate_log` after creating the trusted local artifacts with `python -m medcost.training` and `python -m medcost.calibrate_log`.

This is an exploratory comparison on the same 268 test records previously inspected. The experiment direction was informed by earlier test diagnostics, so this is not fresh independent confirmation. Neither model was refitted and neither margin was changed during evaluation. The app remains unchanged.

| Measure | Original | Log target |
|---|---:|---:|
| MAE | $2,309 | $1,753 |
| RMSE | $3,592 | $3,367 |
| Within $1,000 | 32.5% | 53.4% |
| Within $2,000 | 61.2% | 79.9% |
| Overall nominal 90% range coverage | 244/268 (91.0%) | 245/268 (91.4%) |
| Mean range width after lower-bound clipping | $10,005 | $7,626 |

The overall improvement hides a trade-off. For the 52 records with actual charges >= $20,000, MAE rose from $3,922 to $4,144 and range coverage fell from 76.9% to 71.2%. For records below $10,000, MAE fell from $1,744 to $877. These actual-charge groups are retrospective and cannot be identified from the unknown target at prediction time.

A 90% marginal coverage target does not guarantee 90% coverage within each subgroup. These small-sample results illustrate that distinction directly. Do not claim that either model provides well-calibrated ranges for expensive cases, individual people or current-day populations.

Verification: dataset fingerprints matched both artifacts; original MAE and coverage reproduced the original published report. Saved per-record outputs were used to independently recompute summary MAE and counts. Model, calibration, and production app files were not changed.

Next decisions should use a defined error/cost objective and genuinely independent confirmation rather than further tuning on this test set. Results are educational, not operational pricing validation.
