"""Fit and calibrate the log-target candidate without evaluating test records."""
import json
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from medcost.data import ROOT, FEATURES, load_data, fingerprint
from medcost.training import candidates, pipeline, split_data, conformal_radius


def main():
    fitting, calibration, _ = split_data(load_data())
    model = TransformedTargetRegressor(
        regressor=pipeline(candidates()['Random forest leaf=5']),
        func=np.log1p, inverse_func=np.expm1)
    model.fit(fitting[FEATURES], fitting.charges)
    # Predictions are returned in dollars, so residual calibration is also in dollars.
    prediction = model.predict(calibration[FEATURES])
    radius = conformal_radius(calibration.charges, prediction)
    residuals = np.abs(calibration.charges.to_numpy()-prediction)
    rank = math.ceil((len(calibration)+1)*.9)
    assert abs(np.sort(residuals)[rank-1]-radius)<1e-9
    summary = {
        'candidate': 'Random forest leaf=5 with log1p target',
        'fitting_rows': len(fitting), 'calibration_rows': len(calibration),
        'nominal_coverage': .9, 'residual_rank': rank,
        'radius_USD': radius, 'data_sha256': fingerprint(),
        'test_evaluated': False, 'application_updated': False,
        'limitations': 'Exploratory candidate following prior test diagnostics. Independence and exchangeability assumptions remain necessary. Calibration margin alone is not a performance comparison or individual coverage guarantee.'
    }
    report = ROOT/'reports/log_candidate_calibration.json'
    report.write_text(json.dumps(summary,indent=2)+'\n')
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    with (artifacts/'log_candidate.pkl').open('wb') as handle:
        pickle.dump({'model': model, 'radius': radius, 'calibration': summary},handle)
    pd.DataFrame({'actual': calibration.charges.to_numpy(), 'prediction': prediction,
                  'absolute_error': residuals}).to_csv(artifacts/'log_calibration_predictions.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
