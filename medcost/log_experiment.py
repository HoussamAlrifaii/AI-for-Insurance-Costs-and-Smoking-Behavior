"""Exploratory target transformation comparison on fitting records only."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from medcost.data import ROOT, FEATURES, load_data, fingerprint
from medcost.training import SEED, candidates, pipeline, split_data, scores


def main():
    fit, _, _ = split_data(load_data())
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(fit))
    rows = []
    segments = []
    for name, estimator in candidates().items():
        if name == 'Median baseline':
            continue
        for transform in ['original', 'log1p']:
            prediction = np.empty(len(fit))
            losses = []
            for training, validation in folds:
                model = pipeline(clone(estimator))
                if transform == 'log1p':
                    model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
                model.fit(fit.iloc[training][FEATURES], fit.iloc[training].charges)
                predicted = model.predict(fit.iloc[validation][FEATURES])
                prediction[validation] = predicted
                losses.append(float(np.abs(fit.iloc[validation].charges.to_numpy()-predicted).mean()))
            error = np.abs(fit.charges.to_numpy()-prediction)
            rows.append({'model': name, 'target': transform, 'CV_MAE_USD': float(np.mean(losses)),
                'fold_MAE_SD_USD': float(np.std(losses, ddof=1)),
                **{f'OOF_{key}': value for key, value in scores(fit.charges, prediction).items()},
                'within_1000_pct': float(100*np.mean(error<=1000)),
                'within_2000_pct': float(100*np.mean(error<=2000))})
            for label, mask in [('Below 10000', fit.charges<10000),
                                ('10000 to below 20000', (fit.charges>=10000)&(fit.charges<20000)),
                                ('20000 or more', fit.charges>=20000)]:
                segments.append({'model': name, 'target': transform, 'actual_charge_group': label,
                                 'n': int(mask.sum()), 'OOF_MAE_USD': float(error[mask].mean())})
    report = ROOT/'reports'
    results = pd.DataFrame(rows).sort_values('CV_MAE_USD')
    results.to_csv(report/'log_target_experiment.csv', index=False)
    pd.DataFrame(segments).to_csv(report/'log_target_segments.csv', index=False)
    (report/'log_target_protocol.json').write_text(json.dumps({
        'fitting_rows': len(fit), 'seed': SEED, 'data_sha256': fingerprint(),
        'folds': 5, 'calibration_and_test_used_for_scoring': False,
        'status': 'Exploratory follow-up motivated by previously inspected test diagnostics; no new independent final performance claim.',
        'interpretation': 'Scores in original dollar units after inverse transformation. Log predictions are not bias-corrected conditional means. Deployed model and calibration remain unchanged.'
    }, indent=2)+'\n')
    print(results[['model','target','CV_MAE_USD','OOF_RMSE_USD','within_1000_pct','within_2000_pct']].round(2).to_string(index=False))

if __name__ == '__main__':
    main()
