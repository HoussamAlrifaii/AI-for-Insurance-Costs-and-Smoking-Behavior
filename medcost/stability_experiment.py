"""Paired repeated cross-validation on the original fitting partition only."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from medcost.data import ROOT, FEATURES, load_data, fingerprint
from medcost.training import candidates, pipeline, split_data


def main():
    fitting, _, _ = split_data(load_data())
    estimator = candidates()['Random forest leaf=5']
    results = []
    for seed in [7, 19, 42, 73, 101]:
        folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(fitting))
        for target in ['original', 'log1p']:
            prediction = np.empty(len(fitting))
            fold_errors = []
            for train_index, validation_index in folds:
                model = pipeline(clone(estimator))
                if target == 'log1p':
                    model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
                model.fit(fitting.iloc[train_index][FEATURES], fitting.iloc[train_index].charges)
                values = model.predict(fitting.iloc[validation_index][FEATURES])
                prediction[validation_index] = values
                fold_errors.append(float(np.abs(values-fitting.iloc[validation_index].charges).mean()))
            actual = fitting.charges.to_numpy()
            error = np.abs(actual-prediction)
            results.append({'fold_seed': seed, 'target': target, 'CV_MAE_USD': float(np.mean(fold_errors)),
                'OOF_MAE_USD': float(error.mean()),
                'within_1000_pct': float(100*np.mean(error<=1000)),
                'within_2000_pct': float(100*np.mean(error<=2000)),
                'high_charge_n': int(np.sum(actual>=20000)),
                'high_charge_MAE_USD': float(error[actual>=20000].mean())})
    output = pd.DataFrame(results)
    output.to_csv(ROOT/'reports/stability_experiment.csv', index=False)
    paired = output.pivot(index='fold_seed', columns='target', values='CV_MAE_USD')
    paired['reduction_pct'] = 100*(1-paired.log1p/paired.original)
    paired.to_csv(ROOT/'reports/stability_paired.csv')
    protocol = {'fitting_rows': len(fitting), 'data_sha256': fingerprint(),
        'fold_seeds': [7,19,42,73,101], 'folds_per_repeat': 5, 'forest_seed': 42,
        'calibration_and_test_scored': False,
        'caveat': 'Repeated folds share observations and are not independent new datasets. Exploratory follow-up after prior test diagnostics, not independent final confirmation. Application model unchanged.'}
    (ROOT/'reports/stability_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(paired.round(2).to_string())
    print('\nRepeated-CV averages (not independent-sample confidence intervals):')
    print(output.groupby('target')[['CV_MAE_USD','within_1000_pct','within_2000_pct','high_charge_MAE_USD']].mean().round(2).to_string())

if __name__ == '__main__':
    main()
