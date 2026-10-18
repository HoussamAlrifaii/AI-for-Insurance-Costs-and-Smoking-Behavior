import json
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from medcost.data import ROOT, FEATURES, NUMERIC, CATEGORICAL, load_data, fingerprint

SEED = 42


def pipeline(model):
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)
    ])
    return Pipeline([('preprocess', preprocessing), ('model', model)])


def candidates():
    return {
        'Median baseline': DummyRegressor(strategy='median'),
        'Ridge alpha=1': Ridge(alpha=1),
        'Ridge alpha=100': Ridge(alpha=100),
        'Random forest leaf=5': RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=2),
        'Gradient boosting leaves=7': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7, l2_regularization=10, early_stopping=False, random_state=SEED),
        'Gradient boosting leaves=15': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=10, early_stopping=False, random_state=SEED)
    }


def split_data(frame):
    development, test = train_test_split(frame, test_size=.2, random_state=SEED, stratify=frame['smoker'])
    fit, calibration = train_test_split(development, test_size=.25, random_state=SEED, stratify=development['smoker'])
    return fit, calibration, test


def scores(actual, prediction):
    return {'MAE_USD': float(mean_absolute_error(actual, prediction)),
            'RMSE_USD': float(np.sqrt(mean_squared_error(actual, prediction))),
            'R2': float(r2_score(actual, prediction))}


def conformal_radius(actual, prediction, alpha=.1):
    residuals = np.sort(np.abs(np.asarray(actual)-np.asarray(prediction)))
    rank = math.ceil((len(residuals)+1)*(1-alpha))
    if rank > len(residuals):
        raise ValueError('Insufficient calibration observations')
    return float(residuals[rank-1])


def train():
    frame = load_data()
    fit, calibration, test = split_data(frame)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted = {}
    comparison = []
    for name, estimator in candidates().items():
        model = pipeline(estimator)
        losses = -cross_val_score(model, fit[FEATURES], fit.charges, cv=cv,
                                  scoring='neg_mean_absolute_error', n_jobs=1)
        comparison.append({'model': name, 'CV_MAE_USD': float(losses.mean()),
                           'fold_MAE_SD_USD': float(losses.std(ddof=1))})
        fitted[name] = model
    comparison.sort(key=lambda row: row['CV_MAE_USD'])
    selected = comparison[0]['model']
    model = fitted[selected].fit(fit[FEATURES], fit.charges)
    # No refit on calibration data: retain its independence for interval calibration.
    radius = conformal_radius(calibration.charges, model.predict(calibration[FEATURES]))
    prediction = model.predict(test[FEATURES])
    lower, upper = np.maximum(0, prediction-radius), prediction+radius
    baseline = pipeline(DummyRegressor(strategy='median')).fit(fit[FEATURES], fit.charges)
    output = test.copy()
    output['prediction'] = prediction
    output['lower_90'] = lower
    output['upper_90'] = upper
    output['absolute_error'] = np.abs(output.charges-output.prediction)
    output['covered'] = (output.charges >= lower) & (output.charges <= upper)
    metrics = {
        'selected_model': selected, 'seed': SEED, 'data_sha256': fingerprint(),
        'rows': {'deduplicated': len(frame), 'fit': len(fit), 'calibration': len(calibration), 'test': len(test)},
        'cross_validation': comparison, 'test': scores(test.charges, prediction),
        'test_median_baseline': scores(test.charges, baseline.predict(test[FEATURES])),
        'interval': {'nominal_coverage': .9, 'test_coverage': float(output.covered.mean()),
                     'radius_USD': radius, 'mean_width_USD': float(np.mean(upper-lower))}
    }
    groups = []
    for feature in ['smoker', 'sex', 'region']:
        for value, group in output.groupby(feature):
            groups.append({'feature': feature, 'group': value, 'n': len(group),
                'MAE_USD': float(group.absolute_error.mean()),
                'bias_USD': float((group.prediction-group.charges).mean()),
                'interval_coverage': float(group.covered.mean())})
    importance = permutation_importance(model, test[FEATURES], test.charges,
                    scoring='neg_mean_absolute_error', n_repeats=10, random_state=SEED, n_jobs=1)
    return {'model': model, 'radius': radius, 'metrics': metrics}, output, pd.DataFrame(groups), pd.DataFrame({
        'feature': FEATURES, 'MAE_increase_USD': importance.importances_mean,
        'repeat_SD_USD': importance.importances_std})


def main():
    bundle, predictions, groups, importance = train()
    artifacts = ROOT / 'artifacts'
    reports = ROOT / 'reports'
    artifacts.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    with (artifacts/'model.pkl').open('wb') as handle:
        pickle.dump(bundle, handle)
    (reports/'metrics.json').write_text(json.dumps(bundle['metrics'], indent=2)+'\n')
    predictions.to_csv(artifacts/'test_predictions.csv', index=False)
    groups.to_csv(reports/'subgroup_errors.csv', index=False)
    importance.to_csv(reports/'permutation_importance.csv', index=False)
    print(json.dumps(bundle['metrics'], indent=2))

if __name__ == '__main__':
    main()
