"""Follow-up comparison on an already inspected test partition; no model tuning."""
import json
import pickle
import numpy as np
import pandas as pd
from medcost.data import ROOT, FEATURES, load_data, fingerprint
from medcost.training import split_data, scores


def main():
    _, _, test = split_data(load_data())
    original_metrics = json.loads((ROOT/'reports/metrics.json').read_text())
    if original_metrics['data_sha256'] != fingerprint():
        raise ValueError('Dataset changed since original evaluation')
    results, segments, rows = [], [], []
    for name, filename in [('Original', 'model.pkl'), ('Log target', 'log_candidate.pkl')]:
        # These are locally generated trusted artifacts, never user uploads.
        with (ROOT/'artifacts'/filename).open('rb') as handle:
            bundle = pickle.load(handle)
        metadata = bundle.get('calibration', bundle.get('metrics', {}))
        if metadata.get('data_sha256') != fingerprint():
            raise ValueError('Artifact dataset fingerprint mismatch')
        prediction = bundle['model'].predict(test[FEATURES])
        radius = bundle['radius']
        error = np.abs(test.charges.to_numpy()-prediction)
        lower, upper = np.maximum(0, prediction-radius), prediction+radius
        covered = (test.charges.to_numpy()>=lower)&(test.charges.to_numpy()<=upper)
        metrics = scores(test.charges, prediction)
        results.append({'model': name, 'test_rows': len(test), **metrics,
            'within_1000_pct': float(100*np.mean(error<=1000)),
            'within_2000_pct': float(100*np.mean(error<=2000)),
            'covered_count': int(covered.sum()), 'coverage_pct': float(100*covered.mean()),
            'margin_USD': float(radius), 'mean_interval_width_USD': float(np.mean(upper-lower))})
        if name == 'Original':
            assert abs(metrics['MAE_USD']-original_metrics['test']['MAE_USD'])<1e-8
            assert abs(covered.mean()-original_metrics['interval']['test_coverage'])<1e-10
        for label, mask in [('Below 10000', test.charges<10000),
                            ('10000 to below 20000', (test.charges>=10000)&(test.charges<20000)),
                            ('20000 or more', test.charges>=20000)]:
            segments.append({'model': name, 'actual_charge_group': label, 'n': int(mask.sum()),
                'MAE_USD': float(error[mask].mean()),
                'bias_USD': float((prediction-test.charges.to_numpy())[mask].mean()),
                'coverage_pct': float(100*covered[mask].mean())})
        rows.append(pd.DataFrame({'model': name, 'test_index': test.index,
            'actual': test.charges.to_numpy(), 'prediction': prediction,
            'absolute_error': error, 'lower': lower, 'upper': upper, 'covered': covered}))
    summary=pd.DataFrame(results)
    summary.to_csv(ROOT/'reports/log_test_comparison.csv', index=False)
    pd.DataFrame(segments).to_csv(ROOT/'reports/log_test_segments.csv', index=False)
    pd.concat(rows).to_csv(ROOT/'artifacts/log_test_comparison_predictions.csv', index=False)
    (ROOT/'reports/log_test_protocol.json').write_text(json.dumps({
        'data_sha256': fingerprint(), 'test_rows': len(test), 'model_refitted': False,
        'calibration_recomputed': False, 'application_updated': False,
        'status': 'Exploratory follow-up on an already inspected test set, not independent final confirmation. Do not use these results for further test-driven tuning.'
    },indent=2)+'\n')
    print(summary.round(2).to_string(index=False))
    print('\nBy actual charges:')
    print(pd.DataFrame(segments).round(2).to_string(index=False))

if __name__ == '__main__':
    main()
