from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
NUMERIC = ['age', 'bmi', 'children']
CATEGORICAL = ['sex', 'smoker', 'region']


def load_data():
    frame = pd.read_csv(ROOT / 'insurance.csv')
    required = FEATURES + ['charges']
    if set(frame.columns) != set(required):
        raise ValueError('Unexpected dataset schema')
    if frame['charges'].isna().any() or (frame['charges'] < 0).any():
        raise ValueError('Invalid target')
    # Remove exact repeated records before splitting to avoid train/test copies.
    # Whether identical rows are the same person is not established by this dataset.
    return frame.drop_duplicates().reset_index(drop=True)


def fingerprint():
    return hashlib.sha256((ROOT / 'insurance.csv').read_bytes()).hexdigest()
