import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from medcost.data import FEATURES, load_data
from medcost.training import pipeline, split_data, conformal_radius


def test_splits_are_disjoint_and_target_is_excluded():
    f=load_data();a,b,c=split_data(f)
    assert not set(a.index)&set(b.index)
    assert not set(a.index)&set(c.index)
    assert not set(b.index)&set(c.index)
    assert len(a)+len(b)+len(c)==len(f)
    assert 'charges' not in FEATURES


def test_imputation_learns_training_only_and_unknown_category_works():
    f=load_data().iloc[:30].copy()
    f.loc[f.index[0],'bmi']=np.nan
    m=pipeline(Ridge()).fit(f[FEATURES],f.charges)
    imputer=m.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    assert imputer.statistics_[1]==f.bmi.median()
    row=f[FEATURES].iloc[[0]].copy();row['region']='unseen'
    assert np.isfinite(m.predict(row)).all()


def test_finite_sample_conformal_quantile():
    assert conformal_radius(np.arange(1,11),np.zeros(10))==10
