"""Wrapper class around xgboost to allow user to pass in non-dask collections and still get a model"""

import dask.array as da
import dask.dataframe as dd
import pandas as pd
import xgboost.dask as dxgb


def _add_simple_index(ddf, ix_name="ix", temp_name="_ones"):
    assert (
        temp_name not in ddf.columns
    ), "can't add index, temporary name ``{}`` already in columns".format(temp_name)
    ddf[temp_name] = 1
    ddf[ix_name] = ddf[temp_name].cumsum()
    ddf = ddf.drop(temp_name, axis=1)
    ddf = ddf.set_index(ix_name, drop=True)
    return ddf


def _to_dask_array(X, npartitions=1):
    if isinstance(X, pd.DataFrame):
        ddf = dd.from_pandas(X, npartitions=npartitions)
    elif isinstance(X, pd.Series):
        ddf = dd.from_pandas(pd.DataFrame(X), npartitions=npartitions)
    else:
        if isinstance(X, da.Array):
            return X
        ddf = X
        if not (isinstance(ddf, dd.DataFrame) or isinstance(ddf, dd.Series)):
            raise TypeError("Unsupported type: {}".format(type(ddf)))
    if not ddf.known_divisions:
        ddf = _add_simple_index(ddf)
    return ddf.to_dask_array(lengths=True).persist()


def _to_dask_dataframe(X, npartitions=1):
    if isinstance(X, pd.DataFrame):
        ddf = dd.from_pandas(X, npartitions=npartitions)
    elif isinstance(X, pd.Series):
        ddf = dd.from_pandas(pd.DataFrame(X), npartitions=npartitions)
    else:
        if isinstance(X, da.Array):
            ddf = dd.from_dask_array(X)
        else:
            ddf = X
        if not (isinstance(ddf, dd.DataFrame) or isinstance(ddf, dd.Series)):
            raise TypeError("Unsupported type: {}".format(type(ddf)))
    if not ddf.known_divisions:
        ddf = _add_simple_index(ddf)
    return ddf.persist()


class DaskXGBoostClassifier(dxgb.DaskXGBClassifier):
    def __init__(self, *args, **kwargs):
        super(DaskXGBoostClassifier, self).__init__(*args, **kwargs)

    def fit(self, X, y, **fit_params):
        X = _to_dask_dataframe(X)
        y = _to_dask_dataframe(y)
        return super(DaskXGBoostClassifier, self).fit(X, y, **fit_params)

    def predict(self, X, **kwargs):
        X = _to_dask_dataframe(X)
        return super(DaskXGBoostClassifier, self).predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        X = _to_dask_dataframe(X)
        return super(DaskXGBoostClassifier, self).predict_proba(X, **kwargs)


class DaskXGBoostRegressor(dxgb.DaskXGBRegressor):
    def __init__(self, *args, **kwargs):
        super(DaskXGBoostRegressor, self).__init__(*args, **kwargs)

    def fit(self, X, y, **fit_params):
        X = _to_dask_dataframe(X)
        y = _to_dask_dataframe(y)
        return super(DaskXGBoostRegressor, self).fit(X, y, **fit_params)

    def predict(self, X, **kwargs):
        X = _to_dask_dataframe(X)
        return super(DaskXGBoostRegressor, self).predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        X = _to_dask_dataframe(X)
        return super(DaskXGBoostRegressor, self).predict_proba(X, **kwargs)
