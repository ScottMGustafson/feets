"""
Feature importance by comparing randomly generated features importance over K Folds to those of actual data.
"""

import dask.array as da
import dask.dataframe as dd
import dask_ml
import numpy as np
import pandas as pd
import sklearn

from feets._dask_utils import _to_dask_dataframe


def add_random_feats(df, num_new_feats=10):
    """
    Add random features to dataframe

    Parameters
    ----------
    df : dataframe
    num_new_feats : int

    Returns
    -------
    dataframe, list
    """
    new_cols = ["random_feat_{}".format(i) for i in range(num_new_feats)]
    if isinstance(df, dd.DataFrame):
        sz = df.index.size.compute()
        arr = da.random.random((sz, len(new_cols)))
        new_df = dd.from_dask_array(arr, columns=new_cols)
        df = df.merge(new_df, how="left", left_index=True, right_index=True)
    else:
        for col in new_cols:
            df[col] = np.random.random(df.index.size)
    return df, new_cols


def _dask_kfold(df, feats, target, model_class, kfold_kwargs, model_kwargs, fit_params):
    ddf = _to_dask_dataframe(df).dropna(subset=[target])
    X = ddf[feats].to_dask_array(lengths=True)
    y = ddf[target].to_dask_array(lengths=True)

    kf = dask_ml.model_selection.KFold(**kfold_kwargs)
    model_objects = []

    for train_ix, test_ix in kf.split(X, y):
        mod = model_class(**model_kwargs)
        X_train, y_train = X[train_ix], y[train_ix]
        mod.fit(X_train, y_train, **fit_params)
        mod.feature_names = feats
        mod.get_booster().feature_names = feats
        model_objects.append(mod)
    return model_objects


def _pandas_kfold(df, feats, target, model_class, kfold_kwargs, model_kwargs, fit_params):
    X = df[feats]
    y = df[target]

    kf = sklearn.model_selection.KFold(**kfold_kwargs)
    model_objects = []

    for train_index, test_index in kf.split(X.values):
        X_train, y_train = X.iloc[train_index], y[train_index]
        mod = model_class(**model_kwargs)
        mod.fit(X_train, y_train, **fit_params)
        model_objects.append(mod)
    return model_objects


def kfold_split_train_cv(df, target, feats, model_class, is_dask=True, **kwargs):
    """

    Parameters
    ----------
    df : dataframe
        dask or pandas dataframe
    target : str
        target variable
    feats : list
        list of features
    model_class : class
        class of model having both .fit and .predict methods
    is_dask : bool, default=True
        if true, will convert any dataframes to dask and run with dask_ml.model_selection.KFold
        rather than the sklearn equivalent

    Other Parameters
    ----------------
    kfold_kwargs : dict
        kwargs for skleanr.model_selection.KFold
    model_kwargs : dict
        kwargs for model class.
    fit_params : dict
        params for fit

    Returns
    -------
    list
        list of trained model objects
    """
    kfold_kwargs = kwargs.get("kfold_kwargs")
    if not kfold_kwargs:
        kfold_kwargs = dict(n_splits=5, shuffle=True, random_state=0)
    model_kwargs = kwargs.get("model_kwargs", {})
    fit_params = kwargs.get("fit_params", {})

    if is_dask:
        model_objects = _dask_kfold(
            df, feats, target, model_class, kfold_kwargs, model_kwargs, fit_params
        )
    else:
        model_objects = _pandas_kfold(
            df, feats, target, model_class, kfold_kwargs, model_kwargs, fit_params
        )

    return model_objects


def _flag_important(import_dct, feats, random_cols, num_random_cols_to_beat=9):
    assert num_random_cols_to_beat <= len(random_cols), (
        f"num_random_cols_to_beat ({num_random_cols_to_beat}) must be"
        + f" greater than number of random cols {len(random_cols)}"
    )
    random_importance = sorted([x for k, x in import_dct.items() if k in random_cols])

    if len(random_importance) == 0:
        # case where get_scores returns none of the random cols
        return {k: v for k, v in import_dct.items() if k in feats}
    # case where get_score drops a few, but not all
    diff = len(random_cols) - len(random_importance)
    ix = max(num_random_cols_to_beat - 1 - diff, 0)
    random_thresh = random_importance[ix]
    return {k: v for k, v in import_dct.items() if k in feats and v > random_thresh}


def flag_important(
    model_objects,
    feats,
    random_cols,
    importance_type="gain",
    num_random_cols_to_beat=9,
    min_num_folds=4,
):
    """
    Get important parameters.

    Parameters
    ----------
    model_objects : list
        list of model objects
    feats : list
        list of features
    random_cols : list
        list of random column names generated in add_random_feats
    importance_type : str ("gain")
        importance type recognized by xgboost get_score.  if not xgboost, this is ignored
    num_random_cols_to_beat : int (9)
        number of random column any give column must beat in importance
    min_num_folds : int (4)
        min number of folds that a surviving feature must appear in to be
        considered important

    Returns
    -------
    list
        list of important features

    """
    feat_dict = {}
    assert min_num_folds <= len(
        model_objects
    ), "min_num_folds cannot be less than the number of folds."
    feat_count = {x: 0 for x in feats}
    for i, mod in enumerate(model_objects):
        import_dct = get_import_dct(mod, feats, importance_type=importance_type)
        feat_dict[i] = _flag_important(
            import_dct, feats, random_cols, num_random_cols_to_beat=num_random_cols_to_beat
        )
        for x in feat_dict[i].keys():
            feat_count[x] += 1

    return [k for k, v in feat_count.items() if v > min_num_folds]


def get_mean_dct(dct_lst):
    dct = {}
    for _dct in dct_lst:
        for k, v in _dct.items():
            try:
                dct[k].append(v)
            except KeyError:
                dct[k] = [v]
    return {k: np.mean(v) for k, v in dct.items()}


def get_import_dct(mod, feats, importance_type="gain"):
    if hasattr(mod, "get_booster"):
        return mod.get_booster().get_score(importance_type=importance_type)
    return dict(zip(list(feats), list(mod.feature_importances_)))


def get_mean_importance_dct(model_objects, feats, importance_type="gain"):
    dct_lst = [get_import_dct(mod, feats, importance_type=importance_type) for mod in model_objects]
    mean_dct = get_mean_dct(dct_lst)
    return mean_dct


def run_random_feats(
    df,
    features,
    target,
    model_class,
    num_new_feats=10,
    num_random_cols_to_beat=9,
    min_num_folds=4,
    importance_type="gain",
    **kwargs,
):
    """

    Parameters
    ----------
    df
    features
    target
    model_class
    num_new_feats : int, default=10
    num_random_cols_to_beat : int, default=9
    min_num_folds : int, default=4
    importance_type : str, default="gain"


    Other Parameters
    ----------------
    kfold_kwargs : dict
        kwargs for sklearn.model_selection.KFold
    model_kwargs : dict
        kwargs for model class.
    npartitions : int, default=2
        number of dask partitions if converting from pandas

    Returns
    -------
    dict
    """
    if isinstance(df, pd.DataFrame):
        ddf = _to_dask_dataframe(df, npartitions=kwargs.get("npartitions", 2)).repartition(
            partition_size="100MB"
        )
    else:
        ddf = df
        assert isinstance(ddf, dd.DataFrame), f"{type(ddf)} not supported."

    ddf, random_cols = add_random_feats(ddf, num_new_feats=num_new_feats)
    ddf = ddf.dropna(subset=[target])
    model_objects = kfold_split_train_cv(ddf, target, features + random_cols, model_class, **kwargs)
    import_feats = flag_important(
        model_objects,
        features,
        random_cols,
        importance_type=importance_type,
        num_random_cols_to_beat=num_random_cols_to_beat,
        min_num_folds=min_num_folds,
    )
    dct = get_mean_importance_dct(model_objects, features, importance_type=importance_type)
    dct = {k: v for k, v in dct.items() if k in import_feats}
    return dct, model_objects
