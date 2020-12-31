"""
Feature importance by comparing randomly generated features importance over K Folds to those of actual data.
"""

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import model_selection

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
    for col in new_cols:
        df[col] = da.random.random(df.index.size)
    return df, new_cols


def kfold_split_train(df, target, feats, model_class, **kwargs):
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

    ddf = _to_dask_dataframe(df)
    X = ddf[feats]
    y = ddf[target]

    fit_params = kwargs.get("fit_params", {})
    kf = model_selection.KFold(**kfold_kwargs)
    model_objects = []

    for train_index, test_index in kf.split(X.values):
        X_train, y_train = X.iloc[train_index], y[train_index]
        mod = model_class(**model_kwargs)
        mod.fit(X_train, y_train, **fit_params)
        model_objects.append(mod)
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


def run_random_feats(df, features, target, model_class, **kwargs):
    """

    Parameters
    ----------
    df
    features
    target
    model_class

    Other Parameters
    ----------------
    num_new_feats : int
    importance_type : str
    num_random_cols_to_beat : int
    min_num_folds : int
    kfold_kwargs : dict
        kwargs for sklearn.model_selection.KFold
    model_kwargs : dict
        kwargs for model class.

    Returns
    -------
    dict
    """
    if isinstance(df, pd.DataFrame):
        ddf = _to_dask_dataframe(df, npartitions=kwargs.get("npartitions", 2))
    else:
        ddf = df
        assert isinstance(ddf, dd.DataFrame), f"{type(ddf)} not supported."

    ddf, random_cols = add_random_feats(ddf, num_new_feats=kwargs.get("num_new_feats", 10))
    ddf = ddf.dropna(subset=[target])
    model_objects = kfold_split_train(ddf, target, features + random_cols, model_class, **kwargs)
    importance_type = kwargs.get("importance_type", "gain")
    import_feats = flag_important(
        model_objects,
        features,
        random_cols,
        importance_type=importance_type,
        num_random_cols_to_beat=kwargs.get("num_random_cols_to_beat", 9),
        min_num_folds=kwargs.get("min_num_folds", 4),
    )
    dct = get_mean_importance_dct(model_objects, features, importance_type=importance_type)
    dct = {k: v for k, v in dct.items() if k in import_feats}
    return dct
