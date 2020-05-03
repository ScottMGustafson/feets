import xgboost as xgb
import numpy as np
from sklearn import model_selection


def add_random_feats(df, num_new_feats=10):
    new_cols = ["random_feat_{}".format(i) for i in range(num_new_feats)]
    for col in new_cols:
        df[col] = np.random.sample(df.index.size)
    return df, new_cols


def kfold_split_train(df, target, feats, model_class, **kwargs):
    assert model_class in [xgb.XGBClassifier, xgb.XGBRegressor]
    kfold_kwargs = kwargs.get("kfold_kwargs")
    if not kfold_kwargs:
        kfold_kwargs = dict(n_splits=5, shuffle=False, random_state=0)
    model_kwargs = kwargs.get("model_kwargs", {})

    X = df[feats]
    y = df[target]

    kf = model_selection.KFold(**kfold_kwargs)
    model_objects = []
    for train_index, test_index in kf.split(X.values):
        X_train, y_train = X.iloc[train_index], y[train_index]
        mod = model_class(**model_kwargs)
        mod.fit(X_train, y_train)
        model_objects.append(mod)
    return model_objects


def _flag_important(import_dct, feats, random_cols, num_random_cols_to_beat=9):
    assert num_random_cols_to_beat <= len(random_cols), (
        f"num_random_cols_to_beat ({num_random_cols_to_beat}) must be"
        + f" greater than number of random cols {len(random_cols)}"
    )
    random_importance = sorted([x for k, x in import_dct.items() if k in random_cols])
    random_thresh = random_importance[num_random_cols_to_beat - 1]
    return {k: v for k, v in import_dct.items() if k in feats and v > random_thresh}


def flag_important(
    model_objects,
    feats,
    random_cols,
    importance_type="gain",
    num_random_cols_to_beat=9,
    min_num_folds=4,
):
    feat_dict = {}
    assert min_num_folds <= len(
        model_objects
    ), "min_num_folds cannot be less than the number of folds."
    feat_count = {x: 0 for x in feats}
    for i, mod in enumerate(model_objects):
        import_dct = mod.get_booster().get_score(importance_type=importance_type)
        feat_dict[i] = _flag_important(
            import_dct, feats, random_cols, num_random_cols_to_beat=num_random_cols_to_beat
        )
        for x in feat_dict[i].keys():
            feat_count[x] += 1

    return [k for k, v in feat_count.items() if v > min_num_folds]


def run_gbm_random_feats(df, features, target, model_class, **kwargs):
    df, random_cols = add_random_feats(df, num_new_feats=kwargs.get("num_new_feats", 10))
    model_objects = kfold_split_train(df, target, features + random_cols, model_class, **kwargs)
    import_feats = flag_important(
        model_objects,
        features,
        random_cols,
        importance_type=kwargs.get("importance_type", "gain"),
        num_random_cols_to_beat=kwargs.get("num_random_cols_to_beat", 9),
        min_num_folds=kwargs.get("min_num_folds", 4),
    )
    return import_feats
