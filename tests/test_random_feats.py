from feets.random_feats import _flag_important
from feets import random_feats
from unittest import mock


@mock.patch(
    "feets.random_feats.add_random_feats",
    return_value=("fake_df", ["random1", "random2", "random3"]),
)
@mock.patch("feets.random_feats.kfold_split_train", return_value=["fake", "model", "objects"])
@mock.patch("feets.random_feats.flag_important", return_value=["a", "b", "c"])
@mock.patch(
    "feets.random_feats.get_mean_importance_dct", return_value={"a": 1, "b": 2, "c": 3, "d": 4}
)
def test_run_gbm_feats(mock_imp_dct, mock_import_list, mock_models, mock_random_feat_output):
    output_dct = random_feats.run_random_feats(
        "fake_df", ["a", "b", "c", "d"], "target", "model_class"
    )
    mock_random_feat_output.assert_called_with("fake_df", num_new_feats=10)
    mock_models.assert_called_with(
        "fake_df", "target", ["a", "b", "c", "d"] + ["random1", "random2", "random3"], "model_class"
    )
    mock_import_list.assert_called_with(
        ["fake", "model", "objects"],
        ["a", "b", "c", "d"],
        ["random1", "random2", "random3"],
        importance_type="gain",
        num_random_cols_to_beat=9,
        min_num_folds=4,
    )
    mock_imp_dct.assert_called_with(["fake", "model", "objects"], importance_type="gain")
    assert output_dct == {"a": 1, "b": 2, "c": 3}


def make_fake_importance_dct():
    dct = {k: i for i, k in enumerate("abcdefghijklmnop")}
    random_cols = []
    for i in range(10):
        col = f"random_{i}"
        random_cols.append(col)
        dct[col] = i
    return random_cols, dct


def test_flag_important_no_drop():
    random_cols, dct = make_fake_importance_dct()
    results = _flag_important(dct, list("abcdefghijklmnop"), random_cols, num_random_cols_to_beat=9)
    expected = {"l": 11, "m": 12, "j": 9, "k": 10, "n": 13, "o": 14, "p": 15}
    assert results == expected


def test_flag_important_drop_all():
    random_cols, dct = make_fake_importance_dct()
    dct = {k: v for k, v in dct.items() if not k.startswith("random")}
    results = _flag_important(dct, list("abcdefghijklmnop"), random_cols, num_random_cols_to_beat=9)
    expected = {k: i for i, k in enumerate("abcdefghijklmnop")}
    assert results == expected


def test_flag_important_drop_some():
    random_cols, dct = make_fake_importance_dct()
    del dct["random_0"]
    del dct["random_1"]
    results = _flag_important(dct, list("abcdefghijklmnop"), random_cols, num_random_cols_to_beat=9)
    expected = {"l": 11, "m": 12, "j": 9, "k": 10, "n": 13, "o": 14, "p": 15}
    assert results == expected
