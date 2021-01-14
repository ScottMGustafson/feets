import dask.dataframe as dd
import pandas as pd
from tests import make_data
from feets.random_feats import _flag_important, add_random_feats


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


def test_add_random():
    df = pd.DataFrame(dict(a=list(range(1000))))
    df, new_cols = add_random_feats(df, num_new_feats=30)
    for col in new_cols:
        assert col in df.columns


def test_add_random_dd():
    df = make_data.make_fake_data(to_pandas=False)
    df, new_cols = add_random_feats(df, num_new_feats=30)
    df = df.compute()
    for col in new_cols:
        assert col in df.columns
