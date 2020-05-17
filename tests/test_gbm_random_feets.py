from feets.gbm_random_feats import _flag_important


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
