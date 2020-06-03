import numpy as np

TYPE_MAPPING = [("numeric", "float64"), ("datetime", "M8[us]"), ("object", "object")]


def _validate_mappings(lst):
    allowed_types = set([x[0] for x in TYPE_MAPPING])
    diff = set(lst).difference(allowed_types)
    assert not diff, "{} not recognized type mappings: {}".format(diff, allowed_types)


def classify_feature_types(df, feats=None):
    """
    get inferred feature types by trying df.astype.
    This will try to cast each type as either
        - numeric: encompassing, int, float and bool types or anything else that
            can be successfully cast as float
        - datetime: anything that pandas can successfully cast to datetime
        - object: anything else, typically treated as a string

    Parameters
    ----------
    df : dataframe
    feats : list (optional)
        list of features

    Returns
    -------
    dict
        column name : inferred type (numeric, datetime or object)
    """

    def test_type(ser, _type):
        try:
            _ = ser.astype(_type)
            return True
        except (ValueError, TypeError):
            return False

    if not feats:
        feats = df.columns.tolist()

    _types = {}

    for col in feats:
        for k, _type in TYPE_MAPPING:
            if test_type(df[col], _type):
                _types[col] = k
                break
    return _types


def classify_value_counts(df, col, unique_thresh=0.05, type_dct=None):
    """
    Infer whether a feature is continuous, categorical, uninformative or binary

    Parameters
    ----------
    df : dataframe
    col : str
    unique_thresh : int or float
        threshold of unique values to determine whether a numeric column
        is categorical or continuous.
        If ``unique_thresh > 1``, then this is assumed to be a raw number of
         unique value counts:
            - if length of ``df[col].value_counts() > unique_thresh``, then
            ``col`` is inferred to be continuous, otherwise categorical.
        if ``unique_thresh < 1``, this is assumed to be a percentage, i.e.
            - if ``df[col].value_counts() > unique_thresh * len(df[col])`` is
            inferred to be continuous, else categorical
    type_dct : dict
        this is a mapping of columns to allowed types.

    Returns
    -------
    str
        classification of the feature type.  This will be one of
        ``{'null', 'uninformative', 'binary', 'continuous', 'categorical'}``

    """
    val_counts = df[col].dropna().value_counts()
    if val_counts.empty:
        return "null"
    elif val_counts.size == 1:
        return "uninformative"
    elif val_counts.size == 2:
        return "binary"
    else:
        if not type_dct:
            type_dct = classify_feature_types(df[[col]])
        _validate_mappings(list(type_dct.values()))
        if type_dct[col] == "numeric":
            assert unique_thresh > 0
            if unique_thresh < 1.0:
                unique_thresh = int(unique_thresh * df.index.size)
            if len(val_counts) > unique_thresh:
                return "continuous"
    return "categorical"


def process_feats(df, unique_thresh=0.01, feats=None):
    feat_type_dct = classify_feature_types(df, feats=feats)
    feat_class_dct = {
        k: classify_value_counts(df, k, unique_thresh=unique_thresh, type_dct=feat_type_dct)
        for k in df.columns
    }
    return feat_type_dct, feat_class_dct


def get_correlates(df, thresh=0.9, feats=None, **corr_kwargs):
    """
    get correlate pairs with a correlation coeff greater that ``thresh``

    Parameters
    ----------
    df : dataframe
    thresh : float (0 -> 1)
    feats : list
        list of column names

    Other Parameters
    ----------------
    See parameters for ``pd.DataFrame.corr``

    Returns
    -------
    pd.Series
    """
    if not feats:
        # remove object and datetime types (not comprehensive).
        feats = [f for f in df.columns.tolist() if df[f].dtype not in ["object", "<M8[ns]"]]

    corr_matrix = df[feats].corr(**corr_kwargs).abs()
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        .stack()
        .sort_values(ascending=False)
    )
    return corr_pairs[corr_pairs > thresh]


def get_high_corr_cols(df, rho_thresh, method="spearman"):
    corr_matrix = df.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(upper[column] > rho_thresh)]
