"""
uses sklearn + eli5 permutation importance to eliminate
all feats with importance less than a standard deviation above zero
"""
from eli5.sklearn import PermutationImportance


class PermutationFeatElim:
    def __init__(
        self, target_var, xgb_params, model_class, **kwargs,
    ):
        self.target_var = target_var
        self.xgb_params = xgb_params
        self.ignore_feats = kwargs.get("ignore_feats", [])
        self.xgb_class = model_class
        self.mandatory_feats = kwargs.get("mandatory_feats", [])
        self.initial_feats = kwargs.get("initial_feats")
        self.drop = kwargs.get("drop", False)
        self.train_sample = kwargs.get("train_sample", 0.7)
        self.remaining_feats = None
        raise NotImplementedError("Not yet implemented.")

    def fit(self, X, y=None):
        return self

    def _set_defaults(self, X):
        if not self.initial_feats:
            self.initial_feats = [x for x in X.columns if x not in self.ignore_feats]
            assert len(self.initial_feats) > 1, "must have more than one feature to eliminate"

    @staticmethod
    def _split_tt(df, frac, seed=0):
        train_df = df.sample(frac=frac, random_state=seed)
        test_df = df.loc[~(df.index.isin(train_df.index))]
        return train_df, test_df

    def transform(self, X):
        self._set_defaults(X)
        train_df, test_df = PermutationFeatElim._split_tt(X, self.train_sample)
        mod = self.xgb_class(**self.xgb_params).fit(
            train_df[self.initial_feats], train_df["target"]
        )
        perm = PermutationImportance(mod).fit(test_df[self.initial_feats], test_df["target"])
        feats = [
            self.initial_feats[i]
            for i, x in enumerate(perm.feature_importances_ - perm.feature_importances_std_)
            if x > 0
        ]
        _keep = list(set(self.mandatory_feats + self.ignore_feats + [self.target_var]))
        self.remaining_feats = [x for x in feats if x not in self.ignore_feats + [self.target_var]]
        cols = list(set(feats + _keep))
        if self.drop:
            return X[cols]
        else:
            return X
