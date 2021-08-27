import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from extended_preprocessors.feature_discretizer import FeatureDiscretizer


class NumericalFeatureDiscretizer(FeatureDiscretizer):
    def __init__(
        self, max_bins: int, use_median: bool = False, encode_with_expected=False
    ):
        super().__init__(max_bins, use_median)
        self.discretizer = KBinsDiscretizer(strategy="uniform")
        self.max_bins = max_bins
        self.bins_map = {}
        self.use_median = use_median
        self.encode_with_expected = encode_with_expected

    def fit(self, X, y, **fit_params) -> FeatureDiscretizer:
        assert len(X.shape) == 2
        assert X.shape[1] == 1
        self.discretizer = KBinsDiscretizer(
            n_bins=self._define_n_bins(X), strategy="uniform", encode="ordinal"
        )
        self.discretizer.fit(X)
        bins = self.discretizer.transform(X).flatten()
        self.bins_map = self._make_bins_map(
            bins, y, encode_with_expected=self.encode_with_expected
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert len(X.shape) == 2
        assert X.shape[1] == 1
        bins = self.discretizer.transform(X)
        try:
            mapped_bins = np.array([self.bins_map.get(v) for v in bins.flatten()])
        except:
            raise ValueError()
        return mapped_bins
