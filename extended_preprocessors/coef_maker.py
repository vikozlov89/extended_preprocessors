from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class CoefficientMaker(TransformerMixin):
    """
        Class creating coefficients from a given dataset.
        It takes all columns in columns_to_use list and divides them. For example, provided a list of
        ['col1', 'col2', 'col3'], you will receive a dataframe with 'col1_div_col2', 'col1_div_col3'
        and 'col2_div_col3' columns in addition to existing columns.

        Example:
            >>> ds = pd.DataFrame({'a':[1, 2, 3], 'b':[2, 3, 4]})
            >>> coef_maker = CoefficientMaker(['a', 'b'])
            >>> ds_transformed = coef_maker.transform(ds)
            >>> print(ds_transformed.columns.tolist())
            ['a', 'b', 'a_div_b']
            >>> print(ds_transformed['a_div_b'].round(2).tolist())
            [0.5, 0.67, 0.75]
    """

    def __init__(
        self, columns_to_use: List,
    ):

        self.columns_to_use = columns_to_use
        self.n_columns = len(columns_to_use)

    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None, *args, **kwargs
    ):
        return self

    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        It takes all columns in columns_to_use list and divides them. For example, provided a list of
        ['col1', 'col2', 'col3'], you will receive a dataframe with 'col1_div_col2', 'col1_div_col3'
        and 'col2_div_col3' columns in addition to existing columns.

        :param X: dataframe to preprocess
        :return: pd.DataFrame, original dataframe with additional columns.
        """
        tmp = X.copy()
        cnames_generated = []
        for i in range(self.n_columns - 1):
            for j in range(i + 1, self.n_columns):
                c1 = self.columns_to_use[i]
                c2 = self.columns_to_use[j]
                cname = f"{c1}_div_{c2}"
                tmp[cname] = X[c1].fillna(1) / X[c2].fillna(1).map(
                    lambda x: 1 if x == 0 else x
                )
                cnames_generated.append(cname)
        return tmp
