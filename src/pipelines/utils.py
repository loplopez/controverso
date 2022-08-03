import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_test_split(X: pd.DataFrame, y: pd.Series, group_by: str, test_size: float, random_state: int):
    """
    Provides a train-test split using a column to group by.
    :param df:
    :param group_by:
    :return:
    """
    train_inds, test_inds = next(
        GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=random_state).split(X, groups=X[group_by]))
    return X.iloc[train_inds], X.iloc[test_inds], y.iloc[train_inds], y.iloc[test_inds]
