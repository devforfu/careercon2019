import feather
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from basedir import DATA


TMP = DATA/'tmp'


def to_feather(df, name):
    name = TMP/f'{name}.feather'
    if not name.parent.exists():
        name.parent.mkdir(parents=True)
    df.to_feather(name)
    return name


def from_feather(name, *names):
    names = [name] + list(names)
    data = [feather.read_dataframe(TMP/f'{name}.feather') for name in names]
    return data


def columns(df, pattern):
    index = df.columns.str.match(pattern)
    return df.columns[index]


def starts(df, string):
    return columns(df, f'^{string}')


def dropcols(df, cols, missing_ok=True):
    if missing_ok:
        cols = df.columns[df.columns.isin(cols)]
    return df.drop(columns=cols)


def float64(data): return data.astype(np.float64)


def split(data, target, n_splits=5, seed=None, verbose=True):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(data))
    for i, (trn_idx, val_idx) in enumerate(kfold.split(idx, target), 1):
        if verbose:
            print(f'Running {i:d} of {kfold.get_n_splits():d} folds')
        yield trn_idx, val_idx
