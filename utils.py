import feather
import numpy as np
import pandas as pd

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