import feather
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

