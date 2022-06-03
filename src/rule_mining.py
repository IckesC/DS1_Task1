import pandas as pd
from pyECLAT import ECLAT
from src import DATA
import os

# for debugging

path_norm = os.path.join(DATA, "norm_data.csv")
path_schiz = os.path.join(DATA, "schiz_data.csv")


def eclat(df, minsupport=0.2):
    eclat_instance = ECLAT(df)
    indexes, support = eclat_instance.fit(min_support=minsupport, min_combination=2, max_combination=2)

    return indexes, support


if __name__ == '__main__':
    df = pd.read_csv(path_norm, header=None)
    indexes, support = eclat(df.T)
    print(indexes)