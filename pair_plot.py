import sys
import pandas as pd
import numpy as np
from lib.visualization import Komparator as KP
from lib.utils import *

def init_data(filename):
    df = pd.read_csv(filename, header=None).drop_duplicates().dropna().drop([1,2,3,21], axis=1).iloc[:,:]
    df = df.values
    for i in range(df.shape[1]):
        df[:,i] =  minmax_normalization(df[:,i])
    df = pd.DataFrame(df)
    visu = KP(df)
    visu.scatterplot_([1], df.drop(columns = [1], axis=1).columns)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        # try:
        init_data(sys.argv[1])
        # except:
            # print("That filename doesn't exist")