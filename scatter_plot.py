import sys
import pandas as pd
import numpy as np
from lib.visualization import Komparator as KP

def init_data(filename):
    df = pd.read_csv(filename, header=None).drop_duplicates().dropna()
    visu = KP(df)
    visu.pairplot_([1], df.drop(columns = [1], axis=1).columns)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            init_data(sys.argv[1])
        except:
            print("That filename doesn't exist")