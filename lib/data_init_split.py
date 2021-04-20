import numpy as np
import pandas as pd
from lib.utils import *

def split_data(filename):
    df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
    y = df[1].replace(['B', 'M'], [0, 1])
    x = df.drop([1], axis=1)
    data = data_spliter(x.values, y.values.reshape(-1, 1), 0.6) # , np.array(y_train).reshape(-1, 1), 0.6)
    pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header=None, index=None)
    pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header=None, index=None)
    pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header=None, index=None)
    pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header=None, index=None)



split_data("data.csv")