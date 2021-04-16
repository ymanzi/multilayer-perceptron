import numpy as np
import pandas as pd
import lib.utils


#del 3 21

def split_data(filename):
    df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
    y = df[1].replace(['B', 'M'], [0, 1])
    x = df.drop([1], inplace=True)
    data = data_spliter(x.values, y.values.reshape(-1, 1), 0.6) # , np.array(y_train).reshape(-1, 1), 0.6)
    pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = None)
    pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = None)
    pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = None)
    pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header=None)

df = pd.read_csv("data.csv", header=None ).dropna().drop_duplicates()
print(df.iloc[:,:10].head())