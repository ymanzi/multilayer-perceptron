import sys
import numpy as np
import pandas as pd
from lib.TinyStatistician import *

def get_stat(df):
    list_columns = df.columns
    list_rows = ['Count', 'Mean','Var', 'Std','Min','25%', 'Median' ,'50%','75%','Max']
    list_data = []
    for col in list_columns:
        data = df[col].values
        list_data.append(np.array([df[col].size \
            ,  mean_(data)\
            , var_(data)\
            , std_(data)\
            , min(data.tolist())\
            , quartiles_(data, 25)\
            , median_(data)\
            , quartiles_(data, 50)\
            , quartiles_(data, 75)\
            , max(data)]))
    print(pd.DataFrame(np.array(list_data).transpose(), columns= list_columns, index = list_rows))

def init_data(filename):
    df = pd.read_csv(filename, header=None).drop_duplicates().dropna()
    df[1].replace(['B', 'M'], [0, 1], inplace=True)
    get_stat(df)

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