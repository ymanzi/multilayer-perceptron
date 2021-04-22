import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *

#del 3 21



# df = pd.read_csv("data.csv", header=None ).dropna().drop_duplicates()
# print(df.iloc[:,:10].head())



p = np.array([[1,-1], [3,-3], [2,-2]])
e = np.array([[-5,-5], [3,0], [2,0]])


print(ReLu.derivative(p))

# print(Softmax.fct(arr))

# split_data("resources/data.csv")
# split_validation("resources/x_test.csv", "resources/y_test.csv")


