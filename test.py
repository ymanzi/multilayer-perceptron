import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
from random import *

#del 3 21



# df = pd.read_csv("data.csv", header=None ).dropna().drop_duplicates()
# print(df.iloc[:,:10].head())



p = np.array([[[1,-1, -1], [3,-3, 3], [2,-2, 2]], [[1,-1, -1], [3,-3, 3], [2,-2, 2]]])
m = np.random.binomial(1, 0.5, size = p.shape) 
print(m)
print(p)
print(p*m)

# print(np.sum(p))
# e = np.array([[-5,-5], [3,0], [2,0]])


# print(ReLu.derivative(p))

# print(Softmax.fct(arr))

# split_data("resources/data.csv")
# split_validation("resources/x_test.csv", "resources/y_test.csv")


# data_train, data_test = init_project_data("resources/data.csv")
# print(data_test)
