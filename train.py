import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *

def array_normalization(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = minmax_normalization(arr[:,i])
	return arr


x_train = pd.read_csv("resources/x_train.csv", header=None ).dropna().values
x_test = pd.read_csv("resources/x_test.csv", header=None ).dropna().values

x_train = array_normalization(x_train)
x_test = array_normalization(x_test)


y_train = pd.read_csv("resources/y_train.csv", header=None).dropna().values.reshape(-1, 1)
y_test = pd.read_csv("resources/y_test.csv", header=None).dropna().values.reshape(-1, 1)

training_data = zip(x_train, y_train)
test_data = zip(x_test, y_test)


NN = Network([31, 100, 2], cost=CrossEntropyCost)

# print(NN.biases[0].shape)
NN.mini_batch_gradient(training_data, 1000, 22, learning_rate= 0.02, test_data=test_data)

# print(x_train.shape)
# print(NN.feedforward(x_train[0] ) )


# print(pd.DataFrame(training_data))
# print(y_train.shape)