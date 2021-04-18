import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *

def array_normalization(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = minmax_normalization(arr[:,i])
	return arr

def x_array_reshape(arr):
	return np.array([elem.reshape(-1, 1) for elem in arr])

def y_array_reshape(arr):
	tmp_arr = [[0, 1] if elem == 1 else [1, 0] for elem in arr]
	return np.array([np.array(elem).reshape(-1, 1) for elem in tmp_arr])

x_train = pd.read_csv("resources/x_train.csv", header=None ).dropna().values


x_test = pd.read_csv("resources/x_test.csv", header=None ).dropna().values


x_train = array_normalization(x_train)
x_train = x_array_reshape(x_train)

x_test = array_normalization(x_test)
x_test = x_array_reshape(x_test)

y_train = pd.read_csv("resources/y_train.csv", header=None).dropna().values.reshape(-1, 1)
y_train = y_array_reshape(y_train)

y_test = pd.read_csv("resources/y_test.csv", header=None).dropna().values.reshape(-1, 1)
y_test = y_array_reshape(y_test)

training_data = zip(x_train, y_train)
test_data = zip(x_test, y_test)


NN = Network([31, 10, 2], output_activation=Softmax, cost=CrossEntropyCost)

# print(NN.biases[0].shape)
NN.mini_batch_gradient(training_data, 1000, 1, learning_rate= 0.05, lambda_=0.005, test_data=test_data)

# print(x_train.shape)
# print(NN.feedforward(x_train[0] ) )


# print(pd.DataFrame(training_data))
# print(y_train.shape)