import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.visualization import draw_plot

def array_normalization(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = minmax_normalization(arr[:,i])
	return arr

def x_array_reshape(arr):
	return np.array([elem.reshape(-1, 1) for elem in arr])

def y_array_reshape(arr):
	tmp_arr = [[0, 1] if elem == 1 else [1, 0] for elem in arr]
	return np.array([np.array(elem).reshape(-1, 1) for elem in tmp_arr])

x_train = pd.read_csv("resources/x_train.csv", header=None).dropna().values


x_test = pd.read_csv("resources/x_test.csv", header=None ).dropna().values


x_train = array_normalization(x_train)
x_train = x_array_reshape(x_train)

x_test = array_normalization(x_test)
x_test = x_array_reshape(x_test)

y_train = pd.read_csv("resources/y_train.csv", header=None).dropna().values.reshape(-1, 1)
y_train = y_array_reshape(y_train)

y_test = pd.read_csv("resources/y_test.csv", header=None).dropna().values.reshape(-1, 1)
y_test = y_array_reshape(y_test)

# print(x_test.shape)
# print(x_train.shape)
# exit()


training_data = zip(x_train, y_train)
test_data = zip(x_test, y_test)


NN = Network([31, 31, 2], hidden_activation=Sigmoid, output_activation=Softmax, cost=CrossEntropyCost)

# print(NN.biases[0].shape)
list_cost = []

# cost_tuple = NN.train_(training_data, 100, 10, lambda_=0.01, test_data=test_data, n_epoch_early_stop=50)
# list_cost.append(cost_tuple)

cost_tuple = NN.train_(training_data, 1000, 7, learning_rate=5, lambda_=0.5, test_data=test_data, n_epoch_early_stop=100)
list_cost.append(cost_tuple)

# cost_tuple = NN.train_(training_data, 100, 30, lambda_=0.01, test_data=test_data, n_epoch_early_stop=50)
# list_cost.append(cost_tuple)

draw_plot(cost_tuple)
# print(x_train.shape)
# print(NN.feedforward(x_train[0] ) )


# print(pd.DataFrame(training_data))
# print(y_train.shape)