import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
import pickle

x_train = init_array("resources/x_train.csv", "x")
y_train = init_array("resources/y_train.csv", "y")

x_test = init_array("resources/x_test.csv", "x")
y_test = init_array("resources/y_test.csv", "y")

x_val = init_array("resources/x_validation.csv", 'x')
y_val = init_array("resources/y_validation.csv", 'y')

def ask_function(question):
	reply = "lol"
	while reply not in ['y', 'n']:
		print("------------------------------------------------------")
		reply = str(input(question + " (y/n): "))
		if reply not in ['y', 'n']:
			print("The only accepted replies are 'y' or 'n'. ")
	return reply



def lunch_neural(name, layers, cost=CrossEntropyCost, hidden_activation=Sigmoid, output_activation=Sigmoid, w_init='std',\
			epochs=1000, batch_size=32, learning_rate = 1.0, lambda_=1.0, n_epoch_early_stop = 0,\
				train_data=zip(x_train, y_train), test_data=zip(x_test, y_test), val_data=None):
	if train_data:
		train_data = list(train_data)
	if test_data:
		test_data = list(test_data)
	if val_data:
		val_data = list(val_data)
	while True:
		NN = Network(name, layers, hidden_activation=hidden_activation, output_activation=output_activation, cost=cost, w_init=w_init,\
		epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, lambda_=lambda_, n_epoch_early_stop=n_epoch_early_stop)
		# cost_tuple = NN.train_(zip(x_train, y_train), test_data=zip(x_test, y_test), validation_data = zip(x_val, y_val))
		cost_tuple = NN.train_(train_data, test_data, val_data)
		reply = ask_function("Do you want to save this Network ?")
		if reply == 'y':
			with open("saved_network.pickle", 'wb') as f:
				pickle.dump(NN, f)
			return cost_tuple


# cost_tuple = lunch_neural("mini-batch | Tanh | Xavier | CE" ,[30, 5, 3, 2], hidden_activation=Tanh,\
# 	output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
# 	epochs=1000, batch_size=32, learning_rate=0.1, lambda_=0.0, n_epoch_early_stop=100,\
# 		train_data=zip(x_train, y_train), test_data=zip(x_test, y_test), val_data=zip(x_val, y_val))

cost_tuple = lunch_neural("Stochastic | Sigmoid | xavier | CE" ,[30, 4, 3, 2], hidden_activation=Sigmoid,\
	output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
	epochs=1000, batch_size=1, learning_rate=0.005, lambda_=0, n_epoch_early_stop=100,\
	train_data=zip(x_train, y_train), test_data=zip(x_test, y_test), val_data=None)

# cost_tuple = lunch_neural("mini-batch | Sigmoid | std | CE" ,[31, 4, 3, 2], hidden_activation=Sigmoid,\
# 	output_activation=Softmax, cost=CrossEntropyCost, w_init='std',\
# 	epochs=1000, batch_size=32, learning_rate=0.1, lambda_=1.0, n_epoch_early_stop=100,\
# 	train_data=zip(x_train, y_train), test_data=zip(x_test, y_test), val_data=zip(x_val, y_val))

# cost_tuple = lunch_neural("mini-batch | ReLU | he | CE" ,[31, 4, 3, 2], hidden_activation=ReLu,\
# 	output_activation=Softmax, cost=CrossEntropyCost, w_init='he',\
# 	epochs=1000, batch_size=32, learning_rate=0.5, lambda_=2.0, n_epoch_early_stop=100,\
# 	train_data=zip(x_train, y_train), test_data=zip(x_test, y_test), val_data=zip(x_val, y_val))

# while True:
# 	NN = Network("mini-batch | Tanh | Xavier | CE" ,[31, 4, 3, 2], hidden_activation=Tanh, output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
# 	epochs=1000, batch_size=32, learning_rate=0.1, lambda_=2, n_epoch_early_stop=100)
# 	# cost_tuple = NN.train_(zip(x_train, y_train), test_data=zip(x_test, y_test), validation_data = zip(x_val, y_val))
# 	cost_tuple = NN.train_(zip(x_train, y_train), test_data=zip(x_test, y_test))
# 	reply = ask_function("Do you want to save this Network ?")
# 	if reply == 'y':
# 		with open("saved_network.pickle", 'wb') as f:
# 			pickle.dump(NN, f)
# 		break
# 		# return cost_tuple

# NN2 = Network("Stochastic | Sigmoid | xavier | CE" ,[31, 4, 3, 2], hidden_activation=Sigmoid, output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier')
# cost_tuple = NN2.train_(zip(x_train, y_train), 1000, 1, learning_rate=0.01, lambda_=0.1, test_data=zip(x_test, y_test), n_epoch_early_stop=100)

# NN3 = Network("mini-batch | Sigmoid | std | CE" ,[31, 4, 3, 2], hidden_activation=Sigmoid, output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier')
# cost_tuple = NN3.train_(zip(x_train, y_train), 1000, 32, learning_rate=1, lambda_=0.5, test_data=zip(x_test, y_test), n_epoch_early_stop=100)

# NN4 = Network("mini-batch | ReLU | he | CE" ,[31, 4, 3, 2], hidden_activation=ReLu, output_activation=Softmax, cost=CrossEntropyCost, w_init='he')
# cost_tuple = NN4.train_(zip(x_train, y_train), 1000, 30, learning_rate=0.05, lambda_=0.00, test_data=zip(x_test, y_test), n_epoch_early_stop=100)

# print(x_train.shape)
# print(NN.feedforward(x_train[0] ) )


# print(pd.DataFrame(training_data))
# print(y_train.shape)