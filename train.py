import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
import pickle
import sys

def save_function(question):
	reply = "lol"
	while reply not in ['y', 'n', 'r']:
		print("------------------------------------------------------")
		reply = str(input(question + "(y/n/r): "))
		if reply not in ['y', 'n', 'r']:
			print("The only accepted replies are 'y','n','r'. ")
	return reply


def lunch_neural(name, layers, train_data, test_data, cost=CrossEntropyCost, hidden_activation=Sigmoid, output_activation=Sigmoid, w_init='std',\
			epochs=1000, batch_size=32, learning_rate = 1.0, lambda_=1.0, n_epoch_early_stop = 0, momentum=0,\
			val_data=None):
	if train_data:
		train_data = list(train_data)
	if test_data:
		test_data = list(test_data)
	if val_data:
		val_data = list(val_data)
	while True:
		NN = Network(name, layers, hidden_activation=hidden_activation, output_activation=output_activation, cost=cost, w_init=w_init,\
		epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, lambda_=lambda_, n_epoch_early_stop=n_epoch_early_stop, momentum=momentum)
		cost_tuple = NN.train_(train_data, test_data, val_data)
		reply = save_function("Do you want to save this Network ?")
		if reply in ['y', 'n']:
			if reply == 'y':
				with open("saved_network.pickle", 'wb') as f:
					pickle.dump(NN, f)
				
			return cost_tuple

# def show_big_plot


def main(data_train, data_test):
	list_cost = []

	cost_tuple = lunch_neural("mini-batch|Tanh|Xavier|CE" ,[29, 4, 3, 2], hidden_activation=Tanh,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=1000, batch_size=32, learning_rate=0.1, lambda_=4.0, n_epoch_early_stop=100, momentum=0.9,\
			train_data=data_train, test_data=data_test, val_data=None)
	list_cost.append(cost_tuple)

	reply = ask_function("Do you want to see the bonuses ?")
	if reply == 'y':
		cost_tuple = lunch_neural("Stochastic|Sigmoid|xavier|CE" ,[29, 4, 3, 2], hidden_activation=Sigmoid,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
			# epochs=1000, batch_size=1, learning_rate=0.01, lambda_=0.4, n_epoch_early_stop=100, momentum=0.9,\
			epochs=1000, batch_size=1, learning_rate=0.01, lambda_=1.5, n_epoch_early_stop=100, momentum=0.9,\
			train_data=data_train, test_data=data_test, val_data=None)
		list_cost.append(cost_tuple)

		cost_tuple = lunch_neural("mini-batch|Sigmoid|std|CE" ,[29, 4, 3, 2], hidden_activation=Sigmoid,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='std',\
			epochs=1000, batch_size=32, learning_rate=0.1, lambda_=1.5, n_epoch_early_stop=100, momentum=0.9,\
			# epochs=1000, batch_size=32, learning_rate=0.1, lambda_=0.5, n_epoch_early_stop=100, momentum=0.9,\
			train_data=data_train, test_data=data_test, val_data=None)
		list_cost.append(cost_tuple)

		cost_tuple = lunch_neural("mini-batch|ReLU|he|CE" ,[29, 4, 3, 2], hidden_activation=ReLu,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='he',\
			epochs=1000, batch_size=32, learning_rate=0.1, lambda_=1.5, n_epoch_early_stop=100, momentum=0.9,\
			# epochs=1000, batch_size=32, learning_rate=0.1, lambda_=0.5, n_epoch_early_stop=100, momentum=0.9,\
			train_data=data_train, test_data=data_test, val_data=None)
		list_cost.append(cost_tuple)
	# show_big_plot(list_cost)


if __name__ == "__main__":
	if len(sys.argv) not in [1, 2]:
		print("incorrect number of arguments")
	else:
		if len(sys.argv) == 1:
			data_train, data_test = init_project_data("srcs/data.csv")
		else:
			data_train, data_test = init_project_data(sys.argv[1])
		main(data_train, data_test)

