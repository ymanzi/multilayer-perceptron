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


def lunch_neural(name, layers, datafile, cost=CrossEntropyCost, hidden_activation=Sigmoid, \
	output_activation=Sigmoid, w_init='std', epochs=1000, batch_size=32, \
	learning_rate = 1.0, lambda_=1.0, n_epoch_early_stop=0, momentum=0, dropout=0.0):
	# if train_data:
	# 	train_data = list(train_data)
	# if test_data:
	# 	test_data = list(test_data)
	# if val_data:
	# 	val_data = list(val_data)
	while True:
		train_data, test_data = init_project_train(datafile)
		NN = Network(name, layers, hidden_activation=hidden_activation, output_activation=output_activation, cost=cost, w_init=w_init,\
		epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, lambda_=lambda_, n_epoch_early_stop=n_epoch_early_stop,\
			momentum=momentum, dropout=dropout)
		
		# with open("xavier_drop" + ".pickle", 'wb') as f:
		# 	pickle.dump(NN.weights, f)
		# exit()
		# with open("train_data" + ".pickle", 'rb') as f:
		# 	train_data = pickle.load(f)
		# with open("test_data" + ".pickle", 'rb') as f:
		# 	test_data = pickle.load(f)
		# with open("xavier_drop" + ".pickle", 'rb') as f:
		# 	NN.weights = pickle.load(f)
		NN.train_(train_data, test_data)
		# break
		reply = save_function("Do you want to save this Network ?")
		if reply in ['y', 'n']:
			if reply == 'y':
				with open(NN.name + ".pickle", 'wb') as f:
					pickle.dump(NN, f)
			return NN

# def show_big_plot


def main(filename):
	list_nn = []
	# cost_tuple = lunch_neural("1-mini-batch|Tanh|Xavier|CE" ,[29, 7, 6, 4, 2], hidden_activation=Tanh,\
	cost_tuple = lunch_neural("1" ,[29, 5, 4, 3, 2], hidden_activation=Tanh,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=10000, batch_size=30, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile=filename, dropout=0.5)
	list_nn.append(cost_tuple)

	reply = ask_function("Do you want to see the bonuses ?")
	if reply == 'y':
		# cost_tuple = lunch_neural("2-Stochastic|Sigmoid|xavier|CE" ,[29, 7, 6, 4, 2], hidden_activation=Sigmoid,\
		cost_tuple = lunch_neural("2" ,[29, 5, 4, 3, 2], hidden_activation=Sigmoid,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
			epochs=10000, batch_size=1, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile= filename, dropout=0.5)
		list_nn.append(cost_tuple)

		# cost_tuple = lunch_neural("3-mini-batch|Sigmoid|xavier|CE" ,[29, 7, 6, 4, 2], hidden_activation=Sigmoid,\
		cost_tuple = lunch_neural("3" ,[29, 5, 4, 3, 2], hidden_activation=Sigmoid,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
			epochs=10000, batch_size=32, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile=filename, dropout=0.5)
		list_nn.append(cost_tuple)

		# cost_tuple = lunch_neural("4-mini-batch|ReLU|he|CE" ,[29, 7, 6, 4, 2], hidden_activation=ReLu,\
		cost_tuple = lunch_neural("4" ,[29, 5, 4, 3, 2], hidden_activation=ReLu,\
			output_activation=Softmax, cost=CrossEntropyCost, w_init='he',\
			epochs=10000, batch_size=32, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile=filename, dropout=0.5)
		list_nn.append(cost_tuple)

	# show_big_plot(list_nn)


if __name__ == "__main__":
	if len(sys.argv) not in [1, 2]:
		print("incorrect number of arguments")
	else:
		if len(sys.argv) == 1:
			filename = "data_training.csv"
		else:
			filename = sys.argv[1]
		main(filename)

