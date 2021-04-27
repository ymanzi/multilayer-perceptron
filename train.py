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
	learning_rate = 1.0, lambda_=0.0, n_epoch_early_stop=0, momentum=0, dropout=0.0):

	while True:
		train_data, test_data = init_project_train(datafile)
		NN = Network(name, layers, hidden_activation=hidden_activation, output_activation=output_activation, cost=cost, w_init=w_init,\
		epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, lambda_=lambda_, n_epoch_early_stop=n_epoch_early_stop,\
			momentum=momentum, dropout=dropout)
		input('''========================\n= PRESS ENTER TO TRAIN =\n========================''')
		NN.train_(train_data, test_data)
		reply = save_function("Do you want to save this Network ?")
		if reply in ['y', 'n']:
			if reply == 'y':
				with open("saved_NN/" + NN.name + ".pickle", 'wb') as f:
					pickle.dump(NN, f)
			return NN

# def show_big_plot


def main(filename):
	list_nn = []
	cost_tuple = lunch_neural("1-mini-batch|Tanh" ,[29, 30, 20, 2], hidden_activation=Tanh,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=10000, batch_size=30, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile=filename, dropout=0.4)
	list_nn.append(cost_tuple)

	cost_tuple = lunch_neural("2-5Layers|Tanh" ,[29, 40, 30, 20, 2], hidden_activation=Tanh,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=10000, batch_size=30, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
			datafile=filename, dropout=0.4)
	list_nn.append(cost_tuple)

	reply = ask_function("Do you want to see the bonuses ?")
	if reply == 'y':
		cost_tuple = lunch_neural("3-Stochastic|Sigmoid" ,[29, 40, 30, 20, 2], hidden_activation=Sigmoid,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=10000, batch_size=1, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
		datafile= filename, dropout=0.4)
		list_nn.append(cost_tuple)

		cost_tuple = lunch_neural("4-mini-batch|Sigmoid" ,[29, 30, 20, 2], hidden_activation=Sigmoid,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
		epochs=10000, batch_size=32, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
		datafile= filename, dropout=0.4)
		list_nn.append(cost_tuple)
	
		cost_tuple = lunch_neural("5-mini-batch|ReLU|he" ,[29, 40, 30, 20, 2], hidden_activation=ReLu,\
		output_activation=Softmax, cost=CrossEntropyCost, w_init='he',\
		epochs=10000, batch_size=32, learning_rate=0.01, lambda_=0.0, n_epoch_early_stop=50, momentum=0.9,\
		datafile=filename, dropout=0.4)
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

