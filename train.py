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



while True:
	NN = Network("mini-batch | Tanh | Xavier | CE" ,[31, 4, 3, 2], hidden_activation=Tanh, output_activation=Softmax, cost=CrossEntropyCost, w_init='xavier',\
	epochs=1000, batch_size=32, learning_rate=0.1, lambda_=2, n_epoch_early_stop=100)
	cost_tuple = NN.train_(zip(x_train, y_train), test_data=zip(x_test, y_test), validation_data = zip(x_val, y_val))
	reply = ask_function("Do you want to save this Network ?")
	if reply == 'y':
		# pd.DataFrame(np.array(NN.weights, dtype=list)).to_csv("weights.csv", header=None, index=False)
		# pd.DataFrame( np.array(NN.weights, dtype=list)).to_csv("biases.csv", header=None, index=False)
		with open("saved_network.pickle", 'wb') as f:
			pickle.dump(NN, f)
		break

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