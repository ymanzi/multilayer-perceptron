import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
import pickle


x_val = init_array("resources/x_validation.csv", 'x')
y_val = init_array("resources/y_validation.csv", 'y')


def main
with open("saved_network.pickle", 'rb') as f:
	NN = pickle.load(f)
NN.lunch_test(None, None, list(zip(x_val, y_val)))


if __name__ == "__main__":
	if len(sys.argv) not in [1, 2]:
		print("incorrect number of arguments")
	else:
		if len(sys.argv) == 1:
			data_train, data_test = init_project_data("srcs/data.csv")
		else:
			data_train, data_test = init_project_data(sys.argv[1])
		main(data_predict, filename)
