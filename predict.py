import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
import pickle
import sys

def main(data_predict, network):
	with open(network, 'rb') as f:
		NN = pickle.load(f)
	NN.predict_test(data_predict)


if __name__ == "__main__":
	if len(sys.argv) not in [2, 3]:
		print("incorrect number of arguments")
	else:
		if len(sys.argv) == 2:
			data_predict = init_project_predict("srcs/data.csv")
		else:
			data_predict = init_project_predict(sys.argv[2])
		main(data_predict, sys.argv[1])
