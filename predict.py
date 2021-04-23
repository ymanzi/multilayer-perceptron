import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *
from lib.data_init import *
import pickle


x_val = init_array("resources/x_validation.csv", 'x')
y_val = init_array("resources/y_validation.csv", 'y')

with open("saved_network.pickle", 'rb') as f:
	NN = pickle.load(f)
NN.lunch_test(None, None, list(zip(x_val, y_val)))
