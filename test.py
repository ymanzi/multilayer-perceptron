import numpy as np
import pandas as pd
from lib.utils import *
from lib.my_NN import *

#del 3 21

def split_data(filename):
	df = pd.read_csv(filename, header=None).dropna().drop_duplicates()
	y = df[1].replace(['B', 'M'], [0, 1])
	x = df.drop(columns = [1])
	# print(x)
	data = data_spliter(x.values, y.values.reshape(-1, 1), 0.5) # , np.array(y_train).reshape(-1, 1), 0.6)
	pd.DataFrame(data[0]).to_csv("resources/x_train.csv", header = None, index=False)
	pd.DataFrame(data[1]).to_csv("resources/y_train.csv", header = None, index=False)
	pd.DataFrame(data[2]).to_csv("resources/x_test.csv", header = None, index=False)
	pd.DataFrame(data[3]).to_csv("resources/y_test.csv", header=None, index=False)

def split_validation(file_x, file_y):
	x = pd.read_csv(file_x, header=None)
	y = pd.read_csv(file_y, header=None)
	# print(x)
	# exit()

	data = data_spliter(x.values, y.values.reshape(-1, 1), 0.5) # , np.array(y_train).reshape(-1, 1), 0.6)
	pd.DataFrame(data[0]).to_csv("resources/x_test.csv", header = None, index=False)
	pd.DataFrame(data[1]).to_csv("resources/y_test.csv", header = None, index=False)
	pd.DataFrame(data[2]).to_csv("resources/x_validation.csv", header = None, index=False)
	pd.DataFrame(data[3]).to_csv("resources/y_validation.csv", header=None, index=False)


# df = pd.read_csv("data.csv", header=None ).dropna().drop_duplicates()
# print(df.iloc[:,:10].head())

# arr = np.array([1, 3, 2])
# print(Softmax.fct(arr))

split_data("resources/data.csv")
split_validation("resources/x_test.csv", "resources/y_test.csv")
