import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def mean_(x):
	length = x.shape[0]
	if length == 0:
		return None
	return float(np.sum(x) / length)

def median_(x):
	length = x.size
	if length == 0:
		return None
	y = np.sort(x)
	if (length % 2):
		ret = y[int(length / 2)]
	else:
		ret = (y[int(length / 2) - 1] + y[int(length / 2)]) / 2
	return float(ret)

def quartiles_(x, percentile):
	length = x.size
	if length == 0:
		return None
	y = np.sort(x)
	per = (percentile / 100) * length
	if (per % 1.0):
		return float(y[int(per)])
	else:
		return float(y[int(per) - 1])

def var_(x):
	length = x.size
	if length == 0:
		return None
	mean = float(np.sum(x) / length)
	return np.sum(np.power(np.subtract(x, mean), 2)) / length

def std_(x):
	length = x.size
	if length == 0:
		return None
	mean = float(np.sum(x) / length)
	return np.sqrt(np.sum(np.power(np.subtract(x, mean), 2)) / length)

def skew_(x):
	return 3 * (mean_(x) - median_(x)) / std_(x)


def pearson_corr(data, feature1, feature2):
       """
       Pearson correlation works well if the relationship between variables are linear and variables are roughly normal. 
       But it is not robust, if there are outliers
       """
       print('Pearson correlation: \n', data[[feature1, feature2]].corr(method= "pearson"))

def spearman_corr(data, feature1, feature2):
       """
       Pearson correlation works well if the relationship between variables are linear and variables are roughly normal. 
       But it is not robust, if there are outliers
       To compute spearman's correlation we need to compute rank of each value
       """
       ranked_data = data.rank()
       print('Spearman correlation: \n', ranked_data[[feature1, feature2]].corr(method= "spearman"))


def delete_outliers(data, list_feature):
	for feature in list_feature:
		data_benign = data[data["diagnosis"] == "B"]
		data_malignant = data[data["diagnosis"] == "M"]

		desc = data_benign[feature].describe()
		Q1 = desc[4]
		Q3 = desc[6]

		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5*IQR
		upper_bound = Q3 + 1.5*IQR

		# box_plot_(data, 'diagnosis', [feature])
		data[data["diagnosis"] == "B"] = data_benign[(data_benign[feature] > lower_bound) & (data_benign[feature] < upper_bound)]
		
		desc = data_malignant[feature].describe()
		Q1 = desc[4]
		Q3 = desc[6]

		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5*IQR
		upper_bound = Q3 + 1.5*IQR
		
		data[data["diagnosis"] == "M"] = data_malignant[(data_malignant[feature] > lower_bound) & (data_malignant[feature] < upper_bound)]
		# box_plot_(data, 'diagnosis', [feature])
	return data

def effect_size_(data, cat_var, features):
       ''' 
       Cohen Effect Size 
       Cohen suggest that if d(effect size)= 0.2, it is small effect size, d = 0.5 medium effect size, d = 0.8 large effect size.]
       This summary statistics offers a simple way of quantifying the difference between two groups
              In an other saying, effect size emphasises the size of the difference
       '''
       print("[ EFFECT SIZE ]")
       data_malignant = data[data[cat_var] == 'M']
       data_benign = data[data[cat_var] == 'B']

       dic_ = {}
       for i, f1 in enumerate(features):
              mean_diff = data_malignant[f1].mean() - data_benign[f1].mean()
              var_benign = data_benign[f1].var()
              var_malignant = data_malignant[f1].var()
              var_pooled = (len(data_benign)*var_benign +len(data_malignant)*var_malignant)\
                     / float(len(data_benign)+ len(data_malignant))
              effect_size = mean_diff/np.sqrt(var_pooled)
              dic_[f1] = effect_size
       dic_ = dict(sorted(dic_.items(), key=lambda item: item[1], reverse=True))
       for key, value in dic_.items():
              print("Effect size of {:23}: {}".format(key, value))
