# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from lib.TinyStatistician import *

def box_plot_(data, cat_var, features):
       for col in features:
              melted_data = pd.melt(data,id_vars =cat_var ,value_vars =col)#['radius_mean', 'texture_mean'])
              plt.figure(figsize = (15,10))
              sns.boxplot(x = "variable", y = "value", hue=cat_var ,data= melted_data)
              plt.title(col)
              plt.show()

def histogram_(data, cat_var, features):
       for col in features:
              m = plt.hist(data[data[cat_var] == "M"][col].values,bins=30,fc = (1,0,0,0.5),label = "Malignant")
              b = plt.hist(data[data[cat_var] == "B"][col].values,bins=30,fc = (0,1,0,0.5),label = "Benign")
              plt.legend()
              plt.xlabel(col)
              plt.ylabel("Frequency")
              plt.title("Histogram of " + col + " for Benign and Malignant Tumors")
              plt.show()


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


def heatmap_(data, cat_var, features):
       """ 
       Pearson Correlation 
       
       Strength of the relationship between two variables
       * Meaning of 1 is two variable are positively correlated with each other
       * Meaning of zero is there is no correlation between variables 
       * Meaning of -1 is two variables are negatively correlated with each other
       """

       if cat_var not in features:
              features.insert(0, cat_var)

       f,ax=plt.subplots(figsize = (15,10))
       sns.heatmap(data[features].corr(),annot=True,linewidths=0.5,fmt = ".2f", ax=ax, cmap="YlGnBu")
       plt.xticks(rotation=90)
       plt.yticks(rotation=0)
       plt.title('Correlation Map')
       plt.savefig('graph.png')
       plt.show()




def covariance_list(data, features_list):
       print("[ COVARIANCE ]")
       dic_ = {}
       for i, f1 in enumerate(features_list):
              for f2 in features_list[i:]:
                     if f1 != f2:
                            ret = covariance_(data, f1, f2)
                            dic_[ret[0]] = ret[1]
       dic_ = dict(sorted(dic_.items(), key=lambda item: item[1], reverse=True))
       for key, value in dic_.items():
              print("{:50}: {}".format(key, value))

def covariance_(data, feature1, feature2):
       ''' 
       Covariance is measure of the tendency of two variables to vary together
       * So covariance is maximized if two vectors are identical
       * Covariance is zero if they are orthogonal.
       * Covariance is negative if they point in opposite direction
       '''
       return ("{} and {}".format(feature1, feature2) , data[feature1].cov(data[feature2]))


columns = ['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

data = pd.read_csv("data.csv", header=None)
data[1].replace(['B', 'M'], [0, 1], inplace=True)
data = pd.DataFrame(data.values, columns=columns, dtype=float)
data['diagnosis'].replace([0, 1], ['B', 'M'], inplace=True)
data = data.drop(['id'], axis=1)
# print(data.head())

# m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
# b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Benign")
# plt.legend()
# plt.xlabel("Radius Mean Values")
# plt.ylabel("Frequency")
# plt.title("Histogram of Radius Mean for Benign and Malignant Tumors")
# plt.show()








f_list = ["radius_mean",
"radius_worst",
"radius_se",
"perimeter_mean",
"perimeter_worst",
"area_mean",
"area_worst",
"concavity_mean",
"concavity_worst",
"concave points_mean",
"concave points_worst"]

print(data.values.shape())
data = delete_outliers(data, f_list )
print(data.count())


# cp_col = columns

# cp_col.remove('diagnosis')
# cp_col.remove('id')

# histogram_(data, 'diagnosis',cp_col)
# box_plot_(data, 'diagnosis',cp_col)
# effect_size_(data, 'diagnosis',cp_col)


# f_list = ["radius_mean",
# "radius_worst",
# "radius_se",
# "perimeter_mean",
# "perimeter_worst",
# "area_mean",
# "area_worst",
# "concavity_mean",
# "concavity_worst",
# "concave points_mean",
# "concave points_worst"]

# covariance_list(data, f_list)
# print(covariance_(data, "area_worst", "area_worst"))

# print(data.head())
# heatmap_(data.replace(['M', 'B'],[1, 0]), 'diagnosis', f_list)
# covariance_(data, 'radius_mean', 'area_mean')
# pearson_corr(data, 'radius_mean', 'area_mean')
# spearman_corr(data, 'radius_mean', 'area_mean')
