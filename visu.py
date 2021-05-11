import sys
import pandas as pd
import numpy as np
from lib.visualization import * #Komparator as KP
from lib.visualization import Komparator as KP

from lib.utils import *

def init_data(filename):
    columns = ['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

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

    data = pd.read_csv(filename, header=None)
    data[1].replace(['B', 'M'], [0, 1], inplace=True)
    data = pd.DataFrame(data.values, columns=columns, dtype=float)
    data['diagnosis'].replace([0, 1], ['B', 'M'], inplace=True)
    data = data.drop(['id'], axis=1)

    cp_col = columns
    cp_col.remove('diagnosis')
    cp_col.remove('id')

    heatmap_(data.replace(['M', 'B'],[1, 0]), 'diagnosis', f_list)
    # heatmap_(data.replace(['M', 'B'],[1, 0]), 'diagnosis', cp_col)
    # histogram_(data, 'diagnosis',cp_col)
    # box_plot_(data, 'diagnosis',cp_col)
    visu = KP(data)
    visu.scatterplot_('diagnosis', f_list)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("incorrect number of arguments")
    elif sys.argv[1][-3:] != "csv":
        print("wrong data file extension")
    else:
        try:
            init_data(sys.argv[1])
        except:
            print("That filename doesn't exist")