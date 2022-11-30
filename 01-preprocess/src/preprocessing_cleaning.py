## -------------------------------------------------------------------##
## 0. PACKAGES

import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split("Kaggle_taxi_regression",1)[0] + "Kaggle_taxi_regression/")
from utils.utils_functions import *

CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split("Kaggle_taxi_regression",1)[0] + "Kaggle_taxi_regression/01-preprocess/src")
os.chdir(CURRENT_PATH + "/01-preprocess/src")

import pandas as pd
import pandas_profiling as pdp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
plt.style.use("ggplot")
rcParams['figure.figsize'] = (15, 12)

## -------------------------------------------------------------------##
## 1. LOAD RAW DATA

raw_data_train = pd.read_csv("../../exdata/train/train.csv")
raw_data_test = pd.read_csv("../../exdata/test/test.csv")
train = raw_data_train.copy()
test = raw_data_test.copy()

## -------------------------------------------------------------------##
## 2. Save a report profile of the raw data

if not os.path.exists("../results/raw_exploration/report_raw_data.html"):
    folder_create("../results/raw_exploration/")
    report_train = pdp.ProfileReport(raw_data_train, title="Report",minimal=True)
    report_train.to_file("../results/raw_exploration/report_raw_data.html")
    report_test = pdp.ProfileReport(raw_data_test, title="Report",minimal=True)
    report_test.to_file("../results/raw_exploration/report_raw_data_test.html")

## -------------------------------------------------------------------##
## 3. Identify output column
diff_cols = set(train.columns) ^ set(test.columns)
print("The possible output columns are: {}".format(diff_cols))

## -------------------------------------------------------------------##
## 4. Check data quality