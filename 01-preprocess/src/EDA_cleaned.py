## -------------------------------------------------------------------##
## 0. PACKAGES

import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split("Kaggle_taxi_regression",1)[0] + "Kaggle_taxi_regression/")
from utils.utils_functions import *

CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split("Kaggle_taxi_regression",1)[0] + "Kaggle_taxi_regression/01-preprocess/src")
import preprocessing_cleaning as pc

import pandas_profiling as pdp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
plt.style.use("ggplot")
rcParams['figure.figsize'] = (15, 12)

## -------------------------------------------------------------------##
## 1. LOAD DATA CLEANED

train = pc.train.copy()
test = pc.test.copy()

folder_create("../data/EDA")

## -------------------------------------------------------------------##
## 2. Save a report profile of the cleaned data

report_train = pdp.ProfileReport(train, title="Report",minimal=True)
report_train.to_file("../data/EDA/report_train.html")

report_test = pdp.ProfileReport(test, title="Report",minimal=True)
report_test.to_file("../data/EDA/report_test.html")

## -------------------------------------------------------------------##
## 3. Exploratory Data Analysis (EDA) of the cleaned and preprocessed data


