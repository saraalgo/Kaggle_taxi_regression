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
rcParams["figure.figsize"] = (15, 12)

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
## 3. Exploration of the data

print("The training data has a total of {} observations and {} variables".format(train.shape[0], train.shape[1]))
print("The testing data has a total of {} observations and {} variables".format(test.shape[0], test.shape[1]))

diff_cols = set(train.columns) ^ set(test.columns)
print("The missing columns in the testing data are: {}".format(diff_cols))

train.info()


## -------------------------------------------------------------------##
## 4. Feature engineering

### id

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

### vendor_id and passanger_count

train.vendor_id.value_counts()
train.passenger_count.value_counts()

### pickup_datetime and dropoff_datetime

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["minute"] = train["pickup_datetime"].dt.minute
train["minute_oftheday"] = train["hour"] * 60 + train["minute"]
train["day_week"] =train["pickup_datetime"].dt.dayofweek
train["month"] = train["pickup_datetime"].dt.month

test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["minute"] = test["pickup_datetime"].dt.minute
test["minute_oftheday"] = test["hour"] * 60 + test["minute"]
test["day_week"] =test["pickup_datetime"].dt.dayofweek
test["month"] = test["pickup_datetime"].dt.month

train.drop(["pickup_datetime", "dropoff_datetime"], axis=1, inplace=True)
test.drop(["pickup_datetime"], axis=1, inplace=True)

### pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude


