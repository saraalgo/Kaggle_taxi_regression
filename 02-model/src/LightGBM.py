## -------------------------------------------------------------------##
## 0. PACKAGES

# Data working packages
import numpy as np
import pandas as pd
import lightgbm as lgb

## -------------------------------------------------------------------##
## 1. LOAD DATA 
import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split('Kaggle_taxi_regression',1)[0] + 'Kaggle_taxi_regression/')
from utils.utils_functions import *

CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split('Kaggle_taxi_regression',1)[0] + 'Kaggle_taxi_regression/01-preprocess/src')
if not os.path.exists('../results/train_clean.npz'):
    if not os.path.exists('../results/preprocessed_data_train.npz'):
        import preprocessing_featureeng as pc
        train = pc.train.copy()
        test = pc.test.copy()
    else:
        import EDA_cleaning as EDA
        train = EDA.train.copy()
        test = EDA.test.copy()
else:
    npz = np.load('../results/train_clean.npz')
    train = pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index').T
    npz = np.load('../results/test_clean.npz')
    test = pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index').T

CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split('Kaggle_taxi_regression',1)[0] + 'Kaggle_taxi_regression/02-model/src')

## -------------------------------------------------------------------##
## 1. Division of the data
X_train = train.drop('trip_duration', axis = 1)
y_train = train.trip_duration

X_test = test


## -------------------------------------------------------------------##
# 2. Building and the model 

#initialize lightgbm model 
model = lgb.LGBMRegressor(objective='regression',
                          metric = 'rmse')

#fit model 
model.fit(X_train, y_train)

#predict on train and test 
y_train_pred = model.predict(X_train)

#calculate performance in train
train_mse = np.sqrt(np.mean((y_train_pred - y_train)**2))
print('Train MSE: {}'.format(train_mse))


## -------------------------------------------------------------------##
# 3. Prediction of the model 

y_pred = model.predict(X_test)

# Apply revert_log2() function to y_pred to know the predicted duration in seconds
revert = np.vectorize(revert_log2)
y_pred = revert(y_pred)
y_pred

# Save our output predictions to send the submission.csv to the kaggle competition
submit_data = pd.DataFrame(columns=[['id', 'trip_duration']])
raw_data_test = pd.read_csv('../../exdata/test/test.csv')
submit_data['id'] = raw_data_test['id']
submit_data['trip_duration'] = y_pred
submit_data

# Save submission.csv
folder_create('../results')
submit_data.to_csv('../results/submission.csv', index=False)