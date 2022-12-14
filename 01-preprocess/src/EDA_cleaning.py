## -------------------------------------------------------------------##
## 0. PACKAGES

import os
CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split('Kaggle_taxi_regression',1)[0] + 'Kaggle_taxi_regression/')
from utils.utils_functions import *

CURRENT_PATH = os.getcwd()
os.chdir(CURRENT_PATH.split('Kaggle_taxi_regression',1)[0] + 'Kaggle_taxi_regression/01-preprocess/src')

# Data working packages
import numpy as np
import pandas as pd
import pandas_profiling as pdp

# Graph packages
import matplotlib.pyplot as plt
import seaborn as sns

## -------------------------------------------------------------------##
## 1. LOAD DATA PREPROCESSED (if preprocessing done, load it)

if not os.path.exists('../results/preprocessed_data_train.npz'):
    import preprocessing_featureeng as pc
    train = pc.train.copy()
    test = pc.test.copy()
else:
    npz = np.load('../results/preprocessed_data_train.npz')
    train = pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index').T
    npz = np.load('../results/preprocessed_data_test.npz')
    test = pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index').T

folder_create('../results/EDA')

## -------------------------------------------------------------------##
## 2. Exploratory Data Analysis (EDA) to clean the preprocessed data

# 2.1 Check the duplicated rows and leave only one in train data
print('Before train had {} and test had {} rows duplicated'.format(train.duplicated().sum(), test.duplicated().sum()))
train.drop_duplicates(keep='first', inplace=True)
print('Now train has {} and test has {} rows duplicated'.format(train.duplicated().sum(), test.duplicated().sum()))


# 2.2 Check output feature
print(train.trip_duration.describe())
## Transform to log2 to have a better distribution
##filter 1: log2 to trip duration
trip_duration_fil1 = train.copy()
trip_duration_fil1['trip_duration'] = np.log2(trip_duration_fil1['trip_duration']+1)
##calculate mean and standard deviation
mean = trip_duration_fil1['trip_duration'].mean()
sd = trip_duration_fil1['trip_duration'].std()
lower_bound = mean - (3 * sd)
upper_bound = mean + (3 * sd)
##filter out the outliers (filter 2)
trip_duration_fil2 = trip_duration_fil1[(trip_duration_fil1['trip_duration'] > lower_bound) & (trip_duration_fil1['trip_duration'] < upper_bound)]
## plot train trip duration
fig, axs = plt.subplots(1, 3, figsize=(20,4))
axs[0].hist(train.trip_duration, edgecolor='black')
axs[1].hist(trip_duration_fil1.trip_duration, edgecolor='black')
axs[2].hist(trip_duration_fil2.trip_duration, edgecolor='black')
sns.despine(left=True, bottom=True, ax=axs[0])
sns.despine(left=True, bottom=True, ax=axs[1])
sns.despine(left=True, bottom=True, ax=axs[2])
axs[0].set_title('Original trip duration train data')
axs[1].set_title('Log2 to trip duration')
axs[2].set_title('Without outliers trip duration')
plt.savefig('../results/EDA/trip_duration_train.png')
## save new trip duration in train data
train = trip_duration_fil2.reset_index(drop=True)


## 2.3 Check distance feature
if 'distance' in train:
    ## Convert km to meters
    train['distance'] = train['distance'] * 1000
    test['distance'] = test['distance'] * 1000
    ## Check the min and max distance in train and test
    print('Train minimun distance is {} and maximum is {} meters'.format(min(train.distance), max(train.distance)))
    print('Test minimun distance is {} and maximum is {} meters'.format(min(test.distance), max(test.distance)))
    ## Transform to log2 to have a better distribution
    #convert to log2
    train_f = train.copy()
    train_f['distance'] = np.log2(train_f['distance']+1)
    #calculate mean and standard deviation for train
    mean = train_f['distance'].mean()
    sd = train_f['distance'].std()
    lower_bound = mean - (3 * sd)
    upper_bound = mean + (3 * sd)
    #filter out the outliers
    train_f = train_f[(train_f['distance'] > lower_bound) & (train_f['distance'] < upper_bound)]
    ## plot train distance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    train.distance.plot.hist(ax=axes[0], title='Original train distance')
    train_f.distance.plot.hist(ax=axes[1], title='Distance without outliers train')
    plt.tight_layout()
    plt.savefig('../results/EDA/distance_train.png')
    ## Apply log2 in test and save both train and test new distances
    train = train_f.reset_index(drop=True)
    test['distance'] = np.log2(test['distance']+1)


## 2.4 Check distance and duration of googlemaps features
# DISTANCE MAPS
if 'distance_maps' in train:
    ## Convert km to meters
    train['distance_maps'] = train['distance_maps'] * 1000
    test['distance_maps'] = test['distance_maps'] * 1000
    ## Transform to log2 to have a better distribution
    #convert to log2
    train_f = train.copy()
    train_f['distance_maps'] = np.log2(train_f['distance_maps']+1)
    #calculate mean and standard deviation for train
    mean = train_f['distance_maps'].mean()
    sd = train_f['distance_maps'].std()
    lower_bound = mean - (3 * sd)
    upper_bound = mean + (3 * sd)
    #filter out the outliers
    train_f = train_f[(train_f['distance_maps'] > lower_bound) & (train_f['distance_maps'] < upper_bound)]
    ## plot train distance
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    train.distance.plot.hist(ax=axes[0], title='Original train distance maps')
    train_f.distance.plot.hist(ax=axes[1], title='Distance maps without outliers train')
    plt.tight_layout()
    plt.savefig('../results/EDA/distancemaps_train.png')
    ## Apply log2 in test and save both train and test new maps distances
    train = train_f.reset_index(drop=True)
    test['distance_maps'] = np.log2(test['distance_maps']+1)

# DURATION MAPS
if 'duration_maps' in train:
    ##filter 1: log2 to trip duration
    trip_duration_fil1 = train.copy()
    trip_duration_fil1['duration_maps'] = np.log2(trip_duration_fil1['duration_maps']+1)
    ##calculate mean and standard deviation
    mean = trip_duration_fil1['duration_maps'].mean()
    sd = trip_duration_fil1['duration_maps'].std()
    lower_bound = mean - (3 * sd)
    upper_bound = mean + (3 * sd)
    ##filter out the outliers (filter 2)
    trip_duration_fil2 = trip_duration_fil1[(trip_duration_fil1['duration_maps'] > lower_bound) & (trip_duration_fil1['duration_maps'] < upper_bound)]
    ## plot train trip duration
    fig, axs = plt.subplots(1, 3, figsize=(20,4))
    axs[0].hist(train.trip_duration, edgecolor='black')
    axs[1].hist(trip_duration_fil1.trip_duration, edgecolor='black')
    axs[2].hist(trip_duration_fil2.trip_duration, edgecolor='black')
    sns.despine(left=True, bottom=True, ax=axs[0])
    sns.despine(left=True, bottom=True, ax=axs[1])
    sns.despine(left=True, bottom=True, ax=axs[2])
    axs[0].set_title('Original duration maps train data')
    axs[1].set_title('Log2 to duration maps')
    axs[2].set_title('Without outliers duration maps')
    plt.savefig('../results/EDA/durationmaps_train.png')
    ## Apply log2 in test and save both train and test new maps distances
    train = trip_duration_fil2.reset_index(drop=True)
    test['duration_maps'] = np.log2(test['duration_maps']+1)


## 2.5 Check colinearity 
#Calculate the correlation matrix
corr_matrix = train.corr()
#Plot corr matrix
fig = plt.figure(figsize=(20,20))
ax = sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
ax.set_xticklabels(train.columns, rotation=90)
ax.set_yticklabels(train.columns)
fig.savefig('../results/EDA/corr_matrix.png', bbox_inches='tight')

# remove feature with correlation more than 0.8
columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[0]):
        if corr_matrix.iloc[i,j] >= 0.8:
            if columns[j]:
                columns[j] = False
selected_columns = train.columns[columns]
train = train[selected_columns]
selected_columns = selected_columns.drop('trip_duration')
test = test[selected_columns]

## -------------------------------------------------------------------##
## 3. Save a report profile of the cleaned data

report_train = pdp.ProfileReport(train, title='Report',minimal=True)
report_train.to_file('../results/EDA/report_train.html')

report_test = pdp.ProfileReport(test, title='Report',minimal=True)
report_test.to_file('../results/EDA/report_test.html')

## -------------------------------------------------------------------##
## 4. Save cleaned data

np.savez(os.path.join('../results/train_clean.npz'), **train)
np.savez(os.path.join('../results/test_clean.npz'), **test)