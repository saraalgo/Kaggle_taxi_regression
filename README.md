# Kaggle_taxi_regression

This repository is already functional with the current version.

## About

NYC Taxi Duration - Regression LightGBM ML model

This repository is a improved version of the notebook presented to the Kaggle competition [New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration), can be found in the [THIS](https://www.kaggle.com/code/salvarezgonz/nyc-taxi-duration-regression-lightgbm-ml-model#4.-Training-of-LightGBM-model-for-Machine-Learning-(ML)) link.

By applying this repository, the **output.csv** will be created in the format required by the Kaggle competition.

## Prerequisites

1. Install python 3.7.9

## Installation

1. Clone the repository in your personal device using the following command:

```sh
git clone https://github.com/saraalgo/Kaggle_taxi_regression.git
```

2. Create and activate python environment if you do not count with the beforehand mentioned Python version. Otherwise, you could skip this step.

```sh
python3.7.9 -m venv Kaggle_taxi_regression/
source bin/activate
```

3. Upgrade pip and install project requirements 
```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project workflow

### How to run
You can run this repository script by script or all by once:
1. Script by script:
  - First execute *01-preprocess/src/preprocessing_featureeng.py*
```sh
cd 01-preprocess/src/
python preprocessing_featureeng.py
```
  - Then execute *01-preprocess/src/EDA_cleaning.py*
```sh
python EDA_cleaning.py
```
  - Finally execute *02-model/src/LightGBM.py*
```sh
cd ../../02-model/src/
python LightGBM.py
```

2. Run only the *02-model/src/LightGBM.py* script:
```sh
cd 02-model/src/
python LightGBM.py
```

### Output generated
Any of both options to run will result in a **02-model/results/submission.csv** in the format required for the above-metioned Kaggle competition