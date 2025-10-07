# *****************************  IMPORT LIBRARIES AND MODULES  *****************************

import os
import os.path as osp
import numpy as np
import pandas as pd
from time import time
import joblib

from scipy.io import arff
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# *****************************  READ DATA  *****************************

# function to read the arff file
def read_arff_file(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    return df, meta


# combine the arff files data in dataframe
def combine_arrf_df(data_fol):
    ll = os.listdir(data_fol)
    df_ll = []

    for name in ll:
        path = osp.join(data_fol, name)
        year = int(name[0])     # extract the year after which bank goes bankrupt
        df, _ = read_arff_file(path)

        df["year"] = year
        df_ll.append(df)

    df = pd.concat(df_ll, ignore_index=True)
    df = df.drop_duplicates()

    return df


data_fol = "data"

df = combine_arrf_df(data_fol)
print("Data shape: ", df.shape)

# *****************************  DATA IMPUTATION  *****************************

# fill the missing values in numerical dataframe
df = df.fillna(df.median())

# *****************************  DATA PREPARATION  *****************************

# change the data type 
df['class'] = df['class'].astype(int)

df['y'] = df['year']
df['y'][df['class']==0]=0   # set the value to 0 where there is no bankruptcy 

# drop the unusable columns
df = df.drop(columns=["year", "class"])

# get the input feature columns
input_columns = list(df.columns)
input_columns.remove("y")

# separate the input data and target variable
X = df[input_columns]
y = df['y']
y[y!=0]=1           # merge all bankruptcy years in single class

# split data into train/test
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size=0.2, random_state=42)


# SMOTE upsampling data
smote = SMOTE(random_state=42)
train_X2, train_y2 = smote.fit_resample(train_X, train_y)
train_X2.shape, train_y2.shape

# Data preprocessing 
sc = MinMaxScaler()
train_X2 = sc.fit_transform(train_X2)
test_X2 = sc.transform(test_X)

# save the scaler for later use
joblib.dump(sc, "min_max_scaler_full.pkl")


# *****************************  MODEL BUILDING  *****************************

metrics_dkt = {}

# XGBOOST TRAINING

print("\n\n")
print("-"*30)
print("XGBOOST Training")

xgb = XGBClassifier()

# select the parameters to run the model on
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [3, 7, 10, 20],
}


# do grid search on the parameters
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=4,
    n_jobs=-1
)

# fit the model for CV
t1 = time()
grid_search.fit(train_X2, train_y2)     
t2=time()

print("Time taken: ", t2-t1)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# get the best model with the best parameters trained
xgb = grid_search.best_estimator_


# Evaluate on test set
y_pred = xgb.predict(test_X2)
pre, rec, f1 = precision_score(test_y, y_pred), recall_score(test_y, y_pred), f1_score(test_y, y_pred)
pre, rec, f1 = round(pre, 3), round(rec, 3), round(f1, 3)

print("\n\n")
print("Test F1:", f1)
print("Test Precision:", pre)
print("Test Recall:", rec)

joblib.dump(xgb, "xgb.pkl")

metrics_dkt["xgboost"] = {"precision":pre, "recall":rec, "f1":f1}

# -------------------------------------------------------------

# LightGBM TRAINING 
print("\n\n")
print("-"*30)
print("LightGBM Training")

lgb = LGBMClassifier()

param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [-1, 5, 10],
}


grid_search = GridSearchCV(
    estimator=lgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=4,
    n_jobs=-1
)

t1 = time()
grid_search.fit(train_X2, train_y2)
t2=time()

print("Time taken: ", t2-t1)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

lgb = grid_search.best_estimator_


# Evaluate on test set
y_pred = lgb.predict(test_X2)
pre, rec, f1 = precision_score(test_y, y_pred), recall_score(test_y, y_pred), f1_score(test_y, y_pred)
pre, rec, f1 = round(pre, 3), round(rec, 3), round(f1, 3)

print("\n\n")
print("Test F1:", f1)
print("Test Precision:", pre)
print("Test Recall:", rec)

joblib.dump(lgb, "lgb.pkl")

metrics_dkt["lgb"] = {"precision":pre, "recall":rec, "f1":f1}

# -------------------------------------------------------------

# CATBOOST TRAINING
print("\n\n")
print("-"*30)
print("CatBoost Training")

cb = CatBoostClassifier()

param_grid = {
    'iterations': [100, 200, 500, 1000],     
    'depth': [4, 6, 8, 10],           
}


grid_search = GridSearchCV(
    estimator=cb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=4,
    n_jobs=-1
)

t1 = time()
grid_search.fit(train_X2, train_y2)
t2=time()

print("Time taken: ", t2-t1)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

cb = grid_search.best_estimator_

# Evaluate on test set
y_pred = cb.predict(test_X2)
pre, rec, f1 = precision_score(test_y, y_pred), recall_score(test_y, y_pred), f1_score(test_y, y_pred)
pre, rec, f1 = round(pre, 3), round(rec, 3), round(f1, 3)

print("\n\n")
print("Test F1:", f1)
print("Test Precision:", pre)
print("Test Recall:", rec)

joblib.dump(cb, "cb.pkl")

metrics_dkt["cb"] = {"precision":pre, "recall":rec, "f1":f1}

joblib.dump(cb, "cb.pkl")


# *****************************  RESULTS DISPLAY  *****************************
 
joblib.dump(metrics_dkt, "model_metrics.pkl") 

print("\n"*2)
print("*"*20)
print("Model test-data results\n")

for k,v in metrics_dkt.items():
    print("Model : ", k)
    print("Metrics: ", v)

