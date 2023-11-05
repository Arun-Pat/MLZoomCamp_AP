#!/usr/bin/env python
# coding: utf-8
# pip install xgboost

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

output_file = 'model_A.bin'

#Data Prep

df = pd.read_csv('autism_screening.csv')
# Make column names lowercase and replace spaces with underscore in name
# Also correct  some misspellings of column names

df.columns=df.columns.str.strip().str.lower().str.replace(' ','_').str.replace('/','_')
df = df.rename(columns={'austim' : 'autism', 'jundice': 'jaundice', 'contry_of_res' : 'country_of_res'})

# Change all categorical data to small case with space replaced by underscore
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(' ', '_')

# age column - Replace null and sole outlier with mean value
mean_age = int(df['age'].mean())
df.age = df.age.fillna(mean_age)
df.loc[df.age == df.age.max(), 'age'] = mean_age

# relation Column - Replace ? with most common value 
mode_relation = df['relation'].mode().iloc[0]
df['relation'].replace('?', mode_relation, inplace=True)

# ethnicity column - Replace ? with 'unknown' value 
df['ethnicity'].replace('?', 'unknown', inplace=True)

df.class_asd = (df.class_asd == 'yes').astype(int)


# ## DATA SPLIT and OneHot encoding
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

categorical = ['gender', 'ethnicity', 'jaundice', 'autism',  'country_of_res', 'relation']
numerical =['a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score', 
            'a6_score', 'a7_score', 'a8_score', 'a9_score', 'a10_score', 'age' ]

# Training

dv = DictVectorizer(sparse=False)
df_full_train_dict = df_full_train[numerical+categorical].to_dict(orient='records')
X_train = dv.fit_transform(df_full_train_dict)

test_dict = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)

y_train = df_full_train.class_asd.values
y_test = df_test.class_asd.values

# TRAIN XG Boost

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
print('Training with XG Boost:')

xgb_params = {
    'eta': 0.6,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)
y_pred = model.predict(dtest)

print('validation results:')
score = roc_auc_score(y_test, y_pred)
print('auc: ', score)
# ## SAVING Model and DV to The FILE using Pickle

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')


