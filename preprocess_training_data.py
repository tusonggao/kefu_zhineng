import pickle
import time
import gc
import os
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


random_seed = 42
np.random.seed(random_seed)


def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()


def split_by_user_id(df_merged, train_ratio=0.67):
    df_merged['md5_val'] = df_merged['buy_user_id'].apply(convert_2_md5)

    print('df_merged.dtypes is ', df_merged.dtypes)

    df_merged_sorted = df_merged.sort_values(by=['md5_val'])
    # print('df_merged_sorted head is ', df_merged_sorted.head(5))
    df_merged_sorted.to_csv('./data/hive_sql_merged_instances_sorted.csv', sep='\t', date_format='%Y/%m/%d', index=0)  # date_format='%Y-%m-%d %H:%M:%s'
    row_n = df_merged.shape[0]
    train_num = int(row_n*train_ratio)
    pivot_val = df_merged_sorted.ix[train_num, 'md5_val']
    pivot_val = 'ac3a1976ceca523950645655fd18a927'
    # print('train_num is: ', train_num, 'pivot_val is: ', pivot_val)

    df_merged_train = df_merged_sorted[df_merged_sorted['md5_val']<=pivot_val]
    df_merged_test = df_merged_sorted[df_merged_sorted['md5_val']>pivot_val]
    df_merged_train.to_csv('./data/hive_sql_merged_instances_train.csv', sep='\t', index=0)
    df_merged_test.to_csv('./data/hive_sql_merged_instances_test.csv', sep='\t', index=0)

    return df_merged_train, df_merged_test


def merged_data_info(df_origin):
    pass

# df_pos = pd.read_csv('./data/hive_sql_pos_instances_data.csv')
# df_neg = pd.read_csv('./data/hive_sql_neg_instances_data_modified.csv')
# df_neg = df_neg.sample(n=600000)
# df_pos['y'] = 1
# df_neg['y'] = 0
# df_merged = pd.concat([df_pos, df_neg])
# print('df_pos shape is ', df_pos.shape)
# print('df_pos head is ', df_pos.head(3))
# print('df_neg is ', df_neg.shape)
# print('df_neg head is ', df_neg.head(3))

print('hello world')
df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', sep='\t')
# df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', parse_dates=[1],
#     infer_datetime_format=True,
#     sep='\t')
df_merged['creation_date'] = pd.to_datetime(df_merged['creation_date'], 
    format='%Y-%m-%d %H:%M:%S', errors='ignore')
print('df_merged shape is ', df_merged.shape)
print('df_merged dtypes is ', df_merged.dtypes)
print('df_merged head is ', df_merged['creation_date'].head(10))

# split_by_user_id(df_merged)

# df_recency = pd.read_csv('./data/hive_sql_R_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
df_recency = pd.read_csv('./data/hive_sql_R_data.csv')


df_recency['creation_date'] = pd.to_datetime(df_recency['creation_date'], 
    format='%Y-%m-%d %H:%M:%S', errors='ignore')
df_recency['recency_date'] = pd.to_datetime(df_recency['recency_date'], 
    format='%Y-%m-%d %H:%M:%S', errors='ignore')
df_recency['gap_days'] = (df_recency['creation_date'] - df_recency['recency_date']).dt.days


print('df_recency dtypes is ', df_recency.dtypes)
print('df_recency sample is ', df_recency['gap_days'].tail(10))

print('finished process!')


# train_df = merged_df[merged_df['prediction_pay_price']!=-99999]
# train_y = train_df['prediction_pay_price'].values
# train_X = train_df.drop(['prediction_pay_price'], axis=1).values

# test_df = merged_df[merged_df['prediction_pay_price']==-99999]
# test_X = test_df.drop(['prediction_pay_price'], axis=1).values

#del merged_df, train_df, test_df
#gc.collect()

# X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
#             train_y, test_size=0.25, random_state=42)
# print('X_train_new.shape is {}, X_val_new.shape is {}, test_X.shape is {}'.format(
#       X_train_new.shape, X_val_new.shape, test_X.shape))


#lgbm = lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1, learning_rate=0.08, 
#                          random_state=42, max_depth=13, min_child_samples=400,
#                          num_leaves=700, subsample=0.7, colsample_bytree=0.85,
#                          silent=-1, verbose=-1)


#lgbm = lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1, learning_rate=0.05, 
#                          random_state=42, max_depth=7, min_child_samples=150,
#                          num_leaves=3000, subsample=0.7, colsample_bytree=0.8,
#                          silent=-1, verbose=-1)

#lgbm = lgb.LGBMRegressor(n_estimators=6000, n_jobs=-1, learning_rate=0.08, 
#                          random_state=42, max_depth=13, min_child_samples=800,
#                          num_leaves=151, subsample=0.8, colsample_bytree=0.9,
#                          boosting_type='dart', reg_alpha=0.1, reg_lambda=0.05,
#                          silent=-1, verbose=-1)

#lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#         eval_metric=rmse, verbose=200, early_stopping_rounds=600)

#lgbm.fit(train_X, train_y, eval_set=[(X_train_new, y_train_new), 
#        (X_val_new, y_val_new)], eval_metric=rmse, 
#        verbose=200, early_stopping_rounds=500)
    
#lgbm.fit(X_train_new, y_train_new)



