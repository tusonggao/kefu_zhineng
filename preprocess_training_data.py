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
    # df_merged_sorted.to_csv('./data/hive_sql_merged_instances_sorted.csv', sep='\t', date_format='%Y/%m/%d', index=0)  # date_format='%Y-%m-%d %H:%M:%s'
    row_n = df_merged.shape[0]
    train_num = int(row_n*train_ratio)
    pivot_val = df_merged_sorted.ix[train_num, 'md5_val']
    pivot_val = 'ac3a1976ceca523950645655fd18a927'
    # print('train_num is: ', train_num, 'pivot_val is: ', pivot_val)

    df_merged_train = df_merged_sorted[df_merged_sorted['md5_val']<=pivot_val]
    df_merged_test = df_merged_sorted[df_merged_sorted['md5_val']>pivot_val]
    # df_merged_train.to_csv('./data/hive_sql_merged_instances_train.csv', sep='\t', index=0)
    # df_merged_test.to_csv('./data/hive_sql_merged_instances_test.csv', sep='\t', index=0)

    return df_merged_train, df_merged_test

def merged_data_info(df_origin):
    pass


def compute_top_multiple(label_y, predict_y, top_ratio=0.1):
    df = pd.DataFrame()
    df['label_y'] = label_y
    df['predict_y'] = predict_y
    df.sort_values(by=['predict_y'], ascending=False, inplace=True)
    df_top_10_percent = df[:int(top_ratio*df.shape[0])]
    ratio = sum(df['label_y'])/df.shape[0]
    ratio_top_10_percent = sum(df_top_10_percent['label_y'])/df_top_10_percent.shape[0]
    ratio_mutiple = ratio_top_10_percent/ratio
    return ratio_mutiple


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
# df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', sep='\t')
# df_merged.to_csv('./data/hive_sql_merged_instances_comma.csv', index=0)

# df_merged = pd.read_csv('./data/hive_sql_merged_instances_comma.csv')

df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', parse_dates=[1],
    infer_datetime_format=True,
    sep='\t')
# df_merged['creation_date'] = pd.to_datetime(df_merged['creation_date'], 
#     format='%Y-%m-%d %H:%M:%S')
# df_merged['gap_days'] = (df_merged['creation_date'] - df_merged['creation_date']).dt.days

print('df_merged shape is ', df_merged.shape)
print('df_merged dtypes is ', df_merged.dtypes)
# print('df_merged head is ', df_merged['gap_days'].head(10))

# split_by_user_id(df_merged)

###----------------------------###
###------ 加上R的feature  -----###
###----------------------------###
df_recency = pd.read_csv('./data/hive_sql_R_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
# df_recency = pd.read_csv('./data/hive_sql_R_data.csv')
# df_recency['creation_date'] = pd.to_datetime(df_recency['creation_date'], 
#     format='%Y-%m-%d %H:%M:%S', errors='ignore')
# df_recency['recency_date'] = pd.to_datetime(df_recency['recency_date'], 
#     format='%Y-%m-%d %H:%M:%S', errors='ignore')

# df_recency['gap_days'] = (df_recency['creation_date'] - df_recency['recency_date']).dt.days
# df_merged = pd.merge(df_merged, df_recency, how='left', on=['buy_user_id', 'creation_date'])
# df_merged.drop(['recency_date'], axis=1, inplace=True)
# print('df_merged.shape after add R is ', df_merged.shape)
# print('df_merged.dtypes after add R is ', df_merged.dtypes)

# df_merged.drop(['gap_days'], axis=1, inplace=True)

###----------------------------###
###------ 加上F的feature  -----###
###----------------------------###
df_frequency = pd.read_csv('./data/hive_sql_F_data.csv', parse_dates=[1], infer_datetime_format=True)
df_merged = pd.merge(df_merged, df_frequency, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add frequency is ', df_merged.shape)
print('df_merged.dtypes after add frequency is ', df_merged.dtypes)

###----------------------------###
###------ 加上M的feature  -----###
###----------------------------###
df_monetary = pd.read_csv('./data/hive_sql_M_data.csv', parse_dates=[1], infer_datetime_format=True)
df_merged = pd.merge(df_merged, df_monetary, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add monetary is ', df_merged.shape)
print('df_merged.dtypes after add monetary is ', df_merged.dtypes)


###----------------------------###
###-- 加上first order的feature -###
###----------------------------###
df_first_order = pd.read_csv('./data/hive_sql_first_order_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
df_first_order['gap_days_first_order'] = (df_first_order['creation_date'] - df_first_order['order_dt']).dt.days
df_first_order.drop(['order_dt'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_first_order, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add first order is ', df_merged.shape)
print('df_merged.dtypes after add first order is ', df_merged.dtypes)


###----------------------------###
###-- 加上last order的feature -###
###----------------------------###
df_last_order = pd.read_csv('./data/hive_sql_last_order_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
df_last_order['gap_days_last_order'] = (df_last_order['creation_date'] - df_last_order['order_dt']).dt.days
df_last_order.drop(['order_dt'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_last_order, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add last order is ', df_merged.shape)
print('df_merged.dtypes after add last order is ', df_merged.dtypes)


###----------------------------###
###------ 加上地址的feature  ---###
###----------------------------###
df_address = pd.read_csv('./data/hive_sql_address_data.csv')
df_address['address_code'] = df_address['rand_address_code'].apply(str)
df_address.drop(['rand_address_code'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_address, how='left', on=['buy_user_id'])
print('df_merged.shape after add address code is ', df_merged.shape)
print('df_merged.dtypes after add address code is ', df_merged.dtypes)


###----------------------------------------------------------###
###------ 加上class_code 和 branch_code的feature -------------###
###----------------------------------------------------------###
df_class_code = pd.read_csv('./data/hive_sql_patient_class_data.csv')
df_class_code['class_code'] = df_class_code['class_code'].apply(str)
df_class_code['branch_code'] = df_class_code['branch_code'].apply(str)
df_merged = pd.merge(df_merged, df_class_code, how='left', on=['buy_user_id'])
print('df_merged.shape after add class_code, branch_code code is ', df_merged.shape)
print('df_merged.dtypes after add class_code, branch_code code is ', df_merged.dtypes)


###----------------------------------------------------------###
###------ 加上start_app count的feature -----------------------###
###----------------------------------------------------------###
df_start_app_cnt = pd.read_csv('./data/hive_sql_startapp_cnt_data.csv')
df_start_app_cnt.rename(columns={'cnt':'start_app_cnt'}, inplace = True)
df_merged = pd.merge(df_merged, df_start_app_cnt, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add start_app count, branch_code code is ', df_merged.shape)
print('df_merged.dtypes after add start_app count, branch_code code is ', df_merged.dtypes)

print('\n-------------------------------------\n'
      '     data preprocess finished          \n'
      '---------------------------------------\n');

# df_merged = pd.get_dummies(df_merged)
#为了加快训练速度，进行采样
# df_merged = df_merged.sample(100000) 

df_merged = pd.get_dummies(df_merged, columns=['address_code', 'class_code', 'branch_code'])
print('afte get_dummies, df_merged.shape is ', df_merged.shape)

df_merged_train, df_merged_test = split_by_user_id(df_merged)
df_merged_train.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)
df_merged_test.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)

print('df_merged_train.shape df_merged_test.shape: ', df_merged_train.shape, df_merged_test.shape)

# df_merged_train = pd.get_dummies(df_merged_train)
# df_merged_test = pd.get_dummies(df_merged_test)

df_train_y = df_merged_train['y']
df_train_X = df_merged_train.drop(['y'], axis=1)

df_test_y = df_merged_test['y']
df_test_X = df_merged_test.drop(['y'], axis=1)


print('start training')
start_t = time.time()

lgbm = lgb.LGBMClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.08, 
                         random_state=42, max_depth=13, min_child_samples=400,
                         num_leaves=100, subsample=0.7, colsample_bytree=0.85,
                         silent=-1, verbose=-1)

# lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#         eval_metric='auc', verbose=200, early_stopping_rounds=600)

lgbm.fit(df_train_X, df_train_y)
print('training ends, cost time: ', time.time()-start_t)

start_t = time.time()
print('predict starting')
y_predictions = lgbm.predict_proba(df_test_X)
auc_score = roc_auc_score(df_test_y, y_predictions[:, 1])



print('auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)

ratio_multiple_10 = compute_top_multiple(df_test_y, y_predictions[:, 1], top_ratio=0.1)
ratio_multiple_20 = compute_top_multiple(df_test_y, y_predictions[:, 1], top_ratio=0.2)
ratio_multiple_30 = compute_top_multiple(df_test_y, y_predictions[:, 1], top_ratio=0.3)

print('ratio_multiple 10 is ', ratio_multiple_10, 
      'ratio_multiple 20 is ', ratio_multiple_20)

# y_pred_proba = clf.predict_proba(df_test)
# auc = roc_auc_score(y_test, y_pred_proba[:, 1])


feature_names = df_train_X.columns.values.tolist()

# print(pd.DataFrame({
#         'column': feature_names,
#         'importance': lgbm.feature_importances_,
#     }).sort_values(by='importance', ascending=False))

df_feat_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgbm.feature_importances_,
    }).sort_values(by='importance', ascending=False)
df_feat_importance.to_csv('./model_output/df_feat_importance.csv', index=0, sep='\t')


# print(pd.DataFrame({
#         'column': feature_names,
#         'importance': lgbm.feature_importance(),
#     }).sort_values(by='importance'))

# gbdt.feature_importances_

print('program ends')