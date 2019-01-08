# https://datawhatnow.com/feature-importance/

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
    return str(hashlib.md5(str(value).encode('utf-8')).hexdigest())

def split_by_user_id(df_merged, train_ratio=0.67):
    df_merged['md5_val'] = df_merged['buy_user_id'].apply(convert_2_md5)

    print('df_merged.dtypes is ', df_merged.dtypes)

    df_merged_sorted = df_merged.sort_values(by=['md5_val'])
    # print('df_merged_sorted head is ', df_merged_sorted.head(5))
    df_merged_sorted.to_csv('./data/hive_sql_merged_instances_sorted.csv', 
        sep='\t', date_format='%Y/%m/%d', index=0)  # date_format='%Y-%m-%d %H:%M:%s'
    train_num = int(df_merged.shape[0]*train_ratio)
    pivot_val = df_merged_sorted.iloc[train_num]['md5_val']
    pivot_val_1 = df_merged_sorted.at[train_num, 'md5_val']
    # pivot_val = 'ac3a1976ceca523950645655fd18a927'

    print('train_num is: ', train_num, 'pivot_val is: ', pivot_val, pivot_val_1)

    df_merged_train = df_merged_sorted[df_merged_sorted['md5_val']<=pivot_val]
    df_merged_test = df_merged_sorted[df_merged_sorted['md5_val']>pivot_val]

    # df_merged_train = df_merged_sorted[:train_num]
    # df_merged_test = df_merged_sorted[train_num:]

    df_merged_train.to_csv('./data/hive_sql_merged_instances_train.csv', sep='\t', index=0)
    df_merged_test.to_csv('./data/hive_sql_merged_instances_test.csv', sep='\t', index=0)

    return df_merged_train, df_merged_test

def merged_data_info(df_origin):
    pass


def compute_top_multiple(label_y, predict_y, threshold=10, by_percentage=True):
    df = pd.DataFrame()
    df['label_y'] = label_y
    df['predict_y'] = predict_y
    df.sort_values(by=['predict_y'], ascending=False, inplace=True)
    ratio_whole = sum(df['label_y'])/df.shape[0]
    if by_percentage:        
        df_top = df[:int(threshold*0.01*df.shape[0])]
    else:
        df_top = df[:threshold]        
    ratio_top = sum(df_top['label_y'])/df_top.shape[0]
    ratio_mutiple = ratio_top/ratio_whole
    return ratio_mutiple


def compute_bottom_multiple(label_y, predict_y, threshold=10, by_percentage=True):
    df = pd.DataFrame()
    df['label_y'] = label_y
    df['predict_y'] = predict_y
    df.sort_values(by=['predict_y'], ascending=False, inplace=True)
    ratio_whole = sum(df['label_y'])/df.shape[0]
    if by_percentage:        
        df_bottom = df[-int(threshold*0.01*df.shape[0]):]
    else:
        df_bottom = df[-threshold:]        
    ratio_bottom = sum(df_bottom['label_y'])/df_bottom.shape[0]
    ratio_mutiple = ratio_bottom/ratio_whole
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
df_address = pd.read_csv('./data/hive_sql_address_data.csv', dtype={'rand_address_code': str})
df_address.rename(columns={'rand_address_code':'address_code'}, inplace = True)
# df_address['address_code'] = df_address['rand_address_code'].apply(str)
# df_address.drop(['rand_address_code'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_address, how='left', on=['buy_user_id'])
print('df_merged.shape after add address code is ', df_merged.shape)
print('df_merged.dtypes after add address code is ', df_merged.dtypes)


###----------------------------------------------------------###
###------ 加上class_code 和 branch_code的feature -------------###
###----------------------------------------------------------###
df_class_code = pd.read_csv('./data/hive_sql_patient_class_data.csv', 
                            dtype={'class_code': str, 'branch_code': str})
# df_class_code['class_code'] = df_class_code['class_code'].apply(str)
# df_class_code['branch_code'] = df_class_code['branch_code'].apply(str)
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
print('top 200 ratio_multiple is',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False), 
      'top 500 ratio_multiple is',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=500, by_percentage=False), 
      'ratio_multiple top 1 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=1), 
      'ratio_multiple top 5 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=5), 
      'ratio_multiple top 10 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=10), 
      'ratio_multiple top 20 is ',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=20),
      'ratio_multiple top 30 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=30),
      'ratio_multiple top 40 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=40),
      'ratio_multiple top 50 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=50),
      'bottom 200 ratio_multiple is',
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False), 
      'ratio_multiple bottom 1 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=1), 
      'ratio_multiple bottom 5 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=5), 
      'ratio_multiple bottom 10 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=10), 
      'ratio_multiple bottom 20 is ',
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=20),
      'ratio_multiple bottom 30 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=30),
      'ratio_multiple bottom 40 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=40),
      'ratio_multiple bottom 50 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=50),
      )

# y_pred_proba = clf.predict_proba(df_test)
# auc = roc_auc_score(y_test, y_pred_proba[:, 1])


feature_names = df_train_X.columns.values.tolist()
df_feat_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgbm.feature_importances_,
    }).sort_values(by='importance', ascending=False)
df_feat_importance.to_csv('./model_output/df_feat_importance.csv', index=0, sep='\t')

# df_feat_importance[:22].plot.bar(x='column', y='importance', rot=0)
# plt.show()

def show_features_importance_bar(features, feature_importance):
    plt.figure(figsize=(25, 6))
    #plt.yscale('log', nonposy='clip')
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), features, rotation='vertical')
    plt.title('Feature importance')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()
    

show_features_importance_bar(df_feat_importance['column'][:25],
                             df_feat_importance['importance'][:25])


lgbm = lgb.LGBMClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.08, 
                         random_state=42, max_depth=7, min_child_samples=500,
                         num_leaves=55, subsample=0.7, colsample_bytree=0.85,
                         silent=-1, verbose=-1)

# lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#         eval_metric='auc', verbose=200, early_stopping_rounds=600)

lgbm.fit(df_train_X, df_train_y)
print('new training ends, cost time: ', time.time()-start_t)
start_t = time.time()
y_predictions = lgbm.predict_proba(df_test_X)
auc_score = roc_auc_score(df_test_y, y_predictions[:, 1])

print('new auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)
print('top 200 ratio_multiple is',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False), 
      'top 500 ratio_multiple is',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=500, by_percentage=False), 
      'ratio_multiple top 1 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=1), 
      'ratio_multiple top 5 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=5), 
      'ratio_multiple top 10 is ', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=10), 
      'ratio_multiple top 20 is ',
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=20),
      'ratio_multiple top 30 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=30),
      'ratio_multiple top 40 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=40),
      'ratio_multiple top 50 is', 
      compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=50),
      'bottom 200 ratio_multiple is',
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False), 
      'ratio_multiple bottom 1 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=1), 
      'ratio_multiple bottom 5 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=5), 
      'ratio_multiple bottom 10 is ', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=10), 
      'ratio_multiple bottom 20 is ',
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=20),
      'ratio_multiple bottom 30 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=30),
      'ratio_multiple bottom 40 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=40),
      'ratio_multiple bottom 50 is', 
      compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=50),
      )

print('program ends')