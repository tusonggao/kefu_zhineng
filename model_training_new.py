# https://datawhatnow.com/feature-importance/
# https://github.com/Microsoft/LightGBM/issues/826

import pickle
import time
import gc
import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)

def show_features_importance_bar(features, feature_importance):
    df_feat_importance = pd.DataFrame({
            'column': features,
            'importance': feature_importance,
        }).sort_values(by='importance', ascending=False)
    df_feat_importance.to_csv('./model_output/df_feat_importance.csv', index=0, sep='\t')

    plt.figure(figsize=(25, 6))
    #plt.yscale('log', nonposy='clip')
    plt.bar(range(len(feature_importance)), feature_importance, align='center')
    plt.xticks(range(len(feature_importance)), features, rotation='vertical')
    plt.title('Feature importance')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def split_by_user_id(df_merged, train_ratio=0.67):
    df_merged['md5_val'] = df_merged['buy_user_id'].apply(convert_2_md5)

    df_merged_sorted = df_merged.sort_values(by=['md5_val'])
    # print('df_merged_sorted head is ', df_merged_sorted.head(5))
    # df_merged_sorted.to_csv('./data/hive_sql_merged_instances_sorted.csv', 
    #     sep='\t', date_format='%Y/%m/%d', index=0)  # date_format='%Y-%m-%d %H:%M:%s'
    train_num = int(df_merged.shape[0]*train_ratio)
    pivot_val = df_merged_sorted.iloc[train_num]['md5_val']

    print('train_num is: ', train_num, 'pivot_val is: ', pivot_val)

    df_merged_train = df_merged_sorted[df_merged_sorted['md5_val']<=pivot_val]
    df_merged_test = df_merged_sorted[df_merged_sorted['md5_val']>pivot_val]

    # df_merged_train = df_merged_sorted[:train_num]
    # df_merged_test = df_merged_sorted[train_num:]

    # df_merged_train.to_csv('./data/hive_sql_merged_instances_train.csv', sep='\t', index=0)
    # df_merged_test.to_csv('./data/hive_sql_merged_instances_test.csv', sep='\t', index=0)

    return df_merged_train, df_merged_test


def compute_density_multiple(y_true, y_predict, threshold=10, by_percentage=True, top=True):
    df = pd.DataFrame({'y_true': y_true, 'y_predict': y_predict})
    df.sort_values(by=['y_predict'], ascending=False, inplace=True)

    density_whole = sum(df['y_true'])/df.shape[0]
    if by_percentage:
        if top:
            df_target = df[:int(threshold*0.01*df.shape[0])]
        else:
            df_target = df[-int(threshold*0.01*df.shape[0]):]
    else:
        if top:
            df_target = df[:threshold]
        else:
            df_target = df[-threshold:]
    density_partial = sum(df_target['y_true'])/df_target.shape[0]
    density_mutiple = density_partial/density_whole
    return density_mutiple


print('hello world')
df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', parse_dates=[1],
    infer_datetime_format=True, sep='\t')
# df_merged['creation_date'] = pd.to_datetime(df_merged['creation_date'], 
#     format='%Y-%m-%d %H:%M:%S')
# df_merged['gap_days'] = (df_merged['creation_date'] - df_merged['creation_date']).dt.days

sample_num = 2000000

#抽样100万做训练集
df_merged = df_merged.sample(n=sample_num, random_state=42)
print('df_merged shape is ', df_merged.shape)
print('df_merged dtypes is ', df_merged.dtypes)
# print('df_merged head is ', df_merged['gap_days'].head(10))

# split_by_user_id(df_merged)

###----------------------------###
###------ 加上R的feature  -----###
###----------------------------###
# df_recency = pd.read_csv('./data/hive_sql_R_data.csv', parse_dates=[1, 2], infer_datetime_format=True)
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
###------ 加上F的feature  ----- ###
###----------------------------###
df_frequency = pd.read_csv('./data/hive_sql_F_data.csv', parse_dates=[1], infer_datetime_format=True)
df_merged = pd.merge(df_merged, df_frequency, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add frequency is ', df_merged.shape)
print('df_merged.dtypes after add frequency is ', df_merged.dtypes)
del df_frequency

###----------------------------###
###------ 加上M的feature  -----###
###----------------------------###
df_monetary = pd.read_csv('./data/hive_sql_M_data.csv', parse_dates=[1], infer_datetime_format=True)
df_merged = pd.merge(df_merged, df_monetary, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add monetary is ', df_merged.shape)
print('df_merged.dtypes after add monetary is ', df_merged.dtypes)
del df_monetary


###---------------------------------###
###   加上first order的features      ###
###   包含：                         ###
###   1. 订单距离电话回访时间的天数    ###
###   2. 订单金额                    ###
###   3. 订单来源                    ###
###   4. 订单支付方式                ###
###---------------------------------###
df_first_order = pd.read_csv('./data/hive_sql_first_order_data.csv', 
                             parse_dates=[1, 2], infer_datetime_format=True,
                             dtype={'first_origin_type': str, 'first_payment_type': str})
df_first_order['gap_days_first_order'] = (df_first_order['creation_date'] - df_first_order['order_dt']).dt.days
df_first_order.drop(['order_dt'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_first_order, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add first order is ', df_merged.shape)
print('df_merged.dtypes after add first order is ', df_merged.dtypes)
del df_first_order


###---------------------------------###
###   加上last order的features       ###
###   包含：                         ###
###   1. 订单距离电话回访时间的天数    ###
###   2. 订单金额                    ###
###   3. 订单来源                    ###
###   4. 订单支付方式                ###
###---------------------------------###
df_last_order = pd.read_csv('./data/hive_sql_last_order_data.csv', 
                            parse_dates=[1, 2], infer_datetime_format=True,
                            dtype={'last_origin_type': str, 'last_payment_type': str})
df_last_order['gap_days_last_order'] = (df_last_order['creation_date'] - df_last_order['order_dt']).dt.days
df_last_order.drop(['order_dt'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_last_order, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add last order is ', df_merged.shape)
print('df_merged.dtypes after add last order is ', df_merged.dtypes)
del df_last_order


###----------------------------###
###--- 收货地址省份的feature  --###
###----------------------------###
df_address = pd.read_csv('./data/hive_sql_address_data.csv', dtype={'rand_address_code': str})
df_address.rename(columns={'rand_address_code':'address_code'}, inplace = True)
# df_address['address_code'] = df_address['rand_address_code'].apply(str)
# df_address.drop(['rand_address_code'], axis=1, inplace=True)
df_merged = pd.merge(df_merged, df_address, how='left', on=['buy_user_id'])
print('df_merged.shape after add address code is ', df_merged.shape)
print('df_merged.dtypes after add address code is ', df_merged.dtypes)
del df_address


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
del df_class_code


###----------------------------------------------------------###
###------ 加上start_app count的feature -----------------------###
###----------------------------------------------------------###
df_start_app_cnt = pd.read_csv('./data/hive_sql_startapp_cnt_data.csv')
df_start_app_cnt.rename(columns={'cnt':'start_app_cnt'}, inplace = True)
df_merged = pd.merge(df_merged, df_start_app_cnt, how='left', on=['buy_user_id', 'creation_date'])
print('df_merged.shape after add start_app count, branch_code code is ', df_merged.shape)
print('df_merged.dtypes after add start_app count, branch_code code is ', df_merged.dtypes)
del df_start_app_cnt


###----------------------------------------------------------###
###------ 加上电话回访时间所在的月份的feature -----------------###
###----------------------------------------------------------###
df_merged['call_month'] = df_merged['creation_date'].dt.month.apply(str)
df_merged['call_weekday'] = df_merged['creation_date'].dt.weekday.apply(str)
print('df_merged.shape after add start_app count, branch_code code is ', df_merged.shape)
print('df_merged.dtypes after add start_app count, branch_code code is ', df_merged.dtypes)


###----------------------------###
###--- 收货地址个数的feature  --###
###----------------------------###
df_address_num = pd.read_csv('./data/hive_sql_address_num_data.csv')
df_merged = pd.merge(df_merged, df_address_num, how='left', on=['buy_user_id'])
print('df_merged.shape after add address number feature is ', df_merged.shape)
print('df_merged.dtypes after add address number feature is ', df_merged.dtypes)
del df_address_num

print('\n-------------------------------------\n'
      '     data preprocess finished          \n'
      '---------------------------------------\n')

df_merged_train, df_merged_test = split_by_user_id(df_merged)
del df_merged

df_merged_train.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)
df_merged_test.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)

print('df_merged_train.shape df_merged_test.shape: ', df_merged_train.shape, df_merged_test.shape)

df_train_y = df_merged_train['y']
df_train_X = df_merged_train.drop(['y'], axis=1)

df_test_y = df_merged_test['y']
df_test_X = df_merged_test.drop(['y'], axis=1)


feature_names = df_train_X.columns.tolist()

d_train = lgb.Dataset(df_train_X.values, label=df_train_y.values, feature_name = feature_names, 
                categorical_feature=['address_code', 'class_code', 'branch_code', 
                'call_month', 'call_weekday', 'first_payment_type', 'first_origin_type', 
                'last_payment_type', 'last_origin_type', 'first_order_status', 'last_order_status'])

params = {'learning_rate':0.08, 'boosting_type':'gbdt', 'objective':'binary',
          'metric':'binary_logloss', 'sub_feature':0.85, 'sub_sample':0.7,
          'num_leaves':100, 'min_data':400, 'max_depth':13, 'random_state':42}

# lgbm = lgb.LGBMClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.08, 
#                          random_state=42, max_depth=13, min_child_samples=400,
#                          num_leaves=100, subsample=0.7, colsample_bytree=0.85,
#                          silent=-1, verbose=-1, boosting_type='gbdt')

print('lgb training starts')
start_t = time.time()
clf = lgb.train(params, d_train, 500)
print('lgb training ends, cost time', time.time()-start_t)

start_t = time.time()
y_pred=clf.predict(df_test_X.values)
print('y_pred.shape is ', y_pred.shape)
auc_score = roc_auc_score(df_test_y, y_pred)
print('auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)

print('top 200 ratio_multiple is',
      compute_density_multiple(df_test_y, y_pred, threshold=200, by_percentage=False),
      'top 500 ratio_multiple is',
      compute_density_multiple(df_test_y, y_pred, threshold=500, by_percentage=False),
      'ratio_multiple top 1 is ', 
      compute_density_multiple(df_test_y, y_pred, threshold=1), 
      'ratio_multiple top 5 is ', 
      compute_density_multiple(df_test_y, y_pred, threshold=5), 
      'ratio_multiple top 10 is ', 
      compute_density_multiple(df_test_y, y_pred, threshold=10),
      'ratio_multiple top 20 is ',
      compute_density_multiple(df_test_y, y_pred, threshold=20),
      'ratio_multiple top 30 is', 
      compute_density_multiple(df_test_y, y_pred, threshold=30)
)

importance = clf.feature_importance(importance_type='split')
feature_name = clf.feature_name()
feature_importance = pd.DataFrame({
                         'feature_name':feature_name,
                         'importance':importance}
                     ).sort_values(by='importance', ascending=False)
feature_importance.to_csv('./model_output/lgb_feat_importance_split.csv',index=False)

importance = clf.feature_importance(importance_type='gain')
feature_name = clf.feature_name()
feature_importance = pd.DataFrame({
                         'feature_name':feature_name,
                         'importance':importance}
                     ).sort_values(by='importance', ascending=False)
feature_importance.to_csv('./model_output/lgb_feat_importance_gain.csv',index=False)


# plt.figure(figsize=(12,6))
lgb.plot_importance(clf, max_num_features=30,  importance_type='split')
# plt.title("Feature importances")
plt.show()

print('program ends')