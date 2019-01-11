import numpy as np
import pandas as pd

random_seed = 42
np.random.seed(random_seed)

df_pos = pd.read_csv('./data/hive_sql_pos_instances_data.csv')
df_pos.drop(['orders_code', 'order_cost'], axis=1, inplace=True)

df_neg = pd.read_csv('./data/hive_sql_neg_instances_data.csv')

# 按照正负样例1:2进行筛选
df_neg = df_neg.sample(n=df_pos.shape[0]*2)  

df_pos['y'] = 1
df_neg['y'] = 0

df_merged = pd.concat([df_pos, df_neg])

print('df_pos shape is ', df_pos.shape)
print('df_pos head is ', df_pos.head(3))

print('df_neg is ', df_neg.shape)
print('df_neg head is ', df_neg.head(3))

print('df_merged is ', df_merged.shape)

df_merged[['buy_user_id', 'creation_date', 'y']].to_csv(
    './data/hive_sql_merged_instances.csv', sep='\t', index=0)

# df_merged_sampled = df_merged.sample(n=1000000)  #筛选出100万
# print('df_merged sample is ', df_merged.sample(10))
# df_merged_sampled[['buy_user_id', 'creation_date', 'y']].to_csv(
#     './data/hive_sql_merged_instances_sampled.csv', sep='\t', index=0)

print('prog ends')