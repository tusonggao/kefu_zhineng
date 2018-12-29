import numpy as np
import pandas as pd

random_seed = 42
np.random.seed(random_seed)

df_pos = pd.read_csv('./data/hive_sql_pos_instances_data.csv')
df_neg = pd.read_csv('./data/hive_sql_neg_instances_data_modified.csv')
df_neg = df_neg.sample(n=600000)

df_pos['y'] = 1
df_neg['y'] = 0

df_merged = pd.concat([df_pos, df_neg])

print('df_pos shape is ', df_pos.shape)
print('df_pos head is ', df_pos.head(3))

print('df_neg is ', df_neg.shape)
print('df_neg head is ', df_neg.head(3))

print('df_merged is ', df_merged.shape)
print('df_merged sample is ', df_merged.sample(10))

df_merged[['buy_user_id', 'creation_date', 'y']].to_csv('./data/hive_sql_merged_instances.csv')