import numpy as np
import pandas as pd

seed = 42


df_pos = pd.read_csv('./data/hive_sql_pos_instances_data.csv')
df_neg = pd.read_csv('./data/hive_sql_neg_instances_data_modified.csv')
df_neg = df_neg.sample(n=600000, random_state=42)

df_pos['y'] = 1
df_neg['y'] = 0

print('df_pos shape is ', df_pos.shape)
print('df_pos head is ', df_pos.head(3))

print('df_neg is ', df_neg.shape)
print('df_neg head is ', df_neg.head(3))

df_merged = 

print('df_pos shape is ', df_pos.shape)
print('df_pos head is ', df_pos.head(3))

print('df_neg is ', df_neg.shape)
print('df_neg head is ', df_neg.head(3))