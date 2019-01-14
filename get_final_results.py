import pandas as pd
import numpy as np

branch_code_map = {}
with open('F:/kefu_zhineng/sql_scripts_get_final_results/branch_code_map.txt', encoding='utf-8') as file:
    for line in file:
        branch_code_chinese, branch_code = line.strip().split()
        branch_code = branch_code.strip()
        branch_code_map[branch_code] = branch_code_chinese
print(branch_code_map)

# df_results = pd.read_csv('./data/hive_sql_scripts_get_final_results_data.csv', 
#                          dtype={'branch_code': int})
# print(df_results.head(10))

df_results = pd.read_csv('./data/hive_sql_scripts_get_final_results_new_data.csv', 
                         dtype={'branch_code': str, 'orders_code': str})
print(df_results.head(10))

print('df_results.columns.tolist() ', df_results.columns.tolist())

# print('uniques() is ', df_results['branch_code'].value_counts())
print('uniques() is ', df_results['branch_code'].unique())

top = 3000
df_results = df_results.iloc[:top]

df_results['branch_code_chinese'] = df_results['branch_code'].map(branch_code_map)
df_results = df_results[['buy_user_id', 'orders_code', 'last_orders_date', 'last_order_cost', 
                         'branch_code_chinese', 'branch_code', 'dept_code_cn', 'possibility']]


# buy_user_id,branch_code,possibility,orders_code,last_orders_date,last_order_cost

## 改变dataframe的列名：
df_results.rename(columns={'branch_code': '科室代码', 'branch_code_chinese': '科室名', 
                           'buy_user_id': '客户编码', 'possibility': '模型输出概率',
                           'orders_code': '订单编号', 'last_orders_date': '最近购买时间',
                           'last_order_cost': '购买金额', 'dept_code_cn': '所属部门'},
                  inplace=True)

# df_results.to_csv('./data/final_results_data_processed.csv', index=False, sep='\t')

df_results.to_excel('./data/final_results_data_processed_new.xlsx', index=False)

df_results[:500].to_excel('./data/final_results_data_processed_new_top500.xlsx', index=False)

df_results[:1000].to_excel('./data/final_results_data_processed_new_top1000.xlsx', index=False)


