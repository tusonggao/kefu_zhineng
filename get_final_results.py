import pandas as pd
import numpy as np

branch_code_map = {}
with open('F:/kefu_zhineng/sql_scripts_get_final_results/branch_code_map.txt', encoding='utf-8') as file:
    for line in file:
        branch_code_chinese, branch_code = line.strip().split()
        branch_code = int(branch_code)
        branch_code_map[branch_code] = branch_code_chinese
print(branch_code_map)

df_results = pd.read_csv('./data/hive_sql_scripts_get_final_results_data.csv', 
                         dtype={'branch_code': int})
print(df_results.head(10))

print('df_results.columns.tolist() ', df_results.columns.tolist())

# print('uniques() is ', df_results['branch_code'].value_counts())
print('uniques() is ', df_results['branch_code'].unique())

top = 500
df_results = df_results.iloc[:top]
df_results['branch_code_chinese'] = df_results['branch_code'].map(branch_code_map)
df_results = df_results[['buy_user_id', 'branch_code_chinese', 'branch_code', 'possibility']]

## 改变dataframe的列名：
df_results.rename(columns={'branch_code': '科室代码', 'branch_code_chinese': '科室名', 
                           'buy_user_id': '客户编码', 'possibility': '模型输出概率'},
                  inplace=True)

# df_results.to_csv('./data/final_results_data_processed.csv', index=False, sep='\t')
df_results.to_excel('./data/final_results_data_processed.xlsx', index=False)


