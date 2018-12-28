import os
import json
import time
import pandas as pd
from sqlalchemy import create_engine

###############################################################

# 得到用户app最近3天以来app启动次数

###############################################################

def read_sql_script(sql_file):
    sql_str = ''
    with open(sql_file, encoding='UTF-8') as file:
        for line in file:
            content = line.strip()
            if content.find('/*')!=-1 and content.find('*/')!=-1:
                start_idx = content.index('/*')
                end_idx = content.index('*/')
                print('start_idx, end_idx ', start_idx, end_idx)
                content = content[:start_idx] + content[end_idx+2:]
            sql_str += ' ' + content
    return sql_str

def basefilename(full_file_name):
    file_name = os.path.split(full_file_name)[-1]
    raw_file_name = os.path.splitext(file_name)[0]
    print('raw_file_name is', raw_file_name)
    return raw_file_name

# print(basefilename('test.csv'))
# print(basefilename('c:/csv/test.csv'))
# print(basefilename('./test.csv'))

def read_and_store_df_from_impala(sql_file):
    print('reading start...')

    start_t = time.time()
    conn_impala = create_engine('impala://172.21.57.127:21050')
    sql = read_sql_script(sql_file)
    print('sql is ', sql)
    df = pd.read_sql(sql, conn_impala)
    end_t = time.time()

    print('read data from impala cost time ', end_t-start_t)
    print('df.shape is', df.shape)
    print('df.head() is', df.head())

    output_file = basefilename(sql_file) + '_data.csv'
    df.to_csv('./data/'+output_file, index=0)

    return df


# with open("./111.json", 'r', encoding='UTF-8') as f:
# with open('./wanzheng.json', 'r', encoding='UTF-8') as f:
#     temp = json.loads(f.read())
#     #print(temp)
#     #print(temp['RECORDS'])
#     print(len(temp['RECORDS']))
#     print(temp['RECORDS'][1]['Id'])
# print('hello world!')
# conn_impala = create_engine('hive://172.21.57.127:21050')
# sql = 'show databases;'
# sql = read_sql_script('./sql_scripts/hive_sql_5.txt')


# './sql_scripts/hive_sql_pos_instances.txt')

# df = read_and_store_df_from_impala('./sql_scripts/hive_sql_1.txt')
# df = read_and_store_df_from_impala('./sql_scripts/hive_sql_pos_instances.txt')
df = read_and_store_df_from_impala('./sql_scripts/hive_sql_neg_instances.txt')


# end_t = time.time()
# print('df.shape is', df.shape)
# print('df.head() is', df.head())
# print('read data cost time ', end_t-start_t)
# df.to_csv('./data/hive_sql_1_output.csv', index=0)
