# dn01: 172.21.57.123
# dn02: 172.21.57.124
# dn03: 172.21.57.125
# dn04: 172.21.57.126
# dn05: 172.21.57.127

# nm01: 172.21.57.128
# nm02: 172.21.57.129

###############################################################

from sqlalchemy import create_engine
import json
import time
import pandas as pd


def read_sql_script(sql_file):
    sql_str = ''
    with open(sql_file) as file:
        for line in file:
            content = line.strip()
            if content.find('/*')!=-1 and content.find('*/')!=-1:
                start_idx = content.index('/*')
                end_idx = content.index('*/')
                print('start_idx, end_idx ', start_idx, end_idx)
                content = content[:start_idx] + content[end_idx+2:]
            sql_str += ' ' + content
    return sql_str

print(read_sql_script('./sql_scripts/hive_sql_1_test.txt'))
sss = 'aaa ttt bbb ccc'
print('sss part of is ', sss[100:])


# with open("./111.json", 'r', encoding='UTF-8') as f:
# with open('./wanzheng.json', 'r', encoding='UTF-8') as f:
#     temp = json.loads(f.read())
#     #print(temp)
#     #print(temp['RECORDS'])
#     print(len(temp['RECORDS']))
#     print(temp['RECORDS'][1]['Id'])
# print('hello world!')


# start_t = time.time()

# print('reading start...')
# conn_impala = create_engine('impala://172.21.57.127:21050')
# # conn_impala = create_engine('hive://172.21.57.127:21050')
# # sql = 'show databases;'
# # sql = read_sql_script('./sql_scripts/hive_sql_5.txt')
# sql = read_sql_script('./sql_scripts/hive_sql_1.txt')
# print('sql is ', sql)
# df = pd.read_sql(sql, conn_impala)

# end_t = time.time()
# print('df.shape is', df.shape)

# print('df.head() is', df.head())
# print('read data cost time ', end_t-start_t)

# df.to_csv('./data/hive_sql_1_output.csv', index=0)
