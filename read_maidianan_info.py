from sqlalchemy import create_engine
import pandas as pd
import time

print('hello world!')

conn_sa = create_engine('impala://172.25.66.11:21050')

sql_tag = '/*SA(production)*/'

start_t = time.time()

sql = '''select u.second_id account_id, date, count(1) cnt 
from events e inner join users u on u.id = e.user_id 
where event='$AppStart' and date>='2017-01-01'
group by u.second_id, date order by u.second_id, date {sql_tag}; '''.format(sql_tag=sql_tag)

print(sql)
# df = pd.read_sql(sql, conn_sa)

# end_t = time.time()
# print('read_sql cost time: ', end_t-start_t)

# start_t = time.time()
# df.to_csv('./data/df_maidian_info.csv')
# end_t = time.time()
# print('read_sql cost time: ', end_t-start_t)

# print(df.head(5))
# print('df.shape is ', df.shape)

df.read_csv()


















# 172.25.66.11 data01.jianke.sa data01
# 172.25.66.12 data02.jianke.sa data02
# 172.25.66.13 data03.jianke.sa data03

# import sqlalchemy
# def conn():
#     return connect(host='some_host', 
#                    port=21050,
#                    database='default',
#                    timeout=20,
#                    use_ssl=True,
#                    ca_cert='some_pem',
#                    user=user, password=pwd,
#                    auth_mechanism='PLAIN')

# engine = sqlalchemy.create_engine('impala://', creator=conn)

# sql_tag = '/*SA(production)*/'
# sql = 'select * from events {sql_tag}'.format(sql_tag)


# sql = "select event from events e where event='$AppStart' and date>='2018-05-01' and date<='2018-05-03'"