import pandas as pd
import numpy as np
import os


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def check_outlier(file_name):
    print('check file: ', file_name)
    line_cnt = 0
    with open(file_name) as file:
        for line in file:
            line = line.strip()
            if not is_ascii(line):
                print(file_name, line_cnt, ' not ascii ')
            line_cnt += 1
            if line_cnt%1000000==0:
                print('current line_cnt is ', line_cnt)


extra_chars = [',', ';', '\t', '_', '-', '*']
file_names = []
for root, dirs, files in os.walk('./data/'):  
    # print(root)   # 当前目录路径  
    # print(dirs)   # 当前路径下所有子目录  
    # print(files)  # 当前路径下所有非目录子文件
    file_names += files


# hive_sql_neg_instances_data_modified.csv

file_names = ['hive_sql_patient_class_data.csv', 
              'hive_sql_pos_instances_data.csv',
              'hive_sql_R_data.csv', 
              'hive_sql_startapp_cnt_data.csv']

print('files is ', file_names)
for file_n in file_names:
    check_outlier('./data/' + file_n)



