# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os
import datetime
import calendar
from time import time
import time

import matplotlib.pyplot as plt

def make_columns(df):
    column_list = ['측정날짜', '측정소명', '수온', '수소이온농도','전기전도도', '용존산소', '총유기탄소', '총질소', '총인', '클로로필-a']
    list_df = pd.DataFrame(columns=column_list)
    list_df
    df = df.drop(columns=df.columns.difference(column_list))
    new_column = list_df.columns.difference(df.columns)
#     print(new_column)
    if not new_column.empty :
        print("Make_columns")
        for i in range(new_column.shape[0]):
            df[new_column[i]] = pd.Series()
#     print('columns')
#     print(df.columns)
    return df


# 녹조 조류 모니터링, 오염원
def make_timeseries_by_day(df, interpolation=None):

    if interpolation[1] == False:
        return df

    time_day = []
    #year = int(str(df.iloc[0, 0]).split('-')[0])
    year = int(str(df.iloc[0, 0])[0:4])
    #print(year)

    if df.shape[0] == 1:
        print('row in')
        ori_time = str(df.iloc[0, 0]) + "-06-15 12:00"
        ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
        df.iloc[0, 0] = ori_time
        return df
    elif df.shape[0] == 2:
        #print('ddddddddddd')
        #print(df.iloc[0, 0], df.iloc[1, 0])
        if df.iloc[0, 0] == df.iloc[1, 0]:
            ori_time = str(df.iloc[0, 0]) + "-03-15 12:00"
            ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
            df.iloc[0, 0] = ori_time

            ori_time = str(df.iloc[1, 0]) + "-09-15 12:00"
            ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
            df.iloc[1, 0] = ori_time
            return df


    div_cnt = df[df.columns[0]].value_counts().sort_index().tolist()

    for i in range(len(div_cnt)):

        month = calendar.monthrange(year, i + 1)
        for j in range(div_cnt[i]):
            time_val = datetime.timedelta(days=(month[1] / div_cnt[i] * j) + 1)

            time_day.append(
                '-' + str(time_val.days) + ' ' + str(time.strftime('%H:00:00', time.gmtime(time_val.seconds))))

    df.iloc[:, 0] = df.iloc[:, 0] + time_day

    return df

'''
web용
# example
# 0:자동, 1:수질 , 2:총량, 3:유량, 4:수위
de_iloc_val = ['9', '15', '17', '1', '1']
de_iloc_val[0]


def divideDataFrame(df_ori, de_iloc_val=0):
    df = df_ori
    #     print(de_iloc_val)
    #     print(type(de_iloc_val))
    df.set_index(df.columns[0], inplace=True)
    #     print(df)

    df_d = []
    for i in range(4):
        #         print((i*de_iloc_val))
        #         print((i*de_iloc_val)+de_iloc_val)
        #         print('aaaaaaa')
        df_tmp = df.iloc[:, (i * de_iloc_val):(i * de_iloc_val) + de_iloc_val]
        #         df_tmp.reset_index(inplace=True)
        df_d.append(df_tmp)

    return df_d
'''

def make_timeseries(df, interpolation=None, iloc_val= None):

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].notna()]
    df = df.dropna(thresh=3)

    year = pd.DatetimeIndex(df[date_col]).year.astype(np.int64)

    start = str(year[0]) + "-01-01 00:00"    #     start
    end = str(year[-1]) + "-12-31 23:00"#     end
    #print(year)

    print('time range in files : ', start, ' ~ ', end)

    time_series = pd.date_range(start=start, end=end, freq='H')

    time_series = pd.DataFrame(time_series)
    time_series.columns = [date_col]

    time_series = pd.concat([time_series, df], axis=0)
    time_series = time_series.drop_duplicates([date_col], keep="last")
    time_series = time_series.sort_values([date_col], axis=0)


    if interpolation[0]:
        for i in range(1, df.shape[1], 1):
            idx = df.iloc[:, i].dropna()

            if idx.shape[0] != 0:
                time_series.iloc[0, i] = idx.iloc[0]
                time_series.iloc[-1, i] = idx.iloc[-1]

    return time_series


def make_dataframe(directory_path, file_names, iloc_val, interpolation=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []


    for loc in range(len(file_names)):

        df_loc = []
        #print(file_names[loc])
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])

            df_tmp = pd.read_excel(path)

            print(path)
#
            df_tmp = make_timeseries_by_day(df = df_tmp, interpolation=interpolation)

            df_loc.append(df_tmp)


        df_loc = pd.concat(df_loc)
#
        df_loc = make_timeseries(df_loc, interpolation=interpolation, iloc_val=iloc_val)


        #print(df_loc)
        df_full.append(df_loc)


        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        #print(file_names[loc])

        #print(df[loc].shape, timestamp_s.shape)
        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year)))
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year)))

        df[loc] = df[loc].reset_index(drop=True)

        if interpolation[0]:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')

        #print('time_series.iloc[:, 10], time_series.iloc[:, 11]')
        #print(df[loc].iloc[:, 0], df[loc].iloc[:, 1])


    return df, date_time.reset_index(drop=True)




def make_dataframe_temp_12days(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        #print(file_names[loc])
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])
            #print(file_names[loc][y])
            #df_tmp = make_timeseries(pd.read_excel(path), interpolate=None)
            df_tmp = pd.read_excel(path)
#            print(df_tmp.shape)
            #         df_tmp = pd.read_excel(path).dropna(axis='columns', how = 'all')
            #         print(df_tmp.head)
            df_loc.append(df_tmp)
        #             df_loc.append(pd.read_excel(path))

        #if interpolate == True:

        df_loc = pd.concat(df_loc)
        #df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val, directory_path=directory_path)

        #print(df_loc)
        df_full.append(df_loc)

        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        #print(date_time)

        #print(df[loc].shape, timestamp_s.shape)
        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year)))
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year)))

        df[loc] = df[loc].reset_index(drop=True)

        #print('make_dataframe in file_open.py')
        #print(df[loc].head())
        #print(df[loc].tail())

        #print('시작과 끝 데이터')
        #print(df[loc].iloc[:1, :])
        #print(df[loc].iloc[-1:, :])

        if interpolate:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')
        #df[loc] = df[loc].interpolate(method='polynomial', order=3)



        #df[loc] = df[loc].interpolate(method='linear', order=3, axis=0)

        # plt.figure()
        # df[loc].plot()
        # plt.show()



    return df, date_time.reset_index(drop=True)





def make_dataframe_in_test(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        #print(file_names[loc])
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])
            #print(file_names[loc][y])
            #df_tmp = make_timeseries(pd.read_excel(path), interpolate=None)
            df_tmp = pd.read_excel(path)
#            print(df_tmp.shape)
            #         df_tmp = pd.read_excel(path).dropna(axis='columns', how = 'all')
            #         print(df_tmp.head)
            df_loc.append(df_tmp)
        #             df_loc.append(pd.read_excel(path))

        df_loc = pd.concat(df_loc)
        df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val, directory_path=directory_path)

        df_full.append(df_loc)

        #inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(df_full[loc])

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        #print(df[loc].shape, timestamp_s.shape)
        #df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day)))
        #df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day)))
        #df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year)))
        #df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year)))

        df[loc] = df[loc].reset_index(drop=True)

        if interpolate:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')

    return df, date_time.reset_index(drop=True)





#ori_time = str(df.iloc[0,0])+ "-06-15 12:00"
#ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
#df.iloc[0,0] = ori_time
#df = make_timeseries(df)