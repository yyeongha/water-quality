# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os
import datetime
import calendar
from time import time
import time

import matplotlib.pyplot as plt



def make_timeseries_by_day(df, interpolation=None):

    if interpolation[1] == False:
        return df

    time_day = []
    year = int(str(df.iloc[0, 0])[0:4])

    if df.shape[0] == 1:
        print('row in')
        ori_time = str(df.iloc[0, 0]) + "-06-15 12:00"
        ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
        df.iloc[0, 0] = ori_time
        return df
    elif df.shape[0] == 2:
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


def make_timeseries(df, interpolation=None, iloc_val= None, loc=0, first_file_no=0, month=12, day=31):

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].notna()]
    df = df.dropna(thresh=3)

    year = pd.DatetimeIndex(df[date_col]).year.astype(np.int64)
    if loc==0 and first_file_no==0:
        month_tmp = pd.DatetimeIndex(df[date_col]).month.astype(np.int64)
        day_tmp = pd.DatetimeIndex(df[date_col]).day.astype(np.int64)
        month = month_tmp[-1] 
        day = day_tmp[-1] 
    else: 
        month = month
        day = day

    start = str(year[0]) + "-01-01 00:00"    #     start
    end = str(year[-1]) + "-" +str(month) + "-" + str(day) + " 23:00"#     end

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

    return time_series, month, day


def make_dataframe(directory_path, file_names, iloc_val, interpolation=None, first_file_no=0, month=12, day=31):
    day_for_sincos = 24 * 60 * 60
    year_for_sincos = (365.2425) * day_for_sincos

    df_full = []
    df = []


    for loc in range(len(file_names)):

        df_loc = []
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])

            df_tmp = pd.read_excel(path)

            df_tmp = make_timeseries_by_day(df = df_tmp, interpolation=interpolation)

            df_loc.append(df_tmp)


        df_loc = pd.concat(df_loc)

        df_loc, month, day = make_timeseries(df_loc, interpolation=interpolation, iloc_val=iloc_val, month=month, day=day, loc=loc, first_file_no=first_file_no)


        df_full.append(df_loc)


        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day_for_sincos)))
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day_for_sincos)))
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year_for_sincos)))
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year_for_sincos)))

        df[loc] = df[loc].reset_index(drop=True)

        if interpolation[0]:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')



    return df, date_time.reset_index(drop=True), month, day




def make_dataframe_temp_12days(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []

        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])

            df_tmp = pd.read_excel(path)

            df_loc.append(df_tmp)

        df_loc = pd.concat(df_loc)

        df_full.append(df_loc)

        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)


        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year)))
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year)))

        df[loc] = df[loc].reset_index(drop=True)


        if interpolate:
            
            df[loc] = df[loc].interpolate(method='pchip', order=3, limit_direction='both')




    return df, date_time.reset_index(drop=True)





def make_dataframe_in_test(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []

        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])

            df_tmp = pd.read_excel(path)

            df_loc.append(df_tmp)

        df_loc = pd.concat(df_loc)
        df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val, directory_path=directory_path)

        df_full.append(df_loc)


        df.append(df_full[loc])

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        df[loc] = df[loc].reset_index(drop=True)

        if interpolate:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')

    return df, date_time.reset_index(drop=True)


