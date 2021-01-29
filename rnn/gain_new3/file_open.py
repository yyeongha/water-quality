import pandas as pd
import numpy as np

import os
import datetime

import matplotlib.pyplot as plt

def make_timeseries(df, interpolate=None, iloc_val= None, directory_path = ''):

    #print(df.shape)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].notna()]
    #print(df.shape)


    #print('directory_path')
    directory_path = directory_path.split('/')[1]
    if directory_path == 'ASOS':
        df = df.dropna(thresh=3)
    elif directory_path == 'AWS':
        df = df.dropna(thresh=3)
    elif directory_path == '수위':
        df = df.dropna(thresh=3)

    #print(df.head())

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



    if interpolate:
        for i in range(1, df.shape[1], 1):
            idx = df.iloc[:, i].dropna()

            if idx.shape[0] != 0:
                time_series.iloc[0, i] = idx.iloc[0]
                time_series.iloc[-1, i] = idx.iloc[-1]


    return time_series


def make_dataframe(directory_path, file_names, iloc_val, interpolate=None):
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

            df_loc.append(df_tmp)


        df_loc = pd.concat(df_loc)
        df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val, directory_path=directory_path)

        #print(df_loc)
        df_full.append(df_loc)


        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)


        print(df[loc].shape, timestamp_s.shape)
        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day)))
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year)))
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year)))

        df[loc] = df[loc].reset_index(drop=True)

        if interpolate:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both')


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

        print(df[loc].shape, timestamp_s.shape)
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
