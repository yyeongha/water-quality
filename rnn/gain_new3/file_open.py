import pandas as pd
import numpy as np

import os
import datetime

import matplotlib.pyplot as plt

def make_timeseries(df, interpolate=None, iloc_val= None):

    #print(df.shape)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].notna()]
    #print(df.shape)


    year = pd.DatetimeIndex(df[date_col]).year.astype(np.int64)

    start = str(year[0]) + "-01-01 00:00"    #     start
    end = str(year[-1]) + "-12-31 23:00"#     end
    #print(year)

    print('time range in files : ', start, ' ~ ', end)
    #print(start, end)
    #print('time range in files')
    #print(start, end)

    #freq = 'H'
    #if interpolate:
#        freq = 'M'

    time_series = pd.date_range(start=start, end=end, freq='H')

    time_series = pd.DataFrame(time_series)
    time_series.columns = [date_col]

    time_series = pd.concat([time_series, df], axis=0)
    time_series = time_series.drop_duplicates([date_col], keep="last")
    time_series = time_series.sort_values([date_col], axis=0)

   # print(time_series.shape)

    #df.append(eval(f"df_full[loc].iloc[{iloc_val}]"))

    #print (df.shape[0])
    #print(df.shape)

    #print(df.iloc[0][1])
    #print(df.iloc[0][2])

    if interpolate:
        for i in range(1, df.shape[1], 1):
            idx = df.iloc[:, i].dropna()

            if idx.shape[0] != 0:
                time_series.iloc[0, i] = idx.iloc[0]
                time_series.iloc[-1, i] = idx.iloc[-1]


            #print(idx[-1])
            #time_series.iloc[0][i] = df.iloc[0][1 + i]
            #time_series.iloc[-1][i] = df.iloc[-1][1 + i]
            #print(df.iloc[0][1+i])

        #print(df.iloc[0][1:])
        #print(time_series.iloc[0][1:].shape)

        #time_series.iloc[0][1:] = df.iloc[0][1:] #불러온 값의 첫값삽입
        #time_series.iloc[-1][1:] = df.iloc[-1][1:] #마지막값삽입

        #print(time_series.tail())


        #time_series = time_series.interpolate(method='polynomial', order=3, limit_direction='both') #보간 3차
        #time_series = time_series.interpolate(metod='spline', order=3, limit_direction='both')  # 보간 3차

        #print(time_series.shape)

        #testset = eval(f"time_series.iloc[{iloc_val}]")
        #print(testset.shape)
        #print(type(testset))
        #testset = testset.interpolate(method='linear')
        #print(testset.head())


        #time_series.iloc[:,28:30] = time_series.iloc[:,28:30].interpolate(method='polynomial', order=3)

        #time_series = time_series.interpolate(method='polynomial', order=3)
        #time_series = time_series.interpolate(method = 'polynomial', order=3, limit_direction='both')  # 보간 3차
        #time_series = time_series.interpolate(method='polynomial', order=3, limit=10000)  # 보간 3차
        #time_series[0] = time_series[0].interpolate(method='linear', order=2, axis=0,limit_direction='forward')  # 보간 3차


    #print(pd.__version__)
    #print('----------------------')
    #print(time_series)
    #print('----------------------')

    #plt.figure()
    #plt.plot(range())

    #print(df.shape)
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
        df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val)

        #print(df_loc)
        df_full.append(df_loc)

        #print(df_full[0].shape)
        #if interpolate:

        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        #plt.figure()
        #inter.plot()
 #       plt.subplot(3, 1, 1)
  #      inter.iloc[:, 0:4].plot()
   #     plt.subplot(3, 1, 2)
    #    inter.iloc[:, 4:8].plot()
     #   plt.subplot(3, 1, 3)
      #  inter.iloc[:, 8:12].plot()
        #plt.show()

        #print(inter.shape)
        #print(df_full[loc].iloc[:, 0])
        #     df.append(df_full[loc].iloc[:,2:11])
        #df.append(eval(f"df_full[loc].iloc[{iloc_val}]"))
        df.append(inter)
        #     df.append(df_full[loc].iloc[:, [2,3,4,5,6,7,10]])
        #     df.append(df_full[loc].iloc[:, 2:11])
        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        #print('timestamp_s')
        #print(timestamp_s)

        #df[loc]['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        #df[loc]['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        #df[loc]['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        #df[loc]['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
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

    return df
