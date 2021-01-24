import pandas as pd
import numpy as np

import os
import datetime


def make_timeseries(df, interpolate=None):
    #print(df.shape)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].notna()]

    year = pd.DatetimeIndex(df[date_col]).year.astype(np.int64)

    start = str(year[0]) + "-01-01 00:00"
    #     start

    end = str(year[0]) + "-12-31 23:00"
    #     end

    time_series = pd.date_range(start=start, end=end, freq='H')
    time_series = pd.DataFrame(time_series)
    time_series.columns = [date_col]

    df = pd.concat([time_series, df], axis=0)
    df = df.drop_duplicates([date_col], keep="last")
    df = df.sort_values([date_col], axis=0)

    if interpolate:
        print('interpolation')
        df = df.interpolate(metod='spline', order=2, limit_direction='both')

    #print(df.shape)
    return df


def make_dataframe(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y])
            print(file_names[loc][y])
            df_tmp = make_timeseries(pd.read_excel(path), interpolate=interpolate)

            #         df_tmp = pd.read_excel(path).dropna(axis='columns', how = 'all')
            #         print(df_tmp.head)
            df_loc.append(df_tmp)
        #             df_loc.append(pd.read_excel(path))
        df_full.append(pd.concat(df_loc))
        #     df.append(df_full[loc].iloc[:,2:11])
        df.append(eval(f"df_full[loc].iloc[{iloc_val}]"))
        #     df.append(df_full[loc].iloc[:, [2,3,4,5,6,7,10]])
        #     df.append(df_full[loc].iloc[:, 2:11])
        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)
        df[loc]['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df[loc]['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df[loc]['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df[loc]['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df[loc] = df[loc].reset_index(drop=True)

    return df
