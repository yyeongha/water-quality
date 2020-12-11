import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from window_generator import WindowGenerator
from utils import *
from miss_data import MissData
from gain_data_generator import GainDataGenerator

def main():
    folder = 'data'
    file_names = ['가평_2019.xlsx', '의암호_2019.xlsx']

    # 1. sin cos
    day = 24 * 60 * 60
    year = (365.2425) * day
    df_full = []
    df = []
    for i in range(len(file_names)):
        path = os.path.join(folder, file_names[i])
        df_full.append(pd.read_excel(path))
        df.append(df_full[i].iloc[:, 2:11])
        date_time = pd.to_datetime(df_full[i].iloc[:, 0], format='%Y.%m.%d %H:%M')
        timestamp_s = date_time.map(datetime.datetime.timestamp)
        df[i]['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df[i]['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df[i]['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df[i]['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # 2. normalization
    df_all = pd.concat(df)
    train_mean = df_all.mean()
    train_std = df_all.std()
    for i in range(len(file_names)):
        df[i] = (df[i]-train_mean)/train_std
    train_df = df[0]
    val_df = df[0]
    test_df = df[0]

    # 3. ???
    w2 = WindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_width=6, 
        label_width=1, 
        shift=1,
        label_columns=None
    )

    print(w2)

    # 4. miss data
    norm_df = pd.concat(df,axis=0)
    norm_data = norm_df.to_numpy()
    MissData.save(norm_data, max_tseq = 12)

    # 5. gain data generator
    print(df)
    dgen = GainDataGenerator(df)


if __name__ == "__main__":
    main()