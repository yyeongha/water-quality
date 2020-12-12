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
from gain import GAIN

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

    # 3.
    train_df = df_all
    val_df = df_all
    test_df = df_all

    # 4.
    wide_window = WindowGenerator(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_width=24*3, 
        label_width=24*3, 
        shift=0
    )

    # ???
    wide_window.plot(plot_col='총질소')

    # 5.
    val_performance = {}
    performance = {}

    # issue
    gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
    gain.compile(loss=GAIN.RMSE_loss)

    # 6.
    MAX_EPOCHS = 300

    def compile_and_fit(model, window, patience=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

        model.compile(loss=GAIN.RMSE_loss)
        history = model.fit(window.train, 
                            epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history
        
    history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS//5)
    val_performance['Gain'] = gain.evaluate(wide_window.val)
    performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)



if __name__ == "__main__":
    main()