import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from core.data_generator import DataGenerator
from core.window_generator import WindowGenerator
from core.utils import *

# font for korean
import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothicCoding-Bold'


target_col = '총유기탄소'

input_step = 24*7
OUT_STEPS = 24*3
fill_no = 3
# input_step = 2
# OUT_STEPS = 1

MAX_EPOCHS = 1
batch_size = 32

gain_folder = './data/test/gain'
original_folder = './data/test/origin'
# file_list = [['서상_2018.xlsx', '서상_2019.xlsx'], ['의암호_2018.xlsx','의암호_2019.xlsx']]
# file_list = [['서상_2019.xlsx'], ['의암호_2019.xlsx']]
file_list = [['서상_2019.xlsx']]

df = createDataFrame(gain_folder, file_list)
origin_df = createDataFrame(original_folder, file_list)

# origin_df.to_excel("./origin_df.xlsx", index=False)
# df.to_excel("./df.xlsx", index=False)
# interpolate(origin_df, max_gap=2)

# print(type(df))
num_features = df.shape[1]
label_columns_indices = {name: i for i, name in enumerate(df)}
target_col_idx = label_columns_indices[target_col]
# print(target_col_idx)



dgen = DataGenerator(df, origin_data= origin_df, fill_no=fill_no, input_width=input_step, label_width=OUT_STEPS, target_col_idx=target_col_idx)
n = len(df)
# print(n)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

multi_window = WindowGenerator(
    df=df,
    train_df=train_df,
    val_df=train_df,
    test_df=train_df,
    input_width=input_step,
    label_width=OUT_STEPS,
    shift=1,
    label_columns=target_col
)


a , b = multi_window.example

def compile_and_fit(model, window, patience=3):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=0.0001, patience=5, verbose=1)
    adam = tf.keras.optimizers.Adam(learning_rate=0.1)
    # model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanSquaredError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS, batch_size = batch_size, validation_data=window.val)

    return history


multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])


history = compile_and_fit(multi_lstm_model, multi_window, 10)

multi_window.plot(multi_lstm_model)
