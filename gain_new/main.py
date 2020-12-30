import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *


# input parameter
DIR = 'data'
FILE_LIST = [['의암호_2018.xlsx'], ['의암호_2019.xlsx']]
MAX_EPOCHS = 100
DEBUG = True

df_list, df_full, df_all = create_dataframe(DIR, FILE_LIST)

standard_normalization(df_list, df_all)

wide_window = WindowGenerator(
    df=df_list,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=24 * 5,
    label_width=24 * 5,
    shift=0
)

_ = wide_window.example

val_performance = {}
performance = {}

gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)

def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=GAIN.RMSE_loss)
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    return history

history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS // 5)

if DEBUG:
    figure_loss(history)

val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

gain.save(save_dir='save')

# gain.evaluate(wide_window.test.repeat(), steps=100)

if DEBUG:
    figure_gain(df_list, wide_window, gain)
