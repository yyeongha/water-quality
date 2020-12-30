import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *


# input parameter
parameters_dir = './input'
parameters_file = 'input.json'
parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)
with open(parameters_path, encoding='utf8') as json_file:
    parameters = json.load(json_file)

# init input parameter
gain_parameters = parameters['gain']

DEBUG = gain_parameters['debug']
max_epochs = gain_parameters['max_epochs']
input_width = gain_parameters['input_width']
label_width = gain_parameters['label_width']
shift = gain_parameters['shift']
use_train = gain_parameters['use_train']
batch_size = gain_parameters['batch_size']
miss_pattern = gain_parameters['miss_pattern']
miss_rate = gain_parameters['miss_rate']
fill_no = gain_parameters['fill_no']

df_list, df_full, df_all = create_dataframe(gain_parameters['data_dir'], gain_parameters['data_file'])

standard_normalization(df_list, df_all)

wide_window = WindowGenerator(
    df=df_list,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=input_width,
    label_width=label_width,
    shift=shift,
    batch_size=batch_size,
    miss_pattern=miss_pattern,
    miss_rate=miss_rate,
    fill_no=fill_no
)

_ = wide_window.example

val_performance = {}
performance = {}

gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)

if use_train:
    def compile_and_fit(model, window, patience=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
        model.compile(loss=GAIN.RMSE_loss)
        history = model.fit(window.train, epochs=max_epochs, validation_data=window.val, callbacks=[early_stopping])
        return history

    history = compile_and_fit(gain, wide_window, patience=max_epochs // 5)

    if DEBUG:
        figure_loss(history)

    val_performance['Gain'] = gain.evaluate(wide_window.val)
    performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

    gain.save(save_dir='save')
else:
    gain.load(save_dir='save') 

if DEBUG:
    figure_gain(df_list, wide_window, gain)