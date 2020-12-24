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


folder = 'data'
file_names = [['가평_2019.xlsx']]

df, df_full, df_all = createDataFrame(folder, file_names)

standardNormalization(df, df_all)

dgen = GainDataGenerator(df)

train_df = df[0]
val_df = df[0]
test_df = df[0]

wide_window = WindowGenerator(
    df=df,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=24 * 5,
    label_width=24 * 5,
    shift=0
)
wide_window.plot(plot_col='총질소')  # create dg issue

val_performance = {}
performance = {}

gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)
MAX_EPOCHS = 1


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


history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS // 5)

val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)
gain.save(save_dir='save')

gain.evaluate(wide_window.test.repeat(), steps=100)
wide_window.plot(gain, plot_col='클로로필-a')

cnt = 0
for i in df:
    data = i.to_numpy()
    print('data--------------', data.shape)
    # total_n = wide_window.dg.data.shape[0]
    total_n = len(data)
    print(type(total_n))
    print('total_n', total_n)
    unit_shape = wide_window.dg.shape[1:]
    print('unit_shape', unit_shape)
    dim = wide_window.dg.shape[1]
    print('dim', dim)
    n = (total_n // dim) * dim
    print('n', n)

    x = data[0:n].copy()
    y_true = data[0:n].copy()
    x_reshape = x.reshape((-1,) + unit_shape)
    isnan = np.isnan(x_reshape)
    isnan = np.isnan(y_true)

    x_remain = data[-wide_window.dg.shape[1]:].copy()
    x_remain_reshape = x_remain.reshape((-1,) + unit_shape)
    x_remain_reshape.shape

    # zero loss is normal because there is no ground truth in the real dataset
    gain.evaluate(x_reshape, y_true.reshape((-1,) + unit_shape))

    y_pred = gain.predict(x_reshape)
    y_remain_pred = gain.predict(x_remain_reshape)

    y_pred = y_pred.reshape(y_true.shape)
    if y_pred.shape[0] != 8760:
        y_remain_pred = y_remain_pred.reshape(x_remain.shape)
        y_pred = np.append(y_pred, y_remain_pred[-(total_n - n):], axis=0)

    # Denormalized
    train_mean = df_all.mean()
    train_std = df_all.std()

    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = df_all.columns
    result = y_pred * train_std + train_mean

    result.pop("Day sin")
    result.pop("Day cos")
    result.pop("Year sin")
    result.pop("Year cos")

    df_date = pd.DataFrame(df_full[0]['측정날짜'][:len(result.index)])
    # df_location = pd.DataFrame(df_full[0]['측정소명'][:len(result.index)])
    result = pd.concat([df_date, result], axis=1)
    result.to_excel('./data/' + file_names[cnt][0][:8] + '_' + str(MAX_EPOCHS) + '_result.xlsx',
                    index=False)
    cnt += 1
