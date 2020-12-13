import pandas as pd
import numpy as np
from glob import glob
import os
import datetime
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *



''' main '''
folder = 'data'
file_names = ['가평_2019.xlsx', '의암호_2019.xlsx']

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

# normalize data

df_all = pd.concat(df)
df_all

train_mean = df_all.mean()
train_std = df_all.std()
for i in range(len(file_names)):
    df[i] = (df[i]-train_mean)/train_std




dgen = GainDataGenerator(df)

train_df = df_all
val_df = df_all
test_df = df_all

wide_window = WindowGenerator(
    df=df,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=24*3, label_width=24*3, shift=0,
    #label_columns=['T (degC)']
)

wide_window.plot(plot_col='총질소')

# plt.figure(figsize=(9,10))
# isnan = np.isnan(norm_data).astype(int)
# data = isnan
# n = data.shape[0]
# seq_len = n//8
# for i in range(8):
#     plt.subplot(181+i)
#     plt.imshow(data[i*seq_len:(i+1)*seq_len, 0:7], aspect='auto')
#     plt.yticks([])
# plt.show()

# plt.figure(figsize=(9,10))
# n = wide_window.dg.data_m.shape[0]
# train = n//8
# for i in range(8):
#     plt.subplot(181+i)
#     plt.imshow(wide_window.dg.data_m[i*train:(i+1)*train, 0:7], aspect='auto')
#     plt.yticks([])
# plt.show()

val_performance = {}
performance = {}

gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)

MAX_EPOCHS = 50

def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

    #model.compile(loss=tf.losses.MeanSquaredError(),
                    #optimizer=tf.optimizers.Adam(),
                    #metrics=[tf.metrics.MeanAbsoluteError()])
    model.compile(loss=GAIN.RMSE_loss)

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS//5)

val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax2 = ax.twinx()
# ax.plot(history.history['gen_loss'], label='gen_loss')
# ax.plot(history.history['disc_loss'], label='disc_loss')
# ax2.plot(history.history['rmse'], label='rmse', color='green')
# ax2.plot(history.history['val_loss'], label='val_loss', color='red')
# ax.legend(loc='upper center')
# ax2.legend(loc='upper right')
# ax.set_xlabel("epochs")
# ax.set_ylabel("loss")
# ax2.set_ylabel("rmse")
# plt.show()

gain.evaluate(wide_window.test.repeat(), steps=100)

total_n = wide_window.dg.data.shape[0]
print(total_n)
unit_shape = wide_window.dg.shape[1:]
print(unit_shape)
dim = np.prod(wide_window.dg.shape[1:]).astype(int)
print(dim)
n = (total_n//dim)*dim
print(n)
x = wide_window.dg.data[0:n].copy()
y = wide_window.dg.data[0:n].copy()
m = wide_window.dg.data_m[0:n]
x[m == 0] = np.nan
print('x.shape =', x.shape)
x = x.reshape((-1,)+unit_shape)
y_true = y.reshape((-1,)+unit_shape)
print('x.shape =', x.shape)

y_pred = gain.predict(x)
y_pred = y_pred.reshape((n, 13))
x = x.reshape((n, 13))
# plt.figure()
# plt.plot(x[:, 8])
# plt.plot(y_pred[:, 8])
# plt.show()

norm_df = pd.concat(df,axis=0)

data = norm_df.to_numpy()
x = data[0:n].copy()
y_true = data[0:n].copy()
isnan = np.isnan(x)
x[isnan] = np.nan

total_n = wide_window.dg.data.shape[0]
print(total_n)
unit_shape = wide_window.dg.shape[1:]
print(unit_shape)
dim = np.prod(wide_window.dg.shape[1:]).astype(int)
print(dim)
n = (total_n//dim)*dim

print('x.shape =', x.shape)
x_reshape = x.reshape((-1,)+unit_shape)
print('x_reshape.shape =', x_reshape.shape)

y_pred = gain.predict(x_reshape)
y_pred = y_pred.reshape(y_true.shape)

n = 8
plt.figure(figsize=(9,20))
for i in range(n):
    plt.subplot(811+i)
    plt.plot(x[:, i])
    plt.plot(y_pred[:, i])
plt.show()