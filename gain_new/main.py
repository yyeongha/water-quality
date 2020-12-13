import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *


folder = 'data'
file_names = ['가평_2019.xlsx', '의암호_2019.xlsx']

df, df_full, df_all = createDataFrame(folder, file_names)

standardNormalization(df, df_all)

dgen = GainDataGenerator(df)

train_df = df_all
val_df = df_all
test_df = df_all

wide_window = WindowGenerator(
    df=df,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=24*3, 
    label_width=24*3, 
    shift=0
)
wide_window.plot(plot_col='총질소') # create dg issue

''' miss data plt '''
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
    model.compile(loss=GAIN.RMSE_loss)

    history = model.fit(window.train, 
                        epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS//5)

val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

''' loss plt '''
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

''' 학습데이터로 테스트 '''
total_n = wide_window.dg.data.shape[0]
unit_shape = wide_window.dg.shape[1:]
dim = np.prod(wide_window.dg.shape[1:]).astype(int)
n = (total_n//dim)*dim
x = wide_window.dg.data[0:n].copy()
y = wide_window.dg.data[0:n].copy()
m = wide_window.dg.data_m[0:n]
x[m == 0] = np.nan
x = x.reshape((-1,)+unit_shape)
y_true = y.reshape((-1,)+unit_shape)
y_pred = gain.predict(x)
y_pred = y_pred.reshape((n, 13))
x = x.reshape((n, 13))

''' 원본데이터로 테스트 '''
norm_df = pd.concat(df,axis=0)
data = norm_df.to_numpy()
x = data[0:n].copy()
y_true = data[0:n].copy()
isnan = np.isnan(x)
x[isnan] = np.nan
total_n = wide_window.dg.data.shape[0]
unit_shape = wide_window.dg.shape[1:]
dim = np.prod(wide_window.dg.shape[1:]).astype(int)
n = (total_n//dim)*dim
x_reshape = x.reshape((-1,)+unit_shape)
y_pred = gain.predict(x_reshape)
y_pred = y_pred.reshape(y_true.shape)

''' result plt '''
n = 8
plt.figure(figsize=(9,20))
for i in range(n):
    plt.subplot(811+i)
    plt.plot(x[:, i])
    plt.plot(y_pred[:, i])
plt.show()