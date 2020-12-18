import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# font for korean
import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothicCoding-Bold'


target_col = '총유기탄소'

input_step = 24*7
OUT_STEPS = 24*3
# input_step = 2
# OUT_STEPS = 1

MAX_EPOCHS = 100

# mpl.rcParams['figure.figsize'] = (8, 6)
# mpl.rcParams['axes.grid'] = False
# df = pd.read_excel("./data/8/가평_2018-2019.xlsx")
# df = pd.read_excel("./data/8/2.xlsx")
# df = pd.read_csv("./data/8/df4.csv",encoding='utf-8-sig')

df1 = pd.read_excel("./data/9/1.xlsx")
df = pd.read_excel("./data/9/2.xlsx")
# df = pd.read_excel("./data/8/1.xlsx")


# dgen = GainDataGenerator(df1)


# print(dgen)

#
# def comfareDf(df1, df2, fill_cnt):
#         if fill_cnt != 0:
#             mask = df1.copy()
#             for i in df1.columns:
#                 dfx = pd.DataFrame( df1[i] )
#                 dfx['new'] = ((dfx.notnull() != dfx.shift().notnull()).cumsum())
#                 dfx['ones'] = 1
#                 mask[i] = (dfx.groupby('new')['ones'].transform('count') < fill_cnt + 1) | df1[i].notnull()
#             df = df2.bfill()[mask]
#         return df
#
# # print(df1)
# # print(df2)
# df3 = comfareDf(df1, df, 2)
# print(df3)

# slice [start:stop:step], starting from index 5 take every 6th record.
#df1 = df1[5::6]
#df1['측정날짜']
date_time = pd.to_datetime(df.pop('측정날짜'), format='%Y.%m.%d %H:%M:%S', utc=True )
# date_time = pd.to_datetime(df['측정날짜'], format='%Y.%m.%d %H:%M:%S')
# print(date_time)
print(df)

timestamp_s = date_time.map(datetime.datetime.timestamp)

day = 24*60*60
week = day * 7
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
# df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
# df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# plt.plot(np.array(df['Day sin'])[:25])
# plt.plot(np.array(df['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')
# plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}
# print(column_indices)

# print(df)




n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


train_mean = train_df.mean()
train_std = train_df.std()


# print('std, mean')
# print(train_mean)
# print(train_std)
# print('std, mean')
# print('NONONONONONONONONONONONONONONONONO')
# print(test_df)


train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# print(train_std)
# print(train_std['총유기탄소'])

# print(train_df)
# print(val_df)
# print(test_df)
#
# print('DEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDE')
# print(test_df * train_std + train_mean)
# print('DEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDE')




# Data Window
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.

    # print(train_df)


    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # print(type(train_df))

    # print('gen')
    # print(self.test_df[target_col])

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}


    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)

    # print('########################################################################')
    # print(self.input_slice)

    self.input_indices = np.arange(self.total_window_size)[self.input_slice]


    # print('self.input_indices')
    # print(self.input_indices)

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


# w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
#                      label_columns=['총유기탄소'])
# print(w1)
#
# w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
#                      label_columns=['총유기탄소'])
# print(w2)


# Data 분할
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]

  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# Stack three slices, the length of the total window:
# example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
#                            np.array(train_df[100:100+w1.total_window_size]),
#                            np.array(train_df[200:200+w1.total_window_size])])
#

# example_inputs, example_labels = w1.split_window(example_window)

# print('All shapes are: (batch, time, features)')
# print(f'Window shape: {example_window.shape}')
# print(f'Inputs shape: {example_inputs.shape}')
# print(f'labels shape: {example_labels.shape}')

# w1.example = example_inputs, example_labels


def plot(self, model=None, plot_col=target_col, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')
  plt.show()


WindowGenerator.plot = plot
# plot()
# w1.plot()




def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
#
# print(w1.train.take(1))

# for example_inputs, example_labels in w1.train.take(1):
#   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#   print(f'Labels shape (batch, time, features): {example_labels.shape}')


# wide_window = WindowGenerator(
#     input_width=24, label_width=24, shift=1,
#     label_columns=['총유기탄소'])

# wide_window = WindowGenerator(
#     input_width=24, label_width=24, shift=1)

# print(wide_window)

# CONV_WIDTH = 3
# conv_window = WindowGenerator(
#     input_width=CONV_WIDTH,
#     label_width=1,
#     shift=1,
#     label_columns=['총유기탄소'])

# print(conv_window)
val_performance = {}
performance = {}
multi_val_performance = {}
multi_performance = {}


#input_step = 24*7
#OUT_STEPS = 24
multi_window = WindowGenerator(input_width=input_step,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=[target_col])
# multi_window = WindowGenerator(input_width=input_step,
#                               label_width=OUT_STEPS,
#                               shift=OUT_STEPS)

# multi_window.plot()

# print(multi_window.column_indices.get(target_col, None))

# test_input, test_label = multi_window.example


# print('********************************')
# print(multi_window.train_df)
# print('********************************')
# print(multi_window.val_df)
# print('********************************')
# print(multi_window.test_df)
# print('********************************')





def compile_and_fit(model, window, patience=3):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         min_lr=0.0001, patience=5, verbose=1)

  adam = tf.keras.optimizers.Adam(learning_rate=0.1)

  # model.compile(loss=tf.losses.MeanSquaredError(),
  #               optimizer=tf.optimizers.Adam(),
  #               metrics=[tf.metrics.MeanAbsoluteError()])

  model.compile(loss=tf.losses.MeanSquaredError(),
               optimizer=tf.optimizers.Adam(),
               metrics=[tf.metrics.MeanSquaredError()])

  #model.compile(loss=tf.losses.MeanSquaredError(),
  #              optimizer=adam,
  #              metrics=[tf.metrics.MeanAbsoluteError()])

  #history = model.fit(window.train, epochs=MAX_EPOCHS,
  #                    validation_data=window.val,
  #                    callbacks=[early_stopping, reduce_lr])

  #history = model.fit(window.train, epochs=MAX_EPOCHS,
  #                    validation_data=window.val,
  #                    callbacks=[reduce_lr])
  history = model.fit(window.train, epochs=MAX_EPOCHS, batch_size = 256,
                           validation_data=window.val)

  return history




# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ])
# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=num_features)
# ])

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

# print(history.history['loss'])
# print(history.history['val_loss'])
#
# plt.plot(history.history['loss'], label='loss')
# plt.show()
#
#
#
# multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
# multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
# multi_window.plot(multi_lstm_model)


# test_input, test_label = multi_window.example
#
# pred = multi_lstm_model.predict(test_input)
#
# col_num = multi_window.column_indices.get(target_col, None)
#
#
# print('***************1111')
# print(test_label[0,:,col_num])
# print('***************22222')
# print(test_label[:,col_num]*train_std[target_col] + train_mean[target_col])
#
# # # print(pred[0])
# # print(pred[0,:,4])
# #
# #
# # print('***************22222')
#
# # print(WindowGenerator.label_columns_indices.get(target_col, None))
# # predictions[n, :, label_col_index]
#
# # print('***************')
# # tt = pd.DataFrame(pred[0])
# # print(tt)
#
# tt = pd.DataFrame(pred[0,:,col_num])
# print(tt)
#
#
# print('***************33333')
# ttt = tt *train_std[target_col] + train_mean[target_col]
#
# print(ttt)
#
# tt = pred * train_std + train_mean
# print(tt)
# plt.figure()
# plt.plot(pred[0])
# # #print(pred)
# plt.show()

''' Dense
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model)
'''


''' CONV
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

# IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)
'''
