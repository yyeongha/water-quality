# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from core.util import *

from core.rnn_data_generator import WaterDataGenerator
from core.gain_data_generator import GainDataGenerator

# for Korean in plot
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location = '/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf'
#font_location = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
fprop = fm.FontProperties(fname=font_location)


class WindowGenerator():
    def __init__(self, input_width=24*5, label_width=24*5, shift=24*5,
               #train_df=train_df, val_df=val_df, test_df=test_df,
            train_df=None, val_df=None, test_df=None, df=None, out_num_features=0,out_features=0, #model_save_path=None,
                 test_df2=None, miss_rate=0.2, fill_no=3, batch_size = 32, save_dir = 'save/',
#                out_features = None,
               label_columns=None):
    # Store the raw data.
        #print("insert train_df = ", train_df.shape)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.test_df2 = test_df2
        self.df = df

        self.miss_rate = miss_rate
        self.fill_no = fill_no
        self.batch_size = batch_size

        #print('model_save_path : ', model_save_path)
        #self.model_save_path = model_save_path
    #
        self.out_num_features = out_num_features
        self.out_features = out_features

        self.train_std = None
        self.train_mean = None
        self.save_dir = save_dir
    # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

            #print('-------------------------------------------')
            #print(self.label_columns_indices)

            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

            #print(self.column_indices)

    # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        #self.example # create self.dg
       # print("++++++++++++++++++++++++++++++++++++")
        #print(self.example[0].shape)

       # print("++++++++++++++++++++++++++++++++++++")
        #print(self.example[0].shape)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

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

#WindowGenerator.split_window = split_window

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(10, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]', fontproperties=fprop)
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

#WindowGenerator.plot = plot

#    @property
#    def train(self):
#        return self.make_dataset(self.train_df)

    @property
    def train(self):
        return self.make_dataset(self.train_df, train=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def test2(self):
        return self.make_dataset(self.test_df2)

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

    def example2(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example2', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.val))
            # And cache it for next time
            self._example = result
        return result

    @property
    def example3(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example3', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result

    @property
    def example_test(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example_test', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test2))
            # And cache it for next time
            self._example = result
        return result

    def hour_to_day_mean(self, array):
        time = 24
        return tf.reduce_mean(tf.reshape(array, [array.shape[0] // time, time]), 1)

    def plot24(self, model=None, plot_col=0, max_subplots=3, plot_out_col=0):
        inputs, labels = self.example
        plt.figure(figsize=(10, 8))
        #plot_col_index = self.column_indices[plot_col]
        #plot_out_col_index = self.column_indices[plot_out_col]
        plot_col_index = 0
        plot_out_col_index = 0
        max_n = min(max_subplots, len(inputs))

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
            label_out_col_index = self.label_columns_indices.get(plot_out_col, None)
        else:
            label_col_index = plot_col_index
            label_out_col_index = plot_out_col_index

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            #plt.ylabel(f'{self.df_all.columns[plot_col]} [normed]', fontproperties=fprop)
#            plt.ylabel(f'{self.df_all.columns[plot_col]} [normed]')

            input_temp = self.hour_to_day_mean(inputs[n, :, plot_col_index])
            input_temp = input_temp * self.train_std[plot_col] + self.train_mean[plot_col]

            plt.plot(
                self.hour_to_day_mean(self.input_indices),
                input_temp,
                label='Inputs', marker='.', zorder=-10)

            if label_col_index is None:
                continue

            label_temp = self.hour_to_day_mean(labels[n, :, label_out_col_index])
            label_temp = label_temp * self.train_std[plot_col] + self.train_mean[plot_col]

            plt.plot(
                self.hour_to_day_mean(self.label_indices),
                label_temp,
                label='Labels', marker='.', zorder=-10, c='#2ca02c')

            if model is not None:
                predictions = model(inputs)

                # predictions = predictions * train_std[plot_col] * train_mean[plot_col]
                predict_temp = self.hour_to_day_mean(predictions[n, :, label_out_col_index])
                predict_temp = predict_temp * self.train_std[plot_col] + self.train_mean[plot_col]

                plt.plot(
                    self.hour_to_day_mean(self.label_indices),
                    predict_temp,
                    label='Predictions', marker='.', zorder=-10, c='#ff7f0e')

            if n == 0:
                plt.legend()

        plt.xlabel('Time [day]')
        plt.show()



    def compa(self, model=None, plot_col=0, windows=None, target_std=None, target_mean=None, predict_day=4):
        if windows is not None:
            inputs, labels = windows
        else:
            inputs, labels = self.example

        if model is None:
            return

        pred_arr = []
        label_arr = []
        mae = mse = rmse = mape = 0

        o = o1 = p = 0
        nse_sum1 = nse_sum2 = 0
        pbias_sum1 = pbias_sum2 = 0

        predictions = model(inputs)

        predictions = predictions * target_std[plot_col] + target_mean[plot_col]
        labels = labels * target_std[plot_col] + target_mean[plot_col]

        o1 = np.mean(labels.numpy())

        for n in range(len(inputs)):
            pred_temp = self.hour_to_day_mean(predictions[n, :, :])
            label_temp = self.hour_to_day_mean(labels[n, :, :])

            o = label_temp[predict_day].numpy()
            p = pred_temp[predict_day].numpy()

            temp_m = o - p

            mae += np.abs(temp_m)
            mse += temp_m ** 2

            nse_sum1 += temp_m ** 2
            nse_sum2 += (o - o1) ** 2

            pbias_sum1 += temp_m
            pbias_sum2 += o

        nse = 1 - (nse_sum1 / nse_sum2)
        pbias = (pbias_sum1 / pbias_sum2) * 100

        mae /= len(inputs)
        mse /= len(inputs)

        rmse = np.sqrt(mse)

        return nse, np.abs(pbias), pred_temp.numpy(), label_temp.numpy()




#class GainWindowGenerator(WindowGenerator):
def make_dataset_gain(self, data, train=False):
    dg = GainDataGenerator(
        self.df,
        #data,
        input_width=self.input_width,
        label_width=self.label_width,
        batch_size=self.batch_size,
        normalize=False,
        miss_pattern=True,
        miss_rate=self.miss_rate,
        fill_no=self.fill_no,
        model_save_path = self.save_dir
    )
    self.dg = dg
    #print('asdfasdfasdfasfd')

    ds = tf.data.Dataset.from_generator(
        lambda: dg,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            dg.shape,
            dg.shape
            # [batch_size, train_generator.dim],
            # [batch_size, train_generator.dim],
        )
    )
    return ds

WindowGenerator.make_dataset = make_dataset_gain


#class WaterWindowGenerator(WindowGenerator):
def make_dataset_water(self, data, train=False):

    dg = WaterDataGenerator(
        data,
        # self.train_df,
        batch_size=self.batch_size,
        input_width=self.input_width,
        label_width=self.label_width,
        shift=self.label_width,
        # out_features = out_features,
        out_features=self.out_features,
        # out_num_features = out_num_features,
        # out_num_features = g_out_num_features,
        out_num_features=self.out_num_features,
    )

    #print('self.input_width, self.label_width11')
    #print(self.input_width, self.label_width)

    # print('input_shape', 'label_shape')
    # print(dg.input_shape, dg.label_shape)

    # self.dg = dg
    ds = tf.data.Dataset.from_generator(
        lambda: dg,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            dg.input_shape,
            dg.label_shape
            # [batch_size, train_generator.dim],
            # [batch_size, train_generator.dim],
        )
    )

    #print('self.input_width, self.label_width')
    #print(self.input_width, self.label_width)
    #print('ds-----------------------------')
    #print(ds)
    if train:
        # return ds.repeat(10).prefetch(3)
        return ds.repeat(-1).prefetch(5)
    else:
        return ds.prefetch(5)

    #return ds



