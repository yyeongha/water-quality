import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from core.util import *

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               #train_df=train_df, val_df=val_df, test_df=test_df,
            train_df=None, val_df=None, test_df=None, df=None, out_num_features=0,out_features=0,
#                out_features = None,
               label_columns=None):
    # Store the raw data.
        #print("insert train_df = ", train_df.shape)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.df = df

    #
        self.out_num_features = out_num_features
        self.out_features = out_features


    # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

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


        self.example[0] # create self.dg

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



class MissData(object):
    def __init__(self, load_dir=None):
        if load_dir:
            self.missarr = np.load(os.path.join(load_dir, 'miss.npy'))
            self.idxarr = np.load(os.path.join(load_dir, 'idx.npy'))

    def make_missdata(self, data_x, missrate=0.2):
        data = data_x.copy()
        rows, cols = data_x.shape
        total_no = rows * cols
        total_miss_no = np.round(total_no * missrate).astype(int)
        total_idx = self.idxarr.shape[0]
        idxarr = self.idxarr
        missarr = self.missarr
        # print(total_miss_no)
        miss_no = 0
        cum_no = self.idxarr[:, 3:4]
        cum_no = cum_no.reshape((total_idx))
        cum_sum = np.max(cum_no)
        # print(cum_no)
        # print(total_idx)
        while True:
            loc_count = np.around(np.random.random() * cum_sum)
            # print('loc_count =', loc_count)
            idx = len(cum_no[cum_no <= loc_count]) - 1
            # print(cum_no[cum_no <= loc_count])
            # print('idx =', idx)
            startnan = idxarr[idx][0]
            nanlen = idxarr[idx][2]
            loc = np.around(np.random.random() * (rows - nanlen)).astype(int)
            # print('loc =', loc)
            # print(loc_count, idx)
            # print(idxarr[idx])
            # data_copy = data[loc:loc+nanlen].copy()
            data_copy = data[loc:loc + nanlen]
            # print('startnan=', startnan)
            # isnan = missarr[startnan:startnan+nanlen].copy()
            isnan = missarr[startnan:startnan + nanlen]
            # print('isnan =',isnan)
            miss_no += idxarr[idx][1]
            if (miss_no > total_miss_no):
                break
            data_copy[isnan == 1] = np.nan
            data[loc:loc + nanlen] = data_copy
        # print('miss_data =', data)
        return data

    def save(data, max_tseq, save_dir='save'):
        no, dim = data.shape
        # print((no, dim))
        isnan = np.isnan(data).astype(int)
        isany = np.any(isnan, axis=1).astype(int)
        shifted = np.roll(isany, 1)
        shifted[0] = 1
        # print(isnan)
        # print(isany.astype(int))
        # print(shifted)
        startnan = ((isany == 1) & (shifted == 0)).astype(int)
        # print(startnan)
        group = startnan.cumsum()
        group = group * isany
        #         print(group)
        n = np.max(group)
        #         print(n)
        missarr = None
        cum_no = 0
        rowidx = 0
        for i in range(1, n + 1):
            g = (group == i).astype(int)
            i = np.argmax(g)
            rows = g.sum()
            # print(len)
            # print(i)
            # print(type(missarr))
            if rows <= max_tseq:
                nanseq = isnan[i:i + rows, :]
                no = np.sum(nanseq)
                # print(no)
                if missarr is None:
                    missarr = nanseq
                    idxarr = np.array([[rowidx, no, rows, cum_no]])
                else:
                    missarr = np.concatenate((missarr, nanseq))
                    idxarr = np.concatenate((idxarr, [[rowidx, no, rows, cum_no]]), axis=0)
                cum_no += no
                rowidx += rows

        #print(idxarr)
        miss_npy_file = os.path.join(save_dir, 'miss.npy')
        idx_npy_file = os.path.join(save_dir, 'idx.npy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(miss_npy_file, missarr)
        np.save(idx_npy_file, idxarr)
        #print('miss_data file saved')



def interpolate(np_data, max_gap=3):
    # n = np_data.shape[1]
    data = pd.DataFrame(np_data)
    # data[0][0] = np.nan
    # data[0][1] = np.nan
    # data[0][2] = np.nan
    # data[data.columns[0]][0] = np.nan
    # data[data.columns[0]][1] = np.nan
    # data[data.columns[0]][2] = np.nan

    # create mask
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in data.columns:
        mask[i] = (grp.groupby(i)['ones'].transform('count') < max_gap) | data[i].notnull()
    data = data.interpolate(method='polynomial', order=5, limit=max_gap, axis=0).bfill()[mask]
    return data.to_numpy()
    # return data



class GainDataGenerator(tf.keras.utils.Sequence):
    'Generates data for GAIN'

    def __init__(self,
                 data_list,
                 batch_size=32,
                 input_width=24 * 3,
                 label_width=24 * 3,
                 shift=0,
                 fill_no=4,
                 miss_rate=0.2,
                 hint_rate=0.9,
                 normalize=True,
                 miss_pattern=None,
                 miss_data_load_dir='save',
                 alpha=100.):
        'Initialization'
        window_size = input_width

        # interpollation
        filled_data = []
        for data in data_list:
            data = interpolate(data, max_gap=fill_no)
            filled_data.append(data)

        data_list = filled_data

        # whole data
        self.data = np.concatenate(data_list)

        # TO-DO

        # pre calculation for  sequence data
        last_cum = 0
        cums = []
        for data in data_list:
            isnan = np.isnan(data)
            isany = np.any(isnan, axis=1)
            # shift same as pd.shift(isany, fill_value=True)
            shifted = np.roll(isany, 1)
            shifted[0] = True  # set to nan

            start_seq = ((isany == False) & (shifted == True)).astype(int)
            cum = start_seq.cumsum()
            cum += last_cum
            last_cum = np.max(cum)
            cum[isany] = 0
            cums.append(cum)

        # Define mask matrix
        if miss_pattern is None:
            print("pattern none")
            self.data_m = binary_sampler(1 - miss_rate, self.data.shape)
        else:
            # MissData.save(self.data, max_tseq = 12)
            print("load save")
            self.miss = MissData(load_dir=miss_data_load_dir)
            self.miss_rate = miss_rate
            miss_data = self.miss.make_missdata(self.data, self.miss_rate)
            self.data_m = 1. - np.isnan(miss_data).astype(float)

            self.data_m_rand = binary_sampler(1 - (miss_rate / 10.), self.data.shape)
            self.data_m[self.data_m_rand == 0.] = 0.
        self.miss_pattern = miss_pattern

        # sequence data
        self.ids = np.concatenate(cums)
        data_idx = np.empty((0), dtype=int)
        for i in range(1, last_cum + 1):
            seq_len = (self.ids == i).sum()
            start_id = np.argmax(self.ids == i)
            # possible data number in seqeunce
            time_len = seq_len - window_size + 1
            start_ids = np.arange(start_id, start_id + time_len)
            data_idx = np.append(data_idx, start_ids)

        # start index set for sequence data
        self.data_idx = data_idx
        self.input_width = input_width
        self.no = len(data_idx)

        # print('self.no = ', self.no)

        self.batch_size = batch_size

        # random shuffling  index
        self.batch_idx = sample_batch_index(self.no, self.no)
        self.batch_id = 0
        self.shape = (batch_size, self.input_width) + self.data.shape[1:]
        # self.hint_rate = hint_rate

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(128/self.batch_size)
        # return 2
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # print('index =', index)
        # Sample batch
        x = np.empty((0, self.input_width, self.data.shape[1]))
        # m = np.empty((0, self.input_width, self.data.shape[1]))
        # h = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.input_width, self.data.shape[1]))
        # print(x.shape)
        # print(self.data.shape)
        # print(self.input_width)
        # self.batch_idx = sample_batch_index(self.no, self.batch_size)
        for cnt in range(0, self.batch_size):
            i = self.batch_idx[self.batch_id]
            self.batch_id += 1
            # self.batch_id %= self.batch_size
            self.batch_id %= self.no
            if self.miss_pattern and (self.batch_id == 0):
                self.batch_idx = sample_batch_index(self.no, self.no)
                miss_data = self.miss.make_missdata(self.data, self.miss_rate)
                self.data_m = 1. - np.isnan(miss_data).astype(float)
                self.data_m_rand = binary_sampler(1 - self.miss_rate / 10., self.data.shape)
                self.data_m[self.data_m_rand == 0.] = 0.
            idx1 = self.data_idx[i]
            idx2 = self.data_idx[i] + self.input_width
            # print(idx1, idx2)

            Y_mb = self.data[idx1:idx2].copy()
            X_mb = Y_mb.copy()
            M_mb = self.data_m[idx1:idx2]
            Z_mb = uniform_sampler(0, 0.01, shape=X_mb.shape)
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            # H_mb_temp = binary_sampler(self.hint_rate, shape=X_mb.shape)
            # H_mb = M_mb * H_mb_temp
            X_mb[M_mb == 0] = np.nan
            Y_mb[M_mb == 1] = np.nan
            x = np.append(x, [X_mb], axis=0)
            # m = np.append(m, [M_mb], axis=0)
            # h = np.append(h, [H_mb], axis=0)
            y = np.append(y, [Y_mb], axis=0)

        # return [x, m, h], y
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


def make_dataset_gain(self, data):
    dg = GainDataGenerator(
        self.df,
        input_width = self.input_width,
        label_width = self.label_width,
        batch_size = 128,
        normalize = False,
        miss_pattern = True,
        miss_rate = 0.15,
        fill_no = 3,
    )
    self.dg = dg

    ds = tf.data.Dataset.from_generator(
        lambda: dg,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            dg.shape,
            dg.shape
            #[batch_size, train_generator.dim],
            #[batch_size, train_generator.dim],
        )
    )
    return ds

#WindowGenerator.make_dataset = make_dataset_gain




##------------------- water data gn
class WaterDataGenerator(tf.keras.utils.Sequence):
    'Generates data for water'

    def __init__(self,
                 imputed_data,
                 ori_data=None,
                 batch_size=32,
                 input_width=24 * 7,
                 label_width=24 * 3,
                 shift=24 * 3,
                 skip_time=None,
                 shuffle=True,
                 out_features=None,
                 out_num_features=None,
                 ):
        'Initialization'
        self.window_size = input_width + shift
        self.total_no = imputed_data.shape[0]
        self.data = imputed_data
        self.input_width = input_width
        self.label_width = label_width
        self.batch_size = batch_size
        self.input_shape = (batch_size, input_width, self.data.shape[1])
        self.out_num_features = out_num_features
        #         print("out_features")
        #         print(out_features)
        if out_features:
            self.out_features = out_features
        else:
            self.out_features = [i for i in range(out_num_features)]
        self.label_shape = (batch_size, label_width, self.out_num_features)


        if (skip_time):
            # TO-DO
            self.no = self.total_no - self.window_size
            self.data_idx = np.arange(0, self.no)
        else:
            self.no = self.total_no - self.window_size
            self.data_idx = np.arange(0, self.no)

        if shuffle:
            self.batch_idx = np.random.permutation(self.no)
        else:
            self.batch_idx = np.arange(0, self.no)
        self.batch_id = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(128/self.batch_size)
        # return 2
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # print('index =', index)
        # print('self.no =', self.no)
        # print('self.total_no =', self.total_no)
        # print('self.batch_id =', self.batch_id)
        # Sample batch
        label_width = self.label_width
        batch_idx = self.batch_idx

        x = np.empty((0, self.input_width, self.data.shape[1]))

        # y = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.label_width, self.out_num_features))

        # print(x.shape)
        # print(y.shape)

        for cnt in range(0, self.batch_size):
            i = self.batch_id
            self.batch_id += 1
            idx1 = self.data_idx[batch_idx[i]]
            idx2 = idx1 + self.input_width

            X = self.data[idx1:idx2].to_numpy()

            idx1 = self.data_idx[batch_idx[i]] + self.window_size - label_width
            idx2 = idx1 + label_width

            # Y = self.data[idx1:idx2,:,:out_num_features]
            Y = self.data.iloc[idx1:idx2, self.out_features].to_numpy()
            # Y = self.data[idx1:idx2]
            # print('Y.shape = ', Y.shape)
            # Y = Y.iloc[:,:out_num_features]

            self.batch_id %= self.no
            # print("x.shape=", x.shape)
            # print('X.shape=', X.shape)
            # print(type(x), type(X))
            x = np.append(x, [X], axis=0)

            y = np.append(y, [Y], axis=0)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


def make_dataset_water(self, data):
#   print("make_dataset_water")
#   print(out_features)

    #print("make_dataset_water")
    #print(data)

    #print("make_datset_water : ", self.out_num_features)

    dg = WaterDataGenerator(
        data,
        #self.train_df,
        batch_size = 128,
        input_width = self.input_width,
        label_width = self.label_width,
        shift = self.label_width,
        #out_features = out_features,
        out_features = self.out_features,
        #out_num_features = out_num_features,
        #out_num_features = g_out_num_features,
        out_num_features = self.out_num_features,
    )

    #print('input_shape', 'label_shape')
    #print(dg.input_shape, dg.label_shape)


  #self.dg = dg
    ds = tf.data.Dataset.from_generator(
        lambda: dg,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            dg.input_shape,
            dg.label_shape
            #[batch_size, train_generator.dim],
            #[batch_size, train_generator.dim],
        )
    )

    return ds

#WindowGenerator.make_dataset = make_dataset_water


class GainWindowGenerator(WindowGenerator):
    def make_dataset(self, data):
        return make_dataset_gain(self, data)

class WaterWindowGenerator(WindowGenerator):
    def make_dataset(self, data):
        return make_dataset_water(self, data)





#--------------------------------