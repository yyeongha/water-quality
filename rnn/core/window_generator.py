import tensorflow as tf
import numpy as np

from core.data_generator import DataGenerator
from core.utils import *
import matplotlib.pyplot as plt


''' font '''
import matplotlib
import matplotlib.font_manager as fm

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                # train_df=train_df, val_df=val_df, test_df=test_df,
                train_df=None, val_df=None, test_df=None, df=None, batch_size=32, fill_no=2,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.df = df
        self.batch_size = batch_size
        self.fill_no = fill_no

        self.label_columns = label_columns
        # Work out the label column indices.
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        self.label_columns_idx = self.column_indices[label_columns]

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


    def plot(self, model=None, plot_col='총유기탄소', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(10, 8))
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

    # WindowGenerator.plot = plot

    @property
    def train(self):
        print('@@@ train')
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        print('@@@ val')
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        print('@@@ test')
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


    # def __len__(self):
    #     'Denotes the number of batches per epoch'
    #     return 1
    #
    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     x = np.empty((0, self.input_width, self.data.shape[1]))
    #     y = np.empty((0, self.label_width, 1))
    #     for cnt in range(0, self.batch_size):
    #         i = self.batch_idx[self.batch_id]
    #         self.batch_id += 1
    #         self.batch_id %= self.no
    #         if (self.batch_id == 0):
    #             self.batch_idx = sample_batch_index(self.no, self.no)
    #
    #         idx1 = self.data_idx[i]
    #         idx2 = self.data_idx[i]+self.input_width
    #         idx3 = idx2 + 1
    #         idx4 = idx3 + self.label_width
    #
    #         X_mb = self.data[idx1:idx2]
    #         # Y_mb = self.data[idx1:idx2]
    #         Y_mb = self.data[idx2:self.idx4, 0:1]
    #         # Y_mb = self.data[idx3:idx4]
    #     x = np.append(x, [X_mb], axis=0)
    #     y = np.append(y, [Y_mb], axis=0)
    #
    #     return x, y
    #
    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     return

    def make_dataset(self, data):
        dg = DataGenerator(
            self.df,
            input_width = self.input_width,
            label_width = self.label_width,
            batch_size = self.batch_size,
            normalize = False,
            fill_no = self.fill_no,
            shift=self.shift,
            target_col_idx = self.label_columns_idx
        )
        self.dg = dg
        # print(dg.shape)
        ds = tf.data.Dataset.from_generator(
            lambda: dg,
            output_types=(tf.float32, tf.float32),
            output_shapes=( dg.shape, [dg.shape[0], self.label_width , 1 ] )
                # dg.shape,
                # dg[dg.shape[0],self.label_width,self.label_columns:self.label_columns+1]

                # [idx2:self.total_window_size, self.target_col_idx:self.target_col_idx]
                # dg.shape
                #[batch_size, train_generator.dim],
                #[batch_size, train_generator.dim],
            # )
        )
        return ds
