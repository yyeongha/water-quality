import tensorflow as tf
import numpy as np

from core.gain_data_generator import GainDataGenerator
from core.utils import *
import matplotlib.pyplot as plt


''' font '''
import matplotlib
import matplotlib.font_manager as fm

# 나눔 폰트 리스트업
avail_font = []
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for font in font_list:
    if font.find('Nanum') != -1:
        avail_font.append(font)

# font list
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf'
# font_location = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
# font_location = 'C:/Users/hackx/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf' # For Windows
fm.get_fontconfig_fonts()
font_location = avail_font[0] # 나눔 폰트 index 0 사용 (리스트 확인 후 변경가능)
fprop = fm.FontProperties(fname=font_location)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                # train_df=train_df, val_df=val_df, test_df=test_df,
                train_df=None, val_df=None, test_df=None, df=None,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.df = df

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




    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(10, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
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

    def make_dataset(self, data):
        dg = GainDataGenerator(
            self.df,
            input_width = self.input_width,
            label_width = self.label_width,
            batch_size = 128,
            normalize = False,
            miss_pattern = True,
            miss_rate = 0.2,
            fill_no = 2,
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