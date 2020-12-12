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


folder = 'data'
file_names = ['가평_2019.xlsx', '의암호_2019.xlsx']

day = 24*60*60
year = (365.2425)*day

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

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                # train_df=train_df, val_df=val_df, test_df=test_df,
                train_df=None, val_df=None, test_df=None,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

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

WindowGenerator.split_window = split_window

import matplotlib
import matplotlib.font_manager as fm

# font list
fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf'
#font_location = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
font_location = 'C:/Users/hackx/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf' # For Windows
fprop = fm.FontProperties(fname=font_location)

''' 
폰트 한글 적용 테스트
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for font in font_list:
    if font.find('Nanum') != -1:
        print(font)

plt.ylabel('한글', fontproperties=fprop)
plt.show()
exit(0)
'''

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

WindowGenerator.plot = plot

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

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
        - total: total number of samples
        - batch_size: batch size

    Returns:
        - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

def binary_sampler(p, shape):
    '''Sample binary random variables.
    
    Args:
        - p: probability of 1
        - shape: matrix shape
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size = shape)
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

def uniform_sampler(low, high, shape):
    '''Sample uniform random variables.
    
    Args:
        - low: low limit
        - high: high limit
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size = shape)

def normalization (data, parameters=None):
    '''Normalize data in [0, 1] range.
    
    Args:
        - data: original data
    
    Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                        'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters

class MissData(object):
    def __init__(self, load_dir=None):
        if load_dir:
            self.missarr = np.load(os.path.join(load_dir, 'miss.npy'))
            self.idxarr = np.load(os.path.join(load_dir, 'idx.npy'))
            
    def make_missdata(self, data_x, missrate=0.2):
        data = data_x.copy()
        rows, cols = data_x.shape
        total_no = rows*cols
        total_miss_no = np.round(total_no*missrate).astype(int)
        total_idx = self.idxarr.shape[0]
        idxarr = self.idxarr
        missarr = self.missarr
        #print(total_miss_no)
        miss_no = 0
        cum_no = self.idxarr[:,3:4]
        cum_no = cum_no.reshape((total_idx))
        cum_sum = np.max(cum_no)
        #print(cum_no)
        #print(total_idx)
        while True:
            loc_count = np.around(np.random.random()*cum_sum)
            #print('loc_count =', loc_count)
            idx = len(cum_no[cum_no <= loc_count])-1
            #print(cum_no[cum_no <= loc_count])
            #print('idx =', idx)
            startnan = idxarr[idx][0]
            nanlen = idxarr[idx][2]
            loc = np.around(np.random.random()*(rows-nanlen)).astype(int)
            #print('loc =', loc)
            #print(loc_count, idx)
            #print(idxarr[idx])
            #data_copy = data[loc:loc+nanlen].copy()
            data_copy = data[loc:loc+nanlen]
            #print('startnan=', startnan)
            #isnan = missarr[startnan:startnan+nanlen].copy()
            isnan = missarr[startnan:startnan+nanlen]
            #print('isnan =',isnan)
            miss_no += idxarr[idx][1]
            if (miss_no > total_miss_no):
                break
            data_copy[isnan==1] = np.nan
            data[loc:loc+nanlen] = data_copy
        #print('miss_data =', data)
        return data
    
    def save(data, max_tseq, save_dir='save'):
        no, dim = data.shape
        #print((no, dim))
        isnan = np.isnan(data).astype(int)
        isany = np.any(isnan, axis=1).astype(int)
        shifted = np.roll(isany, 1)
        shifted[0] = 1
        #print(isnan)
        #print(isany.astype(int))
        #print(shifted)
        startnan = ((isany == 1) & (shifted ==0)).astype(int)
        #print(startnan)
        group = startnan.cumsum()
        group = group*isany
        #print(group)
        n = np.max(group)
        #print(n)
        missarr = None
        cum_no = 0
        rowidx = 0
        for i in range(1, n+1):
            g = (group == i).astype(int)
            i = np.argmax(g)
            rows = g.sum()
            #print(len)
            #print(i)
            #print(type(missarr))
            if rows <= max_tseq:
                nanseq = isnan[i:i+rows, :]
                no = np.sum(nanseq)
                #print(no)
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
        print('miss_data file saved')

norm_df = pd.concat(df,axis=0)
norm_data = norm_df.to_numpy()
MissData.save(norm_data, max_tseq = 12)

def interpolate(np_data, max_gap=3):
    data = pd.DataFrame(np_data)

    # create mask
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in data.columns:
        mask[i] = (grp.groupby(i)['ones'].transform('count') < max_gap) | data[i].notnull()
    data = data.interpolate(method='polynomial', order=5, limit=max_gap, axis=0).bfill()[mask]
    return data

from tensorflow import keras

class GainDataGenerator(keras.utils.Sequence):
    'Generates data for GAIN'
    def __init__(self,
                 data_list,
                 batch_size=32,
                 input_width=24*3,
                 label_width=24*3,
                 shift=0,
                 fill_no=2,
                 miss_rate=0.2,
                 hint_rate=0.9,
                 normalize=True,
                 miss_pattern=None,
                 alpha=100.):
        'Initialization'
        window_size = input_width
        
        # print('data_list => ', data_list)
        # interpollation
        filled_data = []
        for data in data_list:
            # print('loop data => ', data)
            # print('fill_no => ', fill_no)
            data = interpolate(data, max_gap=fill_no)
            # print('interpolate data => ', data)
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
            shifted = np.roll(isany, 1)
            shifted[0] = True # set to nan
            start_seq = ((isany == False) & (shifted == True)).astype(int)
            cum = start_seq.cumsum()
            cum += last_cum
            last_cum = np.max(cum)
            cum[isany == 1] = np.nan
            cums.append(cum)
            
        
        # normlize for spam
        if normalize:
            self.data, norm_param = normalization(self.data)
        #print(norm_param)
        
        # Define mask matrix
        if miss_pattern is None:
            self.data_m = binary_sampler(1-miss_rate, self.data.shape)
        else:
            #MissData.save(self.data, max_tseq = 12)
            self.miss = MissData(load_dir='save')
            self.miss_rate = miss_rate
            miss_data = self.miss.make_missdata(self.data, self.miss_rate)
            self.data_m = 1. - np.isnan(miss_data).astype(float)
        
        # sequence data
        self.ids = np.concatenate(cums)
        data_idx = np.empty((0), dtype=int)
        for i in range(1, last_cum+1):
            seq_len = (self.ids == i).sum()
            start_id = np.argmax(self.ids == i)
            time_len = seq_len - window_size + 1
            start_ids = np.arange(start_id, start_id+time_len)
            data_idx = np.append(data_idx, start_ids)
            
        # start index set for sequence data
        self.data_idx = data_idx
        self.input_width = input_width
        self.no = len(data_idx)
        
        #print('self.no = ', self.no)
        
        self.batch_size = batch_size
        
        # random shuffling  index
        self.batch_idx = sample_batch_index(self.no, self.no)
        self.batch_id = 0
        self.shape = (batch_size,self.input_width)+self.data.shape[1:]
        #self.hint_rate = hint_rate
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(128/self.batch_size)
        #return 2
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        #print('index =', index)
        # Sample batch
        x = np.empty((0, self.input_width, self.data.shape[1]))
        #m = np.empty((0, self.input_width, self.data.shape[1]))
        #h = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.input_width, self.data.shape[1]))
        #print(x.shape)
        #print(self.data.shape)
        #print(self.input_width)
        #self.batch_idx = sample_batch_index(self.no, self.batch_size)
        for cnt in range(0, self.batch_size):
            i = self.batch_idx[self.batch_id]
            self.batch_id += 1
            #self.batch_id %= self.batch_size
            self.batch_id %= self.no
            if (self.batch_id == 0):
                self.batch_idx = sample_batch_index(self.no, self.no)
                #miss_data = self.miss.make_missdata(self.data, self.miss_rate)
                #self.data_m = 1. - np.isnan(miss_data).astype(float)
            idx1 = self.data_idx[i]
            idx2 = self.data_idx[i]+self.input_width
            #print(idx1, idx2)
        
            Y_mb = self.data[idx1:idx2]
            X_mb = Y_mb.copy()
            M_mb = self.data_m[idx1:idx2]
            Z_mb = uniform_sampler(0, 0.01, shape=X_mb.shape)
            X_mb = M_mb*X_mb + (1-M_mb)*Z_mb
            #H_mb_temp = binary_sampler(self.hint_rate, shape=X_mb.shape)
            #H_mb = M_mb * H_mb_temp
            X_mb[M_mb == 0] = np.nan
            x = np.append(x, [X_mb], axis=0)
            #m = np.append(m, [M_mb], axis=0)
            #h = np.append(h, [H_mb], axis=0)
            y = np.append(y, [Y_mb], axis=0)
            
        #return [x, m, h], y
        return x, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return

dgen = GainDataGenerator(df)

class GAIN(keras.Model):
    def __init__(self, shape, alpha=100., load=False, hint_rate=0.9, gen_sigmoid=True, **kwargs):
        super(GAIN, self).__init__(**kwargs)
        self.shape = shape
        self.dim = np.prod(shape).astype(int)
        self.h_dim = self.dim
        self.gen_sigmoid = gen_sigmoid
        self.build_generator()
        self.build_discriminator()
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.generator_optimizer = Adam()
        self.discriminator_optimizer = Adam()

    ## GAIN models
    def build_generator(self):
        last_activation = 'sigmoid' if self.gen_sigmoid else None
        xavier_initializer = tf.keras.initializers.GlorotNormal()

        x = Input(shape=(self.dim,), name='generator_input_x')
        m = Input(shape=(self.dim,), name='generator_input_m')

        a = Concatenate()([x, m])

        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #a = keras.layers.BatchNormalization()(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #a = keras.layers.BatchNormalization()(a)
        G_prob = Dense(self.dim, activation=last_activation, kernel_initializer=xavier_initializer)(a)
        self.generator = keras.models.Model([x, m], G_prob, name='generator')

    def build_discriminator(self):
        xavier_initializer = tf.keras.initializers.GlorotNormal()

        x = Input(shape=(self.dim,), name='discriminator_input_x')
        h = Input(shape=(self.dim,), name='discriminator_input_h')

        a = Concatenate()([x, h])

        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        D_prob = Dense(self.dim, activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        self.discriminator = keras.models.Model([x, h], D_prob, name='discriminator')
        
    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        shape = inputs.shape
        dims = np.prod(shape[1:])
        input_width = shape[1]
        #print('inputs.shape=',inputs.shape)
        x = inputs
        #x = x.reshape((n, -1))
        #print('dims=',dims)
        x = keras.layers.Reshape((dims,))(x)
        #x = keras.layers.Reshape(tf.TensorShape((self.dim,)))(x)
        #print('x =', x)
        #print('x.shape = ', x.shape)
        #x = keras.layers.Reshape(tf.TensorShape([57]))(x)
        
        isnan = tf.math.is_nan(x)
        #m = 1.- keras.backend.cast(isnan, dtype=tf.float32)
        m = tf.where(isnan, 0., 1.)
        z = keras.backend.random_uniform(shape=tf.shape(x), minval=0.0, maxval=0.01)
        x = tf.where(isnan, z, x)
        #z = uniform_sampler(0, 0.01, shape=x.shape)
        #z = tf.keras.backend.random_uniform(shape=x.shape, minval=0.0, maxval=0.01)
        imputed_data = self.generator([x, m], training=False)
        #imputed_data = m*x + (1-m)*imputed_data
        imputed_data = tf.where(isnan, imputed_data, np.nan)
        imputed_data = keras.layers.Reshape(shape[1:])(imputed_data)
        #print('imputed_data.shape = ', imputed_data.shape)
        
        return imputed_data
    
    def D_loss(M, D_prob):
        ## GAIN loss
        return -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                         + (1-M) * tf.keras.backend.log(1. - D_prob + 1e-8))
    
    def G_loss(self, M, D_prob, X, G_sample):
        G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / (tf.reduce_mean(M) + 1e-8)
        #G_loss_temp = GAIN.G_loss_bincross(M, D_prob)
        #MSE_loss = GAIN.MSE_loss(M, X, G_sample)
        G_loss = G_loss_temp + self.alpha * MSE_loss
        return G_loss
        
    def RMSE_loss(y_true, y_pred):
        isnan = tf.math.is_nan(y_pred)
        M = tf.where(isnan, 1., 0.)
        return tf.sqrt(tf.reduce_sum(tf.where(isnan, 0., y_pred-y_true)**2)/tf.reduce_sum(1-M))
    
    def train_step(self, data):
        #[x, m, h], y = data
        x, y = data
        #X = keras.layers.Reshape((self.dim,), input_shape=self.shape)(x)
        #Y = keras.layers.Reshape((self.dim,), input_shape=self.shape)(y)
        X = keras.layers.Flatten()(x)
        Y = keras.layers.Flatten()(y)
        #X = tf.reshape(x, shape=(x.shape[0], -1))
        #Y = tf.reshape(y, shape=(x.shape[0], -1))
        isnan = tf.math.is_nan(X)
        #M = 1 - keras.backend.cast(isnan, dtype=tf.float32)
        M = tf.where(isnan, 0., 1.)
        Z = keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=0.01)
        #H_temp = binary_sampler(self.hint_rate, shape=X.shape)
        H_rand = keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=1.)
        #H_temp = 1*keras.backend.cast((H_rand < self.hint_rate), dtype=tf.float32)
        H_temp = tf.where(H_rand < self.hint_rate, 1., 0.)
        
        H = M * H_temp
        #X = M * X + (1-M) * Z
        X = tf.where(isnan, Z, X)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_sample = self.generator([X, M], training=True)

            # Combine with observed data
            #Hat_X = tf.where(isnan, G_sample, X)
            Hat_X = X * M + G_sample * (1-M)
            D_prob = self.discriminator([Hat_X, H], training=True)
            gen_loss = self.G_loss(M, D_prob, X, G_sample)
            disc_loss = tf.keras.backend.mean(tf.keras.losses.binary_crossentropy(M, D_prob))
            #disc_loss = GAIN.D_loss(M, D_prob)
            #disc_loss = GAIN.D_loss(M, D_prob)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        rmse = tf.sqrt(tf.reduce_sum(tf.where(isnan, G_sample - Y, 0.)**2)/tf.reduce_sum(1-M))
        return {
                 'gen_loss':     gen_loss,
                 'disc_loss':    disc_loss,
                 'rmse':         rmse,
               }
    
    def save(self, save_dir='savedta'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        disc_savefile = os.path.join(save_dir, 'discriminator.h5')
        gen_savefile = os.path.join(save_dir, 'generator.h5')
        self.discriminator.save_weights(disc_savefile)
        self.generator.save_weights(gen_savefile)

    def load(self, save_dir='savedata'):
        disc_savefile = os.path.join(save_dir, 'discriminator.h5')
        gen_savefile = os.path.join(save_dir, 'generator.h5')
        try:
            self.discriminator.load_weights(disc_savefile)
            self.generator.load_weights(gen_savefile)
            print('model weights loaded')
        except:
            print('model loadinng error')

def make_dataset_gain(self, data):
    print('@@@ make_dataset_gain')
    print('data => ', data)
    print('data type => ', type(data))    
    dg = GainDataGenerator(
        df,
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

WindowGenerator.make_dataset = make_dataset_gain

train_df = df_all
val_df = df_all
test_df = df_all

wide_window = WindowGenerator(
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