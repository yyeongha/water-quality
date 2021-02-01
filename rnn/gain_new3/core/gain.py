# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import os

#-----------------------

class GAIN(tf.keras.Model):
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

        shape = self.shape
        # x = Input(shape=(self.dim,), name='generator_input_x')
        # m = Input(shape=(self.dim,), name='generator_input_m')
        x = Input(shape=shape, name='generator_input_x')
        m = Input(shape=shape, name='generator_input_m')

        x_f = tf.keras.layers.Flatten()(x)
        m_f = tf.keras.layers.Flatten()(m)

        a = Concatenate()([x_f, m_f])

        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        # a = keras.layers.BatchNormalization()(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        # a = keras.layers.BatchNormalization()(a)
        a = Dense(self.dim, activation=last_activation, kernel_initializer=xavier_initializer)(a)
        G_prob = tf.keras.layers.Reshape(shape)(a)
        self.generator = tf.keras.models.Model([x, m], G_prob, name='generator')

    def build_discriminator(self):
        xavier_initializer = tf.keras.initializers.GlorotNormal()
        shape = self.shape

        # x = Input(shape=(self.dim,), name='discriminator_input_x')
        # h = Input(shape=(self.dim,), name='discriminator_input_h')
        x = Input(shape=shape, name='discriminator_input_x')
        h = Input(shape=shape, name='discriminator_input_h')

        x_f = tf.keras.layers.Flatten()(x)
        h_f = tf.keras.layers.Flatten()(h)

        a = Concatenate()([x_f, h_f])

        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.dim, activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        D_prob = tf.keras.layers.Reshape(shape)(a)
        self.discriminator = tf.keras.models.Model([x, h], D_prob, name='discriminator')

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        shape = inputs.shape
        dims = np.prod(shape[1:])
        input_width = shape[1]
        # print('inputs.shape=',inputs.shape)
        x = inputs
        # x = x.reshape((n, -1))
        # print('dims=',dims)
        # x = keras.layers.Reshape((dims,))(x)
        # x = keras.layers.Reshape(tf.TensorShape((self.dim,)))(x)
        # print('x =', x)
        # print('x.shape = ', x.shape)
        # x = keras.layers.Reshape(tf.TensorShape([57]))(x)

        isnan = tf.math.is_nan(x)
        # m = 1.- keras.backend.cast(isnan, dtype=tf.float32)
        m = tf.where(isnan, 0., 1.)
        z = tf.keras.backend.random_uniform(shape=tf.shape(x), minval=0.0, maxval=0.01)
        x = tf.where(isnan, z, x)
        # z = uniform_sampler(0, 0.01, shape=x.shape)
        # z = tf.keras.backend.random_uniform(shape=x.shape, minval=0.0, maxval=0.01)
        imputed_data = self.generator([x, m], training=False)
        # imputed_data = m*x + (1-m)*imputed_data
        # imputed_data = tf.where(isnan, imputed_data, np.nan)
        imputed_data = tf.where(isnan, imputed_data, x)
        # imputed_data = keras.layers.Reshape(shape[1:])(imputed_data)
        # print('imputed_data.shape = ', imputed_data.shape)

        return imputed_data

    def D_loss(M, D_prob):
        ## GAIN loss
        return -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                               + (1 - M) * tf.keras.backend.log(1. - D_prob + 1e-8))

    def G_loss(self, M, D_prob, X, G_sample):
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.keras.backend.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / (tf.reduce_mean(M) + 1e-8)
        # G_loss_temp = GAIN.G_loss_bincross(M, D_prob)
        # MSE_loss = GAIN.MSE_loss(M, X, G_sample)
        G_loss = G_loss_temp + self.alpha * MSE_loss
        return G_loss

    def RMSE_loss(y_true, y_pred):
        isnan = tf.math.is_nan(y_true)
        M = tf.where(isnan, 1., 0.)
        return tf.sqrt(tf.reduce_sum(tf.where(isnan, 0., y_pred - y_true) ** 2) / tf.reduce_sum(1 - M))

    def train_step(self, data):
        # [x, m, h], y = data
        x, y = data
        # X = keras.layers.Reshape((self.dim,), input_shape=self.shape)(x)
        # Y = keras.layers.Reshape((self.dim,), input_shape=self.shape)(y)
        # X = keras.layers.Flatten()(x)
        # Y = keras.layers.Flatten()(y)
        X = x
        Y = y
        # X = tf.reshape(x, shape=(x.shape[0], -1))
        # Y = tf.reshape(y, shape=(x.shape[0], -1))
        isnan = tf.math.is_nan(X)
        # M = 1 - keras.backend.cast(isnan, dtype=tf.float32)
        M = tf.where(isnan, 0., 1.)
        Z = tf.keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=0.01)
        # H_temp = binary_sampler(self.hint_rate, shape=X.shape)
        H_rand = tf.keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=1.)
        # H_temp = 1*keras.backend.cast((H_rand < self.hint_rate), dtype=tf.float32)
        H_temp = tf.where(H_rand < self.hint_rate, 1., 0.)

        H = M * H_temp
        # X = M * X + (1-M) * Z
        X = tf.where(isnan, Z, X)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_sample = self.generator([X, M], training=True)

            # Combine with observed data
            # Hat_X = tf.where(isnan, G_sample, X)
            Hat_X = X * M + G_sample * (1 - M)
            D_prob = self.discriminator([Hat_X, H], training=True)
            gen_loss = self.G_loss(M, D_prob, X, G_sample)
            disc_loss = tf.keras.backend.mean(tf.keras.losses.binary_crossentropy(M, D_prob))
            # disc_loss = GAIN.D_loss(M, D_prob)
            # disc_loss = GAIN.D_loss(M, D_prob)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        rmse = tf.sqrt(tf.reduce_sum(tf.where(isnan, G_sample - Y, 0.) ** 2) / tf.reduce_sum(1 - M))
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'rmse': rmse,
        }

    def save(self, save_dir='save'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        disc_savefile = os.path.join(save_dir, 'discriminator.h5')
        gen_savefile = os.path.join(save_dir, 'generator.h5')
        self.discriminator.save_weights(disc_savefile)
        self.generator.save_weights(gen_savefile)

    def load(self, save_dir='save'):
        disc_savefile = os.path.join(save_dir, 'discriminator.h5')
        gen_savefile = os.path.join(save_dir, 'generator.h5')
        try:
            self.discriminator.load_weights(disc_savefile)
            self.generator.load_weights(gen_savefile)
           # print('model weights loaded')
        except:
            print('model loadinng error')

    #def compile_and_fit(model, window, patience=10, epochs=100):
    def compile_and_fit(self, window, patience=10, epochs=100):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

        #self.compile(loss=GAIN.RMSE_loss)
        self.compile(loss=GAIN.RMSE_loss)

        history = self.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[early_stopping])
        return history

#------------------------

def dataset_slice(df, train_ratio, val_ratio, test_ratio):
    total_no = df.shape[0]
    train_no = int(total_no * train_ratio)

    #val_no = int(total_no * (train_ratio + val_ratio))

    #train_slice = slice(0, train_no)    #0.8
    #val_slice = slice(train_no, val_no)   #0.1
    #test_slice = slice(val_no, None)
    train_slice = slice(0, train_no)  # 0.8
    val_slice = slice(train_no, None)  # 0.1
    test_slice = slice(0, None)

    train = pd.DataFrame(df[train_slice])
    val = pd.DataFrame(df[val_slice])
    test = pd.DataFrame(df[test_slice])

    test_slice2 = slice(total_no - 288, None)
    test2 = pd.DataFrame(df[test_slice2])

    #print('total_no :1111111111111111111111111 ')
    #print('total_no : ', total_no)
    #print('train : ', train.shape)
    #print('val : ', val.shape)
    #print('test : ', test.shape)


    return train, val, test, test2



def create_dataset_with_gain(gain, df, window = None, shape = None):

    if window == None:
        unit_shape = shape
    else:
        unit_shape = window.dg.shape[1:]
    #unit_shape = window

    time_seq = unit_shape[0]
    # ----------
    gans = []
    oris = []

    #print(type(df))
    #print(type(df[0]))

    for i in range(len(df)):
        x = df[i].to_numpy()
        total_n = x.shape[0]
        n = (total_n // time_seq) * time_seq
        #x = x[0:n]
        #x_block = x.reshape((-1,) + unit_shape)
        #y = gain.predict(x_block)
        #y_gan = y.reshape(x.shape)

        x_block = x[0:n].copy()
        x_block_shape = x_block.shape
        x_block = x_block.reshape((-1,) + unit_shape)

        x_block_remain = x[-time_seq:].copy()
        x_block_remain_shape = x_block_remain.shape
        x_block_remain = x_block_remain.reshape((-1,) + unit_shape)

        y = gain.predict(x_block)
        y_remain = gain.predict(x_block_remain)

        y_gan = y.reshape(x_block_shape)
        y_remain_gan = y_remain.reshape(x_block_remain_shape)
        y_gan = np.append(y_gan, y_remain_gan[-(total_n-n):], axis=0)

        #x_block_remain = x[-time_seq]

        # cut off sin, cos data
        if (i > 0):
            #x = x[:, :-4]
            #y_gan = y_gan[:, :-4]
            x = x[:, :]
            y_gan = y_gan[:, :]
        gans.append(y_gan)
        oris.append(x)

    #print('create_dataset_with_gain org')
    #print(oris[0])

    #print('create_dataset_with_gain gan')
    #print(gans[0])

    ori = np.concatenate(oris, axis=1)
    gan = np.concatenate(gans, axis=1)

    #print('------------------------')
    #print(gan.shape)
    #print('------------------------')

    return ori, gan

    #print(x.shape)
    #print(y_gan.shape)



def create_dataset_interpol(df, window=24*5):

    time_seq = window
    # ----------
    gans = []
    oris = []
    for i in range(len(df)):
        x = df[i].to_numpy()
        #total_n = x.shape[0]
        #n = (total_n // time_seq) * time_seq
        #x = x[0:n]

        # cut off sin, cos data
        #if (i > 0):
            #x = x[:, :-4]


        oris.append(x)

    ori = np.concatenate(oris, axis=1)

    return ori



def model_GAIN(shape, gen_sigmoid, window=None, training_flag=False, epochs = 100, model_save_path='../save'):
    gain = GAIN(shape=shape, gen_sigmoid=gen_sigmoid)

    if training_flag == True:

        #MAX_EPOCHS = 2000

        # gain.compile(loss=GAIN.RMSE_loss)
        #history = gain.compile_and_fit(window, patience=MAX_EPOCHS // 5, epochs=epochs)
        history = gain.compile_and_fit(window, patience=epochs//5, epochs=epochs)
        gain.save(save_dir=model_save_path)
    else:
        #print('gain load')
        gain.load(save_dir=model_save_path)
        gain.compile(loss=GAIN.RMSE_loss)

    return gain


#-------------------------------------------------11111
'''
norm_df = pd.concat(df,axis=0)

data = norm_df.to_numpy()

total_n = wide_window.dg.data.shape[0]
#print("total_n : ",total_n)
unit_shape = wide_window.dg.shape[1:]
#print(unit_shape)
#dim = np.prod(wide_window.dg.shape[1:]).astype(int)
dim = wide_window.dg.shape[1]
#print(dim)
n = (total_n//dim)*dim

x = data[0:n].copy()
y_true = data[0:n].copy()

#x = interpolate(x, max_gap=3)

#print('x.shape =', x.shape)
x_reshape = x.reshape((-1,)+unit_shape)
#print('x_reshape.shape =', x_reshape.shape)
isnan = np.isnan(x_reshape)
#print(isnan.sum())
#print('y_true.shape=', y_true.shape)
isnan = np.isnan(y_true)
#print(isnan.sum())

x_remain = data[-wide_window.dg.shape[1]:].copy()
x_remain_reshape = x_remain.reshape((-1,)+unit_shape)
x_remain_reshape.shape

gain.evaluate(x_reshape, y_true.reshape((-1,)+unit_shape))

y_pred = gain.predict(x_reshape)
y_remain_pred = gain.predict(x_remain_reshape)

y_pred = y_pred.reshape(y_true.shape)
y_remain_pred = y_remain_pred.reshape(x_remain.shape)
#print(y_pred.shape, y_remain_pred.shape)
y_pred = np.append(y_pred, y_remain_pred[-(total_n-n):], axis=0)
#print(y_pred.shape)

y_pred[~np.isnan(data)] = np.nan

#n = 8
#plt.figure(figsize=(9,20))
#for i in range(n):
#    #plt.subplot('%d1%d'%(n,i))
#    plt.subplot(811+i)
#    plt.plot(x[:, i])
#    plt.plot(y_pred[:, i])
#plt.show()

total_n = wide_window.dg.data.shape[0]
#print(total_n)
unit_shape = wide_window.dg.shape[1:]
#print('unit_shape=', unit_shape)
time_seq = unit_shape[0]
#print(time_seq)
n = (total_n // time_seq) * time_seq
#print('n=', n)
'''
#-------------------------------------------------11111