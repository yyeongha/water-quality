import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

## use - class gain_cnn
from tensorflow.keras.layers import Conv1D


class GAIN(keras.Model):
    def __init__(self, 
                shape, 
                alpha=100., 
                load=False, 
                hint_rate=0.9, 
                gen_sigmoid=True, 
                **kwargs):
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

        x = Input(shape=shape, name='generator_input_x')
        m = Input(shape=shape, name='generator_input_m')

        x_f = Flatten()(x)
        m_f = Flatten()(m)

        a = Concatenate()([x_f, m_f])
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.dim, activation=last_activation, kernel_initializer=xavier_initializer)(a)
        G_prob = keras.layers.Reshape(shape)(a)
        self.generator = keras.models.Model([x, m], G_prob, name='generator')

    def build_discriminator(self):
        xavier_initializer = tf.keras.initializers.GlorotNormal()
        shape = self.shape

        x = Input(shape=shape, name='discriminator_input_x')
        h = Input(shape=shape, name='discriminator_input_h')

        x_f = Flatten()(x)
        h_f = Flatten()(h)

        a = Concatenate()([x_f, h_f])
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.dim, activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        D_prob = keras.layers.Reshape(shape)(a)
        self.discriminator = keras.models.Model([x, h], D_prob, name='discriminator')

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        shape = inputs.shape
        dims = np.prod(shape[1:])
        input_width = shape[1]
        x = inputs
        isnan = tf.math.is_nan(x)
        m = tf.where(isnan, 0., 1.)
        z = keras.backend.random_uniform(shape=tf.shape(x), minval=0.0, maxval=0.01)
        x = tf.where(isnan, z, x)
        imputed_data = self.generator([x, m], training=False)
        imputed_data = tf.where(isnan, imputed_data, x)
        return imputed_data

    def D_loss(M, D_prob):
        ## GAIN loss
        return -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                               + (1 - M) * tf.keras.backend.log(1. - D_prob + 1e-8))

    def G_loss(self, M, D_prob, X, G_sample):
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.keras.backend.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / (tf.reduce_mean(M) + 1e-8)
        G_loss = G_loss_temp + self.alpha * MSE_loss
        return G_loss

    def RMSE_loss(y_true, y_pred):
        isnan = tf.math.is_nan(y_true)
        M = tf.where(isnan, 1., 0.)
        return tf.sqrt(tf.reduce_sum(tf.where(isnan, 0., y_pred - y_true) ** 2) / tf.reduce_sum(1 - M))

    def train_step(self, data):
        x, y = data
        X = x
        Y = y
        isnan = tf.math.is_nan(X)
        M = tf.where(isnan, 0., 1.)
        Z = keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=0.01)
        H_rand = keras.backend.random_uniform(shape=tf.shape(X), minval=0.0, maxval=1.)
        H_temp = tf.where(H_rand < self.hint_rate, 1., 0.)
        H = M * H_temp
        X = tf.where(isnan, Z, X)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_sample = self.generator([X, M], training=True)

            # Combine with observed data
            Hat_X = X * M + G_sample * (1 - M)
            D_prob = self.discriminator([Hat_X, H], training=True)
            gen_loss = self.G_loss(M, D_prob, X, G_sample)
            disc_loss = tf.keras.backend.mean(tf.keras.losses.binary_crossentropy(M, D_prob))
            
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

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
            print('model weights loaded')
        except:
            print('model loadinng error')


class GAIN_cnn(keras.Model):
    def __init__(self, shape, alpha=100., load=False, hint_rate=0.9, gen_sigmoid=True, **kwargs):
        super(GAIN_cnn, self).__init__(**kwargs)
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
        shape = self.shape
        last_activation = 'sigmoid' if self.gen_sigmoid else None
        xavier_initializer = tf.keras.initializers.GlorotNormal()

        x = Input(shape=shape, name='generator_input_x')
        m = Input(shape=shape, name='generator_input_m')

        a = Concatenate()([x, m])

        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        #a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #a = keras.layers.BatchNormalization()(a)
        #a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #a = keras.layers.BatchNormalization()(a)
        #G_prob = Conv1D(filters=shape[1], kernel_size=(7,), padding='same', activation=last_activation, kernel_initializer=xavier_initializer)(a)
        a = Dense(shape[-1]*2, activation='relu', kernel_initializer=xavier_initializer)(a)
        G_prob = Dense(shape[-1], activation=last_activation, kernel_initializer=xavier_initializer)(a)
        self.generator = keras.models.Model([x, m], G_prob, name='generator')

    def build_discriminator(self):
        xavier_initializer = tf.keras.initializers.GlorotNormal()
        shape = self.shape

        x = Input(shape=shape, name='discriminator_input_x')
        h = Input(shape=shape, name='discriminator_input_h')

        a = Concatenate()([x, h])
        
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)
        a = Conv1D(filters=shape[1]*2, kernel_size=(7,), padding='same', activation='relu')(a)

        #a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        #D_prob = Dense(self.dim, activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        #D_prob = Conv1D(filters=shape[1], kernel_size=(7,), padding='same', activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        a = Dense(shape[-1]*2, activation='relu', kernel_initializer=xavier_initializer)(a)
        D_prob = Dense(shape[-1], activation='sigmoid', kernel_initializer=xavier_initializer)(a)
        self.discriminator = keras.models.Model([x, h], D_prob, name='discriminator')
        
    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        shape = inputs.shape
        #dims = np.prod(shape[1:])
        #input_width = shape[1]
        # print('inputs.shape=',inputs.shape)
        x = inputs
        #x = x.reshape((n, -1))
        #print('dims=',dims)
        #x = keras.layers.Reshape((dims,))(x)
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
        #imputed_data = tf.where(isnan, imputed_data, np.nan)
        imputed_data = tf.where(isnan, imputed_data, x)
        #imputed_data = keras.layers.Reshape(shape[1:])(imputed_data)
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
        #X = keras.layers.Flatten()(x)
        #Y = keras.layers.Flatten()(y)
        X = x
        Y = y
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
          print('model weights loaded')
        except:
          print('model loadinng error')