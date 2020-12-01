'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from utils import getUseTrain

from tensorflow import keras

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
import tensorflow
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import os

class GDense(tf.keras.layers.Layer):
    def __init__(self, w, b):
        super(GDense, self).__init__()
        self.w = w
        self.b = b

    def call(self, x):
        return tf.matmul(x, self.w) + self.b

class GAIN():
    def __init__(self, dim, alpha, load=False):
        self.dim = dim
        self.alpha = alpha
        self.h_dim = int(dim)

        self.build_generator()
        #self.generator.summary()
        self.build_discriminator()
        #self.discriminator.summary()

        self.generator_optimizer = Adam()
        self.discriminator_optimizer = Adam()
        if load == True:
            self.load()

    ## GAIN models
    def build_generator(self):

        xavier_initializer = tf.keras.initializers.GlorotNormal()

        x = Input(shape=(self.dim,), name='generator_input_x')
        m = Input(shape=(self.dim,), name='generator_input_m')

        a = Concatenate()([x, m])

        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        a = Dense(self.h_dim, activation='relu', kernel_initializer=xavier_initializer)(a)
        G_prob = Dense(self.dim, activation='sigmoid', kernel_initializer=xavier_initializer)(a)
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

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def G_loss_bincross(M, D_prob):
        return -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))

    def MSE_loss(M, X, G_sample):
        return tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    def G_loss(self, M, D_prob, X, G_sample):
        G_loss_temp = -tf.reduce_mean((1-M) * tf.keras.backend.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
        #G_loss_temp = GAIN.G_loss_bincross(M, D_prob)
        #MSE_loss = GAIN.MSE_loss(M, X, G_sample)
        G_loss = G_loss_temp + self.alpha * MSE_loss 
        return G_loss

    # Transform train_on_batch return value
    # to dict expected by on_batch_end callback
    def named_logs(model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def D_loss(M, D_prob):
        ## GAIN loss
        return -tf.reduce_mean(M * tf.keras.backend.log(D_prob + 1e-8) \
                        + (1-M) * tf.keras.backend.log(1. - D_prob + 1e-8)) 

    def save(self, save_dir='savedata'):
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

    # `tf.function`이 어떻게 사용되는지 주목해 주세요.
    # 이 데코레이터는 함수를 "컴파일"합니다.
    @tf.function
    def train_step(self, inputs):
        X, M, H = inputs
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_sample = self.generator([X, M], training=True)

            # Combine with observed data
            Hat_X = X * M + G_sample * (1-M)
            D_prob = self.discriminator([Hat_X, H], training=True)
            gen_loss = self.G_loss(M, D_prob, X, G_sample)
            disc_loss = GAIN.D_loss(M, D_prob)
            #disc_loss = tf.keras.backend.mean(tf.keras.losses.binary_crossentropy(M, D_prob))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        #return gen_loss, disc_loss


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_x, batch_size, hint_rate):
        'Initialization'
        self.no, self.dim = data_x.shape
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.data_m = 1 - np.isnan(data_x)
        self.norm_data_x = np.nan_to_num(self.norm_data, 0)
        self.idx = 0
        self.batch_cnt = self.no // self.batch_size
        self.shuffle_idx = np.random.permutation(self.no)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_idx = self.shuffle_idx[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
        self.idx = (self.idx+1)%self.batch_cnt
        if (self.idx == 0):
            self.shuffle_idx = np.random.permutation(self.no)
        X_mb = self.norm_data_x[batch_idx, :]  
        M_mb = self.data_m[batch_idx, :]  

        # Sample random vectors  
        Z_mb = uniform_sampler(0, 0.01, self.batch_size, self.dim)
        H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, self.dim)
        H_mb = M_mb * H_mb_temp
          
        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

        return X_mb, M_mb, H_mb

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return


def gain (train_data, test_data, gain_parameters):
    '''Impute missing values in data_x
    
    Args:
        - data_x: original data with missing values
        - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
        - imputed_data: imputed data
    '''
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    useTrain = getUseTrain(gain_parameters) 

    # Define mask matrix
    if useTrain:
        train_mask = 1 - np.isnan(train_data)
        train_row, dim = train_data.shape # 4287, 27
        train_data = np.nan_to_num(train_data, 0)
        h_dim = int(dim)

    test_mask = 1 - np.isnan(test_data)
    test_row, dim = test_data.shape # 4287, 27
    test_data = np.nan_to_num(test_data, 0)
    
    ''' 학습 '''
    # main logic
    train_generator = DataGenerator(train_data, batch_size, hint_rate)

    # break point
    exit(0)

    ds = tf.data.Dataset.from_generator(
        #lambda: train_generator.__iter__(),
        lambda: train_generator,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(
            [batch_size, train_generator.dim],
            [batch_size, train_generator.dim],
            [batch_size, train_generator.dim],
        )
    ).repeat(-1).prefetch(10)

    # it = iter(ds)
    # X_mb, M_mb, H_mb = next(it)
  
    ## Iterations
    gain = GAIN(dim, alpha, load=True)

    it_ds = iter(ds)

    # Start Iterations
    progress = tqdm(range(iterations))
    for it in progress:
        X_mb, M_mb, H_mb = next(it_ds)
        gain.train_step([X_mb, M_mb, H_mb])
    gain.save()
      
    ''' 테스트 '''
    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, test_row, dim) 
    M_mb = test_mask
    X_mb = test_data          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
    X_mb = X_mb.astype(np.float32)
    M_mb = M_mb.astype(np.float32)

    imputed_data = gain.generator.predict([X_mb, M_mb])
    imputed_data = test_mask * test_data + (1 - test_mask) * imputed_data
    return imputed_data
