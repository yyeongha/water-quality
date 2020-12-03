'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow.compat.v1 as tf

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from utils import getUseTrain


tf.disable_v2_behavior()


def gain (train_data, test_data, gain_parameters):
    '''Impute missing values in data_x

    Args:
        - train_data: original train data with missing values
        - test_data: original test data with missing values
        - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations

    Returns:
        - imputed_data: imputed data
    '''

    # Debug
    # print('train_data = ', train_data)
    # print('test_data = ', test_data)
    # print('train_data.shape = ', train_data.shape) # (4287, 27)
    # print('test_data.shape = ', test_data.shape) # (1837, 27)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    dir_name = gain_parameters['dir_name']
    useTrain = getUseTrain(gain_parameters) 

    # Define mask matrix
    if useTrain:
        train_mask = 1 - np.isnan(train_data)
        train_row, dim = train_data.shape # 4287, 27
        train_data = np.nan_to_num(train_data, 0)

        # debug
        # tddf = pd.DataFrame(train_data)
        # tddf.to_excel('./output/train_data.xlsx', index=False)

        # Hidden state dimensions
        h_dim = int(dim)

        # Debug
        print('train_data = ', train_data)

    test_mask = 1 - np.isnan(test_data)

    # tddf = pd.DataFrame(test_mask)
    # tddf.to_excel('./output/test_mask.xlsx', index=False)

    test_row, dim = test_data.shape # 4287, 27
    test_data = np.nan_to_num(test_data, 0)

    # debug
    # tddf = pd.DataFrame(test_data)
    # tddf.to_excel('./output/test_data.xlsx', index=False)

    # Debug
    print('test_data = ', test_data)

    ''' GAIN architecture'''
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape = [None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, dim])

    if useTrain:
        # Discriminator variables
        D_W1 = tf.Variable(xavier_init([dim * 2, h_dim])) # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

        D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))

        D_W3 = tf.Variable(xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

        # Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))

        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    else:
        D_W1 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_W1.npy".format(dir_name=dir_name)))
        D_b1 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_b1.npy".format(dir_name=dir_name)))
        D_W2 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_W2.npy".format(dir_name=dir_name)))
        D_b2 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_b2.npy".format(dir_name=dir_name)))
        D_W3 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_W3.npy".format(dir_name=dir_name)))
        D_b3 = tf.Variable(np.load("./classfy_weight/{dir_name}/D_b3.npy".format(dir_name=dir_name)))
        G_W1 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_W1.npy".format(dir_name=dir_name)))
        G_b1 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_b1.npy".format(dir_name=dir_name)))
        G_W2 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_W2.npy".format(dir_name=dir_name)))
        G_b2 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_b2.npy".format(dir_name=dir_name)))
        G_W3 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_W3.npy".format(dir_name=dir_name)))
        G_b3 = tf.Variable(np.load("./classfy_weight/{dir_name}/G_b3.npy".format(dir_name=dir_name)))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ''' GAIN functions'''
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

        # MinMax normalized output
        # G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        G_prob = tf.matmul(G_h2, G_W3) + G_b3
        return G_prob


    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values = [x, h], axis = 1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ''' GAIN structure'''
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if useTrain:
        # Start Iterations
        for it in tqdm(range(iterations)):
            # Sample batch
            batch_idx = sample_batch_index(train_row, batch_size)
            X_mb = train_data[batch_idx, :]
            M_mb = train_mask[batch_idx, :]

            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict = {M: M_mb, X: X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict = {X: X_mb, M: M_mb, H: H_mb})

        # Save weights
        # np.save("./weight/D_W1", D_W1.eval(session=sess))
        # np.save("./weight/D_b1", D_b1.eval(session=sess))
        # np.save("./weight/D_W2", D_W2.eval(session=sess))
        # np.save("./weight/D_b2", D_b2.eval(session=sess))
        # np.save("./weight/D_W3", D_W3.eval(session=sess))
        # np.save("./weight/D_b3", D_b3.eval(session=sess))
        # np.save("./weight/G_W1", G_W1.eval(session=sess))
        # np.save("./weight/G_b1", G_b1.eval(session=sess))
        # np.save("./weight/G_W2", G_W2.eval(session=sess))
        # np.save("./weight/G_b2", G_b2.eval(session=sess))
        # np.save("./weight/G_W3", G_W3.eval(session=sess))
        # np.save("./weight/G_b3", G_b3.eval(session=sess))

        os.makedirs("./classfy_weight/{dir_name}".format(dir_name=dir_name), exist_ok=True)
        np.save("./classfy_weight/{dir_name}/D_W1".format(dir_name=dir_name), D_W1.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/D_b1".format(dir_name=dir_name), D_b1.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/D_W2".format(dir_name=dir_name), D_W2.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/D_b2".format(dir_name=dir_name), D_b2.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/D_W3".format(dir_name=dir_name), D_W3.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/D_b3".format(dir_name=dir_name), D_b3.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_W1".format(dir_name=dir_name), G_W1.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_b1".format(dir_name=dir_name), G_b1.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_W2".format(dir_name=dir_name), G_W2.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_b2".format(dir_name=dir_name), G_b2.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_W3".format(dir_name=dir_name), G_W3.eval(session=sess))
        np.save("./classfy_weight/{dir_name}/G_b3".format(dir_name=dir_name), G_b3.eval(session=sess))

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, test_row, dim)
    M_mb = test_mask
    X_mb = test_data
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    # X_mb = M_mb * X_mb + (1 - M_mb)
    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
    imputed_data = test_mask * test_data + (1 - test_mask) * imputed_data
    # imputed_data = test_data + imputed_data

    return imputed_data
