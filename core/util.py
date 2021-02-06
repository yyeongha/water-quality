# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(df):
    # normalize data
    df_all = pd.concat(df)
    #print('_____________________________')
    #print(df_all.shape)
    #print(df[0].shape)

    train_mean = df_all.mean()
    train_std = df_all.std()
    #for i in range(len(file_names)):
    for i in range(len(df)):
        df[i] = (df[i] - train_mean) / train_std

    #print('_____________________________')
    #print(df_all.shape)
    #print(df[0].shape)

    return df_all, train_mean, train_std, df




def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
        - total: total number of samples
        - batch_size: batch size

    Returns:
        - batch_idx: batch index
    '''


    total_idx = np.random.permutation(total)
    #print('total_idx11111111111111')
    #print(total_idx)
    batch_idx = total_idx[:batch_size]
    #print(batch_idx)
    return batch_idx


def binary_sampler(p, shape):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - shape: matrix shape

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=shape)
    binary_random_matrix = 1 * (unif_random_matrix < p)
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
    return np.random.uniform(low, high, size=shape)



def interpolate(np_data, max_gap=3):
    data = pd.DataFrame(np_data)
    # create mask
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in data.columns:
        mask[i] = (grp.groupby(i)['ones'].transform('count') < max_gap) | data[i].notnull()
    data = data.interpolate(method='polynomial', order=5, limit=max_gap, axis=0).bfill()[mask]
    return data.to_numpy()
    # return data

def hour_to_day_mean(array):
    time = 24
    array = array.reshape((array.shape[0], array.shape[1] // time, time, array.shape[2]))
    array = array.mean(2)
    return array

def evaluate_predict(model=None,df = None, plot_col=0, input_width=7*24, label_width=5*24, target_std=None, target_mean=None, predict_day=4):

    print(df.shape)
    print(plot_col)

    width = input_width + label_width

    length = df.shape[0]
    length -= width

    inputs = []
    labels = []

    for i in range(0,length,24):
#         print('i = ', i)
        dataset = df.iloc[i:i+width].to_numpy()
        input = dataset[:input_width]
        label = dataset[input_width:, plot_col:plot_col+1]

        input = input.reshape((-1,)+input.shape)
        label = label.reshape((-1,)+label.shape)

        inputs.append(input)
        labels.append(label)

    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    predictions = model(inputs)

    predictions = predictions.numpy() * target_std[plot_col] + target_mean[plot_col]
    labels = labels * target_std[plot_col] + target_mean[plot_col]

    pred_day = hour_to_day_mean(predictions)

    label_day = hour_to_day_mean(labels)

    inputs_target = inputs[:,:,plot_col:plot_col+1]
    inputs_target = inputs_target * target_std[plot_col] + target_mean[plot_col]

    inputs_day = hour_to_day_mean(inputs_target)


    o1 = np.mean(label_day[:,predict_day,0])
    nse1 = ((pred_day-label_day)**2).sum(axis=0)
    nse2 = ((label_day - o1)**2).sum(axis=0)
    nse3 = 1 - (nse1[predict_day]/nse2[predict_day])

    pbias1 = (label_day - pred_day).sum(axis=0)
    pbias2 = (label_day).sum(axis=0)
    pbias3 = (pbias1[predict_day]/pbias2[predict_day])*100



    return float(nse3), float(np.abs(pbias3)), pred_day, label_day

def plot_pred(model=None, pred_day=None, label_day=None, predict_target_day=4, target_column = None):
        plt.figure()
        plt.title(model+'-'+target_column)
        plt.plot(label_day[:, predict_target_day, :], label='label')
        plt.plot(pred_day[:, predict_target_day, :], label='pred')
        plt.legend()
        plt.show()
