
import pandas as pd
import numpy as np


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
    # n = np_data.shape[1]

    #print("interpolate------")
    #print(type(np_data))
    #print(np_data)

    data = pd.DataFrame(np_data)
    #data = np_data

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