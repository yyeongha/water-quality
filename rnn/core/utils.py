import os
import datetime
import pandas as pd
import numpy as np


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


def interpolate(np_data, max_gap=3):
    # data = pd.DataFrame(np_data)
    data = np_data
    # create mask
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in data.columns:
        mask[i] = (grp.groupby(i)['ones'].transform('count') < max_gap) | data[i].notnull()
    data = data.interpolate(method='polynomial', order=5, limit=max_gap, axis=0).bfill()[mask]
    return data

def createDataFrame(folder, file_list):
    day = 24 * 60 * 60
    week = day * 7
    year = (365.2425) * day

    da = db = [[] for i in range(len(file_list))]
    for i in range(len(file_list)):
        for filename in file_list[i]:
            path = os.path.join(folder, filename)
            da[i].append(pd.read_excel(path))
        db[i] = pd.concat(da[i])
        db[i].pop('측정소명')
    for i in range(1 , len(db)):
        db[i] = db[i].iloc[:,2:].rename(columns = lambda x: x + '_'+str(i))
    df = pd.concat(db, axis=1)

    date_time = pd.to_datetime(df.pop('측정날짜'), format='%Y.%m.%d %H:%M',utc=True)
    timestamp_s = date_time.map(datetime.datetime.timestamp)
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    # df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    # df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df

# def createDataFrame(folder, file_names):
#     day = 24 * 60 * 60
#     year = (365.2425) * day
#
#     df_full = []
#     df = []
#
#     for i in range(len(file_names)):
#         path = os.path.join(folder, file_names[i])
#         df_full.append(pd.read_excel(path))
#         df.append(df_full[i].iloc[:, 2:11])
#         date_time = pd.to_datetime(df_full[i].iloc[:, 0], format='%Y.%m.%d %H:%M')
#         timestamp_s = date_time.map(datetime.datetime.timestamp)
#         df[i]['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
#         df[i]['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
#         df[i]['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
#         df[i]['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
#
#     df_all = pd.concat(df)
#
#     return df, df_full, df_all


def standardNormalization(df, df_all):
    train_mean = df_all.mean()
    train_std = df_all.std()
    for i in range(len(df)):
        df[i] = (df[i]-train_mean)/train_std
