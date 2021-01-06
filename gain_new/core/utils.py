import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    return np.random.uniform(low, high, size = shape)


def normalization(data, parameters=None):
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
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


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


def create_dataframe(folder, file_names):
    day = 24 * 60 * 60
    year = (365.2425) * day

    df_full = []
    df_list = []

    for loc in range(len(file_names)):
        df_loc = []
        for y in range(len(file_names[loc])):
            path = os.path.join(folder, file_names[loc][y])
            df_loc.append(pd.read_excel(path))
        df_full.append(pd.concat(df_loc))
        df_list.append(df_full[loc].iloc[:, 2:11])
        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)
        df_list[loc]['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df_list[loc]['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df_list[loc]['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df_list[loc]['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        df_list[loc] = df_list[loc].reset_index(drop=True)

    df_all = pd.concat(df_list)
    return df_list, df_full, df_all


def standard_normalization(df_list, df_all):
    train_mean = df_all.mean()
    train_std = df_all.std()
    for i in range(len(df_list)):
        df_list[i] = (df_list[i]-train_mean)/train_std


def figure_loss(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(history.history['gen_loss'], label='gen_loss')
    ax.plot(history.history['disc_loss'], label='disc_loss')
    ax2.plot(history.history['rmse'], label='rmse', color='green')
    ax2.plot(history.history['val_loss'], label='val_loss', color='red')
    ax.legend(loc='upper center')
    ax2.legend(loc='upper right')
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax2.set_ylabel("rmse")
    # plt.show()
    plt.savefig('./debug/loss.png')


def figure_gain(df_list, wide_window, gain):
    norm_df = pd.concat(df_list, axis=0)
    data = norm_df.to_numpy()
    total_n = wide_window.dg.data.shape[0]
    unit_shape = wide_window.dg.shape[1:]
    dim = wide_window.dg.shape[1]
    n = (total_n//dim)*dim
    x = data[0:n].copy()
    y_true = data[0:n].copy()
    x_reshape = x.reshape((-1,)+unit_shape)
    isnan = np.isnan(x_reshape)
    isnan = np.isnan(y_true)
    x_remain = data[-wide_window.dg.shape[1]:].copy()
    x_remain_reshape = x_remain.reshape((-1,)+unit_shape)
    x_remain_reshape.shape
    gain.evaluate(x_reshape, y_true.reshape((-1,)+unit_shape))
    y_pred = gain.predict(x_reshape)
    y_remain_pred = gain.predict(x_remain_reshape)
    y_pred = y_pred.reshape(y_true.shape)
    y_remain_pred = y_remain_pred.reshape(x_remain.shape)
    y_pred = np.append(y_pred, y_remain_pred[-(total_n-n):], axis=0)
    y_pred = y_pred[:data.shape[0]] # issue fix
    y_pred[~np.isnan(data)] = np.nan # issue (not match row)
    n = 8
    plt.figure(figsize=(9,20))
    for i in range(n):
        plt.subplot(811+i)
        plt.plot(x[:, i])
        plt.plot(y_pred[:, i])
    # plt.show()
    plt.savefig('./debug/gain.png')