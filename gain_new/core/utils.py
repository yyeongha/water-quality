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
    data = pd.DataFrame(np_data)

    # create mask
    mask = data.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for i in data.columns:
        mask[i] = (grp.groupby(i)['ones'].transform('count') < max_gap) | data[i].notnull()
    data = data.interpolate(method='polynomial', order=5, limit=max_gap, axis=0).bfill()[mask]
    return data