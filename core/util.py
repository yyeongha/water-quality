#####################################################################################
# 시계열 데이터를 처리하고, 모델을 평가하는 다양한 유틸리티 함수들을 포함
# 정규화, 샘플링, 보간, 일별 평균 계산 및 모델 평가 등이 포함됨
#####################################################################################

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 데이터프레임을 정규화
def normalize(df):
    # normalize data
    df_all = pd.concat(df)

    train_mean = df_all.mean()
    train_std = df_all.std()

    for i in range(len(df)):
        df[i] = (df[i] - train_mean) / train_std

    return df_all, train_mean, train_std, df

# 미니배치 인덱스 샘플링
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

# 이진 랜덤 변수 샘플링
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

# 균등 랜덤 변수 샘플링
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

# 결측데이터를 보간
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

# 시간별 데이터를 일별 평균으로 변환
def hour_to_day_mean(array):
    time = 24
    array = array.reshape((array.shape[0], array.shape[1] // time, time, array.shape[2]))
    array = array.mean(2)
    return array

# 모델 성능 평가하고 성능지표 계산
def compa(model=None,df = None, plot_col=0, input_width=7*24, label_width=5*24, target_std=None, target_mean=None, predict_day=4):
    width = input_width + label_width
    length = df.shape[0] - width
    inputs = []
    labels = []

    for i in range(0, length, 24):
        dataset = df.iloc[i:i+width].to_numpy()
        input = dataset[:input_width]
        label = dataset[input_width:, plot_col:plot_col+1]

        input = input.reshape((-1,)+input.shape)
        label = label.reshape((-1,)+label.shape)

        inputs.append(input)
        labels.append(label)

    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    predictions = model(inputs).numpy()

    predictions = predictions * target_std[plot_col] + target_mean[plot_col]
    labels = labels * target_std[plot_col] + target_mean[plot_col]

    pred_day = hour_to_day_mean(predictions)

    label_day = hour_to_day_mean(labels)

    inputs_target = inputs[:,:,plot_col:plot_col+1]
    inputs_target = inputs_target * target_std[plot_col] + target_mean[plot_col]
    inputs_day = hour_to_day_mean(inputs_target)

    o1 = np.mean(label_day[:,predict_day,:])
    nse1 = ((label_day - pred_day)**2).sum(axis=0)
    nse2 = ((label_day - o1)**2).sum(axis=0)
    nse3 = 1 - (nse1[predict_day]/nse2[predict_day])
    pbias1 = (label_day - pred_day).sum(axis=0)
    pbias2 = (label_day).sum(axis=0)
    pbias3 = (pbias1[predict_day]/pbias2[predict_day])*100
        
    labels_test = labels.mean(axis=1)
    predis_test = inputs_target.mean(axis=1)

    nse2_1 = ((labels_test - predis_test)**2).sum()
    nse2_2 = ((labels_test - o1)**2).sum()
    nse2_3 = 1 - (nse2_1/nse2_2)

    pbias2_1 = (labels_test - predis_test).sum()
    pbias2_2 = labels_test.sum()
    pbias2_3 = pbias2_1/pbias2_2 * 100

    mae = (np.abs(label_day - pred_day)).mean()
    mse = ((label_day - pred_day)**2).mean()
    rmse = np.sqrt(((label_day - pred_day)**2).mean())

    o_ = np.mean(label_day[:,predict_day,:])
    p_ = np.mean(pred_day[:,predict_day,:])

    oi = label_day[:,predict_day,:]
    pi = pred_day[:,predict_day,:]

    high = ((oi-o_)*(pi-p_)).sum()
    low = np.sqrt( ( (oi-o_)**2 ).sum() )
    low = low * np.sqrt( ( (pi-p_)**2 ).sum() )

    R = high/low
    RS = (R)**2

    return nse3, pbias3, pred_day, labels, mae, rmse, RS, R
