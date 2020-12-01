from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from gain import gain
from utils import rmse_loss, data_loader, init_preprocess, getUseTrain


def main (args):
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: directory or file
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''
    # Make Parameter Dictionary for gain
    gain_parameters = {
        'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'iterations': args.iterations
    }

    # Make Parameter Dictionary for Pre-Process
    preprocess_parameters = {
        'input': args.data_name,
        'target': list(map(int, args.target.split(','))),
        'time': args.time,
        'target_all': args.target_all,
        'fill_cnt': args.fill_cnt
    }

    # Return status of use train flag
    useTrain = getUseTrain(gain_parameters) 

    # Init Pre-Process and Make Traing Data
    preprocess = init_preprocess(preprocess_parameters)
    output_np = preprocess.getDataSet()

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(output_np, args.miss_rate)

    # Read the raw excel, Not use miss_data_x created data_loader
    if useTrain:
        miss_data_x = preprocess.getRawDataSet()

    # Numpy object reshape
    miss_data_x = preprocess.getReshapeNp(miss_data_x)
    ori_data_x = preprocess.getReshapeNp(ori_data_x)
    data_m = preprocess.getReshapeNp(data_m)

    # Impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters)
    
    # Report the RMSE performance
    if useTrain:
        rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)

    # Make result excel
    data_df = pd.DataFrame(data=imputed_data_x)
    data_df.to_excel('./output/result_reshape.xlsx', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # AS-IS parameters
    parser.add_argument(
        '--data_name',
        default='spam.csv',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)

    # TO-BE parameters
    parser.add_argument(
        '--target',
        help='blar blar blar',
        default='0', # string array, ex) 2,3,4,5
        type=str)
    parser.add_argument(
        '--time',
        help='blar blar blar',
        default=120, # 5 day
        type=int)
    parser.add_argument(
        '--target_all',
        help='blar blar blar',
        default=False, # utils.py line 210
        type=bool)
    parser.add_argument(
        '--fill_cnt',
        help='blar blar blar',
        default=2,
        type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
