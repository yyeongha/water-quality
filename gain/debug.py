from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''select gain version
gain - gain source by origin
gain_shevious - gain source by shevious
'''
# from gain import gain
from gain_shevious import gain

from utils import rmse_loss, data_loader, init_preprocess, getUseTrain
from miss_data import MissData 


def main (parameters):

    # Make Parameter Dictionary for gain
    gain_parameters = {
        'batch_size': parameters['batch_size'],
        'hint_rate': parameters['hint_rate'],
        'alpha': parameters['alpha'],
        'iterations': parameters['iterations']
    }

    # Make Parameter Dictionary for Pre-Process
    preprocess_parameters = {
        'input': parameters['data_name'],
        'target': list(map(int, parameters['target'].split(','))),
        'time': parameters['time'],
        'target_all': parameters['target_all'],
        'fill_cnt': parameters['fill_cnt']
    }

    data_name = parameters['data_name']
    debug_mode = parameters['debug_mode']
    miss_rate = parameters['miss_rate']
    max_tseq = parameters['max_tseq']
    plt_show = parameters['plt_show']

    # Just debug, remove code after complete development
    print('[debug] debug_mode = ', debug_mode)
    print('[debug] miss_rate = ', miss_rate)

    # Return status of use train flag
    # if iterations == 0 then False
    # if iterations != 0 then True
    useTrain = getUseTrain(parameters) 
    
    # Just debug, remove code after complete development
    print('[debug] gain_parameters = ', gain_parameters)
    print('[debug] preprocess_parameters = ', preprocess_parameters)
    print('[debug] useTrain = ', useTrain)

    # Init Pre-Process and Make Traing Data
    # Using "input", "target", "time", "target_all", "fill_cnt"
    preprocess = init_preprocess(preprocess_parameters)

    # Make group shift data for train
    # check init function and getDataFrame function 
    output_df = preprocess.getDataFrame()
    
    # Just debug, remove code after complete development
    print('[debug] output_df = ', output_df)
    if debug_mode == True:
        output_df.to_excel('./debug/output_df.xlsx', index=False)

    # Save miss data pattern
    miss_data_save = MissData(load_dir=False)
    miss_data_save.save(preprocess.getSelfTargetDf(), max_tseq)

    # get mean and standard
    M, S = preprocess.getMeanAndStand()
    
    # Just debug, remove code after complete development
    print('[debug] 각 열에 대한 평균 = ', M)
    print('[debug] 각 열에 대한 표준편차 = ', S)

    if useTrain:
        # make normalization dataframe
        normalization_df = preprocess.normalization(output_df, M, S)

        # Just debug, remove code after complete development
        print('[debug] normalization_df = ', normalization_df)
        if debug_mode == True:
            normalization_df.to_excel('./debug/normalization_df.xlsx', index=False)

        # Make miss data
        miss_data_load = MissData(load_dir=True)
        miss_normalization_df = miss_data_load.make_missdata( 
            data_x=normalization_df, 
            missrate=miss_rate
        )

        # Just debug, remove code after complete development
        print('[debug] miss_normalization_df = ', miss_normalization_df)
        if debug_mode == True:
            miss_normalization_df.to_excel('./debug/miss_normalization_df.xlsx', index=False)

        # Division train data set, test data set
        output_df_70, output_df_30 = preprocess.splitDf(miss_normalization_df)

        # Just debug, remove code after complete development
        print('[debug] output_df_70 = ', output_df_70)
        print('[debug] output_df_30 = ', output_df_30)
        if debug_mode == True:
            output_df_70.to_excel('./debug/output_df_70.xlsx', index=False)
            output_df_30.to_excel('./debug/output_df_30.xlsx', index=False)

        # Processing division data
        data_list = []
        output_df_list = [output_df_70, output_df_30]
        idx = 0
        for output_df in output_df_list:
            # Discard data for reshape
            discard_df = preprocess.getDiscardDf(output_df)

            # Convert dataframe to numpy
            discard_np = preprocess.getNp(discard_df)

            # Numpy object reshape
            reshape_np = preprocess.getReshapeNp(discard_np)

            # Load data and introduce missingness
            ori_data_x, miss_data_x, data_m = data_loader(reshape_np, miss_rate)

            # Save the return data of data loader 
            data = { 
                'ori_data_x': ori_data_x, 
                'miss_data_x': reshape_np, 
                'data_m': data_m, 
                'M': M, 
                'S': S 
            }
            data_list.append(data)
            idx += 1

        # init data variable
        train_data = data_list[0]
        test_data = data_list[1]

        # core logic
        imputed_data_x = gain(
            train_data = train_data['miss_data_x'], 
            test_data = test_data['miss_data_x'], 
            gain_parameters = gain_parameters
        )

        # calculate rmse
        rmse = rmse_loss(
            test_data['ori_data_x'], 
            imputed_data_x, 
            test_data['data_m']
        )

        # Just debug, remove code after complete development
        print('[debug] rmse = ', rmse)
        print('[debug] rmse (round) = ', round(rmse, 4))

        return False
    else:
        output_df = preprocess.getRawDataFrame()

        # Just debug, remove code after complete development
        print('[debug] output_df = ', output_df)
        if debug_mode == True:
            output_df.to_excel('./debug/output_df.xlsx', index=False)

        # Standard normal distribution normalization
        normalization_df = preprocess.normalization(output_df, M, S)
        print('[debug] normalization_df = ', normalization_df)
        if debug_mode == True:
            normalization_df.to_excel('./debug/normalization_df.xlsx', index=False)

        # Convert dataframe to numpy
        normalization_np = preprocess.getNp(normalization_df)

        # Numpy object reshape
        reshape_np = preprocess.getReshapeNp(normalization_np)

        # Load data and introduce missingness
        ori_data_x, miss_data_x, data_m = data_loader(reshape_np, miss_rate)

        # Save the return data of data loader 
        test_data = {
            'ori_data_x': ori_data_x, 
            'miss_data_x': reshape_np, 
            'data_m': data_m, 
            'M': M, 
            'S': S 
        }

        imputed_data_x = gain(
            train_data = None, 
            test_data = test_data['miss_data_x'], 
            gain_parameters = gain_parameters
        )
        
    # Just debug, remove code after complete development
    print('[debug] imputed_data_x.shape = ', imputed_data_x.shape)

    # reverse reshape
    imputed_data = preprocess.reverseReShape(imputed_data_x)

    # Just debug, remove code after complete development
    print('[debug] imputed_data.shape = ', imputed_data.shape)

    # Make result dataframe
    imputed_df = pd.DataFrame(data=imputed_data)
    imputed_df.columns = preprocess.getTargetName()

    # Just debug, remove code after complete development
    print('[debug] imputed_df = ', imputed_df)
    if debug_mode == True:
        imputed_df.to_excel('./debug/imputed_df.xlsx', index=False)

    # denormalization
    result_df = preprocess.denormalization(imputed_df, test_data['M'], test_data['S'])
    print('[debug] result_df = ', result_df)
    if debug_mode == True:
        result_df.to_excel('./debug/result_df.xlsx', index=False)

    # write excel for result_df
    preprocess.addTimeFormat(result_df, './output/가평_2019.xlsx')

    # visualize debug
    if plt_show:
        # input
        origin_input = data_name
        origin_df = pd.read_excel(origin_input)
        
        # output
        result_input = './output/가평_2019.xlsx'
        result_df = pd.read_excel(result_input)

        # write png file to visual directory
        for col in result_df.columns.tolist()[1:]:
            diff_column = col
            origin_list = origin_df[diff_column].to_numpy()
            result_list = result_df[diff_column].to_numpy()
            idx_list = list(range(0, len(result_list)))
            imputed = result_list.copy()
            imputed[np.isnan(origin_list)==False] = np.nan
            plt.figure()
            plt.plot(idx_list, origin_list, 'b')
            plt.plot(idx_list, imputed, 'r')
            plt.savefig('./visual/{}.png'.format(col))


# usage example
# python debug.py
# check variable "parameters_file"
if __name__ == '__main__':
    parameters_dir = './parameters'

    # 파일에 대한 학습
    # parameters_file = 'train_file.json'

    # 파일에 대한 테스트
    # parameters_file = 'test_file.json'

    # 디렉토리에 대한 학습
    # parameters_file = 'train_dir.json'

    # 디렉토리에 대한 테스트
    # parameters_file = 'test_dir.json'

    # parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)
    # with open(parameters_path, encoding='utf8') as json_file:
    #     parameters = json.load(json_file)
  
    # main(parameters)

    # 파라미터 수정 테스트
    # parameters_file = 'train_file.json'
    parameters_file = 'train_dir.json'
    parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)
    with open(parameters_path, encoding='utf8') as json_file:
        parameters = json.load(json_file)
  
    main(parameters)

    parameters_file = 'test_file.json'
    parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)
    with open(parameters_path, encoding='utf8') as json_file:
        parameters = json.load(json_file)
  
    main(parameters)
    

