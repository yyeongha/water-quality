from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import json
import numpy as np
import pandas as pd

# from gain import gain
from origin_gain_custom import gain
from utils import rmse_loss, data_loader, init_preprocess, getUseTrain


# usage example
# python debug.py
# check your file named parameters.json
def main (parameters):
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
        'batch_size': parameters['batch_size'],
        'hint_rate': parameters['hint_rate'],
        'alpha': parameters['alpha'],
        'iterations': parameters['iterations'],
        'dir_name': parameters['data_name'].split('/')[-1]
    }

    # Make Parameter Dictionary for Pre-Process
    preprocess_parameters = {
        'input': parameters['data_name'],
        'target': list(map(int, parameters['target'].split(','))),
        'time': parameters['time'],
        'target_all': parameters['target_all'],
        'fill_cnt': parameters['fill_cnt']
    }

    # Return status of use train flag
    useTrain = getUseTrain(parameters) 

    print('[debug] gain_parameters = ', gain_parameters)
    print('[debug] preprocess_parameters = ', preprocess_parameters)
    print('[debug] useTrain = ', useTrain)

    # Init Pre-Process and Make Traing Data
    preprocess = init_preprocess(preprocess_parameters)
    # output_df = preprocess.getDataFrame()

    start_time = time.time() # time check
    output_df = preprocess.getDataFrame()
    
    # Debug
    print('output_df => ', output_df)
    # output_df.to_excel('./output/merge.xlsx', index=False)
    print("---{}s seconds---".format(time.time()-start_time))
    # exit(0)

    # temp
    before_shift_df = pd.read_excel('./output/before_shift.xlsx')
    M, S = preprocess.getMeanAndStand(before_shift_df)
    print('[debug] M = ', M)
    print('[debug] S = ', S)

    # Read the raw excel, Not use miss_data_x created data_loader (차후 디렉토리 검토)
    if useTrain:

        # print('[debug] $$ output_df = ', output_df)
        normalization_df = preprocess.normalization(output_df, M, S)
        # print('[debug] $$ normalization_df = ', normalization_df)


        # debug
        # normalization_df.to_excel('./output/before_random.xlsx', index=False)

        # Division train data set, test data set
        output_df_70, output_df_30 = preprocess.splitDf(normalization_df)

        # Debug
        # output_df_70.to_excel('./output/70.xlsx', index=False)
        # output_df_30.to_excel('./output/30.xlsx', index=False)

        # Processing division data
        data_list = []
        output_df_list = [output_df_70, output_df_30]
        idx = 0
        for output_df in output_df_list:
            # Standard normal distribution normalization
            # normalization_df, M, S = preprocess.normalization(output_df)

            # Discard data for reshape
            normalization_df = preprocess.getDiscardDf(normalization_df)

            # if idx == 0:
            #     tempDf2 = pd.DataFrame(normalization_df)
            #     tempDf2.to_excel('./output/normalization_df.xlsx', index=False)

            # Convert dataframe to numpy
            normalization_np = preprocess.getNp(normalization_df)

            # Numpy object reshape
            normalization_np_reshape = preprocess.getReshapeNp(normalization_np)

            # Load data and introduce missingness
            ori_data_x, miss_data_x, data_m = data_loader(normalization_np_reshape, parameters['miss_rate'])
            # if idx == 0:
            #     tempDf2 = pd.DataFrame(miss_data_x)
            #     tempDf2.to_excel('./output/miss_data_x.xlsx', index=False)

            # Save the return data of data loader 
            data = { 'ori_data_x': ori_data_x, 'miss_data_x': miss_data_x, 'data_m': data_m, 'M': M, 'S': S }
            data_list.append(data)
            idx += 1

        train_data = data_list[0]
        test_data = data_list[1]
  
        # Debug
        # preprocess.npToExcel(train_data['ori_data_x'], './output/ori_data_x_70.xlsx')
        # preprocess.npToExcel(train_data['miss_data_x'], './output/miss_data_x_70.xlsx')
        # preprocess.npToExcel(train_data['data_m'], './output/data_m_70.xlsx')
        # preprocess.npToExcel(test_data['ori_data_x'], './output/ori_data_x_30.xlsx')
        # preprocess.npToExcel(test_data['miss_data_x'], './output/miss_data_x_30.xlsx')
        # preprocess.npToExcel(test_data['data_m'], './output/data_m_30.xlsx')

        imputed_data_x = gain(
            train_data = train_data['miss_data_x'], 
            test_data = test_data['miss_data_x'], 
            gain_parameters = gain_parameters
        )

        # break point
        # exit(0)

        rmse = rmse_loss(
            test_data['ori_data_x'], 
            imputed_data_x, 
            test_data['data_m']
        )
        print('[debug] rmse = ', round(rmse, 4))
    else:
        output_df = preprocess.getRawDataFrame()
        print('[debug] output_df = ', output_df)

        # Debug
        # tempDf3 = pd.DataFrame(normalization_np)
        output_df.to_excel('./output/output_df.xlsx', index=False)

        # Standard normal distribution normalization
        normalization_df = preprocess.normalization(output_df, M, S)
        print('[debug] normalization_df = ', normalization_df)
        normalization_df.to_excel('./output/normalization_df.xlsx', index=False)

        # temp
        x_df = preprocess.denormalization(normalization_df, M, S)
        x_df.to_excel('./output/x_df.xlsx', index=False)

        # Discard data for reshape
        normalization_df = preprocess.getDiscardDf(normalization_df)

        # Convert dataframe to numpy
        normalization_np = preprocess.getNp(normalization_df)

        # Debug
        tempDf3 = pd.DataFrame(normalization_np)
        tempDf3.to_excel('./output/normalization_np.xlsx', index=False)

        # Numpy object reshape
        normalization_np_reshape = preprocess.getReshapeNp(normalization_np)

        # Debug
        tempDf = pd.DataFrame(normalization_np_reshape)
        tempDf.to_excel('./output/normalization_np_reshape.xlsx', index=False)

        # Load data and introduce missingness
        ori_data_x, miss_data_x, data_m = data_loader(normalization_np_reshape, parameters['miss_rate'])

        # Debug
        tempDf2 = pd.DataFrame(miss_data_x)
        tempDf2.to_excel('./output/miss_data_x.xlsx', index=False)

        # Save the return data of data loader 
        test_data = { 'ori_data_x': ori_data_x, 'miss_data_x': normalization_np_reshape, 'data_m': data_m, 'M': M, 'S': S }

        imputed_data_x = gain(
            train_data = None, 
            test_data = test_data['miss_data_x'], 
            gain_parameters = gain_parameters
        )
        
    # Debug
    print('imputed_data_x.shape = ', imputed_data_x.shape)

    # fix
    imputed_data_x = preprocess.reverseReShape(imputed_data_x)

    # Debug
    print('imputed_data_x.shape = ', imputed_data_x.shape)

    # Make result dataframe
    imputed_df = pd.DataFrame(data=imputed_data_x)
    imputed_df.columns = preprocess.getTargetName()
    # imputed_df.to_excel('./output/before_denormal.xlsx', index=False)

    # denormalization
    imputed_df = preprocess.denormalization(imputed_df, test_data['M'], test_data['S'])

    # Make result excel
    # imputed_df.to_excel('./output/result_reshape.xlsx', index=False)
    # preprocess.npToExcel(imputed_df, './output/가평_2019.xlsx', True)


if __name__ == '__main__':
    # parameters_path = './parameters.json'
    # parameters_path = './parameters_train.json'
    # parameters_path = './parameters_test.json'
    parameters_path = './parameters_train_dir.json'
    with open(parameters_path, encoding='utf8') as json_file:
        parameters = json.load(json_file)
  
    inject_data_name_list = [
        # './classfy/2546087781705908382', no
        # './classfy/2753029681805320370', yes
        # './classfy/1868617106871301337', yes
        # './classfy/3464694848574796523', no
        # './classfy/1530373521792743778', no
        # './classfy/6759354446250977078', no
        # './classfy/6992528307228056028', no
        # './classfy/5682481196855854459', no
        # './classfy/3054652060032968249', yes
        # './classfy/6551795653332579301', no
        # './classfy/4245234164211946156', yes
        # './classfy/8800330005351799323', no
        # './classfy/7705551329510827169', no
        # './classfy/9592735753895240', yes
        # './classfy/2053391929074247952', no
        # './classfy/9041759046289129891', no
        # './classfy/1692182177243629754', no
        # './classfy/7989656796853978982', no
        # './classfy/7684844176936782008', no
        # './classfy/1614748674163524857', no
        # './classfy/7339366309677307016', no
        # './classfy/6724235619583583672', yes
        # './classfy/355326747347266483', no
        # './classfy/6685387669464724421', no
        # './classfy/511305924109714059', no
        # './classfy/6325248669386095902', no
        # './classfy/7483783171719641308', yes
        # './classfy/6552429313030480690', yes
        # './classfy/3045107374348052150', no
        # './classfy/4214509389030386299', no
        # './classfy/1381859547695611593', no
        # './classfy/3270375920874359045', no
        # './classfy/3228431611117038571', no
        # './classfy/99488762247561991', no
        # './classfy/2124000557585122617', yes
        # './classfy/679348088764031629', no
        # './classfy/3223433592508912514', no
        # './classfy/6160567172594793923', no
        # './classfy/8546483254449251397' no
    ]
    for inject_data_name in inject_data_name_list:
        parameters['data_name'] = inject_data_name
        # Calls main function
        main(parameters)
        exit(0)
