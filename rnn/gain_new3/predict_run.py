# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import shutil

from core.gain import *
from core.rnn_predic import *
from core.models import *
from core.util import *
#from core.window import WindowGenerator, MissData, make_dataset_water, WaterDataGenerator
from core.window import WindowGenerator, make_dataset_gain, make_dataset_water
from file_open import make_dataframe, make_dataframe_temp_12days
from core.miss_data import MissData
import json

import datetime


def hour_to_day_mean(array):
    time = 24
    array = array.reshape((array.shape[0], array.shape[1] // time, time, array.shape[2]))
    array = array.mean(2)
    return array

class prediction_for_webpage():


    model_prediction_path = ["save/han/models/",""]

    dataframe_all_data_2016_2019 = None



    def __init__(self):
        self.gain_train = False
        self.rnn_train = False
        #self.mais_path = 'save/'

        self.base_path = 'save/'

        self.watershed_path = ['han/', 'nak/']

        self.options = [
            #[3,  9, 2, 6, 4, 8,  5, 6, 2], #최과장님이준거 방사성 2개로들어감 직접컬럼날림
            [3, 9, 1, 6, 4, 8, 5, 6, 2],
            [8+4, 16+4, 1+4, 7+4, 1+4, 1+4, 17+4, 6+4, 5+4],
            ["자동/", "수질/", "방사성/", "TMS/", "유량/", "수위/", "총량/", "퇴적물/", "조류/"],
            [False, True, True, True, False, False, True, True, True],
        ]

        #                         2                   4          5       6         7
        #tmpr_value, ph_value, do_value, ec_value, toc_value, 총질소_값, 총인_값, 클로로필-a_값

        self.target_column_index = [2, 4, 5, 6, 7]
        self.target_model_path = ["do/", "toc/", "nitrogen/", "phosphorus/", "chlorophyll-a/"]



        self.push_checker = 0

        sum = 0
        for i in range(len(self.options[0])):
            column_num = self.options[0][i]
            measurement_num = self.options[1][i]
            sum += measurement_num*column_num


        print('sumsumsumsumsumsumsumsumsumsum')
        print(sum)

        self.loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

    def dataframe_concat(self, df1, df2):
        #df2 = pd.concat([df2])
        df2 = df2.reset_index(drop=True)
        if self.push_checker == 0:
            df1 = df2
            self.push_checker += 1
        else:  # Day sin, Day cos, Year sin, Year cos
            df1 = pd.concat([df1, df2], axis=1)
            self.push_checker += 1
        return df1

    def run(self, dataframe=dataframe_all_data_2016_2019, target = 0, watershed = 0):
        print(dataframe.shape)
        self.push_checker = 0
        real_df_all = pd.DataFrame([])

        target_index = self.target_column_index[target]

        target_data = dataframe.iloc[:,target_index+1:target_index+2]
        target_columns = target_data.columns



        target_mean = target_data.mean()
        target_std = target_data.std()
        print("예측항목 : ", target_columns)

        target_mean = target_mean.to_numpy()
        target_std = target_std.to_numpy()




        #target_data = dataframe.iloc[:,1:self.options[1][0]+1]
        #target_columns = target_data.columns


        #target_mean

        # normalize
        #df_all, target_mean, target_std, df = normalize(dataframe)

        date = dataframe.iloc[:,:1]
        data = dataframe.iloc[:,1:]

        target_name = self.target_model_path[target]
        base_path = self.base_path + self.watershed_path[watershed]

        for i in range(len(self.options[0])):
            column_num = self.options[0][i]
            measurement_num = self.options[1][i]
            measerement_name = self.options[2][i]
            gain_skip = self.options[3][i]

            #print(column_num)
            #print(measurement_num)
            #print(measerement_name)
            #print(gain_skip)

            df = []
            mean = []
            std = []

            for j in range(column_num):
                df_tmp = data.iloc[:,:measurement_num]
                data = data.iloc[:,measurement_num:]

                df.append(df_tmp)

                if gain_skip == False:

                    for file in self.loadfiles:
                        if os.path.isfile(base_path + measerement_name + file):
                            shutil.copyfile(base_path + measerement_name + file, 'save/' + file)
                        else:
                            print('can not load file name : save/' + measerement_name + file)

                    gain = model_GAIN(shape=(120, measurement_num), gen_sigmoid=False, model_save_path='save/')

                    _, gan = create_dataset_with_gain(gain=gain, window=None, shape=(120, measurement_num),df=[df_tmp])

                    real_df_all = self.dataframe_concat(real_df_all, pd.DataFrame(gan))
                else:
                    real_df_all = self.dataframe_concat(real_df_all, df_tmp)


        print(real_df_all)

        train_df, val_df, test_df, test_df2 = dataset_slice(real_df_all, 0.8, 0.1, 0.1)
        model_path = "save/" + self.watershed_path[watershed] + "models/" + self.target_model_path[target] + "gru.ckpt"
        print(model_path)
        gru_model = model_gru(OUT_STEPS=5*24, checkpoint_path=model_path)

        input_hour = real_df_all.iloc[:24*7,:].to_numpy()
        input_hour = input_hour.reshape((1,) + input_hour.shape)
        print(input_hour.shape)


        #label_hour = real_df_all.iloc[24 * 7:, target_index:target_index+1].to_numpy()
        label_hour = real_df_all.iloc[24 * 7 : 24 * 7 + 24 * 5, target_index:target_index + 1].to_numpy()
        label_hour = label_hour.reshape((1,) + label_hour.shape)


        pred_hour = gru_model.predict(input_hour)

        pred_hour = pred_hour * target_std + target_mean

        pred_day = hour_to_day_mean(pred_hour)

        print(pred_hour.shape)
        print(pred_day.shape)

        #train_mean
        #train_std

        print(type(label_hour))
        print(label_hour.shape)
        label_day = hour_to_day_mean(label_hour)
        print(label_day.shape)

        print(pred_day)
        print(label_day)
        print(np.abs(pred_day-label_day))


        return 1, 1, 1, 1

#
    def __call__(self):
        return self.run()
