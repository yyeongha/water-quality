# -*- coding: utf-8 -*-

import shutil

from gain_new.core.gain import *
from gain_new.core.rnn_predic import *
from gain_new.core.util import *
from django.conf import settings


def hour_to_day_mean(array):
    time = 24
    array = array.reshape((array.shape[0], array.shape[1] // time, time, array.shape[2]))
    array = array.mean(2)
    return array


class prediction_for_webpage():

    def __init__(self):
        self.gain_train = False
        self.rnn_train = False
        self.base_path = settings.PREDICT_ROOT + '/save/'

        self.watershed_path = ['han/', 'nak/', 'geum/', 'yeong/']

        self.options = [
            [
                [3],
                [8 + 4, 6 + 4, 7 + 4],
                ["자동/", "수질/", "총량/"],
                [False, True, True],
            ],
            [
                [3, 3, 3, 2],
                [8 + 4, 7 + 4, 6 + 4, 3 + 4],
                ["자동/", "총량/", "수질/", "조류/"],
                [False, True, True, True],
            ],
            [
                [2],
                [8 + 4, 7 + 4, 6 + 4],
                ["자동/", "총량/", "수질/"],
                [False, True, True],
            ],
            [
                [3],
                [8 + 4, 7 + 4, 6 + 4, 3 + 4],
                ["자동/", "총량/", "수질/", "조류/"],
                [False, True, True, True],
            ]
        ]

        # tmpr_value, ph_value, do_value, ec_value, toc_value, 총질소_값, 총인_값, 클로로필-a_값

        self.target_column_index = [2, 4, 5, 6, 7]
        self.target_model_path = ["do/", "toc/", "nitrogen/", "phosphorus/", "chlorophyll-a/"]

        self.push_checker = 0

        self.loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

    def dataframe_concat(self, df1, df2):
        df2 = df2.reset_index(drop=True)
        if self.push_checker == 0:
            df1 = df2
            self.push_checker += 1
        else:
            df1 = pd.concat([df1, df2], axis=1)
            self.push_checker += 1
        return df1

    def run(self, dataframe=None, target=0, watershed=0):
        self.push_checker = 0
        real_df_all = pd.DataFrame([])

        target_index = self.target_column_index[target]

        target_data = dataframe.iloc[:, target_index + 1:target_index + 2]
        target_columns = target_data.columns

        target_mean = target_data.mean()
        target_std = target_data.std()

        target_mean = target_mean.to_numpy()
        target_std = target_std.to_numpy()

        date = dataframe.iloc[:, :1]
        data = dataframe.iloc[:, 1:]

        target_name = self.target_model_path[target]
        base_path = self.base_path + self.watershed_path[watershed]

        for i in range(len(self.options[watershed][0])):
            column_num = self.options[watershed][0][i]
            measurement_num = self.options[watershed][1][i]
            measerement_name = self.options[watershed][2][i]
            gain_skip = self.options[watershed][3][i]

            mean = []
            std = []

            for j in range(column_num):
                df_tmp = data.iloc[:, :measurement_num]
                data = data.iloc[:, measurement_num:]

                mean.append(df_tmp.mean())
                std.append(df_tmp.std())

                df_tmp = (df_tmp - mean[j]) / std[j]

                if gain_skip == False:

                    for file in self.loadfiles:
                        if os.path.isfile(base_path + measerement_name + file):
                            shutil.copyfile(base_path + measerement_name + file, self.base_path + file)
                        else:
                            print('can not load file name : ' + base_path + measerement_name + file)

                    gain = model_GAIN(shape=(120, measurement_num), gen_sigmoid=False, model_save_path=self.base_path)

                    _, gan = create_dataset_with_gain(gain=gain, window=None, shape=(120, measurement_num), df=[df_tmp])

                    real_df_all = self.dataframe_concat(real_df_all, pd.DataFrame(gan))
                else:
                    real_df_all = self.dataframe_concat(real_df_all, df_tmp)

        model_path = base_path + "models/" + self.target_model_path[target] + "gru.ckpt"
        gru_model = model_gru(OUT_STEPS=5 * 24, checkpoint_path=model_path)

        input_hour = real_df_all.iloc[:24 * 10, :].to_numpy()
        input_hour = input_hour.reshape((1,) + input_hour.shape)

        label_hour = real_df_all.iloc[24 * 10: 24 * 10 + 24 * 5, target_index:target_index + 1].to_numpy()
        label_hour = label_hour.reshape((1,) + label_hour.shape)
        label_hour = label_hour * target_std + target_mean
        label_day = hour_to_day_mean(label_hour)

        pred_hour = gru_model.predict(input_hour)
        pred_hour = pred_hour * target_std + target_mean
        pred_day = hour_to_day_mean(pred_hour)

        input_hour = input_hour[:, :, target_index:target_index + 1]
        input_hour = input_hour * target_std + target_mean
        input_day = hour_to_day_mean(input_hour)

        nse_sum1 = 0
        nse_sum2 = 0
        pbias_sum1 = 0
        pbias_sum2 = 0

        o1 = np.mean(label_day)

        length = pred_day.shape[1]
        for n in range(length):
            o = label_day[0, n, 0]
            p = pred_day[0, n, 0]

            temp_m = o - p

            nse_sum1 += temp_m ** 2
            nse_sum2 += (o - o1) ** 2

            pbias_sum1 += temp_m
            pbias_sum2 += o

        nse = 1 - (nse_sum1 / nse_sum2)
        pbias = (pbias_sum1 / pbias_sum2) * 100

        return nse, np.abs(pbias), input_day.reshape(-1), label_day.reshape(-1), pred_day.reshape(-1)

    def __call__(self):
        return self.run()
