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
from core.predic import *
from core.models import *
from core.util import *
#from core.window import WindowGenerator, MissData, make_dataset_water, WaterDataGenerator
from core.window import *
from file_open import make_dataframe


import os
import datetime

import time


__GAIN_TRAINING__ = False
__RNN_TRAINING__ = True

pd.set_option('display.max_columns', 1000)

interpolation_option = [False, True, False, True, False]
#han = [':,26:31', ':,28:44', ':,26:29', ':,29:31', ':,28:50', ':,27:38']
han = [':,[26,27,28,29,30]', ':,[28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44]', ':,[26,27,28]', ':,[28,29]', ':,[27,28,30,31,32,33,35]']

#한강 조류 데이터 없음

gang = 'han/'
folder = [gang+'자동/', gang+'수질/', gang+'AWS/', gang+'방사성/', gang+'TMS/']

iloc_val = han

# 0:자동, 1:수질, 2:aws, 3:방사성,  tkrwp 4:조류, 4:TMS
#run_num = [3]
#run_num = [0]
run_num = [0, 1, 2, 3, 4]
#run_num = [0, 1, 2, 3, 4, 5]


file_names = [
    [  # 자동
        ['가평_2016.xlsx', '가평_2017.xlsx', '가평_2018.xlsx', '가평_2019.xlsx'],
        #['서상_2016.xlsx', '서상_2017.xlsx', '서상_2018.xlsx', '서상_2019.xlsx'],
        #['의암호_2016.xlsx', '의암호_2017.xlsx', '의암호_2018.xlsx', '의암호_2019.xlsx'],
    ],
    [  # 수질
        ['가평천3_2016.xlsx', '가평천3_2017.xlsx', '가평천3_2018.xlsx', '가평천3_2019.xlsx'],
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
    ],
    [  # AWS
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
        ['청평_2016.xlsx', '청평_2017.xlsx', '청평_2018.xlsx', '청평_2019.xlsx'],
    ],
    [  # 방사성  ############################################## --
        ['의암댐_2016.xlsx', '의암댐_2017.xlsx', '의암댐_2018.xlsx', '의암댐_2019.xlsx'],
    ],# ERROR  시계열값이 엑셀에서 년만 존재하는데 이것이 읽어드리는데 읽지 못하고 1970년을 뱉어냄
    # [  # 조류
    #     ['의암호_2016.xlsx', '의암호_2017.xlsx', '의암호_2018.xlsx', '의암호_2019.xlsx'],
    # ],
    [  # TMS
        ['가평신천하수_2016.xlsx', '가평신천하수_2017.xlsx', '가평신천하수_2018.xlsx', '가평신천하수_2019.xlsx'],
        ['가평청평하수_2016.xlsx', '가평청평하수_2017.xlsx', '가평청평하수_2018.xlsx', '가평청평하수_2019.xlsx'],
    ],
]


real_df_all = pd.DataFrame([])

#start = time.time()

#df = []

for i in range(len(run_num)):

    idx = run_num[i]

    print('interpol flag : ', interpolation_option[idx])
    print('folder : ', folder[idx])
    print('colum_idx : ', iloc_val[idx])
    print('file_names[idx] : ', file_names[idx])

    #start = time.time()

    df = make_dataframe(folder[idx], file_names[idx], iloc_val[idx], interpolate=interpolation_option[idx])
    if i == 0:
        dfff = df

    print('-------df[0].shape-------- : ', df[0].shape)

    #start = time.time()
    df_all, train_mean, train_std, df = normalize(df)

    if interpolation_option[idx] == False:

        loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

        if __GAIN_TRAINING__ == True:
            #print('pd.concat(df, axis=0).to_numpy().shape')
            #print(pd.concat(df,axis=0).to_numpy().shape)
            #norm_df = pd.concat(df, axis=0)
            #norm_data = norm_df.to_numpy()
            #MissData.save(norm_data, max_tseq=24, save_dir='save/')

            MissData.save(pd.concat(df,axis=0).to_numpy(), max_tseq = 24, save_dir='save/')

            #MissData.save(df.to_numpy(), max_tseq=24, save_dir='save/')
        else:
            for file in loadfiles:
                if os.path.isfile(folder[idx]+file):
                    shutil.copyfile(folder[idx]+file, 'save/'+file)
                else:
                    MissData.save(pd.concat(df, axis=0).to_numpy(), max_tseq=24, save_dir='save/')
                  #  print('"',folder[i]+file,'" file is not')
                    #exit(1)

        wide_window = GainWindowGenerator(input_width=24 * 5, label_width=24 * 5, shift=0, train_df = df_all, val_df = df_all, test_df = df_all, df = df)

        gain = model_GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False, training_flag=__GAIN_TRAINING__, window=wide_window, model_save_path='save/')

        if __GAIN_TRAINING__ == True:
            for file in loadfiles:
                shutil.copyfile('save/' + file, folder[idx] + file)

        ori, gan = create_dataset_with_gain(gain=gain, window=wide_window, df=df)
    else:
        gan = create_dataset_interpol(window=24*5, df=df)

    if i == 0 :
        #print('in')
        gan = pd.DataFrame(gan)
        gan = pd.DataFrame(gan).fillna(0)
        real_df_all = gan
    else:  # Day sin, Day cos, Year sin, Year cos
        #real_df_all = pd.concat([real_df_all, pd.DataFrame(gan[:,:-4])], axis=1)
        gan = pd.DataFrame(gan)

        #print('gan shape : ', gan.shape)
        #print('-------------------------gan.head()----------------')
        #print(gan.head())

        gan = pd.DataFrame(gan).fillna(0) # 이미 흘러나온 것에 대해서는 축을 없에거나 하지않고 nan에 대하여 0을 채워준다 마땅한값이 0이라서
        #gan = gan.dropna(thresh=2, axis=1 )
        real_df_all = pd.concat([real_df_all, gan], axis=1)

        print('--------filena-----------gan.head()----------------')
        print(gan.head())


    #print(i, " : ")
   # print(real_df_all)
    #print(i , ' --- data sets shape : ', real_df_all.shape)

#print( time.time() - start)
#exit(1000)

train_df, val_df, test_df = dataset_slice(real_df_all, 0.8, 0.1, 0.1)
#train_ori_df, val_ori_df, test_ori_df = dataset_slice(ori, 0.7)


print('-------------------prediction')
print('-------------------prediction')
print('-------------------prediction')


print('real_df_all.type : ', type(real_df_all))
print('train_df.type : ', type(train_df))
print('train_df.shape : ', train_df.shape, 'val_df.shape : ', val_df.shape, 'test_df.shape : test_df.shape')
#print(train_df.head())

#exit(1)


label_columns_indices = {name: i for i, name in enumerate(dfff[0])}

print("label_columns_indices:")
print(label_columns_indices)

#target_col = 'iem_mesrin_splore_qy'
target_col = 'do_value'
#target_col = 'chlorophylla'

print('target columns : ', target_col)
num_features = dfff[0].shape[1]

target_col_idx = label_columns_indices[target_col]
# target_col_idx
out_features = [target_col_idx]
out_num_features = len(out_features)

print("target_col_idx : ", target_col_idx)
print('out_num_features : ', out_num_features)

OUT_STEPS = 24*3
MAX_EPOCHS = 400
#MAX_EPOCHS = 15

multi_window = WaterWindowGenerator(
    input_width=24*7,label_width=OUT_STEPS, shift=OUT_STEPS,
    train_df=train_df, val_df=val_df, test_df=test_df,
    out_features=out_features, out_num_features=out_num_features
)

multi_linear_model = model_multi_linear(
    window=multi_window, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_linear.ckpt")

elman_model = model_elman(
    window=multi_window, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/elman.ckpt")

gru_model = model_gru(
    window=multi_window, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/gru.ckpt")

multi_lstm_model = model_multi_lstm(
    window=multi_window, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_lstm.ckpt")

multi_conv_model = model_multi_conv(
    window=multi_window, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_conv.ckpt")

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val.repeat(-1), steps=100)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test.repeat(-1), verbose=0, steps=100)

multi_val_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.val.repeat(-1), steps=100)
multi_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)

multi_val_performance['GRU'] = gru_model.evaluate(multi_window.val.repeat(-1), steps=100)
multi_performance['GRU'] = gru_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)


#multi_window.SetStandMean(std=train_std, mean=train_mean)
#multi_window.compa3(multi_linear_model, plot_col=out_features[0])

#multi_window.plot24(multi_linear_model, plot_col=out_features[0])

x = np.arange(len(multi_performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]
plt.figure()
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
plt.show()