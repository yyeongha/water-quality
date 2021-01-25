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


__GAIN_TRAINING__ = False
#__GAIN_SKIP__ = False
__RNN_TRAINING__ = True
interpolation_option = [False, True, False]


han = [':,26:31', ':,28:44', ':,26:29']
iloc_val = [han[2],han[1],han[0]]

nakdong = [':,2:11', ':,2:12']

#folder = 'han/aws/'
#folder = ['han/auto/', 'han/aws/']
folder = ['han/aws/','han/수질/' ,'han/auto/']
#folder = ['han/aws/']

#inter = [ ]

file_names = [

    [  # AWS
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
        ['청평_2016.xlsx', '청평_2017.xlsx', '청평_2018.xlsx', '청평_2019.xlsx'],
    ],

    [  # 수질
        ['가평천3_2016.xlsx', '가평천3_2017.xlsx', '가평천3_2018.xlsx', '가평천3_2019.xlsx'],
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
    ],

    [ # 자동
        ['가평_2016.xlsx', '가평_2017.xlsx', '가평_2018.xlsx', '가평_2019.xlsx'],
        # ['서상_2016.xlsx','서상_2017.xlsx','서상_2018.xlsx','서상_2019.xlsx' ],
        ['의암호_2016.xlsx', '의암호_2017.xlsx', '의암호_2018.xlsx', '의암호_2019.xlsx'],
    ],


    #['화천_2016.xlsx', '화천_2017.xlsx', '화천_2018.xlsx', '화천_2019.xlsx']

]



# 0: 자동, 1: 수질, 2: AWS(ASOS),
#han = [':,2:11', ':,28:45', ':,26:29']

# 간격이 3이아니면 터짐 이유 아직 모름 안찾아봐서



real_df_all = pd.DataFrame([])

for i in range(len(folder)):

    df = make_dataframe(folder[i], file_names[i], iloc_val[i], interpolate=interpolation_option[i])
    df_all, train_mean, train_std, df = normalize(df)
    #print(((df_all - train_mean) / train_std).head())
   # print(df[1].head())



    #print("df_all.shape") #print(df_all.shape) #print(df.shape)

    if interpolation_option[i] == False:

        loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

        if __GAIN_TRAINING__ == True:
            MissData.save(pd.concat(df,axis=0).to_numpy(), max_tseq = 24, save_dir='save/')
        else:
            for file in loadfiles:
                if os.path.isfile(folder[i]+file):
                    shutil.copyfile(folder[i]+file, 'save/'+file)
                #else:
                  #  print('"',folder[i]+file,'" file is not')
                    #exit(1)

        wide_window = GainWindowGenerator(input_width=24 * 5, label_width=24 * 5, shift=0, train_df = df_all, val_df = df_all, test_df = df_all, df = df)

        gain = model_GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False, training_flag=__GAIN_TRAINING__, window=wide_window, model_save_path='save/')

        if __GAIN_TRAINING__ == True:
            for file in loadfiles:
                shutil.copyfile('save/' + file, folder[i] + file)

        ori, gan = create_dataset_with_gain(gain=gain, window=wide_window, df=df)

       # print(pd.DataFrame(gan).head())

        #exit(2)

    else:
        gan = create_dataset_interpol(window=wide_window, df=df)

    if i == 0 :
        real_df_all = pd.DataFrame(gan)
    else:  # Day sin, Day cos, Year sin, Year cos
        real_df_all = pd.concat([real_df_all, pd.DataFrame(gan[:,:-4])], axis=1, ignore_index=True)

    print('real_df_all : ', real_df_all.shape)

exit(1000)

train_df, val_df, test_df = dataset_slice(real_df_all, 0.7)
#train_ori_df, val_ori_df, test_ori_df = dataset_slice(ori, 0.7)

#print(train_df.shape, val_df.shape, test_df.shape)
#print(train_df.head())

exit(1)


label_columns_indices = {name: i for i, name in enumerate(df[0])}

print("label_columns_indices:")
print(label_columns_indices)

target_col = 'do_value'

print('target columns :')
print(target_col)
num_features = df[0].shape[1]

target_col_idx = label_columns_indices[target_col]
# target_col_idx
out_features = [target_col_idx]
out_num_features = len(out_features)

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