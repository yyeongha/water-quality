import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


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
__GAIN_SKIP__ = True
__RNN_TRAINING__ = False


origin_path = 'save/dataset_origin.xlsx'
gain_path = 'save/dataset_gain.xlsx'
columns_path = 'save/columns.xlsx'

folder = 'han/aws/'

file_names = [['남이섬_2016.xlsx','남이섬_2017.xlsx','남이섬_2018.xlsx', '남이섬_2019.xlsx'],
              ['청평_2016.xlsx', '청평_2017.xlsx', '청평_2018.xlsx', '청평_2019.xlsx'],
              ['화천_2016.xlsx', '화천_2017.xlsx', '화천_2018.xlsx', '화천_2019.xlsx']]

#target_col = '총유기탄소'
# 각각의 강 마다, 측정소 별로 column을 맞춰야함
# 0: 자동, 1: 수질, 2: AWS(ASOS),
han = [':,2:11', ':,28:45', ':,26:29']
nakdong = [':,2:11', ':,2:12']
# gain_han = ['save/han/1', 'save/han/2', 'save/han/3']

# interpolation_option = True
interpolation_option = False
# gain_save_path = gain_han[2]

iloc_val = han[2]
#iloc_val = han[0]
#iloc_val



##for normal
# df = make_dataframe(file_names, iloc_val)
## for interpolation


#gin ori = 0

if __GAIN_SKIP__ == False:

    df = make_dataframe(folder, file_names, iloc_val, interpolate=interpolation_option)
    df_all, train_mean, train_std = normalize(df)

    #print(train_mean, train_std)
    #---------
    #norm_df = pd.concat(df,axis=0)
    #norm_data = norm_df.to_numpy()
    #MissData.save(norm_data, max_tseq = 24)
    # 1: 자동, 2: 수질,
    #dgen = GainDataGenerator(df)

    #train_df = val_df = test_df = df_all
    #print(df_all.columns)
    #target_col =
    #wide_window = WindowGenerator(input_width=24*5, label_width=24*5, shift=0, train_df=train_df, val_df=val_df, test_df=test_df, df=df)

    wide_window = GainWindowGenerator(input_width=24 * 5, label_width=24 * 5, shift=0, train_df = df_all, val_df = df_all, test_df = df_all, df = df)

    gain = model_GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False, training_flag=__GAIN_TRAINING__, window=wide_window)

    #val_performance = {} #performance = {}
    #val_performance['Gain'] = gain.evaluate(wide_window.val)
    #performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

    ori, gan = create_dataset_with_gain(gain=gain, window=wide_window, df=df)

    # save files
    #pd.DataFrame(ori).to_excel(origin_path, index = False)
    #pd.DataFrame(gan).to_excel(gain_path, index=False)
    #pd.DataFrame(columns=df_all.columns).to_excel(columns_path, index=False)
else:
    print('load created ori and gan files by gain')
    #df = make_dataframe(folder, file_names, iloc_val, interpolate=interpolation_option)#?컬럼때문?에 불러와야함
    df = [pd.read_excel(columns_path)]
    ori = pd.read_excel(origin_path).to_numpy()
    gan = pd.read_excel(gain_path).to_numpy()


print('----------------------------------------------')

train_df, val_df, test_df = dataset_slice(gan, 0.7)
train_ori_df, val_ori_df, test_ori_df = dataset_slice(ori, 0.7)

label_columns_indices = {name: i for i, name in enumerate(df[0])}

print("label_columns_indices:")
print(label_columns_indices)

#num_features = train_df.shape[1]

target_col = 'rn60m_value'

print('target columns :')
print(target_col)
num_features = df[0].shape[1]
#print("num_features:") #print(num_features)
#pd.DataFrame(df[0][0]).to_excel('save/columns.xlsx', index=False)

target_col_idx = label_columns_indices[target_col]
# target_col_idx
out_features = [target_col_idx]
out_num_features = len(out_features)

print('out_num_features : ', out_num_features)
#print('test_ori_df.columns:')
#print(test_ori_df.columns[1])


OUT_STEPS = 24*3
MAX_EPOCHS = 400
#MAX_EPOCHS = 15

multi_window = WaterWindowGenerator(
    input_width=24*7,label_width=OUT_STEPS, shift=OUT_STEPS,
    train_df=train_df, val_df=val_df, test_df=test_df,
    out_features=out_features, out_num_features=out_num_features
)

multi_val_performance = {}
multi_performance = {}

#last_baseline = MultiStepLastBaseline(OUT_STEPS=OUT_STEPS, out_features=out_features)
#last_baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
#multi_val_performance['BaseLine'] = last_baseline.evaluate(multi_window.val.repeat(-1), steps=100)
#multi_performance['BaseLine'] = last_baseline.evaluate(multi_window.test.repeat(-1), verbose=0, steps=100)


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


#x = np.arange(len(multi_performance))
#width = 0.3
#metric_name = 'mean_absolute_error'
#metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')
#val_mae = [v[metric_index] for v in multi_val_performance.values()]
#test_mae = [v[metric_index] for v in multi_performance.values()]
#plt.figure()
#plt.bar(x - 0.17, val_mae, width, label='Validation')
#plt.bar(x + 0.17, test_mae, width, label='Test')
#plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
#plt.ylabel(f'MAE (average over all times and outputs)')
#_ = plt.legend()
#plt.show()