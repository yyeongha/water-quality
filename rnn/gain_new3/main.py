

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

from datetime import datetime

import os
import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import time




pd.set_option('display.max_columns', 1000)

interpolation_option_han = [False, True, False,
                        True, True, False,
                        False, True, True
                        ]

interpolation_option_nak = [False, True, False,
                        False, False, False,
                        True, True , True
                        ]

interpolation_option = interpolation_option_han
#han = [':,26:31', ':,28:44', ':,26:29', ':,29:31', ':,28:50', ':,27:38']
han = [':,[26,27,28,29,30]', ':,[28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44]', ':,[26,27,28]',
       ':,[28,29]', ':,[27,28,30,31,32,33,35]', ':,[26]',
       ':,[26]',':,28:45', ':,[27,28,29,30,31,33]'
       ]

nak = [':,[26,27,28,29,30]', ':,[28,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44]', ':,[27]',
       ':,[26,27]', ':,[26]', ':,[26]',
       ':,28:45', ':,[27,28,29,30,31,33]', ':,[30]' #':,[27,28,30,31,32,33,35]'
       ]

#한강 조류 데이터 없음

#gang = 'han/'
gang_han = 'han/'
folder_han = [gang_han+'자동/', gang_han+'수질/', gang_han+'AWS/',
          gang_han+'방사성/', gang_han+'TMS/', gang_han+'유량/',
          gang_han+'수위/', gang_han+'총량/', gang_han+'퇴적물/']

gang_nak = 'nak/'
folder_nak = [gang_nak+'자동/', gang_nak+'수질/', gang_nak+'ASOS/',
           gang_nak+'보/', gang_nak+'유량/', gang_nak+'수위/',
              gang_nak+'총량/', gang_nak+'퇴적물/', gang_nak+'TMS/']

#del_folder_han = [
        # 현재 [4] 번 TMS 먼가이상해서 확인해야함
        #run_num = [6] # 수위 시간 처리해야함 아직안함
 #   folder_han+'댐/', # 데이터 없음
#    folder_han+'조류/', # 시간도 맞지 않을 뿐더러 데이터 없음  ERROR  시계열값이 엑셀에서 년만 존재하는데 이것이 읽어드리는데 읽지 못하고 1970년을 뱉어냄
#]

#del_folder_nak = [
#    folder_nak+'방사성/', #시간 포멧 안맞음
#    folder_nak+'조류/', # 시간도 맞지 않을 뿐더러 데이터 없음  ERROR  시계열값이 엑셀에서 년만 존재하는데 이것이 읽어드리는데 읽지 못하고 1970년을 뱉어냄
#]

run_num_han = [0, 1, 2, 3, 4, 5, 6, 7, 8]
run_num_nak = [0, 1, 2, 3, 4, 5, 6, 7 , 8]


file_names_han = [
    [  # 자동 0
        ['가평_2016.xlsx', '가평_2017.xlsx', '가평_2018.xlsx', '가평_2019.xlsx'],
        #['서상_2016.xlsx', '서상_2017.xlsx', '서상_2018.xlsx', '서상_2019.xlsx'],
        #['의암호_2016.xlsx', '의암호_2017.xlsx', '의암호_2018.xlsx', '의암호_2019.xlsx'],
    ],
    [  # 수질 1
        ['가평천3_2016.xlsx', '가평천3_2017.xlsx', '가평천3_2018.xlsx', '가평천3_2019.xlsx'],
        #['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
    ],
    [  # AWS 2
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
        #['청평_2016.xlsx', '청평_2017.xlsx', '청평_2018.xlsx', '청평_2019.xlsx'],
    ],
    [  # 방사성  ############################################## -- 3
        ['의암댐_2016.xlsx', '의암댐_2017.xlsx', '의암댐_2018.xlsx', '의암댐_2019.xlsx'],
    ],
    [  # TMS 4
        ['가평신천하수_2016.xlsx', '가평신천하수_2017.xlsx', '가평신천하수_2018.xlsx', '가평신천하수_2019.xlsx'],
        #['가평청평하수_2016.xlsx', '가평청평하수_2017.xlsx', '가평청평하수_2018.xlsx', '가평청평하수_2019.xlsx'],
    ],
    [  #  유량 5
        ['가평군(가평교)_2016.xlsx', '가평군(가평교)_2017.xlsx', '가평군(가평교)_2018.xlsx', '가평군(가평교)_2019.xlsx']
    ],
    [  #  수위 6
        ['가평군(가평교)_2016.xlsx', '가평군(가평교)_2017.xlsx', '가평군(가평교)_2018.xlsx', '가평군(가평교)_2019.xlsx']
    ],
    [  #  총량 7
        ['조종천3_2016.xlsx', '조종천3_2017.xlsx', '조종천3_2018.xlsx', '조종천3_2019.xlsx']
    ],
    [  #  퇴적물 8
        ['의암댐2_2016.xlsx', '의암댐2_2017.xlsx', '의암댐2_2018.xlsx', '의암댐2_2019.xlsx']
    ],
]



file_names_temp = [
    [  # 자동 0
        ['[0601-0612]가평_2019.xlsx'],
    ],
    [  # 수질 1
        ['[0601-0612]가평천3_2019.xlsx'],
    ],
    [  # AWS 2
        ['[0601-0612]남이섬_2019.xlsx'],
    ],
    [  # 방사성  ############################################## -- 3
        ['[0601-0612]의암댐_2019.xlsx'],
    ],
    [  # TMS 4
        ['[0601-0612]가평신천하수_2019.xlsx'],
    ],
    [  #  유량 5
        ['[0601-0612]가평군(가평교)_2019.xlsx']
    ],
    [  #  수위 6
        ['[0601-0612]가평군(가평교)_2019.xlsx']
    ],
    [  #  총량 7
        ['[0601-0612]조종천3_2019.xlsx']
    ],
    [  #  퇴적물 8
        ['[0601-0612]의암댐2_2019.xlsx']
    ],
]


file_names_nak = [
    [  # 자동 0
        ['도개_2016.xlsx', '도개_2017.xlsx', '도개_2018.xlsx', '도개_2019.xlsx'],
    ],
    [  # 수질 1
        ['상주2_2016.xlsx', '상주2_2017.xlsx', '상주2_2018.xlsx', '상주2_2019.xlsx'],
    ],
    [  # ASOS 2
        ['구미_2016.xlsx', '구미_2017.xlsx', '구미_2018.xlsx', '구미_2019.xlsx'],
    ],
    [  # 보 3
        ['구미보_2016.xlsx', '구미보_2017.xlsx', '구미보_2018.xlsx', '구미보_2019.xlsx'],
    ],
    [  #  유량 4
        ['병성_2016.xlsx', '병성_2017.xlsx', '병성_2018.xlsx', '병성_2019.xlsx']
    ],
    [  #  수위 5
        ['상주시(병성교)_2016.xlsx', '상주시(병성교)_2017.xlsx', '상주시(병성교)_2018.xlsx', '상주시(병성교)_2019.xlsx']
    ],
    [  #  총량 6
        ['병성천-1_2016.xlsx', '병성천-1_2017.xlsx', '병성천-1_2018.xlsx', '병성천-1_2019.xlsx']
    ],
    [  #  퇴적물 7
        ['낙단_2016.xlsx', '낙단_2017.xlsx', '낙단_2018.xlsx', '낙단_2019.xlsx']
    ],
    [   # TMS
        ['상주하수_2016.xlsx', '상주하수_2017.xlsx', '상주하수_2018.xlsx', '상주하수_2019.xlsx']
    ],
]


__GAIN_TRAINING__ = True
#__GAIN_TRAINING__ = True
#__RNN_TRAINING__ = True
__RNN_TRAINING__ = True

target_col = 'do_value'
gain_epochs = 15
rnn_out_steps = 24*5
rnn_epochs = 17

#watershed = '한강_12days_test'
watershed = '한강'

if watershed == '한강':
    folder = folder_han
    interpolation_option = interpolation_option_han
    iloc_val = han
    run_num = run_num_han
    file_names = file_names_han
if watershed == '낙동강':
    folder = folder_nak
    interpolation_option = interpolation_option_nak
    iloc_val = nak
    run_num = run_num_nak
    file_names = file_names_nak
if watershed == '한강_12days_test':
    folder = folder_han
    interpolation_option = interpolation_option_han
    iloc_val = han
    run_num = run_num_han
    file_names = file_names_temp


#iloc_val = han
#run_num = [3]
#run_num = [0, 1, 2, 3, 4, 5]
#run_num_nak = [0, 6]
#run_num_nak = [2]





real_df_all = pd.DataFrame([])

#start = time.time()

#df = []

target_std = 0
target_mean = 0
target_all = 0

time_array = 0

for i in range(len(run_num)):

    idx = run_num[i]

    print('interpol flag : ', interpolation_option[idx])
    print('folder : ', folder[idx])
    print('colum_idx : ', iloc_val[idx])
    print('file_names[idx] : ', file_names[idx])

    #start = time.time()

    if watershed == '한강_12days_test':
        df, times = make_dataframe_temp_12days(folder[idx], file_names[idx], iloc_val[idx], interpolate=interpolation_option[idx])
    else:
        df, times = make_dataframe(folder[idx], file_names[idx], iloc_val[idx], interpolate=interpolation_option[idx])


    if i == 0:
        dfff = df
        time_array = times

   # print('11123123123412414')
    #print(times.shape)
    #print(times)

    print('-------df[0].shape-------- : ', df[0].shape)

    #df[0].iloc[:,0].to_excel('testset.xlsx')

    #start = time.time()
    df_all, train_mean, train_std, df = normalize(df)
    if i == 0:
        target_all = df_all
        target_std = train_std
        target_mean = train_mean

    if interpolation_option[idx] == False:

        loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

        gain_calc_falg = True

        if __GAIN_TRAINING__ == True:
            #print('pd.concat(df, axis=0).to_numpy().shape')
            #print(pd.concat(df,axis=0).to_numpy().shape)
            #norm_df = pd.concat(df, axis=0)
            #norm_data = norm_df.to_numpy()
            #MissData.save(norm_data, max_tseq=24, save_dir='save/')

            gain_calc_falg = MissData.save(pd.concat(df, axis=0).to_numpy(), max_tseq=24, save_dir='save/')
            print(folder[idx], ': training ', 'Miss date save : ', gain_calc_falg)
            #MissData.save(df.to_numpy(), max_tseq=24, save_dir='save/')
        else:
            for file in loadfiles:
                if os.path.isfile('save/' + folder[idx]+file):
                    shutil.copyfile('save/' + folder[idx]+file, 'save/'+file)
                    print('load file name : save/' + folder[idx]+file)
                else:
                    if file == 'miss.npy':
                        gain_calc_falg = MissData.save(pd.concat(df, axis=0).to_numpy(), max_tseq=24, save_dir='save/')
                        print(folder[idx], ': is not miss.npy ', 'Miss date save : ', gain_calc_falg)
                  #  print('"',folder[i]+file,'" file is not')
                    #exit(1)

        if gain_calc_falg == True:
            print('GainWindowGenerator in main')
            WindowGenerator.make_dataset = make_dataset_gain
            wide_window = WindowGenerator(input_width=24 * 5, label_width=24 * 5, shift=0, train_df = df_all, val_df = df_all, test_df = df_all, df = df)
            print('model_GAIN in main')
            #print(wide_window.dg.shape[1:])
            gain = model_GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False, epochs=gain_epochs, training_flag=__GAIN_TRAINING__, window=wide_window, model_save_path='save/')
            #gain = model_GAIN(shape=(24 * 5, df[0].shape[1]), gen_sigmoid=False, epochs=100, training_flag=__GAIN_TRAINING__, window=wide_window, model_save_path='save/')

            print('file proc in main')
            if __GAIN_TRAINING__ == True:
                #dir = 'save/'+folder[i]
                if not os.path.exists('save/' + folder[idx]):
                    os.makedirs('save/' + folder[idx])
                for file in loadfiles:
                    shutil.copyfile('save/' + file, 'save/' + folder[idx] + file)

            print('create_dataset_with_gain in main')
            ori, gan = create_dataset_with_gain(gain=gain, window=wide_window, df=df)


            #times_t = create_dataset_interpol(window=24*5, df=times)
            #print('time_t : ', times)

        else:
            gan = create_dataset_interpol(window=24*5, df=df)
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

        print(real_df_all.shape)

        #print('--------filena-----------gan.head()----------------')
        #print(gan.head())


    #print(i, " : ")
   # print(real_df_all)
    #print(i , ' --- data sets shape : ', real_df_all.shape)

#print( time.time() - start)
#exit(1000)


#2021-01-08 인풋 날자로 하여 정상작동함을 확인
#time_array = pd.Series(range(time_array.shape[0]), index=time_array)
#time_array = time_array['2019-12-01':'2019-12-07']
#real_df_all = real_df_all.iloc[time_array,:]

print('------------predic real_df_all.shape : ', real_df_all)


#train_df, val_df, test_df, test_df2 = dataset_slice(real_df_all, 0.8, 0.1, 0.1)

train_df = val_df = test_df = test_df2 = pd.DataFrame(real_df_all)
#train_df.shape = val_df = test_df = (1, 94)

#train_df, val_df, test_df = dataset_slice(real_df_all, 0, 0, 1)
#train_df, val_df, test_df = dataset_slice(real_df_all, 0.8, 0.1, 0.1)
#train_ori_df, val_ori_df, test_ori_df = dataset_slice(ori, 0.7)


print('-------------------prediction')
print('-------------------prediction')
print('-------------------prediction')


print('real_df_all.type : ', type(real_df_all))
print('train_df.type : ', type(train_df))
print('train_df.shape : ', train_df.shape, 'val_df.shape : ', val_df.shape, 'test_df.shape:' ,test_df.shape)
#print(train_df.head())

#exit(1)


label_columns_indices = {name: i for i, name in enumerate(dfff[0] )}

print("label_columns_indices:")
print(label_columns_indices)

#target_col = 'iem_mesrin_splore_qy'

#target_col = 'chlorophylla'

print('target columns : ', target_col)
num_features = dfff[0].shape[1]

target_col_idx = label_columns_indices[target_col]
# target_col_idx
out_features = [target_col_idx]
out_num_features = len(out_features)

print("target_col_idx : ", target_col_idx)
print('out_num_features : ', out_num_features)

#OUT_STEPS =

#MAX_EPOCHS = 15



#
# multi_linear_model = model_multi_linear(
#      window=None, OUT_STEPS=OUT_STEPS, out_num_features=out_num_features, epochs=MAX_EPOCHS,
#      training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_linear.ckpt")


#test_df = test_df.to_numpy()
#test_df = test_df.reshape(-1,test_df.shape[0],test_df.shape[1])
#yy = multi_linear_model.predict(test_df)

#print('-----------------------------------')
#print(yy.shape)
#print(yy)
#print(yy.reshape[-1,:])


WindowGenerator.make_dataset = make_dataset_water
# multi_window = WindowGenerator(
#     input_width=24*7,label_width=OUT_STEPS, shift=OUT_STEPS,
#     train_df=train_df, val_df=val_df, test_df=test_df,
#     out_features=out_features, out_num_features=out_num_features,
#     label_columns=dfff[0].columns
# )
multi_window = WindowGenerator(
    input_width=24*7,label_width=rnn_out_steps, shift=rnn_out_steps,
    train_df=train_df, val_df=val_df, test_df=test_df, test_df2=test_df2,
    out_features=out_features, out_num_features=out_num_features,
    label_columns=dfff[0].columns
)

multi_linear_model = model_multi_linear(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_linear.ckpt")
elman_model = model_elman(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/elman.ckpt")
gru_model = model_gru(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/gru.ckpt")
multi_lstm_model = model_multi_lstm(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_lstm.ckpt")
multi_conv_model = model_multi_conv(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path="save/models/multi_conv.ckpt")

multi_val_performance = {}
multi_performance = {}

a, b = multi_window.example3

print('aaaaaaaaa')
print(a)
print(a.shape)
print('bbbbbbbbb')
print(b)
print(b.shape)

val_nse = {}
val_pbias = {}
# val_nse['Linear'], val_pbias['Linear'] = multi_window.compa(
#      multi_linear_model, plot_col=out_features[0], windows=multi_window.example,
#      min_max_normailze=False, target_std=target_std, target_mean=target_mean)
# val_nse['ELMAN'], val_pbias['ELMAN'] = multi_window.compa(
#      elman_model, plot_col=out_features[0], windows=multi_window.example3,
#      min_max_normailze=False, target_std=target_std, target_mean=target_mean)
# val_nse['GRU'], val_pbias['GRU'] = multi_window.compa(
#      gru_model, plot_col=out_features[0], windows=multi_window.example3,
#      min_max_normailze=False, target_std=target_std, target_mean=target_mean)
# val_nse['LSTM'], val_pbias['LSTM'] = multi_window.compa(
#      multi_lstm_model, plot_col=out_features[0], windows=multi_window.example3,
#      min_max_normailze=False, target_std=target_std, target_mean=target_mean)
# val_nse['CONV'], val_pbias['CONV'] = multi_window.compa(
#      multi_conv_model, plot_col=out_features[0], windows=multi_window.example3,
#      min_max_normailze=False, target_std=target_std, target_mean=target_mean)

val_nse['Linear'], val_pbias['Linear'] = multi_window.compa(
     multi_linear_model, plot_col=out_features[0], windows=multi_window.example,
     min_max_normailze=False, target_std=target_std, target_mean=target_mean)
val_nse['ELMAN'], val_pbias['ELMAN'] = multi_window.compa(
     elman_model, plot_col=out_features[0], windows=multi_window.example3,
     min_max_normailze=False, target_std=target_std, target_mean=target_mean)
val_nse['GRU'], val_pbias['GRU'] = multi_window.compa(
     gru_model, plot_col=out_features[0], windows=multi_window.example3,
     min_max_normailze=False, target_std=target_std, target_mean=target_mean)
val_nse['LSTM'], val_pbias['LSTM'] = multi_window.compa(
     multi_lstm_model, plot_col=out_features[0], windows=multi_window.example3,
     min_max_normailze=False, target_std=target_std, target_mean=target_mean)
val_nse['CONV'], val_pbias['CONV'] = multi_window.compa(
     multi_conv_model, plot_col=out_features[0], windows=multi_window.example3,
     min_max_normailze=False, target_std=target_std, target_mean=target_mean)



x = np.arange(len(val_nse))
width = 0.35
plt.figure()
plt.bar(x, val_pbias.values(), 0.3, label='PBIAS' )
plt.bar(x + width, val_nse.values(), 0.3, label='NSE')
plt.xticks(x,val_nse.keys(), rotation=45)
_ = plt.legend()
plt.show()

#
# multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val.repeat(-1), steps=100)
# multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test.repeat(-1), verbose=0, steps=100)
#
# multi_val_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.val.repeat(-1), steps=100)
# multi_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)
#
# multi_val_performance['GRU'] = gru_model.evaluate(multi_window.val.repeat(-1), steps=100)
# multi_performance['GRU'] = gru_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)
#
# multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
# multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
#
# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)




#multi_window.SetStandMean(std=train_std, mean=train_mean)
#multi_window.compa3(multi_linear_model, plot_col=out_features[0])

#multi_window.plot24(multi_linear_model, plot_col=out_features[0])
#
# x = np.arange(len(multi_performance))
# width = 0.3
# metric_name = 'mean_absolute_error'
# metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')
# val_mae = [v[metric_index] for v in multi_val_performance.values()]
# test_mae = [v[metric_index] for v in multi_performance.values()]
# plt.figure()
# plt.bar(x - 0.17, val_mae, width, label='Validation')
# plt.bar(x + 0.17, test_mae, width, label='Test')
# plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
# plt.ylabel(f'MAE (average over all times and outputs)')
# _ = plt.legend()
# plt.show()