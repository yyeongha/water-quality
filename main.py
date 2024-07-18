# -*- coding: utf-8 -*-

import shutil
from core.gain import *
from core.rnn_predic import *
from core.models import *
from core.util import *
from core.window import WindowGenerator, make_dataset_gain, make_dataset_water
from core.file_open import make_dataframe
from core.miss_data import MissData
import json
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# input parameter
data_path = 'data/'
parameters_dir = 'input'

parameters_file = 'input.json'
parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)


#parameters = json.load(parameters_path)
with open(parameters_path, encoding='utf8') as json_file:
    parameters = json.load(json_file)

gain_parameters = parameters['gain']
rnn_parameters = parameters['rnn']
file_parameters = parameters['file']

parameters_path = parameters_dir+'/'+ file_parameters['watershed'] + '.json'
with open(parameters_path, encoding='utf8') as json_file:
    parameters = json.load(json_file)

data_parameters = parameters['data']

interpolation_option = data_parameters['interpolation']
colum_idx = data_parameters['columns']
watershed = data_parameters['watershed']
file_names = data_parameters['files']
folder = data_parameters['directorys']
for i in range(len(folder)):
    folder[i] = watershed+folder[i]

__GAIN_TRAINING__ = gain_parameters['train']
gain_epochs = gain_parameters['max_epochs']
gain_in_setps = gain_parameters['input_width']
gain_out_setps = gain_parameters['label_width']
gain_batch_size = gain_parameters['batch_size']
gain_fill_no = gain_parameters['fill_width']
gain_shift = gain_parameters['shift_width']
gain_miss_rate = gain_parameters['miss_rate']

__RNN_TRAINING__ = rnn_parameters['train']
rnn_epochs = rnn_parameters['max_epochs']
rnn_in_setps = rnn_parameters['input_width']
rnn_out_steps = rnn_parameters['label_width']
rnn_batch_size = rnn_parameters['batch_size']
rnn_predict_day = rnn_parameters['predict_day']
rnn_target_column = rnn_parameters['target_column']

if rnn_predict_day < 3 or rnn_predict_day >5:
    print('predict_day err')
    exit(88)
rnn_predict_day -= 1


if file_parameters['watershed'] == 'nak':
    run_num = range(len(folder))
else:
    run_num = [0]


real_df_all = pd.DataFrame([])
target_all = target_mean = target_std = 0

gain_val_performance = {}
gain_performance = {}

ddday = 31
mmmonth = 12

length = len(run_num)
for i in range(length):

    idx = run_num[i]

    print('interpol flag : ', interpolation_option[idx])
    print('folder : ', data_path + folder[idx])
    print('colum_idx : ', colum_idx[idx])
    print('file_names[idx] : ', file_names[idx])

    df, times, mmmonth, ddday = make_dataframe(data_path+folder[idx], file_names[idx],
                               colum_idx[idx], interpolation=interpolation_option[idx],
                              first_file_no=i, month=mmmonth, day=ddday)

    df_all, train_mean, train_std, df = normalize(df)

    if i == 0:
        dfff = df
        target_all = df_all
        target_std = train_std
        target_mean = train_mean
        start_year = str(times.iloc[0].year)
        end_year = str(times.iloc[-1].year)

    if interpolation_option[idx][0] == False:

        loadfiles = ['idx.npy', 'miss.npy', 'discriminator.h5', 'generator.h5']

        gain_calc_falg = True

        if __GAIN_TRAINING__ == True:
            gain_calc_falg = MissData.save(pd.concat(df, axis=0).to_numpy(), max_tseq=24, save_dir='save/')
        else:
            for file in loadfiles:
                if os.path.isfile('save/' + folder[idx]+file):
                    shutil.copyfile('save/' + folder[idx]+file, 'save/'+file)
                else:
                    if file == 'miss.npy':
                        gain_calc_falg = MissData.save(pd.concat(df, axis=0).to_numpy(), max_tseq=24, save_dir='save/')

        if gain_calc_falg == True:
            WindowGenerator.make_dataset = make_dataset_gain
            wide_window = WindowGenerator(input_width=gain_in_setps, label_width=gain_out_setps, shift=gain_shift,
                                          fill_no=gain_fill_no, miss_rate=gain_miss_rate, batch_size=gain_batch_size,
                                          train_df = df_all, val_df = df_all, test_df = df_all, df = df)

            gain = model_GAIN(shape=(gain_in_setps, df_all.shape[1]), gen_sigmoid=False, epochs=gain_epochs,
                              training_flag=__GAIN_TRAINING__, window=wide_window, model_save_path='save/')

            gain_val_performance[str(i)] = gain.evaluate(wide_window.val)
            gain_performance[str(i)] = gain.evaluate(wide_window.test, verbose=0)

            if __GAIN_TRAINING__ == True:
                if not os.path.exists('save/' + folder[idx]):
                    os.makedirs('save/'+folder[idx])
                for file in loadfiles:
                    shutil.copyfile('save/' + file, 'save/' + folder[idx] + file)

            ori, gan = create_dataset_with_gain(gain=gain, shape=(gain_in_setps, df_all.shape[1]), df=df)

        else:
            gan = create_dataset_interpol(window=gain_in_setps, df=df)
    else:
        gan = create_dataset_interpol(window=gain_in_setps, df=df)

    if i == 0 :
        real_df_all = pd.DataFrame(gan)
    else:
        real_df_all = pd.concat([real_df_all, pd.DataFrame(gan)], axis=1)


train_df, val_df, test_df = dataset_slice(real_df_all, 0.7, 0.1, 0.2)
idx = [2, 4, 5, 6, 7]
pa = ["do/", "toc/", "nitrogen/", "phosphorus/", "chlorophyll-a/"]
indices = {name: i for i, name in enumerate(idx)}

target_dic = {"do":"do_value", "toc":"toc_value", "tn":"총질소_값", "tp":"총인_값", "chl-a":"클로로필-a_값"}
label_columns_indices = {name: i for i, name in enumerate(dfff[0])}
target_col_idx = label_columns_indices[target_dic[rnn_target_column]]
out_features = [target_col_idx]
out_num_features = len(out_features)


WindowGenerator.make_dataset = make_dataset_water
multi_window = WindowGenerator(
    input_width=rnn_in_setps,label_width=rnn_out_steps, shift=rnn_out_steps,out_features=out_features,
    out_num_features=out_num_features,label_columns=dfff[0].columns, batch_size=rnn_batch_size,
    train_df=train_df, val_df=val_df, test_df=test_df)


model_path = "save/" + watershed + "models/" + pa[indices[target_col_idx]]
if __RNN_TRAINING__:
    if not os.path.exists(model_path):
        os.makedirs(model_path)

val_nse = {}
val_pbias = {}
val_mae = {}
val_rmse = {}
val_r = {}
val_rs = {}


# multi_linear_model = model_multi_linear(
#     window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
#     training_flag=__RNN_TRAINING__, checkpoint_path=model_path+"multi_linear.ckpt", continue_train=False)
# elman_model = model_elman(
#     window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
#     training_flag=__RNN_TRAINING__, checkpoint_path=model_path+"elman.ckpt", continue_train=False)
gru_model = model_gru(
    window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
    training_flag=__RNN_TRAINING__, checkpoint_path=model_path+"gru.ckpt")
# multi_lstm_model = model_multi_lstm(
#     window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
#     training_flag=__RNN_TRAINING__, checkpoint_path=model_path+"multi_lstm.ckpt", continue_train=False)
# multi_conv_model = model_multi_conv(
#     window=multi_window, OUT_STEPS=rnn_out_steps, out_num_features=out_num_features, epochs=rnn_epochs,
#     training_flag=__RNN_TRAINING__, checkpoint_path=model_path+"multi_conv.ckpt", continue_train=False)


# val_nse['Linear'], val_pbias['Linear'], pred, label, val_mae['Linear'], val_rmse['Linear'], val_rs['Linear'], val_r['Linear'] = compa(
#     model=gru_model,df=val_df, plot_col=out_features[0], target_std=target_std, target_mean=target_mean, predict_day = rnn_predict_day)
#
# val_nse['ELMAN'], val_pbias['ELMAN'], pred, label, val_mae['ELMAN'], val_rmse['ELMAN'], val_rs['ELMAN'], val_r['ELMAN'] = compa(
#     model=gru_model,df=val_df, plot_col=out_features[0], target_std=target_std, target_mean=target_mean, predict_day = rnn_predict_day)

val_nse['GRU'], val_pbias['GRU'], pred, label, val_mae['GRU'], val_rmse['GRU'], val_rs['GRU'], val_r['GRU'] = compa(
    model=gru_model,df=test_df, plot_col=out_features[0], target_std=target_std, target_mean=target_mean, predict_day = rnn_predict_day)

# val_nse['LSTM'], val_pbias['LSTM'], pred, label, val_mae['LSTM'], val_rmse['LSTM'], val_rs['LSTM'], val_r['LSTM'] = compa(
#     model=gru_model,df=val_df, plot_col=out_features[0], target_std=target_std, target_mean=target_mean, predict_day = rnn_predict_day)
#
# val_nse['CONV'], val_pbias['CONV'], pred, label, val_mae['CONV'], val_rmse['CONV'], val_rs['CONV'], val_r['CONV'] = compa(
#     model=gru_model,df=val_df, plot_col=out_features[0], target_std=target_std, target_mean=target_mean, predict_day = rnn_predict_day)


print("watershed : ", watershed)
print("year : " + start_year + " ~ "+ end_year)
print('target : ', rnn_target_column)

print('GAIN RMSE : ', gain_performance)
print('GRU NSE : ', val_nse['GRU'])
print('GRU PBIAS : ', val_pbias['GRU'])
print('GRU MAE : ', val_mae['GRU'])
print('GRU RMSE : ', val_rmse['GRU'])
print('GRU R^2 : ', val_rs['GRU'])
