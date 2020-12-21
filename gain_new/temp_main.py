import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *

folder = 'data'
file_names = [['의암호_2016.xlsx'],['의암호_2017.xlsx'],['의암호_2018.xlsx'],['의암호_2019.xlsx'],
            ['서상_2016.xlsx'],['서상_2017.xlsx'],['서상_2018.xlsx'],['서상_2019.xlsx']]
# file_names = [['의암호_2016.xlsx'],['의암호_2017.xlsx'],['의암호_2018.xlsx'],['의암호_2019.xlsx'],
#             ['화천_2016.xlsx'],['화천_2017.xlsx'],['화천_2018.xlsx'],['화천_2019.xlsx']]
# file_names = [['의암호_2016.xlsx'],['의암호_2017.xlsx'],['의암호_2018.xlsx'],['의암호_2019.xlsx'],
#             ['가평_2016.xlsx'],['가평_2017.xlsx'],['가평_2018.xlsx'],['가평_2019.xlsx']]
# file_names = [['의암호_2017.xlsx'], ['의암호_2018.xlsx'], ['의암호_2019.xlsx'],
#               ['가평_2017.xlsx'], ['가평_2018.xlsx'], ['가평_2019.xlsx']]
# file_names = [['가평_2016.xlsx','가평_2017.xlsx','가평_2018.xlsx', '가평_2019.xlsx'], ['의암호_2016.xlsx','의암호_2017.xlsx','의암호_2018.xlsx', '의암호_2019.xlsx']]
# file_names = [['의암호_2018.xlsx']]


# file_names = [['의암호_2017.xlsx'],['의암호_2016.xlsx']]

# file_names = [['가평_2016.xlsx','가평_2017.xlsx','가평_2018.xlsx','가평_2019.xlsx'],
# ['화천_2016.xlsx','화천_2017.xlsx','화천_2018.xlsx','화천_2019.xlsx'],
# ['의암호_2016.xlsx','의암호_2017.xlsx','의암호_2018.xlsx','의암호_2019.xlsx'],
# ['서상_2016.xlsx','서상_2017.xlsx','서상_2018.xlsx','서상_2019.xlsx']]

df, df_full, df_all = createDataFrame(folder, file_names)

standardNormalization(df, df_all)

dgen = GainDataGenerator(df)

train_df = df[0]
val_df = df[0]
test_df = df[0]

wide_window = WindowGenerator(
    df=df,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=24*5,
    label_width=24*5,
    shift=0
)
wide_window.plot(plot_col='총질소')  # create dg issue

''' miss data plt '''
# plt.figure(figsize=(9,10))
# n = wide_window.dg.data_m.shape[0]
# train = n//8
# for i in range(8):
#     plt.subplot(181+i)
#     plt.imshow(wide_window.dg.data_m[i*train:(i+1)*train, 0:7], aspect='auto')
#     plt.yticks([])
# plt.show()

val_performance = {}
performance = {}

gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)
# if load_yn !=True:
MAX_EPOCHS = 1


def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=GAIN.RMSE_loss)

    history = model.fit(window.train,
                        epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS // 5)

val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)
gain.save(save_dir='save')

''' 학습 loss history 출력'''
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax2 = ax.twinx()
# ax.plot(history.history['gen_loss'], label='gen_loss')
# ax.plot(history.history['disc_loss'], label='disc_loss')
# ax2.plot(history.history['rmse'], label='rmse', color='green')
# ax2.plot(history.history['val_loss'], label='val_loss', color='red')
# ax.legend(loc='upper center')
# ax2.legend(loc='upper right')
# ax.set_xlabel("epochs")
# ax.set_ylabel("loss")
# ax2.set_ylabel("rmse")
# plt.show()

gain.evaluate(wide_window.test.repeat(), steps=100)
wide_window.plot(gain, plot_col='클로로필-a')
cnt = 0
for i in df:
    # print('iiiiiiiii',i)
    norm_df = i
    print('-----------------4', norm_df.shape)
    data = norm_df.to_numpy()
    print('data', data.shape[0])
    total_n = wide_window.dg.data.shape[0]
    print('total_n', total_n)
    unit_shape = wide_window.dg.shape[1:]
    print('unit_shape', unit_shape)
    dim = wide_window.dg.shape[1]
    print('dim', dim)
    n = (total_n // dim) * dim
    print('n', n)

    x = data[0:n].copy()
    y_true = data[0:n].copy()
    x_reshape = x.reshape((-1,)+unit_shape)
    isnan = np.isnan(x_reshape)
    isnan = np.isnan(y_true)

    x_remain = data[-wide_window.dg.shape[1]:].copy()
    x_remain_reshape = x_remain.reshape((-1,) + unit_shape)
    x_remain_reshape.shape

    # zero loss is normal because there is no ground truth in the real dataset
    gain.evaluate(x_reshape, y_true.reshape((-1,) + unit_shape))

    y_pred = gain.predict(x_reshape)
    y_remain_pred = gain.predict(x_remain_reshape)

    y_pred = y_pred.reshape(y_true.shape)
    y_remain_pred = y_remain_pred.reshape(x_remain.shape)

    y_pred = np.append(y_pred, y_remain_pred[-(total_n - n):], axis=0)
    print('-----------------1', y_pred.shape)
    print('-----------------2', x)
    print('-----------------3', )
    print('-----------------4', )
    print('-----------------5', )

    pd.DataFrame(y_pred).to_excel(
        '/Users/jhy/workspace/' + file_names[cnt][0][:8] + '_' + str(MAX_EPOCHS) + '_result.xlsx', index=False)

    # Denormalized
    # train_mean = df_all.mean()
    # train_std = df_all.std()

    # result = pd.DataFrame(y_pred) * train_std + train_mean
    # result.pop("Day sin")
    # result.pop("Day cos")
    # result.pop("Year sin")
    # result.pop("Year cos")

    # df_date = pd.DataFrame(df_full[0]['측정날짜'][:len(result.index)])
    # df_location = pd.DataFrame(df_full[0]['측정소명'][:len(result.index)])
    # result = pd.concat([df_date,result],axis=1)
    # result.to_excel('/Users/jhy/workspace/'+file_names[cnt][0][:5]+'_'+str(MAX_EPOCHS)+'_result.xlsx', index=False)
    cnt += 1

# ''' 원본데이터로 테스트 '''
# norm_df = pd.concat(df,axis=0)
# data = norm_df.to_numpy()
# total_n = wide_window.dg.data.shape[0]
# unit_shape = wide_window.dg.shape[1:]
# dim = wide_window.dg.shape[1]
# n = (total_n//dim)*dim
# x = data[0:n].copy()
# y_true = data[0:n].copy()
# x_reshape = x.reshape((-1,)+unit_shape)
# isnan = np.isnan(x_reshape)
# isnan = np.isnan(y_true)
# # x[isnan] = np.nan

# ''' 잘린 부분 추가로 append '''
# # # --------기존
# # y_pred = gain.predict(x_reshape)
# # y_pred = y_pred.reshape(y_true.shape)

# # # --------remain
# # x_remain = data[-wide_window.dg.shape[1]:].copy()
# # x_remain_reshape = x_remain.reshape((-1,)+unit_shape)
# # x_remain_reshape.shape
# # # 잘린부분 predict, 나머지 predict
# # y_pred = gain.predict(x_reshape)
# # y_remain_pred = gain.predict(x_remain_reshape)
# # # 잘린부분 reshape, 나머지 reshape
# # y_pred = y_pred.reshape(y_true.shape)
# # y_remain_pred = y_remain_pred.reshape(x_remain.shape)
# # # 잘린 부분 shape
# # print('기존 잘린 부분 shpe',y_pred.shape, y_remain_pred.shape)
# # y_pred = np.append(y_pred, y_remain_pred[-(total_n-n):], axis=0)
# # # append 이후 shape
# # print('append 이후 shape',y_pred.shape)
# # print(np.nan_to_num(y_pred))
# ##### x, t_pred 병합
# df_y = []
# '''
# 일단 임시
# '''
# # n = (total_n//dim)*dim
# # x = data[0:n].copy()
# # y_true = data[0:n].copy()
# # x_reshape = x.reshape((-1,)+unit_shape)
# # isnan = np.isnan(x_reshape)
# # isnan = np.isnan(y_true)
# dim = wide_window.dg.shape[1]
# unit_shape = wide_window.dg.shape[1:]
# # cnt = 0
# # for df_x in df:

# #     ###origin ---
# #     x = df_x.to_numpy()
# #     total_n = x.shape[0]
# #     print('xxxxx',np.shape(x))
# #     n = (total_n//dim)*dim
# #     print('nnn',n)
# #     # x = x[0:n]
# #     ## ----- 여기까지 origin


# #     x_copy = x[0:n].copy()
# #     y_true = x[0:n].copy()
# #     x_reshape = x_copy.reshape((-1,)+unit_shape)
# #     print('x_reshape',x_reshape.shape)
# #     isnan = np.isnan(x_reshape)
# #     isnan = np.isnan(y_true)
# #     # y_pred = gain.predict(x_reshape)
# #     # y_pred = y_pred.reshape(y_true.shape)
# #     x_remain = x[-wide_window.dg.shape[1]:].copy()
# #     x_remain_reshape = x_remain.reshape((-1,)+unit_shape)
# #     x_remain_reshape.shape
# #     # 잘린부분 predict, 나머지 predict
# #     y_pred = gain.predict(x_reshape)

# #     y_remain_pred = gain.predict(x_remain_reshape)

# #     # 잘린부분 reshape, 나머지 reshape
# #     y_pred = y_pred.reshape(y_true.shape)
# #     print('y_pred',y_pred.shape)
# #     y_remain_pred = y_remain_pred.reshape(x_remain.shape)
# #     print('y_remain_pred',y_remain_pred.shape)
# #     y_pred = np.append(y_pred, y_remain_pred[-(total_n-n):], axis=0)

# #     #### origin --
# #     # x_block = x.reshape((-1,)+unit_shape)
# #     # y = gain.predict(x_block)
# #     # y_nan = y.reshape(x.shape)
# #     # print('y_nany_nany_nan',y_nan)
# #     # y_gan = np.nan_to_num(y_nan)
# #     print('-----------------1',y_pred.shape)
# #     print('-----------------2',x.shape)
# #     print('-----------------3',)
# #     print('-----------------4',)
# #     print('-----------------5',)
# #     # y_gan += np.nan_to_num(x)
# #     y_gan = np.nan_to_num(y_pred)
# #     y_gan += np.nan_to_num(x)
# #     ## ----- 여기까지 origin
# #     print("=====================")
# #     print(df_all.shape)
# #     # 컬럼명 변환
# #     y_gan = pd.DataFrame(y_gan)
# #     y_gan.columns = df_all.columns
# #     print(y_gan.shape)
# #     print("=====================")

# #     # Denormalized
# #     print("''''''''''''''''''''''''''''''''''''")
# #     train_mean = df_all.mean()
# #     train_std = df_all.std()
# #     print(train_mean)
# #     print(train_std)
# #     print("''''''''''''''''''''''''''''''''''''")

# #     result = y_gan * train_std + train_mean
# #     result.pop("Day sin")
# #     result.pop("Day cos")
# #     result.pop("Year sin")
# #     result.pop("Year cos")

# #     df_date = pd.DataFrame(df_full[0]['측정날짜'][:len(result.index)])
# #     # df_location = pd.DataFrame(df_full[0]['측정소명'][:len(result.index)])
# #     result2 = pd.concat([df_date,result],axis=1)

# #     result2.to_excel('/Users/jhy/workspace/'+file_names[cnt][0]+'_'+str(MAX_EPOCHS)+'_result.xlsx', index=False)
# #     cnt+=1
# #     # result.to_excel('/Users/jhy/workspace/result.xlsx', index=False)
# #     # y_gan.to_excel('/Users/jhy/workspace/noralize_result.xlsx', index=False)


# # print("=====================")
# # print(df_all.shape)
# # # 컬럼명 변환
# # y_gan = pd.DataFrame(y_gan)
# # y_gan.columns = df_all.columns
# # print(y_gan.shape)
# # print("=====================")

# # # Denormalized
# # print("''''''''''''''''''''''''''''''''''''")
# # train_mean = df_all.mean()
# # train_std = df_all.std()
# # print(train_mean)
# # print(train_std)
# # print("''''''''''''''''''''''''''''''''''''")

# # result = y_gan * train_std + train_mean
# # result.pop("Day sin")
# # result.pop("Day cos")
# # result.pop("Year sin")
# # result.pop("Year cos")

# # df_date = pd.DataFrame(df_full[0]['측정날짜'][:len(result.index)])
# # # df_location = pd.DataFrame(df_full[0]['측정소명'][:len(result.index)])
# # result2 = pd.concat([df_date,result],axis=1)

# # result2.to_excel('/Users/jhy/workspace/'+file_names[0][0]+'_'+str(MAX_EPOCHS)+'_result.xlsx', index=False)
# # result.to_excel('/Users/jhy/workspace/result.xlsx', index=False)
# # y_gan.to_excel('/Users/jhy/workspace/noralize_result.xlsx', index=False)


# ''' result plt '''
# n = 8
# plt.figure(figsize=(9,20))
# for i in range(n):
#     #plt.subplot('%d1%d'%(n,i))
#     plt.subplot(811+i)
#     plt.plot(x[:, i])
#     plt.plot(y_pred[:, i])
# plt.show()
