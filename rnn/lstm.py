import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# df = pd.read_csv('data/cansim-0800020-eng-6674700030567901031.csv',
#                  skiprows=6, skipfooter=9,
#                  engine='python')

path = "./data/1/"
#
excelfiles = []
df = []

# timestep = 7*24



def normalization(df):
    M = df.mean() # 평균
    S = df.std(ddof=0) # 표준편차
    normalization_df = (df - M)/S
    normalization_df = normalization_df.fillna(0)
    return normalization_df, M, S

def denormalization(normalization_df, M, S):
    df = normalization_df * S + M
    df = df.fillna(0)
    return df

def create_dataset(origin_data, look_back=1):
    # start_row = origin_data.shape[0]
    # print("origin_data.shape[0] : " + str(origin_data.shape[0]))
    # print(origin_data.head())
    # end_row = origin_data.shape[0] - look_back
    end_row = math.floor(origin_data.shape[0]/2)
    dataX = origin_data.iloc[:end_row, :]
    dataY = origin_data.iloc[end_row:end_row*2, :]



for root, dirs, files in os.walk(path):
    for filename in files:
        if filename == '.DS_Store':
            continue
        excelfiles.append(path+filename)
# print(excelfiles)

for fname in excelfiles:

    # print(os.path.splitext(fname)[1])

    if os.path.splitext(fname)[1] == '.csv':
        df = pd.read_csv(fname,encoding='utf-8-sig')
    elif os.path.splitext(fname)[1] == '.xlsx':
        df = pd.read_excel(fname)
    rows = df.shape[0]
    cols = df.shape[1]

    columnlist = df.columns.tolist()


predict_rate = 24*7
test_rate = 24 * 7
timestep = 1

print("rows : " + str(rows))
print("predict_rate : " + str(rows - predict_rate))

# split_rate = int(math.ceil(rows * 0.96))
# split_rate2 = int(math.ceil((rows-split_rate) * 0.5))

# split_rate = rows - (predict_rate*2 + test_rate)
#
# print("split_rate : " + str(split_rate))
#
# train_x = df.iloc[:split_rate,1:]
# train_y = df.iloc[split_rate:split_rate+predict_rate,1: ]
# test_x = df.iloc[split_rate+predict_rate:split_rate+predict_rate+test_rate,1: ]
# test_y = df.iloc[split_rate+predict_rate+test_rate:,1: ]

# df, mean, std = normalization(df)


split_rate = int(math.ceil((rows - predict_rate*2)/2))

print("split_rate : " + str(split_rate))

train_x = df.iloc[:split_rate,1:]
train_y = df.iloc[split_rate:split_rate+split_rate,1: ]
test_x = df.iloc[split_rate+split_rate:split_rate+split_rate+predict_rate,1: ]
test_y = df.iloc[split_rate+predict_rate+split_rate:,1: ]






print("train_x")
print(train_x.shape)
print("train_y")
print(train_y.shape)

# print(columnlist)

# train = df.iloc[:split_rate,:]
# test = df.iloc[split_rate:,:]

# train = df.iloc[:split_rate,1:]
# # test = df.iloc[split_rate:,6 ]
# test = df.iloc[split_rate:split_rate+split_rate2,1: ]
#
# real = df.iloc[split_rate+split_rate2:,1: ]

# train_x.to_csv("./train_x.csv", header=False, index=False, encoding='utf-8-sig')
# train_y.to_csv("./train_y.csv", header=False, index=False, encoding='utf-8-sig')
# test_x.to_csv("./test_x.csv", header=False, index=False, encoding='utf-8-sig')
# test_y.to_csv("./test_y.csv", header=False, index=False, encoding='utf-8-sig')


# train_sc_df, mean_train, std_train = normalization(train)
# test_sc_df, mean_test, std_test = normalization(test)
# real_sc_df, mean_real, std_real = normalization(real)

# train_x_sc_df, mean_train_x, std_train_x = normalization(train_x)
# train_y_sc_df, mean_train_y, std_train_y = normalization(train_y)
# test_x_sc_df, mean_test_x, std_test_x = normalization(test_x)
# test_y_sc_df, mean_test_y, std_test_y = normalization(test_y)

# print("mean_test_y1")
# print(mean_test_y)
# print(std_test_y)

#
# train_x_sc_df.to_csv('./train_x_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# test_x_sc_df.to_csv('./test_x_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# train_y_sc_df.to_csv('./train_y_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# test_y_sc_df.to_csv('./test_y_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
#
# de_train_x_sc_df = denormalization(train_x_sc_df, mean_train_x, std_train_x)
# de_train_y_sc_df = denormalization(train_y_sc_df, mean_train_y, std_train_y)
# de_test_x_sc_df = denormalization(test_x_sc_df, mean_test_x, mean_test_x)
# de_test_y_sc_df = denormalization(test_y_sc_df, mean_test_y, std_test_y)
#
#
# de_train_x_sc_df.to_csv('./de_train_x_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# de_test_x_sc_df.to_csv('./de_test_x_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# de_train_y_sc_df.to_csv('./de_train_y_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# de_test_y_sc_df.to_csv('./de_test_y_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# #
#
# de_train_sc_df = denormalization(train_sc_df, mean_train, std_train)
# de_test_sc_df = denormalization(test_sc_df, mean_test, std_test)
#
# de_train_sc_df.to_csv('./de_train_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')
# test_sc_df.to_csv('./de_test_sc_df.csv', header=False, index=False ,encoding='utf-8-sig')

column_list = list(train_x)

print(column_list)

'''
for s in range(1, timestep + 1):
    tmp_train_x = train_x[column_list].shift(s)
    tmp_train_y = train_y[column_list].shift(s)
    tmp_test_x = test_x[column_list].shift(s)
    tmp_test_y = test_y[column_list].shift(s)
    tmp_train_x.columns = "shift_" + tmp_train_x.columns + "_" + str(s)
    tmp_train_y.columns = "shift_" + tmp_train_y.columns + "_" + str(s)
    tmp_test_x.columns = "shift_" + tmp_test_x.columns + "_" + str(s)
    tmp_test_y.columns = "shift_" + tmp_test_y.columns + "_" + str(s)
    train_x[tmp_train_x.columns] = train_x[column_list].shift(-s)
    train_y[tmp_train_y.columns] = train_y[column_list].shift(-s)
    test_x[tmp_test_x.columns] = test_x[column_list].shift(-s)
    test_y[tmp_test_y.columns] = test_y[column_list].shift(-s)
'''



# for s in range(1, timestep+1):
#     train_x_sc_df['shift_{}'.format(s)] = None
#     train_y_sc_df['shift_{}'.format(s)] = None
#     test_x_sc_df['shift_{}'.format(s)] = None
#     test_y_sc_df['shift_{}'.format(s)] = None
#     train_x_sc_df['shift_{}'.format(s)] = train_x_sc_df[column_list].shift(s)
#     train_y_sc_df['shift_{}'.format(s)] = train_y_sc_df.shift(s)
#     test_x_sc_df['shift_{}'.format(s)] = test_x_sc_df.shift(s)
#     test_y_sc_df['shift_{}'.format(s)] = test_y_sc_df.shift(s)

# print(train_sc_df.head(13))


# train_x_sc_df.to_csv('./train_x_sc_df1.csv', header=False, index=False ,encoding='utf-8-sig')
# test_x_sc_df.to_csv('./test_x_sc_df1.csv', header=False, index=False ,encoding='utf-8-sig')



# X_train1 = train_x_sc_df.dropna()
# X_train = train_x_sc_df.dropna().drop(['총유기탄소', '수온'], axis=1)
# X_train = train_x.dropna().drop(column_list, axis=1)
X_train = train_x.dropna()
y_train = train_y.dropna()[['총유기탄소']]

# X_train1.to_csv('./X_train1.csv', header=False, index=False ,encoding='utf-8-sig')
# X_train.to_csv('./X_train.csv', header=False, index=False ,encoding='utf-8-sig')
# y_train.to_csv('./y_train.csv', header=False, index=False ,encoding='utf-8-sig')

# X_test = test_x_sc_df.dropna(axis=1)
# X_test = test_x_sc_df.dropna().drop(['총유기탄소', '수온'], axis=1)
# X_test = test_x.dropna().drop(column_list, axis=1)
X_test = test_x.dropna()
y_test = test_y.dropna()[['총유기탄소']]


y_train.to_csv('./y_train.csv', header=False, index=False ,encoding='utf-8-sig')
X_test.to_csv('./y_test.csv', header=False, index=False ,encoding='utf-8-sig')


X_train, mean_train_x, std_train_x = normalization(X_train)
y_train, mean_train_y, std_train_y = normalization(y_train)
X_test, mean_test_x, std_test_x = normalization(X_test)
y_test, mean_test_y, std_test_y = normalization(y_test)



# result = denormalization(y_test, mean_test_y, std_test_y )
#
# result.to_csv('./result.csv', header=False, index=False ,encoding='utf-8-sig')

# print('y_test')
# print(y_test)

# y_test1 = pd.DataFrame(sc.inverse_transform(y_test))
# y_test1 = denormalization(y_test, mean_test, std_test)
# y_test1 = pd.DataFrame(denormalization(y_test, mean, std))

# X_test.to_csv('./X_test.csv', header=False, index=False ,encoding='utf-8-sig')
# y_test.to_csv('./y_test.csv', header=False, index=False ,encoding='utf-8-sig')
# y_test1.to_csv('./y_test1.csv', header=False, index=False ,encoding='utf-8-sig')

# X_real = real_sc_df.dropna(axis=1)

# X_real = real_sc_df.dropna().drop('총유기탄소', axis=1)
# y_real = real_sc_df.dropna()[['총유기탄소']]
# y_real.to_csv('./y_real.csv', header=False, index=False ,encoding='utf-8-sig')


# print(X_train.head())
# print(y_train.head())


X_train = X_train.values
y_train = y_train.values

print("X_train")
print(X_train.shape)
print("y_train")
print(y_train.shape)

# X_real= X_real.values

X_test= X_test.to_numpy()
y_test = y_test.to_numpy()
# y_test = y_test.values
# y_real = y_real.values


# print(X_train.shape[0])
# print(X_train)
# print(y_train.shape[0])
# print(y_train)





# X_train_t = X_train.reshape(X_train.shape[0], X_train.shape[1], 1).astype(float)
# X_test_t = X_test.reshape(X_test.shape[0], X_test.shape[1], 1).astype(float)
# X_real_t = X_test.reshape(X_real.shape[0], X_real.shape[1], 1).astype(float)

print((len(column_list)))

X_train_t = X_train.reshape(X_train.shape[0], timestep, len(column_list)).astype(float)
X_test_t = X_test.reshape(X_test.shape[0], timestep, len(column_list)).astype(float)
# y_train_t = X_test.reshape(-1, timestep, 1).astype(float)

print("X_train_t")
print(X_train_t.shape)

print("X_test_t")
print(X_test_t.shape)



# print("최종 DATA")
# print(X_train_t.shape)
# print(X_train_t)
# print(y_train)

from tensorflow import keras
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
# import keras


# K.clear_session()
# model = Sequential() # Sequeatial Model
# model.add(LSTM(32, input_shape=(timestep, len(column_list)), return_sequences = True, activation='tanh')) # (timestep, feature)
# model.add(LSTM(32, input_shape=(timestep, len(column_list)), activation='tanh'  )) # (timestep, feature)
# model.add(Dense(1))# output = 1
#
# opt = keras.optimizers.Adam(learning_rate=0.1)
# model.compile(loss='mean_squared_error', optimizer=opt)
# # model.compile(loss='mean_squared_error', optimizer='adam')
#
# model.summary()
#
# early_stop = EarlyStopping(monitor='loss', patience=50, verbose=1)
#
# reduceLR = ReduceLROnPlateau(
#     monitor='loss',  # 검증 손실을 기준으로 callback이 호출됩니다
#     factor=0.5,          # callback 호출시 학습률을 1/2로 줄입니다
#     patience=10,         # epoch 10 동안 개선되지 않으면 callback이 호출됩니다
#     min_lr=0.001,
# )
#
# # model.fit(X_train_t, y_train, epochs=10000,
# #           batch_size=128, verbose=1, callbacks=[`early_stop`])
#
# # hist =  model.fit(X_train_t, y_train, epochs=10000,
# #           batch_size=64, verbose=1)
#
# hist =  model.fit(X_train_t, y_train, epochs=100, shuffle=False,
#           batch_size=64, verbose=1, callbacks=[reduceLR, early_stop])


# print(hist.history['loss'])
#
# fig, loss_ax = plt.subplots()
# loss_ax.plot(hist.history['loss'], label='train loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')
# plt.show()
#
# with open("loss_curr.txt", "a") as myfile:
#     print(hist.history['loss'], file=myfile)


model = Sequential()
model.add(LSTM(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train_t, y_train, epochs=15, verbose=1)

plt.plot(history.history['loss'], label="loss")
plt.legend(loc="upper right")
plt.show()


# print(X_test_t)


y_pred = model.predict(X_test_t)

# print(type(y_pred))
# print(type(y_test))

# pd.DataFrame()

# print("mean_test_y2")
# print(mean_test_y)
# print(std_test_y)
# y_test1 = pd.DataFrame(y_test)
# y_pred = denormalization(pd.DataFrame(y_pred), mean_test_y, std_test_y)
# y_test2 = denormalization(y_test1, mean_test_y, std_test_y)
#
# print(type(y_pred))
# print(type(y_test))

# y_test2.to_csv('./y_test1111.csv', header=False, index=False ,encoding='utf-8-sig')
# y_pred.to_csv('./y_pred.csv', header=False, index=False ,encoding='utf-8-sig')

plt.plot(y_test)
plt.plot(y_pred)
plt.legend(['y_test','y_pred'])

plt.show()

y_test   = pd.DataFrame(y_test, columns=['총유기탄소'])
result = pd.DataFrame(y_pred, columns=['총유기탄소'])
# result = pd.DataFrame(sc.inverse_transform(y_pred, 1))
result.to_csv('./result111.csv', header=False, index=False ,encoding='utf-8-sig')

y_test = denormalization(y_test, mean_test_y, std_test_y)
result = denormalization(result, mean_test_y, std_test_y)

plt.plot(y_test)
plt.plot(result)
plt.legend(['real','predict'])

plt.show()

# result = denormalization(result, mean, std)
# print(y_pred)
# plt.plot(y_test1)
# plt.plot(result)
# plt.legend(['inv_y_test','inv_result'])
#
# plt.show()




result.to_csv('./result.csv', header=False, index=False ,encoding='utf-8-sig')
