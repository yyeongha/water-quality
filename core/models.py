#####################################################################################
# 시계열 데이터를 처리하고 예측하는 다양한 딥러닝 모델을 구현
# 모델을 학습하고 평가하기 위한 유틸리티 함수와 콜백을 포함
#####################################################################################

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback

# 학습 도중 모델의 성능이 향상될때마다 모델의 가중치를 저장
class checkpoint_save(Callback):
    def __init__(self, model=None, save_path=None, val_nse=None):
        super(Callback, self).__init__() 
        self.best_score = val_nse
        self.model = model
        self.save_path=save_path

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_nse')
        if current_score > self.best_score:
            print('save model: nse from %.3f to %.3f' %(self.best_score, current_score) )
            self.best_score = current_score
            self.model.save_weights(self.save_path)

# nse 메트릭 계산
def nse(y_true, y_pred):
    mean = tf.reduce_mean(y_true)
    return 1. - tf.reduce_sum(tf.square(y_true-y_pred))/tf.reduce_sum(tf.square(y_true-mean))

# 모델 컴파일 및 학습 수행
def compile_and_fit(model, window, patience=1000, epochs=400, save_path=None, val_nse=-10000, steps_per_epoch = 10):

    checkpoint = checkpoint_save(model=model, save_path=save_path, val_nse = val_nse)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')

    model.compile(
        loss=tf.losses.MeanSquaredError(),

        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(), nse])

    history = model.fit(
        window.train, epochs=epochs, steps_per_epoch=steps_per_epoch,
        validation_data=window.val,
        callbacks=[early_stopping, checkpoint])
    return history

# 모델 컴파일
def compile(model):

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(), nse])

# 학습과정에서 손실값과 메트릭 변화를 시각화
def plot_history(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['mean_absolute_error'], label='mae')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.plot(history.history['val_mean_absolute_error'], label='val_mae')
    ax.legend()
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    plt.show()


# 베이스라인 모델, 마지막 입력 시점의 값을 반복하여 예측값으로 사용
class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, out_features, OUT_STEPS):
        self.out_features = out_features
        self.OUT_STEPS = OUT_STEPS

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, (self.out_features[0]):(self.out_features[0]+1)], [1, self.OUT_STEPS, 1])

# 단순한 선형 모델 정의
def MultiLinearModel(OUT_STEPS, out_num_features):
    multi_linear_model = tf.keras.Sequential([

        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * out_num_features),
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return multi_linear_model

# Elman RNN을 사용한 시계열 예측 모델 정의
def ElmanModel(OUT_STEPS, out_num_features):
    elman_model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(128, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS*out_num_features),
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return elman_model

# GRU를 사용한 시계열 예측 모델 정의
def GRUModel(OUT_STEPS, out_num_features):
    gru_model = tf.keras.Sequential([
        tf.keras.layers.GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.5),
        tf.keras.layers.GRU(256, return_sequences=False, dropout=0.1, recurrent_dropout=0.5),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(OUT_STEPS * out_num_features),
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return gru_model

# LSTM을 사용한 시계열 예측 모델 정의
def MultiLSTMModel(OUT_STEPS, out_num_features):
    multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * out_num_features),
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return multi_lstm_model

# 1D Convolution을 사용한 시계열 예측 모델 정의
def MultiConvModel(OUT_STEPS, out_num_features):
    CONV_WIDTH = 7
    CONV_LAYER_NO = 1
    multi_conv_model = tf.keras.Sequential([

                                               tf.keras.layers.Lambda(
                                                   lambda x: x[:, -(CONV_WIDTH * CONV_LAYER_NO - CONV_LAYER_NO + 1):,
                                                             :]),
                                           ] + [
                                               tf.keras.layers.Conv1D(1024, activation='relu', kernel_size=(CONV_WIDTH))
                                               for i in range(CONV_LAYER_NO)
                                           ] + [
                                               tf.keras.layers.Dense(OUT_STEPS * out_num_features),
                                               tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
                                           ])
    return multi_conv_model
