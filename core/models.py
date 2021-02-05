# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


def nse(y_true, y_pred):
    mean = tf.reduce_mean(y_true)
    return 1. - tf.reduce_sum(tf.square(y_true-y_pred))/tf.reduce_sum(tf.square(y_true-mean))

def compile_and_fit(model, window, patience=1000, epochs=400, save_path=None):
    checkpoint = keras.callbacks.ModelCheckpoint(
        save_path, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only= True, mode='auto', period=1)
        #save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(), nse])

    history = model.fit(
        #window.train, epochs=epochs,
        window.train, epochs=epochs, steps_per_epoch=10,
        validation_data=window.val,
        callbacks=[early_stopping, checkpoint])
    return history

def compile(model):

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()])


def plot_history(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['mean_absolute_error'], label='mae')
#     ax.plot(history.history['mean_squared_error'], label='mse')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.plot(history.history['val_mean_absolute_error'], label='val_mae')
#     ax.plot(history.history['val_mean_squared_error'], label='val_mse')
    #plt.legend(history.history.keys(), loc='upper right')
    #ax.legend(loc='upper center')
    ax.legend()
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    plt.show()



class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, out_features, OUT_STEPS):
        self.out_features = out_features
        self.OUT_STEPS = OUT_STEPS

    def call(self, inputs):
        #print(inputs[:, -1:, 0:1])
        #return tf.tile(inputs[:, -1:, :out_num_features], [1, OUT_STEPS, 1])
        return tf.tile(inputs[:, -1:, (self.out_features[0]):(self.out_features[0]+1)], [1, self.OUT_STEPS, 1])
        #return tf.tile(inputs[:, -1:, out_features[0]:(out_features[1]+1)], [1, OUT_STEPS, 1])

#def LastBaseLine(OUT_STEPS, out_features):



def MultiLinearModel(OUT_STEPS, out_num_features):
    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * out_num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return multi_linear_model

def ElmanModel(OUT_STEPS, out_num_features):
    elman_model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.SimpleRNN(128, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS*out_num_features,
                          kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return elman_model

#def GRUModel(OUT_STEPS, out_num_features):
#    gru_model = tf.keras.Sequential([
#        tf.keras.layers.GRU(128, return_sequences=False),
#        tf.keras.layers.Dense(OUT_STEPS * out_num_features,
#                              kernel_initializer=tf.initializers.zeros),
#        # Shape => [batch, out_steps, features]
#        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
#    ])
#    return gru_model

def GRUModel(OUT_STEPS, out_num_features):
    gru_model = tf.keras.Sequential([
        tf.keras.layers.GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.5),
        tf.keras.layers.GRU(256, return_sequences=False, dropout=0.1, recurrent_dropout=0.5),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(OUT_STEPS * out_num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return gru_model




def MultiLSTMModel(OUT_STEPS, out_num_features):
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        # tf.keras.layers.LSTM(32, return_sequences=False),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        #     tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * out_num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
    ])
    return multi_lstm_model


def MultiConvModel(OUT_STEPS, out_num_features):
    CONV_WIDTH = 7
    CONV_LAYER_NO = 1
    multi_conv_model = tf.keras.Sequential([
                                               # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                                               tf.keras.layers.Lambda(
                                                   lambda x: x[:, -(CONV_WIDTH * CONV_LAYER_NO - CONV_LAYER_NO + 1):,
                                                             :]),
                                           ] + [
                                               # Shape => [batch, 1, conv_units]
                                               tf.keras.layers.Conv1D(1024, activation='relu', kernel_size=(CONV_WIDTH))
                                               for i in range(CONV_LAYER_NO)
                                           ] + [
                                               # Shape => [batch, 1,  out_steps*features]
                                               tf.keras.layers.Dense(OUT_STEPS * out_num_features,
                                                                     kernel_initializer=tf.initializers.zeros),
                                               # Shape => [batch, out_steps, features]
                                               tf.keras.layers.Reshape([OUT_STEPS, out_num_features])
                                           ])
    return multi_conv_model