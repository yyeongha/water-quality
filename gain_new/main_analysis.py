import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *


# input parameter
parameters_dir = './input'
parameters_file = 'input.json'
parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)
with open(parameters_path, encoding='utf8') as json_file:
    parameters = json.load(json_file)

# init input parameter
gain_parameters = parameters['gain']

DEBUG = gain_parameters['debug']
max_epochs = gain_parameters['max_epochs']
input_width = gain_parameters['input_width']
label_width = gain_parameters['label_width']
shift = gain_parameters['shift']
use_train = gain_parameters['use_train']
batch_size = gain_parameters['batch_size']
miss_pattern = gain_parameters['miss_pattern']
miss_rate = gain_parameters['miss_rate']
fill_no = gain_parameters['fill_no']

# df_list 는 Day sin, Day cos, Year sin, Year cos 컬럼을 추가한 각각의 dataframe을 포함하고 있음 (2~10 컬럼 하드코딩)
# df_full 은 입력으로 받은 원본 그대로를 각각의 dataframe을 포함하고 있음
# df_all 은 df_list 를 행 기준으로 concat 한 하나의 dataframe
df_list, df_full, df_all = create_dataframe(gain_parameters['data_dir'], gain_parameters['data_file'])

# df_all 의 평균, 표준편차를 사용함
# df_list 를 대상으로 df_all 의 평균, 표준편차로 정규화 진행
# 리스트 아이템의 주소를 참조하기 때문에 return 없이 값이 변경됨
standard_normalization(df_list, df_all)

# WindowGenerator의 init 함수 호출
wide_window = WindowGenerator(
    df=df_list,
    train_df=df_all,
    val_df=df_all,
    test_df=df_all,
    input_width=input_width,
    label_width=label_width,
    shift=shift,
    batch_size=batch_size,
    miss_pattern=miss_pattern,
    miss_rate=miss_rate,
    fill_no=fill_no
)

# 단순히 self.dg 를 생성하기 위한 코드
# example -> train -> make_dataset -> GainDataGenerator 순서로 호출됨
_ = wide_window.example

# 모델 평가를 저장할 변수 초기화
val_performance = {}
performance = {}

# gain 객체를 생성함
# __init__ -> build_generator -> build_discriminator 순서로 호출됨
gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)

# 학습 사용 구분
if use_train:
    # 학습 함수
    def compile_and_fit(model, window, patience=10):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
        model.compile(loss=GAIN.RMSE_loss)
        history = model.fit(window.train, epochs=max_epochs, validation_data=window.val, callbacks=[early_stopping])
        return history

    # 학습을 시작
    # train -> make_dataset -> val -> make_dataset_gain
    # train_step -> G_loss -> __len__ -> __getitem__
    history = compile_and_fit(gain, wide_window, patience=max_epochs // 5)

    # loss 곡선을 출력하는 사용자 커스텀 유틸 함수
    if DEBUG:
        figure_loss(history)

    # 모델의 평가를 변수에 저장
    val_performance['Gain'] = gain.evaluate(wide_window.val)
    performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

    # gain 모델 자체를 아래의 두 파일로 저장함
    # discriminator.h5, generator.h5 저장
    gain.save(save_dir='save')
else:
    gain.load(save_dir='save') 

# 성능 측정
# gain.evaluate(wide_window.test.repeat(), steps=100)

# predict 결과를 출력하는 사용자 커스텀 유틸 함수
if DEBUG:
    figure_gain(df_list, wide_window, gain)