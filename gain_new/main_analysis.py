import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from glob import glob
from tensorflow.keras.layers import Input, Concatenate, Dot, Add, ReLU, Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from core.gain import GAIN
from core.gain_data_generator import GainDataGenerator
from core.window_generator import WindowGenerator
from core.utils import *


# input parameter
DIR = 'data'
FILE_LIST = [['의암호_2018.xlsx'], ['의암호_2019.xlsx']]
MAX_EPOCHS = 100
DEBUG = True

# df_list 는 Day sin, Day cos, Year sin, Year cos 컬럼을 추가한 각각의 dataframe을 포함하고 있음 (2~10 컬럼 하드코딩)
# df_full 은 입력으로 받은 원본 그대로를 각각의 dataframe을 포함하고 있음
# df_all 은 df_list 를 행 기준으로 concat 한 하나의 dataframe
df_list, df_full, df_all = create_dataframe(DIR, FILE_LIST)

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
    input_width=24 * 5,
    label_width=24 * 5,
    shift=0
)

# 단순히 self.dg 를 생성하기 위한 코드
# example -> train -> make_dataset -> GainDataGenerator 순서로 호출됨
_ = wide_window.example

val_performance = {}
performance = {}

# gain 객체를 생성함
# __init__ -> build_generator -> build_discriminator 순서로 호출됨
gain = GAIN(shape=wide_window.dg.shape[1:], gen_sigmoid=False)
gain.compile(loss=GAIN.RMSE_loss)

# 학습 함수
def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=GAIN.RMSE_loss)
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    return history

# 학습을 시작
# train -> make_dataset -> val -> make_dataset_gain
# train_step -> G_loss -> __len__ -> __getitem__
history = compile_and_fit(gain, wide_window, patience=MAX_EPOCHS // 5)

# loss 곡선을 출력하는 사용자 커스텀 유틸 함수
if DEBUG:
    figure_loss(history)

# 모델의 평가를 변수에 저장
val_performance['Gain'] = gain.evaluate(wide_window.val)
performance['Gain'] = gain.evaluate(wide_window.test, verbose=0)

# gain 모델 자체를 아래의 두 파일로 저장함
# discriminator.h5, generator.h5 저장
gain.save(save_dir='save')

# 성능 측정
# gain.evaluate(wide_window.test.repeat(), steps=100)

# predict 결과를 출력하는 사용자 커스텀 유틸 함수
if DEBUG:
    figure_gain(df_list, wide_window, gain)
