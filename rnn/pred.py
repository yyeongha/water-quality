import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from core.data_generator import DataGenerator
from core.window_generator import WindowGenerator
from core.utils import *

# font for korean
import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothicCoding-Bold'


target_col = '총유기탄소'

input_step = 24*7
OUT_STEPS = 24*3
# input_step = 2
# OUT_STEPS = 1

MAX_EPOCHS = 1

gain_folder = './data/test/gain'
original_folder = './data/test/origin'
# file_list = [['서상_2018.xlsx', '서상_2019.xlsx'], ['의암호_2018.xlsx','의암호_2019.xlsx']]
file_list = [['서상_2019.xlsx'], ['의암호_2019.xlsx']]

df = createDataFrame(gain_folder, file_list)
origin_df = createDataFrame(original_folder, file_list)

origin_df.to_excel("./origin_df.xlsx", index=False)
# interpolate(origin_df, max_gap=2)

# print(type(df))
num_features = df.shape[1]
label_columns_indices = {name: i for i, name in enumerate(df)}
target_col_idx = label_columns_indices[target_col]
print(target_col_idx)



dgen = DataGenerator(df, origin_data= origin_df, fill_no=2, input_width=24*7, label_width=24*3, target_col_idx=target_col_idx)

n = len(df)
print(n)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

wide_window = WindowGenerator(
    df=df,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    input_width=24*7,
    label_width=24*3,
    shift=24,
    label_columns=target_col
)

wide_window.plot()
