
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
from file_open import make_dataframe, make_dataframe_in_test
from core.miss_data import MissData

from datetime import datetime

import os
import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import time


__GAIN_TRAINING__ = False
#__GAIN_TRAINING__ = True
__RNN_TRAINING__ = False

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
#run_num_han = [0]



file_names_han = [
    [  # 자동 0
        ['가평_2016.xlsx', '가평_2017.xlsx', '가평_2018.xlsx', '가평_2019.xlsx'],
    ],
    [  # 수질 1
        ['가평천3_2016.xlsx', '가평천3_2017.xlsx', '가평천3_2018.xlsx', '가평천3_2019.xlsx'],
    ],
    [  # AWS 2
        ['남이섬_2016.xlsx', '남이섬_2017.xlsx', '남이섬_2018.xlsx', '남이섬_2019.xlsx'],
    ],
    [  # 방사성  ############################################## -- 3
        ['의암댐_2016.xlsx', '의암댐_2017.xlsx', '의암댐_2018.xlsx', '의암댐_2019.xlsx'],
    ],
    [  # TMS 4
        ['가평신천하수_2016.xlsx', '가평신천하수_2017.xlsx', '가평신천하수_2018.xlsx', '가평신천하수_2019.xlsx'],
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


    df, times = make_dataframe_in_test(folder[idx], file_names[idx], iloc_val[idx], interpolate=interpolation_option[idx])
    #if i == 0:
    #    dfff = df
    #    time_array = times
    print('---------------------------------------')
    print(df[0].shape)
    #print(df[0].head())

    #print(times)



    times = pd.Series(range(times.shape[0]), index=times)
    times = times['2019-07-01':'2019-07-12']
    real_df_all = df[0].iloc[times,:]
    real_df_all = pd.DataFrame(real_df_all)
    print(real_df_all.shape)
    #print(folder[idx]+'test.xlsx')
    real_df_all.to_excel(folder_han[idx]+'/[0601-0612]'+file_names[idx][0][3], index=False)
