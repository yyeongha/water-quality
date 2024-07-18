#####################################################################################
# WaterDataGenerator 클래스를 정의하여 시계열 데이터의 배치를 생성, 이를 학습에 사용할 수 있도록 준비
# 이 클래스는 TensorFlow의 Sequence를 상속받아 구현되어, 모델 학습 중 데이터 생성기를 효율적으로 사용할 수 있게 함
#####################################################################################

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 데이터 생성기 초기화, 배치 생성을 위한 기본 설정을 수행함
class WaterDataGenerator(tf.keras.utils.Sequence):
    'Generates data for water'

    def __init__(self,
                 imputed_data,
                 ori_data=None,
                 batch_size=32,
                 input_width=24 * 7,
                 label_width=24 * 3,
                 shift=24 * 3,
                 skip_time=None,
                 shuffle=True,
                 out_features=None,
                 out_num_features=None,
                 ):
        'Initialization'
        self.window_size = input_width + shift
        self.total_no = imputed_data.shape[0]
        self.data = imputed_data
        self.input_width = input_width
        self.label_width = label_width
        self.batch_size = batch_size
        self.input_shape = (batch_size, input_width, self.data.shape[1])
        self.out_num_features = out_num_features
        if out_features:
            self.out_features = out_features
        else:
            self.out_features = [i for i in range(out_num_features)]
        self.label_shape = (batch_size, label_width, self.out_num_features)

        if (skip_time):
            # TO-DO
            self.no = self.total_no - self.window_size + 1
            if self.no < 1 :
                self.no = 1
            self.data_idx = np.arange(0, self.no)
        else:
            self.no = self.total_no - self.window_size + 1
            if self.no < 1 :
                self.no = 1
            self.data_idx = np.arange(0, self.no)

        if shuffle:
            self.batch_idx = np.random.permutation(self.no)
        else:
            self.batch_idx = np.arange(0, self.no)
        self.batch_id = 0

    # 에폭당 배치 수를 반환
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.no//self.batch_size

    # 인덱스에 해당하는 배치를 생성하여 반환
    def __getitem__(self, index):
        'Generate one batch of data'
        # Sample batch
        label_width = self.label_width
        batch_idx = self.batch_idx

        x = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.label_width, self.out_num_features))

        for cnt in range(0, self.batch_size):
            i = self.batch_id
            self.batch_id += 1

            idx1 = self.data_idx[batch_idx[i]]
            idx2 = idx1 + self.input_width

            X = self.data[idx1:idx2].to_numpy()

            idx1 = self.data_idx[batch_idx[i]] + self.window_size - label_width
            idx2 = idx1 + label_width

            Y = self.data.iloc[idx1:idx2, self.out_features].to_numpy()
            self.batch_id %= self.no
            x = np.append(x, [X], axis=0)
            y = np.append(y, [Y], axis=0)

        return x, y

    # 에폭이 끝날 때 호출되는 함수, 인덱스를 업데이트함
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return
