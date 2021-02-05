# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


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
        #         print("out_features")
        #         print(out_features)
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(128/self.batch_size)
        # return 2
        #return 1
        return self.no//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # print('index =', index)
        # print('self.no =', self.no)
        # print('self.total_no =', self.total_no)
        # print('self.batch_id =', self.batch_id)
        # Sample batch
        label_width = self.label_width
        batch_idx = self.batch_idx

        x = np.empty((0, self.input_width, self.data.shape[1]))
        # y = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.label_width, self.out_num_features))

        #print('--------------------------------23123')
        #print(x.shape)
        #print(y.shape)

        for cnt in range(0, self.batch_size):
            i = self.batch_id
            self.batch_id += 1

            idx1 = self.data_idx[batch_idx[i]]
            idx2 = idx1 + self.input_width

            X = self.data[idx1:idx2].to_numpy()

            idx1 = self.data_idx[batch_idx[i]] + self.window_size - label_width
            idx2 = idx1 + label_width

            # Y = self.data[idx1:idx2,:,:out_num_features]
            Y = self.data.iloc[idx1:idx2, self.out_features].to_numpy()
            # Y = self.data[idx1:idx2]
            # print('Y.shape = ', Y.shape)
            # Y = Y.iloc[:,:out_num_features]


            self.batch_id %= self.no
            # print("x.shape=", x.shape)
            # print('X.shape=', X.shape)
            # print(type(x), type(X))
            x = np.append(x, [X], axis=0)
            y = np.append(y, [Y], axis=0)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return
