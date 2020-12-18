import numpy as np

from tensorflow import keras
from core.utils import *



class DataGenerator(keras.utils.Sequence):
    'Generates data for GAIN'
    def __init__(self,
                 data_list,
                 origin_data=None,
                 batch_size=32,
                 input_width=24*7,
                 label_width=24*3,
                 shift=0,
                 fill_no=2,
                 normalize=True,
                 target_col_idx=None
                 ):
        'Initialization'
        window_size = input_width
        self.total_window_size = input_width + shift

        self.target_col_idx = target_col_idx

        # self.data_idx = data_idx

        # interpollation for original data
        if origin_data is not None:
            ori_data = interpolate(origin_data, max_gap=fill_no)
            for_idx_data = ori_data.to_numpy()

        else :
            for_idx_data = data_list.to_numpy()
            # ori_data.to_excel("./interpolate.xlsx")


        self.data = data_list.to_numpy()

        # TO-DO
        # pre calculation for  sequence data
        last_cum = 0
        cums = []

        isnan = np.isnan(for_idx_data)
        isany = np.any(isnan, axis=1)
        shifted = np.roll(isany, 1)
        shifted[0] = True # set to nan
        start_seq = ((isany == False) & (shifted == True)).astype(int)
        cum = start_seq.cumsum()
        cum += last_cum
        last_cum = np.max(cum)
        cum[isany == 1] = np.isnan(123) #np.isnan(123) instead of np.nan
        cums.append(cum)

        # sequence data
        self.ids = np.concatenate(cums)

        data_idx = np.empty((0), dtype=int)
        for i in range(1, last_cum+1):
            seq_len = (self.ids == i).sum()
            start_id = np.argmax(self.ids == i)
            time_len = seq_len - window_size + 1
            start_ids = np.arange(start_id, start_id+time_len)
            data_idx = np.append(data_idx, start_ids)

        # start index set for sequence data
        self.data_idx = data_idx
        # print(data_idx)
        self.input_width = input_width
        self.label_width = label_width
        self.no = len(data_idx)
        self.batch_size = batch_size

        # random shuffling  index
        self.batch_idx = sample_batch_index(self.no, self.no)
        self.batch_id = 0
        self.shape = (batch_size,self.input_width)+self.data.shape[1:]
        #self.hint_rate = hint_rate

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'

        x = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.label_width, self.data.shape[1]))
        for cnt in range(0, self.batch_size):
            i = self.batch_idx[self.batch_id]
            self.batch_id += 1
            self.batch_id %= self.no
            if (self.batch_id == 0):
                self.batch_idx = sample_batch_index(self.no, self.no)
            idx1 = self.data_idx[i]
            idx2 = self.data_idx[i]+self.input_width

            X_mb = self.data[idx1:idx2]
            # Y_mb = self.data[idx2+1:self.total_window_size, self.target_col_idx:self.target_col_idx+1]
            Y_mb = self.data[idx2+1:self.total_window_size]

            x = np.append(x, [X_mb], axis=0)
            y = np.append(y, [Y_mb], axis=0)


        # x = np.empty((0, self.input_width, self.data.shape[1]))
        # y = np.empty((0, self.input_width, self.data.shape[1]))
        # for cnt in range(0, self.batch_size):
        #     i = self.batch_idx[self.batch_id]
        #     self.batch_id += 1
        #     self.batch_id %= self.no
        #     if (self.batch_id == 0):
        #         self.batch_idx = sample_batch_index(self.no, self.no)
        #     idx1 = self.data_idx[i]
        #     idx2 = self.data_idx[i]+self.input_width
        #
        #     X_mb =
        #
        #     Y_mb = self.data[idx1:idx2]
        #     X_mb = Y_mb.copy()
        #     M_mb = self.data_m[idx1:idx2]
        #     Z_mb = uniform_sampler(0, 0.01, shape=X_mb.shape)
        #     X_mb = M_mb*X_mb + (1-M_mb)*Z_mb
        #     X_mb[M_mb == 0] = np.nan
        #     x = np.append(x, [X_mb], axis=0)
        #     y = np.append(y, [Y_mb], axis=0)

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(x)
        print('yyyyyyyyyyyyyyyyyyyyyyyyyyy')
        print(y)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return
