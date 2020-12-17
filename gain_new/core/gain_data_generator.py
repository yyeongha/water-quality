import numpy as np

from tensorflow import keras
from core.utils import *
from core.miss_data import MissData


class GainDataGenerator(keras.utils.Sequence):
    'Generates data for GAIN'
    def __init__(self,
                 data_list,
                 batch_size=32,
                 input_width=24*3,
                 label_width=24*3,
                 shift=0,
                 fill_no=4,
                 miss_rate=0.2,
                 hint_rate=0.9,
                 normalize=True,
                 miss_pattern=None,
                 alpha=100.):
        'Initialization'
        window_size = input_width
        
        # interpollation
        filled_data = []
        for data in data_list:
            data = interpolate(data, max_gap=fill_no)
            filled_data.append(data)
            
        data_list = filled_data
        
        # whole data
        self.data = np.concatenate(data_list)

        # TO-DO
        
        # pre calculation for  sequence data
        last_cum = 0
        cums = []
        for data in data_list:
            # nan 변환
            isnan = np.isnan(data)
            # 하나라도 nan 인 경우를 잡기위함
            isany = np.any(isnan, axis=1)
            shifted = np.roll(isany, 1)
            shifted[0] = True # set to nan
            start_seq = ((isany == False) & (shifted == True)).astype(int)
            cum = start_seq.cumsum()
            cum += last_cum
            last_cum = np.max(cum)
            cum[isany] = 0
            cums.append(cum)
            
        
        # normlize for spam
        if normalize:
            self.data, norm_param = normalization(self.data)
        #print(norm_param)
        
        # Define mask matrix
        if miss_pattern is None:
            self.data_m = binary_sampler(1-miss_rate, self.data.shape)
        else:
            #MissData.save(self.data, max_tseq = 12)
            self.miss = MissData(load_dir='save')
            self.miss_rate = miss_rate
            miss_data = self.miss.make_missdata(self.data, self.miss_rate)
            self.data_m = 1. - np.isnan(miss_data).astype(float)
            
            self.data_m_rand = binary_sampler(1-(miss_rate/10.), self.data.shape)
            self.data_m[self.data_m_rand==0.] = 0.
        self.miss_pattern = miss_pattern
        
        # sequence data
        self.ids = np.concatenate(cums)
        data_idx = np.empty((0), dtype=int)
        for i in range(1, last_cum+1):
            seq_len = (self.ids == i).sum()
            start_id = np.argmax(self.ids == i)
            # possible data number in seqeunce
            time_len = seq_len - window_size + 1
            start_ids = np.arange(start_id, start_id+time_len)
            data_idx = np.append(data_idx, start_ids)
        # start index set for sequence data
        self.data_idx = data_idx
        self.input_width = input_width
        self.no = len(data_idx)
        
        #print('self.no = ', self.no)
        
        self.batch_size = batch_size
        
        # random shuffling  index
        self.batch_idx = sample_batch_index(self.no, self.no)
        self.batch_id = 0
        self.shape = (batch_size,self.input_width)+self.data.shape[1:]
        #self.hint_rate = hint_rate
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(128/self.batch_size)
        #return 2
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        #print('index =', index)
        # Sample batch
        x = np.empty((0, self.input_width, self.data.shape[1]))
        #m = np.empty((0, self.input_width, self.data.shape[1]))
        #h = np.empty((0, self.input_width, self.data.shape[1]))
        y = np.empty((0, self.input_width, self.data.shape[1]))
        #print(x.shape)
        #print(self.data.shape)
        #print(self.input_width)
        #self.batch_idx = sample_batch_index(self.no, self.batch_size)
        for cnt in range(0, self.batch_size):
            i = self.batch_idx[self.batch_id]
            self.batch_id += 1
            #self.batch_id %= self.batch_size
            self.batch_id %= self.no
            if self.miss_pattern and (self.batch_id == 0):
                self.batch_idx = sample_batch_index(self.no, self.no)
                miss_data = self.miss.make_missdata(self.data, self.miss_rate)
                self.data_m = 1. - np.isnan(miss_data).astype(float)
                self.data_m_rand = binary_sampler(1-self.miss_rate/10., self.data.shape)
                self.data_m[self.data_m_rand==0.] = 0.
            idx1 = self.data_idx[i]
            idx2 = self.data_idx[i]+self.input_width
            #print(idx1, idx2)
        
            Y_mb = self.data[idx1:idx2].copy()
            X_mb = Y_mb.copy()
            M_mb = self.data_m[idx1:idx2]
            Z_mb = uniform_sampler(0, 0.01, shape=X_mb.shape)
            X_mb = M_mb*X_mb + (1-M_mb)*Z_mb
            #H_mb_temp = binary_sampler(self.hint_rate, shape=X_mb.shape)
            #H_mb = M_mb * H_mb_temp
            X_mb[M_mb == 0] = np.nan
            Y_mb[M_mb == 1] = np.nan
            x = np.append(x, [X_mb], axis=0)
            #m = np.append(m, [M_mb], axis=0)
            #h = np.append(h, [H_mb], axis=0)
            y = np.append(y, [Y_mb], axis=0)
        #return [x, m, h], y
        return x, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return