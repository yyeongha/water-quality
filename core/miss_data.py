#####################################################################################
# 결측 데이터를 생성하고 이를 파일로 저장하거나 로드하는 기능
# 이를 통해 GAIN 와 같은 모델이 결측 데이터를 학습하고 예측할 수 있도록 지원
# 결측 패턴을 파일로 저장함으로써 동일한 패턴의 결측 데이터를 재생성할 수 있어 일관된 평가 및 학습이 가능
#####################################################################################

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

class MissData(object):
    def __init__(self, load_dir=None):
        if load_dir:

            self.missarr = np.load(os.path.join(load_dir, 'miss.npy'))
            self.idxarr = np.load(os.path.join(load_dir, 'idx.npy'))
            print('MissData : ', load_dir, " miss : ", self.missarr.shape)

    # 주어진 데이터에 결측 데이터를 생성
    def make_missdata(self, data_x, missrate=0.2):
        data = data_x.copy() 
        rows, cols = data_x.shape
        total_no = rows * cols
        total_miss_no = np.round(total_no * missrate).astype(int)
        total_idx = self.idxarr.shape[0]
        idxarr = self.idxarr
        missarr = self.missarr
        miss_no = 0
        cum_no = self.idxarr[:, 3:4]
        cum_no = cum_no.reshape((total_idx))
        cum_sum = np.max(cum_no)

        while True:
            loc_count = np.around(np.random.random() * cum_sum)
            idx = len(cum_no[cum_no <= loc_count]) - 1
            startnan = idxarr[idx][0]
            nanlen = idxarr[idx][2]
            loc = np.around(np.random.random() * (rows - nanlen)).astype(int)
            data_copy = data[loc:loc + nanlen]
            isnan = missarr[startnan:startnan + nanlen]
            miss_no += idxarr[idx][1]
            if (miss_no > total_miss_no):
                break
            data_copy[isnan == 1] = np.nan
            data[loc:loc + nanlen] = data_copy

        return data

    # 결측데이터의 결측 패턴을 저장
    def save(data, max_tseq, save_dir='save'):
        no, dim = data.shape
        isnan = np.isnan(data).astype(int)
        isany = np.any(isnan, axis=1).astype(int)
        shifted = np.roll(isany, 1)
        shifted[0] = 1

        startnan = ((isany == 1) & (shifted == 0)).astype(int)
        group = startnan.cumsum()
        group = group * isany
        n = np.max(group)
        missarr = None
        cum_no = 0
        rowidx = 0

        idxarr = None

        for i in range(1, n + 1):
            g = (group == i).astype(int)
            i = np.argmax(g)
            rows = g.sum()
            if rows <= max_tseq:
                nanseq = isnan[i:i + rows, :]
                no = np.sum(nanseq)

                if missarr is None:
                    missarr = nanseq
                    idxarr = np.array([[rowidx, no, rows, cum_no]])
                else:
                    missarr = np.concatenate((missarr, nanseq))
                    idxarr = np.concatenate((idxarr, [[rowidx, no, rows, cum_no]]), axis=0)
                cum_no += no
                rowidx += rows

        miss_npy_file = os.path.join(save_dir, 'miss.npy')
        idx_npy_file = os.path.join(save_dir, 'idx.npy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if missarr is not None:
            np.save(miss_npy_file, missarr)
        else:
            return False
        if idxarr is not None:
            np.save(idx_npy_file, idxarr)
        else:
            return False

        return True

# 데이터의 결측 패턴을 저장
def missdata_save(data, max_tseq, save_dir='save'):
    no, dim = data.shape
    isnan = np.isnan(data).astype(int)
    isany = np.any(isnan, axis=1).astype(int)
    shifted = np.roll(isany, 1)
    shifted[0] = 1

    startnan = ((isany == 1) & (shifted == 0)).astype(int)
    group = startnan.cumsum()
    group = group * isany
    n = np.max(group)
    missarr = None
    cum_no = 0
    rowidx = 0
    for i in range(1, n + 1):
        g = (group == i).astype(int)
        i = np.argmax(g)
        rows = g.sum()
        if rows <= max_tseq:
            nanseq = isnan[i:i + rows, :]
            no = np.sum(nanseq)
            if missarr is None:
                missarr = nanseq
                idxarr = np.array([[rowidx, no, rows, cum_no]])
            else:
                missarr = np.concatenate((missarr, nanseq))
                idxarr = np.concatenate((idxarr, [[rowidx, no, rows, cum_no]]), axis=0)
            cum_no += no
            rowidx += rows

    miss_npy_file = os.path.join(save_dir, 'miss.npy')
    idx_npy_file = os.path.join(save_dir, 'idx.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(miss_npy_file, missarr)
    np.save(idx_npy_file, idxarr)
