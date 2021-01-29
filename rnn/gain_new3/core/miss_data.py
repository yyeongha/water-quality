import os
import numpy as np
import pandas as pd


class MissData(object):
    def __init__(self, load_dir=None):
        #print('MissData : ', load_dir)
        if load_dir:
            #print('MissData : ', load_dir)
            self.missarr = np.load(os.path.join(load_dir, 'miss.npy'))
            self.idxarr = np.load(os.path.join(load_dir, 'idx.npy'))

    def make_missdata(self, data_x, missrate=0.2):
        data = data_x.copy()
        rows, cols = data_x.shape
        total_no = rows * cols
        total_miss_no = np.round(total_no * missrate).astype(int)
        total_idx = self.idxarr.shape[0]
        idxarr = self.idxarr
        missarr = self.missarr
        # print(total_miss_no)
        miss_no = 0
        cum_no = self.idxarr[:, 3:4]
        cum_no = cum_no.reshape((total_idx))
        cum_sum = np.max(cum_no)
        # print(cum_no)
        # print(total_idx)
        while True:
            loc_count = np.around(np.random.random() * cum_sum)
            # print('loc_count =', loc_count)
            idx = len(cum_no[cum_no <= loc_count]) - 1
            # print(cum_no[cum_no <= loc_count])
            # print('idx =', idx)
            startnan = idxarr[idx][0]
            nanlen = idxarr[idx][2]
            loc = np.around(np.random.random() * (rows - nanlen)).astype(int)
            # print('loc =', loc)
            # print(loc_count, idx)
            # print(idxarr[idx])
            # data_copy = data[loc:loc+nanlen].copy()
            # print(data.shape)
            data_copy = data[loc:loc + nanlen]
            # print('startnan=', startnan)
            # isnan = missarr[startnan:startnan+nanlen].copy()
            isnan = missarr[startnan:startnan + nanlen]
            # print('isnan =',isnan)
            miss_no += idxarr[idx][1]
            if (miss_no > total_miss_no):
                break
            data_copy[isnan == 1] = np.nan
            data[loc:loc + nanlen] = data_copy
        # print('miss_data =', data)
        return data

    def save(data, max_tseq, save_dir='save'):
        #save_dir = self.save_directory
        no, dim = data.shape
        #print((no, dim))
        #print(type(data))

        #print(pd.DataFrame(data).head())

        isnan = np.isnan(data).astype(int)
        #print(np.any(isnan, axis=1).astype(int))
        isany = np.any(isnan, axis=1).astype(int)
        shifted = np.roll(isany, 1)
        shifted[0] = 1
        #print(isnan)
        #print(isany.astype(int))
        # print(shifted)
        startnan = ((isany == 1) & (shifted == 0)).astype(int)
        #print(startnan)
        group = startnan.cumsum()
        #print(group)
        group = group * isany
        #         print(group)
        n = np.max(group)

        #print('n:',n)

        #         print(n)
        missarr = None
        cum_no = 0
        rowidx = 0

        idxarr = None

        for i in range(1, n + 1):
            g = (group == i).astype(int)
            i = np.argmax(g)
            rows = g.sum()
            # print(len)
            # print(i)
            # print(type(missarr))
            if rows <= max_tseq:
                nanseq = isnan[i:i + rows, :]
                no = np.sum(nanseq)
                # print(no)
                if missarr is None:
                    missarr = nanseq
                    idxarr = np.array([[rowidx, no, rows, cum_no]])
                else:
                    missarr = np.concatenate((missarr, nanseq))
                    idxarr = np.concatenate((idxarr, [[rowidx, no, rows, cum_no]]), axis=0)
                cum_no += no
                rowidx += rows

        #print(idxarr)
        miss_npy_file = os.path.join(save_dir, 'miss.npy')
        idx_npy_file = os.path.join(save_dir, 'idx.npy')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #print('miss : ',missarr.shape)
        #print('idx : ',idxarr.shape)

        if missarr is not None:
            np.save(miss_npy_file, missarr)
        else:
            return False
        if idxarr is not None:
            np.save(idx_npy_file, idxarr)
        else:
            return False

        return True
        #print('miss_data file saved')




def missdata_save(data, max_tseq, save_dir='save'):
    #save_dir = self.save_directory
    no, dim = data.shape
    # print((no, dim))
    isnan = np.isnan(data).astype(int)
    isany = np.any(isnan, axis=1).astype(int)
    shifted = np.roll(isany, 1)
    shifted[0] = 1
    # print(isnan)
    # print(isany.astype(int))
    # print(shifted)
    startnan = ((isany == 1) & (shifted == 0)).astype(int)
    # print(startnan)
    group = startnan.cumsum()
    group = group * isany
    #         print(group)
    n = np.max(group)
    #         print(n)
    missarr = None
    cum_no = 0
    rowidx = 0
    for i in range(1, n + 1):
        g = (group == i).astype(int)
        i = np.argmax(g)
        rows = g.sum()
        # print(len)
        # print(i)
        # print(type(missarr))
        if rows <= max_tseq:
            nanseq = isnan[i:i + rows, :]
            no = np.sum(nanseq)
            # print(no)
            if missarr is None:
                missarr = nanseq
                idxarr = np.array([[rowidx, no, rows, cum_no]])
            else:
                missarr = np.concatenate((missarr, nanseq))
                idxarr = np.concatenate((idxarr, [[rowidx, no, rows, cum_no]]), axis=0)
            cum_no += no
            rowidx += rows

        #print(idxarr)
    miss_npy_file = os.path.join(save_dir, 'miss.npy')
    idx_npy_file = os.path.join(save_dir, 'idx.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(miss_npy_file, missarr)
    np.save(idx_npy_file, idxarr)
