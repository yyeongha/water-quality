import os
import numpy as np

class MissData(object):
    def __init__(self, load_dir=None):
        print('load_dir',load_dir)
        if load_dir:
            # np_load_old = np.load
            # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
            self.missarr = np.load(os.path.join(load_dir, 'miss.npy'))
            self.idxarr = np.load(os.path.join(load_dir, 'idx.npy'))
    def make_missdata(self, data_x, missrate=None):
        data = data_x.copy()
        rows, cols = data_x.shape
        total_no = rows*cols
        total_miss_no = np.round(total_no*missrate).astype(int)
        total_idx = self.idxarr.shape[0]

        idxarr = self.idxarr
        missarr = self.missarr
        print('self.idxarr',self.idxarr)
        print('self.missarr',self.missarr)
        miss_no = 0
        cum_no = self.idxarr[:,3:4]
        cum_no = cum_no.reshape((total_idx))
        cum_sum = np.sum(cum_no)
        print('cum_no =', cum_no)
        print('cum_sum =', cum_sum)
        print('totla_idx = ', total_idx)
        while True:
            # print ('=====================================')
            loc_count = np.around(np.random.random()*cum_sum)
            # print('loc_count =', loc_count)
            idx = len(cum_no[cum_no <= loc_count])-1
            # print('idx =', idx)
            # print('idxarr[idx]',idxarr[idx])
            # print('cum_no[cum_no <= loc_count]',cum_no[cum_no <= loc_count])
            startnan = idxarr[idx][0]
            nanlen = idxarr[idx][2]
            loc = np.around(np.random.random()*(rows-nanlen)).astype(int)
            # print(loc_count, idx)
            #data_copy = data[loc:loc+nanlen].copy()
            data_copy = data[loc:loc+nanlen]
            # print('startnan=', startnan)
            #isnan = missarr[startnan:startnan+nanlen].copy()
            isnan = missarr[startnan:startnan+nanlen]
            # print('isnan =',isnan)
            miss_no += idxarr[idx][1]
            if (miss_no > total_miss_no):
                break
            data_copy[isnan==1] = np.nan
            data[loc:loc+nanlen] = data_copy
        return data
    # 테스트용
    # def save(self, data, max_tseq, save_dir='save'):    

    # 실전용
    def save(data, max_tseq, save_dir='save'):
        print('data',data)
        print('max_tseq',max_tseq)

        no, dim = data.shape
        #print((no, dim))
        isnan = np.isnan(data).astype(int)
        isany = np.any(isnan, axis=1).astype(int)
        shifted = np.roll(isany, 1)
        #print(isnan)
        #print(isany.astype(int))
        #print(shifted)
        startnan = ((isany == 1) & (shifted ==0)).astype(int)
        #print(startnan)
        group = startnan.cumsum()
        group = group*isany
        #print(group)
        n = np.max(group)
        #print(n)
        missarr = None
        cum_no = 0
        rowidx = 0
        for i in range(1, n+1):
            g = (group == i).astype(int)
            i = np.argmax(g)
            rows = g.sum()
            #print(len)
            #print(i)
            #print(type(missarr))
            if rows <= max_tseq:
                nanseq = isnan[i:i+rows, :]
                no = np.sum(nanseq)
                #print(no)
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
        print('miss_data file saved')
