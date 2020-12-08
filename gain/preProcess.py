import glob
import datetime
import math
import pandas as pd
import numpy as np


class PreProcess:
    def __init__(self, input, target, time, target_all, fill_cnt=0):
        self.fill_cnt = fill_cnt
        self.time = time
        self.df_raw_list = []
        self.time_df = None
        self.day_sin = None
        self.day_cos = None
        self.year_sin = None
        self.year_cos = None

        file_list = glob.glob(input + "/*.xlsx")
        print('[debug] len(file_list) = ', len(file_list))
        
        # file
        if len(file_list) == 0:
            df = pd.read_excel(input)
            fill_df = self.getFillDf(df)
            min_size = fill_df.columns.size

            # 측정시간 추출
            self.time_df = df.iloc[:, 0:1]
            date_time = pd.to_datetime(df['측정날짜'], format='%Y.%m.%d %H:%M:%S')
            timestamp_sr = date_time.map(datetime.datetime.timestamp)
            day1 = 24 * 60 * 60
            week = day1 * 7
            year1 = (365.2425) * day1
            self.day_sin = np.sin(timestamp_sr * (2 * np.pi / day1))
            self.day_cos = np.cos(timestamp_sr * (2 * np.pi / day1))
            self.year_sin = np.sin(timestamp_sr * (2 * np.pi / day1))
            self.year_cos = np.cos(timestamp_sr * (2 * np.pi / day1))

        # directory
        else:
            dn = pd.DataFrame(data={'line': ['0']}) # null padding
            df_list = []
            min_size = 999
            for f in file_list:
                df = pd.read_excel(f)
                self.df_raw_list.append(self.getFillDf(df))
                df_list.append(self.getFillDf(df))
                df_list.append(dn)
                if min_size > df.columns.size:
                    min_size = df.columns.size
            fill_df = pd.concat(df_list).drop('line', axis=1)

        # get all column name list (except 0, 1)
        target_all_list = [n for n in range(2, min_size)]
        print('[debug] target_all_list = ', target_all_list)
        print('[debug] min_size = ', min_size)
             
        if target_all:
            self.target = target_all_list
            self.target_name = fill_df.columns.tolist()[2:min_size]
        else:
            self.target = target
            self.target_name = []
            for t in target:
                self.target_name.append(fill_df.columns.tolist()[t])

        print('[debug] self.target = ', self.target)

        self.group_cnt = self.time * (len(self.target) + 4) # 4 is "sin, cos, sin, cos"
        print('[debug] self.group_cnt (y) = ', self.group_cnt)

        self.target_df = self.getTargetDf(fill_df)
        print('[debug] self.target_df = ', self.target_df)

    def getTargetName(self):
        self.target_name.append('day_sin')
        self.target_name.append('day_cos')
        self.target_name.append('year_sin')
        self.target_name.append('year_cos')
        return self.target_name

    def getFillDf(self, df):
        if self.fill_cnt != 0:
            mask = df.copy()
            for i in df.columns: 
                dfx = pd.DataFrame( df[i] )
                dfx['new'] = ((dfx.notnull() != dfx.shift().notnull()).cumsum())
                dfx['ones'] = 1
                mask[i] = (dfx.groupby('new')['ones'].transform('count') < self.fill_cnt + 1) | df[i].notnull()
            df = df.interpolate().bfill()[mask]
        return df 

    def getTargetDf(self, df):
        return df.iloc[:, self.target]

    def getRelTargetDf(self, df):
        return df.loc[:, self.target_name]

    def getLabelDf(self, target_df):
        x = target_df.isna().any(axis='columns').astype(int)
        y = target_df.isna().any(axis='columns').astype(int).shift(periods=1, fill_value=0)
        z = pd.concat([x, y], axis=1).any(axis='columns').astype(int).cumsum()
        raw_label_df = pd.concat([x, z], axis=1)
        raw_label_df.columns = ['is_nan', 'group']
        new_df = pd.concat([target_df, raw_label_df], axis=1)
        label_df = new_df.where(new_df['is_nan'] != 1).dropna()
        return label_df

    def getSizeSr(self, label_df):
        size_sr = label_df.groupby(label_df['group']).size()
        return size_sr

    def getDiscard(self, target_df):
        print('len(target_df) = ', len(target_df))
        print('len(self.target) = ', len(self.target))
        print('( self.group_cnt ) = ', ( self.group_cnt ))
        return ( len(target_df) * (len(self.target) + 4) ) % ( self.group_cnt )

    def sliceDf(self, target_df, discard):
        return target_df[:len(target_df) - int(discard / len(self.target))]

    def shiftDf(self, label_df, size_sr):
        target_idx_list = size_sr.where(size_sr >= self.time).dropna().index.tolist()
        target_df_list = []
        concat_list = []

        # debug
        print('[debug] group_cnt = ', len(target_idx_list))
        
        for idx in target_idx_list:
            output_df = label_df.where(label_df['group'] == idx).dropna()
            total_cnt = len(label_df.where(label_df['group'] == idx).dropna())

            # debug
            print('[debug] total_cnt = ', total_cnt)

            target_df_list.append({ 'total_cnt': total_cnt, 'output_df': output_df })
        for t in target_df_list:
            for n in range(0, t['total_cnt']-self.time+1):
                split_df = t['output_df'][n:self.time+n]
                concat_list.append(split_df)
                
        # validate
        if len(concat_list) == 0:
            print('[fail] no groups were created. reduce the time option')
            exit(0)

        target_df = pd.concat(concat_list, ignore_index=True).drop('is_nan', axis=1).drop('group', axis=1)
        return target_df

    def getReshapeNp(self, output_np):
        output_np = output_np.reshape(-1, self.group_cnt)
        return output_np

    def getNp(self, output_df):
        output_np = output_df.to_numpy()
        return output_np

    # 삭제 예정
    def getDataSet(self):
        label_df = self.getLabelDf(self.target_df) # diff
        size_sr = self.getSizeSr(label_df) # diff
        target_df = self.shiftDf(label_df, size_sr) # diff
        discard = self.getDiscard(target_df)
        output_df = self.sliceDf(target_df, discard)
        output_np = self.getNp(output_df)
        return output_np

    def getDiscardDf(self, df):
        discard = self.getDiscard(df)
        discard_df = self.sliceDf(df, discard)
        return discard_df

    # custom
    def getDataFrame(self):

        # sin cos 추가
        self.target_df['day_sin'] = self.day_sin
        self.target_df['day_cos'] = self.day_cos
        self.target_df['year_sin'] = self.year_sin
        self.target_df['year_cos'] = self.year_cos

        # temp
        # self.target_df.to_excel('./output/before_shift.xlsx', index=False)
        self.target_df.to_excel('./output/before_shift.xlsx', index=False)
        # print('## self.target_df => ', self.target_df)

        label_df = self.getLabelDf(self.target_df) # diff
        size_sr = self.getSizeSr(label_df) # diff
        target_df = self.shiftDf(label_df, size_sr) # diff

        # print('## target_df => ', target_df)

        discard = self.getDiscard(target_df)
        output_df = self.sliceDf(target_df, discard)
        # return output_df

        # debug
        # print('## output_df => ', output_df)
        return output_df

    # custom
    def getRawDataFrame(self):
        # file
        if len(self.df_raw_list) == 0:
            target_df = self.target_df
            discard = self.getDiscard(target_df)
            output_df = self.sliceDf(target_df, discard)
            return output_df

        # directory (차후 점검)
        else:
            output_np_list = []
            for fill_df in self.df_raw_list:
                target_df = self.getRelTargetDf(fill_df)
                discard = self.getDiscard(target_df)
                output_df = self.sliceDf(target_df, discard)
                output_np = self.getNp(output_df)
                output_np_list.append(output_np)
            return output_np_list

    # 삭제 예정
    def getRawDataSet(self):
        # file
        if len(self.df_raw_list) == 0:
            target_df = self.target_df
            discard = self.getDiscard(target_df)
            output_df = self.sliceDf(target_df, discard)
            output_np = self.getNp(output_df)
            return output_np

        # directory
        else:
            output_np_list = []
            for fill_df in self.df_raw_list:
                target_df = self.getRelTargetDf(fill_df)
                discard = self.getDiscard(target_df)
                output_df = self.sliceDf(target_df, discard)
                output_np = self.getNp(output_df)
                output_np_list.append(output_np)
            return output_np_list

    # def npToExcel(self, input_np, save_path, timeFormat=False):
    #     df = pd.DataFrame(data=input_np)
    #     # df = input_np # temp

    #     if timeFormat:
    #         hour_add = datetime.timedelta(hours = 1)
    #         target = datetime.datetime(2020, 1, 1, 0, 0) - hour_add
    #         date_list = []
    #         for day in range(0, len(df)):
    #             target = (target + hour_add)
    #             date_list.append(target.strftime("%m.%d %H:%S"))
    #         df.insert(loc=0, column='date', value=date_list)

    #     df.to_excel(save_path, index=False)

    def addTimeFormat(self, imputed_df, output_path):
        result_df = pd.concat([self.time_df, imputed_df], axis=1)
        result_df.drop(['day_sin', 'day_cos', 'year_sin', 'year_cos'], axis='columns', inplace=True)
        result_df.to_excel(output_path, index=False)

    def reverseReShape(self, input_np):
        return input_np.reshape(-1, len(self.target) + 4)

    def getMeanAndStand(self, df):
        M = df.mean() # 평균
        S = df.std(ddof=0) # 표준편차
        return M, S

    def normalization(self, df, M, S):
        # M = df.mean() # 평균
        # S = df.std(ddof=0) # 표준편차
        normalization_df = (df - M)/S
        # normalization_df = normalization_df.fillna(0)
        normalization_df = normalization_df
        return normalization_df
        # return normalization_df, M, S

    def denormalization(self, normalization_df, M, S):
        df = normalization_df * S + M
        df = df.fillna(0)
        return df 

    def splitDf(self, df):
        data_100 = len(df)
        split = math.ceil(data_100 * 0.7)
        df_70 = df[:split]
        df_30 = df[split:]
        return df_70, df_30