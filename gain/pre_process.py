import glob
import datetime
import math
import pandas as pd
import numpy as np


class PreProcess:
    def __init__(self, input, target, time, target_all, fill_cnt=0):

        # member variable
        self.fill_cnt = fill_cnt
        self.time = time
        self.df_raw_list = []
        self.time_df = None
        self.day_sin = None
        self.day_cos = None
        self.year_sin = None
        self.year_cos = None

        # check date_name type (file or directory)
        file_list = glob.glob(input + "/*.xlsx")
        
        # if data_name is file type
        if len(file_list) == 0:
            df = pd.read_excel(input)

            # interpolate fill
            fill_df = self.getFillDf(df)

            # get dataframe columns size
            min_size = fill_df.columns.size

            # initial day_sin, day_cos, year_sin, yaer_cos
            self.initSinCos(df)
            
        # if data_name is directory type
        else:
            # null padding dataframe
            dn = pd.DataFrame(data={'line': ['0']})

            # init variable
            df_list = []
            min_size = 999
            
            # loop file
            for f in file_list:
                df = pd.read_excel(f)
                self.df_raw_list.append(self.getFillDf(df))

                # initial day_sin, day_cos, year_sin, yaer_cos
                self.initSinCos(df)

                # interpolate fill and append dataframe
                df_list.append(self.getFillDf(df))

                # append dataframe
                df_list.append(dn)

                # validate logic
                # update min_size
                if min_size > df.columns.size:
                    min_size = df.columns.size

            # concat dataframe and drop null padding dataframe 
            fill_df = pd.concat(df_list).drop('line', axis=1)

        # if target_all True then use target_all_list
        # get all column name list (except 0, 1)
        # 0 is 측정시간
        # 1 is 측정소명
        target_all_list = [n for n in range(2, min_size)]

        # init self.target and self.target_name         
        if target_all:
            self.target = target_all_list
            self.target_name = fill_df.columns.tolist()[2:min_size]
        else:
            self.target = target
            self.target_name = []
            for t in target:
                self.target_name.append(fill_df.columns.tolist()[t])

        # init self.group_cnt
        # "add 4" means "day_sin, day_cos, year_sin, year_cos" 
        # this is important variable
        self.group_cnt = self.time * (len(self.target) + 4)

        # init self.target_df
        self.target_df = self.getTargetDf(fill_df)

        # Just debug, remove code after complete development
        print('[debug] fill_df = ', fill_df)
        print('[debug] self.target = ', self.target)
        print('[debug] self.target_name = ', self.target_name)
        print('[debug] self.group_cnt = ', self.group_cnt)
        print('[debug] self.target_df = ', self.target_df)
        print('[debug] 시간 = ', self.time)
        print('[debug] 컬럼수 = ', len(self.target))
        print('[debug] 컬럼수(sin, cos 포함) = ', len(self.target)+4)
        print('[debug] reshape 후 생성되는 열 개수 = ', self.group_cnt)


    # initial "sin, cos" each day, year
    def initSinCos(self, df):
        self.time_df = df.iloc[:, 0:1]
        # date_time = pd.to_datetime(df['측정날짜'], format='%Y.%m.%d %H:%M:%S', utc=True)
        date_time = pd.to_datetime(df['측정날짜'], format='%Y.%m.%d %H:%M:%S', utc=True)
        timestamp_sr = date_time.map(datetime.datetime.timestamp)
        day1 = 24 * 60 * 60
        week = day1 * 7
        year1 = (365.2425) * day1
        self.day_sin = np.sin(timestamp_sr * (2 * np.pi / day1))
        self.day_cos = np.cos(timestamp_sr * (2 * np.pi / day1))
        self.year_sin = np.sin(timestamp_sr * (2 * np.pi / day1))
        self.year_cos = np.cos(timestamp_sr * (2 * np.pi / day1))


    # interpolate fill
    # this function use init function
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


    # read target parameter and remove not use column
    def getTargetDf(self, df):
        return df.iloc[:, self.target]


    # get pure target dataframe
    def getSelfTargetDf(self):
        return self.target_df


    # get target name list
    def getTargetName(self):
        self.target_name.append('day_sin')
        self.target_name.append('day_cos')
        self.target_name.append('year_sin')
        self.target_name.append('year_cos')
        return self.target_name


    # ???
    def getRelTargetDf(self, df):
        return df.loc[:, self.target_name]


    # labeling dataframe
    def getLabelDf(self, target_df):
        x = target_df.isna().any(axis='columns').astype(int)
        y = target_df.isna().any(axis='columns').astype(int).shift(periods=1, fill_value=0)
        z = pd.concat([x, y], axis=1).any(axis='columns').astype(int).cumsum()
        raw_label_df = pd.concat([x, z], axis=1)
        raw_label_df.columns = ['is_nan', 'group']
        new_df = pd.concat([target_df, raw_label_df], axis=1)
        label_df = new_df.where(new_df['is_nan'] != 1).dropna()
        return label_df


    # group by label
    def getSizeSr(self, label_df):
        size_sr = label_df.groupby(label_df['group']).size()
        return size_sr


    # get discard for reshape
    def getDiscard(self, target_df):
        return ( len(target_df) * (len(self.target) + 4) ) // ( self.group_cnt ) # 121


    # discard row for reshape
    def sliceDf(self, target_df, discard):
        return target_df[:(discard * self.group_cnt) // (len(self.target) + 4)]


    # shift dataframe
    def shiftDf(self, label_df, size_sr):
        target_idx_list = size_sr.where(size_sr >= self.time).dropna().index.tolist()
        target_df_list = []
        concat_list = []

        for idx in target_idx_list:
            output_df = label_df.where(label_df['group'] == idx).dropna()
            total_cnt = len(label_df.where(label_df['group'] == idx).dropna())
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


    # Reshpe dataframe
    def getReshapeNp(self, output_np):
        output_np = output_np.reshape(-1, self.group_cnt)
        return output_np


    # Dataframe to numpy
    def getNp(self, output_df):
        output_np = output_df.to_numpy()
        return output_np


    # Discard row for reshape
    def getDiscardDf(self, df):
        discard = self.getDiscard(df)
        discard_df = self.sliceDf(df, discard)
        return discard_df


    # Make group shift data for train
    def getDataFrame(self):

        # add column day_sin, day_cos, year_sin, year_cos
        self.target_df['day_sin'] = self.day_sin
        self.target_df['day_cos'] = self.day_cos
        self.target_df['year_sin'] = self.year_sin
        self.target_df['year_cos'] = self.year_cos

        # labeling dataframe
        label_df = self.getLabelDf(self.target_df)

        # group by label
        size_sr = self.getSizeSr(label_df)

        # shift dataframe
        target_df = self.shiftDf(label_df, size_sr)

        # get discard for reshape
        discard = self.getDiscard(target_df)

        # discard row for reshape
        output_df = self.sliceDf(target_df, discard)
       
        return output_df


    # Make data for test
    def getRawDataFrame(self):
        # file
        if len(self.df_raw_list) == 0:
            target_df = self.target_df
            discard = self.getDiscard(target_df)
            output_df = self.sliceDf(target_df, discard)
            return output_df

        # directory (not use)
        else:
            output_np_list = []
            for fill_df in self.df_raw_list:
                target_df = self.getRelTargetDf(fill_df)
                discard = self.getDiscard(target_df)
                output_df = self.sliceDf(target_df, discard)
                output_np = self.getNp(output_df)
                output_np_list.append(output_np)
            return output_np_list


    # add column "측정시간" and save excel
    def addTimeFormat(self, imputed_df, output_path):
        result_df = pd.concat([self.time_df, imputed_df], axis=1)
        result_df.drop(['day_sin', 'day_cos', 'year_sin', 'year_cos'], axis='columns', inplace=True)
        result_df.to_excel(output_path, index=False)


    # reshape dataframe
    def reverseReShape(self, input_np):
        return input_np.reshape(-1, len(self.target) + 4)

    
    # get mean and standard
    def getMeanAndStand(self):
        df = self.target_df

        # 평균
        M = df.mean() 

        # 표준편차
        S = df.std(ddof=0)

        return M, S


    # get normalization dataframe
    def normalization(self, df, M, S):
        normalization_df = (df - M)/S
        normalization_df = normalization_df
        return normalization_df


    # get denormalization dataframe
    def denormalization(self, normalization_df, M, S):
        df = normalization_df * S + M
        df = df.fillna(0)
        return df 


    # split dataframe for make train data and test data
    def splitDf(self, df, percent=0.7):
        data_100 = len(df)
        split = math.ceil(data_100 * percent)
        df_70 = df[:split]
        df_30 = df[split:]
        return df_70, df_30