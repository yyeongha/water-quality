#####################################################################################
# 녹조 조류 모니터링과 오염원 분석을 위한 시계열 데이터를 생성하고 처리하는 과정
# 주어진 디렉토리에서 엑셀 파일들을 읽어와 시계열 데이터 프레임을 만들고,
# 필요에 따라 보간(interpolation)을 수행하여 누락된 데이터를 채우는 역할
#####################################################################################

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime
import calendar
from time import time
import time
import matplotlib.pyplot as plt

# 일별 시계열 데이터를 생성하는 함수
def make_timeseries_by_day(df, interpolation=None):
    # 보간이 필요하지 않은 경우, 원본 데이터프레임을 반환
    if interpolation[1] == False:
        return df
    time_day = [] # 시간 조정을 저장할 리스트

    year = int(str(df.iloc[0, 0])[0:4]) # 첫 번째 행에서 연도를 추출
    
    # 데이터프레임에 행이 하나만 있는 경우
    if df.shape[0] == 1:
        print('row in')
        ori_time = str(df.iloc[0, 0]) + "-06-15 12:00" # 임의의 날짜와 시간 추가
        ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M') # 날짜 형식으로 변환
        df.iloc[0, 0] = ori_time # 날짜 형식으로 변환
        return df
    # 데이터프레임에 행이 두 개 있는 경우
    elif df.shape[0] == 2:
        #print('ddddddddddd')
        #print(df.iloc[0, 0], df.iloc[1, 0])
        # 두 행의 값이 동일한 경우
        if df.iloc[0, 0] == df.iloc[1, 0]:
            ori_time = str(df.iloc[0, 0]) + "-03-15 12:00"
            ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
            df.iloc[0, 0] = ori_time

            ori_time = str(df.iloc[1, 0]) + "-09-15 12:00"
            ori_time = datetime.datetime.strptime(ori_time, '%Y-%m-%d %H:%M')
            df.iloc[1, 0] = ori_time
            return df

    div_cnt = df[df.columns[0]].value_counts().sort_index().tolist()  # 월별 값의 개수 리스트

    for i in range(len(div_cnt)):

        month = calendar.monthrange(year, i + 1) # 월의 날짜 수 계산
        for j in range(div_cnt[i]):
            time_val = datetime.timedelta(days=(month[1] / div_cnt[i] * j) + 1) # 시간 값 계산

            time_day.append(
                '-' + str(time_val.days) + ' ' + str(time.strftime('%H:00:00', time.gmtime(time_val.seconds))))

    df.iloc[:, 0] = df.iloc[:, 0] + time_day # 원본 데이터프레임에 시간 리스트 추가

    return df # 수정된 데이터프레임 반환

# 시계열 데이터를 생성하는 함수
def make_timeseries(df, interpolation=None, iloc_val= None, loc=0, first_file_no=0, month=12, day=31):
    date_col = df.columns[0] # 첫 번째 열을 날짜 열로 설정
    df[date_col] = pd.to_datetime(df[date_col])  # 날짜 형식으로 변환
    df = df[df[date_col].notna()] # 날짜가 없는 행 제거
    df = df.dropna(thresh=3) # 결측치가 3개 이상인 행 제거

    year = pd.DatetimeIndex(df[date_col]).year.astype(np.int64) # 연도 추출
    if loc==0 and first_file_no==0:
        month_tmp = pd.DatetimeIndex(df[date_col]).month.astype(np.int64) # 월 추출
        day_tmp = pd.DatetimeIndex(df[date_col]).day.astype(np.int64) # 일 추출
        month = month_tmp[-1]
        day = day_tmp[-1]
    else:
        month = month
        day = day

    start = str(year[0]) + "-01-01 00:00"    # 시작 날짜 설정
    end = str(year[-1]) + "-" +str(month) + "-" + str(day) + " 23:00" # 종료 날짜 설정
    print('time range in files : ', start, ' ~ ', end)

    time_series = pd.date_range(start=start, end=end, freq='H') # 시간 범위 생성

    time_series = pd.DataFrame(time_series) # 데이터프레임으로 변환
    time_series.columns = [date_col] # 열 이름 설정

    time_series = pd.concat([time_series, df], axis=0) # 원본 데이터프레임과 병합
    time_series = time_series.drop_duplicates([date_col], keep="last") # 중복된 날짜 제거
    time_series = time_series.sort_values([date_col], axis=0) # 날짜 순서로 정렬

    if interpolation[0]:
        for i in range(1, df.shape[1], 1):
            idx = df.iloc[:, i].dropna()

            if idx.shape[0] != 0:
                time_series.iloc[0, i] = idx.iloc[0]
                time_series.iloc[-1, i] = idx.iloc[-1]

    return time_series, month, day # 최종 시계열 데이터프레임 반환

# 데이터프레임을 생성하는 함수
def make_dataframe(directory_path, file_names, iloc_val, interpolation=None, first_file_no=0, month=12, day=31):
    day_for_sincos = 24 * 60 * 60 # 하루를 초 단위로 계산
    year_for_sincos = (365.2425) * day_for_sincos # 1년을 초 단위로 계산
    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        #print(file_names[loc])
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y]) # 파일 경로 설정
            df_tmp = pd.read_excel(path) # 엑셀 파일 읽기
            df_tmp = make_timeseries_by_day(df = df_tmp, interpolation=interpolation) # 일별 시계열 데이터 생성
            df_loc.append(df_tmp)

        df_loc = pd.concat(df_loc) # 파일 병합
        df_loc, month, day = make_timeseries(df_loc, interpolation=interpolation, iloc_val=iloc_val, month=month, day=day, loc=loc, first_file_no=first_file_no)
        df_full.append(df_loc)
        inter = eval(f"df_full[loc].iloc[{iloc_val}]")
        df.append(inter)
        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day_for_sincos))) # 하루 주기 추가
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day_for_sincos))) # 하루 주기 추가
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year_for_sincos))) # 1년 주기 추가
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year_for_sincos))) # 1년 주기 추가

        df[loc] = df[loc].reset_index(drop=True)

        if interpolation[0]:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both') # 보간


    return df, date_time.reset_index(drop=True), month, day # 최종 데이터프레임과 날짜 반환

# 12일간의 온도 데이터를 생성하는 함수
def make_dataframe_temp_12days(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60 # 하루를 초 단위로 계산
    year = (365.2425) * day # 1년을 초 단위로 계산

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y]) # 파일 경로 설정
            df_tmp = pd.read_excel(path) # 엑셀 파일 읽기
            df_loc.append(df_tmp)
        df_loc = pd.concat(df_loc) # 파일 병합
        df_full.append(df_loc)
        inter = eval(f"df_full[loc].iloc[{iloc_val}]")

        df.append(inter)

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        df[loc].insert(df[loc].shape[1], 'Day sin', np.sin(timestamp_s * (2 * np.pi / day))) # 하루 주기 추가
        df[loc].insert(df[loc].shape[1], 'Day cos', np.cos(timestamp_s * (2 * np.pi / day))) # 하루 주기 추가
        df[loc].insert(df[loc].shape[1], 'Year sin', np.sin(timestamp_s * (2 * np.pi / year))) # 1년 주기 추가
        df[loc].insert(df[loc].shape[1], 'Year cos', np.cos(timestamp_s * (2 * np.pi / year))) # 1년 주기 추가

        df[loc] = df[loc].reset_index(drop=True)

        if interpolate:
            df[loc] = df[loc].interpolate(method='pchip', order=3, limit_direction='both') # 보간

    return df, date_time.reset_index(drop=True)

# 테스트용 데이터프레임 생성 함수
def make_dataframe_in_test(directory_path, file_names, iloc_val, interpolate=None):
    day = 24 * 60 * 60 # 하루를 초 단위로 계산
    year = (365.2425) * day # 1년을 초 단위로 계산

    df_full = []
    df = []

    for loc in range(len(file_names)):

        df_loc = []
        for y in range(len(file_names[loc])):
            path = os.path.join(directory_path, file_names[loc][y]) # 파일 경로 설정
            df_tmp = pd.read_excel(path) # 엑셀 파일 읽기
            df_loc.append(df_tmp)
        df_loc = pd.concat(df_loc) # 파일 병합
        df_loc = make_timeseries(df_loc, interpolate=interpolate, iloc_val = iloc_val, directory_path=directory_path) # 시계열 데이터 생성

        df_full.append(df_loc)
        df.append(df_full[loc])

        date_time = pd.to_datetime(df_full[loc].iloc[:, 0], format='%Y.%m.%d %H:%M', utc=True)
        timestamp_s = date_time.map(datetime.datetime.timestamp)

        df[loc] = df[loc].reset_index(drop=True)

        if interpolate:
            df[loc] = df[loc].interpolate(method='polynomial', order=3, limit_direction='both') # 보간

    return df, date_time.reset_index(drop=True) # 최종 데이터프레임과 날짜 반환
