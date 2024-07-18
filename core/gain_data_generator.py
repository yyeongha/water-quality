#####################################################################################
# GAIN(Generative Adversarial Imputation Networks)을 위한 데이터 생성기를 구현
# 결측 데이터를 보완하고 시계열 데이터를 처리하여 모델 학습에 사용할 배치(batch) 데이터를 만듦
#####################################################################################

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from core.util import *
from core.miss_data import MissData # MissData 클래스를 가져옴

# GAIN을 위한 데이터 생성기 클래스
class GainDataGenerator(tf.keras.utils.Sequence):
    'Generates data for GAIN'

    def __init__(self,
                 data_list, # 데이터 리스트
                 batch_size=32, # 배치 크기
                 input_width=24 * 3, # 입력 폭 (72)
                 label_width=24 * 3, # 라벨 폭 (72)
                 shift=0, # 이동량
                 fill_no=4, # 보간 최대 간격
                 miss_rate=0.2, # 결측률
                 hint_rate=0.9, # 힌트율
                 normalize=True, # 정규화 여부
                 miss_pattern=None, # 결측 패턴
                 model_save_path='save', # 모델 저장 경로
                 alpha=100.): # 알파 값
        'Initialization'
        window_size = input_width # 윈도우 크기 설정
        # 보간 처리
        filled_data = []
        for data in data_list:
            data = interpolate(data, max_gap=fill_no) # 데이터 보간
            filled_data.append(data) 

        data_list = filled_data

        # 전체 데이터 결합
        self.data = np.concatenate(data_list)
        # TO-DO
        # 시퀀스 데이터에 대한 사전 계산
        last_cum = 0
        cums = []
        for data in data_list: 
            isnan = np.isnan(data) # 결측값 여부 확인
            isany = np.any(isnan, axis=1) # 행 단위로 결측값이 있는지 확인
            shifted = np.roll(isany, 1) # 결측값 여부를 한 칸씩 이동
            shifted[0] = True  # 첫 번째 값을 결측으로 설정

            start_seq = ((isany == False) & (shifted == True)).astype(int) # 시퀀스 시작점 찾기
            cum = start_seq.cumsum() # 누적합 계산
            cum += last_cum
            last_cum = np.max(cum)
            cum[isany] = 0
            cums.append(cum)

        # 마스크 행렬 정의
        if miss_pattern is None:
            self.data_m = binary_sampler(1 - miss_rate, self.data.shape) # 결측 마스크 생성
        else:
            self.miss = MissData(load_dir=model_save_path) # MissData 객체 생성
            self.miss_rate = miss_rate
            miss_data = self.miss.make_missdata(self.data, self.miss_rate) # 결측 데이터 생성
            self.data_m = 1. - np.isnan(miss_data).astype(float) # 결측 마스크 생성

            self.data_m_rand = binary_sampler(1 - (miss_rate / 10.), self.data.shape) # 무작위 마스크 생성
            self.data_m[self.data_m_rand == 0.] = 0.

        self.miss_pattern = miss_pattern

        # 시퀀스 데이터
        self.ids = np.concatenate(cums)
        data_idx = np.empty((0), dtype=int)
        for i in range(1, last_cum + 1):
            seq_len = (self.ids == i).sum()
            start_id = np.argmax(self.ids == i)
            # 시퀀스에서 가능한 데이터 수
            time_len = seq_len - window_size + 1
            start_ids = np.arange(start_id, start_id + time_len)
            data_idx = np.append(data_idx, start_ids)

        # 시퀀스 데이터의 시작 인덱스 설정
        self.data_idx = data_idx
        self.input_width = input_width
        self.no = len(data_idx)
        self.batch_size = batch_size
        self.batch_idx = sample_batch_index(self.no, self.no)
        self.batch_id = 0
        self.shape = (batch_size, self.input_width) + self.data.shape[1:]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1 # 에포크당 배치 수
    def __getitem__(self, index):
        'Generate one batch of data'

        x = np.empty((0, self.input_width, self.data.shape[1])) # 입력 데이터를 저장할 배열 초기화
        y = np.empty((0, self.input_width, self.data.shape[1])) # 라벨 데이터를 저장할 배열 초기화

        for cnt in range(0, self.batch_size):
            i = self.batch_idx[self.batch_id]
            self.batch_id += 1
            self.batch_id %= self.no
            if self.miss_pattern and (self.batch_id == 0):
                self.batch_idx = sample_batch_index(self.no, self.no)
                miss_data = self.miss.make_missdata(self.data, self.miss_rate)
                self.data_m = 1. - np.isnan(miss_data).astype(float)
                self.data_m_rand = binary_sampler(1 - self.miss_rate / 10., self.data.shape)
                self.data_m[self.data_m_rand == 0.] = 0.
            idx1 = self.data_idx[i]
            idx2 = self.data_idx[i] + self.input_width

            Y_mb = self.data[idx1:idx2].copy()
            X_mb = Y_mb.copy()
            M_mb = self.data_m[idx1:idx2]
            Z_mb = uniform_sampler(0, 0.01, shape=X_mb.shape)
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            X_mb[M_mb == 0] = np.nan
            Y_mb[M_mb == 1] = np.nan
            x = np.append(x, [X_mb], axis=0)
            y = np.append(y, [Y_mb], axis=0)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        return # 각 에포크가 끝날 때 인덱스를 업데이트
