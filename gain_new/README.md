# 수질예측 (gain_new)

### 개발환경
```
Python 3.7.9
```

### 프로젝트 실행
```
git clone https://github.com/kotechnia/water-quality
cd water-quality/gain_new
python -m venv venv

# for linux
. venv/bin/activate

# for windows
venv\Scripts\activate

pip install -r requirements.txt

python main_run.py
```

### 디렉토리 설명

core
```
gain 처리에 필요한 핵심 파일을 포함하고 있음
```

data
```
gain 처리 시 입력으로 들어가는 input 데이터를 포함하고 있음
```

output
```
gain 처리 후 산출되는 결과물 데이터인 output 데이터를 포함하고 있음
```

save
```
모델 또는 결측치패턴 파일을 포함하고 있음
```

### 파일 설명

main_run.py
```
실행 스크립트 파일
```

requirements.txt
```
실행 시 필요한 모듈을 포함하고 있음
pip 설치 시 "-r" 옵션을 사용하여 자동으로 모듈을 설치할 수 있음
```

dev.py
```
개발용 (삭제 예정)
```

dev_tmp.py
```
개발용 (삭제 예정)
```