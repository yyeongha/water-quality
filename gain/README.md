# 수질예측 (gain)

## this project not use now

### 개발환경
```
Python 3.7.9
```

### 개발노트

```
- 측정시간을 결과물에 적용하는 로직 추가
- sin cos 각 "day", "year" 추가
- 패턴을 읽어 패턴에 대해 랜덤으로 결측치 생성 추가
=> dev-gain-backup2 브랜치 프리징

- 디버깅을 위한 소스정리 (2020-12-10)
=> dev-gain-backup3 브랜치 프리징

```

### 프로젝트 실행

```
git clone https://github.com/kotechnia/water-quality
cd water-quality/gain
python -m venv venv

# for linux
. venv/bin/activate

# for windows
venv\Scripts\activate

pip install -r requirements.txt

python debug.py
```