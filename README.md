
# web 인터페이스


## water-quality


### Configuring a project

```bash
$ git clone https://github.com/kotechnia/water-quality
```

#### set

* in workspce not in water-quality *
```bash
$ pip install virtualenv
$ python3 -m venv venv
or
$ virtualenv venv
or
$ virtualenv venv --python=python3.7.9

<Linux>
$ . venv/bin/activate

<window>
$ call venv/script/activate
$ pip install -r requirements.txt
$ cd water-quality/web
$ python manage.py runserver IP:PORT

ex) $ python manage.py runserver 127.0.0.1:8000
```

``` bash
python -m venv venv
# for linux or macOS
. venv/bin/activate

$ for windows
venv\Scripts\activate

pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

### gain
```
기존 gain model 프로젝트
```

### gain_new
```
새로운 gain model 프로젝트
```

### rnn
```
rnn 프로젝트
```

### web
```
수질예측 웹 인터페이스 프로젝트
```
-------------------------------

## Development Environment

```
python 3.7.9
[python 은 버전을 맞춰 별도 설치필수(window 경우 32bit or 64bit 맞추지않으면 에러발생)]
```
  



## Getting Started with Project




#### issue
```
<read_excel error >
- xlrd 버전이 1.0.0 이하면 error
- $ pip uninstall xlrd
- $ pip install xlrd==1.2.0

< model error >
- gain_new/save 폴더가 없다면 save폴더를 gain_new 하위에 move(or paste)
```