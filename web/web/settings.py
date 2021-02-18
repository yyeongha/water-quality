import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = 'pk3&%d2cq*5d2@vi7s)ev10%+odlnr@k^3sv0&)+*g59c(9%!v'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'backend'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'web.wsgi.application'
local = 'N'
if local == 'Y':
    DATABASES = {
        'default': {
            # 'ENGINE': 'django.db.backends.sqlite3',
            # 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'water_quality',
            'USER': 'root',
            'PASSWORD': '',
            'HOST': '0.0.0.0',
            'PORT': '3306'
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
            #'ENGINE': 'django.db.backends.mysql',
            #'NAME': 'epai',
            'USER': 'epai',
            'PASSWORD': 'epai',
            'HOST': '112.217.167.123',
            'PORT': '61000'
        }
    }
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'

MEDIA_URL = '/media/'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# MEDIA_URL = '../../gain_new/save/'
#
# MEDIA_ROOT = os.path.join(BASE_DIR, '../gain_new/output')

# UPLOAD_URL = '../../gain_new/save/'
# UPLOAD_ROOT = os.path.join(BASE_DIR, '../gain_new/output')
UPLOAD_URL = '/gain_new/'
UPLOAD_ROOT = os.path.join(BASE_DIR, 'gain_new/output')

PREDICT_URL = '/gain_new/'
PREDICT_ROOT = os.path.join(BASE_DIR, 'gain_new')
