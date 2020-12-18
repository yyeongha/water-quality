from django.urls import path
from django.conf.urls import url
from backend import views as IndexViews


urlpatterns = [
    path('', IndexViews.index, name='index'),
]