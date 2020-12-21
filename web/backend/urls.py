from django.urls import path
from django.conf.urls import url
from backend import views as IndexViews


urlpatterns = [
    path('', IndexViews.index, name='index'),
    path('sample_map', IndexViews.sample_map, name='sample_map'),
    path('sample_datepicker', IndexViews.sample_datepicker, name='sample_datepicker'),
]