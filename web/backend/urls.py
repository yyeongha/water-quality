from django.urls import path
from django.conf.urls import url
from backend import views as IndexViews
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', IndexViews.index, name='index'),
    path('sample_map', IndexViews.sample_map, name='sample_map'),
    path('sample_datepicker', IndexViews.sample_datepicker, name='sample_datepicker'),
    path('file_download', IndexViews.file_download, name='file_download'),
    path('file_upload', IndexViews.file_upload, name='file_upload'),
    path('multi_file_upload', IndexViews.multi_file_upload, name='multi_file_upload'),
    path('load_df', IndexViews.load_df, name='load_df'),
]
# if settings.DEBUG:
#     # urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#     urlpatterns += static(settings.DOWNLOAD_URL, document_root=settings.DOWNLOAD_ROOT)
