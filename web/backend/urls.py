from django.urls import path
from django.conf.urls import url
from backend import views as IndexViews
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', IndexViews.index, name='index'),
    path('file_download', IndexViews.file_download, name='file_download'),
    path('file_upload', IndexViews.file_upload, name='file_upload'),
    path('multi_file_upload', IndexViews.multi_file_upload, name='multi_file_upload'),
    path('load_df', IndexViews.load_df, name='load_df'),
    path('predict', IndexViews.predict, name='predict'),
    path('call_model', IndexViews.call_model, name='call_model'),
    path('deactivate', IndexViews.deactivate, name='deactivate'),
    path('drawing', IndexViews.drawing, name='drawing'),
    path('drawing2', IndexViews.drawing2, name='drawing2'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # urlpatterns += static(settings.DOWNLOAD_URL, document_root=settings.DOWNLOAD_ROOT)
