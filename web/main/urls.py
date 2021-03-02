from django.urls import path
from django.conf.urls import url
from main import views as NewViews
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', NewViews.index, name='index'),
    path('intro', NewViews.intro, name='intro'),
    path('produce', NewViews.produce, name='produce'),
    path('predict_set/<str:model>', NewViews.predict_set, name='predict_set'),
    path('predict_run', NewViews.predict_run, name='predict_run'),


    path('file_download', NewViews.file_download, name='file_download'),
    path('file_upload', NewViews.file_upload, name='file_upload'),
    path('multi_file_upload', NewViews.multi_file_upload, name='multi_file_upload'),
    path('load_df', NewViews.load_df, name='load_df'),
    path('predict', NewViews.predict, name='predict'),
    path('call_model', NewViews.call_model, name='call_model'),
    path('deactivate', NewViews.deactivate, name='deactivate'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # urlpatterns += static(settings.DOWNLOAD_URL, document_root=settings.DOWNLOAD_ROOT)
