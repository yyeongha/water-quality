import os
import zipfile
import pandas as pd
from django.conf import settings
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def index(request):
    context = {}
    try:
        path = settings.MEDIA_ROOT  # insert the path to your directory
        file_list = os.listdir(path)
        file_list_zip = [file for file in file_list if file.endswith(".zip")]
        context['file_list_zip'] = file_list_zip[0][:4]
    except:
        pass

    context['test_html'] = load_df(request)
    return render(request, 'backend/index.html', context)


def sample_map(request):
    context = {}
    return render(request, 'backend/sample_map.html', context)


def sample_datepicker(request):
    context = {}
    return render(request, 'backend/sample_datepicker.html', context)


def file_download(request):
    if request.is_ajax():
        file_path = os.path.join(settings.MEDIA_ROOT, '스크린샷 2020-12-29 오후 3.26.50.png')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                return response
        raise Http404


def file_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        print('myfile', myfile)
        # upload file save default
        fs = FileSystemStorage()
        # fs = FileSystemStorage(location=settings.DOWNLOAD_ROOT, base_url=settings.DOWNLOAD_URL)
        fs.save(myfile.name, myfile)

        # upload file unzip save
        if myfile.name[-4:] == '.zip':
            with zipfile.ZipFile(myfile, 'r') as existing_zip:
                existing_zip.extractall(settings.MEDIA_ROOT)
            return JsonResponse({"rusult": 'success'})
        else:
            return JsonResponse({"rusult": 'fail'})

        # # upload file unzip save
        # if myfile.name[-4:] == '.zip':
        #     with zipfile.ZipFile(myfile, 'r') as existing_zip:
        #         existing_zip.extractall(settings.MEDIA_ROOT)
        #     return JsonResponse({"rusult": 'success'})
        # else:
        #     return JsonResponse({"rusult": 'fail'})


def multi_file_upload(request):
    if request.method == 'POST' and request.FILES.getlist('myfiles'):
        flist = request.FILES.getlist('myfiles')
        for f in flist:
            # upload file save default
            fs = FileSystemStorage()
            # fs = FileSystemStorage(location=settings.DOWNLOAD_ROOT, base_url=settings.DOWNLOAD_URL)
            fs.save(f.name, f)
        return JsonResponse({"rusult": 'success'})
    else:
        return JsonResponse({"rusult": 'fail'})


def load_df(requset):
    # 데이터프레임 샘플
    df_sample = pd.DataFrame(
        {'name': ['Kim', 'LEE', 'Park', 'Choi'],
         'math': [88, 74, 72, 85],
         'english': [80, 90, 78, 80]
         })

    # HTML로 변환하기
    test_html = df_sample.to_html(index=False, justify='center')
    print('test_html;',test_html)
    return test_html
