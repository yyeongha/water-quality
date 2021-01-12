import os
import zipfile
import json
import sys
import pandas as pd
from django.conf import settings
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def index(request):
    context = {}
    # try:
    #     path = settings.MEDIA_ROOT  # insert the path to your directory
    #     file_list = os.listdir(path)
    #     file_list_zip = [file for file in file_list if file.endswith(".zip")]
    #     context['file_list_zip'] = file_list_zip[0][:4]
    # except:
    #     pass

    context['test_html'] = load_df(request)
    context['sample_map'] = sample_map(request)
    return render(request, 'backend/index.html', context)


def sample_map(request):
    testttt = {
        "자동측정망": [
            {'R01': [
                [{'location': '의암호'}, {'x': '127.678647'}, {'y': '37.877653'}],
                [{'location': '한탄강'}, {'x': '127.077585'}, {'y': '38.032802'}],
                [{'location': '경안천'}, {'x': '127.310526'}, {'y': '37.442547'}],
                [{'location': '능서'}, {'x': '127.613025'}, {'y': '37.320053'}],
                [{'location': '평창강'}, {'x': '128.330895'}, {'y': '37.215474'}],
                [{'location': '청미천'}, {'x': '127.71901'}, {'y': '37.203522'}],
                [{'location': '포천'}, {'x': '127.236411'}, {'y': '38.006968'}],
                [{'location': '충주'}, {'x': '127.952738'}, {'y': '37.02413'}],
                [{'location': '서상'}, {'x': '127.685785'}, {'y': '37.955208'}],
                [{'location': '신천'}, {'x': '127.07899'}, {'y': '38.004491'}],
                [{'location': '구리'}, {'x': '127.116663'}, {'y': '37.563112'}],
                [{'location': '원주'}, {'x': '127.951525'}, {'y': '37.423617'}],
                [{'location': '달천'}, {'x': '127.928599'}, {'y': '36.928513'}],
                [{'location': '단양'}, {'x': '128.484395'}, {'y': '37.085671'}],
                [{'location': '인제'}, {'x': '128.200322'}, {'y': '38.159879'}],
                [{'location': '강천'}, {'x': '127.746077'}, {'y': '37.242509'}],
                [{'location': '여주'}, {'x': '127.680034'}, {'y': '37.255051'}],
                [{'location': '가평'}, {'x': '127.38222'}, {'y': '37.673854'}],
                [{'location': '복하천'}, {'x': '127.541519'}, {'y': '37.363618'}],
                [{'location': '화천'}, {'x': '127.709603'}, {'y': '38.09903'}],
                [{'location': '흥천'}, {'x': '127.538081'}, {'y': '37.379891'}],
                [{'location': '미산'}, {'x': '126.982892'}, {'y': '38.02491'}]
            ]}, {'R02': [
                [{'location': '상동'}, {'x': '128.904788'}, {'y': '35.363586'}],
                [{'location': '안동'}, {'x': '128.461752'}, {'y': '36.543947'}],
                [{'location': '봉화'}, {'x': '129.054875'}, {'y': '36.930161'}],
                [{'location': '칠서'}, {'x': '128.438374'}, {'y': '35.387538'}],
                [{'location': '풍양'}, {'x': '128.298197'}, {'y': '36.552512'}],
                [{'location': '고령'}, {'x': '128.386795'}, {'y': '35.751973'}],
                [{'location': '회상'}, {'x': '128.265944'}, {'y': '36.441971'}],
                [{'location': '남강'}, {'x': '128.41557'}, {'y': '35.362076'}],
                [{'location': '구미'}, {'x': '128.397766'}, {'y': '36.08386'}],
                [{'location': '왜관'}, {'x': '128.395499'}, {'y': '35.969196'}],
                [{'location': '해평'}, {'x': '128.358543'}, {'y': '36.19957'}],
                [{'location': '창암'}, {'x': '128.832754'}, {'y': '35.379261'}],
                [{'location': '칠곡'}, {'x': '128.384756'}, {'y': '36.069536'}],
                [{'location': '성서'}, {'x': '128.492083'}, {'y': '35.819474'}],
                [{'location': '신암'}, {'x': '128.307024'}, {'y': '36.405594'}],
                [{'location': '다산'}, {'x': '128.418596'}, {'y': '35.85393'}],
                [{'location': '진주'}, {'x': '128.158665'}, {'y': '35.240937'}],
                [{'location': '적포'}, {'x': '128.360689'}, {'y': '35.610874'}],
                [{'location': '남천'}, {'x': '128.707899'}, {'y': '35.855507'}],
                [{'location': '청암'}, {'x': '128.517692'}, {'y': '35.39299'}],
                [{'location': '강창'}, {'x': '128.50058'}, {'y': '35.882403'}],
                [{'location': '도개'}, {'x': '128.346702'}, {'y': '36.274288'}],
                [{'location': '안동댐하류'}, {'x': '128.764387'}, {'y': '36.580302'}]
            ]},
            {'R03': [
                [{'location': '봉황천'}, {'x': '127.545148'}, {'y': '36.108051'}],
                [{'location': '현도'}, {'x': '127.461732'}, {'y': '36.459901'}],
                [{'location': '미호천'}, {'x': '127.320098'}, {'y': '36.525191'}],
                [{'location': '남면'}, {'x': '127.271225'}, {'y': '36.479886'}],
                [{'location': '옥천천'}, {'x': '127.565773'}, {'y': '36.336869'}],
                [{'location': '장계'}, {'x': '127.637659'}, {'y': '36.377695'}],
                [{'location': '공주'}, {'x': '127.143679'}, {'y': '36.460321'}],
                [{'location': '갑천'}, {'x': '127.388151'}, {'y': '36.441231'}],
                [{'location': '유구천'}, {'x': '127.051685'}, {'y': '36.468364'}],
                [{'location': '이원'}, {'x': '127.668092'}, {'y': '36.239968'}],
                [{'location': '용담호'}, {'x': '127.486157'}, {'y': '35.93601'}],
                [{'location': '부여'}, {'x': '126.951572'}, {'y': '36.325994'}],
                [{'location': '대청호'}, {'x': '127.552556'}, {'y': '36.432077'}]
            ]},
            {'R04': [
                [{'location': '나주'}, {'x': '126.633109'}, {'y': '34.978524'}],
                [{'location': '구례'}, {'x': '127.546596'}, {'y': '35.187937'}],
                [{'location': '탐진호'}, {'x': '126.873564'}, {'y': '34.781822'}],
                [{'location': '서창교'}, {'x': '126.824465'}, {'y': '35.117622'}],
                [{'location': '동복호'}, {'x': '127.098789'}, {'y': '35.085692'}],
                [{'location': '옥정호'}, {'x': '127.106866'}, {'y': '35.608962'}],
                [{'location': '주암호'}, {'x': '127.235383'}, {'y': '35.011809'}],
                [{'location': 'NULL'}, {'x': '127.235333'}, {'y': '35.011888'}],
                [{'location': '우치'}, {'x': '126.888631'}, {'y': '35.242171'}]
            ]}]
    }
    context = {}
    # context['testttt'] = testttt
    # print(testttt)
    return render(request, 'backend/sample_map.html', context)


def sample_datepicker(request):
    context = {}
    return render(request, 'backend/sample_datepicker.html', context)


def file_download(request):
    if request.is_ajax():
        file_path = os.path.join(settings.MEDIA_ROOT, 'sample_excel.txt')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="text/csv")
                print('os.path.basename(file_path)', os.path.basename(file_path))
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                return response
        raise Http404


def file_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        print('myfile', myfile)
        # upload file save default
        # fs = FileSystemStorage()
        fs = FileSystemStorage(location=settings.UPLOAD_ROOT, base_url=settings.UPLOAD_URL)
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
            # fs = FileSystemStorage()
            fs = FileSystemStorage(location=settings.UPLOAD_ROOT, base_url=settings.UPLOAD_URL)
            fs.save(f.name, f)
        return JsonResponse({"rusult": 'success'})
    else:
        return JsonResponse({"rusult": 'fail'})


def load_df(request):
    if request.is_ajax():
        # 데이터프레임 샘플
        df_sample = pd.DataFrame(
            {'name': ['Kim', 'LEE', 'Park', 'Choi'],
             'math': [88, 74, 72, 85],
             'english': [80, 90, 78, 80]
             })
        # HTML로 변환하기
        test_html = df_sample.to_html(index=False, justify='center', table_id="excel_table")
        return JsonResponse({'data': test_html})


def predict(request):
    path = os.path.join(settings.PREDICT_ROOT, 'main.py')
    print('predict_run')
    os.system(path)
    # print('tes',tes)
    # GAIN()
    return JsonResponse({"1": "2"})

def call_model(request):
    if request.is_ajax():
        key=request.POST.get('param')
        print('key',key)
        # input parameter
        parameters_dir = './model_dir/model_A'
        parameters_file = 'a_json.json'
        parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)

        with open(parameters_path, encoding='utf8') as json_file:
            parameters = json.load(json_file)
            print(parameters)
        return JsonResponse({"1": "2"})
##============backup==================
# {'river_id':'R01'},{'key':'의암호'},{'key':'127.678647'},{'key':'37.877653'},
#     {'river_id':'R01'},{'key':'한탄강'},{'key':'127.077585'},{'key':'38.032802'},
#     {'river_id':'R01'},{'key':'경안천'},{'key':'127.310526'},{'key':'37.442547'},
#     {'river_id':'R01'},{'key':'능서'},{'key':'127.613025'},{'key':'37.320053'},
#     {'river_id':'R01'},{'key':'평창강'},{'key':'128.330895'},{'key':'37.215474'},
#     {'river_id':'R01'},{'key':'청미천'},{'key':'127.71901'},{'key':'37.203522'},
#     {'river_id':'R01'},{'key':'포천'},{'key':'127.236411'},{'key':'38.006968'},
#     {'river_id':'R01'},{'key':'충주'},{'key':'127.952738'},{'key':'37.02413'},
#     {'river_id':'R01'},{'key':'서상'},{'key':'127.685785'},{'key':'37.955208'},
#     {'river_id':'R01'},{'key':'신천'},{'key':'127.07899'},{'key':'38.004491'},
#     {'river_id':'R01'},{'key':'구리'},{'key':'127.116663'},{'key':'37.563112'},
#     {'river_id':'R01'},{'key':'원주'},{'key':'127.951525'},{'key':'37.423617'},
#     {'river_id':'R01'},{'key':'달천'},{'key':'127.928599'},{'key':'36.928513'},
#     {'river_id':'R01'},{'key':'단양'},{'key':'128.484395'},{'key':'37.085671'},
#     {'river_id':'R01'},{'key':'인제'},{'key':'128.200322'},{'key':'38.159879'},
#     {'river_id':'R01'},{'key':'강천'},{'key':'127.746077'},{'key':'37.242509'},
#     {'river_id':'R01'},{'key':'여주'},{'key':'127.680034'},{'key':'37.255051'},
#     {'river_id':'R01'},{'key':'가평'},{'key':'127.38222'},{'key':'37.673854'},
#     {'river_id':'R01'},{'key':'복하천'},{'key':'127.541519'},{'key':'37.363618'},
#     {'river_id':'R01'},{'key':'화천'},{'key':'127.709603'},{'key':'38.09903'},
#     {'river_id':'R01'},{'key':'흥천'},{'key':'127.538081'},{'key':'37.379891'},
#     {'river_id':'R01'},{'key':'미산'},{'key':'126.982892'},{'key':'38.02491'},
#     {'river_id':'R02'},{'key':'상동'},{'key':'128.904788'},{'key':'35.363586'},
#     {'river_id':'R02'},{'key':'안동'},{'key':'128.461752'},{'key':'36.543947'},
#     {'river_id':'R02'},{'key':'봉화'},{'key':'129.054875'},{'key':'36.930161'},
#     {'river_id':'R02'},{'key':'칠서'},{'key':'128.438374'},{'key':'35.387538'},
#     {'river_id':'R02'},{'key':'풍양'},{'key':'128.298197'},{'key':'36.552512'},
#     {'river_id':'R02'},{'key':'고령'},{'key':'128.386795'},{'key':'35.751973'},
#     {'river_id':'R02'},{'key':'회상'},{'key':'128.265944'},{'key':'36.441971'},
#     {'river_id':'R02'},{'key':'남강'},{'key':'128.41557'},{'key':'35.362076'},
#     {'river_id':'R02'},{'key':'구미'},{'key':'128.397766'},{'key':'36.08386'},
#     {'river_id':'R02'},{'key':'왜관'},{'key':'128.395499'},{'key':'35.969196'},
#     {'river_id':'R02'},{'key':'해평'},{'key':'128.358543'},{'key':'36.19957'},
#     {'river_id':'R02'},{'key':'창암'},{'key':'128.832754'},{'key':'35.379261'},
#     {'river_id':'R02'},{'key':'칠곡'},{'key':'128.384756'},{'key':'36.069536'},
#     {'river_id':'R02'},{'key':'성서'},{'key':'128.492083'},{'key':'35.819474'},
#     {'river_id':'R02'},{'key':'신암'},{'key':'128.307024'},{'key':'36.405594'},
#     {'river_id':'R02'},{'key':'다산'},{'key':'128.418596'},{'key':'35.85393'},
#     {'river_id':'R02'},{'key':'진주'},{'key':'128.158665'},{'key':'35.240937'},
#     {'river_id':'R02'},{'key':'적포'},{'key':'128.360689'},{'key':'35.610874'},
#     {'river_id':'R02'},{'key':'남천'},{'key':'128.707899'},{'key':'35.855507'},
#     {'river_id':'R02'},{'key':'청암'},{'key':'128.517692'},{'key':'35.39299'},
#     {'river_id':'R02'},{'key':'강창'},{'key':'128.50058'},{'key':'35.882403'},
#     {'river_id':'R02'},{'key':'도개'},{'key':'128.346702'},{'key':'36.274288'},
#     {'river_id':'R02'},{'key':'안동댐하류'},{'key':'128.764387'},{'key':'36.580302'},
#     {'river_id':'R03'},{'key':'봉황천'},{'key':'127.545148'},{'key':'36.108051'},
#     {'river_id':'R03'},{'key':'현도'},{'key':'127.461732'},{'key':'36.459901'},
#     {'river_id':'R03'},{'key':'미호천'},{'key':'127.320098'},{'key':'36.525191'},
#     {'river_id':'R03'},{'key':'남면'},{'key':'127.271225'},{'key':'36.479886'},
#     {'river_id':'R03'},{'key':'옥천천'},{'key':'127.565773'},{'key':'36.336869'},
#     {'river_id':'R03'},{'key':'장계'},{'key':'127.637659'},{'key':'36.377695'},
#     {'river_id':'R03'},{'key':'공주'},{'key':'127.143679'},{'key':'36.460321'},
#     {'river_id':'R03'},{'key':'갑천'},{'key':'127.388151'},{'key':'36.441231'},
#     {'river_id':'R03'},{'key':'유구천'},{'key':'127.051685'},{'key':'36.468364'},
#     {'river_id':'R03'},{'key':'이원'},{'key':'127.668092'},{'key':'36.239968'},
#     {'river_id':'R03'},{'key':'용담호'},{'key':'127.486157'},{'key':'35.93601'},
#     {'river_id':'R03'},{'key':'부여'},{'key':'126.951572'},{'key':'36.325994'},
#     {'river_id':'R03'},{'key':'대청호'},{'key':'127.552556'},{'key':'36.432077'},
#     {'river_id':'R04'},{'key':'나주'},{'key':'126.633109'},{'key':'34.978524'},
#     {'river_id':'R04'},{'key':'구례'},{'key':'127.546596'},{'key':'35.187937'},
#     {'river_id':'R04'},{'key':'탐진호'},{'key':'126.873564'},{'key':'34.781822'},
#     {'river_id':'R04'},{'key':'서창교'},{'key':'126.824465'},{'key':'35.117622'},
#     {'river_id':'R04'},{'key':'동복호'},{'key':'127.098789'},{'key':'35.085692'},
#     {'river_id':'R04'},{'key':'옥정호'},{'key':'127.106866'},{'key':'35.608962'},
#     {'river_id':'R04'},{'key':'주암호'},{'key':'127.235383'},{'key':'35.011809'},
#     {'river_id':'R04'},{'key':'NULL'},{'key':'NULL'},{'key':'NULL'},
#     {'river_id':'R04'},{'key':'우치'},{'key':'126.888631'},{'key':'35.242171'},
#     {'river_id':'R04'},{'key':'용봉'},{'key':'126.777789'},{'key':'35.073675'}
