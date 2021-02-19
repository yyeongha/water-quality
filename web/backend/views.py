import datetime
import json
import os
import zipfile

import pandas as pd
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse, Http404
from django.shortcuts import render

from gain_new.core.predict_run import prediction_for_webpage


def index(request):
    context = {}

    context['test_html'] = load_df(request)
    return render(request, 'backend/index.html', context)


def load_df(request):
    if request.is_ajax():
        target = request.POST.get('target')
        model = request.POST.get('model')
        for i in ['A', 'B', 'C', 'D']:
            if model == i:
                parameters_dir = './model_dir/model_' + i
        parameters_file = 'json_info.json'
        parameters_path = '{dir}/{file}'.format(dir=parameters_dir, file=parameters_file)

        with open(parameters_path, encoding='utf8') as json_file:
            parameters = json.load(json_file)
        target_list = []
        # addr_list = []
        location_list = []
        x_list = []
        y_list = []
        cat_id_list = []
        cat_did_list = []
        rch_id_list = []
        rch_did_list = []
        node_id_list = []
        node_did_list = []

        if target == 'all':
            for i in parameters['web_info']['map_info']:
                if i['target'] == 'target_a':
                    target_name = '자동측정망'
                elif i['target'] == 'target_b':
                    target_name = '수질측정망'
                elif i['target'] == 'target_c':
                    target_name = '총량측정망'
                elif i['target'] == 'target_d':
                    target_name = '녹조 조류'
                target_list.append(target_name)
                location_list.append(i['location'])
                x_list.append(i['x'])
                y_list.append(i['y'])
                cat_id_list.append(i['cat_id'])
                cat_did_list.append(format(i['cat_did'], '.53g'))
                rch_id_list.append(i['rch_id'])
                rch_did_list.append(i['rch_did'])
                node_id_list.append(i['node_id'])
                node_did_list.append(i['node_did'])
        else:
            for i in parameters['web_info']['map_info']:
                #  임시로 a 삽입
                # map_list = [d for d in i if i['target'] == target]

                if i['target'] == target:
                    try:
                        if i['target'] == 'target_a':
                            target_name = '자동측정망'
                        elif i['target'] == 'target_b':
                            target_name = '수질측정망'
                        elif i['target'] == 'target_c':
                            target_name = '총량측정망'
                        elif i['target'] == 'target_d':
                            target_name = '녹조 조류'
                        target_list.append(target_name)
                        location_list.append(i['location'])
                        x_list.append(i['x'])
                        y_list.append(i['y'])
                        cat_id_list.append(i['cat_id'])
                        cat_did_list.append(format(i['cat_did'], '.53g'))
                        rch_id_list.append(i['rch_id'])
                        rch_did_list.append(i['rch_did'])
                        node_id_list.append(i['node_id'])
                        node_did_list.append(i['node_did'])
                    except:
                        pass

        # 데이터프레임 샘플
        df_sample = pd.DataFrame(
            {'종류': target_list,
             'location': location_list,
             '위도': y_list,
             '경도': x_list,
             'cat_id': cat_id_list,
             'cat_did': cat_did_list,
             'rch_id': rch_id_list,
             'rch_did': rch_did_list,
             'node_id': node_id_list,
             'node_did': node_did_list
             })
        # HTML로 변환하기
        test_html = df_sample.to_html(index=False, justify='center', table_id="excel_table")
        return JsonResponse({'data': test_html})


def predict(request):
    key = request.POST.get('key')
    upload_df = request.POST.get('upload_df')
    start_date = request.POST.get('start_date')
    end_date = request.POST.get('end_date')
    predict_start_date = request.POST.get('predict_start_date')
    predict_end_date = request.POST.get('predict_end_date')
    try:
        model = request.POST.get('model')
        for i in ['A', 'B', 'C', 'D']:
            if model == i:
                model_dir = './model_dir/model_' + i
        parameters_file = 'json_info.json'
        excel_file = 'river.xlsx'
        parameters_path = '{dir}/{file}'.format(dir=model_dir, file=parameters_file)
        excel_path = '{dir}/{file}'.format(dir=model_dir, file=excel_file)

        if upload_df == 'Y':
            df = read_xlsx(settings.UPLOAD_ROOT, start_date, predict_end_date, 'N')
            print('upload_df', df)
        else:
            df = read_xlsx(excel_path, start_date, predict_end_date, 'Y')
            print('df', df)
        if df == 0:
            return JsonResponse({"return": "date_fail"})
        # 강우량,기온
        rain_list, temp_list = load_rain(key, start_date, predict_end_date, model_dir)
        '''
        watershed 
            0:한강, 1:낙동강, 2:금강, 3:영산강
        Target index 
            0: 용존산소(DO), 1:총유기탄소(TOC) 2:총질소(TN) 3:총인(TP), 4:클로로필-a(Chl-a)
        '''
        watershed = {"A": 0, "B": 2, "C": 3, "D": 1}
        target = {"do": 0, "toc": 1, "tn": 2, "tp": 3, "chl": 4}
        prediction = prediction_for_webpage()

        nse, pbias, input_data, label, pred = prediction.run(dataframe=df, watershed=watershed[model],
                                                             target=target[key])
        # 수질등급별 색상
        color = color_list(key, pred)

        input_data = list(input_data)
        label = list(label)
        pred = list(pred)


    except Exception as e:
        print(e)
        return JsonResponse({"return": "fail"})

    with open(parameters_path, encoding='utf8') as json_file:
        parameters = json.load(json_file)
    try:
        '''
        toc = 총유기 탄소량
        chl = 클로로필 -a
        do = 용존 산소량
        tn = 총질소
        tp = 총인
        '''
        # data
        data = [None, None, None, None, None, None, None, None, None]
        predict_cahrt = {"origin": input_data,
                         "origin_2": data + [input_data[9]] + label,
                         "predict": data + [input_data[9]] + pred}

        rain_chart = {"rain_list": rain_list,
                      "temp_list": temp_list}
        predict_water = pred
        if key == 'toc':
            value = {"1": "2"}
    except Exception as e:
        print(e)
        return JsonResponse({"return": "fail"})
    return JsonResponse({"predict_cahrt": predict_cahrt, "predict_water": predict_water,
                         "column": parameters['web_info']['columns'][key], "color": color, "rain_chart": rain_chart})


def call_model(request):
    if request.is_ajax():
        key = request.POST.get('param')
        # input parameter
        for i in ['A', 'B', 'C', 'D']:
            if key == i:
                model_dir = './model_dir/model_' + i

        if key == 'A':
            x = 37.868107908588094
            y = 127.68186772257765
        elif key == 'B':
            x = 36.477884511195995
            y = 127.48034948700723
        elif key == 'C':
            x = 34.91217642389682
            y = 126.77230673652103
        elif key == 'D':
            x = 36.437705725281816
            y = 128.21182763524286
        # parameters_dir = './model_dir/model_A'
        parameters_file = 'json_info.json'
        parameters_path = '{dir}/{file}'.format(dir=model_dir, file=parameters_file)

        with open(parameters_path, encoding='utf8') as json_file:
            parameters = json.load(json_file)

        return JsonResponse({'web_info': parameters['web_info'], "x": x, "y": y})


def load_rain(key, start_date, end_date, model_dir):
    rain_file = 'rain.xlsx'
    rain_path = '{dir}/{file}'.format(dir=model_dir, file=rain_file)

    start_date = datetime.datetime.strptime(start_date, '%Y.%m.%d')

    start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')

    end_date = datetime.datetime.strptime(end_date, '%Y.%m.%d')
    end_date += datetime.timedelta(days=1)
    end_date = datetime.datetime.strftime(end_date, '%Y-%m-%d')

    # dataframe convert
    df_loc = pd.DataFrame(pd.read_excel(rain_path)).filter(["aws_dt", "rn60m_value", "ta_value"])
    # set date
    first_column = str(df_loc.columns[0])
    after_start_date = df_loc[first_column] >= start_date
    before_end_date = df_loc[first_column] < end_date
    between_two_dates = after_start_date & before_end_date

    filtered_dates = df_loc.loc[between_two_dates]
    filtered_dates.aws_dt = pd.to_datetime(filtered_dates.aws_dt)
    filtered_dates = filtered_dates.set_index('aws_dt')

    newDf = filtered_dates.resample('D').mean()
    rn60m_value = list(newDf["rn60m_value"])
    ta_value = list(newDf["ta_value"])
    return rn60m_value, ta_value


def read_xlsx(files_Path, start_date=None, end_date=None, predict=None):
    # dateformat convert
    end_date = datetime.datetime.strptime(end_date, '%Y.%m.%d')
    end_date += datetime.timedelta(days=1)
    end_date = datetime.datetime.strftime(end_date, '%Y.%m.%d')
    end_date = str(end_date)

    if predict != 'Y':
        files_Path = files_Path + "/"
        file_name_and_time_lst = []

        # 해당 경로에 있는 파일들의 생성시간을 함께 리스트로 넣어줌
        for f_name in os.listdir(f"{files_Path}"):
            written_time = os.path.getctime(f"{files_Path}{f_name}")
            file_name_and_time_lst.append((f_name, written_time))
        # 생성시간 역순
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        # sort
        recent_file = sorted_file_lst[0]
        recent_file_name = recent_file[0]

        # file_open
        path = os.path.join(files_Path, recent_file_name)
    else:
        path = files_Path
    # dataframe convert
    df_loc = pd.DataFrame(pd.read_excel(path))

    # set date
    first_column = str(df_loc.columns[0])
    after_start_date = df_loc[first_column] >= start_date
    before_end_date = df_loc[first_column] < end_date
    between_two_dates = after_start_date & before_end_date

    filtered_dates = df_loc.loc[between_two_dates]
    print('filtered_dates', filtered_dates)
    return 0
    # return filtered_dates


def file_download(request):
    if request.is_ajax():
        file_path = os.path.join(settings.MEDIA_ROOT, 'han.txt')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="text/csv")
                print('os.path.basename(file_path)', os.path.basename(file_path))
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                return HttpResponse
        raise Http404


def deactivate(request):
    if request.method == 'POST':
        return JsonResponse({"return": "success"})


def file_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        # upload file save default
        # fs = FileSystemStorage()
        fs = FileSystemStorage(location=settings.UPLOAD_ROOT, base_url=settings.UPLOAD_URL)
        fs.save(myfile.name, myfile)
        # df = read_xlsx(settings.UPLOAD_ROOT, request.POST.get('start_date'), request.POST.get('end_date'))

        return JsonResponse({"rusult": 'success'})
        # ------------
        # upload file unzip save
        if myfile.name[-4:] == '.zip':
            with zipfile.ZipFile(myfile, 'r') as existing_zip:
                existing_zip.extractall(settings.MEDIA_ROOT)
            return JsonResponse({"rusult": 'success'})
        else:
            return JsonResponse({"rusult": 'fail'})


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


def color_list(key, data_list):
    color = []
    if key == 'toc':
        for i in data_list:
            if i <= 2:
                color.append('rgba(66,238,46,0.95)')
            elif 2 < i <= 3:
                color.append('rgba(152,235,78)')
            elif 3 < i <= 4:
                color.append('rgba(178,238,84)')
            elif 4 < i <= 5:
                color.append('rgba(235,229,97)')
            elif 5 < i <= 6:
                color.append('rgba(235,200,79)')
            elif 6 < i <= 8:
                color.append('rgba(235,127,80)')
            elif 8 < i:
                color.append('rgba(235,20,37)')
    elif key == 'chl':
        for i in data_list:
            if i <= 5:
                color.append('rgba(66,238,46,0.95)')
            elif 5 < i <= 9:
                color.append('rgba(152,235,78)')
            elif 9 < i <= 14:
                color.append('rgba(178,238,84)')
            elif 14 < i <= 20:
                color.append('rgba(235,229,97)')
            elif 20 < i <= 42:
                color.append('rgba(235,200,79)')
            elif 42 < i <= 70:
                color.append('rgba(235,127,80)')
            elif 70 < i:
                color.append('rgba(235,20,37)')
    elif key == 'do':
        for i in data_list:
            if i >= 7.5:
                color.append('rgba(66,238,46,0.95)')
            elif 7.5 > i >= 5:
                color.append('rgba(152,235,78)')
            elif 7.5 > i >= 5:
                color.append('rgba(178,238,84)')
            elif 7.5 > i >= 5:
                color.append('rgba(235,229,97)')
            elif 2.0 <= i < 5:
                color.append('rgba(235,200,79)')
            elif 2.0 <= i < 5:
                color.append('rgba(235,127,80)')
            elif 2.0 > i:
                color.append('rgba(235,20,37)')
    elif key == 'tn':
        for i in data_list:
            if i <= 0.2:
                color.append('rgba(66,238,46,0.95)')
            elif 0.2 < i <= 0.3:
                color.append('rgba(152,235,78)')
            elif 0.3 < i <= 0.4:
                color.append('rgba(178,238,84)')
            elif 0.4 < i <= 0.6:
                color.append('rgba(235,229,97)')
            elif 0.6 < i <= 1.0:
                color.append('rgba(235,200,79)')
            elif 1.0 < i <= 1.5:
                color.append('rgba(235,127,80)')
            elif 1.5 < i:
                color.append('rgba(235,20,37)')
    elif key == 'tp':
        for i in data_list:
            if i <= 0.02:
                color.append('rgba(66,238,46,0.95)')
            elif 0.02 < i <= 0.04:
                color.append('rgba(152,235,78)')
            elif 0.04 < i <= 0.1:
                color.append('rgba(178,238,84)')
            elif 0.1 < i <= 0.2:
                color.append('rgba(235,229,97)')
            elif 0.2 < i <= 0.3:
                color.append('rgba(235,200,79)')
            elif 0.3 < i <= 0.5:
                color.append('rgba(235,127,80)')
            elif 0.5 < i:
                color.append('rgba(235,20,37)')
    return color
