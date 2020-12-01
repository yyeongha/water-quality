import json

with open('./parameters.json', encoding='utf8') as json_file:
    parameters = json.load(json_file)
    for key, value in parameters.items():
        print('key = {key}, value = {value}, type = {type}'.format(key=key, value=value, type=type(value)))