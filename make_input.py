import argparse
import json

def main (args):

    watershed = args.watershed
    train_gain = args.train_gain
    train_rnn = args.train_rnn
    target_col = args.target_col
    predict_day = args.predict_day


    json_parameters={}
    with open('input/input.json', encoding='utf8') as json_file:
        parameters = json.load(json_file)

        json_parameters['file'] = parameters['file']
        json_parameters['gain'] = parameters['gain']
        json_parameters['rnn'] = parameters['rnn']

        json_parameters['file']['watershed'] = watershed
        json_parameters['gain']['train'] = train_gain
        json_parameters['rnn']['train'] = train_rnn
        json_parameters['rnn']['target_column'] = target_col
        json_parameters['rnn']['predict_day'] = predict_day

        print(json_parameters)

    with open('input/input.json', 'w', encoding='utf-8') as make_file:
        json.dump(json_parameters, make_file, indent=4)



if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--watershed',
        choices=['han','nak', 'geum', 'yeong'],
        default='han',
        type=str)
    parser.add_argument(
        '--train_gain',
        help='GAIN training option(True/False)',
        default=False,
        type=bool)
    parser.add_argument(
        '--train_rnn',
        help='RNN training option(True/False)',
        default=False,
        type=bool)
    parser.add_argument(
        '--target_col',
        choices=['do','toc', 'tn', 'tp', 'chl-a'],
        default='do',
        type=str)
    parser.add_argument(
        '--predict_day',
        help='number of predict day',
        default=5,
        type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
