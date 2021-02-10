# -*- coding: utf-8 -*-

#import gain as Gain
from gain_new.core.models import *



def model_multi_linear(OUT_STEPS, out_num_features, window = None, epochs = 2000, training_flag = False, checkpoint_path = 'save/model', continue_train = False, steps_per_epoch = 10):
    model = MultiLinearModel(OUT_STEPS, out_num_features)

    if training_flag == True:
        if continue_train:
            model.load_weights(checkpoint_path)
            compile(model)
            model_evaluate_value = model.evaluate(window.val)
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path,
                                      val_nse=model_evaluate_value[2], steps_per_epoch=steps_per_epoch)
        else:
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path)
            
        model.load_weights(checkpoint_path)
    else :
        model.load_weights(checkpoint_path)
        compile(model)
    return model


#print('elman_model train')
def model_elman(OUT_STEPS, out_num_features, window, epochs, training_flag, checkpoint_path, continue_train = False, steps_per_epoch = 10):
    model = ElmanModel(OUT_STEPS, out_num_features)

    if training_flag == True:
        if continue_train :
            model.load_weights(checkpoint_path)
            compile(model)
            model_evaluate_value = model.evaluate(window.val)
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path,
                                      val_nse=model_evaluate_value[2], steps_per_epoch=steps_per_epoch)
        else:
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path)
            
        model.load_weights(checkpoint_path)
    else :
        model.load_weights(checkpoint_path)
        compile(model)
    return model
#multi_val_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.val.repeat(-1), steps=100)
#multi_performance['ELMAN_RNN'] = elman_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)

#print('gru_model train')

def model_gru(OUT_STEPS=24*5, out_num_features=1, window=None, epochs=100, training_flag=False, checkpoint_path="save/", continue_train = False, steps_per_epoch = 10):
    model = GRUModel(OUT_STEPS, out_num_features)

    if training_flag == True:
        model_evaluate_value = 0
        if continue_train :
            model.load_weights(checkpoint_path)
            compile(model)
            model_evaluate_value = model.evaluate(window.val)
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path,
                                      val_nse=model_evaluate_value[2], steps_per_epoch=steps_per_epoch)
        else:
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path, steps_per_epoch=steps_per_epoch)

        model.load_weights(checkpoint_path)
    else:
        model.load_weights(checkpoint_path)
        compile(model)
    return model

#multi_val_performance['GRU'] = gru_model.evaluate(multi_window.val.repeat(-1), steps=100)
#multi_performance['GRU'] = gru_model.evaluate(multi_window.test.repeat(-1), verbose=1, steps=100)


#print('multi_lstm_model train')
def model_multi_lstm(OUT_STEPS, out_num_features, window, epochs, training_flag, checkpoint_path, continue_train = False, steps_per_epoch = 10):
    model = MultiLSTMModel(OUT_STEPS, out_num_features)
    #checkpoint_path = "save/multi_lstm_model.ckpt"
#    model.load_weights(checkpoint_path)
    if training_flag == True:
        if continue_train:
            model.load_weights(checkpoint_path)
            compile(model)
            model_evaluate_value = model.evaluate(window.val)
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path,
                                      val_nse=model_evaluate_value[2], steps_per_epoch=steps_per_epoch)
        else:
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path)
        model.load_weights(checkpoint_path)
    else:
        model.load_weights(checkpoint_path)
        compile(model)
    return model

#multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
#multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

def model_multi_conv(OUT_STEPS, out_num_features, window, epochs, training_flag, checkpoint_path, continue_train = False, steps_per_epoch = 10):
    model = MultiConvModel(OUT_STEPS, out_num_features)
    #checkpoint_path = "save/multi_conv_model.ckpt"
#    model.load_weights(checkpoint_path)
    if training_flag == True:
        if continue_train:
            model.load_weights(checkpoint_path)
            compile(model)
            model_evaluate_value = model.evaluate(window.val)
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path,
                                      val_nse=model_evaluate_value[2], steps_per_epoch=steps_per_epoch)
        else:
            history = compile_and_fit(model,window, epochs=epochs, save_path = checkpoint_path)
        model.load_weights(checkpoint_path)
    else:
        model.load_weights(checkpoint_path)
        compile(model)
    return model

#multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
#multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)

#x = np.arange(len(multi_performance))
#width = 0.3
#metric_name = 'mean_absolute_error'
#metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')
#val_mae = [v[metric_index] for v in multi_val_performance.values()]
#test_mae = [v[metric_index] for v in multi_performance.values()]
#plt.figure()
#plt.bar(x - 0.17, val_mae, width, label='Validation')
#plt.bar(x + 0.17, test_mae, width, label='Test')
#plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
#plt.ylabel(f'MAE (average over all times and outputs)')
#_ = plt.legend()
#plt.show()
