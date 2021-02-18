# -*- coding: utf-8 -*-

from core.models import *

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

def model_gru(OUT_STEPS=24*5, out_num_features=1, window=None, epochs=100, training_flag=True, checkpoint_path="save/", continue_train = True, steps_per_epoch = 5):
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

def model_multi_lstm(OUT_STEPS, out_num_features, window, epochs, training_flag, checkpoint_path, continue_train = False, steps_per_epoch = 10):
    model = MultiLSTMModel(OUT_STEPS, out_num_features)

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

def model_multi_conv(OUT_STEPS, out_num_features, window, epochs, training_flag, checkpoint_path, continue_train = False, steps_per_epoch = 10):
    model = MultiConvModel(OUT_STEPS, out_num_features)

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
