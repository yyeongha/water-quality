#####################################################################################
# 다양한 시계열 예측 모델을 정의하고, 이를 학습하고 평가하는 함수를 제공
# 각 모델은 특정 구조를 가지고 있으며, 학습을 위한 유틸리티 함수와 함께 사용됨
#####################################################################################

# -*- coding: utf-8 -*-

from core.models import *

# multi_linear 모델을 생성하고 학습하거나 평가
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

# elman 모델을 생성하고 학습하거나 평가
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

# gru 모델을 생성하고 학습하거나 평가
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

# lstm 모델을 생성하고 학습하거나 평가
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

# 1D Convolution 모델을 생성하고 학습하거나 평가
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
