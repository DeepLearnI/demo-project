"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from model import Model, save_freeze_tensorflow_model_for_inference, convert_to_tensor_rt, inference_from_tensor_rt_graph
from preprocessing import preprocess
from utils import init_configuration, download_data, get_log_dir, get_arguments_as_dict

# read the parameters from the config file
all_params = init_configuration(config_file='config/config.yaml')

# getting log directory to save the model and results
log_dir = get_log_dir(all_params)

print('downloading data')
train_path, test_path = download_data(reload=True)

print('preprocessing data')
dataset_train, dataset_test, dataset_train_lengths, dataset_test_lengths, dataset_test_for_predict, dataset_test_lengths_for_predict, x_test = preprocess(train_path, test_path, all_params)

print('initialize and train the model')
model = Model(log_dir, all_params)
model.train(dataset_train, dataset_test, dataset_train_lengths, dataset_test_lengths)

model.predict(dataset_test_for_predict, dataset_test_lengths_for_predict)


frozen_graph, your_outputs = save_freeze_tensorflow_model_for_inference(log_dir)

convert_to_tensor_rt(log_dir, frozen_graph, your_outputs)

