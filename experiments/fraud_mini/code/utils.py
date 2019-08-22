"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import logging
import configargparse
import sys
import os
from datetime import datetime
import urllib.request

def get_arguments_as_dict(args):
    return {k:str(v) for k,v in vars(args).items()}


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def get_log_dir(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def init_configuration(config_file="./config/config.yaml"):
    parser = configargparse.ArgParser(default_config_files=['config.yaml'],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', default=config_file, is_config_file=True, help='config file path')
    parser.add('--batch_size', default=256, type=int)
    parser.add('--max_time_steps', default=100, type=int)
    parser.add('--tf_log_level', default=1, type=int)
    parser.add('--base_dir', default='')
    parser.add('--name', default='LSTM')
    parser.add('--model', default='LSTM')
    parser.add('--dense_units', action='append', default=[], type=int)
    parser.add('--dropout', action='append', default=[], type=float)
    parser.add('--learning_rate', default=0.001, type=float)
    parser.add('--min_learning_rate', default=0.00001, type=float)
    parser.add('--decay_steps', default=52000, type=float)
    parser.add('--decay_rate', default=0.5, type=float)
    parser.add('--reg_weight', default=0.000001, type=float)
    parser.add('--beta1', default=0.9, type=float)
    parser.add('--beta2', default=0.999, type=float)
    parser.add('--epsilon', default=0.000001, type=float)
    parser.add('--clip_gradients', action='store_true')
    parser.add('--grad_clip_val', default=1., type=float)
    parser.add('--train_steps', default=300000, type=int)
    parser.add('--restore', action='store_true')
    parser.add('--checkpoint_interval', default=10000, type=int)
    parser.add('--eval_interval', default=30000, type=int)
    parser.add('--summary_interval', default=250, type=int)
    parser.add('--pos_weight', default=5, type=float)
    parser.add('--recurrency_type', default='LSTM')
    parser.add('--lstm_units', default=[], action='append', type=int)
    parser.add('--zoneout', default=[], action='append', type=float)
    parser.add('--capture_weight', default=40, type=float)
    parser.add('--hidden_activation', default='selu')
    parser.add('--use_ema', action='store_true')
    parser.add('--ema_decay', default=0.9999, type=float)
    parser.add('--label_smoothing', action='store_true')
    parser.add('--positive_smoothing', default=0.2, type=float)
    parser.add('--negative_smoothing', default=0., type=float)
    parser.add('--random_seed', default=42, type=int)

    return parser.parse_args()


def download_data(reload=True):
    train_path = "/tmp/creditcard_train.csv"
    test_path = "/tmp/creditcard_test.csv"
    if reload:
        urllib.request.urlretrieve("https://www.dropbox.com/s/3dl28vfsh6lo8s9/creditcard_train.csv?dl=1", train_path)
        urllib.request.urlretrieve("https://www.dropbox.com/s/ie4dhmtcxgsrb50/creditcard_test.csv?dl=1", test_path)
    return train_path, test_path


class ValueWindow:
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []