"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def _prepare_2d(inputs, max_len, pad=None):
    if pad is None:
        pad = 0.

    return np.stack([_pad_2d(x, max_len, pad) for x in tqdm(inputs)]).astype(np.float32), np.array([x.shape[0] for x in inputs], dtype=np.int32)


def _pad_2d(t, length, pad):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=pad)


def adjust_time_resolution(data, max_time_steps):
    data = [data[i: i+max_time_steps] for i in range(0, len(data), max_time_steps)]
    return data


def sequential_reshape(data, all_params):
    tqdm.pandas()
    data = data.groupby(data.customer_id).progress_apply(lambda x: x.drop('customer_id', axis=1).values)

    matrices = data.values
    data = []

    for matrix in tqdm(matrices):
        data += adjust_time_resolution(matrix, all_params.max_time_steps)

    data, lengths = _prepare_2d(data, max_len=all_params.max_time_steps, pad=0.)

    return data, lengths


def preprocess(train_in_path, test_in_path, all_params):
    df_train = pd.read_csv(train_in_path)
    df_test = pd.read_csv(test_in_path)

    df_train = select_and_order_columns(df_train)
    df_test = select_and_order_columns(df_test)

    x_train, lengths_train = sequential_reshape(df_train, all_params)
    x_test, lengths_test = sequential_reshape(df_test, all_params)

    dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
    dataset_test = tf.data.Dataset.from_tensor_slices(x_test)

    dataset_train_lengths = tf.data.Dataset.from_tensor_slices(lengths_train)
    dataset_test_lengths = tf.data.Dataset.from_tensor_slices(lengths_test)

    assert len(lengths_test) == len(x_test)
    assert len(lengths_train) == len(x_train)

    train_seed = np.random.randint(0, 100000)
    dataset_train = dataset_train.shuffle(10000, seed=train_seed).batch(all_params.batch_size).repeat(-1).prefetch(32)

    dataset_test_for_predict = dataset_test.batch(1)

    dataset_test_lengths_for_predict = dataset_test_lengths.batch(1)

    dataset_test = dataset_test.batch(len(x_test) // 4).repeat(-1)
    dataset_train_lengths = dataset_train_lengths.shuffle(10000, seed=train_seed).batch(all_params.batch_size).repeat(-1).prefetch(32)
    dataset_test_lengths = dataset_test_lengths.batch(len(lengths_test) // 4).repeat(-1)




    return dataset_train, dataset_test, dataset_train_lengths, dataset_test_lengths, dataset_test_for_predict, dataset_test_lengths_for_predict, x_test


def select_and_order_columns(df):
    feature_cols = ['Amount']
    feature_cols += [col for col in df.columns if col[0] == 'V']
    cols = ['customer_id'] + feature_cols + ['Class']

    return df[cols]

