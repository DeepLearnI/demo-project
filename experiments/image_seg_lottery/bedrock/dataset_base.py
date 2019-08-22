# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A base class for managing datasets."""

import tensorflow as tf
from satellite_segmentation.constants import NUM_EPOCHS


class DatasetSplit(object):
  """A split of a dataset, for example just the training data."""

  def __init__(self, data, batch_size=None, shuffle=False, seed=None):
    size = data[0].shape[0] // 4
    #size = 4096 # HN - can change if memory intensive
    height = data[0].shape[1]
    width = data[0].shape[2]
    channels = data[0].shape[3]
    classes = data[1].shape[3]

    index = 0
    def generator():
      index = 0
      epoch = 0
      X, y = data
      while True:
        if index >= len(X):
          index = 0
          epoch += 1

        X_r, y_r = X[index: index+batch_size], y[index: index+batch_size]
        index += batch_size

        if epoch >= NUM_EPOCHS:
          break

        yield X_r, y_r

    # Build the dataset.
    self._dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), (tf.TensorShape([None, height, width, channels]), 
                                                                                         tf.TensorShape([None, height, width, classes])))

    # Shuffle if applicable.
    if shuffle:
      self._dataset = self._dataset

    # Create an iterator for the dataset.
    self._iterator = self._dataset.make_initializable_iterator()
    self._initializer = self._iterator.initializer

  @property
  def initializer(self):
    return self._initializer

  @property
  def dataset(self):
    return self._dataset

  def get_handle(self, sess):
    return sess.run(self._iterator.string_handle())


class DatasetBase(object):
  """The base class for a dataset."""

  def __init__(self,
               train,
               train_batch_size,
               test,
               validate=None,
               train_order_seed=None):
    # Create datasets for each segment of the data.
    self._train = DatasetSplit(
        train, train_batch_size, shuffle=True, seed=train_order_seed)
    self._test = DatasetSplit(test, train_batch_size)
    if validate:
      self._validate = DatasetSplit(validate, train_batch_size)
    else:
      self._validate = None

    # Create the overall iterator.
    self._handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        self._handle, self._train.dataset.output_types,
        self._train.dataset.output_shapes)
    self._placeholders = iterator.get_next()

  @property
  def train_initializer(self):
    return self._train.initializer

  @property
  def test_initializer(self):
    return self._test.initializer

  @property
  def validate_initializer(self):
    return self._validate.initializer

  def get_train_handle(self, sess):
    return self._train.get_handle(sess)

  def get_test_handle(self, sess):
    return self._test.get_handle(sess)

  def get_validate_handle(self, sess):
    if self._validate is None:
      return None
    else:
      return self._validate.get_handle(sess)

  @property
  def handle(self):
    return self._handle

  @property
  def placeholders(self):
    return self._placeholders
