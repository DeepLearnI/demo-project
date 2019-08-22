"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import tensorflow as tf


class CausalConv1D(tf.layers.Conv1D):
	def __init__(self, filters,
		kernel_size,
		strides=1,
		dilation_rate=1,
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		trainable=True,
		name=None,
		**kwargs):
		super(CausalConv1D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='valid',
			data_format='channels_last',
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			trainable=trainable,
			name=name, **kwargs)

	def build(self, input_shape):
		'''Build layer
		'''
		input_shape = tf.TensorShape(input_shape).as_list()

		#Build parent layer to create initial filters
		super(CausalConv1D, self).build(input_shape)

	def call(self, inputs, queued=False, convolution_queue=None):
		'''Causal Conv !D call has two modes: normal and AR

		mormal mode just excecutes the network on a sequence of [batch_size, time, channels]
		AR (autoregressive) mode executes the network on subsequences of [batch_size, 1, channels] but keeps track of previous computations to use in next step
		AR mode is inspired by fast wavenet paper
		'''
		if queued:
			return self.queue_call(inputs, convolution_queue)

		padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
		inputs = tf.pad(inputs, tf.constant([(0, 0), (padding, 0), (0, 0)]))
		return super(CausalConv1D, self).call(inputs)
