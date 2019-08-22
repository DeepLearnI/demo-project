"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import tensorflow as tf
from causal_conv import CausalConv1D


class LayerNorm:
    def __init__(self, axis=-1, reuse=False, epsilon=1e-8, name='LayerNorm'):
        self.epsilon = epsilon
        self.axis = axis
        self.reuse = reuse
        self.scope = name
        
    def __call__(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            inputs_shape = inputs.shape[-1]
    
            beta = tf.get_variable("beta", [inputs_shape], initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", [inputs_shape], initializer=tf.ones_initializer())
            
            mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=self.axis, keepdims=True)
            
            normalized = (inputs - mean) * tf.rsqrt(variance + self.epsilon)
            outputs = gamma * normalized + beta
        
        return outputs


class VanillaTCNBlock(tf.layers.Layer):
	def __init__(self, out_channels, kernel_size, strides=1, dilation_rate=1, dropout=0.5, spatial_dropout=True,
		trainable=True, name=None, dtype=None,
		activity_regularizer=None, **kwargs):
		super(VanillaTCNBlock, self).__init__(
			trainable=trainable, dtype=dtype,
			activity_regularizer=activity_regularizer,
			name=name, **kwargs)

		self.dropout = dropout
		self.out_channels = out_channels
		self.resample = None #This will determine the option of using conv1x1 for residual connection (to change number of channels)
		self.spatial_dropout = spatial_dropout

		assert type(dilation_rate) == type(kernel_size) == type(strides) == int

		if dilation_rate != 1:
			assert strides == 1

		self.conv1 = CausalConv1D(
			out_channels, kernel_size, strides=strides,
			dilation_rate=dilation_rate, activation=tf.nn.relu,
			name='{}_conv1'.format(name))

		self.conv2 = CausalConv1D(
			out_channels, kernel_size, strides=strides,
			dilation_rate=dilation_rate, activation=tf.nn.relu,
			name='{}_conv2'.format(name))

		self.ln1 = LayerNorm(axis=-1, name='{}_ln1'.format(name))
		self.ln2 = LayerNorm(axis=-1, name='{}_ln2'.format(name))

	def build(self, input_shape):
		input_shape = tf.TensorShape(input_shape).as_list()

		#If input and output channels do not match, we need to change input channels to be able to apply residual connections
		if input_shape[-1] != self.out_channels:
			self.resample = CausalConv1D(self.out_channels, kernel_size=1, activation=None, 
				name='{}_resample_conv1x1'.format(self.name))
		super(VanillaTCNBlock, self).build(input_shape)

	def call(self, inputs, training=True, queued=False, states=None):
		noise_shape = [tf.shape(inputs)[0], tf.constant(1), tf.constant(self.out_channels)] if self.spatial_dropout else None
		if queued:
			return self.queue_call(inputs, training=training, states=states)

		x = self.conv1(inputs)
		x = self.ln1(x)
		x = tf.layers.dropout(x, self.dropout, noise_shape=noise_shape, training=training)

		x = self.conv2(x)
		x = self.ln2(x)
		x = tf.layers.dropout(x, self.dropout, noise_shape=noise_shape, training=training)

		if self.resample is not None:
			inputs = self.resample(inputs)

		return tf.nn.relu(x + inputs)

