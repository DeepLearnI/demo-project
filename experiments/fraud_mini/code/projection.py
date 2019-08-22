"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import tensorflow as tf


class Projection:
    """Projection to a scalar and through a sigmoid activation
    """
    
    def __init__(self, is_training, drop_rate=0., shape=1, activation=tf.nn.sigmoid, using_selu=True, scope=None):
        """
        Args:
            is_training: Boolean, to control the use of sigmoid function as it is useless to use it
                during training since it is integrate inside the sigmoid_crossentropy loss
            shape: integer, dimensionality of output space. Defaults to 1 (scalar)
            activation: callable, activation function. only used during inference
            scope: Projection scope.
        """
        super(Projection, self).__init__()
        self.is_training = is_training
        
        self.shape = shape
        self.drop_rate = drop_rate
        self.activation = activation if not using_selu else tf.nn.selu
        self.using_selu = using_selu
        self.scope = 'projection_layer' if scope is None else scope
    
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') if self.using_selu else None,
                                     activation=None, name='dense_{}'.format(self.scope))
            
            if self.activation is not None:
                output = self.activation(output)
            
            if self.using_selu:
                if self.is_training:
                    output = tf.contrib.nn.alpha_dropout(output, keep_prob=(1 - self.drop_rate),
                                                         name='dropout_{}'.format(self.scope))
            
            else:
                output = tf.layers.dropout(output, rate=self.drop_rate, training=self.is_training,
                                           name='dropout_{}'.format(self.scope))
            
            return output
