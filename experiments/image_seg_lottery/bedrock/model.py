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

"""A fully-connected neural network model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bedrock import model_base
import tensorflow as tf


class ModelU(model_base.ModelBase):
    """A fully-connected network with user-specifiable hyperparameters."""
    
    def __init__(self,
                 hyperparameters,
                 input_placeholder,
                 label_placeholder,
                 presets=None,
                 masks=None):
        """Creates a fully-connected network.
    
        Args:
          hyperparameters: A dictionary of hyperparameters for the network.
            For this class, a single hyperparameter is available: 'layers'. This
            key's value is a list of (# of units, activation function) tuples
            for each layer in order from input to output. If the activation
            function is None, then no activation will be used.
          input_placeholder: A placeholder for the network's input.
          label_placeholder: A placeholder for the network's expected output.
          presets: Preset initializations for the network as in model_base.py
          masks: Masks to prune the network as in model_base.py.
        """
        # Call parent constructor.
        super(ModelU, self).__init__(presets=presets, masks=masks)
        
        # Build the network layer by layer.
        layers = hyperparameters['layers']
        
        # Encoder section
        current_layer, skip1 = self.encoder_block(0, input_placeholder, layers[0])
        print("skip1", skip1.shape)
        current_layer, skip2 = self.encoder_block(1, current_layer, layers[1])
        print("skip2", skip2.shape)
        current_layer, skip3 = self.encoder_block(2, current_layer, layers[2])
        print("skip3", skip3.shape)
        current_layer, skip4 = self.encoder_block(3, current_layer, layers[3])
        print("skip4", skip4.shape)
        
        # Bottom layer
        current_layer = self.conv2d_block(current_layer, layers[4], name='bottom_layer')
        
        # Decoder section
        current_layer = self.decoder_block(5, current_layer, skip4, layers[5])
        current_layer = self.decoder_block(6, current_layer, skip3, layers[6])
        current_layer = self.decoder_block(7, current_layer, skip2, layers[7])
        current_layer = self.decoder_block(8, current_layer, skip1, layers[8])
        
        current_layer = self.Conv2D('output_layer', current_layer, layers[9], kernel_size=(1, 1))
        self.outputs = tf.nn.sigmoid(current_layer)
        self.targets = label_placeholder
        self.inputs = input_placeholder
        
        # Compute the loss and accuracy.
        self.create_loss(label_placeholder, current_layer)  # params (label_placeholder, output_logits)
        
    def encoder_block(self, block_number, inputs, n_filters, kernel_size=3, batchnorm=True):
        print("Creating encoder block ".format('encoder_conv_{}'.format(block_number)))
        skip_outputs = self.conv2d_block(inputs, n_filters, kernel_size=kernel_size, batchnorm=batchnorm, name='encoder_conv_{}'.format(block_number))
        outputs = tf.keras.layers.MaxPooling2D((2, 2), name='max_pool_{}'.format(block_number))(skip_outputs)#, strides=(1, 1), padding='same', name='max_pool_{}'.format(block_number))(skip_outputs)
        # return tf.layers.Dropout(rate=0.1)(outputs), skip_outputs
        return outputs, skip_outputs
    
    def decoder_block(self, block_number, inputs, skip_input, n_filters, kernel_size=3, batchnorm=True):
        print("Creating decoder block ".format('decoder_conv_{}'.format(block_number)))
        outputs = self.Conv2DTranspose('transpose_conv2d_{}'.format(block_number), inputs, n_filters, kernel_size=(kernel_size, kernel_size),
                                       strides=(1, 2, 2, 1))
        print("output block {}, shape {}".format(block_number, outputs.shape))
        outputs = tf.concat([outputs, skip_input], axis=-1)
        # outputs = tf.layers.Dropout(rate=0.1)(outputs)
        outputs = self.conv2d_block(outputs, n_filters, kernel_size=kernel_size, batchnorm=batchnorm, name='decoder_conv_{}'.format(block_number))
        return outputs

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True, name='conv2d_block'):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = self.Conv2D(name='{}_conv1'.format(name), inputs=input_tensor, channels=n_filters, kernel_size=(kernel_size, kernel_size),
                        kernel_initializer=None)#(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
    
        # second layer
        x = self.Conv2D(name='{}_conv2'.format(name), inputs=x, channels=n_filters, kernel_size=(kernel_size, kernel_size),
                        kernel_initializer=None)#(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
    
        return x


class ModelFc(model_base.ModelBase):
  """A fully-connected network with user-specifiable hyperparameters."""

  def __init__(self,
               hyperparameters,
               input_placeholder,
               label_placeholder,
               presets=None,
               masks=None):
    """Creates a fully-connected network.

    Args:
      hyperparameters: A dictionary of hyperparameters for the network.
        For this class, a single hyperparameter is available: 'layers'. This
        key's value is a list of (# of units, activation function) tuples
        for each layer in order from input to output. If the activation
        function is None, then no activation will be used.
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
    # Call parent constructor.
    super(ModelFc, self).__init__(presets=presets, masks=masks)

    # Build the network layer by layer.
    current_layer = input_placeholder
    for i, (units, activation) in enumerate(hyperparameters['layers']):

      if i < 2: # first two layers are convolutional
          print("Creating convolutional layer {}".format(i))
          #print("this is conv layer {}".format(i))
          current_layer = self.Conv2D('layer{}'.format(i),
                                      inputs=current_layer,
                                      channels=units,
                                      kernel_size=(3,3),
                                      activation=activation,
                                      kernel_initializer=tf.initializers.he_normal()
                                      )

      
      if i == 2:
          print("Adding Flatten after convolutional layers")
          current_layer = tf.reshape(current_layer, [tf.shape(current_layer)[0],
                                                     current_layer.shape[1] * current_layer.shape[2] * current_layer.shape[-1]]
                                     )
      if i >= 2:
          current_layer = self.dense_layer(
              'layer{}'.format(i), #name
              current_layer, #inputs
              units,
              activation,
              kernel_initializer=tf.initializers.he_normal() #tf.contrib.layers.xavier_initializer(uniform=False)
          )

    # Compute the loss and accuracy.
    self.create_loss_and_accuracy(label_placeholder, current_layer)  # params (label_placeholder, output_logits)
