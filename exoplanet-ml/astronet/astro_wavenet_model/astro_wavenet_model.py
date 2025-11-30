# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A model for classifying light curves using WaveNet architecture.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
    (WaveNet blocks 1)        (WaveNet blocks 2)        ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features

WaveNet blocks use dilated convolutions with exponentially increasing dilation
rates, skip connections, and residual connections to efficiently capture
long-range dependencies in time series data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model


class AstroWaveNetModel(astro_model.AstroModel):
  """A model for classifying light curves using WaveNet architecture."""

  def _build_wavenet_layers(self, inputs, hparams, scope="wavenet"):
    """Builds WaveNet layers with dilated convolutions.

    The layers are defined by blocks, each containing multiple dilated
    convolutional layers with exponentially increasing dilation rates.
    Each layer has residual connections, and outputs from all layers are
    combined via skip connections.

    Args:
      inputs: A Tensor of shape [batch_size, length] or
        [batch_size, length, ndims].
      hparams: Object containing WaveNet hyperparameters.
      scope: Prefix for operation names.

    Returns:
      A Tensor of shape [batch_size, output_size], where the output size depends
      on the input size and the number of filters.
    """
    with tf.variable_scope(scope):
      net = inputs
      if net.shape.rank == 2:
        net = tf.expand_dims(net, -1)  # [batch, length] -> [batch, length, 1]
      if net.shape.rank != 3:
        raise ValueError(
            "Expected inputs to have rank 2 or 3. Got: {}".format(inputs))

      num_filters = int(hparams.wavenet_num_filters)
      kernel_size = int(hparams.wavenet_kernel_size)
      
      # Initial 1x1 convolution to match filter dimensions
      net = tf.layers.conv1d(
          inputs=net,
          filters=num_filters,
          kernel_size=1,
          padding="same",
          activation=None,
          name="initial_conv")

      skip_connections = []

      # Build WaveNet blocks
      for block_idx in range(hparams.wavenet_num_blocks):
        with tf.variable_scope("block_{}".format(block_idx + 1)):
          # Each block has layers with exponentially increasing dilation rates
          for layer_idx in range(hparams.wavenet_num_layers_per_block):
            dilation_rate = 2 ** layer_idx
            layer_name = "layer_{}".format(layer_idx + 1)
            
            with tf.variable_scope(layer_name):
              # Store input for residual connection
              residual = net
              
              # Dilated convolution
              conv_out = tf.layers.conv1d(
                  inputs=net,
                  filters=num_filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  padding="same",
                  activation=tf.nn.relu,
                  name="dilated_conv")
              
              # Skip connection (1x1 convolution)
              skip = tf.layers.conv1d(
                  inputs=conv_out,
                  filters=num_filters,
                  kernel_size=1,
                  padding="same",
                  activation=None,
                  name="skip_conv")
              skip_connections.append(skip)
              
              # Residual connection (1x1 convolution + add)
              if hparams.get("use_residual_connections", True):
                residual_conv = tf.layers.conv1d(
                    inputs=conv_out,
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    activation=None,
                    name="residual_conv")
                net = residual + residual_conv
              else:
                net = conv_out

      # Combine skip connections
      if hparams.get("use_skip_connections", True):
        skip_sum = tf.add_n(skip_connections, name="skip_sum")
        skip_combined = tf.nn.relu(skip_sum, name="skip_relu")
        
        # 1x1 convolution to combine skip connections
        net = tf.layers.conv1d(
            inputs=skip_combined,
            filters=num_filters,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            name="skip_combine")

      # Flatten
      net.shape.assert_has_rank(3)
      net_shape = net.shape.as_list()
      output_dim = net_shape[1] * net_shape[2]
      net = tf.reshape(net, [-1, output_dim], name="flatten")

    return net

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
      time_series_hidden_layers[name] = self._build_wavenet_layers(
          inputs=time_series,
          hparams=self.hparams.time_series_hidden[name],
          scope=name + "_hidden")

    self.time_series_hidden_layers = time_series_hidden_layers
