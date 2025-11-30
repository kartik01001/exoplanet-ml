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

"""Tests for astro_wavenet_model.AstroWaveNetModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.astro_wavenet_model import astro_wavenet_model
from astronet.astro_wavenet_model import configurations
from astronet.ops import input_ops
from astronet.ops import testing
from tf_util import configdict


class AstroWaveNetModelTest(tf.test.TestCase):

  def assertShapeEquals(self, shape, tensor_or_array):
    """Asserts that a Tensor or Numpy array has the expected shape.

    Args:
      shape: Numpy array or anything that can be converted to one.
      tensor_or_array: tf.Tensor, tf.Variable, or Numpy array.
    """
    if isinstance(tensor_or_array, (np.ndarray, np.generic)):
      self.assertAllEqual(shape, tensor_or_array.shape)
    elif isinstance(tensor_or_array, (tf.Tensor, tf.Variable)):
      self.assertAllEqual(shape, tensor_or_array.shape.as_list())
    else:
      raise TypeError("tensor_or_array must be a Tensor or Numpy ndarray")

  def testOneTimeSeriesFeature(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        }
    }
    hidden_spec = {
        "time_feature_1": {
            "wavenet_num_blocks": 2,
            "wavenet_num_layers_per_block": 3,
            "wavenet_num_filters": 8,
            "wavenet_kernel_size": 3,
            "use_skip_connections": True,
            "use_residual_connections": True,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_wavenet_model.AstroWaveNetModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()

    # TODO(shallue): TensorFlow 2.0 doesn't have global variable collections.
    # If we want to keep testing variable shapes in 2.0, we must keep track of
    # the individual Keras Layer objects in the model class.
    variables = {v.op.name: v for v in tf.global_variables()}

    # Validate some Tensor shapes for dilated convolutions.
    # Initial conv should be (1, input_channels, num_filters)
    initial_conv = variables["time_feature_1_hidden/initial_conv/kernel"]
    self.assertShapeEquals((1, 1, 8), initial_conv)

    # Block 1, Layer 1: dilation_rate=1
    block_1_layer_1_conv = variables[
        "time_feature_1_hidden/block_1/layer_1/dilated_conv/kernel"]
    self.assertShapeEquals((3, 8, 8), block_1_layer_1_conv)

    # Block 1, Layer 2: dilation_rate=2
    block_1_layer_2_conv = variables[
        "time_feature_1_hidden/block_1/layer_2/dilated_conv/kernel"]
    self.assertShapeEquals((3, 8, 8), block_1_layer_2_conv)

    # Block 1, Layer 3: dilation_rate=4
    block_1_layer_3_conv = variables[
        "time_feature_1_hidden/block_1/layer_3/dilated_conv/kernel"]
    self.assertShapeEquals((3, 8, 8), block_1_layer_3_conv)

    # Block 2, Layer 1: dilation_rate=1
    block_2_layer_1_conv = variables[
        "time_feature_1_hidden/block_2/layer_1/dilated_conv/kernel"]
    self.assertShapeEquals((3, 8, 8), block_2_layer_1_conv)

    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    # Output shape should be (batch_size, length * num_filters) = (None, 20 * 8)
    self.assertShapeEquals((None, 160),
                           model.time_series_hidden_layers["time_feature_1"])
    self.assertEqual(len(model.aux_hidden_layers), 0)
    self.assertIs(model.time_series_hidden_layers["time_feature_1"],
                  model.pre_logits_concat)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)

  def testTwoTimeSeriesFeatures(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 5,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    hidden_spec = {
        "time_feature_1": {
            "wavenet_num_blocks": 2,
            "wavenet_num_layers_per_block": 2,
            "wavenet_num_filters": 6,
            "wavenet_kernel_size": 3,
            "use_skip_connections": True,
            "use_residual_connections": True,
        },
        "time_feature_2": {
            "wavenet_num_blocks": 1,
            "wavenet_num_layers_per_block": 2,
            "wavenet_num_filters": 4,
            "wavenet_kernel_size": 3,
            "use_skip_connections": True,
            "use_residual_connections": True,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_wavenet_model.AstroWaveNetModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()

    # TODO(shallue): TensorFlow 2.0 doesn't have global variable collections.
    # If we want to keep testing variable shapes in 2.0, we must keep track of
    # the individual Keras Layer objects in the model class.
    variables = {v.op.name: v for v in tf.global_variables()}

    # Validate Tensor shapes.
    feature_1_initial_conv = variables["time_feature_1_hidden/initial_conv/kernel"]
    self.assertShapeEquals((1, 1, 6), feature_1_initial_conv)

    feature_1_block_1_layer_1_conv = variables[
        "time_feature_1_hidden/block_1/layer_1/dilated_conv/kernel"]
    self.assertShapeEquals((3, 6, 6), feature_1_block_1_layer_1_conv)

    feature_2_initial_conv = variables["time_feature_2_hidden/initial_conv/kernel"]
    self.assertShapeEquals((1, 1, 4), feature_2_initial_conv)

    feature_2_block_1_layer_1_conv = variables[
        "time_feature_2_hidden/block_1/layer_1/dilated_conv/kernel"]
    self.assertShapeEquals((3, 4, 4), feature_2_block_1_layer_1_conv)

    self.assertItemsEqual(["time_feature_1", "time_feature_2"],
                          model.time_series_hidden_layers.keys())
    # time_feature_1: length=20, num_filters=6 -> output=120
    self.assertShapeEquals((None, 120),
                           model.time_series_hidden_layers["time_feature_1"])
    # time_feature_2: length=5, num_filters=4 -> output=20
    self.assertShapeEquals((None, 20),
                           model.time_series_hidden_layers["time_feature_2"])
    self.assertItemsEqual(["aux_feature_1"], model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    # pre_logits_concat: 120 + 20 + 1 = 141
    self.assertShapeEquals((None, 141), model.pre_logits_concat)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)


if __name__ == "__main__":
  tf.test.main()
