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

"""Configurations for model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.astro_model import configurations as parent_configs


def base():
  """Base configuration for a WaveNet model with a single global view."""
  config = parent_configs.base()

  # Add configuration for the WaveNet layers of the global_view feature.
  config["hparams"]["time_series_hidden"] = {
      "global_view": {
          "wavenet_num_blocks": 4,
          "wavenet_num_layers_per_block": 5,
          "wavenet_num_filters": 32,
          "wavenet_kernel_size": 3,
          "use_skip_connections": True,
          "use_residual_connections": True,
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 512
  return config


def local_global():
  """Base configuration for a WaveNet model with separate local/global views."""
  config = parent_configs.base()

  # Override the model features to be local_view and global_view time series.
  config["inputs"]["features"] = {
      "local_view": {
          "length": 201,
          "is_time_series": True,
          "subcomponents": []
      },
      "global_view": {
          "length": 2001,
          "is_time_series": True,
          "subcomponents": []
      },
  }

  # Add configurations for the WaveNet layers of time series features.
  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          "wavenet_num_blocks": 3,
          "wavenet_num_layers_per_block": 4,
          "wavenet_num_filters": 32,
          "wavenet_kernel_size": 3,
          "use_skip_connections": True,
          "use_residual_connections": True,
      },
      "global_view": {
          "wavenet_num_blocks": 4,
          "wavenet_num_layers_per_block": 5,
          "wavenet_num_filters": 32,
          "wavenet_kernel_size": 3,
          "use_skip_connections": True,
          "use_residual_connections": True,
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 4
  config["hparams"]["pre_logits_hidden_layer_size"] = 512
  return config
