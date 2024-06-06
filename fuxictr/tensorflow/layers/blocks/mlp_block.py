# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================


import tensorflow as tf
from fuxictr.tensorflow.tf_utils import get_activation, get_initializer, get_regularizer
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LayerNormalization, Dropout


class MLP_Block(Layer):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 layer_norm=False,
                 norm_before_activation=True,
                 use_bias=True,
                 initializer="glorot_normal",
                 regularizer=None):
        super(MLP_Block, self).__init__()
        self.mlp = tf.keras.Sequential()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.mlp.add(Dense(hidden_units[idx + 1], use_bias=use_bias, 
                               kernel_initializer=get_initializer(initializer), 
                               kernel_regularizer=get_regularizer(regularizer),
                               bias_regularizer=get_regularizer(regularizer)))
            if norm_before_activation:
                if batch_norm:
                    self.mlp.add(BatchNormalization(hidden_units[idx + 1]))
                elif layer_norm:
                    self.mlp.add(LayerNormalization(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                self.mlp.add(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    self.mlp.add(BatchNormalization(hidden_units[idx + 1]))
                elif layer_norm:
                    self.mlp.add(LayerNormalization(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.mlp.add(Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            self.mlp.add(Dense(output_dim, use_bias=use_bias, 
                               kernel_initializer=get_initializer(initializer), 
                               kernel_regularizer=get_regularizer(regularizer),
                               bias_regularizer=get_regularizer(regularizer)))
        if output_activation is not None:
            self.mlp.add(get_activation(output_activation))
    
    def call(self, inputs, training=None):
        return self.mlp(inputs, training=training)

