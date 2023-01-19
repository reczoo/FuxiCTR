# =========================================================================
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
from tensorflow.keras.layers import Layer, Dense


class CrossInteraction(Layer):
    def __init__(self, input_dim):
        super(CrossInteraction, self).__init__()
        self.weight = Dense(1, use_bias=False)
        self.bias = tf.Variable(tf.zeros(input_dim))

    def call(self, X_0, X_i):
        interact_out = self.weight(X_i) * X_0 + self.bias
        return interact_out


class CrossNet(Layer):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = []
        for _ in range(self.num_layers):
            self.cross_net.append(CrossInteraction(input_dim))

    def call(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CrossNetV2(Layer):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = []
        for _ in range(self.num_layers):
            self.cross_layers.append(Dense(input_dim))         

    def call(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i