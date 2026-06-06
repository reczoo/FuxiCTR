# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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


from fuxictr.tensorflow.tf_utils import get_initializer, get_regularizer
from tensorflow.keras.layers import Layer, Dense


class Linear(Layer):
    """Simple dense linear layer wrapping ``tf.keras.layers.Dense``.

    Args:
        output_dim (int): Output dimension.
        use_bias (bool): Whether to use a bias vector. Default: ``True``.
        initializer (str): Kernel initializer name. Default: ``"glorot_normal"``.
        regularizer (optional): Optional kernel/bias regularizer. Default: ``None``.
    """
    def __init__(self,
                 output_dim,
                 use_bias=True,
                 initializer="glorot_normal",
                 regularizer=None):
        super(Linear, self).__init__()
        self.linear = Dense(output_dim, use_bias=use_bias,
                            kernel_initializer=get_initializer(initializer),
                            kernel_regularizer=get_regularizer(regularizer),
                            bias_regularizer=get_regularizer(regularizer))

    def call(self, inputs):
        """Apply the linear transformation.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Transformed tensor.
        """
        return self.linear(inputs)
