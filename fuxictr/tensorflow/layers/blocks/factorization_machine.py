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
from tensorflow.keras.layers import Layer
from .logistic_regression import LogisticRegression
from ..interactions import InnerProductInteraction


class FactorizationMachine(Layer):
    """Factorization Machine layer combining linear and second-order feature interactions.

    Args:
        feature_map (FeatureMap): Feature map object.
        regularizer (optional): Optional regularizer for the logistic regression component. Default: ``None``.
    """
    def __init__(self, feature_map, regularizer=None):
        super(FactorizationMachine, self).__init__()
        self.fm_layer = InnerProductInteraction(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True, regularizer=regularizer)

    def call(self, X, feature_emb):
        """Compute the FM output.

        Args:
            X (dict): Input feature dictionary.
            feature_emb (tf.Tensor): Feature embeddings of shape ``(batch_size, num_fields, emb_dim)``.

        Returns:
            tf.Tensor: FM output tensor of shape ``(batch_size, 1)``.
        """
        lr_out = self.lr_layer(X)
        fm_out = self.fm_layer(feature_emb)
        output = fm_out + lr_out
        return output
