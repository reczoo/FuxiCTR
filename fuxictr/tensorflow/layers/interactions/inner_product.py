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
import numpy as np
from tensorflow.keras.layers import Layer


class InnerProductInteraction(Layer):
    """ output: product_sum (bs x 1), 
                bi_interaction (bs * dim), 
                inner_product (bs x f^2/2), 
                elementwise_product (bs x f^2/2 x emb_dim)
    """
    def __init__(self, num_fields, output="product_sum"):
        super(InnerProductInteraction, self).__init__()
        self.output_type = output
        if output not in ["product_sum", "bi_interaction", "inner_product", "elementwise_product"]:
            raise ValueError("InnerProductInteraction output={} is not supported.".format(output))
        if output == "inner_product":
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.triu_mask = tf.Variable(np.triu(np.ones((num_fields, num_fields)), 1).astype(bool),
                                         trainable=False)
        elif output == "elementwise_product":
            self.triu_index = tf.Variable(np.triu_indices(num_fields, 1), trainable=False)

    def call(self, feature_emb):
        if self.output_type in ["product_sum", "bi_interaction"]:
            sum_of_square = tf.reduce_sum(feature_emb, axis=1) ** 2  # sum then square
            square_of_sum = tf.reduce_sum(feature_emb ** 2, axis=1) # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self.output_type == "bi_interaction":
                return bi_interaction
            else:
                return tf.reduce_sum(bi_interaction, axis=-1, keepdims=True)
        elif self.output_type == "inner_product":
            inner_product_matrix = tf.einsum('bij,bji->bii', feature_emb, feature_emb.transpose(1, 2))
            triu_values = tf.boolean_mask(inner_product_matrix, self.triu_mask)
            return tf.reshape(triu_values, (-1, self.interaction_units))
        elif self.output_type == "elementwise_product":
            emb1 = tf.gather(feature_emb, self.triu_index[0], axis=1)
            emb2 = tf.gather(feature_emb, self.triu_index[1], axis=1)
            return emb1 * emb2

