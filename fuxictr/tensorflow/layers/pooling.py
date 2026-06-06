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
from tensorflow.keras import Model


class MaskedSumPooling(Model):
    """Sum pooling layer for sequence embeddings.

    Computes the sum of embeddings along the sequence dimension.
    """
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        """Apply sum pooling over the sequence axis.

        Args:
            embedding_matrix (tf.Tensor): Tensor of shape ``(batch_size, seq_len, emb_dim)``.

        Returns:
            tf.Tensor: Pooled tensor of shape ``(batch_size, emb_dim)``.
        """
        return tf.reduce_sum(embedding_matrix, axis=1)


