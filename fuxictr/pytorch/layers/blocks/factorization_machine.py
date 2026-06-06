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


import torch
from torch import nn
from .logistic_regression import LogisticRegression
from ..interactions import InnerProductInteraction


class FactorizationMachine(nn.Module):
    """Factorization Machine layer that combines second-order feature interactions with logistic regression.

    ``FactorizationMachine`` computes pairwise inner products of feature embeddings (FM component)
    and adds a linear logistic regression term (LR component) to produce the final output.

    Args:
        feature_map (FeatureMap): A ``FeatureMap`` instance that provides the number of fields
            and feature metadata.

    Example::

        fm = FactorizationMachine(feature_map)
        output = fm(X, feature_emb)
    """

    def __init__(self, feature_map):
        super(FactorizationMachine, self).__init__()
        self.fm_layer = InnerProductInteraction(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)

    def forward(self, X, feature_emb):
        """Compute the FM output.

        Args:
            X (dict): Raw feature inputs.
            feature_emb (torch.Tensor): Feature embeddings of shape (batch_size, num_fields, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        lr_out = self.lr_layer(X)
        fm_out = self.fm_layer(feature_emb)
        output = fm_out + lr_out
        return output

