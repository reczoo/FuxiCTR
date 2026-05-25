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
from fuxictr.pytorch.layers import FeatureEmbedding


class LogisticRegression(nn.Module):
    """Logistic Regression layer that learns a linear weighted sum of input features.

    ``LogisticRegression`` uses 1-dimensional embeddings as feature weights and sums them
    with an optional bias term to produce a linear prediction.

    Args:
        feature_map (FeatureMap): A ``FeatureMap`` instance that provides feature metadata.
        use_bias (bool, optional): Whether to include a learnable bias term. Default: ``True``.

    Example::

        lr = LogisticRegression(feature_map, use_bias=True)
        output = lr(X)
    """

    def __init__(self, feature_map, use_bias=True):
        super(LogisticRegression, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False, use_sharing=False)

    def forward(self, X):
        """Compute the LR output.

        Args:
            X (dict): Raw feature inputs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        embed_weights = self.embedding_layer(X)
        output = embed_weights.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        return output

