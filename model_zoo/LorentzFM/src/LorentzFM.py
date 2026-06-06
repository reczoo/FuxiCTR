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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction
from itertools import combinations


class LorentzFM(BaseModel):
    """Lorentz Factorization Machine model.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"LorentzFM"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``10``.
        embedding_dropout (float): Dropout rate for embeddings. Default: ``0``.
        regularizer (str or None): Regularizer for both embeddings and network. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="LorentzFM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 embedding_dropout=0,
                 regularizer=None,
                 **kwargs):
        super(LorentzFM, self).__init__(feature_map,
                                        model_id=model_id,
                                        gpu=gpu,
                                        embedding_regularizer=regularizer,
                                        net_regularizer=regularizer,
                                        **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.inner_product_layer = InnerProductInteraction(feature_map.num_fields, output="inner_product")
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of LorentzFM.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # bs x field x dim
        inner_product = self.inner_product_layer(feature_emb) # bs x (field x (field - 1) / 2)
        zeroth_components = self.get_zeroth_components(feature_emb) # batch * field
        y_pred = self.triangle_pooling(inner_product, zeroth_components)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_zeroth_components(self, feature_emb):
        """Compute the 0th component for LorentzFM.

        Args:
            feature_emb: Feature embeddings of shape ``(batch, field, dim)``.

        Returns:
            Tensor: Zeroth components of shape ``(batch, field)``.
        """
        sum_of_square = torch.sum(feature_emb ** 2, dim=-1) # batch * field
        zeroth_components = torch.sqrt(sum_of_square + 1) # beta = 1
        return zeroth_components # batch * field

    def triangle_pooling(self, inner_product, zeroth_components):
        """Apply triangle pooling with Lorentzian inner product.

        T(u,v) = (1 - <u, v>L - u0 - v0) / (u0 * v0)
               = (1 + u0 * v0 - inner_product - u0 - v0) / (u0 * v0)
               = 1 + (1 - inner_product - u0 - v0) / (u0 * v0)

        Args:
            inner_product: Inner product tensor.
            zeroth_components: Zeroth component tensor.

        Returns:
            Tensor: Pooled output.
        """
        num_fields = zeroth_components.size(1)
        p, q = zip(*list(combinations(range(num_fields), 2)))
        u0, v0 = zeroth_components[:, p], zeroth_components[:, q]  # batch * (f(f-1)/2)
        score_tensor = 1 + torch.div(1 - inner_product - u0 - v0, u0 * v0) # batch * (f(f-1)/2)
        output = torch.sum(score_tensor, dim=1, keepdim=True) # batch * 1
        return output