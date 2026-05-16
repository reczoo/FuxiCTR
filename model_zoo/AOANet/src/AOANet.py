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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class AOANet(BaseModel):
    """Architecture and Operation Adaptive Network (AOANet) model.

    References:
        - Lang Lang, Zhenlong Zhu, Xuanye Liu, Jianxin Zhao, Jixing Xu, Minghui Shan:
          Architecture and Operation Adaptive Network for Online Recommendations, KDD 2021.
        - [PDF] https://dl.acm.org/doi/pdf/10.1145/3447548.3467133

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"AOANet"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        dnn_hidden_units (list): Hidden units for the DNN tower. Default: ``[64, 64, 64]``.
        dnn_hidden_activations (str): Activation functions for DNN. Default: ``"ReLU"``.
        num_interaction_layers (int): Number of interaction layers. Default: ``3``.
        num_subspaces (int): Number of output subspaces. Default: ``4``.
        net_dropout (float): Dropout rate for the network. Default: ``0``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="AOANet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_hidden_activations="ReLU",
                 num_interaction_layers=3,
                 num_subspaces=4,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AOANet, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=None, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.gin = GeneralizedInteractionNet(num_interaction_layers, 
                                             num_subspaces, 
                                             feature_map.num_fields, 
                                             embedding_dim)
        self.fc = nn.Linear(dnn_hidden_units[-1] + num_subspaces * embedding_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """Forward pass of AOANet.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feat_emb = self.embedding_layer(X)
        dnn_out = self.dnn(feat_emb.flatten(start_dim=1))
        interact_out = self.gin(feat_emb).flatten(start_dim=1)
        y_pred = self.fc(torch.cat([dnn_out, interact_out], dim=-1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class GeneralizedInteractionNet(nn.Module):
    """Generalized Interaction Network for AOANet.

    Args:
        num_layers (int): Number of interaction layers.
        num_subspaces (int): Number of output subspaces.
        num_fields (int): Number of input fields.
        embedding_dim (int): Dimension of feature embeddings.
    """
    def __init__(self, num_layers, num_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteractionNet, self).__init__()
        self.layers = nn.ModuleList([GeneralizedInteraction(num_fields if i == 0 else num_subspaces,
                                                           num_subspaces,
                                                           num_fields,
                                                           embedding_dim) \
                                     for i in range(num_layers)])

    def forward(self, B_0):
        """Forward pass through interaction layers.

        Args:
            B_0: Initial feature embedding tensor.

        Returns:
            torch.Tensor: Output after all interaction layers.
        """
        B_i = B_0
        for layer in self.layers:
            B_i = layer(B_0, B_i)
        return B_i


class GeneralizedInteraction(nn.Module):
    """Generalized Interaction layer for AOANet.

    Args:
        input_subspaces (int): Number of input subspaces.
        output_subspaces (int): Number of output subspaces.
        num_fields (int): Number of input fields.
        embedding_dim (int): Dimension of feature embeddings.
    """
    def __init__(self, input_subspaces, output_subspaces, num_fields, embedding_dim):
        super(GeneralizedInteraction, self).__init__()
        self.input_subspaces = input_subspaces
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.eye(embedding_dim, embedding_dim).unsqueeze(0).repeat(output_subspaces, 1, 1))
        self.alpha = nn.Parameter(torch.ones(input_subspaces * num_fields, output_subspaces))
        self.h = nn.Parameter(torch.ones(output_subspaces, embedding_dim, 1))

    def forward(self, B_0, B_i):
        """Forward pass of generalized interaction.

        Args:
            B_0: Initial feature embedding tensor.
            B_i: Current feature embedding tensor.

        Returns:
            torch.Tensor: Transformed feature embedding.
        """
        outer_product = torch.einsum("bnh,bnd->bnhd",
                                     B_0.repeat(1, self.input_subspaces, 1),
                                     B_i.repeat(1, 1, self.num_fields).view(B_i.size(0), -1, self.embedding_dim)) # b x (field*in) x d x d
        fusion = torch.matmul(outer_product.permute(0, 2, 3, 1), self.alpha) # b x d x d x out
        fusion = self.W * fusion.permute(0, 3, 1, 2) # b x out x d x d
        B_i = torch.matmul(fusion, self.h).squeeze(-1) # b x out x d
        return B_i
