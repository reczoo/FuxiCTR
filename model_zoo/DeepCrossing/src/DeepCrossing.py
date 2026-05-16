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
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation


class DeepCrossing(BaseModel):
    """Deep Crossing model with residual blocks.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"DeepCrossing"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        residual_blocks (list): List of hidden dimensions for residual blocks. Default: ``[64, 64, 64]``.
        hidden_activations (str): Activation functions for hidden layers. Default: ``"ReLU"``.
        net_dropout (float): Dropout rate for the network. Default: ``0``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
        use_residual (bool): Whether to use residual connections. Default: ``True``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="DeepCrossing",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 residual_blocks=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_residual=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DeepCrossing, self).__init__(feature_map,
                                           model_id=model_id,
                                           gpu=gpu,
                                           embedding_regularizer=embedding_regularizer,
                                           net_regularizer=net_regularizer,
                                           **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(residual_blocks)
        layers = []
        input_dim = feature_map.num_fields * embedding_dim
        for hidden_dim, hidden_activation in zip(residual_blocks, hidden_activations):
            layers.append(ResidualBlock(input_dim, 
                                        hidden_dim,
                                        hidden_activation,
                                        net_dropout,
                                        use_residual,
                                        batch_norm))
        layers.append(nn.Linear(input_dim, 1))
        self.crossing_layer = nn.Sequential(*layers) # * used to unpack list
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """Forward pass of DeepCrossing.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.crossing_layer(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class ResidualBlock(nn.Module):
    """Residual block for DeepCrossing.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        hidden_activation (str): Activation function name. Default: ``"ReLU"``.
        dropout_rate (float): Dropout rate. Default: ``0``.
        use_residual (bool): Whether to use residual connections. Default: ``True``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 hidden_activation="ReLU",
                 dropout_rate=0,
                 use_residual=True,
                 batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.activation_layer = get_activation(hidden_activation)
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   self.activation_layer,
                                   nn.Linear(hidden_dim, input_dim))
        self.use_residual = use_residual
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, X):
        """Forward pass of ResidualBlock.

        Args:
            X: Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual block.
        """
        X_out = self.layer(X)
        if self.use_residual:
            X_out = X_out + X
        if self.batch_norm is not None:
            X_out = self.batch_norm(X_out)
        output = self.activation_layer(X_out)
        if self.dropout is not None:
            output = self.dropout(output)
        return output



