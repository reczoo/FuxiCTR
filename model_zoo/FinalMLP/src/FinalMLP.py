# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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


class FinalMLP(BaseModel):
    """FinalMLP model with feature selection and interaction aggregation.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"FinalMLP"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        mlp1_hidden_units (list): Hidden units for the first MLP tower. Default: ``[64, 64, 64]``.
        mlp1_hidden_activations (str): Activation functions for the first MLP. Default: ``"ReLU"``.
        mlp1_dropout (float): Dropout rate for the first MLP. Default: ``0``.
        mlp1_batch_norm (bool): Whether to use batch normalization in the first MLP. Default: ``False``.
        mlp2_hidden_units (list): Hidden units for the second MLP tower. Default: ``[64, 64, 64]``.
        mlp2_hidden_activations (str): Activation functions for the second MLP. Default: ``"ReLU"``.
        mlp2_dropout (float): Dropout rate for the second MLP. Default: ``0``.
        mlp2_batch_norm (bool): Whether to use batch normalization in the second MLP. Default: ``False``.
        use_fs (bool): Whether to use feature selection. Default: ``True``.
        fs_hidden_units (list): Hidden units for feature selection gates. Default: ``[64]``.
        fs1_context (list): Context features for the first feature selection gate. Default: ``[]``.
        fs2_context (list): Context features for the second feature selection gate. Default: ``[]``.
        num_heads (int): Number of attention heads for interaction aggregation. Default: ``1``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="FinalMLP",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 mlp1_hidden_units=[64, 64, 64],
                 mlp1_hidden_activations="ReLU",
                 mlp1_dropout=0,
                 mlp1_batch_norm=False,
                 mlp2_hidden_units=[64, 64, 64],
                 mlp2_hidden_activations="ReLU",
                 mlp2_dropout=0,
                 mlp2_batch_norm=False,
                 use_fs=True,
                 fs_hidden_units=[64],
                 fs1_context=[],
                 fs2_context=[],
                 num_heads=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalMLP, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        feature_dim = embedding_dim * feature_map.num_fields
        self.mlp1 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout,
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=feature_dim,
                              output_dim=None, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_map, 
                                              feature_dim, 
                                              embedding_dim, 
                                              fs_hidden_units, 
                                              fs1_context,
                                              fs2_context)
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """Forward pass of FinalMLP.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class FeatureSelection(nn.Module):
    """Feature selection module for FinalMLP.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        feature_dim (int): Total feature dimension.
        embedding_dim (int): Dimension of feature embeddings.
        fs_hidden_units (list): Hidden units for feature selection gates. Default: ``[]``.
        fs1_context (list): Context features for the first gate. Default: ``[]``.
        fs2_context (list): Context features for the second gate. Default: ``[]``.
    """
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[],
                 fs1_context=[], fs2_context=[]):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs1_context)
        self.fs2_context = fs2_context
        if len(fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs2_context)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        """Forward pass of FeatureSelection.

        Args:
            X: Input feature dict.
            flat_emb: Flattened embedding tensor.

        Returns:
            tuple: Two feature-selected tensors.
        """
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Module):
    """Interaction aggregation module for fusing two MLP outputs.

    Args:
        x_dim (int): Dimension of the first input.
        y_dim (int): Dimension of the second input.
        output_dim (int): Output dimension. Default: ``1``.
        num_heads (int): Number of attention heads. Default: ``1``.
    """
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim,
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        """Forward pass of InteractionAggregation.

        Args:
            x: First input tensor.
            y: Second input tensor.

        Returns:
            torch.Tensor: Aggregated output tensor.
        """
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2),
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output
