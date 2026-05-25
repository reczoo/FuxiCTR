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
import torch.nn.functional as F
import numpy as np
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class FinalNet(BaseModel):
    """FinalNet model with factorized interaction blocks.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"FinalNet"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        block_type (str): Block type, one of ["1B", "2B"]. Default: ``"2B"``.
        batch_norm (bool): Whether to use batch normalization. Default: ``True``.
        use_feature_gating (bool): Whether to use feature gating. Default: ``False``.
        block1_hidden_units (list): Hidden units for the first block. Default: ``[64, 64, 64]``.
        block1_hidden_activations (str or None): Activation functions for the first block. Default: ``None``.
        block1_dropout (float): Dropout rate for the first block. Default: ``0``.
        block2_hidden_units (list): Hidden units for the second block. Default: ``[64, 64, 64]``.
        block2_hidden_activations (str or None): Activation functions for the second block. Default: ``None``.
        block2_dropout (float): Dropout rate for the second block. Default: ``0``.
        residual_type (str): Residual type, one of ["concat", "sum"]. Default: ``"concat"``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="FinalNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 block_type="2B",
                 batch_norm=True,
                 use_feature_gating=False,
                 block1_hidden_units=[64, 64, 64],
                 block1_hidden_activations=None,
                 block1_dropout=0,
                 block2_hidden_units=[64, 64, 64],
                 block2_hidden_activations=None,
                 block2_dropout=0,
                 residual_type="concat",
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FinalNet, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        assert block_type in ["1B", "2B"], "block_type={} not supported.".format(block_type)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.use_feature_gating = use_feature_gating
        if use_feature_gating:
            self.feature_gating = FeatureGating(num_fields, gate_residual="concat")
            gate_out_dim = embedding_dim * num_fields * 2
        self.block_type = block_type
        self.block1 = FinalBlock(input_dim=gate_out_dim if use_feature_gating \
                                           else embedding_dim * num_fields,
                                 hidden_units=block1_hidden_units,
                                 hidden_activations=block1_hidden_activations,
                                 dropout_rates=block1_dropout,
                                 batch_norm=batch_norm,
                                 residual_type=residual_type)
        self.fc1 = nn.Linear(block1_hidden_units[-1], 1)
        if block_type == "2B":
            self.block2 = FinalBlock(input_dim=embedding_dim * num_fields,
                                     hidden_units=block2_hidden_units,
                                     hidden_activations=block2_hidden_activations,
                                     dropout_rates=block2_dropout,
                                     batch_norm=batch_norm,
                                     residual_type=residual_type)
            self.fc2 = nn.Linear(block2_hidden_units[-1], 1)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """Forward pass of FinalNet.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred``, ``y1``, and ``y2`` keys.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred, y1, y2 = None, None, None
        if self.block_type == "1B":
            y_pred = self.forward1(feature_emb)
        elif self.block_type == "2B":
            y1 = self.forward1(feature_emb)
            y2 = self.forward2(feature_emb)
            y_pred = 0.5 * (y1 + y2)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred, "y1": y1, "y2": y2}
        return return_dict

    def forward1(self, X):
        """Forward pass of the first block.

        Args:
            X: Feature embedding tensor.

        Returns:
            torch.Tensor: Prediction logits from the first block.
        """
        if self.use_feature_gating:
            X = self.feature_gating(X)
        block1_out = self.block1(X.flatten(start_dim=1))
        y_pred = self.fc1(block1_out)
        return y_pred

    def forward2(self, X):
        """Forward pass of the second block.

        Args:
            X: Feature embedding tensor.

        Returns:
            torch.Tensor: Prediction logits from the second block.
        """
        block2_out = self.block2(X.flatten(start_dim=1))
        y_pred = self.fc2(block2_out)
        return y_pred

    def add_loss(self, return_dict, y_true):
        """Compute total loss including consistency losses for 2B mode.

        Args:
            return_dict: Dictionary returned by forward pass.
            y_true: Ground truth labels.

        Returns:
            torch.Tensor: Total loss tensor.
        """
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        if self.block_type == "2B":
            y1 = self.output_activation(return_dict["y1"])
            y2 = self.output_activation(return_dict["y2"])
            loss1 = self.loss_fn(y1, return_dict["y_pred"].detach(), reduction='mean')
            loss2 = self.loss_fn(y2, return_dict["y_pred"].detach(), reduction='mean')
            loss = loss + loss1 + loss2
        return loss


class FeatureGating(nn.Module):
    """Feature gating module for FinalNet.

    Args:
        num_fields (int): Number of input fields.
        gate_residual (str): Residual mode, one of ["concat", "sum"]. Default: ``"concat"``.
    """
    def __init__(self, num_fields, gate_residual="concat"):
        super(FeatureGating, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)
        assert gate_residual in ["concat", "sum"]
        self.gate_residual = gate_residual

    def init_weights(self):
        """Initialize gating weights."""
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        """Forward pass of FeatureGating.

        Args:
            feature_emb: Feature embedding tensor of shape (B, F, D).

        Returns:
            torch.Tensor: Gated feature embeddings.
        """
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        if self.gate_residual == "concat":
            out = torch.cat([feature_emb, feature_emb * gates], dim=1) # b x 2f x d
        else:
            out = feature_emb + feature_emb * gates
        return out


class FinalBlock(nn.Module):
    """Final block with factorized interaction layers.

    Args:
        input_dim (int): Input feature dimension.
        hidden_units (list): List of hidden layer dimensions. Default: ``[]``.
        hidden_activations (str or None): Activation functions for hidden layers. Default: ``None``.
        dropout_rates (float or list): Dropout rates for each layer. Default: ``[]``.
        batch_norm (bool): Whether to use batch normalization. Default: ``True``.
        residual_type (str): Residual type, one of ["sum", "concat"]. Default: ``"sum"``.
    """
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None,
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        # Factorized Interaction Block: Replacement of MLP block
        super(FinalBlock, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FactorizedInteraction(hidden_units[idx],
                                                    hidden_units[idx + 1],
                                                    residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        """Forward pass of FinalBlock.

        Args:
            X: Input tensor.

        Returns:
            torch.Tensor: Output tensor after final block.
        """
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class FactorizedInteraction(nn.Module):
    """Factorized interaction layer capturing quadratic feature interactions.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension.
        bias (bool): Whether to use bias. Default: ``True``.
        residual_type (str): Residual type, one of ["sum", "concat"]. Default: ``"sum"``.
    """
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        super(FactorizedInteraction, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        """Forward pass of FactorizedInteraction.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor with factorized interactions.
        """
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            h = torch.cat([h2, h1 * h2], dim=-1)
        elif self.residual_type == "sum":
            h = h2 + h1 * h2
        return h
