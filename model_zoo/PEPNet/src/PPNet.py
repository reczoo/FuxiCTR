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
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation


class PPNet(BaseModel):
    """Parameter Personalized Network (PPNet) model.

    Args:
        feature_map (FeatureMap): A FeatureMap instance used to store feature specs.
        model_id (str): Model identifier string. Default: ``"PPNet"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for training. Default: ``1e-3``.
        embedding_dim (int): Embedding dimension of features. Default: ``10``.
        gate_emb_dim (int): Embedding dimension for gate features. Default: ``10``.
        gate_priors (list): List of feature names used as gate priors. Default: ``[]``.
        gate_hidden_dim (int): Hidden dimension of gate networks. Default: ``64``.
        hidden_units (list): Hidden units of the MLP. Default: ``[64, 64, 64]``.
        hidden_activations (str): Activation function for hidden layers. Default: ``"ReLU"``.
        net_dropout (float): Dropout rate. Default: ``0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="PPNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 gate_emb_dim=10,
                 gate_priors=[],
                 gate_hidden_dim=64,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PPNet, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.gate_embed_layer = FeatureEmbedding(feature_map, gate_emb_dim,
                                                 required_feature_columns=gate_priors)
        gate_input_dim = feature_map.sum_emb_out_dim() + len(gate_priors) * gate_emb_dim
        self.ppn = PPNet_MLP(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             gate_input_dim=gate_input_dim,
                             gate_hidden_dim=gate_hidden_dim,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of PPNet.

        Args:
            inputs: Model inputs.

        Returns:
            dict: Dictionary containing ``y_pred``.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)
        gate_emb = self.gate_embed_layer(X, flatten_emb=True)
        y_pred = self.ppn(feature_emb, gate_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict


class PPNet_MLP(nn.Module):
    """PPNet MLP with gated units.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output dimension. Default: ``1``.
        gate_input_dim (int): Input dimension for gate networks. Default: ``64``.
        gate_hidden_dim (int or None): Hidden dimension of gate networks. Default: ``None``.
        hidden_units (list): Hidden units of the MLP. Default: ``[]``.
        hidden_activations (str): Activation function for hidden layers. Default: ``"ReLU"``.
        dropout_rates (float): Dropout rates. Default: ``0.0``.
        batch_norm (bool): Whether to apply batch normalization. Default: ``False``.
        use_bias (bool): Whether to use bias in linear layers. Default: ``True``.
    """
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 gate_input_dim=64,
                 gate_hidden_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True):
        super(PPNet_MLP, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            layers = [nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx] is not None:
                layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.mlp_layers.append(nn.Sequential(*layers))
            self.gate_layers.append(GateNU(gate_input_dim, gate_hidden_dim,
                                           output_dim=hidden_units[idx + 1]))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))

    def forward(self, feature_emb, gate_emb):
        """Forward pass of PPNet_MLP.

        Args:
            feature_emb: Feature embeddings.
            gate_emb: Gate embeddings.

        Returns:
            Tensor: Output tensor.
        """
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        h = feature_emb
        for i in range(len(self.gate_layers)):
            h = self.mlp_layers[i](h)
            g = self.gate_layers[i](gate_input)
            h = h * g
        out = self.mlp_layers[-1](h)
        return out


class GateNU(nn.Module):
    """Gated Neural Unit for PPNet.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int or None): Hidden dimension. Default: ``None``.
        output_dim (int or None): Output dimension. Default: ``None``.
        hidden_activation (str): Activation function for hidden layer. Default: ``"ReLU"``.
        dropout_rate (float): Dropout rate. Default: ``0.0``.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="ReLU",
                 dropout_rate=0.0):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*layers)

    def forward(self, inputs):
        """Forward pass of GateNU.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor: Gated output scaled by 2.
        """
        return self.gate(inputs) * 2
