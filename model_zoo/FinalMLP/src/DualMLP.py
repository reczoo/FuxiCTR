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


class DualMLP(BaseModel):
    """Dual MLP model with two parallel MLP towers.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"DualMLP"``.
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
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="DualMLP",
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
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DualMLP, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.mlp1 = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                              output_dim=1, 
                              hidden_units=mlp1_hidden_units,
                              hidden_activations=mlp1_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp1_dropout, 
                              batch_norm=mlp1_batch_norm)
        self.mlp2 = MLP_Block(input_dim=embedding_dim * feature_map.num_fields,
                              output_dim=1, 
                              hidden_units=mlp2_hidden_units,
                              hidden_activations=mlp2_hidden_activations,
                              output_activation=None,
                              dropout_rates=mlp2_dropout, 
                              batch_norm=mlp2_batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """Forward pass of DualMLP.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        y_pred = self.mlp1(flat_emb) + self.mlp2(flat_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
