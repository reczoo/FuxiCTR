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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, SqueezeExcitation, \
                                   BilinearInteractionV2, LogisticRegression

class FiBiNET(BaseModel):
    """Feature Importance and Bilinear feature Interaction NETwork (FiBiNET) model.

    Args:
        feature_map (FeatureMap): FeatureMap object containing feature specifications.
        model_id (str): Model identifier string. Default: ``"FiBiNET"``.
        gpu (int): GPU device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate for optimization. Default: ``1e-3``.
        embedding_dim (int): Dimension of feature embeddings. Default: ``10``.
        hidden_units (list): Hidden units for the DNN tower. Default: ``[]``.
        hidden_activations (str): Activation functions for DNN. Default: ``"ReLU"``.
        excitation_activation (str): Activation function for squeeze-and-excitation. Default: ``"ReLU"``.
        reduction_ratio (int): Reduction ratio for squeeze-and-excitation. Default: ``3``.
        bilinear_type (str): Bilinear interaction type. Default: ``"field_interaction"``.
        net_dropout (float): Dropout rate for the network. Default: ``0``.
        batch_norm (bool): Whether to use batch normalization. Default: ``False``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network parameters. Default: ``None``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="FiBiNET",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 excitation_activation="ReLU",
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(FiBiNET, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        self.senet_layer = SqueezeExcitation(num_fields, reduction_ratio, excitation_activation)
        self.bilinear_interaction1 = BilinearInteractionV2(num_fields, embedding_dim, bilinear_type)
        self.bilinear_interaction2 = BilinearInteractionV2(num_fields, embedding_dim, bilinear_type)
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        input_dim = num_fields * (num_fields - 1) * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units, 
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Forward pass of FiBiNET.

        Args:
            inputs: Input data containing features.

        Returns:
            dict: Dictionary with ``y_pred`` key containing the prediction tensor.
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        senet_emb = self.senet_layer(feature_emb)
        bilinear_p = self.bilinear_interaction1(feature_emb)
        bilinear_q = self.bilinear_interaction2(senet_emb)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        dnn_out = self.dnn(comb_out)
        y_pred = self.lr_layer(X) + dnn_out
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict